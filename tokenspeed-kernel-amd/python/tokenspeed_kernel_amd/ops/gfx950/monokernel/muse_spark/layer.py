# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""One Muse Spark decode layer (layers 1 .. 60: MXFP4 experts) of one TP8
rank as one persistent launch.

Rank ``r`` owns hidden columns ``1024 r ..``: it keeps that slice of the FP32
residual, and every rank keeps the whole BF16 attention input. The stages,
in plan order (see each module):

* attention: QKVG, gate, head, attn, merge (``qkvg``, ``attention``);
* output projection and attention tail (``attn_tail``);
* routing, pooler, experts, combine, post-expert projection and boundary
  (``moe_route``, ``moe_pool``, ``moe_experts``, ``moe_tail``).

The peer regions' copy of a launch is ``layer & 1``: 62 layers are an even
count, so it alternates from one monokernel launch to the next, across
steps too (layers 0 and 61 keep the regular path).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    attention,
    attn_tail,
    moe_experts,
    moe_pool,
    moe_route,
    moe_tail,
    qkvg,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import HIDDEN, WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan

SLIDING_SPLITS = 16
FULL_SPLITS = 64
WINDOW = attention.WINDOW

# The first placement: weight-heavy stages on the CTAs that finish their
# earlier tasks first. The trace decides the next one.
BASES = {
    "qkvg": 0,
    "gate": 0,
    "head": 158,
    "attn": 162,
    "merge": 240,
    "oproj": 0,
    "tail": 128,
    "router": 136,
    "topk": 152,
    "pool": 192,
    "ex1": 0,
    "ex2": 0,
    "comb": 64,
    "unpool": 128,
    "out": 216,
}


# Past one row the attention tail has 8 tasks per row and the router 16 per
# pair of rows: both move to CTAs whose only earlier task is oproj (the tail
# off the router's CTAs, whose weight loads would otherwise wait for it).
CHAINED_BASES = {**BASES, "tail": 64, "router": 0}


def _bases(rows: int, *names: str) -> dict[str, int]:
    table = BASES if rows == 1 else CHAINED_BASES
    return {name: table[name] for name in names}


def layer_plan(full: bool, rows: int) -> Plan:
    """The plan of a full-attention (``full``) or sliding-window layer for
    ``rows`` token rows of one request: 1 for decode, ``k + 1`` for a
    speculative verify of ``k`` draft tokens."""
    if not 1 <= rows <= 8:
        raise ValueError(f"a launch serves 1 to 8 rows, got {rows}")
    splits = FULL_SPLITS if full else SLIDING_SPLITS
    regions = (
        qkvg.REGIONS
        + attention.regions(splits)
        + attn_tail.regions()
        + moe_route.regions()
        + moe_pool.regions()
        + moe_experts.regions()
        + moe_tail.regions()
    )
    stages = (
        qkvg.stages(_bases(rows, "qkvg", "gate"))
        + attention.stages(_bases(rows, "head", "attn", "merge"), splits, rows)
        + attn_tail.stages(_bases(rows, "oproj", "tail"), rows)
        + moe_route.stages(_bases(rows, "router", "topk"), rows)
        + moe_pool.stages(_bases(rows, "pool"))
        + moe_experts.stages(_bases(rows, "ex1", "ex2"))
        + moe_tail.stages(_bases(rows, "comb", "unpool", "out"), rows)
    )
    peers = (
        attn_tail.PEERS
        + moe_route.PEERS
        + moe_pool.PEERS
        + moe_experts.PEERS
        + moe_tail.PEERS
    )
    return Plan(
        grid=256,
        stages=stages,
        regions=regions,
        rows=rows,
        chain=rows,
        world=int(WORLD),
        peers=peers,
    )


# The layer index changes every launch: a specialization on it (1, or a
# multiple of 16) would compile new binaries while serving.
@gluon.jit(do_not_specialize=("layer",))
def gluon_muse_spark_layer_gfx950(
    scratch,
    step,
    layer,
    bases,
    attn_ops,
    moe_ops,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    SPLITS: gl.constexpr,
    WIN: gl.constexpr,
    ROPE: gl.constexpr,
):
    """One decode step of one layer on rank ``RANK``.

    ``attn_ops``: (QKVG weight, attention input ``a``, ``seq_len``, KV write
    slot, page table, K cache, V cache, cos / sin, o_proj weight, residual
    slice ``h``, attention gate ``alpha``, ``beta``, pre-FFN gain).
    ``moe_ops``: (router columns, router bias, pooler rows, w13, w13 scales,
    their 4 strides, w2, w2 scales, their 4 strides, pre-expert gain,
    post-expert gain, post-expert projection rows, post-FFN gain, FFN gate
    ``alpha``, ``beta`` (all 8192 columns), next input gain (all 8192
    columns), residual output, next attention input).
    """
    table = trace.allocate(trace_buf)
    peers = mailbox.load_peers(bases, RANK, layer & 1, WORLD)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, layer), PLAN, table, peers)
    (
        w_qkvg,
        a_in,
        seq_len,
        slot,
        page_table,
        k_cache,
        v_cache,
        cos_sin,
        w_o,
        h_in,
        alpha1,
        beta1,
        pre_gain,
    ) = attn_ops
    (
        router_w,
        router_bias,
        w_pool,
        w13,
        w13s,
        s13_we,
        s13_se,
        s13_lin,
        s13_nb,
        w2,
        w2s,
        s2_we,
        s2_se,
        s2_lin,
        s2_nb,
        pre_exp_gain,
        pen_gain,
        w_post,
        post_gain,
        alpha2,
        beta2,
        in_gain,
        h_next,
        a_out,
    ) = moe_ops

    task, active = schedule.stage_task(PLAN, "qkvg")
    if active:
        qkvg.qkvg_stage(mb, task, w_qkvg, a_in, HIDDEN, "qkvg")
    task, active = schedule.stage_task(PLAN, "gate")
    if active:
        qkvg.gate_stage(mb, task, w_qkvg, a_in, HIDDEN, "gate")
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        attention.head_stage(
            mb, task, seq_len, slot, k_cache, v_cache, cos_sin, ROPE, "head"
        )
    task, active = schedule.stage_task(PLAN, "attn")
    if active:
        attention.attn_stage(
            mb,
            task,
            seq_len,
            page_table,
            k_cache,
            v_cache,
            cos_sin,
            ROPE,
            SPLITS,
            WIN,
            "attn",
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        attention.merge_stage(mb, task, SPLITS, "merge")
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        attn_tail.oproj_stage(mb, task, w_o, RANK, "oproj")
    task, active = schedule.stage_task(PLAN, "tail1")
    if active:
        attn_tail.tail_stages(mb, task, h_in, alpha1, beta1, pre_gain, RANK)
    task, active = schedule.stage_task(PLAN, "router")
    if active:
        moe_route.router_stage(
            mb, task, router_w, pre_gain, h_in, alpha1, beta1, RANK, "router"
        )
    task, active = schedule.stage_task(PLAN, "topk")
    if active:
        moe_route.topk_stage(mb, task, router_bias, "topk")
    task, active = schedule.stage_task(PLAN, "pool")
    if active:
        moe_pool.pool_stage(mb, task, w_pool, RANK, "pool")
    # A CTA's ex1 and ex2 tasks run in parts (see moe_experts): both weight
    # streams once routing names the experts, then the A operand meanwhile.
    task1, active1 = schedule.stage_task(PLAN, "ex1")
    task1 = gl.where(active1, task1, 0)
    task2, active2 = schedule.stage_task(PLAN, "ex2")
    task2 = gl.where(active2, task2, 0)
    got1, gain_lo, gain_hi, ids, e1, e2 = moe_experts.ex1_poll(
        mb, task1, task2, active1, pre_exp_gain, "ex1"
    )
    loads1 = moe_experts.ex1_issue(
        task1, active1, e1, w13, w13s, s13_we, s13_se, s13_lin, s13_nb
    )
    loads2 = moe_experts.ex2_issue(
        mb, task2, active2, e2, w2, w2s, s2_we, s2_se, s2_lin, s2_nb, "ex2"
    )
    union = moe_experts.slots(ids)
    a1 = moe_experts.ex1_operand(task1, got1, gain_lo, gain_hi)
    if active1:
        moe_experts.ex1_finish(
            mb,
            task1,
            a1,
            loads1,
            e1,
            union,
            w13,
            w13s,
            s13_we,
            s13_se,
            s13_lin,
            s13_nb,
            "ex1",
        )
    if active2:
        moe_experts.ex2_finish(
            mb,
            task2,
            loads2,
            e2,
            union,
            w2,
            w2s,
            s2_we,
            s2_se,
            s2_lin,
            s2_nb,
            RANK,
            "ex2",
        )
    task, active = schedule.stage_task(PLAN, "comb1")
    if active:
        moe_tail.comb_stages(mb, task, pen_gain, RANK)
    task, active = schedule.stage_task(PLAN, "unpool")
    if active:
        moe_tail.unpool_stages(mb, task, w_post, post_gain, alpha2, beta2, h_next, RANK)
    task, active = schedule.stage_task(PLAN, "out")
    if active:
        moe_tail.out_stage(mb, task, beta2, in_gain, a_out, "out")
    trace.flush(table, trace_buf)


def launch_layer(
    scratch,
    step,
    layer,
    bases,
    attn_ops,
    moe_ops,
    trace_buf,
    rank: int,
    full: bool,
    rows: int,
):
    """Launch one layer step of ``rows`` token rows (``full``: a
    full-attention, NoPE layer)."""
    plan = layer_plan(full, rows)
    gluon_muse_spark_layer_gfx950[(plan.grid,)](
        scratch,
        step,
        layer,
        bases,
        attn_ops,
        moe_ops,
        trace_buf,
        plan.key,
        rank,
        FULL_SPLITS if full else SLIDING_SPLITS,
        0 if full else WINDOW,
        not full,
        num_warps=8,
    )


def launch_gluon_muse_spark_layer_gfx950(
    weights,
    *,
    h,
    a,
    h_out,
    a_out,
    seq_len,
    slot,
    page_table,
    k_cache,
    v_cache,
    cos_sin,
    scratch,
    step,
    peer_bases,
    layer: int,
    trace_buf,
):
    """One step of a packed layer (``weights.LayerWeights``) for the rows of
    one request: one decode row, or a speculative verify's consecutive
    tokens (row ``t`` at position ``seq_len[0] - 1 + t``). The row count
    ``rows`` comes from ``h`` and selects the plan.

    Args:
        weights: From ``weights.pack_layer``.
        h, h_out: FP32 ``[rows, 1024]`` owned residual columns, in and out.
        a, a_out: BF16 ``[rows, 8192]`` attention input of this layer and of
            the next one (normed with the next layer's input gain).
        seq_len, slot: int32 ``[rows]``: each row's length including its
            token, and the KV cache slot of the token.
        page_table: int32 ``[pages]`` of the request (64 tokens a page,
            indexed by absolute position).
        k_cache, v_cache: BF16 ``[slots, 2, 64]``.
        cos_sin: FP32 ``[positions, 64]`` rotary table (unused by a full,
            NoPE layer).
        scratch: int64 mailbox pairs on the Iris symmetric heap.
        step: int32 ``[1]`` step counter.
        peer_bases: int64 ``[8]`` heap bases of the group's ranks.
        layer: The decoder layer index (the tag and the peer copy).
        trace_buf: None, or a trace buffer (``trace.py``).
    """
    rows = h.shape[0] if h.dim() == 2 else 0
    # The launch reads h and a while other CTAs already write the outputs.
    if (
        h.dtype != torch.float32
        or h.shape != (rows, 1024)
        or not h.is_contiguous()
        or h.data_ptr() == h_out.data_ptr()
    ):
        raise ValueError("h: contiguous FP32 [rows, 1024] that does not alias h_out")
    if (
        a.dtype != torch.bfloat16
        or a.shape != (rows, 8192)
        or not a.is_contiguous()
        or a.data_ptr() == a_out.data_ptr()
    ):
        raise ValueError("a: contiguous BF16 [rows, 8192] that does not alias a_out")
    for name, t in (("seq_len", seq_len), ("slot", slot)):
        if t.dtype != torch.int32 or t.numel() != rows:
            raise ValueError(f"{name}: int32 [{rows}], one per row")
    attn_ops, moe_ops = weights.kernel_args(
        h=h,
        a=a,
        seq_len=seq_len,
        slot=slot,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        h_out=h_out,
        a_out=a_out,
    )
    launch_layer(
        scratch,
        step,
        layer,
        peer_bases,
        attn_ops,
        moe_ops,
        trace_buf,
        weights.rank,
        weights.full,
        rows,
    )


__all__ = [
    "BASES",
    "launch_gluon_muse_spark_layer_gfx950",
    "FULL_SPLITS",
    "SLIDING_SPLITS",
    "gluon_muse_spark_layer_gfx950",
    "launch_layer",
    "layer_plan",
]
