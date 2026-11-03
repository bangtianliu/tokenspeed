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
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule, trace, warm
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
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage

SLIDING_SPLITS = 16
# A full layer's attention splits: one request's tasks (2 KV heads x splits)
# fill the grid, so at long context each walks half the keys that 64 splits
# leave it (a full layer at 200k keys: 156 vs 192 us). Several requests' tasks
# run in rounds: the splits, and so each request's sums, do not depend on how
# many requests share the launch.
FULL_SPLITS = 128
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


# The most rows of one launch: two requests' verifies of 7 draft tokens
# (the drafters' block of 8). The GEMVs' MFMAs hold 16 rows.
MAX_ROWS = 16
# KV cache dtypes the attention reads and writes (FP8 at unit scale).
KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


# With warm_next, CTAs 0 .. WARM_TASKS - 1 read the next launch's QKVG and
# gate weights into the memory-side cache (warm.py) after their expert tasks:
# the next launch then starts its projections from the MALL. Earlier, the
# expert weights streaming after them would evict them. The comb tasks start
# at CTA 64.
WARM_TASKS = 64


def layer_plan(full: bool, rows: int, chain: int, warm_next: bool) -> Plan:
    """The plan of a full-attention (``full``) or sliding-window layer for
    ``rows`` token rows: requests of ``chain`` tokens each, 1 for decode,
    ``k + 1`` for a speculative verify of ``k`` draft tokens. With
    ``warm_next`` the launch also warms the next launch's QKVG weights."""
    if not 1 <= rows <= MAX_ROWS or rows % chain:
        raise ValueError(
            f"a launch serves 1 to {MAX_ROWS} rows of requests of {chain}, got {rows}"
        )
    splits = FULL_SPLITS if full else SLIDING_SPLITS
    # Both kinds of layer lay their regions out alike: the attention
    # regions are sized for the full layer's splits. Consecutive launches
    # alternate the peer regions' two copies, which keeps a rank's puts for
    # the next launch off the copy a slower rank still reads; with two
    # layouts in one scratch, a sliding and a full launch overlapped and the
    # group deadlocked.
    regions = (
        qkvg.REGIONS
        + attention.regions(FULL_SPLITS)
        + attn_tail.regions()
        + moe_route.regions()
        + moe_pool.regions()
        + moe_experts.regions()
        + moe_tail.regions()
    )
    stages = (
        qkvg.stages(_bases(rows, "qkvg", "gate"))
        + attention.stages(_bases(rows, "head", "attn", "merge"), splits, rows, chain)
        + attn_tail.stages(_bases(rows, "oproj", "tail"), rows)
        + moe_route.stages(_bases(rows, "router", "topk"), rows)
        + moe_pool.stages(_bases(rows, "pool"))
        + moe_experts.stages(_bases(rows, "ex1", "ex2"))
        + ((Stage("warm", WARM_TASKS, 0),) if warm_next else ())
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
        chain=chain,
        world=int(WORLD),
        peers=peers,
    )


# The layer index changes every launch: a specialization on it (1, or a
# multiple of 16) would compile new binaries while serving. The warm word
# count is the next layer's, a runtime value too.
@gluon.jit(do_not_specialize=("layer", "warm_quads"))
def gluon_muse_spark_layer_gfx950(
    scratch,
    step,
    layer,
    bases,
    attn_ops,
    moe_ops,
    tap,
    tap_col,
    tap_row,
    warm_ptr,
    warm_quads,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    SPLITS: gl.constexpr,
    WIN: gl.constexpr,
    ROPE: gl.constexpr,
    TAP: gl.constexpr,
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
    ``tap``, ``tap_col``, ``tap_row``: with ``TAP``, where the residual
    after the layer also goes, BF16 on every rank (``moe_tail.unpool_stages``).
    ``warm_ptr``, ``warm_quads``: the next launch's packed QKVG weights, read
    (``4 * warm_quads`` int32 words) only when ``PLAN`` has the warm stage.
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
        table_stride,
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
    # Several requests' attention tasks may outnumber the CTAs: rounds.
    G: gl.constexpr = schedule.grid(PLAN)
    attn_task, attn_rounds = schedule.stage_rounds(PLAN, "attn")
    for i in range(attn_rounds):
        attention.attn_stage(
            mb,
            attn_task + i * G,
            seq_len,
            page_table,
            table_stride,
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
    if schedule.has_stage(PLAN, "warm"):
        task, active = schedule.stage_task(PLAN, "warm")
        if active:
            warm.warm_stage(
                mb,
                task,
                warm_ptr,
                warm_quads,
                schedule.stage_tasks(PLAN, "warm"),
                "warm",
            )
    task, active = schedule.stage_task(PLAN, "comb1")
    if active:
        moe_tail.comb_stages(mb, task, pen_gain, RANK)
    task, active = schedule.stage_task(PLAN, "unpool")
    if active:
        moe_tail.unpool_stages(
            mb,
            task,
            w_post,
            post_gain,
            alpha2,
            beta2,
            h_next,
            tap,
            tap_col,
            tap_row,
            RANK,
            TAP,
        )
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
    tap,
    tap_slot,
    warm_next,
    trace_buf,
    rank: int,
    full: bool,
    rows: int,
    chain: int,
    compile_only: bool,
):
    """Launch one layer step of ``rows`` token rows, requests of ``chain``
    tokens (``full``: a
    full-attention, NoPE layer), or with ``compile_only`` only compile the
    launch. ``tap`` (None, or a BF16 ``[rows+, slots, 8192]`` buffer on the
    Iris heap) gets the residual after the layer in slot ``tap_slot`` on
    every rank. ``warm_next`` (None, or the next launch's packed QKVG
    weights) is warmed into the memory-side cache after the experts."""
    plan = layer_plan(full, rows, chain, warm_next is not None)
    if tap is None:
        tap_args = (attn_ops[0], 0, 0)
    else:
        tap_args = (tap, tap_slot * tap.shape[2], tap.stride(0))
    if warm_next is None:
        warm_args = (attn_ops[0], 0)
    else:
        if not warm_next.is_contiguous() or warm_next.nbytes % 16:
            raise ValueError("warm_next: contiguous, a whole number of 16-byte quads")
        # The warm stage reads int32 words: the same bytes, viewed.
        warm_args = (warm_next.view(-1).view(torch.int32), warm_next.nbytes // 16)
    args = (
        scratch,
        step,
        layer,
        bases,
        attn_ops,
        moe_ops,
        *tap_args,
        *warm_args,
        trace_buf,
        plan.key,
        rank,
        FULL_SPLITS if full else SLIDING_SPLITS,
        0 if full else WINDOW,
        not full,
        tap is not None,
    )
    if compile_only:
        gluon_muse_spark_layer_gfx950.warmup(*args, num_warps=8, grid=(plan.grid,))
    else:
        gluon_muse_spark_layer_gfx950[(plan.grid,)](*args, num_warps=8)


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
    chain: int,
    tap,
    tap_slot,
    warm_next,
    trace_buf,
    compile_only: bool,
):
    """One step of a packed layer (``weights.LayerWeights``) for the rows of
    one or more requests of ``chain`` tokens each: a decode row each, or a
    speculative verify's consecutive tokens (row ``t`` of request ``b`` at
    position ``seq_len[b chain] - 1 + t``). The row count ``rows`` comes
    from ``h``; ``rows`` and ``chain`` select the plan.

    Args:
        weights: From ``weights.pack_layer``.
        h, h_out: FP32 ``[rows, 1024]`` owned residual columns, in and out.
        a, a_out: BF16 ``[rows, 8192]`` attention input of this layer and of
            the next one (normed with the next layer's input gain).
        seq_len, slot: int32 ``[rows]``: each row's length including its
            token, and the KV cache slot of the token.
        page_table: int32 ``[requests, pages]`` (or ``[pages]`` for one
            request): each request's page table (64 tokens a page, indexed
            by absolute position).
        k_cache, v_cache: BF16, or FP8 e4m3 holding unit-scale values,
            ``[slots, 2, 64]``.
        cos_sin: FP32 ``[positions, 64]`` rotary table (unused by a full,
            NoPE layer).
        scratch: int64 mailbox pairs on the Iris symmetric heap.
        step: int32 ``[1]`` step counter.
        peer_bases: int64 ``[8]`` heap bases of the group's ranks.
        layer: The decoder layer index (the tag and the peer copy).
        chain: Tokens a request: 1 for decode, ``k + 1`` for a verify.
        tap: None, or a BF16 ``[rows+, slots, 8192]`` buffer on the Iris
            symmetric heap (the same on every rank): the launch also stores
            the residual after the layer, rounded to BF16, into slot
            ``tap_slot`` of every rank's copy (a draft's tap).
        tap_slot: The tap's slot, or None without ``tap``.
        warm_next: None, or the next launch's ``weights.LayerWeights``: this
            launch reads its QKVG weights into the memory-side cache after
            the experts, so the next launch's projections read the MALL.
        trace_buf: None, or a trace buffer (``trace.py``).
        compile_only: Only compile the launch for these operands. A rank's
            launch polls its peers' values for a few seconds before it traps,
            so the ranks compile before any of them launches.
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
    # FP8 caches hold unit-scale e4m3 values (the kernel applies no scale).
    if k_cache.dtype not in KV_DTYPES or v_cache.dtype != k_cache.dtype:
        raise ValueError("k_cache, v_cache: both BF16, or both FP8 e4m3 at unit scale")
    if page_table.dim() == 1:
        page_table = page_table[None]
    if (
        rows % chain
        or page_table.dim() != 2
        or page_table.shape[0] != rows // chain
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            f"page_table: int32 [{rows // chain if chain else '?'}, pages], one "
            f"row a request of {chain} tokens"
        )
    if (tap is None) != (tap_slot is None):
        raise ValueError("tap and tap_slot come together")
    if tap is not None and (
        tap.dtype != torch.bfloat16
        or tap.dim() != 3
        or tap.shape[0] < rows
        or tap.shape[2] != 8192
        or not tap.is_contiguous()
        or not 0 <= tap_slot < tap.shape[1]
    ):
        raise ValueError(f"tap: contiguous BF16 [>={rows}, slots, 8192], a slot in it")
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
        tap,
        tap_slot,
        None if warm_next is None else warm_next.qkvg,
        trace_buf,
        weights.rank,
        weights.full,
        rows,
        chain,
        compile_only,
    )


__all__ = [
    "BASES",
    "MAX_ROWS",
    "launch_gluon_muse_spark_layer_gfx950",
    "FULL_SPLITS",
    "KV_DTYPES",
    "SLIDING_SPLITS",
    "gluon_muse_spark_layer_gfx950",
    "launch_layer",
    "layer_plan",
]
