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

"""A whole DFlash2 draft layer (Muse Spark) in one launch across the 8 ranks."""

from __future__ import annotations

import socket
from datetime import timedelta

import dflash2_draft_reference as ref
import pytest
import test_muse_spark_draft_attn_gfx950 as attn_t
import test_muse_spark_draft_mlp_gfx950 as mlp_t
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monokernel_testing import payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_attn,
    draft_layer,
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, OWNED, HEAD, HALF, Q_HEADS = 8, ref.HIDDEN, 1024, 128, 64, 8
SPLITS, PAGE, WINDOW_KEYS = attn_t.SPLITS, attn_t.PAGE, attn_t.WINDOW_KEYS


@gluon.jit
def layer_kernel(
    scratch,
    step,
    bases,
    x,
    residual,
    residual_out,
    in_gain,
    a_coef,
    a_base,
    qkv_w,
    q_norm,
    k_norm,
    cos_sin,
    positions,
    slots,
    table,
    seq_len,
    k_cache,
    v_cache,
    o_w,
    post_gain,
    m_coef,
    m_base,
    gu_w,
    down_w,
    out,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    SPLITS_: gl.constexpr,
):
    table_ = trace.allocate(trace_buf)
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table_, peers)
    H: gl.constexpr = draft_mlp.HIDDEN
    G: gl.constexpr = schedule.grid(PLAN)
    # The attention half.
    task, active = schedule.stage_task(PLAN, "iscat")
    if active:
        draft_mlp.scat_stage(mb, task, x, RANK, "ared", "iscat")
    r_lo = draft_mlp.residual_zeros()
    r_hi = draft_mlp.residual_zeros()
    sum_task, has_sum = schedule.stage_task(PLAN, "isum1")
    if has_sum:
        r_lo, r_hi = draft_mlp.sum_norm_stages(
            mb,
            sum_task,
            residual,
            residual_out,
            in_gain,
            RANK,
            "ared",
            "ass",
            "ahin",
            "isum1",
            "inorm",
            False,
        )
    task, active = schedule.stage_task(PLAN, "acoef")
    if active:
        draft_mlp.proj_task(
            mb,
            task,
            a_coef,
            "ahin",
            "acpart",
            draft_mlp.COEFS,
            H,
            draft_mlp.COEF_ROWS,
            draft_mlp.COEF_KS,
            "acoef",
        )
    task, active = schedule.stage_task(PLAN, "aconv0")
    if active:
        draft_mlp.conv0_stage(mb, task, a_base, "ahin", "acpart", "ahc", "aconv0")
    task, active = schedule.stage_task(PLAN, "qkv")
    if active:
        draft_mlp.proj_task(
            mb, task, qkv_w, "ahc", "qpart", draft_attn.QKV, H, 128, 512, "qkv"
        )
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        draft_attn.head_stage(
            mb,
            task,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            slots,
            k_cache,
            v_cache,
            "head",
        )
    task, active = schedule.stage_task(PLAN, "attn")
    if active:
        draft_attn.attn_stage(
            mb, task, table, seq_len, k_cache, v_cache, SPLITS_, "attn"
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        draft_attn.merge_stage(mb, task, SPLITS_, "merge")
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        draft_mlp.proj_task(mb, task, o_w, "ao", "opart", H, 1024, 128, 512, "oproj")
    task, active = schedule.stage_task(PLAN, "afin")
    if active:
        draft_mlp.fin_to_owners(
            mb,
            task,
            a_base,
            "opart",
            draft_attn.OPROJ_KPARTS,
            "acpart",
            draft_attn.AFIN_COLS,
            RANK,
            "red",
            "afin",
        )
    # The MLP half: sum1 on isum1's CTAs, with its residual.
    if has_sum:
        draft_mlp.sum_norm_regs(
            mb,
            sum_task,
            r_lo,
            r_hi,
            residual_out,
            post_gain,
            RANK,
            "red",
            "ss",
            "hin",
            "sum1",
            "norm",
            True,
        )
    task, active = schedule.stage_task(PLAN, "coef")
    if active:
        draft_mlp.proj_task(
            mb,
            task,
            m_coef,
            "hin",
            "cpart",
            draft_mlp.COEFS,
            H,
            draft_mlp.COEF_ROWS,
            draft_mlp.COEF_KS,
            "coef",
        )
    # The CTAs that host no conv0 task start their first gate/up task right
    # after coef (see draft_mlp).
    gu_task, rounds = schedule.stage_rounds(PLAN, "gu")
    _, has_conv0 = schedule.stage_task(PLAN, "conv0")
    free = not has_conv0
    if free:
        draft_mlp.proj_task(
            mb,
            gu_task,
            gu_w,
            "hc",
            "gpart",
            draft_mlp.GU,
            H,
            draft_mlp.GU_ROWS,
            draft_mlp.GU_KS,
            "gu",
        )
    task, active = schedule.stage_task(PLAN, "conv0")
    if active:
        draft_mlp.conv0_stage(mb, task, m_base, "hin", "cpart", "hc", "conv0")
    for i in range(gl.where(free, 1, 0), rounds):
        draft_mlp.proj_task(
            mb,
            gu_task + i * G,
            gu_w,
            "hc",
            "gpart",
            draft_mlp.GU,
            H,
            draft_mlp.GU_ROWS,
            draft_mlp.GU_KS,
            "gu",
        )
    d_task, d_rounds = schedule.stage_rounds(PLAN, "down")
    a_task, has_act = schedule.stage_task(PLAN, "act")
    I: gl.constexpr = draft_mlp.INTER
    if has_act:
        tiles = draft_mlp.proj_issue(
            d_task, down_w, H, I, draft_mlp.DOWN_ROWS, draft_mlp.DOWN_KS
        )
        draft_mlp.act_stage(mb, a_task, "act")
        xs = draft_mlp.proj_poll(mb, d_task, "act", I, draft_mlp.DOWN_KS, "down")
        draft_mlp.proj_compute(
            mb,
            d_task,
            tiles,
            xs,
            "dpart",
            H,
            I,
            draft_mlp.DOWN_ROWS,
            draft_mlp.DOWN_KS,
            "down",
        )
    for i in range(gl.where(has_act, 1, 0), d_rounds):
        draft_mlp.proj_task(
            mb,
            d_task + i * G,
            down_w,
            "act",
            "dpart",
            H,
            I,
            draft_mlp.DOWN_ROWS,
            draft_mlp.DOWN_KS,
            "down",
        )
    task, active = schedule.stage_task(PLAN, "fin")
    if active:
        draft_mlp.fin_stage(
            mb,
            task,
            m_base,
            out,
            "dpart",
            draft_mlp.DOWN_KPARTS,
            "cpart",
            draft_mlp.FIN_COLS,
            "fin",
        )
    trace.flush(table_, trace_buf)


BASES = {**attn_t.BASES, **{k: v for k, v in mlp_t.BASES.items() if k != "scat"}}
BASES["sum"] = BASES["isum"]


def plan(rows: int) -> Plan:
    return Plan(
        grid=256,
        stages=draft_layer.stages(BASES, SPLITS, rows),
        regions=draft_layer.regions(SPLITS),
        rows=rows,
        chain=rows,
        world=WORLD,
        peers=draft_layer.PEERS,
    )


def layer_inputs(prefix: int, rank: int, rows: int, device):
    """Every input of one launch on ``rank``: the shared tensors from the two
    halves' tests, this rank's weights, and the paged context cache."""
    parts, residual, in_gain, a_coef, a_base, q_norm, k_norm = attn_t._half_shared(
        prefix, rows
    )
    qkv_w, o_w, ctx_k, ctx_v = attn_t._half_rank(prefix, rank, rows)
    _, _, post_gain, m_coef, m_base = mlp_t._shared(prefix, rows)
    gu_w, down_w = mlp_t._rank_weights(prefix, rank)
    end = prefix + rows
    pages = (end + PAGE - 1) // PAGE
    g = torch.Generator().manual_seed(prefix)
    table = torch.randperm(4 * pages + 8, generator=g)[: pages + 1].to(torch.int32)
    k_cache = torch.zeros((4 * pages + 8) * PAGE, 1, HEAD, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    tok = torch.arange(prefix)
    cslot = table[tok // PAGE].long() * PAGE + tok % PAGE
    k_cache[cslot, 0] = ctx_k[:prefix]
    v_cache[cslot, 0] = ctx_v[:prefix]
    btok = prefix + torch.arange(rows)
    slots = table[btok // PAGE].long() * PAGE + btok % PAGE
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    host = dict(
        parts=parts,
        residual=residual,
        in_gain=in_gain,
        post_gain=post_gain,
        a_coef=a_coef,
        a_base=a_base,
        m_coef=m_coef,
        m_base=m_base,
        q_norm=q_norm,
        k_norm=k_norm,
        qkv_w=qkv_w,
        o_w=o_w,
        gu_w=gu_w,
        down_w=down_w,
        ctx_k=ctx_k,
        ctx_v=ctx_v,
        positions=btok.to(torch.int32),
        slots=slots,
        cos_sin=attn_t.rope_table(end + 8),
        end=end,
    )
    args = [
        parts[rank].to(device),
        residual.to(device),
        torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        in_gain[own].float().to(device),
        bf16_tiles(a_coef).to(device),
        a_base.to(device),
        bf16_tiles(qkv_w).to(device),
        q_norm.to(device),
        k_norm.to(device),
        host["cos_sin"].to(device),
        host["positions"].to(device),
        slots.to(device),
        table.to(device),
        torch.full((1,), end, dtype=torch.int32, device=device),
        k_cache.to(device),
        v_cache.to(device),
        bf16_tiles(o_w).to(device),
        post_gain[own].float().to(device),
        bf16_tiles(m_coef).to(device),
        m_base.to(device),
        bf16_tiles(gu_w).to(device),
        bf16_tiles(down_w).to(device),
        torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device),
    ]
    return host, args


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    """Within one BF16 step, or a step of the largest value where a sum
    cancels to near zero."""
    torch.testing.assert_close(
        got.float(),
        want.float(),
        rtol=2**-7,
        atol=want.float().abs().max().item() * 2**-9,
    )


def _worker(rank: int, port: int, prefix: int) -> None:
    rows = 8
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        p = plan(rows)
        scratch = ctx.zeros((p.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        host, args = layer_inputs(prefix, rank, rows, device)
        step = 31
        layer_kernel[(p.grid,)](
            scratch,
            torch.full((1,), step, dtype=torch.int32, device=device),
            group.bases,
            *args,
            None,
            p.key,
            rank,
            SPLITS,
            num_warps=8,
        )
        torch.cuda.synchronize()
        tag = launch_tag(step, 0)
        residual_out, out = args[2], args[-1]

        def words(name, width, copy=0):
            return torch.stack(
                [
                    payloads(scratch, p, name, width, tag, row=r, copy=copy).cpu()
                    for r in range(rows)
                ]
            )

        def bf(name, width, copy=0):
            return words(name, width, copy).view(torch.bfloat16)

        # The attention half up to its output (the half's own test covers
        # its stages one by one), then this rank's attention partial.
        _, res1 = ref.tp_sum_norm(
            host["parts"], host["residual"], host["in_gain"], round_tp_sum=True
        )
        h = bf("ahin", HIDDEN // 2, copy=step & 1)
        cpart = words("acpart", 2 * ref.COEFFICIENTS).view(torch.float32)
        adl = (cpart[:, : ref.COEFFICIENTS] + cpart[:, ref.COEFFICIENTS :]).to(
            torch.bfloat16
        )
        _close(adl, (h.float() @ host["a_coef"].float().T).to(torch.bfloat16))
        a_delta = adl.view(rows, 2, ref.TAPS, ref.GROUPS)
        ao = bf("ao", Q_HEADS * HALF).view(rows, -1)
        attn_part = ref.grouped_conv(
            (ao.float() @ host["o_w"].float().T).to(torch.bfloat16),
            a_delta[:, 1],
            host["a_base"][1],
        )
        # The MLP half's input: every rank's partial, summed in rank order.
        gathered = [torch.empty_like(attn_part) for _ in range(WORLD)]
        dist.all_gather(gathered, attn_part)
        want_h, want_res = ref.tp_sum_norm(
            gathered, res1, host["post_gain"], round_tp_sum=True
        )
        _close(residual_out[:, own].cpu(), want_res[:, own])
        hin = bf("hin", HIDDEN // 2, copy=step & 1)
        _close(hin, want_h)
        # The MLP half's stages on the kernel's own input.
        mcp = words("cpart", 2 * ref.COEFFICIENTS).view(torch.float32)
        mdl = (mcp[:, : ref.COEFFICIENTS] + mcp[:, ref.COEFFICIENTS :]).to(
            torch.bfloat16
        )
        _close(mdl, (hin.float() @ host["m_coef"].float().T).to(torch.bfloat16))
        m_delta = mdl.view(rows, 2, ref.TAPS, ref.GROUPS)
        hc = bf("hc", HIDDEN // 2)
        _close(hc, ref.grouped_conv(hin, m_delta[:, 0], host["m_base"][0]))
        act = bf("act", ref.INTER_RANK // 2)
        gate, up = (
            (hc.float() @ host["gu_w"].float().T)
            .to(torch.bfloat16)
            .float()
            .split(ref.INTER_RANK, dim=-1)
        )
        _close(act, (torch.nn.functional.silu(gate) * up).to(torch.bfloat16))
        partial = (act.float() @ host["down_w"].float().T).to(torch.bfloat16)
        _close(out.cpu(), ref.grouped_conv(partial, m_delta[:, 1], host["m_base"][1]))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("prefix", [3000])
def test_draft_layer_in_one_launch_hands_the_attention_partial_to_the_mlp(
    prefix: int,
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, prefix), nprocs=WORLD, join=True)
