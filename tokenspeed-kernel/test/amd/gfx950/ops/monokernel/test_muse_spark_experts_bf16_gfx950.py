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

"""Muse Spark BF16 experts (moe_experts_bf16) across the 8 ranks, on the
regular BF16 route path's weight layout."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monokernel_testing import fill, payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_experts,
    moe_experts_bf16,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

# 48 experts: 8 rows of random picks share about 37 of them, so the slot
# lanes run several slots.
WORLD, POOLED, INTER, EXPERTS, TOP_K, OWNED = 8, 4096, 512, 48, 8, 512


@gluon.jit
def experts_kernel(
    scratch,
    step,
    bases,
    w13,
    w2,
    s13_we,
    s2_we,
    pre_gain,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "slt")
    if active:
        moe_experts.slot_table_stage(mb, "slt")
    task1, active1 = schedule.stage_task(PLAN, "ex1")
    task1 = gl.where(active1, task1, 0)
    task2, active2 = schedule.stage_task(PLAN, "ex2")
    task2 = gl.where(active2, task2, 0)
    got1, gain_lo, gain_hi, ids, e1, e2 = moe_experts.ex1_poll(
        mb, task1, task2, active1, pre_gain, "ex1"
    )
    loads1 = moe_experts_bf16.ex1_issue(task1, active1, e1, w13, s13_we)
    union = moe_experts.union_of(ids)
    a1 = moe_experts_bf16.ex1_operand(task1, got1, gain_lo, gain_hi)
    if active1:
        moe_experts_bf16.ex1_finish(mb, task1, a1, loads1, union, w13, s13_we, "ex1")
    task, active = schedule.stage_task(PLAN, "exr")
    if active:
        moe_experts.exr_stage(mb, task, union, True, "exr")
    loads2 = moe_experts_bf16.ex2_issue(mb, task2, active2, e2, w2, s2_we, "ex2")
    if active2:
        moe_experts_bf16.ex2_finish(mb, task2, loads2, union, w2, s2_we, RANK, "ex2")


def _experts(rank: int):
    """Random BF16 expert shards of ``rank`` in the regular route path's
    layout: w13 ``[E, 2 I, H]`` (gate rows, then up rows), w2 ``[E, H, I]``."""
    g = torch.Generator().manual_seed(500 + rank)
    w13 = (torch.randn(EXPERTS, 2 * INTER, POOLED, generator=g) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(EXPERTS, POOLED, INTER, generator=g) * 0.04).to(torch.bfloat16)
    return w13, w2


def _inputs(launch: int, rows: int):
    """Each row's pooled input and 8 picks, and the pre-expert gain."""
    g = torch.Generator().manual_seed(77 + launch)
    pooled = torch.randn(rows, POOLED, generator=g).to(torch.bfloat16)
    gain = (1.0 + 0.1 * torch.randn(POOLED, generator=g)).to(torch.bfloat16).float()
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(rows)]
    ).to(torch.int32)
    return pooled, gain, ids


def _reference(experts, pooled, gain, ids, cols, device):
    """BF16 partials ``[rows, 8, cols]`` of every row's 8 picks (output
    columns ``cols``) with the regular route path's rounding: the expert
    input and the SwiGLU intermediate in BF16, FP32 products."""
    x = pooled.float().to(device)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain.to(device)
    x = x.to(torch.bfloat16).float()
    w13, w2 = (t.to(device) for t in experts)
    out = torch.empty(ids.shape[0], TOP_K, cols.stop - cols.start, device=device)
    for e in ids.unique().tolist():
        gate, up = w13[e, :INTER].float(), w13[e, INTER:].float()
        down = w2[e, cols].float()
        for row, k in (ids == e).nonzero().tolist():
            g_, u = gate @ x[row], up @ x[row]
            h = (g_ / (1.0 + torch.exp(-g_)) * u).to(torch.bfloat16).float()
            out[row, k] = down @ h
    return out.to(torch.bfloat16).cpu()


def _worker(rank: int, port: int, rows: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=600),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        plan = Plan(
            grid=256,
            # The layer's placement: exr over every CTA past one row.
            stages=moe_experts.stages(
                {
                    "slt": 255,
                    "xq": 240,
                    "ex1": 0,
                    "exr": 192 if rows == 1 else 0,
                    "ex2": 0,
                },
                rows,
                1 if rows == 1 else int(moe_experts.EXR_PARTS),
                True,
            ),
            regions=(("route", 2 * TOP_K), ("cg", POOLED // 2)) + moe_experts.regions(),
            rows=rows,
            chain=rows,
            external=("route", "cg"),
            world=WORLD,
            peers=("cg",) + moe_experts.PEERS,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        experts = [_experts(r) for r in range(WORLD)]
        w13, w2 = (t.to(device) for t in experts[rank])
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        for launch in range(2):
            step = 10 + launch
            tag = launch_tag(step, 0)
            pooled, gain, ids = _inputs(launch, rows)
            for row in range(rows):
                route = torch.cat(
                    [ids[row], torch.full((TOP_K,), 0.125).view(torch.int32)]
                )
                fill(scratch, plan, "route", route.to(device), tag, row=row)
                fill(
                    scratch,
                    plan,
                    "cg",
                    pooled[row].view(torch.int32).to(device),
                    tag,
                    copy=step & 1,
                    row=row,
                )
            experts_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                w13,
                w2,
                w13.stride(0),
                w2.stride(0),
                gain.to(device),
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            dist.barrier()
            want = [
                _reference(experts[src], pooled, gain, ids, own, device)
                for src in range(WORLD)
            ]
            for row in range(rows):
                got = payloads(
                    scratch,
                    plan,
                    "rr",
                    WORLD * TOP_K * OWNED // 2,
                    tag,
                    row=row,
                    copy=step & 1,
                )
                got = got.view(torch.bfloat16).float().view(WORLD, TOP_K, OWNED).cpu()
                for src in range(WORLD):
                    w = want[src][row].float()
                    torch.testing.assert_close(
                        got[src], w, rtol=2**-7, atol=2**-7 * float(w.abs().max())
                    )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_bf16_experts_match_the_route_path_math_on_its_weight_layout(rows: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows), nprocs=WORLD, join=True)
