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

"""Muse Spark expert routing across the 8 ranks (column-sharded router)."""

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
    moe_route,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

WORLD, HIDDEN, OWNED, EXPERTS, TOP_K = 8, 8192, 1024, 256, 8


@gluon.jit
def route_kernel(scratch, step, bases, w, bias, PLAN: gl.constexpr, RANK: gl.constexpr):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "router")
    if active:
        moe_route.router_stage(mb, task, w, RANK, "router")
    task, active = schedule.stage_task(PLAN, "topk")
    if active:
        moe_route.topk_stage(mb, bias, "topk")


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(HIDDEN, generator=g)
    w = torch.randn(EXPERTS, HIDDEN, generator=g) / HIDDEN**0.5
    bias = torch.randn(EXPERTS, generator=g) * 0.1
    return u, w, bias


def _worker(rank: int, port: int) -> None:
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
        plan = Plan(
            grid=256,
            stages=moe_route.stages({"router": 0, "topk": 16}),
            regions=(("u", OWNED),) + moe_route.regions(),
            rows=1,
            chain=1,
            external=("u",),
            world=WORLD,
            peers=moe_route.PEERS,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        for k, seed in enumerate((1, 2, 3)):
            step = 4 + k
            u, w, bias = _inputs(seed)
            fill(
                scratch,
                plan,
                "u",
                u[own].view(torch.int32).to(device),
                launch_tag(step, 0),
            )
            route_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                w[:, own].contiguous().to(device),
                bias.to(device),
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            got = payloads(scratch, plan, "route", 2 * TOP_K, launch_tag(step, 0)).cpu()
            logit = (w.double() @ u.double()) * 0.17677669529663687
            score = torch.sigmoid(logit)
            ids = torch.topk(score + bias.double(), TOP_K).indices
            weights = score[ids] / score[ids].sum()
            assert torch.equal(got[:TOP_K].long(), ids)
            torch.testing.assert_close(
                got[TOP_K:].view(torch.float32).double(), weights, rtol=1e-5, atol=1e-6
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_routing_sums_the_ranks_partial_logits_and_picks_the_top_8():
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)
