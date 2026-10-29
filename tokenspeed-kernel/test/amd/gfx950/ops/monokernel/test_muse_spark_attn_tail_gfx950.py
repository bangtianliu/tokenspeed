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

"""Muse Spark output projection and attention tail across the 8 ranks."""

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
    attn_tail,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, Q, OWNED = 8, 8192, 1024, 1024
LAUNCHES = 2


@gluon.jit
def tail_kernel(
    scratch,
    step,
    bases,
    w,
    h,
    h_out,
    alpha,
    beta,
    pre_gain,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        attn_tail.oproj_stage(mb, task, w, RANK, "oproj")
    task, active = schedule.stage_task(PLAN, "tail1")
    if active:
        attn_tail.tail_stages(mb, task, h, h_out, alpha, beta, pre_gain, RANK)


def _plan() -> Plan:
    return Plan(
        grid=256,
        stages=attn_tail.stages({"oproj": 0, "tail": 128}),
        regions=(("z", Q // 2),) + attn_tail.regions(),
        rows=1,
        chain=1,
        external=("z",),
        world=WORLD,
        peers=attn_tail.PEERS,
    )


def _rank_inputs(rank: int, launch: int):
    g = torch.Generator().manual_seed(100 * launch + rank)
    z = torch.randn(Q, generator=g).to(torch.bfloat16)
    w = (torch.randn(HIDDEN, Q, generator=g) / Q**0.5).to(torch.bfloat16)
    return z, w


def _shared_inputs(launch: int):
    g = torch.Generator().manual_seed(9000 + launch)
    h = torch.randn(HIDDEN, generator=g) * 3
    alpha = torch.rand(HIDDEN, generator=g) + 0.5
    beta = torch.rand(HIDDEN, generator=g) + 0.5
    pre_gain = (torch.randn(HIDDEN, generator=g) * 0.1 + 1.0).to(torch.bfloat16).float()
    return h, alpha, beta, pre_gain


def _reference(launch: int):
    b = torch.zeros(HIDDEN)
    for r in range(WORLD):
        z, w = _rank_inputs(r, launch)
        b = b + (z.float() @ w.float().T).to(torch.bfloat16).float()
    b = b.to(torch.bfloat16).float()
    b = b * torch.rsqrt(b.pow(2).mean() + 1e-8)
    h, alpha, beta, pre_gain = _shared_inputs(launch)
    hp = beta * b + alpha * h
    u = hp * torch.rsqrt(hp.pow(2).mean() + 1e-5) * pre_gain
    return hp, u


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
        plan = _plan()
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        # Each launch has new inputs and a new step (the peer regions'
        # copies alternate). z is this launch's input, filled by the host.
        for k in range(LAUNCHES):
            step = 7 + k
            z, w = _rank_inputs(rank, k)
            h, alpha, beta, pre_gain = _shared_inputs(k)
            fill(
                scratch, plan, "z", z.view(torch.int32).to(device), launch_tag(step, 0)
            )
            h_out = torch.empty(OWNED, device=device)
            tail_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                bf16_tiles(w).to(device),
                h[own].to(device),
                h_out,
                alpha[own].to(device),
                beta[own].to(device),
                pre_gain[own].to(device),
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            u = payloads(scratch, plan, "u", OWNED, launch_tag(step, 0)).view(
                torch.float32
            )
            want_hp, want_u = _reference(k)
            torch.testing.assert_close(h_out.cpu(), want_hp[own], rtol=2**-7, atol=1e-3)
            torch.testing.assert_close(u.cpu(), want_u[own], rtol=2**-7, atol=1e-3)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_attention_tail_reduces_scatters_and_normalizes_across_ranks():
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)
