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

"""Muse Spark MXFP4 experts across the 8 ranks, on the regular path's
preprocessed expert weights."""

from __future__ import annotations

import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monokernel_testing import fill, payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.numerics.torch.quantization import (  # noqa: E402
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (  # noqa: E402
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_experts,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

# 48 experts: 8 rows of random picks share about 37 of them, as many as a
# verify's 8 consecutive tokens, so the slot lanes run several slots.
WORLD, POOLED, INTER, EXPERTS, TOP_K, OWNED = 8, 4096, 512, 48, 8, 512


@gluon.jit
def experts_kernel(
    scratch,
    step,
    bases,
    w13,
    w13s,
    w2,
    w2s,
    s13_we,
    s13_se,
    s13_lin,
    s13_nb,
    s2_we,
    s2_se,
    s2_lin,
    s2_nb,
    pre_gain,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    # A CTA's ex1 and ex2 tasks run in parts (see moe_experts): both weight
    # streams once routing names the experts, then the A operand (xq).
    task, active = schedule.stage_task(PLAN, "slt")
    if active:
        moe_experts.slot_table_stage(mb, "slt")
    task, active = schedule.stage_task(PLAN, "xq")
    if active:
        moe_experts.xq_stage(mb, task, pre_gain, "xq")
    task1, active1 = schedule.stage_task(PLAN, "ex1")
    task1 = gl.where(active1, task1, 0)
    task2, active2 = schedule.stage_task(PLAN, "ex2")
    task2 = gl.where(active2, task2, 0)
    ids, e1, e2 = moe_experts.ex1_poll_routes(mb, task1, task2, active1, "ex1")
    loads1 = moe_experts.ex1_issue(
        task1, active1, e1, w13, w13s, s13_we, s13_se, s13_lin, s13_nb
    )
    loads2 = moe_experts.ex2_issue(
        mb, task2, active2, e2, w2, w2s, s2_we, s2_se, s2_lin, s2_nb, "ex2"
    )
    union = moe_experts.union_of(ids)
    a1 = moe_experts.ex1_xq(mb, task1, active1, "ex1")
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
    task, active = schedule.stage_task(PLAN, "exr")
    if active:
        moe_experts.exr_stage(mb, task, union, False, "exr")
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


def _raw_experts(rank: int):
    """Random MXFP4 expert shards of ``rank``: w13 rows interleave gate / up."""
    g = torch.Generator().manual_seed(500 + rank)
    w13 = torch.randint(
        0, 256, (EXPERTS, 2 * INTER, POOLED // 2), dtype=torch.uint8, generator=g
    )
    s13 = torch.randint(
        122, 127, (EXPERTS, 2 * INTER, POOLED // 32), dtype=torch.uint8, generator=g
    )
    w2 = torch.randint(
        0, 256, (EXPERTS, POOLED, INTER // 2), dtype=torch.uint8, generator=g
    )
    s2 = torch.randint(
        118, 123, (EXPERTS, POOLED, INTER // 32), dtype=torch.uint8, generator=g
    )
    return w13, s13, w2, s2


def _preprocessed(rank: int, device: torch.device):
    w13, s13, w2, s2 = _raw_experts(rank)
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    for name, t in (
        ("w13_weight", w13),
        ("w13_weight_scale", s13),
        ("w2_weight", w2),
        ("w2_weight_scale", s2),
    ):
        setattr(module, name, torch.nn.Parameter(t.to(device), requires_grad=False))
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return (
        _extract_gluon_raw_w(module.w13_weight_triton_tensor),
        _extract_gluon_raw_s(module.w13_precision_config.b_mx_scale),
        _extract_gluon_raw_w(module.w2_weight_triton_tensor),
        _extract_gluon_raw_s(module.w2_precision_config.b_mx_scale),
    )


def _inputs(launch: int, rows: int):
    """Each row's pooled input and 8 picks (rows ``[rows, ...]``), and the
    pre-expert gain."""
    g = torch.Generator().manual_seed(77 + launch)
    pooled = torch.randn(rows, POOLED, generator=g).to(torch.bfloat16)
    gain = (1.0 + 0.1 * torch.randn(POOLED, generator=g)).to(torch.bfloat16).float()
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(rows)]
    ).to(torch.int32)
    return pooled, gain, ids


def _reference(rank: int, pooled, gain, ids, cols, device):
    """Rank ``rank``'s BF16 partials ``[rows, 8, cols]`` of every row's 8
    picks (output columns ``cols``), computed on ``device``. Each picked
    expert is dequantized once."""
    x = pooled.float().to(device)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain.to(device)
    xq = fake_quantize_mxfp4_even(x.to(torch.bfloat16).float())
    w13, s13, w2, s2 = (t.to(device) for t in _raw_experts(rank))
    out = torch.empty(ids.shape[0], TOP_K, cols.stop - cols.start, device=device)
    for e in ids.unique().tolist():
        up = dequantize_mxfp4(w13[e], s13[e]).float()
        down = dequantize_mxfp4(w2[e][cols], s2[e][cols]).float()
        for row, k in (ids == e).nonzero().tolist():
            acc = up @ xq[row]
            h = acc[0::2] / (1.0 + torch.exp(-acc[0::2])) * acc[1::2]
            hq = fake_quantize_mxfp4_even(h.to(torch.bfloat16).float().view(1, -1))
            out[row, k] = down @ hq.view(-1)
    return out.to(torch.bfloat16).cpu()


def _worker(rank: int, port: int, rows: int, exr: int, exr_parts: int) -> None:
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
            # The layer's placements of exr (layer.BASES, CHAINED_BASES and
            # NEXT_BASES).
            stages=moe_experts.stages(
                {"slt": 255, "xq": 240, "ex1": 0, "exr": exr, "ex2": 0},
                rows,
                exr_parts,
            ),
            regions=(("route", 2 * TOP_K), ("cg", POOLED // 2)) + moe_experts.regions(),
            rows=rows,
            chain=rows,
            external=("route", "cg"),
            world=WORLD,
            peers=("cg",) + moe_experts.PEERS,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        w13, s13, w2, s2 = _preprocessed(rank, device)
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
                s13,
                w2,
                s2,
                w13.stride(0),
                s13.stride(0),
                s13.stride(1),
                s13.stride(2),
                w2.stride(0),
                s2.stride(0),
                s2.stride(1),
                s2.stride(2),
                gain.to(device),
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            dist.barrier()
            want = [
                _reference(src, pooled, gain, ids, own, device) for src in range(WORLD)
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
                        got[src], w, rtol=2**-6, atol=2**-6 * float(w.abs().max())
                    )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows, exr, exr_parts", [(1, 192, 1), (8, 0, 4), (8, 128, 2)])
def test_experts_match_the_regular_path_math_on_its_weight_layout(
    rows: int, exr: int, exr_parts: int
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows, exr, exr_parts), nprocs=WORLD, join=True)
