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

"""A whole Muse Spark decode layer as one persistent launch per TP8 rank."""

from __future__ import annotations

import socket
from datetime import timedelta

import muse_spark_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monokernel_testing import payloads
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import layer  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, POOLED, OWNED, EXPERTS = 8, 8192, 4096, 1024, 16
QKVG, Q = 2304, 1024


def _shared(seed: int, length: int):
    """Inputs every rank shares."""
    g = torch.Generator().manual_seed(seed)

    def gain(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16).float()

    a_in = torch.randn(HIDDEN, generator=g).to(torch.bfloat16)
    h = torch.randn(HIDDEN, generator=g) * 2
    gates = [torch.rand(HIDDEN, generator=g) + 0.5 for _ in range(4)]
    router = torch.randn(256, HIDDEN, generator=g) / HIDDEN**0.5
    bias = torch.randn(256, generator=g) * 0.1
    bias[EXPERTS:] = -1e4  # the test holds 16 experts per rank
    pages = (length + 63) // 64
    table = (torch.randperm(pages + 3, generator=g)[:pages] + 1).to(torch.int32)
    return dict(
        a_in=a_in,
        h=h,
        alpha1=gates[0],
        beta1=gates[1],
        alpha2=gates[2],
        beta2=gates[3],
        pre_gain=gain(HIDDEN),
        pre_exp_gain=gain(POOLED),
        pen_gain=gain(POOLED),
        post_gain=gain(HIDDEN),
        in_gain=gain(HIDDEN),
        router=router,
        bias=bias,
        table=table,
    )


def _rank_weights(rank: int, length: int):
    g = torch.Generator().manual_seed(1000 + rank)

    def bf(*shape, k):
        return (torch.randn(*shape, generator=g) / k**0.5).to(torch.bfloat16)

    slots = ((length + 63) // 64 + 5) * 64
    return dict(
        w_qkvg=bf(QKVG, HIDDEN, k=HIDDEN),
        w_o=bf(HIDDEN, Q, k=Q),
        w_pool=bf(512, HIDDEN, k=HIDDEN),
        w_post=bf(OWNED, POOLED, k=POOLED),
        k_cache=torch.randn(slots, 2, 64, generator=g).to(torch.bfloat16),
        v_cache=torch.randn(slots, 2, 64, generator=g).to(torch.bfloat16),
    )


def _reference(sh, length, full, kernel_pooled, kernel_hp):
    """The layer's pooled row, FP32 residual ``h''`` and next BF16 attention
    input.

    MXFP4 quantization is discontinuous: a pooled value one BF16 step from
    the reference's can quantize one E2M1 step away and move every expert
    output a little. So the pooled row is checked on its own, and the expert
    math downstream of it starts from the kernel's ``kernel_pooled``. In the
    same way the FFN residual gate starts from the kernel's ``h'``
    (``kernel_hp``), which is checked on its own.
    """
    window = 0 if full else layer.WINDOW
    splits = layer.FULL_SPLITS if full else layer.SLIDING_SPLITS
    cs = ref.rope_table(length + 1)
    b = torch.zeros(HIDDEN)
    for r in range(WORLD):
        w = _rank_weights(r, length)
        qkvg = (sh["a_in"].float() @ w["w_qkvg"].float().T).to(torch.bfloat16)
        z, _, _, _ = ref.attention_z(
            qkvg, w["k_cache"], w["v_cache"], sh["table"], length, window, splits, cs
        )
        b = b + (z.float() @ w["w_o"].float().T).to(torch.bfloat16).float()
    b = b.to(torch.bfloat16).float()
    b = b * torch.rsqrt(b.pow(2).mean() + 1e-8)
    hp = sh["beta1"] * b + sh["alpha1"] * sh["h"]
    u = hp * torch.rsqrt(hp.pow(2).mean() + 1e-5) * sh["pre_gain"]
    score = torch.sigmoid((sh["router"].double() @ u.double()) * 0.17677669529663687)
    ids = torch.topk(score + sh["bias"].double(), 8).indices
    weights = (score[ids] / score[ids].sum()).float()
    w_pool = torch.cat([_rank_weights(r, length)["w_pool"] for r in range(WORLD)])
    pooled = (u.to(torch.bfloat16).float() @ w_pool.float().T).to(torch.bfloat16)
    routes = torch.zeros(8, POOLED)
    for r in range(WORLD):
        routes = (
            routes
            + ref.expert_partials(
                r, EXPERTS, kernel_pooled, sh["pre_exp_gain"], ids
            ).float()
        )
    p = routes.to(torch.bfloat16).float() * sh["pen_gain"]
    p = p * torch.rsqrt(p.pow(2).mean(1, keepdim=True) + 1e-8)
    combined = (p * weights[:, None]).sum(0).to(torch.bfloat16)
    w_post = torch.cat([_rank_weights(r, length)["w_post"] for r in range(WORLD)])
    y = (combined.float() @ w_post.float().T).to(torch.bfloat16).float() * sh[
        "post_gain"
    ]
    y = y * torch.rsqrt(y.pow(2).mean() + 1e-8)
    h2 = sh["beta2"] * y + sh["alpha2"] * kernel_hp
    a = (h2 * torch.rsqrt(h2.pow(2).mean() + 1e-5) * sh["in_gain"]).to(torch.bfloat16)
    return pooled, hp, (sh["beta1"] * b).abs(), h2, a, (sh["beta2"] * y).abs()


def _worker(rank: int, port: int, length: int, full: bool) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=900),
    )
    try:
        dev = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
        plan = layer.layer_plan(full)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        sh = _shared(42, length)
        w = _rank_weights(rank, length)
        w13, w13s, w2, w2s = ref.preprocessed_experts(rank, EXPERTS, dev)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        own_pool = slice(rank * 512, (rank + 1) * 512)
        L = length - 1
        slot = int(sh["table"][L // 64]) * 64 + L % 64
        k_cache, v_cache = w["k_cache"].to(dev), w["v_cache"].to(dev)
        h_next = torch.empty(OWNED, device=dev)
        a_out = torch.empty(HIDDEN, dtype=torch.bfloat16, device=dev)
        attn_ops = (
            bf16_tiles(w["w_qkvg"]).to(dev),
            sh["a_in"].to(dev),
            torch.full((1,), length, dtype=torch.int32, device=dev),
            torch.full((1,), slot, dtype=torch.int32, device=dev),
            sh["table"].to(dev),
            k_cache,
            v_cache,
            ref.rope_table(length + 1).to(dev),
            bf16_tiles(w["w_o"]).to(dev),
            sh["h"][own].to(dev),
            sh["alpha1"][own].to(dev),
            sh["beta1"][own].to(dev),
            sh["pre_gain"][own].to(dev),
        )
        moe_ops = (
            sh["router"][:, own].contiguous().to(dev),
            sh["bias"].to(dev),
            bf16_tiles(w["w_pool"]).to(dev),
            w13,
            w13s,
            w13.stride(0),
            w13s.stride(0),
            w13s.stride(1),
            w13s.stride(2),
            w2,
            w2s,
            w2.stride(0),
            w2s.stride(0),
            w2s.stride(1),
            w2s.stride(2),
            sh["pre_exp_gain"].to(dev),
            sh["pen_gain"][own_pool].to(dev),
            bf16_tiles(w["w_post"]).to(dev),
            sh["post_gain"][own].to(dev),
            sh["alpha2"][own].to(dev),
            sh["beta2"][own].to(dev),
            sh["in_gain"][own].to(dev),
            h_next,
            a_out,
        )
        group.finish_prepare()
        lay = 1 if full else 2
        layer.launch_layer(
            scratch,
            torch.full((1,), 3, dtype=torch.int32, device=dev),
            lay,
            group.bases,
            attn_ops,
            moe_ops,
            None,
            rank,
            full,
        )
        torch.cuda.synchronize()
        dist.barrier()
        kernel_pooled = (
            payloads(scratch, plan, "cg", POOLED // 2, launch_tag(3, lay), copy=lay & 1)
            .view(torch.bfloat16)
            .cpu()
        )
        hp_own = (
            payloads(scratch, plan, "hp", OWNED, launch_tag(3, lay))
            .view(torch.float32)
            .cpu()
        )
        slices = [torch.empty(OWNED) for _ in range(WORLD)]
        dist.all_gather(slices, hp_own)
        kernel_hp = torch.cat(slices)
        pooled, hp, gated1, h2, a, gated = _reference(
            sh, length, full, kernel_pooled, kernel_hp
        )
        # h' sums 8 ranks' BF16 partials: where they cancel, a step of one
        # partial is an absolute error at the row's scale, which the norm
        # carries into beta b.
        err1 = (hp_own - hp[own]).abs()
        bound1 = 2**-7 * hp[own].abs() + 2**-6 * float(gated1.max())
        assert torch.all(err1 <= bound1), float(err1.max())
        # The pooled row is the BF16 projection of bf16(u), which carries the
        # attention tail's BF16 steps: up to two steps, and near zero an
        # absolute error of the row's scale (the sum cancels).
        atol = 2**-9 * float(pooled.float().abs().max())
        torch.testing.assert_close(
            kernel_pooled.float(), pooled.float(), rtol=2**-6, atol=atol
        )
        err = (h_next.cpu() - h2[own]).abs()
        # In the same way, a BF16 step of the projection is an absolute
        # error at the scale of beta y.
        bound = 2**-7 * h2[own].abs() + 2**-6 * float(gated.max())
        assert torch.all(err <= bound), float(err.max())
        # The next input is BF16 of h'' after a norm: the same row-scale error.
        atol = 2**-8 * float(a.float().abs().max())
        torch.testing.assert_close(
            a_out.float().cpu(), a.float(), rtol=2**-6, atol=atol
        )
        dist.barrier()
        # Other layers run the same binary: the index is a runtime argument.
        # Each step flips the peer copy (layer & 1), as serving does.
        with assert_no_triton_compile(layer.gluon_muse_spark_layer_gfx950):
            for other in (lay + 1, lay + 18, lay + 35):
                layer.launch_layer(
                    scratch,
                    torch.full((1,), 3, dtype=torch.int32, device=dev),
                    other,
                    group.bases,
                    attn_ops,
                    moe_ops,
                    None,
                    rank,
                    full,
                )
            torch.cuda.synchronize()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("full", [False, True])
def test_layer_matches_the_regular_path_math(full):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, 3000, full), nprocs=WORLD, join=True)
