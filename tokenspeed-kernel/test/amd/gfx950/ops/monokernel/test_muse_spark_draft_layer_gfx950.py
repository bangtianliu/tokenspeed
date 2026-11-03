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

import contextlib
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
from utils import assert_no_triton_compile, is_cdna4

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


def plan(rows: int, last: bool) -> Plan:
    return draft_layer.layer_plan(rows, last)


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
    weights = draft_layer.pack_draft_layer(
        {
            "input_norm": in_gain,
            "post_norm": post_gain,
            "attn_coef": a_coef,
            "mlp_coef": m_coef,
            "attn_base": a_base,
            "mlp_base": m_base,
            "qkv": qkv_w,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "o_proj": o_w,
            "gate_up": gu_w,
            "down": down_w,
        },
        rank,
    )
    weights = draft_layer.DraftLayerWeights(
        rank=rank,
        **{
            f: getattr(weights, f).to(device)
            for f in weights.__dataclass_fields__
            if f != "rank"
        },
    )
    runtime = dict(
        x=parts[rank].to(device),
        residual=residual.to(device),
        residual_out=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        out=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device),
        final_gain=None,
        normed=None,
        seq_len=torch.full((1,), end, dtype=torch.int32, device=device),
        positions=host["positions"].to(device),
        slots=slots.to(device),
        page_table=table.to(device),
        k_cache=k_cache.to(device),
        v_cache=v_cache.to(device),
        cos_sin=host["cos_sin"].to(device),
    )
    return host, weights, runtime


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
        # The last layer's plan is the larger; both share the scratch.
        p = plan(rows, True)
        scratch = ctx.zeros((p.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        step = 31
        for layer in (2, 3, 4):
            host, weights, rt = layer_inputs(prefix + layer, rank, rows, device)
            last = layer == 4
            if last:
                # The last layer runs the draft's final norm in place of
                # storing its partial.
                g = torch.Generator().manual_seed(prefix)
                final_w = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(
                    torch.bfloat16
                )
                host["final_w"] = final_w
                rt["out"] = None
                rt["final_gain"] = final_w[own].float().to(device)
                rt["normed"] = torch.zeros(
                    rows, HIDDEN, dtype=torch.bfloat16, device=device
                )
            # The next layer, at another length, runs the first one's binary.
            no_compile = (
                assert_no_triton_compile(
                    draft_layer.gluon_muse_spark_draft_layer_gfx950
                )
                if layer == 3
                else contextlib.nullcontext()
            )
            with no_compile:
                draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
                    weights,
                    **rt,
                    scratch=scratch,
                    step=torch.full((1,), step, dtype=torch.int32, device=device),
                    peer_bases=group.bases,
                    layer=layer,
                    trace_buf=None,
                    compile_only=False,
                )
            torch.cuda.synchronize()
            _check(plan(rows, last), scratch, step, layer, host, rt, rank, rows)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _check(p, scratch, step, layer, host, rt, rank, rows) -> None:
    """Every stage of one launch on the kernel's own input."""
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    tag = launch_tag(step, layer)
    copy = (step * 5 + layer) & 1
    residual_out = rt["residual_out"]

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
    h = bf("ahin", HIDDEN // 2, copy=copy)
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
    hin = bf("hin", HIDDEN // 2, copy=copy)
    _close(hin, want_h)
    # The MLP half's stages on the kernel's own input.
    mcp = words("cpart", 2 * ref.COEFFICIENTS).view(torch.float32)
    mdl = (mcp[:, : ref.COEFFICIENTS] + mcp[:, ref.COEFFICIENTS :]).to(torch.bfloat16)
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
    mlp_part = ref.grouped_conv(partial, m_delta[:, 1], host["m_base"][1])
    if rt["normed"] is None:
        _close(rt["out"].cpu(), mlp_part)
        return
    # The last layer: the final norm of the ranks' MLP partials and the
    # residual after the attention sublayer, the whole rows on every rank.
    parts = [torch.empty_like(mlp_part) for _ in range(WORLD)]
    dist.all_gather(parts, mlp_part)
    want_normed, _ = ref.tp_sum_norm(
        parts, want_res, host["final_w"], round_tp_sum=True
    )
    _close(rt["normed"].cpu(), want_normed)


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
