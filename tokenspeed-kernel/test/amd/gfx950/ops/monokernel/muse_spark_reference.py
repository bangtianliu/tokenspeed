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

"""Torch references of the Muse Spark decode math (the regular path's
rounding points) for the monokernel tests, one TP8 rank at a time."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)

HEAD, HEADS, KV_HEADS, GROUP, PAGE = 64, 16, 2, 8, 64
HIDDEN, POOLED, INTER, TOP_K, OWNED = 8192, 4096, 512, 8, 1024
K_ROW, V_ROW, GATE_ROW = 1024, 1152, 1280


def rope_table(positions: int) -> torch.Tensor:
    """FP32 ``[positions, 64]``: cos of the 32 pairs, then sin (theta 500000)."""
    inv = 1.0 / 500000.0 ** (torch.arange(0, HEAD, 2, dtype=torch.float32) / HEAD)
    t = torch.arange(positions, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([t.cos(), t.sin()], dim=1)


def norm_rope(
    x: torch.Tensor, gain: float, cs: torch.Tensor, rope: bool
) -> torch.Tensor:
    """Per-head RMS norm, then interleaved RoPE: FP32, one BF16 rounding."""
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain
    if rope:
        c, s = cs[:32], cs[32:]
        lo, hi = y[..., 0::2], y[..., 1::2]
        y = torch.stack([lo * c - hi * s, hi * c + lo * s], dim=-1).flatten(-2)
    return y.to(torch.bfloat16)


def split_attention(q, keys, values, splits: int) -> torch.Tensor:
    """The monokernel's split attention: chunks of 128 keys per split with
    an online base-2 softmax whose weights round to BF16, then the LSE merge."""
    n = keys.shape[0]
    per = -(-n // splits)
    scores = (q.float() @ keys.T) * (0.125 * math.log2(math.e))
    outs, lses = [], []
    for s in range(splits):
        lo, hi = s * per, min(s * per + per, n)
        m, l, acc = float("-inf"), torch.zeros(()), torch.zeros(HEAD)
        for c0 in range(lo, hi, 128):
            sc = scores[c0 : min(c0 + 128, hi)]
            m_new = max(m, float(sc.max()))
            alpha = 2.0 ** (m - m_new)
            p = torch.exp2(sc - m_new)
            l = l * alpha + p.sum()
            acc = (
                acc * alpha + p.to(torch.bfloat16).float() @ values[c0 : c0 + p.numel()]
            )
            m = m_new
        if hi > lo:
            outs.append(acc / l)
            lses.append(m + math.log2(float(l)))
    lse = torch.tensor(lses)
    w = torch.exp2(lse - lse.max())
    return (torch.stack(outs) * w[:, None]).sum(0) / w.sum()


def attention_z(qkvg, k_cache, v_cache, table, length, window, splits, cs):
    """One rank's gated head-norm output ``z`` [1024] and the token's K, V.

    ``k_cache`` / ``v_cache``: BF16 ``[slots, 2, 64]`` before the token's write.
    """
    rope = window != 0
    L = length - 1
    q = norm_rope(qkvg[:K_ROW].view(HEADS, HEAD), 5.473006248474121, cs[L], rope)
    k = norm_rope(qkvg[K_ROW:V_ROW].view(KV_HEADS, HEAD), 1.0, cs[L], rope)
    v = qkvg[V_ROW:GATE_ROW].view(KV_HEADS, HEAD)
    gate = qkvg[GATE_ROW:].view(HEADS, HEAD).float()
    slot = int(table[L // PAGE]) * PAGE + L % PAGE
    flat_k, flat_v = k_cache.clone(), v_cache.clone()
    flat_k[slot], flat_v[slot] = k, v
    first = max(0, L - window + 1) if window else 0
    pos = torch.arange(first, L + 1)
    slots = table[pos // PAGE].long() * PAGE + pos % PAGE
    keys, values = flat_k[slots].float(), flat_v[slots].float()
    z = torch.empty(HEADS, HEAD)
    for h in range(HEADS):
        a = split_attention(q[h], keys[:, h // GROUP], values[:, h // GROUP], splits)
        a = a.to(torch.bfloat16).float()
        z[h] = a * torch.rsqrt(a.pow(2).mean() + 1e-5) * torch.sigmoid(gate[h])
    return z.reshape(-1).to(torch.bfloat16), k, v, slot


def raw_experts(rank: int, experts: int):
    """Random MXFP4 expert shards of ``rank``: w13 rows interleave gate / up."""
    g = torch.Generator().manual_seed(500 + rank)
    w13 = torch.randint(
        0, 256, (experts, 2 * INTER, POOLED // 2), dtype=torch.uint8, generator=g
    )
    s13 = torch.randint(
        122, 127, (experts, 2 * INTER, POOLED // 32), dtype=torch.uint8, generator=g
    )
    w2 = torch.randint(
        0, 256, (experts, POOLED, INTER // 2), dtype=torch.uint8, generator=g
    )
    s2 = torch.randint(
        118, 123, (experts, POOLED, INTER // 32), dtype=torch.uint8, generator=g
    )
    return w13, s13, w2, s2


def preprocessed_experts(rank: int, experts: int, device: torch.device):
    """``raw_experts`` through the regular path's MoE weight preprocessing."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )

    w13, s13, w2, s2 = raw_experts(rank, experts)
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


def expert_partials(rank: int, experts: int, pooled, gain, ids) -> torch.Tensor:
    """Rank ``rank``'s BF16 route partials ``[8, 4096]``."""
    x = pooled.float()
    x = x * torch.rsqrt(x.pow(2).mean() + 1e-5) * gain
    xq = fake_quantize_mxfp4_even(x.to(torch.bfloat16).float().view(1, -1)).view(-1)
    w13, s13, w2, s2 = raw_experts(rank, experts)
    out = torch.empty(TOP_K, POOLED)
    for k, e in enumerate(ids.tolist()):
        acc = dequantize_mxfp4(w13[e], s13[e]).float() @ xq
        h = acc[0::2] / (1.0 + torch.exp(-acc[0::2])) * acc[1::2]
        hq = fake_quantize_mxfp4_even(h.to(torch.bfloat16).float().view(1, -1)).view(-1)
        out[k] = dequantize_mxfp4(w2[e], s2[e]).float() @ hq
    return out.to(torch.bfloat16)
