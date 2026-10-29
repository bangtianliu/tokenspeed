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

"""Muse Spark attention stages (head, attn, merge) of one rank, one row."""

from __future__ import annotations

import math

import pytest
import torch
from monokernel_testing import fill, payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    attention,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

HEAD, HEADS, KV_HEADS, GROUP, PAGE = 64, 16, 2, 8, 64
QKVG = 2304
K_ROW, V_ROW, GATE_ROW = 1024, 1152, 1280
SLIDING_SPLITS, FULL_SPLITS = 16, 64


@gluon.jit
def attention_kernel(
    scratch,
    step,
    seq_len,
    slot,
    table,
    k_cache,
    v_cache,
    cos_sin,
    PLAN: gl.constexpr,
    SPLITS: gl.constexpr,
    WINDOW: gl.constexpr,
    ROPE: gl.constexpr,
):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        attention.head_stage(
            mb, task, seq_len, slot, k_cache, v_cache, cos_sin, ROPE, "head"
        )
    task, active = schedule.stage_task(PLAN, "attn")
    if active:
        attention.attn_stage(
            mb, task, seq_len, table, k_cache, v_cache, SPLITS, WINDOW, "attn"
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        attention.merge_stage(mb, task, SPLITS, "merge")


def _plan(splits: int) -> Plan:
    return Plan(
        grid=256,
        stages=attention.stages({"head": 0, "attn": 32, "merge": 18}, splits),
        regions=(("qkvg", QKVG // 2),) + attention.regions(splits),
        rows=1,
        chain=1,
        external=("qkvg",),
    )


def _rope_table(positions: int) -> torch.Tensor:
    inv = 1.0 / 500000.0 ** (torch.arange(0, HEAD, 2, dtype=torch.float32) / HEAD)
    t = torch.arange(positions, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([t.cos(), t.sin()], dim=1)


def _norm_rope(x, gain, cs, rope):
    """Reference per-head norm then interleaved RoPE: FP32, one BF16 rounding."""
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * gain
    if rope:
        c, s = cs[:32], cs[32:]
        lo, hi = y[..., 0::2], y[..., 1::2]
        y = torch.stack([lo * c - hi * s, hi * c + lo * s], dim=-1).flatten(-2)
    return y.to(torch.bfloat16)


def _split_attention(q, keys, values, splits):
    """The kernel's split attention: per split, chunks of 128 keys with an
    online base-2 softmax whose weights round to BF16 before the PV product,
    then the LSE merge of the splits."""
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


@pytest.mark.parametrize(
    "length,window,splits",
    [
        (6, 2048, SLIDING_SPLITS),
        (3000, 2048, SLIDING_SPLITS),
        (3000, 0, FULL_SPLITS),
        (9000, 0, FULL_SPLITS),
    ],
)
def test_attention_matches_the_regular_path_math(length, window, splits):
    rope = window != 0
    g = torch.Generator().manual_seed(length + window)
    L = length - 1  # the token's position
    pages = (length + PAGE - 1) // PAGE + 3
    perm = (
        torch.randperm(pages, generator=g)[: (length + PAGE - 1) // PAGE].to(
            torch.int32
        )
        + 1
    )
    table = perm.clone()
    k_cache = torch.randn(pages + 2, PAGE, KV_HEADS, HEAD, generator=g).to(
        torch.bfloat16
    )
    v_cache = torch.randn(pages + 2, PAGE, KV_HEADS, HEAD, generator=g).to(
        torch.bfloat16
    )
    qkvg = torch.randn(QKVG, generator=g).to(torch.bfloat16)
    cs = _rope_table(length + 1)
    slot = int(table[L // PAGE]) * PAGE + L % PAGE

    plan = _plan(splits)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    step = 5
    tag = launch_tag(step, 0)
    fill(scratch, plan, "qkvg", qkvg.view(torch.int32).cuda(), tag)
    kc, vc = k_cache.clone().cuda(), v_cache.clone().cuda()
    attention_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), step, dtype=torch.int32, device="cuda"),
        torch.full((1,), length, dtype=torch.int32, device="cuda"),
        torch.full((1,), slot, dtype=torch.int32, device="cuda"),
        table.cuda(),
        kc,
        vc,
        cs.cuda(),
        plan.key,
        splits,
        window,
        rope,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # Reference: the regular path's math with its BF16 rounding points.
    q = _norm_rope(qkvg[:K_ROW].view(HEADS, HEAD), 5.473006248474121, cs[L], rope)
    k = _norm_rope(qkvg[K_ROW:V_ROW].view(KV_HEADS, HEAD), 1.0, cs[L], rope)
    v = qkvg[V_ROW:GATE_ROW].view(KV_HEADS, HEAD)
    gate = qkvg[GATE_ROW:].view(HEADS, HEAD).float()
    flat_k = k_cache.view(-1, KV_HEADS, HEAD).clone()
    flat_v = v_cache.view(-1, KV_HEADS, HEAD).clone()
    flat_k[slot], flat_v[slot] = k, v
    first = max(0, L - window + 1) if window else 0
    pos = torch.arange(first, L + 1)
    slots = table[pos // PAGE].long() * PAGE + pos % PAGE
    keys, values = flat_k[slots].float(), flat_v[slots].float()  # [n, 2, 64]
    z = torch.empty(HEADS, HEAD)
    for h in range(HEADS):
        a = _split_attention(q[h], keys[:, h // GROUP], values[:, h // GROUP], splits)
        a = a.to(torch.bfloat16).float()
        z[h] = a * torch.rsqrt(a.pow(2).mean() + 1e-5) * torch.sigmoid(gate[h])

    got = (
        payloads(scratch, plan, "z", HEADS * HEAD // 2, tag).view(torch.bfloat16).cpu()
    )
    # FP32 sums in another order: within about one BF16 rounding step.
    torch.testing.assert_close(got.float().view(HEADS, HEAD), z, rtol=2**-7, atol=1e-3)
    # The cache gets the rotated k and the raw v of the token.
    got_k = kc.view(-1, KV_HEADS, HEAD)[slot].cpu()
    got_v = vc.view(-1, KV_HEADS, HEAD)[slot].cpu()
    torch.testing.assert_close(got_k.float(), k.float(), rtol=2**-7, atol=1e-2)
    assert torch.equal(got_v, v)
