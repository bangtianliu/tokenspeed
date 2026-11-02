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

"""Muse Spark attention stages (head, attn, merge) of one rank, for one
decode row or the chained rows of a speculative verify."""

from __future__ import annotations

import muse_spark_reference as ref
import pytest
import torch
from monokernel_testing import fill, payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    attention,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    qkvg as qkvg_stage,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.layer import (  # noqa: E402
    FULL_SPLITS,
    SLIDING_SPLITS,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

HEAD, HEADS, KV_HEADS, GROUP, PAGE = 64, 16, 2, 8, 64
QKVG = 2304
K_ROW, V_ROW, GATE_ROW = 1024, 1152, 1280


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
            mb,
            task,
            seq_len,
            table,
            k_cache,
            v_cache,
            cos_sin,
            ROPE,
            SPLITS,
            WINDOW,
            "attn",
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        attention.merge_stage(mb, task, SPLITS, "merge")


def _plan(splits: int, rows: int) -> Plan:
    return Plan(
        grid=256,
        stages=attention.stages({"head": 0, "attn": 32, "merge": 18}, splits, rows),
        regions=qkvg_stage.REGIONS + attention.regions(splits),
        rows=rows,
        chain=rows,
        external=("qkvg", "gate"),
    )


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize(
    "length,window,splits",
    [
        (6, 2048, SLIDING_SPLITS),
        (3000, 2048, SLIDING_SPLITS),
        (3000, 0, FULL_SPLITS),
        (9000, 0, FULL_SPLITS),
        # Several key chunks a split.
        (50000, 0, FULL_SPLITS),
    ],
)
def test_attention_matches_the_regular_path_math(length, window, splits, rows):
    rope = window != 0
    g = torch.Generator().manual_seed(length + window)
    L0 = length - 1  # row 0's position; row t is the token at L0 + t
    end = length + rows - 1
    used = (end + PAGE - 1) // PAGE
    pages = used + 3
    table = torch.randperm(pages, generator=g)[:used].to(torch.int32) + 1
    k_cache = torch.randn(pages + 2, PAGE, KV_HEADS, HEAD, generator=g).to(
        torch.bfloat16
    )
    v_cache = torch.randn(pages + 2, PAGE, KV_HEADS, HEAD, generator=g).to(
        torch.bfloat16
    )
    qkvg = torch.randn(rows, QKVG, generator=g).to(torch.bfloat16)
    cs = ref.rope_table(end + 1)
    positions = torch.arange(L0, L0 + rows)
    slots = table[positions // PAGE].long() * PAGE + positions % PAGE

    plan = _plan(splits, rows)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    step = 5
    tag = launch_tag(step, 0)
    # The projection's two K halves: each half of the value (exact), so
    # their FP32 sum is the BF16 value.
    for row in range(rows):
        for region, part in (
            ("qkvg", qkvg[row, :GATE_ROW]),
            ("gate", qkvg[row, GATE_ROW:]),
        ):
            half = (part.float() / 2).view(torch.int32)
            fill(scratch, plan, region, torch.cat([half, half]).cuda(), tag, row=row)
    kc, vc = k_cache.clone().cuda(), v_cache.clone().cuda()
    attention_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), step, dtype=torch.int32, device="cuda"),
        (positions + 1).to(torch.int32).cuda(),
        slots.to(torch.int32).cuda(),
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

    # Reference: the regular path's math with its BF16 rounding points. The
    # cache slots of the launch's tokens hold stale values: the kernel must
    # take those keys from the projection.
    z, k, v, _ = ref.attention_z(
        qkvg,
        k_cache.view(-1, KV_HEADS, HEAD),
        v_cache.view(-1, KV_HEADS, HEAD),
        table,
        length,
        window,
        splits,
        cs,
    )
    for t in range(rows):
        got = payloads(scratch, plan, "z", HEADS * HEAD // 2, tag, row=t)
        # FP32 sums in another order: within about one BF16 rounding step.
        torch.testing.assert_close(
            got.view(torch.bfloat16).cpu().float(),
            z[t].float(),
            rtol=2**-7,
            atol=1e-3,
        )
        # The cache gets the rotated k and the raw v of each row's token.
        got_k = kc.view(-1, KV_HEADS, HEAD)[slots[t]].cpu()
        got_v = vc.view(-1, KV_HEADS, HEAD)[slots[t]].cpu()
        torch.testing.assert_close(got_k.float(), k[t].float(), rtol=2**-7, atol=1e-2)
        assert torch.equal(got_v, v[t])
