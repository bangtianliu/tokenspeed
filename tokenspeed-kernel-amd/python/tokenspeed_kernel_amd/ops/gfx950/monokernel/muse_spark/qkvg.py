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

"""QKVG: the fused q | k | v | output-gate projection of one rank.

The rank's BF16 weight is ``[2304, 8192]``: 144 blocks of 16 rows. A task
takes one block and one half of K (128 KiB): one CU cannot stream 256 KiB
as fast as the memory can, so the q | k | v blocks spread over 160 CTAs.

* qkvg, 160 tasks: the q | k | v blocks (rows 0 .. 1279).
* gate, 128 tasks: the output-gate blocks (rows 1280 .. 2303), on CTAs
  that run a qkvg task first. Attention does not wait for the gate, so its
  weights load after the q | k | v weights.

A task issues its weight loads first, then reads its K half of the
attention input rows that the previous launch wrote, and publishes its 16
FP32 partial sums per row. A consumer adds the two halves and rounds the
sum to BF16 (:func:`rows`), as the regular path rounds its FP32 sums.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    GATE_ROW,
    HIDDEN,
    QKVG,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

KSPLIT = gl.constexpr(2)
QKV_BLOCKS = gl.constexpr(GATE_ROW // 16)  # 80
GATE_BLOCKS = gl.constexpr((QKVG - GATE_ROW) // 16)  # 64

# FP32 partial sums [K half][row]: of q | k | v, and of the gate.
REGIONS = (
    ("qkvg", int(KSPLIT) * int(GATE_ROW)),
    ("gate", int(KSPLIT) * (int(QKVG) - int(GATE_ROW))),
)


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    """QKVG and gate, placed by ``bases``."""
    qkv, gate = placement(bases, ("qkvg", "gate"))
    return (
        Stage("qkvg", int(KSPLIT * QKV_BLOCKS), qkv, writes=("qkvg",)),
        Stage("gate", int(KSPLIT * GATE_BLOCKS), gate, writes=("gate",)),
    )


@gluon.jit
def _block_stage(
    mb, task, w, a, a_stride, FIRST: gl.constexpr, N: gl.constexpr, region: gl.constexpr
):
    """Task ``task`` of a stage of ``N`` rows from block ``FIRST``: block
    ``FIRST + task // 2``, K half ``task % 2``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = task // KSPLIT
    half = task % KSPLIT
    KITS: gl.constexpr = HIDDEN // 32 // KSPLIT
    tiles = bf16.load_weight_tiles(
        w,
        nb0=FIRST + blk,
        K=HIDDEN,
        kit0=half * KITS,
        KITS=KITS,
        SPLIT=8,
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )
    x_smem = bf16.load_global_rows(
        a, a_stride, half * (HIDDEN // KSPLIT), HIDDEN // KSPLIT, R
    )
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # FP32 [RP, 16]
    return out, half * N + blk * 16


@gluon.jit
def _put_rows(mb, region: gl.constexpr, out, at):
    R: gl.constexpr = schedule.rows(mb.plan)
    rows = gl.arange(0, out.shape[0], gl.SliceLayout(1, out.type.layout))[:, None]
    for r in gl.static_range(R):
        v = gl.sum(gl.where(rows == r, out, 0.0), axis=0)
        i = at + gl.arange(0, 16, v.type.layout)
        mb.row(r).put(region, i, v.to(gl.int32, bitcast=True))


@gluon.jit
def qkvg_stage(mb, task, w, a, a_stride, STAGE: gl.constexpr):
    """QKVG task ``task``: half ``task % 2`` of K of q | k | v rows
    ``16 (task // 2) ..`` of every token row.

    Args:
        w: The rank's fused weight, BF16 ``[2304, 8192]`` (``weights.bf16_tiles``).
        a: The attention input, BF16 ``[rows, 8192]`` with row stride
            ``a_stride`` (the previous launch's output).
    """
    mb.stamp(STAGE, trace.START)
    out, at = _block_stage(mb, task, w, a, a_stride, 0, GATE_ROW, "qkvg")
    mb.stamp(STAGE, trace.COMPUTED)
    _put_rows(mb, "qkvg", out, at)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def gate_stage(mb, task, w, a, a_stride, STAGE: gl.constexpr):
    """Gate task ``task``: half ``task % 2`` of K of gate rows ``16 (task //
    2) ..`` (rows ``1280 + 16 (task // 2) ..`` of ``w``), as :func:`qkvg_stage`."""
    mb.stamp(STAGE, trace.START)
    out, at = _block_stage(
        mb, task, w, a, a_stride, QKV_BLOCKS, QKVG - GATE_ROW, "gate"
    )
    mb.stamp(STAGE, trace.COMPUTED)
    _put_rows(mb, "gate", out, at)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def rows(got, AXIS: gl.constexpr):
    """The BF16-rounded projection values (FP32) from the polled partial
    sums ``got``, whose ``AXIS`` holds (half 0, half 1)."""
    halves = got.to(gl.float32, bitcast=True)
    return gl.sum(halves, axis=AXIS).to(gl.bfloat16).to(gl.float32)
