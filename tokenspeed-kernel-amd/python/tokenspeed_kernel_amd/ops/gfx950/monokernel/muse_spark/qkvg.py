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

144 tasks, each one 16-row block of the rank's [2304, 8192] BF16 weight
(256 KiB a task, K split over 8 waves). A task issues its weight loads
first, then reads the attention input rows that the previous launch wrote,
and publishes its 16 BF16 results per row in region qkvg (BF16 pairs).
The regular path computes the same FP32 sums and rounds them to BF16.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import HIDDEN, QKVG
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

TASKS = gl.constexpr(QKVG // 16)

REGIONS = (("qkvg", int(QKVG) // 2),)  # BF16 pairs of q | k | v | gate


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    """QKVG, placed by ``bases``."""
    (qkvg,) = placement(bases, ("qkvg",))
    return (Stage("qkvg", int(TASKS), qkvg, writes=("qkvg",)),)


@gluon.jit
def qkvg_stage(mb, task, w, a, a_stride, STAGE: gl.constexpr):
    """QKVG task ``task``: rows ``16 task .. 16 task + 15`` of every token row.

    Args:
        w: The rank's fused weight, BF16 ``[2304, 8192]`` (``weights.bf16_tiles``).
        a: The attention input, BF16 ``[rows, 8192]`` with row stride
            ``a_stride`` (the previous launch's output).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    mb.stamp(STAGE, trace.START)
    tiles = bf16.load_weight_tiles(
        w,
        nb0=task,
        K=HIDDEN,
        kit0=0,
        KITS=HIDDEN // 32,
        SPLIT=8,
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )
    x_smem = bf16.load_global_rows(a, a_stride, 0, HIDDEN, R)
    mb.stamp(STAGE, trace.READY)
    outs = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16_rows(mb, "qkvg", task * 8, outs)
    mb.stamp(STAGE, trace.END)
