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

"""Pre-expert projection (the pooler) of one Muse Spark rank (one row).

pool, 32 tasks: one 16-row block of the rank's 512 pooler rows (BF16
``[512, 8192]``, 256 KiB a task) times the whole BF16 MoE input ``xin``,
rounded to BF16 like the regular path. A task puts its 16 values into every
rank's ``cg``, so every rank gets the whole pooled row (4096, rank-major).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import HIDDEN, WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

POOLED = gl.constexpr(4096)
POOL_ROWS = gl.constexpr(512)  # pooled columns a rank computes: POOLED / WORLD
POOL_TASKS = gl.constexpr(32)

PEERS = ("cg",)


def regions() -> tuple[tuple[str, int], ...]:
    return (("cg", int(POOLED) // 2),)  # BF16 pairs of the pooled row


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    (pool,) = placement(bases, ("pool",))
    return (Stage("pool", int(POOL_TASKS), pool, reads=("xin",), writes=("cg",)),)


@gluon.jit
def pool_stage(mb, task, w, RANK: gl.constexpr, STAGE: gl.constexpr):
    """Pool task ``task``: pooled columns ``512 RANK + 16 task ..``.

    Args:
        w: This rank's pooler rows, BF16 ``[512, 8192]`` (``weights.bf16_tiles``).
    """
    gl.static_assert(schedule.rows(mb.plan) == 1, "the pooler serves one row")
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
    x_smem, _ = bf16.load_rows(mb, "xin", 0, HIDDEN, 1)
    mb.stamp(STAGE, trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=1)  # FP32 [1, 16]
    mb.stamp(STAGE, trace.COMPUTED)
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    values = gl.convert_layout(out.to(gl.bfloat16), pairs)
    word = gl.reshape(mx.pack_bf16_pairs(gl.reshape(values, [1, 8, 2])), [8])
    p = gl.arange(0, 8, word.type.layout)
    for r in gl.static_range(WORLD):
        mb.put_peer("cg", r, RANK * (POOL_ROWS // 2) + task * 8 + p, word)
    mb.stamp(STAGE, trace.END)
