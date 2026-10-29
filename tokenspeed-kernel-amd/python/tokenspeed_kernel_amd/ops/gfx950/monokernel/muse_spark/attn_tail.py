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

"""Output projection and attention tail of one Muse Spark rank (one row).

Rank ``r`` owns hidden columns ``1024 r .. 1024 r + 1023``: it keeps only
that slice of the FP32 residual, and it reduces only those columns of the
output projection (a reduce-scatter).

* oproj, 128 tasks of 64 output columns: the rank's ``o_proj`` partial
  (BF16 ``[8192, 1024]``, 128 KiB a task) of the gated head-norm output
  ``z``, rounded to BF16 like the regular path's partial. A task puts its
  columns into the owner rank's ``red`` slot of this rank.
* tail1, tail2, tail3, 8 tasks of 128 owned columns on the same CTAs (each
  task keeps its values in registers from one stage to the next):

  - tail1 sums the 8 ranks' partials in rank order, rounds to BF16 (``b``),
    and puts ``sum b^2`` of its columns into every rank's ``ss1``.
  - tail2: ``b rsqrt(mean(b^2) + 1e-8)`` (the weightless post-attention
    norm over all 8192 columns), then the residual gate ``h' = beta b +
    alpha h``, stored as the rank's next residual slice; puts ``sum h'^2``.
  - tail3: ``u = h' rsqrt(mean(h'^2) + 1e-5) bf16(w + 1)`` (the pre-FFN
    norm), published as FP32 in ``u`` for the MoE.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import HIDDEN, WORLD, Q
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

OWNED = gl.constexpr(1024)  # hidden columns a rank owns: HIDDEN / WORLD
OPROJ_TASKS = gl.constexpr(128)
OPROJ_COLS = gl.constexpr(64)
TAIL_TASKS = gl.constexpr(8)
TAIL_COLS = gl.constexpr(128)
POST_EPS = gl.constexpr(1e-8)
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("red", "ss1", "ss2")


def regions() -> tuple[tuple[str, int], ...]:
    world, owned, tasks = int(WORLD), int(OWNED), int(TAIL_TASKS)
    return (
        ("red", world * owned // 2),  # BF16 pairs [source rank][owned column]
        ("ss1", world * tasks),  # FP32 partial sums of squares [rank][task]
        ("ss2", world * tasks),
        ("u", owned),  # FP32 owned slice of the pre-FFN norm output
    )


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    oproj, tail = placement(bases, ("oproj", "tail"))
    tasks = int(TAIL_TASKS)
    return (
        Stage("oproj", int(OPROJ_TASKS), oproj, reads=("z",), writes=("red",)),
        Stage("tail1", tasks, tail, reads=("red",), writes=("ss1",)),
        Stage("tail2", tasks, tail, reads=("ss1",), writes=("ss2",)),
        Stage("tail3", tasks, tail, reads=("ss2",), writes=("u",)),
    )


@gluon.jit
def oproj_stage(mb, task, w, RANK: gl.constexpr, STAGE: gl.constexpr):
    """Oproj task ``task``: output columns ``64 task ..``.

    Args:
        w: The rank's ``o_proj``, BF16 ``[8192, 1024]`` (``weights.bf16_tiles``).
    """
    gl.static_assert(schedule.rows(mb.plan) == 1, "the attention tail serves one row")
    mb.stamp(STAGE, trace.START)
    NW: gl.constexpr = OPROJ_COLS // 16
    tiles = bf16.load_weight_tiles(
        w,
        nb0=task * NW,
        K=Q,
        kit0=0,
        KITS=Q // 32,
        SPLIT=gl.num_warps() // NW,
        NW=NW,
        GROUP=NW,
        GROUP_STRIDE=0,
    )
    x_smem, _ = bf16.load_rows(mb, "z", 0, Q, 1)
    mb.stamp(STAGE, trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=1)  # FP32 [1, 64]
    mb.stamp(STAGE, trace.COMPUTED)
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    values = gl.convert_layout(out.to(gl.bfloat16), pairs)
    word = gl.reshape(
        mx.pack_bf16_pairs(gl.reshape(values, [1, OPROJ_COLS // 2, 2])),
        [OPROJ_COLS // 2],
    )
    p = gl.arange(0, OPROJ_COLS // 2, word.type.layout)
    cols_per_rank: gl.constexpr = OWNED // OPROJ_COLS
    owner = task // cols_per_rank
    at = RANK * (OWNED // 2) + (task % cols_per_rank) * (OPROJ_COLS // 2) + p
    mb.put_peer("red", owner, at, word)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _put_all(mb, region: gl.constexpr, index, value):
    """Put the scalar ``value`` at ``index`` of every rank's ``region``."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    one = gl.arange(0, 1, lanes)
    word = (gl.zeros([1], gl.float32, lanes) + value).to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer(region, p, index + one, word)


@gluon.jit
def _total(mb, region: gl.constexpr):
    """Sum of every rank's partials in ``region``, rank-major order."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = gl.arange(0, WORLD * TAIL_TASKS, lanes)
    (got,) = mb.poll_many((region,), (i,))
    return gl.sum(got.to(gl.float32, bitcast=True), axis=0)


@gluon.jit
def tail_stages(mb, task, h, h_out, alpha, beta, pre_gain, RANK: gl.constexpr):
    """Tail tasks ``task`` of tail1, tail2 and tail3: owned columns
    ``128 task ..`` (of the rank's 1024).

    Args:
        h: The rank's FP32 residual slice ``[1024]``.
        h_out: Where ``h'`` goes (not ``h``: the waves of a task read ``h``
            while others store ``h'``).
        alpha, beta: FP32 residual-gate coefficients of the owned columns.
        pre_gain: FP32 ``bf16(w + 1)`` of the pre-FFN norm, owned columns.
    """
    mb.stamp("tail1", trace.START)
    nw: gl.constexpr = gl.num_warps()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, rl))
    j = gl.arange(0, TAIL_COLS // 2, gl.SliceLayout(0, rl))
    (got,) = mb.poll_many(
        ("red",), (src[:, None] * (OWNED // 2) + task * (TAIL_COLS // 2) + j[None, :],)
    )
    mb.stamp("tail1", trace.READY)
    lo_all = (got << 16).to(gl.float32, bitcast=True)
    hi_all = (got & -65536).to(gl.float32, bitcast=True)
    # The rank order of the regular path's reduction: 0, then 1, ...
    lo = gl.zeros([TAIL_COLS // 2], gl.float32, gl.SliceLayout(0, rl))
    hi = gl.zeros([TAIL_COLS // 2], gl.float32, gl.SliceLayout(0, rl))
    for p in gl.static_range(WORLD):
        lo += gl.sum(gl.where(src[:, None] == p, lo_all, 0.0), axis=0)
        hi += gl.sum(gl.where(src[:, None] == p, hi_all, 0.0), axis=0)
    lo = lo.to(gl.bfloat16).to(gl.float32)
    hi = hi.to(gl.bfloat16).to(gl.float32)
    _put_all(mb, "ss1", RANK * TAIL_TASKS + task, gl.sum(lo * lo + hi * hi, axis=0))
    mb.stamp("tail1", trace.END)

    mb.stamp("tail2", trace.START)
    inv = gl.rsqrt(_total(mb, "ss1") / HIDDEN + POST_EPS)
    mb.stamp("tail2", trace.READY)
    c = task * TAIL_COLS + 2 * j
    hp_lo = gl.fma(gl.load(beta + c), lo * inv, gl.load(alpha + c) * gl.load(h + c))
    hp_hi = gl.fma(
        gl.load(beta + c + 1), hi * inv, gl.load(alpha + c + 1) * gl.load(h + c + 1)
    )
    gl.store(h_out + c, hp_lo)
    gl.store(h_out + c + 1, hp_hi)
    _put_all(
        mb,
        "ss2",
        RANK * TAIL_TASKS + task,
        gl.sum(hp_lo * hp_lo + hp_hi * hp_hi, axis=0),
    )
    mb.stamp("tail2", trace.END)

    mb.stamp("tail3", trace.START)
    inv2 = gl.rsqrt(_total(mb, "ss2") / HIDDEN + PRE_EPS)
    mb.stamp("tail3", trace.READY)
    u_lo = hp_lo * inv2 * gl.load(pre_gain + c)
    u_hi = hp_hi * inv2 * gl.load(pre_gain + c + 1)
    mb.put("u", c, u_lo.to(gl.int32, bitcast=True))
    mb.put("u", c + 1, u_hi.to(gl.int32, bitcast=True))
    mb.stamp("tail3", trace.END)
