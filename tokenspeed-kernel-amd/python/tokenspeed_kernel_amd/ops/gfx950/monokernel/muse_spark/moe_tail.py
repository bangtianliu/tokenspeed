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

"""Combine, post-expert projection and layer boundary of one Muse Spark rank.

Rank ``r`` owns pooled columns ``512 r ..`` of the expert outputs and hidden
columns ``1024 r ..`` of the residual (one row).

* comb1, comb2, 8 tasks of 64 owned pooled columns (same CTAs): the rank-order
  sum of the 8 ranks' BF16 route partials, rounded to BF16, times the
  post-expert gain; then each route's RMS norm over all 4096 columns (one
  exchange of partial sums of squares per route), and the weighted sum of
  the 8 routes (a tree within a wave, not the regular path's pick order),
  rounded to BF16, put into every rank's ``cmb``.
* unpool, fin, 64 tasks of 16 owned hidden columns (same CTAs): the
  post-expert projection (BF16 ``[1024, 4096]``, 128 KiB a task) of the whole
  combined row, rounded to BF16, times the post-FFN gain. Each task puts
  four partial sums into every rank's ``ss4``: of ``y^2``, ``(beta y)^2``,
  ``(alpha h')(beta y)`` and ``(alpha h')^2``. Then, after one exchange: the
  post-FFN norm (RMS over 8192 columns), the residual gate ``h'' = beta y +
  alpha h'`` (stored: the launch's residual output), and the next layer's
  input norm, whose sum of squares is the expansion of ``(beta y inv + alpha
  h')^2`` in the four sums. The result is rounded to BF16 and put into every
  rank's ``anext``.
* out, 16 tasks: the whole next attention input to memory, for the next
  launch.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.attn_tail import OWNED
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import HIDDEN, WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.moe_experts import (
    OWNED_OUT,
    TOP_K,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.moe_pool import POOLED
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

COMB_TASKS = gl.constexpr(8)
COMB_COLS = gl.constexpr(64)
UNPOOL_TASKS = gl.constexpr(64)
UNPOOL_COLS = gl.constexpr(16)
SUMS = gl.constexpr(4)
OUT_TASKS = gl.constexpr(16)
POST_EPS = gl.constexpr(1e-8)
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("ss3", "cmb", "ss4", "anext")


def regions() -> tuple[tuple[str, int], ...]:
    world, k = int(WORLD), int(TOP_K)
    return (
        ("ss3", world * int(COMB_TASKS) * k),  # FP32 [rank][task][route]
        ("cmb", int(POOLED) // 2),  # BF16 pairs of the combined row
        ("ss4", world * int(UNPOOL_TASKS) * int(SUMS)),  # FP32 [rank][task][sum]
        ("anext", int(HIDDEN) // 2),  # BF16 pairs of the next attention input
    )


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    comb, unpool, out = placement(bases, ("comb", "unpool", "out"))
    c, u = int(COMB_TASKS), int(UNPOOL_TASKS)
    return (
        Stage("comb1", c, comb, reads=("rr", "route"), writes=("ss3",)),
        Stage("comb2", c, comb, reads=("ss3",), writes=("cmb",)),
        Stage("unpool", u, unpool, reads=("cmb", "hp"), writes=("ss4",)),
        Stage("fin", u, unpool, reads=("ss4",), writes=("anext",)),
        Stage("out", int(OUT_TASKS), out, reads=("anext",)),
    )


@gluon.jit
def _put_all(mb, region: gl.constexpr, index, values):
    """Put ``values`` (FP32) at ``index`` of every rank's ``region``."""
    word = values.to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer(region, p, index, word)


@gluon.jit
def _bf16_word(lo, hi):
    return (lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    )


@gluon.jit
def comb_stages(mb, task, pen_gain, RANK: gl.constexpr):
    """Comb1 and comb2 task ``task``: owned pooled columns ``64 task ..``.

    Args:
        pen_gain: FP32 ``bf16(w + 1)`` of the post-expert norm, owned columns.
    """
    mb.stamp("comb1", trace.START)
    nw: gl.constexpr = gl.num_warps()
    gl.static_assert(nw == 8, "comb lays out 8 waves")
    # [source, route, pair]: the 8 source ranks in a thread's registers, so
    # the rank-order sum stays in a thread; the routes over 2 lanes and 4
    # waves; the pairs over 32 lanes (2 waves repeat them).
    L3: gl.constexpr = gl.BlockedLayout([WORLD, 1, 1], [1, 2, 32], [1, 4, 2], [2, 1, 0])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
    k = gl.arange(0, TOP_K, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    j = gl.arange(0, COMB_COLS // 2, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[
        None, None, :
    ]
    (got,) = mb.poll_many(
        ("rr",), (((src * TOP_K + k) * OWNED_OUT + task * COMB_COLS) // 2 + j,)
    )
    mb.stamp("comb1", trace.READY)
    lo_all = (got << 16).to(gl.float32, bitcast=True)
    hi_all = (got & -65536).to(gl.float32, bitcast=True)
    L2: gl.constexpr = gl.SliceLayout(0, L3)
    # The rank order of the regular path's reduction: 0, then 1, ...
    lo = gl.zeros([TOP_K, COMB_COLS // 2], gl.float32, L2)
    hi = gl.zeros([TOP_K, COMB_COLS // 2], gl.float32, L2)
    for p in gl.static_range(WORLD):
        lo += gl.sum(gl.where(src == p, lo_all, 0.0), axis=0)
        hi += gl.sum(gl.where(src == p, hi_all, 0.0), axis=0)
    jj = gl.arange(0, COMB_COLS // 2, gl.SliceLayout(0, L2))
    c = task * COMB_COLS + 2 * jj
    lo = lo.to(gl.bfloat16).to(gl.float32) * gl.load(pen_gain + c)[None, :]
    hi = hi.to(gl.bfloat16).to(gl.float32) * gl.load(pen_gain + c + 1)[None, :]
    ssq = gl.sum(lo * lo + hi * hi, axis=1)  # [8 routes]
    kk = gl.arange(0, TOP_K, gl.SliceLayout(1, L2))
    _put_all(mb, "ss3", (RANK * COMB_TASKS + task) * TOP_K + kk, ssq)
    mb.stamp("comb1", trace.END)

    mb.stamp("comb2", trace.START)
    tl_: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [nw, 1], [1, 0])
    rt = gl.arange(0, WORLD * COMB_TASKS, gl.SliceLayout(1, tl_))[:, None]
    kt = gl.arange(0, TOP_K, gl.SliceLayout(0, tl_))[None, :]
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    got_s, got_w = mb.poll_many(
        ("ss3", "route"), (rt * TOP_K + kt, TOP_K + gl.arange(0, TOP_K, lanes))
    )
    mb.stamp("comb2", trace.READY)
    total = gl.sum(got_s.to(gl.float32, bitcast=True), axis=0)  # [8 routes]
    inv = gl.rsqrt(total / POOLED + POST_EPS)
    scale = gl.convert_layout(inv, gl.SliceLayout(1, L2)) * gl.convert_layout(
        got_w.to(gl.float32, bitcast=True), gl.SliceLayout(1, L2)
    )
    out_lo = gl.sum(lo * scale[:, None], axis=0)
    out_hi = gl.sum(hi * scale[:, None], axis=0)
    word = _bf16_word(out_lo, out_hi)
    for p in gl.static_range(WORLD):
        mb.put_peer("cmb", p, (RANK * OWNED_OUT + task * COMB_COLS) // 2 + jj, word)
    mb.stamp("comb2", trace.END)


@gluon.jit
def unpool_stages(
    mb, task, w, post_gain, alpha, beta, in_gain, h_next, RANK: gl.constexpr
):
    """Unpool and fin task ``task``: owned hidden columns ``16 task ..``.

    Args:
        w: This rank's post-expert projection rows, BF16 ``[1024, 4096]``
            (``weights.bf16_tiles``).
        post_gain: FP32 ``bf16(w + 1)`` of the post-FFN norm, owned columns.
        alpha, beta: FP32 residual-gate coefficients of the FFN, owned columns.
        in_gain: FP32 ``bf16(w + 1)`` of the next layer's input norm.
        h_next: Where the owned FP32 residual ``h''`` goes.
    """
    gl.static_assert(schedule.rows(mb.plan) == 1, "the MoE tail serves one row")
    nw: gl.constexpr = gl.num_warps()
    mb.stamp("unpool", trace.START)
    tiles = bf16.load_weight_tiles(
        w,
        nb0=task,
        K=POOLED,
        kit0=0,
        KITS=POOLED // 32,
        SPLIT=8,
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )
    x_smem, _ = bf16.load_rows(mb, "cmb", 0, POOLED, 1)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    j = gl.arange(0, UNPOOL_COLS // 2, lanes)
    c = task * UNPOOL_COLS + 2 * j
    hp_lo, hp_hi = mb.poll_many(("hp", "hp"), (c, c + 1))
    mb.stamp("unpool", trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=1)  # [1, 16]
    mb.stamp("unpool", trace.COMPUTED)
    # split needs a pair's two values in one thread.
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [nw, 1], [1, 0])
    lo, hi = gl.split(
        gl.reshape(gl.convert_layout(out, pairs), [1, UNPOOL_COLS // 2, 2])
    )
    y_lo = gl.convert_layout(gl.sum(lo, axis=0), lanes)
    y_hi = gl.convert_layout(gl.sum(hi, axis=0), lanes)
    y_lo = y_lo.to(gl.bfloat16).to(gl.float32) * gl.load(post_gain + c)
    y_hi = y_hi.to(gl.bfloat16).to(gl.float32) * gl.load(post_gain + c + 1)
    b_lo = gl.load(beta + c)
    b_hi = gl.load(beta + c + 1)
    ah_lo = gl.load(alpha + c) * hp_lo.to(gl.float32, bitcast=True)
    ah_hi = gl.load(alpha + c + 1) * hp_hi.to(gl.float32, bitcast=True)
    by_lo = b_lo * y_lo
    by_hi = b_hi * y_hi
    q = gl.arange(0, SUMS, lanes)
    sums = gl.where(
        q == 0,
        gl.sum(y_lo * y_lo + y_hi * y_hi, axis=0),
        gl.where(
            q == 1,
            gl.sum(by_lo * by_lo + by_hi * by_hi, axis=0),
            gl.where(
                q == 2,
                gl.sum(ah_lo * by_lo + ah_hi * by_hi, axis=0),
                gl.sum(ah_lo * ah_lo + ah_hi * ah_hi, axis=0),
            ),
        ),
    )
    _put_all(mb, "ss4", (RANK * UNPOOL_TASKS + task) * SUMS + q, sums)
    mb.stamp("unpool", trace.END)

    mb.stamp("fin", trace.START)
    sl: gl.constexpr = gl.BlockedLayout([1, SUMS], [64, 1], [nw, 1], [1, 0])
    rt = gl.arange(0, WORLD * UNPOOL_TASKS, gl.SliceLayout(1, sl))[:, None]
    qt = gl.arange(0, SUMS, gl.SliceLayout(0, sl))
    (got,) = mb.poll_many(("ss4",), (rt * SUMS + qt[None, :],))
    mb.stamp("fin", trace.READY)
    total = gl.sum(got.to(gl.float32, bitcast=True), axis=0)  # [4 sums]
    ss_y = gl.sum(gl.where(qt == 0, total, 0.0), axis=0)
    ss_by = gl.sum(gl.where(qt == 1, total, 0.0), axis=0)
    ss_x = gl.sum(gl.where(qt == 2, total, 0.0), axis=0)
    ss_ah = gl.sum(gl.where(qt == 3, total, 0.0), axis=0)
    inv = gl.rsqrt(ss_y / HIDDEN + POST_EPS)
    # The sum of h''^2 over all columns: inv^2 ss_by + 2 inv ss_x + ss_ah.
    ss_h = gl.maximum(inv * inv * ss_by + 2.0 * inv * ss_x + ss_ah, 0.0)
    inv2 = gl.rsqrt(ss_h / HIDDEN + PRE_EPS)
    h_lo = gl.fma(b_lo, y_lo * inv, ah_lo)
    h_hi = gl.fma(b_hi, y_hi * inv, ah_hi)
    gl.store(h_next + c, h_lo)
    gl.store(h_next + c + 1, h_hi)
    word = _bf16_word(
        h_lo * inv2 * gl.load(in_gain + c), h_hi * inv2 * gl.load(in_gain + c + 1)
    )
    for p in gl.static_range(WORLD):
        mb.put_peer("anext", p, (RANK * OWNED + c) // 2, word)
    mb.stamp("fin", trace.END)


@gluon.jit
def out_stage(mb, task, a_out, STAGE: gl.constexpr):
    """Out task ``task``: next attention input values ``512 task ..`` to
    ``a_out`` (BF16 ``[8192]``)."""
    mb.stamp(STAGE, trace.START)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = task * 256 + gl.arange(0, 256, lanes)
    (got,) = mb.poll_many(("anext",), (i,))
    mb.stamp(STAGE, trace.READY)
    gl.store(a_out.to(gl.pointer_type(gl.int32), bitcast=True) + i, got)
    mb.stamp(STAGE, trace.END)
