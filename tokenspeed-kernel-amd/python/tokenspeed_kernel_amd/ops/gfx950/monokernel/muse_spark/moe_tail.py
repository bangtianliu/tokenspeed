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
* unpool, 64 tasks of 16 owned hidden columns: the post-expert projection
  (BF16 ``[1024, 4096]``, 128 KiB a task) of the whole combined row.
* fin1..fin3, 8 tasks of 128 owned hidden columns (same CTAs): the post-FFN
  norm (gain, then RMS over 8192 columns), the residual gate ``h'' = beta y +
  alpha h'`` (stored: the launch's residual output), and the next layer's
  input norm, rounded to BF16 and put into every rank's ``anext``.
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
FIN_TASKS = gl.constexpr(8)
FIN_COLS = gl.constexpr(128)
OUT_TASKS = gl.constexpr(16)
POST_EPS = gl.constexpr(1e-8)
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("ss3", "cmb", "ss4", "ss5", "anext")


def regions() -> tuple[tuple[str, int], ...]:
    world, k = int(WORLD), int(TOP_K)
    return (
        ("ss3", world * int(COMB_TASKS) * k),  # FP32 [rank][task][route]
        ("cmb", int(POOLED) // 2),  # BF16 pairs of the combined row
        ("yv", int(OWNED) // 2),  # BF16 pairs of the owned projection columns
        ("ss4", world * int(FIN_TASKS)),
        ("ss5", world * int(FIN_TASKS)),
        ("anext", int(HIDDEN) // 2),  # BF16 pairs of the next attention input
    )


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    comb, unpool, fin, out = placement(bases, ("comb", "unpool", "fin", "out"))
    c, f = int(COMB_TASKS), int(FIN_TASKS)
    return (
        Stage("comb1", c, comb, reads=("rr", "route"), writes=("ss3",)),
        Stage("comb2", c, comb, reads=("ss3",), writes=("cmb",)),
        Stage("unpool", int(UNPOOL_TASKS), unpool, reads=("cmb",), writes=("yv",)),
        Stage("fin1", f, fin, reads=("yv",), writes=("ss4",)),
        Stage("fin2", f, fin, reads=("ss4", "hp"), writes=("ss5",)),
        Stage("fin3", f, fin, reads=("ss5",), writes=("anext",)),
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
def unpool_stage(mb, task, w, STAGE: gl.constexpr):
    """Unpool task ``task``: owned hidden columns ``16 task ..``.

    Args:
        w: This rank's post-expert projection rows, BF16 ``[1024, 4096]``
            (``weights.bf16_tiles``).
    """
    gl.static_assert(schedule.rows(mb.plan) == 1, "the MoE tail serves one row")
    mb.stamp(STAGE, trace.START)
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
    mb.stamp(STAGE, trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=1)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16_rows(mb, "yv", task * 8, out)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _total(mb, region: gl.constexpr):
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    (got,) = mb.poll_many((region,), (gl.arange(0, WORLD * FIN_TASKS, lanes),))
    return gl.sum(got.to(gl.float32, bitcast=True), axis=0)


@gluon.jit
def fin_stages(mb, task, post_gain, alpha, beta, in_gain, h_next, RANK: gl.constexpr):
    """Fin1..fin3 task ``task``: owned hidden columns ``128 task ..``.

    Args:
        post_gain: FP32 ``bf16(w + 1)`` of the post-FFN norm, owned columns.
        alpha, beta: FP32 residual-gate coefficients of the FFN, owned columns.
        in_gain: FP32 ``bf16(w + 1)`` of the next layer's input norm.
        h_next: Where the owned FP32 residual ``h''`` goes.
    """
    mb.stamp("fin1", trace.START)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    j = gl.arange(0, FIN_COLS // 2, lanes)
    c = task * FIN_COLS + 2 * j
    (got,) = mb.poll_many(("yv",), (c // 2,))
    mb.stamp("fin1", trace.READY)
    y_lo = (got << 16).to(gl.float32, bitcast=True) * gl.load(post_gain + c)
    y_hi = (got & -65536).to(gl.float32, bitcast=True) * gl.load(post_gain + c + 1)
    one = gl.arange(0, 1, lanes)
    _put_all(
        mb,
        "ss4",
        RANK * FIN_TASKS + task + one,
        gl.zeros([1], gl.float32, lanes) + gl.sum(y_lo * y_lo + y_hi * y_hi, axis=0),
    )
    mb.stamp("fin1", trace.END)

    mb.stamp("fin2", trace.START)
    inv = gl.rsqrt(_total(mb, "ss4") / HIDDEN + POST_EPS)
    hp_lo, hp_hi = mb.poll_many(("hp", "hp"), (c, c + 1))
    mb.stamp("fin2", trace.READY)
    h_lo = gl.fma(
        gl.load(beta + c),
        y_lo * inv,
        gl.load(alpha + c) * hp_lo.to(gl.float32, bitcast=True),
    )
    h_hi = gl.fma(
        gl.load(beta + c + 1),
        y_hi * inv,
        gl.load(alpha + c + 1) * hp_hi.to(gl.float32, bitcast=True),
    )
    gl.store(h_next + c, h_lo)
    gl.store(h_next + c + 1, h_hi)
    _put_all(
        mb,
        "ss5",
        RANK * FIN_TASKS + task + one,
        gl.zeros([1], gl.float32, lanes) + gl.sum(h_lo * h_lo + h_hi * h_hi, axis=0),
    )
    mb.stamp("fin2", trace.END)

    mb.stamp("fin3", trace.START)
    inv2 = gl.rsqrt(_total(mb, "ss5") / HIDDEN + PRE_EPS)
    mb.stamp("fin3", trace.READY)
    word = _bf16_word(
        h_lo * inv2 * gl.load(in_gain + c), h_hi * inv2 * gl.load(in_gain + c + 1)
    )
    for p in gl.static_range(WORLD):
        mb.put_peer("anext", p, (RANK * OWNED + c) // 2, word)
    mb.stamp("fin3", trace.END)


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
