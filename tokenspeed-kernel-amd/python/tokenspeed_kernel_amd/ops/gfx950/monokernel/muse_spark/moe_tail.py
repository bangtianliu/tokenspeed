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
columns ``1024 r ..`` of the residual.

* comb1, comb2, 8 tasks of 64 owned pooled columns per row (same CTAs): the
  rank-order
  sum of the 8 ranks' BF16 route partials, rounded to BF16, times the
  post-expert gain; then each route's RMS norm over all 4096 columns (one
  exchange of partial sums of squares per route), and the weighted sum of
  the 8 routes (a tree within a wave, not the regular path's pick order),
  rounded to BF16, put into every rank's ``cmb``.
* unpool, fin, 64 tasks of 16 owned hidden columns (same CTAs): the
  post-expert projection (BF16 ``[1024, 4096]``, 128 KiB a task) of every
  whole combined row, rounded to BF16, times the post-FFN gain. For each
  row, a task puts its
  ``y`` and ``alpha h'`` and four partial sums, of ``y^2``, ``(beta y)^2``,
  ``(alpha h')(beta y)`` and ``(alpha h')^2``, into every rank's ``ffn``, in
  one store per rank. Then, after that one exchange: the post-FFN norm (RMS
  over 8192 columns) and the residual gate ``h'' = beta y + alpha h'`` of the
  owned columns (stored: the launch's residual output).
* out, 16 tasks of 512 hidden columns per row, after the same exchange: the norms,
  the residual gate and the next layer's input norm, whose sum of squares is
  the expansion of ``(beta y inv + alpha h')^2`` in the four sums. The result
  is rounded to BF16 and stored as the next attention input, for the next
  launch. Every rank computes the whole row from the ``ffn`` of all ranks,
  with fin's arithmetic: putting the owned slice of the result to the other
  ranks instead would add a second exchange to the end of the layer.
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
SUMS_AT = gl.constexpr(2 * int(HIDDEN))  # where the sums start in ``ffn``

PEERS = ("ss3", "cmb", "ffn")


def regions() -> tuple[tuple[str, int], ...]:
    world, k = int(WORLD), int(TOP_K)
    return (
        ("ss3", world * int(COMB_TASKS) * k),  # FP32 [rank][task][route]
        ("cmb", int(POOLED) // 2),  # BF16 pairs of the combined row
        # FP32 [y, alpha h'][column], then the sums: [rank][task][sum].
        ("ffn", 2 * int(HIDDEN) + world * int(UNPOOL_TASKS) * int(SUMS)),
    )


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """Comb1, comb2, unpool, fin and out of ``rows`` token rows, placed by
    ``bases``."""
    comb, unpool, out = placement(bases, ("comb", "unpool", "out"))
    c, u = int(COMB_TASKS) * rows, int(UNPOOL_TASKS)
    return (
        Stage("comb1", c, comb, reads=("rr", "route"), writes=("ss3",)),
        Stage("comb2", c, comb, reads=("ss3",), writes=("cmb",)),
        Stage("unpool", u, unpool, reads=("cmb", "hp"), writes=("ffn",)),
        Stage("fin", u, unpool, reads=("ffn",)),
        Stage("out", int(OUT_TASKS) * rows, out, reads=("ffn",)),
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
    """Comb1 and comb2 task ``task``: owned pooled columns ``64 (task % 8)
    ..`` of row ``task // 8``.

    Args:
        pen_gain: FP32 ``bf16(w + 1)`` of the post-expert norm, owned columns.
    """
    mb.stamp("comb1", trace.START)
    row, task, _ = schedule.row_task(mb.plan, task, COMB_TASKS)
    mb = mb.row(row)
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
    L2: gl.constexpr = gl.SliceLayout(0, L3)
    # The gains load before the poll: a load after a poll would start only
    # when the poll returns.
    jj = gl.arange(0, COMB_COLS // 2, gl.SliceLayout(0, L2))
    c = task * COMB_COLS + 2 * jj
    gain_lo = gl.load(pen_gain + c)
    gain_hi = gl.load(pen_gain + c + 1)
    (got,) = mb.poll_many(
        ("rr",), (((src * TOP_K + k) * OWNED_OUT + task * COMB_COLS) // 2 + j,)
    )
    mb.stamp("comb1", trace.READY)
    lo_all = (got << 16).to(gl.float32, bitcast=True)
    hi_all = (got & -65536).to(gl.float32, bitcast=True)
    # The rank order of the regular path's reduction: 0, then 1, ...
    lo = gl.zeros([TOP_K, COMB_COLS // 2], gl.float32, L2)
    hi = gl.zeros([TOP_K, COMB_COLS // 2], gl.float32, L2)
    for p in gl.static_range(WORLD):
        lo += gl.sum(gl.where(src == p, lo_all, 0.0), axis=0)
        hi += gl.sum(gl.where(src == p, hi_all, 0.0), axis=0)
    lo = lo.to(gl.bfloat16).to(gl.float32) * gain_lo[None, :]
    hi = hi.to(gl.bfloat16).to(gl.float32) * gain_hi[None, :]
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
def _ffn_norms(mb):
    """The inverse RMS of the post-FFN norm and of the next layer's input
    norm, from every rank's sums in ``ffn``. Fin and out share this, so both get the
    same values."""
    nw: gl.constexpr = gl.num_warps()
    sl: gl.constexpr = gl.BlockedLayout([1, SUMS], [64, 1], [nw, 1], [1, 0])
    rt = gl.arange(0, WORLD * UNPOOL_TASKS, gl.SliceLayout(1, sl))[:, None]
    qt = gl.arange(0, SUMS, gl.SliceLayout(0, sl))
    (got,) = mb.poll_many(("ffn",), (SUMS_AT + rt * SUMS + qt[None, :],))
    total = gl.sum(got.to(gl.float32, bitcast=True), axis=0)  # [4 sums]
    ss_y = gl.sum(gl.where(qt == 0, total, 0.0), axis=0)
    ss_by = gl.sum(gl.where(qt == 1, total, 0.0), axis=0)
    ss_x = gl.sum(gl.where(qt == 2, total, 0.0), axis=0)
    ss_ah = gl.sum(gl.where(qt == 3, total, 0.0), axis=0)
    inv = gl.rsqrt(ss_y / HIDDEN + POST_EPS)
    # The sum of h''^2 over all columns: inv^2 ss_by + 2 inv ss_x + ss_ah.
    ss_h = gl.maximum(inv * inv * ss_by + 2.0 * inv * ss_x + ss_ah, 0.0)
    return inv, gl.rsqrt(ss_h / HIDDEN + PRE_EPS)


@gluon.jit
def _unpool_row(
    mb, task, y_lo, y_hi, hp_lo, hp_hi, post_lo, post_hi, b_lo, b_hi, a_lo, a_hi, RANK
):
    """One row's unpool epilogue: ``y`` from the raw sums ``y_lo``, ``y_hi``
    of the owned columns' pairs, the four sums, and the put into every rank's
    ``ffn``. Returns ``(y_lo, y_hi, (alpha h')_lo, (alpha h')_hi)`` for fin."""
    nw: gl.constexpr = gl.num_warps()
    y_lo = y_lo.to(gl.bfloat16).to(gl.float32) * post_lo
    y_hi = y_hi.to(gl.bfloat16).to(gl.float32) * post_hi
    ah_lo = a_lo * hp_lo.to(gl.float32, bitcast=True)
    ah_hi = a_hi * hp_hi.to(gl.float32, bitcast=True)
    by_lo = b_lo * y_lo
    by_hi = b_hi * y_hi
    s_y = gl.sum(y_lo * y_lo + y_hi * y_hi, axis=0)
    s_by = gl.sum(by_lo * by_lo + by_hi * by_hi, axis=0)
    s_x = gl.sum(ah_lo * by_lo + ah_hi * by_hi, axis=0)
    s_ah = gl.sum(ah_lo * ah_lo + ah_hi * ah_hi, axis=0)
    # One put per peer: rows y_lo, y_hi, (alpha h')_lo, (alpha h')_hi of the
    # 8 pairs, then the 4 sums in row 4.
    rows: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [1, nw], [1, 0])
    r = gl.arange(0, 8, gl.SliceLayout(1, rows))[:, None]
    jr = gl.arange(0, UNPOOL_COLS // 2, gl.SliceLayout(0, rows))[None, :]
    col = RANK * OWNED + task * UNPOOL_COLS + 2 * jr
    yr_lo = gl.convert_layout(y_lo, gl.SliceLayout(0, rows))[None, :]
    yr_hi = gl.convert_layout(y_hi, gl.SliceLayout(0, rows))[None, :]
    ar_lo = gl.convert_layout(ah_lo, gl.SliceLayout(0, rows))[None, :]
    ar_hi = gl.convert_layout(ah_hi, gl.SliceLayout(0, rows))[None, :]
    sr = gl.where(jr == 0, s_y, gl.where(jr == 1, s_by, gl.where(jr == 2, s_x, s_ah)))
    vals = gl.where(
        r == 0,
        yr_lo,
        gl.where(
            r == 1,
            yr_hi,
            gl.where(r == 2, ar_lo, gl.where(r == 3, ar_hi, sr)),
        ),
    )
    idx = gl.where(
        r < 4,
        (r >> 1) * HIDDEN + col + (r & 1),
        SUMS_AT + (RANK * UNPOOL_TASKS + task) * SUMS + jr,
    )
    word = vals.to(gl.int32, bitcast=True)
    put = (r < 4) | ((r == 4) & (jr < SUMS))
    for p in gl.static_range(WORLD):
        mb.put_peer("ffn", p, idx, word, mask=put)
    return y_lo, y_hi, ah_lo, ah_hi


@gluon.jit
def unpool_stages(mb, task, w, post_gain, alpha, beta, h_next, RANK: gl.constexpr):
    """Unpool and fin task ``task``: owned hidden columns ``16 task ..`` of
    every row.

    Args:
        w: This rank's post-expert projection rows, BF16 ``[1024, 4096]``
            (``weights.bf16_tiles``).
        post_gain: FP32 ``bf16(w + 1)`` of the post-FFN norm, owned columns.
        alpha: FP32 residual-gate coefficient of the FFN, owned columns.
        beta: FP32 residual-gate coefficient of the FFN, all 8192 columns.
        h_next: Where the owned FP32 residual ``h''`` goes, ``[rows, 1024]``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
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
    # The parameters load before the polls: a load after a poll would start
    # only when the poll returns.
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    j = gl.arange(0, UNPOOL_COLS // 2, lanes)
    c = task * UNPOOL_COLS + 2 * j
    post_lo = gl.load(post_gain + c)
    post_hi = gl.load(post_gain + c + 1)
    b_lo = gl.load(beta + RANK * OWNED + c)
    b_hi = gl.load(beta + RANK * OWNED + c + 1)
    a_lo = gl.load(alpha + c)
    a_hi = gl.load(alpha + c + 1)
    x_smem, _ = bf16.load_rows(mb, "cmb", 0, POOLED, R)
    hp = mb.poll_rows(("hp", "hp"), (c, c + 1), R)
    mb.stamp("unpool", trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # [RP, 16]
    mb.stamp("unpool", trace.COMPUTED)
    RP: gl.constexpr = out.shape[0]
    # split needs a pair's two values in one thread.
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [nw, 1], [1, 0])
    lo, hi = gl.split(
        gl.reshape(gl.convert_layout(out, pairs), [RP, UNPOOL_COLS // 2, 2])
    )
    rid = gl.arange(0, RP, gl.SliceLayout(1, lo.type.layout))[:, None]
    kept = ()
    for r in gl.static_range(R):
        kept += (
            _unpool_row(
                mb.row(r),
                task,
                gl.convert_layout(gl.sum(gl.where(rid == r, lo, 0.0), axis=0), lanes),
                gl.convert_layout(gl.sum(gl.where(rid == r, hi, 0.0), axis=0), lanes),
                hp[r][0],
                hp[r][1],
                post_lo,
                post_hi,
                b_lo,
                b_hi,
                a_lo,
                a_hi,
                RANK,
            ),
        )
    mb.stamp("unpool", trace.END)

    mb.stamp("fin", trace.START)
    invs = ()
    for r in gl.static_range(R):
        invs += (_ffn_norms(mb.row(r))[0],)
    mb.stamp("fin", trace.READY)
    for r in gl.static_range(R):
        y_lo, y_hi, ah_lo, ah_hi = kept[r]
        gl.store(h_next + r * OWNED + c, gl.fma(b_lo, y_lo * invs[r], ah_lo))
        gl.store(h_next + r * OWNED + c + 1, gl.fma(b_hi, y_hi * invs[r], ah_hi))
    mb.stamp("fin", trace.END)


@gluon.jit
def out_stage(mb, task, beta, in_gain, a_out, STAGE: gl.constexpr):
    """Out task ``task``: next attention input values ``512 (task % 16) ..``
    of row ``task // 16`` to ``a_out`` (BF16 ``[rows, 8192]``), from every
    rank's ``ffn``.

    Args:
        beta: FP32 residual-gate coefficient of the FFN, all 8192 columns.
        in_gain: FP32 ``bf16(w + 1)`` of the next layer's input norm, all
            8192 columns.
    """
    mb.stamp(STAGE, trace.START)
    row, task, _ = schedule.row_task(mb.plan, task, OUT_TASKS)
    mb = mb.row(row)
    a_out = a_out + row * HIDDEN
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = task * 256 + gl.arange(0, 256, lanes)
    c = 2 * i
    b_lo = gl.load(beta + c)
    b_hi = gl.load(beta + c + 1)
    in_lo = gl.load(in_gain + c)
    in_hi = gl.load(in_gain + c + 1)
    y_lo, y_hi, ah_lo, ah_hi = mb.poll_many(
        ("ffn", "ffn", "ffn", "ffn"), (c, c + 1, HIDDEN + c, HIDDEN + c + 1)
    )
    inv, inv2 = _ffn_norms(mb)
    mb.stamp(STAGE, trace.READY)
    h_lo = gl.fma(
        b_lo,
        y_lo.to(gl.float32, bitcast=True) * inv,
        ah_lo.to(gl.float32, bitcast=True),
    )
    h_hi = gl.fma(
        b_hi,
        y_hi.to(gl.float32, bitcast=True) * inv,
        ah_hi.to(gl.float32, bitcast=True),
    )
    word = _bf16_word(h_lo * inv2 * in_lo, h_hi * inv2 * in_hi)
    gl.store(a_out.to(gl.pointer_type(gl.int32), bitcast=True) + i, word)
    mb.stamp(STAGE, trace.END)
