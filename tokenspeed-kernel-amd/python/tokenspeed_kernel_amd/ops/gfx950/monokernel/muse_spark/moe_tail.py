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
  one store per rank. Then fin, after rnorm: the residual gate ``h'' = beta y
  inv + alpha h'`` of the owned columns (stored: the launch's residual output).
* rnorm, one task per row: the row's 512 x 4 partial sums from every rank's
  ``ffn``, summed once into the inverse RMS of the post-FFN norm (over 8192
  columns) and of the next layer's input norm, whose sum of squares is the
  expansion of ``(beta y inv + alpha h')^2`` in the four sums. Both go to the
  local ``fnorm``. (Every fin and out task polled all 2048 sums of every row
  itself before: at 8 rows, 64 fin CTAs x 128 KB of system-scope loads, and fin
  ended the layer about 6 us after unpool instead of 1 at one row.)
* out, 16 tasks of 512 hidden columns per row: the residual gate with rnorm's
  scales and the next layer's input norm. The result is rounded to BF16 and
  stored as the next attention input, for the next launch. Every rank
  computes the whole row from the ``ffn`` of all ranks, with fin's arithmetic:
  putting the owned slice of the result to the other ranks instead would add a
  second exchange to the end of the layer.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    bf16,
    mailbox,
    mx,
    schedule,
    trace,
)
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
        ("fnorm", 2),  # FP32: the row's post-FFN and next input inverse RMS
    )


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """Comb1, comb2, unpool, rnorm, fin and out of ``rows`` token rows,
    placed by ``bases``. Rnorm's CTAs must not run unpool: the kernel calls
    unpool and fin together (fin keeps unpool's values in registers)."""
    comb, unpool, rnorm, out = placement(bases, ("comb", "unpool", "rnorm", "out"))
    c, u = int(COMB_TASKS) * rows, int(UNPOOL_TASKS)
    if (rnorm - unpool) % 256 < u or (unpool - rnorm) % 256 < rows:
        raise ValueError(f"rnorm's CTAs {rnorm} .. overlap unpool's {unpool} ..")
    return (
        Stage("comb1", c, comb, reads=("rr", "route"), writes=("ss3",)),
        Stage("comb2", c, comb, reads=("ss3",), writes=("cmb",)),
        Stage("unpool", u, unpool, reads=("cmb", "hp"), writes=("ffn",)),
        Stage("rnorm", rows, rnorm, reads=("ffn",), writes=("fnorm",)),
        Stage("fin", u, unpool, reads=("fnorm",)),
        Stage("out", int(OUT_TASKS) * rows, out, reads=("ffn", "fnorm")),
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
    (got,) = mb.poll_many(("ffn",), (_ffn_sums_index(),))
    return _ffn_scales(got)


@gluon.jit
def rnorm_stage(mb, task, STAGE: gl.constexpr):
    """Rnorm task ``task``: row ``task``'s two inverse RMS (:func:`_ffn_norms`)
    into the local ``fnorm``."""
    mb.stamp(STAGE, trace.START)
    mb = mb.row(task)
    _pre_poll_sums(mb)
    inv, inv2 = _ffn_norms(mb)
    mb.stamp(STAGE, trace.READY)
    lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
    vals = gl.where(lanes == 0, inv, inv2)
    mb.put("fnorm", lanes, vals.to(gl.int32, bitcast=True), mask=lanes < 2)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _fnorm_ready(mb):
    """Wave 0 spins on the last row's ``fnorm`` (rnorm publishes every row's
    at about the same time) before a wide poll of the scales. ``mb`` must be
    the launch's mailbox, not a row's (:meth:`Mailbox.row` is relative)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
    mb.row(R - 1).pre_poll("fnorm", lanes % 2)


@gluon.jit
def _pre_poll_sums(mb):
    """:meth:`Mailbox.pre_poll` on 64 of ``ffn``'s sums, spread evenly (each
    unpool task puts its sums last in its one put)."""
    N: gl.constexpr = WORLD * UNPOOL_TASKS * SUMS
    lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
    mb.pre_poll("ffn", SUMS_AT + gl.minimum(lanes * N // 64, N - 1))


@gluon.jit
def _ffn_sums_index():
    """The ``ffn`` pair index of every rank's and task's 4 sums."""
    nw: gl.constexpr = gl.num_warps()
    sl: gl.constexpr = gl.BlockedLayout([1, SUMS], [64, 1], [nw, 1], [1, 0])
    rt = gl.arange(0, WORLD * UNPOOL_TASKS, gl.SliceLayout(1, sl))[:, None]
    qt = gl.arange(0, SUMS, gl.SliceLayout(0, sl))[None, :]
    return SUMS_AT + rt * SUMS + qt


@gluon.jit
def _ffn_scales(got):
    """:func:`_ffn_norms` from the polled sums (:func:`_ffn_sums_index`)."""
    nw: gl.constexpr = gl.num_warps()
    sl: gl.constexpr = gl.BlockedLayout([1, SUMS], [64, 1], [nw, 1], [1, 0])
    qt = gl.arange(0, SUMS, gl.SliceLayout(0, sl))
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
def unpool_stages(
    mb,
    task,
    w,
    post_gain,
    alpha,
    beta,
    h_next,
    tap,
    tap_col,
    tap_row,
    RANK: gl.constexpr,
    TAP: gl.constexpr,
):
    """Unpool and fin task ``task``: owned hidden columns ``16 task ..`` of
    every row.

    Args:
        w: This rank's post-expert projection rows, BF16 ``[1024, 4096]``
            (``weights.bf16_tiles``).
        post_gain: FP32 ``bf16(w + 1)`` of the post-FFN norm, owned columns.
        alpha: FP32 residual-gate coefficient of the FFN, owned columns.
        beta: FP32 residual-gate coefficient of the FFN, all 8192 columns.
        h_next: Where the owned FP32 residual ``h''`` goes, ``[rows, 1024]``.
        tap, tap_col, tap_row: With ``TAP``, a BF16 buffer on the Iris
            symmetric heap where every rank's copy gets ``bf16(h'')`` of the
            owned columns: row ``r``, column ``c`` (of 8192) at ``tap + r
            tap_row + tap_col + c``. A draft reads this residual (the one
            entering the next layer) as a tap. Unused without ``TAP``.
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
    if R > 1:
        # Past one row every wave of many CTAs polls several whole rows:
        # one wave spins on hints first (unthrottled, the flood of polls
        # starves the producers' stores across ranks and the grid livelocks).
        mb.row(R - 1).pre_poll_spread("cmb", POOLED // 2)
    x_smem, _ = bf16.load_rows(mb, "cmb", 0, POOLED, R)
    # Every row's h' of the task's pairs, in the epilogue's [row, pair] layout.
    RPP: gl.constexpr = triton.next_power_of_2(R)
    hl: gl.constexpr = gl.SliceLayout(
        1, gl.BlockedLayout([1, 1, 1], [1, 8, 8], [nw, 1, 1], [2, 1, 0])
    )
    hr = gl.arange(0, RPP, gl.SliceLayout(1, hl))[:, None]
    hc = task * UNPOOL_COLS + 2 * gl.arange(0, UNPOOL_COLS // 2, gl.SliceLayout(0, hl))
    hat = mb.row_offset(hr) + hc[None, :]
    hp = mb.poll_many(("hp", "hp"), (hat, hat + 1))
    mb.stamp("unpool", trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # [RP, 16]
    mb.stamp("unpool", trace.COMPUTED)
    RP: gl.constexpr = out.shape[0]
    # split needs a pair's two values in one thread.
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [nw, 1], [1, 0])
    lo, hi = gl.split(
        gl.reshape(gl.convert_layout(out, pairs), [RP, UNPOOL_COLS // 2, 2])
    )
    # Every row at once: [row, kind, pair], a wave per row. Kinds 0 .. 3 are
    # y_lo, y_hi, (alpha h')_lo, (alpha h')_hi of the 8 pairs; kind 4 holds
    # the row's 4 sums. One put per peer serves every row.
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 8, 8], [nw, 1, 1], [2, 1, 0])
    P2: gl.constexpr = gl.SliceLayout(1, L3)
    r2 = gl.arange(0, RP, gl.SliceLayout(1, P2))[:, None]
    j2 = gl.arange(0, UNPOOL_COLS // 2, gl.SliceLayout(0, P2))[None, :]
    c2 = task * UNPOOL_COLS + 2 * j2
    hp_lo = hp[0].to(gl.float32, bitcast=True)
    hp_hi = hp[1].to(gl.float32, bitcast=True)
    y_lo = gl.convert_layout(lo, P2).to(gl.bfloat16).to(gl.float32)
    y_hi = gl.convert_layout(hi, P2).to(gl.bfloat16).to(gl.float32)
    y_lo = y_lo * gl.convert_layout(post_lo, gl.SliceLayout(0, P2))[None, :]
    y_hi = y_hi * gl.convert_layout(post_hi, gl.SliceLayout(0, P2))[None, :]
    a2_lo = gl.convert_layout(a_lo, gl.SliceLayout(0, P2))[None, :]
    a2_hi = gl.convert_layout(a_hi, gl.SliceLayout(0, P2))[None, :]
    b2_lo = gl.convert_layout(b_lo, gl.SliceLayout(0, P2))[None, :]
    b2_hi = gl.convert_layout(b_hi, gl.SliceLayout(0, P2))[None, :]
    ah_lo = a2_lo * hp_lo
    ah_hi = a2_hi * hp_hi
    by_lo = b2_lo * y_lo
    by_hi = b2_hi * y_hi
    s_y = gl.sum(y_lo * y_lo + y_hi * y_hi, axis=1)[:, None]
    s_by = gl.sum(by_lo * by_lo + by_hi * by_hi, axis=1)[:, None]
    s_x = gl.sum(ah_lo * by_lo + ah_hi * by_hi, axis=1)[:, None]
    s_ah = gl.sum(ah_lo * ah_lo + ah_hi * ah_hi, axis=1)[:, None]
    sr = gl.where(j2 == 0, s_y, gl.where(j2 == 1, s_by, gl.where(j2 == 2, s_x, s_ah)))
    k = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    j3 = j2[:, None, :]
    r3 = r2[:, None, :]
    vals = gl.where(
        k == 0,
        y_lo[:, None, :],
        gl.where(
            k == 1,
            y_hi[:, None, :],
            gl.where(
                k == 2,
                ah_lo[:, None, :],
                gl.where(k == 3, ah_hi[:, None, :], sr[:, None, :]),
            ),
        ),
    )
    col = RANK * OWNED + task * UNPOOL_COLS + 2 * j3
    idx = mb.row_offset(r3) + gl.where(
        k < 4,
        (k >> 1) * HIDDEN + col + (k & 1),
        SUMS_AT + (RANK * UNPOOL_TASKS + task) * SUMS + j3,
    )
    word = vals.to(gl.int32, bitcast=True)
    put = ((k < 4) | ((k == 4) & (j3 < SUMS))) & (r3 < R)
    for p in gl.static_range(WORLD):
        mb.put_peer("ffn", p, idx, word, mask=put)
    mb.stamp("unpool", trace.END)

    mb.stamp("fin", trace.START)
    _fnorm_ready(mb)
    (got,) = mb.poll_many(("fnorm",), (mb.row_offset(r2) + 0 * j2,))
    mb.stamp("fin", trace.READY)
    # [RP, 8]: each pair's row scale (padded rows repeat the last row's).
    inv = got.to(gl.float32, bitcast=True)
    at = h_next + r2 * OWNED + c2
    keep = (r2 < R) & (j2 >= 0)
    h_lo = gl.fma(b2_lo, y_lo * inv, ah_lo)
    h_hi = gl.fma(b2_hi, y_hi * inv, ah_hi)
    gl.store(at, h_lo, mask=keep)
    gl.store(at + 1, h_hi, mask=keep)
    if TAP:
        # The draft's tap of the residual, rounded once to BF16 as the
        # regular capture rounds the FP32 stream, into every rank's buffer.
        # Its reader runs after later collectives of the forward, which
        # order these stores before it.
        t_at = tap + r2.to(gl.int64) * tap_row + tap_col + RANK * OWNED + c2
        for p in gl.static_range(WORLD):
            peer_at = mb.peers.translate(t_at, p)
            gl.store(peer_at, h_lo.to(gl.bfloat16), mask=keep)
            gl.store(peer_at + 1, h_hi.to(gl.bfloat16), mask=keep)
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
    # Before mb.row: Mailbox.row is relative to the mailbox it is called on.
    _fnorm_ready(mb)
    mb = mb.row(row)
    a_out = a_out + row * HIDDEN
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = task * 256 + gl.arange(0, 256, lanes)
    c = 2 * i
    b_lo = gl.load(beta + c)
    b_hi = gl.load(beta + c + 1)
    in_lo = gl.load(in_gain + c)
    in_hi = gl.load(in_gain + c + 1)
    y_lo, y_hi, ah_lo, ah_hi, g1, g2 = mb.poll_many(
        ("ffn", "ffn", "ffn", "ffn", "fnorm", "fnorm"),
        (c, c + 1, HIDDEN + c, HIDDEN + c + 1, 0 * i, 0 * i + 1),
    )
    inv = g1.to(gl.float32, bitcast=True)
    inv2 = g2.to(gl.float32, bitcast=True)
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
