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

"""BF16 experts of one Muse Spark rank (8 picks a row), for the layers whose
checkpoint keeps BF16 experts (layers 0 and 61).

The same tasks, slots and mailbox regions as :mod:`moe_experts` (MXFP4),
whose polls, slot union and K sums (``exr``'s first half) these stages
share. The weights are the regular BF16 route path's, read in place: w13
``[256, 1024, 4096]`` (the rank's 512 gate rows, then its 512 up rows) and
w2 ``[256, 4096, 512]``. A BF16 slot is 4 times an MXFP4 slot's bytes, too
many for a task's registers: the stages stream each slot's weights in
chunks, one chunk ahead.

* ex1, 256 tasks: lane ``l``, column block ``b`` (gate rows ``64 b ..`` and
  the matching up rows), a quarter of K. The normed BF16 rows (no
  quantization) times the block, in four 256-wide K chunks. The FP32 partial
  sums go to ``p1`` in the MXFP4 path's order (gate and up columns
  interleaved), so ``exr`` reads them alike.
* exr, 64 tasks: the K sums and SwiGLU as :func:`moe_experts.exr_stage`,
  then the regular route path's rounding of the intermediate to BF16,
  published as words (256 a row and pick).
* ex2, 256 tasks: lane ``l``, 128 of the 4096 output columns over the
  rank's 512 intermediate values, in two K halves. The BF16 partials go to
  ``rr`` as :mod:`moe_experts`' do.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.moe_experts import (
    BLOCK_N,
    H1_BF16_WORDS,
    INTER,
    K_SLICE,
    KSPLIT,
    LANES,
    M_DUP,
    NB1,
    NB2,
    OWNED_OUT,
    POOLED,
    PRE_EPS,
    TOP_K,
    _cg_layout,
    _routed_order,
    _row_picks,
    _slot_expert,
)

cdna4 = gl.amd.cdna4

# 32-value k-iterations of an ex1 chunk: 256 values of K, 64 KiB of a task's
# weights a slot.
EX1_CHUNK = gl.constexpr(8)
EX1_CHUNKS = gl.constexpr(4)  # K_SLICE // (32 * EX1_CHUNK)
EX2_HALF = gl.constexpr(8)  # k-iterations of an ex2 half: 256 of the 512
NW = gl.constexpr(8)  # N waves of 16 columns: BLOCK_N columns a task
GATE_BLOCKS = gl.constexpr(32)  # 16-row blocks of the gate rows (INTER // 16)


@gluon.jit
def _tiles(
    w,
    expert,
    stride_e,
    nb0,
    GROUP: gl.constexpr,
    GROUP_STRIDE: gl.constexpr,
    K: gl.constexpr,
    kit0,
    KITS: gl.constexpr,
    active,
):
    """Issue ``KITS`` B tiles ``[1, 32, 128]`` (k-iterations ``kit0 ..``) of
    expert ``expert``'s row-major ``[N, K]`` weight at ``w + expert
    stride_e``: N wave ``v`` reads 16-row block ``nb0 + (v // GROUP)
    GROUP_STRIDE + v % GROUP`` (as :func:`bf16.load_row_major_tiles`).
    Nothing loads when ``active`` is false or ``expert`` is -1."""
    bl: gl.constexpr = bf16.b_layout(1, NW)
    s = gl.arange(0, 1, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    k = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    v = n // 16
    row = (nb0 + (v // GROUP) * GROUP_STRIDE + v % GROUP) * 16 + n % 16
    # Offsets within one expert stay 32-bit (a w13 expert is 4M values);
    # the expert's base advances the pointer in 64 bits.
    offset = row[None, None, :] * K + (kit0 + s)[:, None, None] * 32 + k[None, :, None]
    on = active & (expert >= 0)
    base = w + gl.maximum(expert, 0).to(gl.int64) * stride_e
    mask = (offset >= 0) & on
    tiles = ()
    for i in gl.static_range(KITS):
        # Each weight is read once a launch: loads that skip L1 stream faster.
        tiles += (cdna4.buffer_load(base, offset + i * 32, mask=mask, cache=".cg"),)
    return tiles


@gluon.jit
def _mma(x3, kit0: gl.constexpr, tiles, acc):
    """``acc`` plus LDS rows ``x3`` (BF16 ``[16, KP, 32]``) at k-iterations
    ``kit0 ..`` times ``tiles``."""
    al: gl.constexpr = bf16.a_layout(1, NW)
    for i in gl.static_range(len(tiles)):
        a = bf16._a_tile(x3, kit0 + i, 0, M_DUP, 1, al)
        acc = cdna4.mfma(a, tiles[i], acc)
    return acc


@gluon.jit
def _ex1_tiles(task, active, expert, w13, stride_we, chunk: gl.constexpr):
    """Ex1 task ``task``'s w13 tiles of K chunk ``chunk`` of its quarter."""
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    # Waves 0 .. 3 read gate rows 64 blk .., waves 4 .. 7 the up rows 512 +
    # 64 blk ..: result column c < 64 is gate c, 64 + c is up c.
    return _tiles(
        w13,
        expert,
        stride_we,
        blk * 4,
        4,
        GATE_BLOCKS,
        POOLED,
        ks * (K_SLICE // 32) + chunk * EX1_CHUNK,
        EX1_CHUNK,
        active,
    )


@gluon.jit
def ex1_issue(task, active, expert, w13, stride_we):
    """Issue the first K chunk of ex1 task ``task``'s w13 tiles of expert
    ``expert`` (:func:`moe_experts.ex1_issue`'s place)."""
    return _ex1_tiles(task, active, expert, w13, stride_we, 0)


@gluon.jit
def ex1_operand(task, rows, gain_lo, gain_hi):
    """The A operand of ex1 task ``task``: each row's pre-expert RMS norm,
    the task's K quarter rounded to BF16 (the regular path's expert input)
    into LDS as ``[16, 32, 32]`` (rows past the launch's are zero)."""
    ks = task % KSPLIT
    nw: gl.constexpr = gl.num_warps()
    cl: gl.constexpr = _cg_layout(nw)
    G: gl.constexpr = rows[0].shape[0]
    q = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, cl)))[None, :, None]
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_smem = gl.allocate_shared_memory(gl.bfloat16, [M_DUP, K_SLICE], plain)
    for g in gl.static_range(len(rows)):
        got = rows[g]
        lo = (got << 16).to(gl.float32, bitcast=True)
        hi = (got & -65536).to(gl.float32, bitcast=True)
        inv = gl.rsqrt(
            gl.sum(gl.sum(lo * lo + hi * hi, axis=1), axis=1) / POOLED + PRE_EPS
        )[:, None]
        mine = q == ks
        lo = gl.sum(gl.where(mine, lo, 0.0), axis=1) * inv * gain_lo[None, :]
        hi = gl.sum(gl.where(mine, hi, 0.0), axis=1) * inv * gain_hi[None, :]
        x = gl.reshape(gl.join(lo, hi), [G, K_SLICE])
        x_smem.slice(g * G, G, dim=0).store(x.to(gl.bfloat16))
    zl: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, nw], [1, 0])
    RP: gl.constexpr = G * len(rows)
    for b in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << b) < M_DUP:
            x_smem.slice(RP << b, RP << b, dim=0).store(
                gl.zeros([RP << b, K_SLICE], gl.bfloat16, zl)
            )
    gl.barrier()
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    return x_smem.reinterpret(gl.bfloat16, [M_DUP, K_SLICE // 32, 32], flat)


@gluon.jit
def _ex1_put(mb, task, acc, union, expert):
    """Publish one slot's FP32 partials (``acc`` from :func:`_mma`) of the
    rows that picked ``expert``, in the MXFP4 path's ``p1`` order."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    out = mx.gemv_result(acc, M_DUP)  # [16, 128]
    ol: gl.constexpr = out.type.layout
    picks = _row_picks(union, expert, R)
    m = gl.arange(0, M_DUP, gl.SliceLayout(1, ol))
    k = gl.full([M_DUP], -1, gl.int32, m.type.layout)
    for r in gl.static_range(R):
        k = gl.where(m == r, picks[r], k)
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, ol))[None, :]
    col = gl.where(n < BLOCK_N // 2, 2 * n, 2 * (n - BLOCK_N // 2) + 1)
    at = ((k[:, None] * NB1 + blk) * KSPLIT + ks) * BLOCK_N + col
    mb.put(
        "p1",
        mb.row_offset(m[:, None]) + at,
        out.to(gl.int32, bitcast=True),
        mask=(k[:, None] >= 0) & (n >= 0),
    )


@gluon.jit
def ex1_finish(mb, task, x3, tiles, union, w13, stride_we, STAGE: gl.constexpr):
    """Ex1 task ``task`` after :func:`ex1_operand` (``x3``), :func:`ex1_issue`
    (``tiles``, the first chunk of the lane's first slot) and
    :func:`moe_experts.slots` (``union``): every slot of the lane, each in
    K chunks with the next chunk's loads issued before a chunk's products."""
    lane = task // (NB1 * KSPLIT)
    count = union[3]
    for i in range(lane, count, LANES):
        expert = _slot_expert(union, i)
        acc = gl.zeros([1, 16, BLOCK_N], gl.float32, bf16.mfma_layout(1, NW))
        for c in gl.static_range(EX1_CHUNKS):
            if c + 1 < EX1_CHUNKS:
                ahead = _ex1_tiles(task, True, expert, w13, stride_we, c + 1)
            else:
                ahead = _ex1_tiles(
                    task, True, _slot_expert(union, i + LANES), w13, stride_we, 0
                )
            acc = _mma(x3, c * EX1_CHUNK, tiles, acc)
            tiles = ahead
        _ex1_put(mb, task, acc, union, expert)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _ex2_tiles(task, active, expert, w2, stride_we, half: gl.constexpr):
    """Ex2 task ``task``'s w2 tiles of K half ``half``: 128 output rows."""
    blk = task % NB2
    return _tiles(
        w2, expert, stride_we, blk * NW, NW, 0, INTER, half * EX2_HALF, EX2_HALF, active
    )


@gluon.jit
def ex2_issue(mb, task, active, expert, w2, stride_we, STAGE: gl.constexpr):
    """Issue the first K half of ex2 task ``task``'s w2 tiles of expert
    ``expert`` (:func:`moe_experts.ex2_issue`'s place)."""
    if active:
        mb.stamp(STAGE, trace.START)
    return _ex2_tiles(task, active, expert, w2, stride_we, 0)


@gluon.jit
def _ex2_gather(mb, union, expert, x_words):
    """Poll the BF16 intermediates of the rows that picked ``expert`` into
    ``x_words`` (int32 ``[16, 256]``), MFMA row ``m`` the ``m``-th routed
    row; the other rows hold zeros. The caller orders the LDS. Returns
    ``(picks, order)``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    G: gl.constexpr = min(RP, 8)
    nw: gl.constexpr = gl.num_warps()
    picks = _row_picks(union, expert, R)
    order, count = _routed_order(picks, R)
    wl: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [1, nw], [1, 0])
    v = gl.arange(0, H1_BF16_WORDS, gl.SliceLayout(0, wl))[None, :]
    from_rows: gl.constexpr = gl.SliceLayout(1, wl)
    for g in gl.static_range(RP // G):
        m = g * G + gl.arange(0, G, from_rows)
        row = gl.full([G], -1, gl.int32, from_rows)
        pick = gl.full([G], -1, gl.int32, from_rows)
        row0 = gl.to_tensor(-1)
        pick0 = gl.to_tensor(-1)
        for r in gl.static_range(R):
            hit = (picks[r] >= 0) & (m == order[r])
            row = gl.where(hit, r, row)
            pick = gl.where(hit, picks[r], pick)
            first = (picks[r] >= 0) & (order[r] == 0)
            row0 = gl.where(first, r, row0)
            pick0 = gl.where(first, picks[r], pick0)
        routed = m < count
        row = gl.where(routed, row, row0)
        pick = gl.where(routed, pick, pick0)
        (got,) = mb.poll_many(
            ("h1",), (mb.row_offset(row[:, None]) + pick[:, None] * H1_BF16_WORDS + v,)
        )
        x_words.slice(g * G, G, dim=0).store(gl.where(routed[:, None], got, 0))
    for k in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << k) < M_DUP:
            x_words.slice(RP << k, RP << k, dim=0).store(
                gl.zeros([RP << k, H1_BF16_WORDS], gl.int32, wl)
            )
    return picks, order


@gluon.jit
def _ex2_put(mb, task, acc, picks, order, RANK: gl.constexpr):
    """The BF16 partials of one slot (``acc``) to the rank that owns the
    task's columns, as :func:`moe_experts._ex2_compute` puts them."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk0 = task % NB2
    owner = (blk0 * BLOCK_N) // OWNED_OUT
    out = gl.reshape(
        mx.gemv_result(acc, M_DUP).to(gl.bfloat16), [M_DUP, BLOCK_N // 2, 2]
    )
    # A pair in one thread, for the split.
    pairs: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2], [1, 64, 1], [gl.num_warps(), 1, 1], [2, 1, 0]
    )
    word = mx.pack_bf16_pairs(gl.convert_layout(out, pairs))
    wl: gl.constexpr = word.type.layout
    mm = gl.arange(0, M_DUP, gl.SliceLayout(1, wl))[:, None]
    p = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, wl))[None, :]
    row_of = gl.full([M_DUP, 1], 0, gl.int32, wl)
    pick_of = gl.full([M_DUP, 1], -1, gl.int32, wl)
    for r in gl.static_range(R):
        mine = (mm == order[r]) & (picks[r] >= 0)
        row_of = gl.where(mine, r, row_of)
        pick_of = gl.where(mine, picks[r], pick_of)
    col = (blk0 * BLOCK_N) % OWNED_OUT
    at = ((RANK * TOP_K + pick_of) * OWNED_OUT + col) // 2 + p
    mb.put_peer(
        "rr",
        owner,
        mb.row_offset(row_of) + at,
        word,
        mask=(pick_of >= 0) & (p >= 0),
    )


@gluon.jit
def ex2_finish(
    mb, task, tiles, union, w2, stride_we, RANK: gl.constexpr, STAGE: gl.constexpr
):
    """Ex2 task ``task`` after :func:`ex2_issue` (``tiles``, the first half of
    the lane's first slot) and :func:`moe_experts.slots` (``union``): each
    slot's polls, then its two halves with the next half's loads issued
    before a half's products."""
    lane = task // NB2
    count = union[3]
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    x_words = gl.allocate_shared_memory(gl.int32, [M_DUP, H1_BF16_WORDS], plain)
    x3 = x_words.reinterpret(gl.bfloat16, [M_DUP, INTER // 32, 32], flat)
    for i in range(lane, count, LANES):
        expert = _slot_expert(union, i)
        # The previous slot's products read the buffer (and, before the
        # first slot, the LDS may hold ex1's operand).
        gl.barrier()
        picks, order = _ex2_gather(mb, union, expert, x_words)
        gl.barrier()
        if i == lane:
            mb.stamp(STAGE, trace.READY)
        second = _ex2_tiles(task, True, expert, w2, stride_we, 1)
        acc = gl.zeros([1, 16, BLOCK_N], gl.float32, bf16.mfma_layout(1, NW))
        acc = _mma(x3, 0, tiles, acc)
        tiles = _ex2_tiles(task, True, _slot_expert(union, i + LANES), w2, stride_we, 0)
        acc = _mma(x3, EX2_HALF, second, acc)
        _ex2_put(mb, task, acc, picks, order, RANK)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)
