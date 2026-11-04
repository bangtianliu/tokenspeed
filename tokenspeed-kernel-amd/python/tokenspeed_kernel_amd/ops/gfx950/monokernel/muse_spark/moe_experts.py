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

"""MXFP4 experts of one Muse Spark rank (8 picks a row).

At TP8 / EP1 every rank holds all 256 experts, each sharded over its
intermediate dimension: the rank's 512 of 4096. The stages read the regular
path's preprocessed weights in place (gdot128-shuffled E2M1 bytes and
CDNA4-swizzled E8M0 scales, as ``decode_stage1`` / ``decode_stage2`` read
them): a second copy of every expert would not fit.

The launch's rows pick ``U`` distinct experts (8 for one row; consecutive
tokens share many). A pick owns its expert when no earlier row picked it;
slot ``i`` is the ``i``-th owning pick in (row, pick) order. Row 0 owns all
its picks, so slot ``i < 8`` is row 0's pick ``i`` at any row count. A task
of lane ``l`` takes slots ``l, l + 8, ...``: each expert's weights stream
once for every row that picked it. Row ``r``'s result for slot ``i`` goes to
row ``r``'s entry of the pick that chose that expert, so the combine reads
each row's 8 picks as for one row.

* ex1, 256 tasks: lane ``l``, 128 of the 1024 interleaved gate / up columns,
  a quarter of K (64 KiB of weights a slot). A task polls every whole pooled
  row, applies the pre-expert RMS norm, and quantizes its K quarter of every
  row to MXFP4 (the regular path's "even" rule) in LDS, one MFMA row each.
  It publishes the FP32 partial sums of the rows that picked the slot's
  expert.
* exr, 64 tasks: lane ``l``, one ex1 column block. For each slot, a task
  polls the 4 K quarters of every row that picked the slot's expert, sums
  them in K order, applies SwiGLU (``silu(gate) up``) and publishes the
  block's 64 intermediate values a row (``h1``). Summed once here, an ex2
  task reads 512 values a row instead of 4096 partials: polled by all 256
  CTAs at once, the partials took one 0.7 us round trip a row (the
  memory-side bandwidth, not the latency), so a slot that many rows
  picked held its lane up.
* ex2, 256 tasks: lane ``l``, 128 of the 4096 output columns over the rank's
  512 intermediate values (32 KiB a slot). A task polls the intermediates of
  the rows that picked the slot's expert (up to 8 rows a poll) and applies
  the regular path's MXFP4 requantization (groups of 32) itself. The BF16
  partials go to the rank that owns those columns (``rr``), which sums the
  8 ranks' partials.

Expert weights cannot load before routing picks the experts. So a CTA runs
the first slot of its ex1 and ex2 tasks in parts: :func:`ex1_poll` (one
poll for the pooled rows, the first slots' experts and every row's picks),
:func:`ex1_issue`, then :func:`slots` and :func:`ex1_operand` (the slots of
the other picks, and the norm and quantization of the A operand, while the
weights stream), :func:`ex1_finish`, then :func:`ex2_issue` and
:func:`ex2_finish` at ex2's place in the plan. ex2's first weights issue
after ex1's products: issued with ex1's they sat on ex1's critical path (a
full layer at 200k keys took 148.6 us instead of 145.9), and they still
arrive before ex2's poll returns. The loads come before the polls: a poll
waits for every earlier load of its wave (``vmcnt`` counts in order), so a
load after a poll would start only when the poll returns. The finish steps
then run the lane's other slots, one after another.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _cdna4_swizzled_mxfp4_scale_offset,
    _direct_mxfp4_mfma,
    _direct_mxfp4_mfma_layouts,
    _gluon_dot_preshuffled_w_offset,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
    _mxfp4_quantize_tile_in_layout,
    scaled_downcast_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.moe_pool import POOLED
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

TOP_K = gl.constexpr(8)
EXPERTS = gl.constexpr(256)
LANES = gl.constexpr(8)  # slot lanes: the tasks of a column block
INTER = gl.constexpr(512)  # the rank's intermediate values of an expert
W13_N = gl.constexpr(1024)  # interleaved gate / up columns
BLOCK_N = gl.constexpr(128)
M_DUP = gl.constexpr(16)  # MFMA rows; the activation row repeats in each
KSPLIT = gl.constexpr(4)
K_SLICE = gl.constexpr(1024)  # POOLED // KSPLIT
NB1 = gl.constexpr(8)  # W13_N // BLOCK_N
NB2 = gl.constexpr(32)  # POOLED // BLOCK_N
EX1_TASKS = gl.constexpr(256)  # LANES * NB1 * KSPLIT
EXR_TASKS = gl.constexpr(64)  # LANES * NB1
EX2_BLOCKS = gl.constexpr(1)  # 128-column blocks of an ex2 task
EX2_TASKS = gl.constexpr(256)  # LANES * NB2 // EX2_BLOCKS
OWNED_OUT = gl.constexpr(512)  # output columns a rank reduces: POOLED / WORLD
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("rr",)


def regions() -> tuple[tuple[str, int], ...]:
    k, nb1, ks, bn = int(TOP_K), int(NB1), int(KSPLIT), int(BLOCK_N)
    return (
        ("p1", k * nb1 * ks * bn),  # FP32 [pick][block][K quarter][column] a row
        ("h1", k * int(INTER)),  # FP32 SwiGLU intermediate [pick][value] a row
        (
            "rr",
            int(WORLD) * k * int(OWNED_OUT) // 2,
        ),  # BF16 pairs [source][pick][column] a row
    )


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    ex1, exr, ex2 = placement(bases, ("ex1", "exr", "ex2"))
    return (
        Stage("ex1", int(EX1_TASKS), ex1, reads=("route", "cg"), writes=("p1",)),
        Stage("exr", int(EXR_TASKS), exr, reads=("route", "p1"), writes=("h1",)),
        Stage("ex2", int(EX2_TASKS), ex2, reads=("route", "h1"), writes=("rr",)),
    )


@gluon.constexpr_function
def _pick_layout(NP, num_warps):
    """``[pick, pick]`` of the launch's ``NP`` (row, pick) ids: a lane per
    pick along axis 0 (the waves repeat it), every pick in each thread along
    axis 1."""
    return gl.BlockedLayout([1, NP], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def slots(ids):
    """The slots of the launch's picks ``ids [NP]`` (from :func:`ex1_poll`, a
    lane each, rows major). Returns ``(ids, owner, slot, count)``: the ids,
    whether each pick owns its expert, and its slot (a lane each), and the
    number of slots. Only a wave's lanes talk.

    It issues no global load: it runs while the first slots' weights
    stream."""
    NP: gl.constexpr = ids.shape[0]
    L2: gl.constexpr = _pick_layout(NP, gl.num_warps())
    ids = gl.convert_layout(ids, gl.SliceLayout(0, L2))
    q = gl.arange(0, NP, gl.SliceLayout(0, L2))[None, :]
    p = gl.arange(0, NP, gl.SliceLayout(1, L2))[:, None]
    mine = gl.convert_layout(ids, gl.SliceLayout(1, L2))[:, None]
    # A pick owns its expert unless an earlier row picked it (a row's picks
    # are distinct).
    seen = (ids[None, :] == mine) & (q // TOP_K < p // TOP_K)
    # A pick a lane; the scan lowers on a plain layout, not on a slice.
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    owner = gl.convert_layout(1 - gl.max(seen.to(gl.int32), axis=1), lanes)
    slot = gl.cumsum(owner, axis=0) - owner
    # Each pick's row picks: for row r, the pick of r that chose the same
    # expert (or -1). A slot then reads its rows' picks in one reduction
    # (:func:`_row_picks`), not one a row.
    RP: gl.constexpr = NP // TOP_K
    same = ids[None, :] == mine
    k = gl.where(same, q % TOP_K, -1)
    table = gl.max(gl.reshape(k, [NP, RP, TOP_K]), axis=2)
    # The ids stay a lane each, not a copy in every thread: the expert steps
    # run at the register limit past one row, and the copies spill.
    return (
        gl.convert_layout(ids, lanes),
        owner,
        slot,
        gl.sum(owner, axis=0),
        table,
    )


@gluon.jit
def _slot_expert(union, i):
    """The expert of slot ``i``, or -1 past the last slot."""
    ids, owner, slot, _, _ = union
    mine = gl.convert_layout(ids, owner.type.layout)
    return gl.max(gl.where((owner > 0) & (slot == i), mine, -1), axis=0)


@gluon.jit
def _row_picks(union, expert, R: gl.constexpr):
    """The pick of each row that chose ``expert``, or -1: a tuple of ``R``
    scalars. One reduction across the waves gives every thread the whole
    row of the table (:func:`slots`) of the expert's owner pick; the scalars
    then come from the thread's own registers."""
    ids, owner, _, _, table = union
    tl: gl.constexpr = table.type.layout
    mine = gl.convert_layout(ids, gl.SliceLayout(1, tl))
    own = gl.convert_layout(owner, gl.SliceLayout(1, tl))
    hit = ((own > 0) & (mine == expert))[:, None]
    row = gl.max(gl.where(hit, table, -1), axis=0)  # [RP], in every thread
    r = gl.arange(0, table.shape[1], row.type.layout)
    picks = ()
    for i in gl.static_range(R):
        picks += (gl.max(gl.where(r == i, row, -1), axis=0),)
    return picks


@gluon.jit
def _load_b(
    w,
    ws,
    expert,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    n0,
    kt0,
    KT: gl.constexpr,
    N_PHYS: gl.constexpr,
    active=True,
):
    """Issue the B tiles of K tiles ``kt0 ..`` (128 values each) of columns
    ``n0 .. n0 + 127``: ``KT`` (weights, scales) pairs. Nothing loads when
    ``active`` is false or ``expert`` is -1."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    dot_b: gl.constexpr = layouts[2]
    b_scale: gl.constexpr = layouts[4]
    bk = gl.arange(0, 64, gl.SliceLayout(1, dot_b))[:, None]
    bn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(0, dot_b))[None, :]
    sn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale))[:, None]
    sk = gl.arange(0, 4, gl.SliceLayout(0, b_scale))[None, :]
    on = active & (expert >= 0)
    expert = gl.maximum(expert, 0)
    w_off = expert.to(gl.int64) * stride_we
    s_off = expert.to(gl.int64) * stride_se
    tiles = ()
    for i in gl.static_range(KT):
        kt = kt0 + i
        b = gl.amd.cdna4.buffer_load(
            w,
            _gluon_dot_preshuffled_w_offset(w_off, kt * 64 + bk, bn, N_PHYS).to(
                gl.int32
            ),
            mask=(bk >= 0) & on,
        )
        s = gl.amd.cdna4.buffer_load(
            ws,
            _cdna4_swizzled_mxfp4_scale_offset(
                s_off, sn, kt * 4 + sk, stride_slin, stride_snb
            ).to(gl.int32),
            mask=(sk >= 0) & on,
        )
        tiles += ((b, s),)
    return tiles


@gluon.jit
def _mma_rows(packed_smem, scale_smem, tiles):
    """The MXFP4 product of the LDS A rows and the B ``tiles``: FP32 ``[16,
    128]``, one row per A row."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    mfma: gl.constexpr = layouts[0]
    acc = gl.zeros([M_DUP, BLOCK_N], gl.float32, mfma)
    for i in gl.static_range(len(tiles)):
        a = packed_smem.slice(i * 64, 64, 1).load(layouts[1])
        a_s = scale_smem.slice(i * 4, 4, 1).load(layouts[3])
        acc = _direct_mxfp4_mfma(acc, a, tiles[i][0], a_s, tiles[i][1])
    return acc


@gluon.jit
def _stage_a(packed, scales):
    """MXFP4 A rows (``[16, N / 2]`` bytes, ``[16, N / 32]`` scales) to LDS."""
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    p_smem = gl.allocate_shared_memory(gl.uint8, packed.shape, plain)
    s_smem = gl.allocate_shared_memory(gl.uint8, scales.shape, plain)
    p_smem.store(packed)
    s_smem.store(scales)
    gl.barrier()
    return p_smem, s_smem


@gluon.constexpr_function
def _cg_layout(num_warps):
    """``[rows, K quarter, pair]`` of polled pooled rows."""
    return gl.BlockedLayout([1, 1, 1], [1, 1, 64], [1, 1, num_warps], [2, 1, 0])


@gluon.jit
def ex1_poll(mb, task, task2, active, pre_gain, STAGE: gl.constexpr):
    """The inputs of ex1 task ``task`` and the experts of the slots: poll
    every whole pooled row (the norm needs every value), four rows a round
    trip, and every row's expert ids with the first rows (routing ends before
    the pooler, so the ids add no wait). Returns the polled rows (a tuple of
    groups), the task's pre-expert gains (low and high values of the pairs),
    every row's picks (a lane each, for :func:`slots`), then the experts of
    the first slots of ex1 task ``task`` and ex2 task ``task2`` (row 0's
    picks), for :func:`ex1_operand`.

    Every CTA calls it (Gluon cannot carry values out of a branch); an
    inactive CTA (``active`` false) stamps nothing.

    Args:
        pre_gain: FP32 ``bf16(w + 1)`` of the pre-expert norm ``[4096]``.
    """
    if active:
        mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    G: gl.constexpr = min(RP, 4)
    ks = task % KSPLIT
    nw: gl.constexpr = gl.num_warps()
    cl: gl.constexpr = _cg_layout(nw)
    gr = gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, cl)))[:, None, None]
    q = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, cl)))[None, :, None]
    j = gl.arange(0, K_SLICE // 2, gl.SliceLayout(0, gl.SliceLayout(1, cl)))
    c = ks * K_SLICE + 2 * j
    gain_lo = gl.load(pre_gain + c)
    gain_hi = gl.load(pre_gain + c + 1)
    pair = q * (K_SLICE // 2) + j[None, None, :]
    # Every row's 8 ids, a lane each, and the first slots' experts.
    NP: gl.constexpr = RP * TOP_K
    pk = gl.arange(0, NP, gl.SliceLayout(1, _pick_layout(NP, nw)))
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    i = gl.arange(0, 2, lanes)
    first = gl.where(i == 0, task // (NB1 * KSPLIT), task2 // (NB2 // EX2_BLOCKS))
    if R > 1:
        # Past one row every wave of many CTAs polls several whole rows:
        # one wave spins on hints first (unthrottled, the flood of polls
        # starves the producers' stores across ranks and the grid livelocks).
        mb.row(R - 1).pre_poll_spread("cg", POOLED // 2)
    got, firsts, ids = mb.poll_many(
        ("cg", "route", "route"),
        (mb.row_offset(gr) + pair, first, mb.row_offset(pk // TOP_K) + pk % TOP_K),
    )
    rows = (got,)
    for g in gl.static_range(1, RP // G):
        (more,) = mb.poll_many(("cg",), (mb.row_offset(g * G + gr) + pair,))
        rows += (more,)
    if active:
        mb.stamp(STAGE, trace.READY)
    e1 = gl.max(gl.where(i == 0, firsts, -1), axis=0)
    e2 = gl.max(gl.where(i == 1, firsts, -1), axis=0)
    return rows, gain_lo, gain_hi, ids, e1, e2


@gluon.jit
def ex1_operand(task, rows, gain_lo, gain_hi):
    """The A operand of ex1 task ``task`` from :func:`ex1_poll`'s rows and
    gains: apply each row's pre-expert RMS norm and quantize the task's K
    quarter of every row to MXFP4 in LDS (MFMA row ``r`` is row ``r``; the
    rows past the launch's are zero). Returns the LDS (values, scales).

    It issues no global load, and its LDS barriers wait only for LDS, so it
    runs while the weight loads that :func:`ex1_issue` issued before it are
    in flight. The normed rows go through LDS once, in
    BF16 (the quantizer's input precision), into the quantizer's layout: a
    register relayout a row costs two barriers of all the CTA's waves.
    """
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
    # Zero the rows past the launch's in power-of-two blocks: [RP, 2 RP),
    # [2 RP, 4 RP), ... up to M_DUP.
    zl: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, nw], [1, 0])
    RP: gl.constexpr = G * len(rows)
    for b in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << b) < M_DUP:
            x_smem.slice(RP << b, RP << b, dim=0).store(
                gl.zeros([RP << b, K_SLICE], gl.bfloat16, zl)
            )
    gl.barrier()
    dl: gl.constexpr = scaled_downcast_layout(M_DUP, K_SLICE, nw)
    packed, scales = _mxfp4_quantize_tile_in_layout(x_smem.load(dl))
    return _stage_a(packed, scales)


@gluon.jit
def ex1_issue(
    task, active, expert, w13, w13s, stride_we, stride_se, stride_slin, stride_snb
):
    """Issue ex1 task ``task``'s w13 loads (column block, K quarter) of
    expert ``expert``; nothing loads when ``active`` is false or ``expert``
    is -1.

    Args:
        w13, w13s: The rank's preprocessed w13 bytes ``[256, 2048, 1024]`` and
            scales (as ``decode_stage1`` takes them), with their strides.
    """
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    return _load_b(
        w13,
        w13s,
        expert,
        stride_we,
        stride_se,
        stride_slin,
        stride_snb,
        blk * BLOCK_N,
        ks * (K_SLICE // 128),
        K_SLICE // 128,
        W13_N,
        active,
    )


@gluon.jit
def _ex1_slot(mb, task, a, tiles, union, expert):
    """Multiply the A rows by one slot's w13 ``tiles`` and publish the FP32
    partial sums of the rows that picked ``expert``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    p_smem, s_smem = a
    acc = _mma_rows(p_smem, s_smem, tiles)
    picks = _row_picks(union, expert, R)
    m = gl.arange(0, M_DUP, gl.SliceLayout(1, acc.type.layout))
    k = gl.full([M_DUP], -1, gl.int32, m.type.layout)
    for r in gl.static_range(R):
        k = gl.where(m == r, picks[r], k)
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, acc.type.layout))[None, :]
    at = ((k[:, None] * NB1 + blk) * KSPLIT + ks) * BLOCK_N + n
    mb.put(
        "p1",
        mb.row_offset(m[:, None]) + at,
        acc.to(gl.int32, bitcast=True),
        mask=(k[:, None] >= 0) & (n >= 0),
    )


@gluon.jit
def ex1_finish(
    mb,
    task,
    a,
    tiles,
    expert,
    union,
    w13,
    w13s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    STAGE: gl.constexpr,
):
    """Ex1 task ``task`` after :func:`ex1_operand` (``a``) and :func:`ex1_issue`
    (``tiles``, of the lane's first slot, ``expert``) and :func:`slots`
    (``union``): multiply and publish the FP32 partial sums, then the same
    for the lane's other slots."""
    lane = task // (NB1 * KSPLIT)
    count = union[3]
    # Each slot's loads go out before the previous slot's product. With
    # one row the loop never runs, yet this form measured 2 us faster per
    # layer than one without it (a backend schedule effect). Past one row
    # it fits the registers since the slots' ids sit a lane each.
    following = _slot_expert(union, lane + LANES)
    more = ex1_issue(
        task,
        True,
        following,
        w13,
        w13s,
        stride_we,
        stride_se,
        stride_slin,
        stride_snb,
    )
    _ex1_slot(mb, task, a, tiles, union, expert)
    for i in range(lane + LANES, count, LANES):
        expert = following
        tiles = more
        following = _slot_expert(union, i + LANES)
        more = ex1_issue(
            task,
            True,
            following,
            w13,
            w13s,
            stride_we,
            stride_se,
            stride_slin,
            stride_snb,
        )
        _ex1_slot(mb, task, a, tiles, union, expert)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def ex2_issue(
    mb,
    task,
    active,
    expert,
    w2,
    w2s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    STAGE: gl.constexpr,
):
    """Issue ex2 task ``task``'s w2 loads (column block) of expert
    ``expert``; nothing loads (and nothing stamps) when ``active`` is false;
    nothing loads when ``expert`` is -1. Every CTA calls it (Gluon cannot
    carry the tiles out of a branch); :func:`ex2_finish` uses them.

    Args:
        w2, w2s: The rank's preprocessed w2 bytes ``[256, 256, 4096]`` and
            scales, with their strides.
    """
    if active:
        mb.stamp(STAGE, trace.START)
    return _ex2_loads(
        task, active, expert, w2, w2s, stride_we, stride_se, stride_slin, stride_snb
    )


@gluon.jit
def _ex2_loads(
    task, active, expert, w2, w2s, stride_we, stride_se, stride_slin, stride_snb
):
    """:func:`ex2_issue`'s loads, without the trace point."""
    TASKS_PER_LANE: gl.constexpr = NB2 // EX2_BLOCKS
    blk0 = (task % TASKS_PER_LANE) * EX2_BLOCKS
    loads = ()
    for i in gl.static_range(EX2_BLOCKS):
        loads += (
            _load_b(
                w2,
                w2s,
                expert,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
                (blk0 + i) * BLOCK_N,
                0,
                INTER // 128,
                POOLED,
                active,
            ),
        )
    return loads


@gluon.jit
def _slot_rows(mb, picks, RP: gl.constexpr, layout: gl.constexpr):
    """``(row, pick, routed)`` ``[RP]`` in ``layout``: each row that picked
    the slot's expert with its pick; the other rows (and the rows past the
    launch's) stand in for the first routed row, so a poll of them reads
    pairs that are published."""
    R: gl.constexpr = schedule.rows(mb.plan)
    i = gl.arange(0, RP, layout)
    pick = gl.full([RP], -1, gl.int32, layout)
    first = gl.to_tensor(-1)
    first_pick = gl.to_tensor(-1)
    for r in gl.static_range(R - 1, -1, -1):
        pick = gl.where(i == r, picks[r], pick)
        first = gl.where(picks[r] >= 0, r, first)
        first_pick = gl.where(picks[r] >= 0, picks[r], first_pick)
    routed = pick >= 0
    return gl.where(routed, i, first), gl.where(routed, pick, first_pick), routed


@gluon.jit
def exr_stage(mb, task, union, STAGE: gl.constexpr):
    """Exr task ``task`` (slot lane, ex1 column block): for each slot of the
    lane, sum the 4 K quarters of the ex1 partials of every row that picked
    the slot's expert, apply SwiGLU (``silu(gate) up``) and publish the
    block's 64 intermediate values a row. The 32 ex2 tasks of the lane then
    read 512 values a row, not 4 partials of 1024 columns each."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    lane = task // NB1
    blk = task % NB1
    count = union[3]
    # [row, K quarter, column]: the K quarters in a lane's registers.
    pl: gl.constexpr = gl.BlockedLayout(
        [1, KSPLIT, 2], [1, 1, 64], [nw, 1, 1], [2, 1, 0]
    )
    rl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, pl))
    ks = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, pl)))[None, :, None]
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, pl)))[None, None, :]
    hl: gl.constexpr = gl.SliceLayout(
        2, gl.BlockedLayout([1, 1, 2], [1, 64, 1], [nw, 1, 1], [2, 1, 0])
    )
    for i in range(lane, count, LANES):
        expert = _slot_expert(union, i)
        picks = _row_picks(union, expert, R)
        row, pick, _ = _slot_rows(mb, picks, RP, rl)
        at = ((pick[:, None, None] * NB1 + blk) * KSPLIT + ks) * BLOCK_N + n
        (got,) = mb.poll_many(("p1",), (mb.row_offset(row[:, None, None]) + at,))
        if i == lane:
            mb.stamp(STAGE, trace.READY)
        parts = got.to(gl.float32, bitcast=True)
        # The regular path's order: K quarter 0, then 1, ...
        acc = gl.zeros([RP, BLOCK_N], gl.float32, gl.SliceLayout(1, pl))
        for k in gl.static_range(KSPLIT):
            acc += gl.sum(gl.where(ks == k, parts, 0.0), axis=1)
        gate, up = gl.split(gl.reshape(acc, [RP, BLOCK_N // 2, 2]))
        h = gl.convert_layout(gate / (1.0 + gl.exp(-gate)) * up, hl)
        row, pick, routed = _slot_rows(mb, picks, RP, gl.SliceLayout(1, hl))
        j = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, hl))[None, :]
        mb.put(
            "h1",
            mb.row_offset(row[:, None])
            + pick[:, None] * INTER
            + blk * (BLOCK_N // 2)
            + j,
            h,
            mask=routed[:, None] & (j >= 0),
        )
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _ex2_gather(mb, union, expert):
    """The polls of one ex2 slot: the SwiGLU intermediates (:func:`exr_stage`)
    of the rows that picked ``expert``, MFMA row ``m`` the ``m``-th of them,
    in one poll a group of up to 8 rows. Returns ``(h16 [16, 512], picks,
    order)`` for :func:`_ex2_compute`."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    G: gl.constexpr = min(RP, 8)
    nw: gl.constexpr = gl.num_warps()
    picks = _row_picks(union, expert, R)
    # The MFMA row of each routed row: the routed rows before it.
    order = ()
    count = gl.to_tensor(0)
    for r in gl.static_range(R):
        order += (count,)
        count += (picks[r] >= 0).to(gl.int32)
    # MFMA row m's (row, pick): the routed row whose order is m. The others
    # poll MFMA row 0's pairs, then hold zeros.
    ql: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    ml: gl.constexpr = gl.SliceLayout(1, ql)
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    h_smem = gl.allocate_shared_memory(gl.float32, [M_DUP, INTER], plain)
    v = gl.arange(0, INTER, gl.SliceLayout(0, ql))[None, :]
    for g in gl.static_range(RP // G):
        m = g * G + gl.arange(0, G, ml)
        row = gl.full([G], -1, gl.int32, ml)
        pick = gl.full([G], -1, gl.int32, ml)
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
            ("h1",), (mb.row_offset(row[:, None]) + pick[:, None] * INTER + v,)
        )
        h = gl.where(routed[:, None], got.to(gl.float32, bitcast=True), 0.0)
        h_smem.slice(g * G, G, dim=0).store(h)
    for b in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << b) < M_DUP:
            h_smem.slice(RP << b, RP << b, dim=0).store(
                gl.zeros([RP << b, INTER], gl.float32, ql)
            )
    gl.barrier()
    return h_smem.load(ql), picks, order


@gluon.jit
def _ex2_compute(mb, task, loads, gathered, RANK: gl.constexpr):
    """One ex2 slot after :func:`_ex2_gather`: the MXFP4 intermediate times
    the slot's w2 ``loads``, the BF16 partials put to the owner rank."""
    R: gl.constexpr = schedule.rows(mb.plan)
    TASKS_PER_LANE: gl.constexpr = NB2 // EX2_BLOCKS
    blk0 = (task % TASKS_PER_LANE) * EX2_BLOCKS
    h16, picks, order = gathered
    packed, scales = _mxfp4_quantize_tile(h16)
    p_smem, s_smem = _stage_a(packed, scales)
    owner = (blk0 * BLOCK_N) // OWNED_OUT
    for b in gl.static_range(EX2_BLOCKS):
        acc = _mma_rows(p_smem, s_smem, loads[b])
        out = gl.reshape(acc.to(gl.bfloat16), [M_DUP, BLOCK_N // 2, 2])
        word = mx.pack_bf16_pairs(out)
        wl: gl.constexpr = word.type.layout
        mm = gl.arange(0, M_DUP, gl.SliceLayout(1, wl))[:, None]
        p = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, wl))[None, :]
        # The row and pick of each MFMA row.
        row_of = gl.full([M_DUP, 1], 0, gl.int32, wl)
        pick_of = gl.full([M_DUP, 1], -1, gl.int32, wl)
        for r in gl.static_range(R):
            mine = (mm == order[r]) & (picks[r] >= 0)
            row_of = gl.where(mine, r, row_of)
            pick_of = gl.where(mine, picks[r], pick_of)
        col = ((blk0 + b) * BLOCK_N) % OWNED_OUT
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
    mb,
    task,
    loads,
    expert,
    union,
    w2,
    w2s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    RANK: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Ex2 task ``task`` after :func:`ex2_issue` (``loads``, of the lane's
    first slot, ``expert``) and :func:`slots` (``union``): that slot, then the
    lane's other slots."""
    lane = task // (NB2 // EX2_BLOCKS)
    count = union[3]
    # A slot's polls, then the next slot's loads (a poll waits for every
    # earlier load), then the slot's product while they stream.
    gathered = _ex2_gather(mb, union, expert)
    mb.stamp(STAGE, trace.READY)
    following = _slot_expert(union, lane + LANES)
    more = _ex2_loads(
        task, True, following, w2, w2s, stride_we, stride_se, stride_slin, stride_snb
    )
    _ex2_compute(mb, task, loads, gathered, RANK)
    for i in range(lane + LANES, count, LANES):
        loads = more
        gathered = _ex2_gather(mb, union, following)
        following = _slot_expert(union, i + LANES)
        more = _ex2_loads(
            task,
            True,
            following,
            w2,
            w2s,
            stride_we,
            stride_se,
            stride_slin,
            stride_snb,
        )
        _ex2_compute(mb, task, loads, gathered, RANK)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)
