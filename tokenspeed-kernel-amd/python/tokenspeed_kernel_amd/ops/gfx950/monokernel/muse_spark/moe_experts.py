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

* xq, a task a K quarter of a group of up to 4 rows: polls the group's
  whole pooled rows, applies the pre-expert RMS norm, and quantizes the
  quarter to MXFP4 (the regular path's "even" rule), then publishes the
  E2M1 bytes and E8M0 scales as plain words under one flag.
* ex1, 256 tasks: lane ``l``, 128 of the 1024 interleaved gate / up columns,
  a quarter of K (64 KiB of weights a slot). A task issues its first slot's
  weights once routing names the experts, then loads its K quarter of every
  row from xq into LDS, one MFMA row each (136 words a row, not every whole
  pooled row on each of the 256 CTAs). It publishes the FP32 partial sums of
  the rows that picked the slot's expert. The layers with BF16 experts
  (moe_experts_bf16) run no xq: their ex1 polls and norms the pooled rows.
* exr, 64 tasks at one row (lane ``l``, one ex1 column block), 256 past one
  row (each lane's slots split over EXR_PARTS tasks, one a CTA). For each of
  its slots, a task polls the 4 K quarters of every row that picked the
  slot's expert, sums them in K order, applies SwiGLU (``silu(gate) up``)
  and the regular path's MXFP4 requantization (groups of 32: the block's 64
  values are two whole groups), and publishes them as words (``h1``). Done
  once here, an ex2 task reads 72 words a row instead of 4096 partials and
  quantizes nothing: polled by all 256 CTAs at once, the partials took one
  0.7 us round trip a row (the memory-side bandwidth, not the latency), so a
  slot that many rows picked held its lane up, and each of a lane's 32 ex2
  tasks requantized the same rows (1.3 us a slot).
* ex2, 256 tasks: lane ``l``, 128 of the 4096 output columns over the rank's
  512 intermediate values (32 KiB a slot). A task polls the MXFP4
  intermediates of its lane's first 8 slots at once (every row of each)
  into LDS, its first 4 slots' weights issued before that poll, then runs
  those slots with each slot's weights 4 slots ahead and no poll between
  them. Past one row the gather also keeps each slot's row picks in LDS, so
  a slot reads them in one load rather than a reduction a row. The BF16
  partials go to the rank that owns those columns (``rr``), which sums the
  8 ranks' partials.

The slot table (:func:`slots`: each pick's slot, whether it owns its
expert, every row's pick of it) is built once, by one task after the
routing (:func:`slot_table_stage`), and published in ``slt``; built on all
256 CTAs it took ~5 us of each ex1 task's prelude.

Expert weights cannot load before routing picks the experts. So a CTA runs
the first slot of its ex1 and ex2 tasks in parts: :func:`ex1_poll_routes`
(the first slots' experts and the slot table), :func:`ex1_issue`, then
:func:`union_of` and :func:`ex1_xq` (the A operand from xq, while the
weights stream), :func:`ex1_finish`, then :func:`ex2_issue` and
:func:`ex2_finish` at ex2's place in the plan (the BF16 experts:
:func:`ex1_poll` with the pooled rows, then their own operand). ex2's first weights issue
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
# Past one row each lane's slots split over EXR_PARTS exr tasks (a slot of
# every EXR_PARTS-th round each), one task a CTA: an exr task then holds its
# CTA's ex2 task back by about one slot, whatever the slot count. On CTAs
# 192-255 alone, the lanes there ran a round of exr, and when they also had
# the most slots (slots % 8 = 7 or 0, a fifth of served launches) every ex2
# lane waited ~6 us longer. One row has one slot a lane: one part. A launch
# that ends with qnext splits exr in 2 parts instead (layer.NEXT_BASES).
EXR_PARTS = gl.constexpr(4)
EXR_LANE_TASKS = gl.constexpr(64)  # LANES * NB1: one part
# An intermediate row in MXFP4 words: 64 of E2M1 values, then a word per
# ex1 block holding its 2 scale bytes.
H1_WORDS = gl.constexpr(72)  # INTER // 8 + NB1
# A BF16 intermediate row (moe_experts_bf16): 512 values, two a word.
H1_BF16_WORDS = gl.constexpr(256)
EX2_BLOCKS = gl.constexpr(1)  # 128-column blocks of an ex2 task
EX2_TASKS = gl.constexpr(256)  # LANES * NB2 // EX2_BLOCKS
# An ex2 task's first slots (all of them up to 64 distinct experts) are
# gathered in one poll and run unrolled (:func:`ex2_finish`).
EX2_GATHER = gl.constexpr(8)
# Slots whose weights an ex2 task issues before that gather (then one more
# a slot). The gather waits for them anyway (vmcnt counts in order), and it
# waits on exr too, so up to 4 slots a lane (32 distinct experts) stream
# while exr finishes instead of after the gather. 3 gained about half as much.
EX2_AHEAD = gl.constexpr(4)
OWNED_OUT = gl.constexpr(512)  # output columns a rank reduces: POOLED / WORLD
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("rr",)
# The slot table (:func:`slot_table_stage`), plain words in row 0 under one
# flag (sltf): a word a pick (its id, slot, whether it owns its expert, the
# slot count), then each pick's row picks ``[pick, row]``. Sized for 16
# rows. As tagged pairs, all 256 ex1 CTAs polling one CTA's 2176 pairs took
# 9-14 us past the table's end.
SLT_PICKS = gl.constexpr(128)
SLT_WORDS = gl.constexpr(128 + 128 * 16)
# The expert input (:func:`xq_stage`): each row's K quarters normed and
# quantized to MXFP4 once, as plain words in the row's copy of xq: a
# quarter's 512 E2M1 bytes (128 words), then its 32 E8M0 scales (8 words).
# An xq task does one K quarter of a group of up to XQ_ROWS rows (one poll
# round trip) and raises flag ``quarter * XQ_GROUPS + group`` in row 0 of
# xqf. ex1 then loads 136 words a row instead of polling every whole pooled
# row (16 KiB of pairs) on each of its 256 CTAs.
XQ_QUARTER_WORDS = gl.constexpr(136)  # K_SLICE // 8 + K_SLICE // 128
XQ_ROWS = gl.constexpr(4)
XQ_GROUPS = gl.constexpr(4)  # 16 rows / XQ_ROWS


def regions() -> tuple[tuple[str, int], ...]:
    k, nb1, ks, bn = int(TOP_K), int(NB1), int(KSPLIT), int(BLOCK_N)
    return (
        ("slt", int(SLT_WORDS) // 2),  # words (half the region)
        ("sltf", 1),
        ("xq", int(KSPLIT) * int(XQ_QUARTER_WORDS) // 2),  # words
        ("xqf", int(KSPLIT) * int(XQ_GROUPS)),
        ("p1", k * nb1 * ks * bn),  # FP32 [pick][block][K quarter][column] a row
        # The intermediate [pick][word] a row: MXFP4, or BF16 for the
        # layers with BF16 experts (one layout serves both).
        ("h1", k * max(int(H1_WORDS), int(H1_BF16_WORDS))),
        (
            "rr",
            int(WORLD) * k * int(OWNED_OUT) // 2,
        ),  # BF16 pairs [source][pick][column] a row
    )


def xq_tasks(rows: int) -> int:
    """The xq tasks of a launch of ``rows``: a K quarter of a row group each."""
    rp = 1 << (rows - 1).bit_length()
    return int(KSPLIT) * (rp // min(rp, int(XQ_ROWS)))


def stages(
    bases: dict[str, int], rows: int, exr_parts: int, bf16_experts: bool
) -> tuple[Stage, ...]:
    """slt, xq, ex1, exr and ex2 of ``rows`` token rows, placed by ``bases``;
    each lane's slots split over ``exr_parts`` exr tasks (1, 2, or EXR_PARTS
    to spread exr over every CTA). With ``bf16_experts`` (moe_experts_bf16)
    there is no xq: ex1 polls and norms the pooled rows itself."""
    if exr_parts not in (1, 2, int(EXR_PARTS)):
        raise ValueError(f"exr parts: 1, 2 or {int(EXR_PARTS)}, got {exr_parts}")
    slt, xq, ex1, exr, ex2 = placement(bases, ("slt", "xq", "ex1", "exr", "ex2"))
    if bf16_experts:
        head = (
            Stage(
                "ex1",
                int(EX1_TASKS),
                ex1,
                reads=("slt", "sltf", "route", "cg"),
                writes=("p1",),
            ),
        )
    else:
        head = (
            Stage("xq", xq_tasks(rows), xq, reads=("cg",), writes=("xq", "xqf")),
            Stage(
                "ex1",
                int(EX1_TASKS),
                ex1,
                reads=("slt", "sltf", "xq", "xqf", "route"),
                writes=("p1",),
            ),
        )
    return (
        (Stage("slt", 1, slt, reads=("route",), writes=("slt", "sltf")),)
        + head
        + (
            Stage(
                "exr",
                int(EXR_LANE_TASKS) * exr_parts,
                exr,
                reads=("route", "p1"),
                writes=("h1",),
            ),
            Stage("ex2", int(EX2_TASKS), ex2, reads=("route", "h1"), writes=("rr",)),
        )
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
def slot_table_stage(mb, STAGE: gl.constexpr):
    """The launch's slots (:func:`slots`), once: every row's picks, then
    each pick's word (id, slot, owner, slot count) and row picks as plain
    words in ``slt`` (row 0) under one flag (``sltf``). ex1 polls the flag
    with the pooled rows, then loads the words, instead of building the
    table on every CTA (:func:`union_of`)."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NP: gl.constexpr = RP * TOP_K
    nw: gl.constexpr = gl.num_warps()
    if NP <= 64:
        # A pick a lane of every wave: the slots by wave ballots.
        pk = gl.arange(0, NP, gl.BlockedLayout([1], [64], [nw], [0]))
        (ids,) = mb.poll_many(
            ("route",), (mb.row_offset(gl.minimum(pk // TOP_K, R - 1)) + pk % TOP_K,)
        )
        mb.stamp(STAGE, trace.READY)
        ids, owner, slot, count, table = _ballot_slots(ids, RP)
    elif NP == 128:
        # Two picks a lane (2 l and 2 l + 1, one row) of every wave.
        pk = gl.arange(0, NP, gl.BlockedLayout([2], [64], [nw], [0]))
        (ids,) = mb.poll_many(
            ("route",), (mb.row_offset(gl.minimum(pk // TOP_K, R - 1)) + pk % TOP_K,)
        )
        mb.stamp(STAGE, trace.READY)
        ids, owner, slot, count, table = _ballot_slots2(ids, RP)
    else:
        pk = gl.arange(0, NP, gl.SliceLayout(1, _pick_layout(NP, nw)))
        (ids,) = mb.poll_many(
            ("route",), (mb.row_offset(gl.minimum(pk // TOP_K, R - 1)) + pk % TOP_K,)
        )
        mb.stamp(STAGE, trace.READY)
        ids, owner, slot, count, table = slots(ids)
    lanes: gl.constexpr = ids.type.layout
    p = gl.arange(0, NP, lanes)
    word = ids | (slot << 8) | (owner << 15) | (count << 16)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.store_words("slt", p, word)
    tl: gl.constexpr = table.type.layout
    tp = gl.arange(0, NP, gl.SliceLayout(1, tl))[:, None]
    tr = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    mb.store_words("slt", SLT_PICKS + tp * RP + tr, table)
    mb.raise_flag("sltf", 0)
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def _table_layout(NP, RP, num_warps):
    """``[pick, row]`` of the polled table: a pick's row a thread."""
    return gl.BlockedLayout([1, RP], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def union_of(polled):
    """:func:`slots`'s result from the polled slot table (``ex1_poll``)."""
    words, table = polled
    ids = words & 0xFF
    slot = (words >> 8) & 0x7F
    owner = (words >> 15) & 1
    count = gl.max(words >> 16, axis=0)
    return ids, owner, slot, count, table


@gluon.constexpr_function
def _slots_asm(outputs, inputs):
    """:func:`slots` of up to 64 picks (rows of 8) by wave ballots: lane
    ``p`` holds pick ``p``'s id (``NP = 8 len(table)`` picks; lanes past
    them repeat them). For each pick ``q`` (v_readlane), every lane compares
    its id: a hit gives the lane's row-``q // 8`` pick (``q % 8``) and, for a
    lane of a later row, marks its expert as seen in an earlier row. A pick
    owns its expert unless seen; its slot is the owner picks below it
    (v_mbcnt of the owners' ballot), the count all owner picks."""
    (owner,), (slot,), (count,), table = outputs[0:4]
    lane, row, seen, hit, t = (o[0] for o in outputs[4:9])
    (sq,) = outputs[9]
    ((ident,),) = inputs
    RP = len(table)
    NP = RP * 8
    a = [
        "s_nop 4",
        f"v_mbcnt_lo_u32_b32 {lane}, -1, 0",
        f"v_mbcnt_hi_u32_b32 {lane}, -1, {lane}",
        # Past NP lanes repeat the picks: lane l holds pick l % NP.
        f"v_and_b32 {lane}, {NP - 1}, {lane}",
        f"v_lshrrev_b32 {row}, 3, {lane}",
        f"v_mov_b32 {seen}, 0",
    ]
    a += [f"v_mov_b32 {x}, -1" for x in table]
    for r in range(RP):
        a += [f"v_mov_b32 {hit}, 0"]
        for k in range(8):
            a += [
                f"v_readlane_b32 {sq}, {ident}, {r * 8 + k}",
                f"v_cmp_eq_u32_e32 vcc, {sq}, {ident}",
                f"v_cndmask_b32_e64 {table[r]}, {table[r]}, {k}, vcc",
                f"v_cndmask_b32_e64 {hit}, {hit}, 1, vcc",
            ]
        # A lane of a later row saw row r's experts.
        a += [
            f"v_cmp_lt_u32_e32 vcc, {r}, {row}",
            f"v_or_b32 {t}, {seen}, {hit}",
            f"v_cndmask_b32_e32 {seen}, {seen}, {t}, vcc",
        ]
    a += [
        f"v_sub_u32 {owner}, 1, {seen}",
        # The owner picks' ballot, of the first NP lanes (the rest repeat).
        f"v_cmp_ne_u32_e32 vcc, 0, {owner}",
    ]
    if NP == 64:
        a += [
            "s_nop 4",
            f"s_bcnt1_i32_b64 {count}, vcc",
            f"v_mbcnt_lo_u32_b32 {slot}, vcc_lo, 0",
            f"v_mbcnt_hi_u32_b32 {slot}, vcc_hi, {slot}",
        ]
    else:
        # A slot counts the owners below its pick: the ballot's bits below
        # l % NP (a mask of the first NP <= 32 lanes).
        a += [
            "s_nop 4",
            f"s_and_b32 vcc_lo, vcc_lo, {(1 << NP) - 1:#x}",
            f"s_bcnt1_i32_b32 {count}, vcc_lo",
            f"v_bfm_b32 {t}, {lane}, 0",
            f"v_and_b32 {t}, vcc_lo, {t}",
            f"v_bcnt_u32_b32 {slot}, {t}, 0",
        ]
    return "\n".join(a)


@gluon.constexpr_function
def _slots2_asm(outputs, inputs):
    """:func:`_slots_asm` of 128 picks (16 rows of 8), two a lane: lane
    ``l`` holds picks ``2 l`` and ``2 l + 1`` (both of row ``l // 4``). Each
    pick ``q`` (v_readlane of its lane's pick) is compared with both of every
    lane's picks. A pick's slot is the owner picks below it: those of the
    lanes below (both ballots), and for ``2 l + 1`` also ``2 l``."""
    owner, slot = outputs[0], outputs[1]
    (count,) = outputs[2]
    table = outputs[3]
    lane, row = outputs[4][0], outputs[5][0]
    seen, hit, t = outputs[6], outputs[7], outputs[8]
    (sq,) = outputs[9]
    (ident,) = inputs
    RP = len(table) // 2
    a = [
        "s_nop 4",
        f"v_mbcnt_lo_u32_b32 {lane}, -1, 0",
        f"v_mbcnt_hi_u32_b32 {lane}, -1, {lane}",
        f"v_lshrrev_b32 {row}, 2, {lane}",
        f"v_mov_b32 {seen[0]}, 0",
        f"v_mov_b32 {seen[1]}, 0",
    ]
    a += [f"v_mov_b32 {x}, -1" for x in table]
    for r in range(RP):
        a += [f"v_mov_b32 {hit[0]}, 0", f"v_mov_b32 {hit[1]}, 0"]
        for k in range(8):
            q = r * 8 + k
            a += [f"v_readlane_b32 {sq}, {ident[q % 2]}, {q // 2}"]
            for e in range(2):
                a += [
                    f"v_cmp_eq_u32_e32 vcc, {sq}, {ident[e]}",
                    f"v_cndmask_b32_e64 {table[e * RP + r]}, {table[e * RP + r]}, {k}, vcc",
                    f"v_cndmask_b32_e64 {hit[e]}, {hit[e]}, 1, vcc",
                ]
        # A lane of a later row saw row r's experts.
        a += [f"v_cmp_lt_u32_e32 vcc, {r}, {row}"]
        for e in range(2):
            a += [
                f"v_or_b32 {t[e]}, {seen[e]}, {hit[e]}",
                f"v_cndmask_b32_e32 {seen[e]}, {seen[e]}, {t[e]}, vcc",
            ]
    a += [
        f"v_sub_u32 {owner[0]}, 1, {seen[0]}",
        f"v_sub_u32 {owner[1]}, 1, {seen[1]}",
        # Even picks' owners: their count and the ones below each lane.
        f"v_cmp_ne_u32_e32 vcc, 0, {owner[0]}",
        "s_nop 4",
        f"s_bcnt1_i32_b64 {count}, vcc",
        f"v_mbcnt_lo_u32_b32 {t[0]}, vcc_lo, 0",
        f"v_mbcnt_hi_u32_b32 {t[0]}, vcc_hi, {t[0]}",
        # Odd picks' owners: added to both.
        f"v_cmp_ne_u32_e32 vcc, 0, {owner[1]}",
        "s_nop 4",
        f"s_bcnt1_i32_b64 {sq}, vcc",
        f"s_add_u32 {count}, {count}, {sq}",
        f"v_mbcnt_lo_u32_b32 {t[0]}, vcc_lo, {t[0]}",
        f"v_mbcnt_hi_u32_b32 {slot[0]}, vcc_hi, {t[0]}",
        f"v_add_u32 {slot[1]}, {slot[0]}, {owner[0]}",
    ]
    return "\n".join(a)


@gluon.constexpr_function
def _slots2_constraints(RP):
    """:func:`_slots2_asm`'s constraints as one string: a tuple repeats an
    entry for every element of its operand, and with two picks a lane the
    input's entry would repeat its clobbers before the second pick."""
    two = ["=&v", "=&v"]
    outputs = two * 2 + ["=&s"] + ["=&v"] * (2 * RP) + two * 5 + ["=&s"]
    return ",".join(outputs + ["v", "v", "~{vcc}", "~{scc}"])


@gluon.jit
def _ballot_slots2(ids, RP: gl.constexpr):
    """:func:`slots` by :func:`_slots2_asm`: ``ids`` two a lane (128
    picks). Returns the same ``(ids, owner, slot, count, table)``."""
    NP: gl.constexpr = ids.shape[0]
    nw: gl.constexpr = gl.num_warps()
    tt: gl.constexpr = gl.distributed_type(
        gl.int32, [NP, RP], gl.BlockedLayout([2, RP], [64, 1], [nw, 1], [1, 0])
    )
    out = gl.inline_asm(
        _slots2_asm,
        _slots2_constraints(RP),
        [ids],
        [
            ids.type,
            ids.type,
            gl.int32,
            tt,
            ids.type,
            ids.type,
            ids.type,
            ids.type,
            ids.type,
            gl.int32,
        ],
        is_pure=True,
    )
    return ids, out[0], out[1], out[2], out[3]


@gluon.constexpr_function
def _table_lanes_layout(RP, num_warps):
    """``[pick, row]`` with a pick's row picks in its lane's registers."""
    return gl.BlockedLayout([1, RP], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def _ballot_slots(ids, RP: gl.constexpr):
    """:func:`slots` by :func:`_slots_asm`: ``ids`` a lane each (up to 64
    picks). Returns the same ``(ids, owner, slot, count, table)``."""
    NP: gl.constexpr = ids.shape[0]
    nw: gl.constexpr = gl.num_warps()
    lanes: gl.constexpr = ids.type.layout
    tt: gl.constexpr = gl.distributed_type(
        gl.int32, [NP, RP], _table_lanes_layout(RP, nw)
    )
    out = gl.inline_asm(
        _slots_asm,
        (
            "=&v",
            "=&v",
            "=&s",
            "=&v",
            "=&v",
            "=&v",
            "=&v",
            "=&v",
            "=&v",
            "=&s",
            "v,~{vcc},~{scc}",
        ),
        [ids],
        [
            ids.type,
            ids.type,
            gl.int32,
            tt,
            ids.type,
            ids.type,
            ids.type,
            ids.type,
            ids.type,
            gl.int32,
        ],
        is_pure=True,
    )
    return ids, out[0], out[1], out[2], out[3]


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
            # sc0 nt on gfx950: a slot's weights stream once a launch, so
            # they must not evict what the launch reuses from L2 and the
            # memory-side cache (its code, the KV, the warmed next-layer
            # QKVG). The scales stay default: small, and read again.
            cache=".cg",
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


@gluon.constexpr_function
def _cg_layout(num_warps):
    """``[rows, K quarter, pair]`` of polled pooled rows."""
    return gl.BlockedLayout([1, 1, 1], [1, 1, 64], [1, 1, num_warps], [2, 1, 0])


@gluon.jit
def ex1_poll(mb, task, task2, active, pre_gain, STAGE: gl.constexpr):
    """The inputs of ex1 task ``task`` and the experts of the slots: poll
    every whole pooled row (the norm needs every value), four rows a round
    trip, and the slot table's flag (:func:`slot_table_stage`) with the first
    rows, then the table's words.
    Returns the polled rows (a tuple of groups), the task's pre-expert gains
    (low and high values of the pairs), the polled slot table (for
    :func:`union_of`), then the experts of
    the first slots of ex1 task ``task`` and ex2 task ``task2`` (row 0's
    picks), for the BF16 experts' ``ex1_operand``.

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
    # The slot table's words (a lane a pick) and row picks, and the first
    # slots' experts.
    NP: gl.constexpr = RP * TOP_K
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    pk = gl.arange(0, NP, lanes)
    tl: gl.constexpr = _table_layout(NP, RP, nw)
    tp = gl.arange(0, NP, gl.SliceLayout(1, tl))[:, None]
    tr = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    i = gl.arange(0, 2, lanes)
    first = gl.where(i == 0, task // (NB1 * KSPLIT), task2 // (NB2 // EX2_BLOCKS))
    if R > 1:
        # Past one row every wave of many CTAs polls several whole rows:
        # one wave spins on hints first (unthrottled, the flood of polls
        # starves the producers' stores across ranks and the grid livelocks).
        mb.row(R - 1).pre_poll_spread("cg", POOLED // 2)
    got, firsts, _ = mb.poll_many(
        ("cg", "route", "sltf"), (mb.row_offset(gr) + pair, first, i * 0)
    )
    # The table's flag is in: its words load after the poll returned.
    gl.barrier()
    ids = (mb.load_words("slt", pk), mb.load_words("slt", SLT_PICKS + tp * RP + tr))
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
def _xq_quantize(got, ks, gain_lo, gain_hi):
    """K quarter ``ks`` of the polled pooled rows ``got`` (``[G, KSPLIT,
    K_SLICE / 2]`` BF16 pairs, :func:`_cg_layout`) with each row's
    pre-expert RMS norm, quantized to MXFP4 as the regular path does (BF16
    input, groups of 32 along K). Returns the words: E2M1 ``[G, K_SLICE /
    8]`` and E8M0 scales ``[G, K_SLICE / 128]``.

    Only the G rows quantize: groups are per row, so the bytes are the ones
    of a 16-row MFMA tile (which took 2 us of xq's ~3, at 4 rows)."""
    G: gl.constexpr = got.shape[0]
    nw: gl.constexpr = gl.num_warps()
    cl: gl.constexpr = _cg_layout(nw)
    q = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, cl)))[None, :, None]
    lo = (got << 16).to(gl.float32, bitcast=True)
    hi = (got & -65536).to(gl.float32, bitcast=True)
    inv = gl.rsqrt(
        gl.sum(gl.sum(lo * lo + hi * hi, axis=1), axis=1) / POOLED + PRE_EPS
    )[:, None]
    mine = q == ks
    lo = gl.sum(gl.where(mine, lo, 0.0), axis=1) * inv * gain_lo[None, :]
    hi = gl.sum(gl.where(mine, hi, 0.0), axis=1) * inv * gain_hi[None, :]
    x = gl.reshape(gl.join(lo, hi), [G, K_SLICE]).to(gl.bfloat16)
    dl: gl.constexpr = scaled_downcast_layout(G, K_SLICE, nw)
    packed, scales = _mxfp4_quantize_tile_in_layout(gl.convert_layout(x, dl))
    # A word takes 4 consecutive bytes in a thread; the scales come one a
    # thread.
    sl: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [nw, 1], [1, 0])
    return _words(packed), _words(gl.convert_layout(scales, sl))


@gluon.constexpr_function
def _words_layout(cols, num_warps):
    """``[row, word]`` of an xq quarter's words (a word a thread)."""
    lanes = min(64, cols)
    return gl.BlockedLayout([1, 1], [64 // lanes, lanes], [num_warps, 1], [1, 0])


@gluon.jit
def xq_stage(mb, task, pre_gain, STAGE: gl.constexpr):
    """xq task ``task``: K quarter ``task % KSPLIT`` of the rows of group
    ``task // KSPLIT`` (``G = min(RP, XQ_ROWS)`` rows), normed and quantized
    by :func:`_xq_quantize`, published as plain words in each row's copy of
    xq under one flag (xqf)."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    G: gl.constexpr = min(RP, XQ_ROWS)
    ks = task % KSPLIT
    grp = task // KSPLIT
    nw: gl.constexpr = gl.num_warps()
    cl: gl.constexpr = _cg_layout(nw)
    gr = gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, cl)))[:, None, None]
    q = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, cl)))[None, :, None]
    j = gl.arange(0, K_SLICE // 2, gl.SliceLayout(0, gl.SliceLayout(1, cl)))
    c = ks * K_SLICE + 2 * j
    gain_lo = gl.load(pre_gain + c)
    gain_hi = gl.load(pre_gain + c + 1)
    pair = q * (K_SLICE // 2) + j[None, None, :]
    (got,) = mb.poll_many(("cg",), (mb.row_offset(grp * G + gr) + pair,))
    mb.stamp(STAGE, trace.READY)
    words, scales = _xq_quantize(got, ks, gain_lo, gain_hi)
    mb.stamp(STAGE, trace.COMPUTED)
    wl: gl.constexpr = words.type.layout
    pr = grp * G + gl.arange(0, G, gl.SliceLayout(1, wl))[:, None]
    pw = gl.arange(0, K_SLICE // 8, gl.SliceLayout(0, wl))[None, :]
    mb.store_words(
        "xq",
        2 * mb.row_offset(pr) + ks * XQ_QUARTER_WORDS + pw,
        words,
        mask=(pr < R) & (pw >= 0),
    )
    sl: gl.constexpr = scales.type.layout
    sr = grp * G + gl.arange(0, G, gl.SliceLayout(1, sl))[:, None]
    sw = gl.arange(0, K_SLICE // 128, gl.SliceLayout(0, sl))[None, :]
    mb.store_words(
        "xq",
        2 * mb.row_offset(sr) + ks * XQ_QUARTER_WORDS + K_SLICE // 8 + sw,
        scales,
        mask=(sr < R) & (sw >= 0),
    )
    mb.raise_flag("xqf", ks * XQ_GROUPS + grp)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def ex1_poll_routes(mb, task, task2, active, STAGE: gl.constexpr):
    """:func:`ex1_poll` without the pooled rows (their quantized words come
    from xq, :func:`ex1_xq`): the slot table's flag with the experts of the
    first slots of ex1 task ``task`` and ex2 task ``task2``, then the table's
    words. Returns ``(ids, e1, e2)``; the weights can stream from here."""
    if active:
        mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    NP: gl.constexpr = RP * TOP_K
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    pk = gl.arange(0, NP, lanes)
    tl: gl.constexpr = _table_layout(NP, RP, nw)
    tp = gl.arange(0, NP, gl.SliceLayout(1, tl))[:, None]
    tr = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    i = gl.arange(0, 2, lanes)
    first = gl.where(i == 0, task // (NB1 * KSPLIT), task2 // (NB2 // EX2_BLOCKS))
    firsts, _ = mb.poll_many(("route", "sltf"), (first, i * 0))
    # The table's flag is in: its words load after the poll returned.
    gl.barrier()
    ids = (mb.load_words("slt", pk), mb.load_words("slt", SLT_PICKS + tp * RP + tr))
    e1 = gl.max(gl.where(i == 0, firsts, -1), axis=0)
    e2 = gl.max(gl.where(i == 1, firsts, -1), axis=0)
    return ids, e1, e2


@gluon.jit
def ex1_xq(mb, task, active, STAGE: gl.constexpr):
    """ex1 task ``task``'s A operand from xq: poll its K quarter's flags (one
    a row group), then load the quarter's words of every row into LDS, one
    MFMA row each (rows past the launch's rows repeat its last row, rows
    past ``RP`` are zero). Returns the LDS (values, scales)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NG: gl.constexpr = RP // min(RP, XQ_ROWS)
    ks = task % KSPLIT
    nw: gl.constexpr = gl.num_warps()
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    f = gl.arange(0, XQ_GROUPS, lanes)
    mb.poll_many(("xqf",), (ks * XQ_GROUPS + gl.minimum(f, NG - 1),))
    gl.barrier()
    if active:
        mb.stamp(STAGE, trace.READY)
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    PW: gl.constexpr = K_SLICE // 8
    SW: gl.constexpr = K_SLICE // 128
    pl: gl.constexpr = _words_layout(PW, nw)
    sl: gl.constexpr = _words_layout(SW, nw)
    pr = gl.arange(0, M_DUP, gl.SliceLayout(1, pl))[:, None]
    pw = gl.arange(0, PW, gl.SliceLayout(0, pl))[None, :]
    sr = gl.arange(0, M_DUP, gl.SliceLayout(1, sl))[:, None]
    sw = gl.arange(0, SW, gl.SliceLayout(0, sl))[None, :]
    base = ks * XQ_QUARTER_WORDS
    words = mb.load_words("xq", 2 * mb.row_offset(pr) + base + pw)
    scales = mb.load_words("xq", 2 * mb.row_offset(sr) + base + PW + sw)
    p_smem = gl.allocate_shared_memory(gl.uint8, [M_DUP, K_SLICE // 2], plain)
    s_smem = gl.allocate_shared_memory(gl.uint8, [M_DUP, K_SLICE // 32], plain)
    p_smem.reinterpret(gl.int32, [M_DUP, PW], plain).store(gl.where(pr < RP, words, 0))
    s_smem.reinterpret(gl.int32, [M_DUP, SW], plain).store(gl.where(sr < RP, scales, 0))
    gl.barrier()
    return p_smem, s_smem


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
    """Ex1 task ``task`` after :func:`ex1_xq` (``a``) and :func:`ex1_issue`
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
def _words(x):
    """Bytes ``[M, 4 W]`` as little-endian int32 words ``[M, W]``."""
    M: gl.constexpr = x.shape[0]
    W: gl.constexpr = x.shape[1] // 4
    lo, hi = gl.split(gl.reshape(x.to(gl.int32), [M, W, 2, 2]))
    half = lo | (hi << 8)
    lo, hi = gl.split(half)
    return lo | (hi << 16)


@gluon.jit
def exr_stage(mb, task, union, BF16: gl.constexpr, STAGE: gl.constexpr):
    """Exr task ``task`` (slot lane, ex1 column block): for each slot of the
    lane, sum the 4 K quarters of the ex1 partials of every row that picked
    the slot's expert, apply SwiGLU (``silu(gate) up``) and publish the
    block's 64 intermediate values a row. The 32 ex2 tasks of the lane then
    read 512 values a row, not 4 partials of 1024 columns each. With
    ``BF16`` (BF16 experts) the values are rounded to BF16, as the regular
    BF16 route path stores its intermediate, else requantized to MXFP4."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    PARTS: gl.constexpr = schedule.stage_tasks(mb.plan, STAGE) // EXR_LANE_TASKS
    lane = task // (NB1 * PARTS)
    blk = task // PARTS % NB1
    first = lane + LANES * (task % PARTS)
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
    for i in range(first, count, LANES * PARTS):
        expert = _slot_expert(union, i)
        picks = _row_picks(union, expert, R)
        row, pick, _ = _slot_rows(mb, picks, RP, rl)
        at = ((pick[:, None, None] * NB1 + blk) * KSPLIT + ks) * BLOCK_N + n
        (got,) = mb.poll_many(("p1",), (mb.row_offset(row[:, None, None]) + at,))
        if i == first:
            mb.stamp(STAGE, trace.READY)
        parts = got.to(gl.float32, bitcast=True)
        # The regular path's order: K quarter 0, then 1, ...
        acc = gl.zeros([RP, BLOCK_N], gl.float32, gl.SliceLayout(1, pl))
        for k in gl.static_range(KSPLIT):
            acc += gl.sum(gl.where(ks == k, parts, 0.0), axis=1)
        gate, up = gl.split(gl.reshape(acc, [RP, BLOCK_N // 2, 2]))
        inter = gate / (1.0 + gl.exp(-gate)) * up
        if BF16:
            # A pair in one thread, for the split.
            pairs: gl.constexpr = gl.BlockedLayout(
                [1, 1, 2], [1, 64, 1], [nw, 1, 1], [2, 1, 0]
            )
            words = mx.pack_bf16_pairs(
                gl.convert_layout(
                    gl.reshape(inter.to(gl.bfloat16), [RP, BLOCK_N // 4, 2]), pairs
                )
            )
            wl: gl.constexpr = words.type.layout
            row, pick, routed = _slot_rows(mb, picks, RP, gl.SliceLayout(1, wl))
            w = gl.arange(0, BLOCK_N // 4, gl.SliceLayout(0, wl))[None, :]
            mb.put(
                "h1",
                mb.row_offset(row[:, None])
                + pick[:, None] * H1_BF16_WORDS
                + blk * (BLOCK_N // 4)
                + w,
                words,
                mask=routed[:, None] & (w >= 0),
            )
        else:
            # The regular path's requantization: groups of 32 values, and the
            # block's 64 values are two whole groups.
            packed, scales = _mxfp4_quantize_tile(inter)
            words = _words(packed)
            wl: gl.constexpr = words.type.layout
            row, pick, routed = _slot_rows(mb, picks, RP, gl.SliceLayout(1, wl))
            w = gl.arange(0, BLOCK_N // 16, gl.SliceLayout(0, wl))[None, :]
            mb.put(
                "h1",
                mb.row_offset(row[:, None])
                + pick[:, None] * H1_WORDS
                + blk * (BLOCK_N // 16)
                + w,
                words,
                mask=routed[:, None] & (w >= 0),
            )
            # A row's 2 scale bytes in one thread, for the split.
            pl2: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [nw, 1], [1, 0])
            lo, hi = gl.split(gl.convert_layout(scales.to(gl.int32), pl2))
            sl: gl.constexpr = lo.type.layout
            row, pick, routed = _slot_rows(mb, picks, RP, sl)
            mb.put(
                "h1",
                mb.row_offset(row) + pick * H1_WORDS + INTER // 8 + blk,
                lo | (hi << 8),
                mask=routed,
            )
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _routed_order(picks, R: gl.constexpr):
    """The MFMA row of each routed row (the routed rows before it), and the
    number of routed rows (for moe_experts_bf16's polls)."""
    order = ()
    count = gl.to_tensor(0)
    for r in gl.static_range(R):
        order += (count,)
        count += (picks[r] >= 0).to(gl.int32)
    return order, count


@gluon.jit
def _ex2_gather(mb, union, expert):
    """The polls of one ex2 slot: the MXFP4 intermediates (:func:`exr_stage`)
    of the rows that picked ``expert``, MFMA row ``m`` the ``m``-th of them,
    into LDS, one poll a group of up to 8 rows. Returns ``((values, scales)
    in LDS, picks, order)`` for :func:`_ex2_compute`."""
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
    # poll MFMA row 0's words, then hold zeros (E2M1 0 at scale byte 0, as
    # quantized zeros).
    wl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    sl: gl.constexpr = gl.BlockedLayout([1, 2], [8, 8], [1, nw], [1, 0])
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    p_words = gl.allocate_shared_memory(gl.int32, [M_DUP, INTER // 8], plain)
    s_words = gl.allocate_shared_memory(gl.int32, [M_DUP, NB1 // 2], plain)
    v = gl.arange(0, INTER // 8, gl.SliceLayout(0, wl))[None, :]
    b = gl.arange(0, NB1, gl.SliceLayout(0, sl))[None, :]
    for g in gl.static_range(RP // G):
        row, pick, routed = _mfma_rows(
            picks, order, count, g * G, G, R, gl.SliceLayout(1, wl)
        )
        row_s, pick_s, routed_s = _mfma_rows(
            picks, order, count, g * G, G, R, gl.SliceLayout(1, sl)
        )
        got, sc = mb.poll_many(
            ("h1", "h1"),
            (
                mb.row_offset(row[:, None]) + pick[:, None] * H1_WORDS + v,
                mb.row_offset(row_s[:, None])
                + pick_s[:, None] * H1_WORDS
                + INTER // 8
                + b,
            ),
        )
        got = gl.where(routed[:, None], got, 0)
        sc = gl.where(routed_s[:, None], sc, 0)
        lo, hi = gl.split(gl.reshape(sc, [G, NB1 // 2, 2]))
        if g == 0:
            # The previous slot's products read these buffers.
            gl.barrier()
        p_words.slice(g * G, G, dim=0).store(got)
        s_words.slice(g * G, G, dim=0).store(lo | (hi << 16))
    for k in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << k) < M_DUP:
            p_words.slice(RP << k, RP << k, dim=0).store(
                gl.zeros([RP << k, INTER // 8], gl.int32, wl)
            )
            s_words.slice(RP << k, RP << k, dim=0).store(
                gl.zeros([RP << k, NB1 // 2], gl.int32, sl)
            )
    gl.barrier()
    p_smem = p_words.reinterpret(gl.uint8, [M_DUP, INTER // 2], plain)
    s_smem = s_words.reinterpret(gl.uint8, [M_DUP, INTER // 32], plain)
    return (p_smem, s_smem), picks, order


@gluon.jit
def _ex2_gather_lane(mb, union, lane, S: gl.constexpr, KEEP_PICKS: gl.constexpr):
    """The polls of ex2 lane ``lane``'s first ``S`` slots in one round trip:
    each slot's MXFP4 intermediates (:func:`exr_stage`) of every row that
    picked its expert, MFMA row ``r`` row ``r`` (the other rows zero), into
    LDS. Returns the LDS (values ``[S 16, 256]`` and scales ``[S 16, 16]``
    bytes) for :func:`_ex2_compute`, a slot's 16 rows a slice.

    The picks of all ``S`` slots come from the slot table in one reduction.
    A slot past the lane's last, and a row that did not pick the slot's
    expert, poll words that are published (the lane's first slot, the
    slot's first routed row). With ``KEEP_PICKS`` (past one row) it also
    returns those picks in LDS, ``int32 [S, 16]``: row ``k`` is slot
    ``lane + 8 k``, column ``r`` row ``r``'s pick of its expert, or -1 (a
    row that did not pick it, a row past the launch's, or a slot past the
    lane's last), from the same reduction."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    ids, owner, slot, count, table = union
    NP: gl.constexpr = table.shape[0]
    # [slot, pick, row]: a slot a wave, a pick a lane, the rows in registers.
    l3: gl.constexpr = gl.BlockedLayout([1, 1, RP], [1, 64, 1], [nw, 1, 1], [2, 1, 0])
    pl: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, l3))
    j = gl.arange(0, S, gl.SliceLayout(1, gl.SliceLayout(2, l3)))[:, None, None]
    want = lane + j * LANES
    want = gl.where(want < count, want, lane)
    own = gl.convert_layout(owner, pl)[None, :, None]
    mine = gl.convert_layout(slot, pl)[None, :, None]
    t3 = gl.convert_layout(table, gl.SliceLayout(0, l3))[None, :, :]
    picks = gl.max(gl.where((own > 0) & (mine == want), t3, -1), axis=1)  # [S, RP]
    tl: gl.constexpr = picks.type.layout
    if KEEP_PICKS:
        ks = gl.arange(0, S, gl.SliceLayout(1, tl))[:, None]
        plain_p: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        picks_smem = gl.allocate_shared_memory(gl.int32, [S, M_DUP], plain_p)
        picks_smem.slice(0, RP, dim=1).store(
            gl.where(lane + ks * LANES < count, picks, -1)
        )
        if RP < M_DUP:
            # MFMA rows past the launch's rows: no pick.
            fill: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
            picks_smem.slice(RP, M_DUP - RP, dim=1).store(
                gl.full([S, M_DUP - RP], -1, gl.int32, fill)
            )
    r = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    routed = picks >= 0
    first = gl.min(gl.where(routed, r, RP), axis=1)[:, None]
    first_pick = gl.max(gl.where(r == first, picks, -1), axis=1)[:, None]
    row = gl.where(routed, r, first)
    pick = gl.where(routed, picks, first_pick)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 64], [nw, 1, 1], [2, 1, 0])
    sl: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 16, 4], [nw, 1, 1], [2, 1, 0])
    v = gl.arange(0, INTER // 8, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    b = gl.arange(0, NB1, gl.SliceLayout(0, gl.SliceLayout(1, sl)))
    row_w = gl.convert_layout(row, gl.SliceLayout(2, wl))[:, :, None]
    pick_w = gl.convert_layout(pick, gl.SliceLayout(2, wl))[:, :, None]
    row_s = gl.convert_layout(row, gl.SliceLayout(2, sl))[:, :, None]
    pick_s = gl.convert_layout(pick, gl.SliceLayout(2, sl))[:, :, None]
    got, sc = mb.poll_many(
        ("h1", "h1"),
        (
            mb.row_offset(row_w) + pick_w * H1_WORDS + v[None, None, :],
            mb.row_offset(row_s) + pick_s * H1_WORDS + INTER // 8 + b[None, None, :],
        ),
    )
    got = gl.where(gl.convert_layout(routed, gl.SliceLayout(2, wl))[:, :, None], got, 0)
    sc = gl.where(gl.convert_layout(routed, gl.SliceLayout(2, sl))[:, :, None], sc, 0)
    lo, hi = gl.split(gl.reshape(sc, [S, RP, NB1 // 2, 2]))
    # Row-major (unswizzled): the [slot, row, word] buffers are read back as
    # [slot * 16 + row, byte], a slot's 16 MFMA rows a slice.
    plain3: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    p_smem = gl.allocate_shared_memory(gl.uint8, [S, M_DUP, INTER // 2], plain3)
    s_smem = gl.allocate_shared_memory(gl.uint8, [S, M_DUP, INTER // 32], plain3)
    p_words = p_smem.reinterpret(gl.int32, [S, M_DUP, INTER // 8], plain3)
    s_words = s_smem.reinterpret(gl.int32, [S, M_DUP, NB1 // 2], plain3)
    p_words.slice(0, RP, dim=1).store(got)
    s_words.slice(0, RP, dim=1).store(lo | (hi << 16))
    for k in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << k) < M_DUP:
            p_words.slice(RP << k, RP << k, dim=1).store(
                gl.zeros([S, RP << k, INTER // 8], gl.int32, wl)
            )
            s_words.slice(RP << k, RP << k, dim=1).store(
                gl.zeros([S, RP << k, NB1 // 2], gl.int32, sl)
            )
    gl.barrier()
    out = (
        p_smem.reinterpret(gl.uint8, [S * M_DUP, INTER // 2], plain),
        s_smem.reinterpret(gl.uint8, [S * M_DUP, INTER // 32], plain),
    )
    if KEEP_PICKS:
        out += (picks_smem,)
    return out


@gluon.jit
def _ex2_compute_picked(mb, task, loads, a, picks_smem, RANK: gl.constexpr):
    """:func:`_ex2_compute` past one row, with the slot's row picks from LDS
    (``picks_smem`` ``[1, 16]``, kept by :func:`_ex2_gather_lane`): MFMA row ``r`` is
    row ``r``, as there."""
    TASKS_PER_LANE: gl.constexpr = NB2 // EX2_BLOCKS
    blk0 = (task % TASKS_PER_LANE) * EX2_BLOCKS
    p_smem, s_smem = a
    owner = (blk0 * BLOCK_N) // OWNED_OUT
    for b in gl.static_range(EX2_BLOCKS):
        acc = _mma_rows(p_smem, s_smem, loads[b])
        out = gl.reshape(acc.to(gl.bfloat16), [M_DUP, BLOCK_N // 2, 2])
        word = mx.pack_bf16_pairs(out)
        wl: gl.constexpr = word.type.layout
        mm = gl.arange(0, M_DUP, gl.SliceLayout(1, wl))[:, None]
        p = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, wl))[None, :]
        rl: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 64], [gl.num_warps(), 1], [1, 0]
        )
        pick_of = gl.convert_layout(
            gl.reshape(picks_smem.load(rl), [M_DUP]), gl.SliceLayout(1, wl)
        )[:, None]
        col = ((blk0 + b) * BLOCK_N) % OWNED_OUT
        at = ((RANK * TOP_K + pick_of) * OWNED_OUT + col) // 2 + p
        mb.put_peer(
            "rr",
            owner,
            mb.row_offset(mm) + at,
            word,
            mask=(pick_of >= 0) & (p >= 0),
        )


@gluon.jit
def _mfma_rows(
    picks,
    order,
    count,
    M0: gl.constexpr,
    G: gl.constexpr,
    R: gl.constexpr,
    layout: gl.constexpr,
):
    """``(row, pick, routed)`` ``[G]`` of MFMA rows ``M0 ..``: the routed row
    whose order is the MFMA row, else MFMA row 0's."""
    m = M0 + gl.arange(0, G, layout)
    row = gl.full([G], -1, gl.int32, layout)
    pick = gl.full([G], -1, gl.int32, layout)
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
    return gl.where(routed, row, row0), gl.where(routed, pick, pick0), routed


@gluon.jit
def _ex2_compute(mb, task, loads, gathered, RANK: gl.constexpr):
    """One ex2 slot after :func:`_ex2_gather`: the MXFP4 intermediate times
    the slot's w2 ``loads``, the BF16 partials put to the owner rank."""
    R: gl.constexpr = schedule.rows(mb.plan)
    TASKS_PER_LANE: gl.constexpr = NB2 // EX2_BLOCKS
    blk0 = (task % TASKS_PER_LANE) * EX2_BLOCKS
    a, picks, order = gathered
    p_smem, s_smem = a
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
    first slot, ``expert``) and :func:`slots` (``union``): the lane's first
    ``EX2_GATHER`` slots from one gather of their intermediates, the first
    ``EX2_AHEAD`` slots' weights issued before the gather and each later
    slot's that many slots ahead of its product; then (past 64 distinct
    experts) the lane's other slots one at a time.

    A poll waits for every earlier load of its wave (``vmcnt`` counts in
    order): polled a slot at a time between the slots' weights, each slot
    paid its weights' latency and a poll round trip (13 us for 4 slots,
    about 2.4x their stream at 8 rows). The unrolled slots carry no weights
    across a loop, whose copies would wait for the loads in flight."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    S: gl.constexpr = min(RP, EX2_GATHER)
    A: gl.constexpr = max(2, min(S, EX2_AHEAD))
    lane = task // (NB2 // EX2_BLOCKS)
    count = union[3]
    tiles = (loads,)
    for k in gl.static_range(1, A):
        tiles += (
            _ex2_loads(
                task,
                True,
                _slot_expert(union, lane + k * LANES),
                w2,
                w2s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
            ),
        )
    if R > 1:
        # Past one row, each slot's row picks come from LDS (kept by the
        # gather from its one reduction): per slot, the R row scalars of
        # _row_picks were R cross-lane reductions.
        p_smem, s_smem, picks_smem = _ex2_gather_lane(mb, union, lane, S, True)
    else:
        p_smem, s_smem = _ex2_gather_lane(mb, union, lane, S, False)
    mb.stamp(STAGE, trace.READY)
    # MFMA row r is row r (rows that did not pick the slot's expert are zero).
    rows = ()
    for r in gl.static_range(R):
        rows += (r,)
    for k in gl.static_range(S):
        i = lane + k * LANES
        if k + A < S:
            tiles += (
                _ex2_loads(
                    task,
                    True,
                    _slot_expert(union, i + A * LANES),
                    w2,
                    w2s,
                    stride_we,
                    stride_se,
                    stride_slin,
                    stride_snb,
                ),
            )
        if i < count:
            a = (
                p_smem.slice(k * M_DUP, M_DUP, dim=0),
                s_smem.slice(k * M_DUP, M_DUP, dim=0),
            )
            if R > 1:
                _ex2_compute_picked(
                    mb, task, tiles[k], a, picks_smem.slice(k, 1, dim=0), RANK
                )
            else:
                picks = _row_picks(union, _slot_expert(union, i), R)
                _ex2_compute(mb, task, tiles[k], (a, picks, rows), RANK)
    if S < RP:
        following = _slot_expert(union, lane + S * LANES)
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
        for i in range(lane + S * LANES, count, LANES):
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
