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

"""MXFP4 experts of one Muse Spark rank (one row, 8 picks).

At TP8 / EP1 every rank holds all 256 experts, each sharded over its
intermediate dimension: the rank's 512 of 4096. The stages read the regular
path's preprocessed weights in place (gdot128-shuffled E2M1 bytes and
CDNA4-swizzled E8M0 scales, as ``decode_stage1`` / ``decode_stage2`` read
them): a second copy of every expert would not fit.

* ex1, 256 tasks: pick ``k``, 128 of the 1024 interleaved gate / up columns,
  a quarter of K (64 KiB of weights). A task polls the whole pooled row,
  applies the pre-expert RMS norm, and quantizes its K quarter to MXFP4 (the
  regular path's "even" rule) in LDS. It publishes the FP32 partial sums.
* ex2, 256 tasks: pick ``k``, 128 of the 4096 output columns over the rank's
  512 intermediate values (32 KiB). A task polls its pick's ex1 partials,
  sums the 4 in K order, applies SwiGLU (``silu(gate) up``) and the regular
  path's MXFP4 requantization (groups of 32) itself: the 32 tasks of a pick
  repeat this small step, so no stage hands the intermediate over. The BF16
  partial goes to the rank that owns those columns (``rr``), which sums the
  8 ranks' partials.

Expert weights cannot load before routing picks the experts. So that both
weight streams start then, and every CTA reads the same 96 KiB, a CTA runs
its ex1 and ex2 tasks in parts: :func:`ex1_prep` (the pooled row is ready
before routing), then :func:`ex1_issue` and :func:`ex2_issue`, then
:func:`ex1_finish`, and :func:`ex2_finish` at ex2's place in the plan. The
prep polls before the loads are issued: a poll waits for every earlier load
of its wave (``vmcnt`` counts in order).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _cdna4_swizzled_mxfp4_scale_offset,
    _direct_mxfp4_mfma,
    _direct_mxfp4_mfma_layouts,
    _gluon_dot_preshuffled_w_offset,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.moe_pool import POOLED
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

TOP_K = gl.constexpr(8)
INTER = gl.constexpr(512)  # the rank's intermediate values of an expert
W13_N = gl.constexpr(1024)  # interleaved gate / up columns
BLOCK_N = gl.constexpr(128)
M_DUP = gl.constexpr(16)  # MFMA rows; the activation row repeats in each
KSPLIT = gl.constexpr(4)
K_SLICE = gl.constexpr(1024)  # POOLED // KSPLIT
NB1 = gl.constexpr(8)  # W13_N // BLOCK_N
NB2 = gl.constexpr(32)  # POOLED // BLOCK_N
EX1_TASKS = gl.constexpr(256)  # TOP_K * NB1 * KSPLIT
EX2_BLOCKS = gl.constexpr(1)  # 128-column blocks of an ex2 task
EX2_TASKS = gl.constexpr(256)  # TOP_K * NB2 // EX2_BLOCKS
OWNED_OUT = gl.constexpr(512)  # output columns a rank reduces: POOLED / WORLD
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("rr",)


def regions() -> tuple[tuple[str, int], ...]:
    k, nb1, ks, bn = int(TOP_K), int(NB1), int(KSPLIT), int(BLOCK_N)
    return (
        ("p1", k * nb1 * ks * bn),  # FP32 [pick][block][K quarter][column]
        (
            "rr",
            int(WORLD) * k * int(OWNED_OUT) // 2,
        ),  # BF16 pairs [source][pick][column]
    )


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    ex1, ex2 = placement(bases, ("ex1", "ex2"))
    return (
        Stage("ex1", int(EX1_TASKS), ex1, reads=("route", "cg"), writes=("p1",)),
        Stage("ex2", int(EX2_TASKS), ex2, reads=("route", "p1"), writes=("rr",)),
    )


@gluon.jit
def _expert(mb, pick):
    """The expert id of pick ``pick`` (``route``)."""
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    (got,) = mb.poll_many(("route",), (pick + gl.arange(0, 1, lanes),))
    return gl.max(got, axis=0)


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
    ``active`` is false."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    dot_b: gl.constexpr = layouts[2]
    b_scale: gl.constexpr = layouts[4]
    bk = gl.arange(0, 64, gl.SliceLayout(1, dot_b))[:, None]
    bn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(0, dot_b))[None, :]
    sn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale))[:, None]
    sk = gl.arange(0, 4, gl.SliceLayout(0, b_scale))[None, :]
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
            mask=(bk >= 0) & active,
        )
        s = gl.amd.cdna4.buffer_load(
            ws,
            _cdna4_swizzled_mxfp4_scale_offset(
                s_off, sn, kt * 4 + sk, stride_slin, stride_snb
            ).to(gl.int32),
            mask=(sk >= 0) & active,
        )
        tiles += ((b, s),)
    return tiles


@gluon.jit
def _mma_row(packed_smem, scale_smem, tiles):
    """Row 0 of the MXFP4 product of the LDS A rows and the B ``tiles``: FP32
    ``[128]``."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    mfma: gl.constexpr = layouts[0]
    acc = gl.zeros([M_DUP, BLOCK_N], gl.float32, mfma)
    for i in gl.static_range(len(tiles)):
        a = packed_smem.slice(i * 64, 64, 1).load(layouts[1])
        a_s = scale_smem.slice(i * 4, 4, 1).load(layouts[3])
        acc = _direct_mxfp4_mfma(acc, a, tiles[i][0], a_s, tiles[i][1])
    rows = gl.arange(0, M_DUP, gl.SliceLayout(1, mfma))[:, None]
    return gl.sum(gl.where(rows == 0, acc, 0.0), axis=0)


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


@gluon.jit
def ex1_prep(mb, task, active, pre_gain, STAGE: gl.constexpr):
    """The A operand of ex1 task ``task``: poll the whole pooled row (the
    norm needs every value), apply the pre-expert RMS norm, and quantize the
    task's K quarter to MXFP4 in LDS. Returns the LDS (values, scales).

    Every CTA calls it (Gluon cannot carry values out of a branch); an
    inactive CTA (``active`` false) stamps nothing.

    Args:
        pre_gain: FP32 ``bf16(w + 1)`` of the pre-expert norm ``[4096]``.
    """
    if active:
        mb.stamp(STAGE, trace.START)
    ks = task % KSPLIT
    nw: gl.constexpr = gl.num_warps()
    ql: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    q = gl.arange(0, KSPLIT, gl.SliceLayout(1, ql))
    j = gl.arange(0, K_SLICE // 2, gl.SliceLayout(0, ql))
    c = ks * K_SLICE + 2 * j
    gain_lo = gl.load(pre_gain + c)
    gain_hi = gl.load(pre_gain + c + 1)
    (got,) = mb.poll_many(("cg",), (q[:, None] * (K_SLICE // 2) + j[None, :],))
    if active:
        mb.stamp(STAGE, trace.READY)
    lo = (got << 16).to(gl.float32, bitcast=True)
    hi = (got & -65536).to(gl.float32, bitcast=True)
    inv = gl.rsqrt(gl.sum(gl.sum(lo * lo + hi * hi, axis=1), axis=0) / POOLED + PRE_EPS)
    mine = q[:, None] == ks
    lo = gl.sum(gl.where(mine, lo, 0.0), axis=0)
    hi = gl.sum(gl.where(mine, hi, 0.0), axis=0)
    lo = lo * inv * gain_lo
    hi = hi * inv * gain_hi
    x = gl.reshape(gl.join(lo, hi), [K_SLICE])
    x16 = (
        gl.zeros([M_DUP, 1], gl.float32, ql)
        + gl.convert_layout(x, gl.SliceLayout(0, ql))[None, :]
    )
    packed, scales = _mxfp4_quantize_tile(x16)
    return _stage_a(packed, scales)


@gluon.jit
def ex1_issue(
    mb, task, active, w13, w13s, stride_we, stride_se, stride_slin, stride_snb
):
    """Issue ex1 task ``task``'s w13 loads (pick, column block, K quarter);
    nothing loads when ``active`` is false.

    Args:
        w13, w13s: The rank's preprocessed w13 bytes ``[256, 2048, 1024]`` and
            scales (as ``decode_stage1`` takes them), with their strides.
    """
    pick = task // (NB1 * KSPLIT)
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    expert = _expert(mb, pick)
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
def ex1_finish(mb, task, a, tiles, STAGE: gl.constexpr):
    """Ex1 task ``task`` after :func:`ex1_prep` (``a``) and :func:`ex1_issue`
    (``tiles``): multiply and publish the FP32 partial sums."""
    pick = task // (NB1 * KSPLIT)
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    p_smem, s_smem = a
    acc = _mma_row(p_smem, s_smem, tiles)
    mb.stamp(STAGE, trace.COMPUTED)
    n = gl.arange(0, BLOCK_N, acc.type.layout)
    mb.put(
        "p1",
        ((pick * NB1 + blk) * KSPLIT + ks) * BLOCK_N + n,
        acc.to(gl.int32, bitcast=True),
    )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def ex2_issue(
    mb,
    task,
    active,
    w2,
    w2s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    STAGE: gl.constexpr,
):
    """Issue ex2 task ``task``'s w2 loads (pick, column block); nothing
    loads (and nothing stamps) when ``active`` is false. Every CTA calls it
    (Gluon cannot carry the tiles out of a branch); :func:`ex2_finish` uses
    them.

    Args:
        w2, w2s: The rank's preprocessed w2 bytes ``[256, 256, 4096]`` and
            scales, with their strides.
    """
    if active:
        mb.stamp(STAGE, trace.START)
    TASKS_PER_PICK: gl.constexpr = NB2 // EX2_BLOCKS
    pick = task // TASKS_PER_PICK
    blk0 = (task % TASKS_PER_PICK) * EX2_BLOCKS
    expert = _expert(mb, pick)
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
def ex2_finish(mb, task, loads, RANK: gl.constexpr, STAGE: gl.constexpr):
    """Ex2 task ``task`` after :func:`ex2_issue`: poll the pick's ex1
    partials, form the MXFP4 intermediate, multiply, and put the BF16
    partial to the owner rank."""
    TASKS_PER_PICK: gl.constexpr = NB2 // EX2_BLOCKS
    pick = task // TASKS_PER_PICK
    blk0 = (task % TASKS_PER_PICK) * EX2_BLOCKS
    nw: gl.constexpr = gl.num_warps()
    gl.static_assert(nw == NB1, "ex2 sums one ex1 column block a wave")
    # [ex1 block, K quarter, column]: a wave per block, the K quarters in a
    # lane's registers.
    pl: gl.constexpr = gl.BlockedLayout(
        [1, KSPLIT, 2], [1, 1, 64], [nw, 1, 1], [2, 1, 0]
    )
    b1 = gl.arange(0, NB1, gl.SliceLayout(1, gl.SliceLayout(2, pl)))[:, None, None]
    ks = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, pl)))[None, :, None]
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, pl)))[None, None, :]
    (got,) = mb.poll_many(("p1",), (((pick * NB1 + b1) * KSPLIT + ks) * BLOCK_N + n,))
    mb.stamp(STAGE, trace.READY)
    parts = got.to(gl.float32, bitcast=True)
    # The regular path's order: K quarter 0, then 1, ...
    acc = gl.zeros([NB1, BLOCK_N], gl.float32, gl.SliceLayout(1, pl))
    for k in gl.static_range(KSPLIT):
        acc += gl.sum(gl.where(ks == k, parts, 0.0), axis=1)
    gate, up = gl.split(gl.reshape(acc, [NB1, BLOCK_N // 2, 2]))
    h = gl.reshape(gate / (1.0 + gl.exp(-gate)) * up, [INTER])
    ql: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    h16 = (
        gl.zeros([M_DUP, 1], gl.float32, ql)
        + gl.convert_layout(h, gl.SliceLayout(0, ql))[None, :]
    )
    packed, scales = _mxfp4_quantize_tile(h16)
    p_smem, s_smem = _stage_a(packed, scales)
    owner = (blk0 * BLOCK_N) // OWNED_OUT
    for b in gl.static_range(EX2_BLOCKS):
        acc = _mma_row(p_smem, s_smem, loads[b])
        out = gl.reshape(acc.to(gl.bfloat16), [BLOCK_N // 2, 2])
        word = mx.pack_bf16_pairs(out)
        p = gl.arange(0, BLOCK_N // 2, word.type.layout)
        col = ((blk0 + b) * BLOCK_N) % OWNED_OUT
        mb.put_peer(
            "rr", owner, ((RANK * TOP_K + pick) * OWNED_OUT + col) // 2 + p, word
        )
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)
