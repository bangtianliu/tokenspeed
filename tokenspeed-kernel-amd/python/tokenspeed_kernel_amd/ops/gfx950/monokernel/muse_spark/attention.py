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

"""Attention of one Muse Spark rank: 16 q heads, 2 KV heads.

The launch's rows are the consecutive tokens of one or more requests
(``Plan.chain`` a request): one row a request for decode, ``k + 1`` for a
speculative verify of ``k`` draft tokens. Row ``t`` of a request is its
token at position ``L0 + t``, where ``L0`` is the request's own.

* head, 2 tasks per row: KV head ``g``. The per-head RMS norm of k in FP32
  (gain 1), then interleaved RoPE (GPT-J pairs) on sliding layers, then one
  rounding to the cache's dtype (BF16, or FP8 e4m3 at unit scale, as the
  regular path's prologue writes it); the rotated k and the raw v go to the
  KV cache slot of the row's token.
* attn, ``2 * SPLITS`` tasks a request: KV head ``g``, split ``s``, for every
  row of the request (its length and page table row are its own). A
  task forms the rotated q of its 8 q heads (q gain 5.473) of every row,
  one MFMA row per (row, head), and the launch's k and v itself, as head
  does, from the projection: no hand-off between them. It takes the
  launch's k and v from there, not from the cache: a CTA never reads
  another CTA's plain store of the same launch; it rounds them to the
  cache's dtype as head does, so they are the cached values. An FP8 cache's
  keys and values widen to BF16 (exactly) for the BF16 MFMAs. Row ``t`` attends keys
  ``max(0, L0 + t - WINDOW + 1) .. L0 + t`` (from 0 on a full layer). The
  splits share the cached keys before ``L0``, ``ceil(n / SPLITS)`` each
  (from the first key that row 0 attends), in chunks of 128 with an online
  base-2 softmax. The last split then also takes the launch's tokens, as one
  more chunk. A task publishes FP32 ``acc / l`` and the LSE of the 8 q heads
  of its KV head of each row (an empty split publishes ``-inf``).
* merge, 16 tasks per row, one per q head: the LSE merge of the splits,
  BF16, then the regular path's gated head norm ``bf16(v rsqrt(mean(v^2) +
  eps) sigmoid(gate))``, published in ``z`` for the output projection.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    GATE_ROW,
    HEAD_DIM,
    K_ROW,
    KV_HEADS,
    Q_HEADS,
    QKVG,
    V_ROW,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

HEAD = HEAD_DIM
GROUP = gl.constexpr(8)  # q heads per KV head
PAGE = gl.constexpr(64)  # tokens a KV page
KV_STRIDE = gl.constexpr(128)  # cache elements a slot: 2 KV heads x 64
# Keys a chunk, for either cache dtype: 256 FP8 keys (the bytes of 128 BF16
# ones) made a full layer slower below ~200k cached keys (8-GPU bench).
BN = gl.constexpr(128)
WINDOW = 2048  # a sliding layer attends the last 2048 positions, its own included
EPS = gl.constexpr(1e-5)
Q_GAIN = gl.constexpr(5.473006248474121)  # qk_scale_factor as an FP32 norm gain
# exp2 softmax of (q . k) / sqrt(64).
QK_SCALE = gl.constexpr(0.125 * 1.4426950408889634)
HEAD_TASKS = KV_HEADS
MERGE_TASKS = Q_HEADS


def regions(splits: int) -> tuple[tuple[str, int], ...]:
    heads, kv, group, head = int(Q_HEADS), int(KV_HEADS), int(GROUP), int(HEAD)
    return (
        ("part", splits * kv * group * head),  # FP32
        ("lse", splits * heads),  # FP32
        ("z", heads * head // 2),  # BF16 pairs
    )


def stages(
    bases: dict[str, int], splits: int, rows: int, chain: int
) -> tuple[Stage, ...]:
    """Head, attn and merge of ``rows`` token rows (requests of ``chain``
    tokens), placed by ``bases``."""
    head, attn, merge = placement(bases, ("head", "attn", "merge"))
    return (
        Stage("head", int(HEAD_TASKS) * rows, head, reads=("qkvg",)),
        Stage(
            "attn",
            rows // chain * int(KV_HEADS) * splits,
            attn,
            reads=("qkvg",),
            writes=("part", "lse"),
        ),
        Stage(
            "merge",
            int(MERGE_TASKS) * rows,
            merge,
            reads=("part", "lse", "gate"),
            writes=("z",),
        ),
    )


@gluon.constexpr_function
def _pair_layout(rank, nw):
    """Index layouts of polls of projection pairs: the last axis (half 0
    even, half 0 odd, half 1 even, half 1 odd) in one thread. ``rank`` 2:
    ``[32 pairs, 4]``; 3: ``[16 heads, 32 pairs, 4]``."""
    if rank == 2:
        return gl.BlockedLayout([1, 4], [64, 1], [nw, 1], [1, 0])
    return gl.BlockedLayout([1, 1, 4], [2, 32, 1], [nw, 1, 1], [2, 1, 0])


@gluon.jit
def _pair_index(pair, e, N: gl.constexpr, row0):
    """Partial-sum indices of values ``row0 + 2 pair`` and ``+ 1`` of a
    projection region of ``N`` rows a K half (``e``: the last axis)."""
    return (e // 2) * N + row0 + 2 * pair + e % 2


@gluon.jit
def _pair_values(got, e, AX: gl.constexpr):
    """(even, odd) BF16-rounded values (FP32) from the polled partial sums
    of :func:`_pair_index` (``e``: their last axis ``AX``): half 0 plus
    half 1."""
    x = got.to(gl.float32, bitcast=True)
    lo = gl.sum(gl.where(e % 2 == 0, x, 0.0), axis=AX)
    hi = gl.sum(gl.where(e % 2 == 1, x, 0.0), axis=AX)
    return lo.to(gl.bfloat16).to(gl.float32), hi.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _norm_rope(lo, hi, gain, cos_sin, pos, j, ROPE: gl.constexpr):
    """Per-head RMS norm over the 64 values (pairs ``j``), then RoPE, in
    FP32."""
    inv = gl.rsqrt(gl.sum(lo * lo + hi * hi, axis=0) / HEAD + EPS)
    lo = lo * inv * gain
    hi = hi * inv * gain
    if ROPE:
        c = gl.load(cos_sin + pos * HEAD + j)
        s = gl.load(cos_sin + pos * HEAD + HEAD // 2 + j)
        lo, hi = gl.fma(lo, c, -(hi * s)), gl.fma(hi, c, lo * s)
    return lo, hi


@gluon.jit
def head_stage(
    mb,
    task,
    seq_len,
    slot,
    k_cache,
    v_cache,
    cos_sin,
    ROPE: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Head task ``task``: k head ``task % 2`` of row ``task // 2`` to the KV
    cache.

    Args:
        seq_len: int32 ``[rows]``: ``L + 1`` for the row's token at position
            ``L``.
        slot: int32 ``[rows]``: KV cache slot of the row's token.
        k_cache, v_cache: BF16 or FP8 e4m3 (unit scale) ``[slots, 2, 64]``.
        cos_sin: FP32 ``[positions, 64]``: ``cos`` of the 32 pairs, then ``sin``
            (unused when ``ROPE`` is false).
    """
    kv_dtype: gl.constexpr = k_cache.dtype.element_ty
    mb.stamp(STAGE, trace.START)
    row, g, _ = schedule.row_task(mb.plan, task, HEAD_TASKS)
    mb = mb.row(row)
    pl: gl.constexpr = _pair_layout(2, gl.num_warps())
    j = gl.arange(0, HEAD // 2, gl.SliceLayout(1, pl))
    e = gl.arange(0, 4, gl.SliceLayout(0, pl))[None, :]
    pos = gl.load(seq_len + row) - 1
    slot_at = gl.load(slot + row).to(gl.int64) * KV_STRIDE + 2 * j
    kw, vw = mb.poll_many(
        ("qkvg", "qkvg"),
        (
            _pair_index(j[:, None], e, GATE_ROW, K_ROW + g * HEAD),
            _pair_index(j[:, None], e, GATE_ROW, V_ROW + g * HEAD),
        ),
    )
    mb.stamp(STAGE, trace.READY)
    lo, hi = _pair_values(kw, e, 1)
    lo, hi = _norm_rope(lo, hi, 1.0, cos_sin, pos, j, ROPE)
    # The cache keeps the rotated k and the raw v, each rounded once.
    at = slot_at + g * HEAD
    gl.store(k_cache + at, lo.to(kv_dtype))
    gl.store(k_cache + at + 1, hi.to(kv_dtype))
    vlo, vhi = _pair_values(vw, e, 1)
    gl.store(v_cache + at, vlo.to(kv_dtype))
    gl.store(v_cache + at + 1, vhi.to(kv_dtype))
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def _layouts(nw):
    """(qk MFMA, pv MFMA, K operand, V load) layouts."""
    qk_mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, nw]
    )
    pv_mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, nw]
    )
    k_layout = gl.DotOperandLayout(1, qk_mma, 8)
    v_load = gl.BlockedLayout([1, 8], [8, 8], [nw, 1], [1, 0])
    return qk_mma, pv_mma, k_layout, v_load


@gluon.jit
def _load_kv(
    table,
    k_cache,
    v_cache,
    g,
    c0,
    hi,
    dk,
    dv,
    k_layout: gl.constexpr,
    v_load: gl.constexpr,
):
    """Keys ``[64, BN]`` and values ``[BN, 64]`` of chunk ``c0 ..`` from the
    cache (zero from ``hi``), in BF16 (an FP8 cache's widen exactly)."""
    jk = c0 + gl.arange(0, BN, gl.SliceLayout(0, k_layout))
    jv = c0 + gl.arange(0, BN, gl.SliceLayout(1, v_load))
    old_k = jk < hi
    old_v = jv < hi
    page_k = gl.load(table + jk // PAGE, mask=old_k, other=0)
    page_v = gl.load(table + jv // PAGE, mask=old_v, other=0)
    slot_k = page_k.to(gl.int64) * PAGE + jk % PAGE
    slot_v = page_v.to(gl.int64) * PAGE + jv % PAGE
    keys = gl.load(
        k_cache + g * HEAD + slot_k[None, :] * KV_STRIDE + dk[:, None],
        mask=old_k[None, :],
        other=0.0,
    )
    values = gl.load(
        v_cache + g * HEAD + slot_v[:, None] * KV_STRIDE + dv[None, :],
        mask=old_v[:, None],
        other=0.0,
    )
    return keys.to(gl.bfloat16), values.to(gl.bfloat16)


@gluon.jit
def _softmax_step(q, keys, values, valid, m, l, acc, p_layout: gl.constexpr):
    """One chunk of the online base-2 softmax: ``keys [64, N]`` and
    ``values [N, 64]`` (an operand layout and LDS), where ``valid [QM, N]``."""
    pv_mma: gl.constexpr = acc.type.layout
    scores = gl.zeros([q.shape[0], keys.shape[1]], gl.float32, valid.type.layout)
    scores = gl.amd.cdna4.mfma(q, keys, scores)
    scores = gl.where(valid, scores * QK_SCALE, float("-inf"))
    m_new = gl.maximum(m, gl.max(scores, axis=1))
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m - m_safe)
    p = gl.exp2(scores - m_safe[:, None])
    l = l * alpha + gl.sum(p, axis=1)
    pv = gl.zeros(acc.shape, gl.float32, pv_mma)
    pv = gl.amd.cdna4.mfma(gl.convert_layout(p.to(gl.bfloat16), p_layout), values, pv)
    acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, pv_mma))[:, None] + pv
    return m_new, l, acc


@gluon.jit
def attn_stage(
    mb,
    task,
    seq_len,
    table,
    table_stride,
    k_cache,
    v_cache,
    cos_sin,
    ROPE: gl.constexpr,
    SPLITS: gl.constexpr,
    WINDOW: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Attn task ``task`` = (request ``task // (2 SPLITS)``, KV head, split)
    of every row of the request.

    Args:
        seq_len: int32 ``[rows]``, as head_stage takes it (the request's
            first row's is read).
        table: int32 page tables of the requests, one row each (``PAGE``
            tokens a page), ``table_stride`` apart.
        cos_sin, ROPE: As head_stage takes them.
        WINDOW: Attended positions of a sliding layer, 0 on a full layer.
    """
    C: gl.constexpr = schedule.chain(mb.plan)
    gl.static_assert(C <= 16, "the launch's keys are one chunk of 16")
    # One MFMA row per (row, q head): rows past C * GROUP are zero.
    QM: gl.constexpr = max(16, triton.next_power_of_2(C * GROUP))
    KT: gl.constexpr = 16  # the launch's keys, rows past C zero
    mb.stamp(STAGE, trace.START)
    req = task // (KV_HEADS * SPLITS)
    g = task % (KV_HEADS * SPLITS) // SPLITS
    s = task % SPLITS
    # The request's rows and its page table row.
    base = req * C
    table = table + req.to(gl.int64) * table_stride
    nw: gl.constexpr = gl.num_warps()
    qk_mma: gl.constexpr = _layouts(nw)[0]
    pv_mma: gl.constexpr = _layouts(nw)[1]
    k_layout: gl.constexpr = _layouts(nw)[2]
    v_load: gl.constexpr = _layouts(nw)[3]
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 4)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 4)

    L0 = gl.load(seq_len + base) - 1
    # The cached keys of the splits: from row 0's first key, before L0.
    if WINDOW == 0:
        first = 0
    else:
        first = gl.maximum(L0 - WINDOW + 1, 0)
    per = (L0 - first + SPLITS - 1) // SPLITS
    lo = first + s * per
    hi = gl.minimum(lo + per, L0)

    # The first chunk's cache loads go before the poll: they do not depend
    # on this launch, and they arrive while the projections run.
    dk = gl.arange(0, HEAD, gl.SliceLayout(1, k_layout))
    dv = gl.arange(0, HEAD, gl.SliceLayout(0, v_load))
    keys, values = _load_kv(
        table, k_cache, v_cache, g, lo, hi, dk, dv, k_layout, v_load
    )

    # q of the 8 heads of every row, and the launch's k and v. A thread
    # holds one pair, 32 lanes a head: the norm sums in head_stage's order.
    q3: gl.constexpr = _pair_layout(3, nw)
    ql: gl.constexpr = gl.SliceLayout(2, q3)
    qr = gl.arange(0, QM, gl.SliceLayout(1, ql))[:, None]
    qp = gl.arange(0, HEAD // 2, gl.SliceLayout(0, ql))[None, :]
    eq = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, q3)))[None, None, :]
    q_tok = gl.minimum(qr // GROUP, C - 1)
    q_head = qr % GROUP
    kr = gl.arange(0, KT, gl.SliceLayout(1, ql))[:, None]
    k_tok = gl.minimum(kr, C - 1)
    if ROPE:
        # The RoPE rows load before the poll: a load after a poll would
        # start only when the poll returns.
        cq = gl.load(cos_sin + (L0 + q_tok) * HEAD + qp)
        sq = gl.load(cos_sin + (L0 + q_tok) * HEAD + HEAD // 2 + qp)
        ck = gl.load(cos_sin + (L0 + k_tok) * HEAD + qp)
        sk = gl.load(cos_sin + (L0 + k_tok) * HEAD + HEAD // 2 + qp)
    got_q, got_k, got_v = mb.poll_many(
        ("qkvg", "qkvg", "qkvg"),
        (
            mb.row_offset(base + q_tok)[:, :, None]
            + _pair_index(
                qp[:, :, None],
                eq,
                GATE_ROW,
                ((g * GROUP + q_head) * HEAD)[:, :, None],
            ),
            mb.row_offset(base + k_tok)[:, :, None]
            + _pair_index(qp[:, :, None], eq, GATE_ROW, K_ROW + g * HEAD),
            mb.row_offset(base + k_tok)[:, :, None]
            + _pair_index(qp[:, :, None], eq, GATE_ROW, V_ROW + g * HEAD),
        ),
    )
    mb.stamp(STAGE, trace.READY)
    q_lo, q_hi = _pair_values(got_q, eq, 2)
    inv_q = gl.rsqrt(gl.sum(q_lo * q_lo + q_hi * q_hi, axis=1) / HEAD + EPS)[:, None]
    q_lo = q_lo * inv_q * Q_GAIN
    q_hi = q_hi * inv_q * Q_GAIN
    k_lo, k_hi = _pair_values(got_k, eq, 2)
    inv_k = gl.rsqrt(gl.sum(k_lo * k_lo + k_hi * k_hi, axis=1) / HEAD + EPS)[:, None]
    k_lo = k_lo * inv_k * 1.0
    k_hi = k_hi * inv_k * 1.0
    if ROPE:
        q_lo, q_hi = gl.fma(q_lo, cq, -(q_hi * sq)), gl.fma(q_hi, cq, q_lo * sq)
        k_lo, k_hi = gl.fma(k_lo, ck, -(k_hi * sk)), gl.fma(k_hi, ck, k_lo * sk)
    q = gl.reshape(gl.join(q_lo, q_hi), [QM, HEAD])
    q = gl.where(
        gl.arange(0, QM, gl.SliceLayout(1, q.type.layout))[:, None] < C * GROUP,
        q,
        0.0,
    )
    q = gl.convert_layout(q.to(gl.bfloat16), q_layout)
    v_lo, v_hi = _pair_values(got_v, eq, 2)
    # The launch's k and v as the cache holds them: rounded once to its dtype.
    kv_dtype: gl.constexpr = k_cache.dtype.element_ty
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    cur_k = gl.allocate_shared_memory(gl.bfloat16, [KT, HEAD], plain)
    cur_v = gl.allocate_shared_memory(gl.bfloat16, [KT, HEAD], plain)
    cur_k.store(
        gl.reshape(gl.join(k_lo, k_hi), [KT, HEAD]).to(kv_dtype).to(gl.bfloat16)
    )
    cur_v.store(
        gl.reshape(gl.join(v_lo, v_hi), [KT, HEAD]).to(kv_dtype).to(gl.bfloat16)
    )
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [BN, HEAD], plain)

    # Row t's first attended key (a sliding layer's window).
    qi = gl.arange(0, QM, gl.SliceLayout(1, qk_mma))
    qi_tok = gl.minimum(qi // GROUP, C - 1)
    if WINDOW == 0:
        row_first = gl.zeros_like(qi_tok)
    else:
        row_first = L0 + qi_tok - WINDOW + 1
    m = gl.full([QM], float("-inf"), gl.float32, gl.SliceLayout(1, qk_mma))
    l = gl.zeros([QM], gl.float32, gl.SliceLayout(1, qk_mma))
    acc = gl.zeros([QM, HEAD], gl.float32, pv_mma)
    for c0 in range(lo, hi, BN):
        js = c0 + gl.arange(0, BN, gl.SliceLayout(0, qk_mma))
        valid = (js[None, :] < hi) & (js[None, :] >= row_first[:, None])
        # The next chunk's loads go before this chunk's math.
        next_keys, next_values = _load_kv(
            table, k_cache, v_cache, g, c0 + BN, hi, dk, dv, k_layout, v_load
        )
        gl.barrier()
        v_smem.store(values)
        gl.barrier()
        m, l, acc = _softmax_step(
            q, keys, v_smem.load(v_layout), valid, m, l, acc, p_layout
        )
        keys = next_keys
        values = next_values
    # The launch's tokens: token u is the key at L0 + u; row t attends u <= t.
    gl.barrier()
    u = gl.arange(0, KT, gl.SliceLayout(0, qk_mma))
    own = (u[None, :] <= qi_tok[:, None]) & (u[None, :] < C) & (s == SPLITS - 1)
    m, l, acc = _softmax_step(
        q,
        cur_k.permute([1, 0]).load(gl.DotOperandLayout(1, qk_mma, 8)),
        cur_v.load(v_layout),
        own,
        m,
        l,
        acc,
        p_layout,
    )
    mb.stamp(STAGE, trace.COMPUTED)
    l_acc = gl.convert_layout(l, gl.SliceLayout(1, pv_mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse = gl.where(l > 0, m + gl.log2(gl.where(l > 0, l, 1.0)), float("-inf"))
    r = gl.arange(0, QM, gl.SliceLayout(1, pv_mma))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, pv_mma))
    part = ((s * KV_HEADS + g) * GROUP + r[:, None] % GROUP) * HEAD + c[None, :]
    mb.put(
        "part",
        mb.row_offset(base + r[:, None] // GROUP) + part,
        out.to(gl.int32, bitcast=True),
        mask=(r < C * GROUP)[:, None] & (c >= 0)[None, :],
    )
    mb.put(
        "lse",
        mb.row_offset(base + qi // GROUP) + s * Q_HEADS + g * GROUP + qi % GROUP,
        lse.to(gl.int32, bitcast=True),
        mask=qi < C * GROUP,
    )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def merge_stage(mb, task, SPLITS: gl.constexpr, STAGE: gl.constexpr):
    """Merge task ``task`` = q head ``h = task % 16`` of row ``task // 16``:
    the LSE merge, then the gated head norm of the regular path's output
    projection."""
    mb.stamp(STAGE, trace.START)
    row, h, _ = schedule.row_task(mb.plan, task, MERGE_TASKS)
    mb = mb.row(row)
    g = h // GROUP
    hh = h % GROUP
    nw: gl.constexpr = gl.num_warps()
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [nw, 1], [1, 0])
    s = gl.arange(0, SPLITS, gl.SliceLayout(1, ml))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, ml))
    part = ((s[:, None] * KV_HEADS + g) * GROUP + hh) * HEAD + c[None, :]
    gl2: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [nw, 1], [1, 0])
    gc = gl.arange(0, HEAD, gl.SliceLayout(1, gl2))[:, None]
    gh = gl.arange(0, 2, gl.SliceLayout(0, gl2))[None, :]
    got_p, got_l, got_g = mb.poll_many(
        ("part", "lse", "gate"),
        (part, s * Q_HEADS + h, gh * (QKVG - GATE_ROW) + h * HEAD + gc),
    )
    mb.stamp(STAGE, trace.READY)
    lse = got_l.to(gl.float32, bitcast=True)
    top = gl.max(lse, axis=0)
    w = gl.where(lse == float("-inf"), 0.0, gl.exp2(lse - top))
    p = got_p.to(gl.float32, bitcast=True)
    out = gl.sum(p * w[:, None], axis=0) / gl.sum(w, axis=0)
    a = out.to(gl.bfloat16).to(gl.float32)
    inv = gl.rsqrt(gl.sum(a * a, axis=0) / HEAD + EPS)
    gate = gl.sum(got_g.to(gl.float32, bitcast=True), axis=1)
    gate = gl.convert_layout(gate.to(gl.bfloat16).to(gl.float32), c.type.layout)
    z = a * inv * mx.sigmoid(gate)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "z", h * (HEAD // 2), z)
    mb.stamp(STAGE, trace.END)
