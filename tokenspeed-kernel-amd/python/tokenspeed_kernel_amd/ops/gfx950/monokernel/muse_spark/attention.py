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

"""Attention of one Muse Spark rank: 16 q heads, 2 KV heads, one token row.

* head, 18 tasks: q head ``t`` (``t < 16``) or k head ``t - 16``. The per-head
  RMS norm in FP32 (q gain 5.473, k gain 1), then interleaved RoPE (GPT-J
  pairs) on sliding layers, then one BF16 rounding: published in ``qn`` /
  ``kn``. A k task also writes the rotated k and the raw v to the KV cache
  slot of the token. Attention in this launch takes them from the mailbox:
  a CTA never reads another CTA's plain store of the same launch.
* attn, ``2 * SPLITS`` tasks: KV head ``g``, split ``s``. The attended keys
  are ``max(0, L - WINDOW + 1) .. L`` (all of ``0 .. L`` on a full layer);
  split ``s`` takes ``ceil(n / SPLITS)`` of them, in chunks of 128 with an
  online base-2 softmax. It publishes FP32 ``acc / l`` and the LSE of the 8 q
  heads of its KV head (an empty split publishes ``-inf``).
* merge, 16 tasks, one per q head: the LSE merge of the splits, BF16, then
  the regular path's gated head norm ``bf16(v rsqrt(mean(v^2) + eps)
  sigmoid(gate))``, published in ``z`` for the output projection.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    GATE_ROW,
    HEAD_DIM,
    K_ROW,
    KV_HEADS,
    Q_HEADS,
    V_ROW,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

HEAD = HEAD_DIM
GROUP = gl.constexpr(8)  # q heads per KV head
PAGE = gl.constexpr(64)  # tokens a KV page
KV_STRIDE = gl.constexpr(128)  # cache elements a slot: 2 KV heads x 64
BN = gl.constexpr(128)  # keys a chunk
WINDOW = 2048  # a sliding layer attends the last 2048 positions, its own included
EPS = gl.constexpr(1e-5)
Q_GAIN = gl.constexpr(5.473006248474121)  # qk_scale_factor as an FP32 norm gain
# exp2 softmax of (q . k) / sqrt(64).
QK_SCALE = gl.constexpr(0.125 * 1.4426950408889634)
HEAD_TASKS = gl.constexpr(Q_HEADS + KV_HEADS)
MERGE_TASKS = Q_HEADS


def regions(splits: int) -> tuple[tuple[str, int], ...]:
    heads, kv, group, head = int(Q_HEADS), int(KV_HEADS), int(GROUP), int(HEAD)
    return (
        ("qn", heads * head // 2),  # BF16 pairs
        ("kn", kv * head // 2),
        ("part", splits * kv * group * head),  # FP32
        ("lse", splits * heads),  # FP32
        ("z", heads * head // 2),  # BF16 pairs
    )


def stages(bases: dict[str, int], splits: int) -> tuple[Stage, ...]:
    head, attn, merge = placement(bases, ("head", "attn", "merge"))
    return (
        Stage("head", int(HEAD_TASKS), head, reads=("qkvg",), writes=("qn", "kn")),
        Stage(
            "attn",
            int(KV_HEADS) * splits,
            attn,
            reads=("qn", "kn", "qkvg"),
            writes=("part", "lse"),
        ),
        Stage(
            "merge",
            int(MERGE_TASKS),
            merge,
            reads=("part", "lse", "qkvg"),
            writes=("z",),
        ),
    )


@gluon.jit
def _halves(words):
    """FP32 values ``2 p`` and ``2 p + 1`` of BF16 pairs ``words``."""
    return (words << 16).to(gl.float32, bitcast=True), (words & -65536).to(
        gl.float32, bitcast=True
    )


@gluon.jit
def _norm_rope(lo, hi, gain, cos_sin, pos, j, ROPE: gl.constexpr):
    """Per-head RMS norm over the 64 values (pairs ``j``), then RoPE."""
    inv = gl.rsqrt(gl.sum(lo * lo + hi * hi, axis=0) / HEAD + EPS)
    lo = lo * inv * gain
    hi = hi * inv * gain
    if ROPE:
        c = gl.load(cos_sin + pos * HEAD + j)
        s = gl.load(cos_sin + pos * HEAD + HEAD // 2 + j)
        lo, hi = gl.fma(lo, c, -(hi * s)), gl.fma(hi, c, lo * s)
    return lo.to(gl.bfloat16), hi.to(gl.bfloat16)


@gluon.jit
def _pack(lo, hi):
    return (lo.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        hi.to(gl.int16, bitcast=True).to(gl.int32) << 16
    )


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
    """Head task ``task``: q head ``task``, or k head ``task - 16``.

    Args:
        seq_len: int32 ``L + 1`` for the token at position ``L``.
        slot: int32 KV cache slot of the token.
        k_cache, v_cache: BF16 ``[slots, 2, 64]``.
        cos_sin: FP32 ``[positions, 64]``: ``cos`` of the 32 pairs, then ``sin``
            (unused when ``ROPE`` is false).
    """
    mb.stamp(STAGE, trace.START)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    j = gl.arange(0, HEAD // 2, lanes)
    pos = gl.load(seq_len) - 1
    if task < Q_HEADS:
        (words,) = mb.poll_many(("qkvg",), (task * (HEAD // 2) + j,))
        mb.stamp(STAGE, trace.READY)
        lo, hi = _halves(words)
        lo, hi = _norm_rope(lo, hi, Q_GAIN, cos_sin, pos, j, ROPE)
        mb.put("qn", task * (HEAD // 2) + j, _pack(lo, hi))
    else:
        g = task - Q_HEADS
        kw, vw = mb.poll_many(
            ("qkvg", "qkvg"),
            ((K_ROW + g * HEAD) // 2 + j, (V_ROW + g * HEAD) // 2 + j),
        )
        mb.stamp(STAGE, trace.READY)
        lo, hi = _halves(kw)
        lo, hi = _norm_rope(lo, hi, 1.0, cos_sin, pos, j, ROPE)
        mb.put("kn", g * (HEAD // 2) + j, _pack(lo, hi))
        # The cache keeps the rotated k and the raw v.
        at = gl.load(slot).to(gl.int64) * KV_STRIDE + g * HEAD + 2 * j
        gl.store(k_cache + at, lo)
        gl.store(k_cache + at + 1, hi)
        vlo, vhi = _halves(vw)
        gl.store(v_cache + at, vlo.to(gl.bfloat16))
        gl.store(v_cache + at + 1, vhi.to(gl.bfloat16))
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
def attn_stage(
    mb,
    task,
    seq_len,
    table,
    k_cache,
    v_cache,
    SPLITS: gl.constexpr,
    WINDOW: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Attn task ``task`` = (KV head ``task // SPLITS``, split ``task % SPLITS``).

    Args:
        table: int32 page table of the row (``PAGE`` tokens a page).
        WINDOW: Attended positions of a sliding layer, 0 on a full layer.
    """
    gl.static_assert(schedule.rows(mb.plan) == 1, "attention serves one row")
    mb.stamp(STAGE, trace.START)
    g = task // SPLITS
    s = task % SPLITS
    nw: gl.constexpr = gl.num_warps()
    qk_mma: gl.constexpr = _layouts(nw)[0]
    pv_mma: gl.constexpr = _layouts(nw)[1]
    k_layout: gl.constexpr = _layouts(nw)[2]
    v_load: gl.constexpr = _layouts(nw)[3]
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 4)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 4)

    L = gl.load(seq_len) - 1
    if WINDOW == 0:
        n = L + 1
    else:
        n = gl.minimum(L + 1, WINDOW)
    per = (n + SPLITS - 1) // SPLITS
    lo = L + 1 - n + s * per
    hi = gl.minimum(lo + per, L + 1)

    # q of the 8 heads (rows 8 .. 15 zero), and the token's k and v.
    ql: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [nw, 1], [1, 0])
    qr = gl.arange(0, 16, gl.SliceLayout(1, ql))[:, None]
    qp = gl.arange(0, HEAD // 2, gl.SliceLayout(0, ql))[None, :]
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    tp = gl.arange(0, HEAD // 2, lanes)
    got_q, got_k, got_v = mb.poll_many(
        ("qn", "kn", "qkvg"),
        (
            (g * GROUP + gl.minimum(qr, GROUP - 1)) * (HEAD // 2) + qp,
            g * (HEAD // 2) + tp,
            (V_ROW + g * HEAD) // 2 + tp,
        ),
    )
    mb.stamp(STAGE, trace.READY)
    q_lo, q_hi = _halves(got_q)
    q = gl.reshape(gl.join(q_lo, q_hi), [16, HEAD])
    q = gl.where(
        gl.arange(0, 16, gl.SliceLayout(1, q.type.layout))[:, None] < GROUP, q, 0.0
    )
    q = gl.convert_layout(q.to(gl.bfloat16), q_layout)
    line: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    cur_k = gl.allocate_shared_memory(gl.bfloat16, [HEAD], line)
    cur_v = gl.allocate_shared_memory(gl.bfloat16, [HEAD], line)
    k_lo, k_hi = _halves(got_k)
    v_lo, v_hi = _halves(got_v)
    cur_k.store(gl.reshape(gl.join(k_lo, k_hi), [HEAD]).to(gl.bfloat16))
    cur_v.store(gl.reshape(gl.join(v_lo, v_hi), [HEAD]).to(gl.bfloat16))
    gl.barrier()
    k_t = cur_k.load(gl.SliceLayout(1, k_layout))
    v_t = cur_v.load(gl.SliceLayout(0, v_load))
    v_smem = gl.allocate_shared_memory(
        gl.bfloat16, [BN, HEAD], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )

    dk = gl.arange(0, HEAD, gl.SliceLayout(1, k_layout))
    dv = gl.arange(0, HEAD, gl.SliceLayout(0, v_load))
    m = gl.full([16], float("-inf"), gl.float32, gl.SliceLayout(1, qk_mma))
    l = gl.zeros([16], gl.float32, gl.SliceLayout(1, qk_mma))
    acc = gl.zeros([16, HEAD], gl.float32, pv_mma)
    for c0 in range(lo, hi, BN):
        jk = c0 + gl.arange(0, BN, gl.SliceLayout(0, k_layout))
        jv = c0 + gl.arange(0, BN, gl.SliceLayout(1, v_load))
        ok_k = jk < hi
        ok_v = jv < hi
        old_k = ok_k & (jk != L)
        old_v = ok_v & (jv != L)
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
        keys = gl.where((jk == L)[None, :], k_t[:, None], keys)
        values = gl.where((jv == L)[:, None], v_t[None, :], values)
        scores = gl.zeros([16, BN], gl.float32, qk_mma)
        scores = gl.amd.cdna4.mfma(q, keys, scores)
        valid = gl.convert_layout(ok_k, gl.SliceLayout(0, qk_mma))
        scores = gl.where(valid[None, :], scores * QK_SCALE, float("-inf"))
        m_new = gl.maximum(m, gl.max(scores, axis=1))
        m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m - m_safe)
        p = gl.exp2(scores - m_safe[:, None])
        l = l * alpha + gl.sum(p, axis=1)
        gl.barrier()
        v_smem.store(values)
        gl.barrier()
        pv = gl.zeros([16, HEAD], gl.float32, pv_mma)
        pv = gl.amd.cdna4.mfma(
            gl.convert_layout(p.to(gl.bfloat16), p_layout), v_smem.load(v_layout), pv
        )
        acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, pv_mma))[:, None] + pv
        m = m_new
    mb.stamp(STAGE, trace.COMPUTED)
    l_acc = gl.convert_layout(l, gl.SliceLayout(1, pv_mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse = gl.where(l > 0, m + gl.log2(gl.where(l > 0, l, 1.0)), float("-inf"))
    r = gl.arange(0, 16, gl.SliceLayout(1, pv_mma))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, pv_mma))
    part = ((s * KV_HEADS + g) * GROUP + r[:, None]) * HEAD + c[None, :]
    mb.put("part", part, out.to(gl.int32, bitcast=True), mask=(r < GROUP)[:, None])
    rl = gl.arange(0, 16, gl.SliceLayout(1, qk_mma))
    mb.put(
        "lse",
        s * Q_HEADS + g * GROUP + rl,
        lse.to(gl.int32, bitcast=True),
        mask=rl < GROUP,
    )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def merge_stage(mb, task, SPLITS: gl.constexpr, STAGE: gl.constexpr):
    """Merge task ``task`` = q head ``h``: the LSE merge, then the gated head
    norm of the regular path's output projection."""
    mb.stamp(STAGE, trace.START)
    h = task
    g = h // GROUP
    hh = h % GROUP
    nw: gl.constexpr = gl.num_warps()
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [nw, 1], [1, 0])
    s = gl.arange(0, SPLITS, gl.SliceLayout(1, ml))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, ml))
    part = ((s[:, None] * KV_HEADS + g) * GROUP + hh) * HEAD + c[None, :]
    got_p, got_l, got_g = mb.poll_many(
        ("part", "lse", "qkvg"),
        (part, s * Q_HEADS + h, (GATE_ROW + h * HEAD + c) // 2),
    )
    mb.stamp(STAGE, trace.READY)
    lse = got_l.to(gl.float32, bitcast=True)
    top = gl.max(lse, axis=0)
    w = gl.where(lse == float("-inf"), 0.0, gl.exp2(lse - top))
    p = got_p.to(gl.float32, bitcast=True)
    out = gl.sum(p * w[:, None], axis=0) / gl.sum(w, axis=0)
    a = out.to(gl.bfloat16).to(gl.float32)
    inv = gl.rsqrt(gl.sum(a * a, axis=0) / HEAD + EPS)
    z = a * inv * mx.sigmoid(mx.bf16_half(got_g, c))
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "z", h * (HEAD // 2), z)
    mb.stamp(STAGE, trace.END)
