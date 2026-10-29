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

"""Expert routing of one Muse Spark rank (one row).

The router multiplies the FP32 pre-FFN norm output ``u = h' s g`` (``s =
rsqrt(mean(h'^2) + 1e-5)``, ``g = bf16(w + 1)``) by the FP32 router weight.
Rank ``r`` holds ``h'`` only for its 1024 owned columns, so it takes the
matching 1024 columns of the router weight (1 MiB, not the replicated 8 MiB)
and every rank adds the 8 partial logits. ``u`` is not rounded on this path,
and ``h' = beta b inv + alpha h`` (see attn_tail) is linear in the
post-attention norm's ``inv``. So the router multiplies ``beta b g`` and
``alpha h g``, and top-8 applies the scalars ``inv`` and ``s``: the router
waits for neither norm exchange, only for ``b``.

* router, 16 tasks of 16 experts: this rank's partial logits of ``alpha h
  g`` (from the launch's inputs, at once) and of ``beta b g``, put into
  every rank's ``lg2`` and ``lg`` slots of this rank.
* topk, 1 task: the rank-order sum of ``inv lg + lg2`` times ``s`` (both
  from ``ss1``) times ``output_multiplier``, the sigmoid
  scores, and the 8 experts of the largest ``score + bias`` (8 argmax passes,
  a tie to the lowest id). The weights are the bias-free scores of the 8,
  divided by their sum. Every rank gets the same bits, published in ``route``
  as 8 ids, then 8 FP32 weights.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.attn_tail import (
    OWNED,
    norm_scales,
    sums_index,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

EXPERTS = gl.constexpr(256)
TOP_K = gl.constexpr(8)
ROUTER_TASKS = gl.constexpr(16)
ROUTER_ROWS = gl.constexpr(16)  # experts a router task
ROUTER_SCALE = gl.constexpr(0.17677669529663687)  # output_multiplier

PEERS = ("lg", "lg2")


def regions() -> tuple[tuple[str, int], ...]:
    return (
        # FP32 partial logits [source rank][expert] of beta b and of alpha h
        ("lg", int(WORLD) * int(EXPERTS)),
        ("lg2", int(WORLD) * int(EXPERTS)),
        ("route", 2 * int(TOP_K)),  # int32 ids, then FP32 weights
    )


def stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    router, topk = placement(bases, ("router", "topk"))
    return (
        Stage("router", int(ROUTER_TASKS), router, reads=("bq",), writes=("lg", "lg2")),
        Stage("topk", 1, topk, reads=("lg", "lg2", "ss1"), writes=("route",)),
    )


@gluon.jit
def router_stage(
    mb, task, w, pre_gain, h, alpha, beta, RANK: gl.constexpr, STAGE: gl.constexpr
):
    """Router task ``task``: experts ``16 task ..``.

    ``h' = beta b inv + alpha h`` (see attn_tail), so the logits of ``h' g``
    are ``inv`` times those of ``beta b g`` plus those of ``alpha h g``. The
    second part needs only the launch's inputs, so it goes out first; the
    first needs only ``b``, not the norm exchange. Top-8 applies ``inv``.

    Args:
        w: This rank's router columns, FP32 ``[256, 1024]`` (row ``e`` is
            expert ``e``'s weights for the owned columns).
        pre_gain: FP32 ``bf16(w + 1)`` of the pre-FFN norm, owned columns.
        h, alpha, beta: The owned FP32 residual and residual-gate
            coefficients (as tail_stages takes them).
    """
    mb.stamp(STAGE, trace.START)
    nw: gl.constexpr = gl.num_warps()
    # The waves split the columns: each wave polls only its 128 values of
    # h' (fewer polls of the same lines), and one sum crosses the waves.
    wl: gl.constexpr = gl.BlockedLayout([1, 4], [2, 32], [1, nw], [1, 0])
    e = task * ROUTER_ROWS + gl.arange(0, ROUTER_ROWS, gl.SliceLayout(1, wl))
    k = gl.arange(0, OWNED, gl.SliceLayout(0, wl))
    weights = gl.load(w + e[:, None] * OWNED + k[None, :])
    gain = gl.load(pre_gain + k)
    ahg = gl.load(alpha + k) * gl.load(h + k) * gain
    bg = gl.load(beta + k) * gain
    word2 = gl.sum(weights * ahg[None, :], axis=1).to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer("lg2", p, RANK * EXPERTS + e, word2)
    (got,) = mb.poll_many(("bq",), (k,))
    mb.stamp(STAGE, trace.READY)
    v = got.to(gl.float32, bitcast=True) * bg
    partial = gl.sum(weights * v[None, :], axis=1)
    mb.stamp(STAGE, trace.COMPUTED)
    word = partial.to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer("lg", p, RANK * EXPERTS + e, word)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def topk_stage(mb, bias, STAGE: gl.constexpr):
    """The top-8 of the summed logits (``bias``: FP32 ``[256]``)."""
    mb.stamp(STAGE, trace.START)
    nw: gl.constexpr = gl.num_warps()
    ll: gl.constexpr = gl.BlockedLayout([WORLD, 1], [1, 64], [1, nw], [0, 1])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, ll))
    e = gl.arange(0, EXPERTS, gl.SliceLayout(0, ll))
    # Every wave holds all 256 experts (4 a lane), so each argmax below
    # reduces within a wave: no LDS round trip and barrier per pass.
    wl: gl.constexpr = gl.BlockedLayout([EXPERTS // 64], [64], [nw], [0])
    ew = gl.arange(0, EXPERTS, wl)
    bias_w = gl.load(bias + ew)  # before the poll, not after it
    at = src[:, None] * EXPERTS + e[None, :]
    # The alpha h part goes out at the router's start: poll it first, alone.
    (got2,) = mb.poll_many(("lg2",), (at,))
    got, ss = mb.poll_many(("lg", "ss1"), (at, sums_index(nw)))
    mb.stamp(STAGE, trace.READY)
    # The norm scales, from the attention tail's partial sums.
    inv, s = norm_scales(ss)
    parts = got.to(gl.float32, bitcast=True)
    parts2 = got2.to(gl.float32, bitcast=True)
    logit = gl.zeros([EXPERTS], gl.float32, gl.SliceLayout(0, ll))
    for p in gl.static_range(WORLD):
        mine = src[:, None] == p
        logit += gl.fma(
            gl.sum(gl.where(mine, parts, 0.0), axis=0),
            inv,
            gl.sum(gl.where(mine, parts2, 0.0), axis=0),
        )
    logit = gl.convert_layout(logit * s, wl)
    # An exact division: the scores decide which experts a near tie picks.
    score = 1.0 / (1.0 + gl.exp(-(logit * ROUTER_SCALE)))
    choice = score + bias_w
    slot = gl.arange(0, TOP_K, gl.BlockedLayout([1], [64], [nw], [0]))
    ids = gl.zeros([TOP_K], gl.int32, slot.type.layout)
    picked = gl.zeros([TOP_K], gl.float32, slot.type.layout)
    for k in gl.static_range(TOP_K):
        best = gl.max(choice, axis=0)
        first = gl.min(gl.where(choice == best, ew, EXPERTS), axis=0)
        ids = gl.where(slot == k, first, ids)
        picked = gl.where(
            slot == k, gl.sum(gl.where(ew == first, score, 0.0), axis=0), picked
        )
        choice = gl.where(ew == first, float("-inf"), choice)
    total = gl.sum(picked, axis=0)
    weight = picked / gl.where(total == 0.0, 1.0, total)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.put("route", slot, ids)
    mb.put("route", TOP_K + slot, weight.to(gl.int32, bitcast=True))
    mb.stamp(STAGE, trace.END)
