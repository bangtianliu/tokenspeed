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

"""Whole-layer decode monokernels.

A monokernel runs one decoder layer's decode step as one persistent launch.
Every stage of the layer (norms, projections, attention, MoE) is a set of
tasks placed on fixed CTAs. The stages hand data to each other through tagged
mailboxes in a scratch buffer, not through kernel boundaries.
See ``docs/design/monokernel.md`` for the design, and ``README.md`` in this
package for adding a model's monokernel.

The runtime packs a layer once after weight loading
(:func:`prepare_qwen4_exp_delta_layer`) and then runs it per step
(:func:`qwen4_exp_delta_layer_decode`). The prepare function returns None
when the platform has no monokernel for the layer, so the caller keeps its
regular path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "FusedDecodeLayer",
    "muse_spark_draft_context_decode",
    "muse_spark_draft_head_decode",
    "muse_spark_draft_layer_decode",
    "muse_spark_layer_decode",
    "prepare_muse_spark_draft_context",
    "prepare_muse_spark_draft_head",
    "prepare_muse_spark_draft_layer",
    "prepare_muse_spark_experts",
    "prepare_muse_spark_layer",
    "qwen4_exp_mtp_front",
    "select_qwen4_exp_mtp_front",
    "prepare_qwen4_exp_delta_layer",
    "prepare_qwen4_exp_qsa_layer",
    "qwen4_exp_delta_layer_decode",
    "qwen4_exp_qsa_layer_decode",
]


@dataclass(frozen=True)
class FusedDecodeLayer:
    """A decoder layer packed for its monokernel.

    Attributes:
        decode: The selected kernel. The ``*_decode`` function calls it.
        weights: The kernel's packed weights (opaque to the caller).
        scratch_pairs: int64 mailbox pairs the kernel needs. One scratch
            buffer serves every layer of a step.
        late_weights: int32 words of the packed weights that the layer's
            launch reads last, for the launch before it to warm (``warm`` of
            :func:`qwen4_exp_delta_layer_decode`), or None.
    """

    decode: Any
    weights: Any
    scratch_pairs: int
    late_weights: torch.Tensor | None


def _select(mode: str, max_rows: int, override: str | None):
    signature = format_signature(h=dense_tensor_format(torch.bfloat16))
    return select_kernel(
        "monokernel", mode, signature, traits={"rows": max_rows}, override=override
    )


def _prepare(kind: str, max_rows: int, tensors, dense_tiled: bool, override):
    try:
        prepare = _select(f"{kind}_prepare", max_rows, override)
        decode = _select(f"{kind}_decode", max_rows, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    weights, scratch_pairs, late_weights = prepare(
        tensors=tensors, dense_tiled=dense_tiled, max_rows=max_rows
    )
    return FusedDecodeLayer(
        decode=decode,
        weights=weights,
        scratch_pairs=scratch_pairs,
        late_weights=late_weights,
    )


def prepare_qwen4_exp_delta_layer(
    *,
    max_rows: int,
    tensors: dict[str, Any],
    dense_tiled: bool,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack a Qwen3.8-Flash-Next Gated DeltaNet layer for its monokernel.

    Args:
        max_rows: The most request rows (one token each) a decode step will
            run the layer with. A step may run fewer.
        tensors: The layer's weights by role (see
            ``tokenspeed_kernel_amd...qwen4_exp.weights.pack_delta_layer``).
            ``tensors["experts"]`` names the MoE block's format: ``"mxfp4"``
            (``pack_moe``'s tensors) or ``"bf16"`` (``pack_moe_bf16``'s).
        dense_tiled: Whether the dense MXFP4 weights are already in N16 tile
            order (the MFMA linear layout) rather than checkpoint order.
        override: Optional registered kernel name to force.

    Returns:
        The packed layer, or None when no monokernel serves ``max_rows`` here.
    """
    return _prepare("qwen4_exp_delta_layer", max_rows, tensors, dense_tiled, override)


def qwen4_exp_delta_layer_decode(
    layer: FusedDecodeLayer,
    h: torch.Tensor,
    h_out: torch.Tensor,
    *,
    chain: int,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    conv_out: torch.Tensor,
    conv_out_rows: torch.Tensor,
    replay: torch.Tensor | None,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    ple: dict[str, torch.Tensor] | None,
    warm: torch.Tensor | None,
) -> None:
    """One decode step, or one speculative verify, of a packed Gated
    DeltaNet layer.

    Args:
        layer: From :func:`prepare_qwen4_exp_delta_layer`.
        h: Hyper-connection input streams of ``rows`` request rows (at most
            the packed ``max_rows``). The shape is BF16 ``[rows, 4 * hidden]``,
            or ``[rows, hidden]``: one row per request that stands for all
            four streams (the first layer's embedding).
        h_out: Output streams, same shape. ``h_out`` must not alias ``h``.
        chain: Consecutive rows per request: 1 for decode, ``k + 1`` for a
            verify of ``k`` draft tokens (the rows chain the layer's state).
        conv_state, ssm_state: The layer's state pools (pages first).
        state_in, state_out: int32 ``[requests]`` input and output state
            pages, -1 for padding requests. The state after a request's last
            token goes to its output page (-1 in a verify).
        conv_out, conv_out_rows: Each row's conv window after its token goes
            to row ``conv_out_rows[row]`` (int32 ``[rows]``, -1: none) of
            ``conv_out``. In decode, these are ``conv_state`` and
            ``state_out``. In a verify, these are the verify scratch and its
            rows.
        replay: None, or the layer's replay payload a verify fills (each
            row's ``[k | v | a | b]``).
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``.
        step: int32 counter (``[1]``, or ``[2]`` for the launch that
            advances it) that changes between steps (CUDA graph safe: the
            kernel reads it on device).
        layer_index: Decoder layer index, below 64.
        ple: For the layer with a PLE block (packed with ``tensors["ple"]``),
            its per-step operands by name:

            * ``token``: each row's token ``[rows]``.
            * ``context``: the int64 context pool ``[pages, 2]``.
            * ``conv_state``: the conv state pool
              ``[pages, hc_count * hidden, 9]``.
            * ``page_in``: int32 ``[requests]``, each request's input page.
            * ``page_out``: int32 ``[rows]``, each row's output row.
            * ``context_out``: ``[*, 2]``.
            * ``conv_out``: ``[*, hc_count * hidden, 9]``.

            The state after each row's token goes to its output row of the
            two outputs. In decode, these are the pools and the output pages.
            In a verify, these are the verify scratch and its rows. ``ple``
            is None for any other layer. The kernel does the n-gram lookup
            itself.
        warm: The next launch's ``late_weights`` (None for the last layer).
            The launch may read them into the memory-side cache while it has
            little HBM traffic, so the next launch reads them sooner. It
            leaves them unchanged.
    """
    layer.decode(
        weights=layer.weights,
        h=h,
        h_out=h_out,
        chain=chain,
        conv_state=conv_state,
        ssm_state=ssm_state,
        state_in=state_in,
        state_out=state_out,
        conv_out=conv_out,
        conv_out_rows=conv_out_rows,
        replay=replay,
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        ple=ple,
        warm=warm,
    )


def prepare_qwen4_exp_qsa_layer(
    *,
    max_rows: int,
    tensors: dict[str, Any],
    dense_tiled: bool,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack a Qwen3.8-Flash-Next QSA attention layer for its monokernel.

    Args and returns as in :func:`prepare_qwen4_exp_delta_layer`. ``tensors``
    follows ``...qwen4_exp.weights.pack_qsa_layer``.
    """
    return _prepare("qwen4_exp_qsa_layer", max_rows, tensors, dense_tiled, override)


def qwen4_exp_qsa_layer_decode(
    layer: FusedDecodeLayer,
    h: torch.Tensor,
    h_out: torch.Tensor,
    *,
    chain: int,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    seq_len: torch.Tensor,
    full_table: torch.Tensor,
    page_size: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qsa_table: torch.Tensor,
    recent_table: torch.Tensor,
    compressed: torch.Tensor,
    raw_key: torch.Tensor,
    rope_position: torch.Tensor,
    staging: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    draft: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    accepted: torch.Tensor | None,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    mixed_out: torch.Tensor | None,
    advance_step: bool,
) -> None:
    """One decode step, one speculative verify, or one MTP draft step of a
    packed QSA attention layer.

    Args:
        layer, h, h_out, chain, scratch, step, layer_index: As in
            :func:`qwen4_exp_delta_layer_decode`.
        cos_sin: Rotary cache, FP32 ``[max_position, 64]``.
        positions: int64 ``[3, rows]`` MRoPE positions of each row's token.
        seq_len: int32 ``[requests]``: each request's length including its
            last token (token ``t`` of a request's ``chain`` rows is at
            ``seq_len - chain + t``).
        full_table, page_size: The requests' KV page tables ``[requests,
            pages]`` and the page size.
        k_cache, v_cache: The layer's ``[slots, kv_heads, head_dim]`` caches.
        qsa_table, recent_table: The requests' QSA compressed and recent page
            tables ``[requests, pages]``.
        compressed, raw_key, rope_position: The layer's QSA indexer caches.
        staging: None in decode. In a verify, the rows of TokenSpeed's
            ``QSAVerifyState`` ``(this layer's raw index keys [rows, dim],
            positions [rows, 3], logical positions [rows], recent locations
            [rows])`` that the kernel writes instead of the recent ring.
        draft: None, except in an MTP draft step (``chain`` 1): the requests'
            QSA draft scratch ``(raw keys [rows, 4, 1, dim], group-start
            positions [rows, 3], position tags [rows, 4])``. The kernel stages
            the index keys there instead of the recent ring, and its
            compression reads the members that earlier draft steps staged.
        accepted: None, except in an MTP draft catch-up step (the draft's
            first step over one request's verified rows): the request's int32
            accepted length ``[1]``. Rows at or past it write K / V but no
            index keys. As TokenSpeed's draft narrowing, only the live row
            (the last accepted token's) runs the attention and the rest of
            the layer, so ``h_out`` and ``mixed_out`` hold that one row.
        mixed_out: When :func:`prepare_qwen4_exp_qsa_layer` packed the layer
            with the model's final mixer (``tensors["final_mixer"]``, the last
            layer), the BF16 mixer output (the LM head's input), with the
            rows of ``h_out``. Else ``None``.
        advance_step: Whether the launch moves ``step`` on when it ends (the
            last launch of a forward); ``step`` is then int32 ``[2]`` (the
            step and a count that stays 0 between launches).
    """
    layer.decode(
        weights=layer.weights,
        h=h,
        h_out=h_out,
        chain=chain,
        cos_sin=cos_sin,
        positions=positions,
        seq_len=seq_len,
        full_table=full_table,
        page_size=page_size,
        k_cache=k_cache,
        v_cache=v_cache,
        qsa_table=qsa_table,
        recent_table=recent_table,
        compressed=compressed,
        raw_key=raw_key,
        rope_position=rope_position,
        staging=staging,
        draft=draft,
        accepted=accepted,
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        mixed_out=mixed_out,
        advance_step=advance_step,
    )


def prepare_muse_spark_layer(
    *, tensors: dict[str, Any], max_rows: int, override: str | None = None
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's Muse Spark decoder layer for its monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.weights.pack_layer``).
        max_rows: The most token rows of one request a launch will serve:
            1 for decode, ``k + 1`` for a speculative verify of ``k`` draft
            tokens. The scratch is sized for it.
        override: Optional registered kernel name to force.

    Returns:
        The packed layer, or None when no monokernel serves ``max_rows``
        here.
    """
    try:
        prepare = _select("muse_spark_layer_prepare", max_rows, override)
        decode = _select("muse_spark_layer_decode", max_rows, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    weights, scratch_pairs, late_weights = prepare(tensors=tensors, max_rows=max_rows)
    return FusedDecodeLayer(
        decode=decode,
        weights=weights,
        scratch_pairs=scratch_pairs,
        late_weights=late_weights,
    )


def prepare_muse_spark_experts(
    *, tensors: dict[str, Any], override: str | None = None
) -> tuple[torch.Tensor, ...] | None:
    """Prepare one TP8 rank's slices of all 256 Muse Spark experts for the
    monokernel, as the regular gluon MoE prepares them on an EP1 server. An
    EP8 server, whose ranks hold whole experts, gathers the slices first.

    Args:
        tensors: The slices in the MoE loader's layout by role (see
            ``tokenspeed_kernel_amd...muse_spark.weights.prepare_experts``).
            The function consumes them.
        override: Optional registered kernel name to force.

    Returns:
        ``prepare_muse_spark_layer``'s ``experts`` tuple, or None when no
        monokernel serves the layer here.
    """
    try:
        prepare = _select("muse_spark_experts_prepare", 1, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    return prepare(tensors=tensors)


def muse_spark_layer_decode(
    layer: FusedDecodeLayer,
    *,
    h: torch.Tensor,
    a: torch.Tensor,
    h_out: torch.Tensor,
    a_out: torch.Tensor,
    seq_len: torch.Tensor,
    slot: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    layer_index: int,
    chain: int,
    tap: torch.Tensor | None,
    tap_slot: int | None,
    compile_only: bool,
) -> None:
    """One step of a packed Muse Spark layer for the rows of one or more
    requests (a decode row each, or a speculative verify's consecutive
    tokens), on every rank of the TP8 group at once (the launches exchange
    values).

    Args:
        layer: From :func:`prepare_muse_spark_layer`.
        h, h_out: FP32 ``[rows, 1024]``: the rank's owned residual columns
            (``1024 rank ..``) before and after the layer.
        a, a_out: BF16 ``[rows, 8192]``: the layer's attention input (normed
            with its input norm) and the next layer's. Outputs alias no
            input.
        seq_len, slot: int32 ``[rows]``: each row's length including its
            token, and the token's KV cache slot. Row ``t`` of request ``b``
            is the token at position ``seq_len[b chain] - 1 + t``.
        page_table: int32 ``[requests, pages]`` (or ``[pages]`` for one
            request): each request's page table of the layer's cache group
            (64 tokens a page, absolute positions).
        k_cache, v_cache: The layer's BF16 ``[slots, 2, 64]`` caches.
        cos_sin: FP32 ``[positions, 64]`` rotary table (cos, then sin).
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``,
            on the Iris symmetric heap (the same offset on every rank).
        step: int32 ``[1]`` counter that changes once per forward (CUDA graph
            safe: the kernel reads it on device).
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        layer_index: Decoder layer index, below 64. The launches of one
            forward alternate its parity.
        chain: Tokens a request: 1 for decode, ``k + 1`` for a verify of
            ``k`` draft tokens. The rows are whole requests.
        tap: None, or a BF16 ``[rows+, slots, 8192]`` buffer on the Iris
            symmetric heap: the launch also stores the residual after the
            layer (BF16) into slot ``tap_slot`` of every rank's copy, the
            tap a draft reads.
        tap_slot: The tap's slot, or None without ``tap``.
        compile_only: Only compile the launch for these operands, without
            running it. A launch polls its peers' values for a few seconds
            before it traps, so a rank that compiles much longer than its
            peers stalls them: compile on every rank, synchronize the ranks,
            then launch.
    """
    layer.decode(
        weights=layer.weights,
        h=h,
        a=a,
        h_out=h_out,
        a_out=a_out,
        seq_len=seq_len,
        slot=slot,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        layer_index=layer_index,
        chain=chain,
        tap=tap,
        tap_slot=tap_slot,
        compile_only=compile_only,
    )


def prepare_muse_spark_draft_context(
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    max_rows: int,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's DFlash2 context write (``fc`` and ``hidden_norm``
    of the target's taps, then every draft layer's K and V into the draft
    cache) for its monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.draft_context
            .pack_draft_context``).
        rank: The rank in the TP8 group.
        max_rows: The verify rows of one request.
        override: Optional registered kernel name to force.

    Returns:
        The packed write, or None when no monokernel serves ``max_rows``
        here.
    """
    try:
        prepare = _select("muse_spark_draft_context_prepare", max_rows, override)
        decode = _select("muse_spark_draft_context_decode", max_rows, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    weights, scratch_pairs, late_weights = prepare(
        tensors=tensors, rank=rank, max_rows=max_rows
    )
    return FusedDecodeLayer(
        decode=decode,
        weights=weights,
        scratch_pairs=scratch_pairs,
        late_weights=late_weights,
    )


def muse_spark_draft_context_decode(
    context: FusedDecodeLayer,
    *,
    taps: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_caches: tuple[torch.Tensor, ...],
    v_caches: tuple[torch.Tensor, ...],
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    compile_only: bool,
) -> None:
    """The DFlash2 context write of one request's verify rows, on every rank
    of the TP8 group at once (the launches exchange the norm).

    Args:
        context: From :func:`prepare_muse_spark_draft_context`.
        taps: BF16 ``[rows, 40960]``: the target's tapped hidden states.
        positions: int32 or int64 ``[rows]``: the rows' positions.
        slots: int32 or int64 ``[rows]``: the rows' draft cache slots.
        k_caches, v_caches: The draft layers' BF16 ``[slots, 1, 128]``
            caches of the rank's KV head.
        cos_sin: FP32 ``[positions, 128]`` RoPE table, cos then sin.
        scratch: int64 mailbox scratch of at least ``context.scratch_pairs``
            on the Iris symmetric heap.
        step: int32 ``[1]`` counter that changes from launch to launch.
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        compile_only: Only compile the launch for these operands (compile on
            every rank, synchronize the ranks, then launch).
    """
    context.decode(
        weights=context.weights,
        taps=taps,
        positions=positions,
        slots=slots,
        k_caches=k_caches,
        v_caches=v_caches,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        compile_only=compile_only,
    )


def prepare_muse_spark_draft_head(
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    max_rows: int,
    scale: float,
    softcap: float,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's DFlash2 draft head (the vocabulary-parallel top 16
    of each draft step and the candidate selector's scores) for its
    monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.draft_head.pack_draft_head``).
        rank: The rank in the TP8 group.
        max_rows: The draft steps of one request's block.
        scale: The logits' multiplier.
        softcap: The final logit softcap.
        override: Optional registered kernel name to force.

    Returns:
        The packed head, or None when no monokernel serves ``max_rows`` here.
    """
    try:
        prepare = _select("muse_spark_draft_head_prepare", max_rows, override)
        decode = _select("muse_spark_draft_head_decode", max_rows, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    weights, scratch_pairs, late_weights = prepare(
        tensors=tensors, rank=rank, max_rows=max_rows, scale=scale, softcap=softcap
    )
    return FusedDecodeLayer(
        decode=decode,
        weights=weights,
        scratch_pairs=scratch_pairs,
        late_weights=late_weights,
    )


def muse_spark_draft_head_decode(
    head: FusedDecodeLayer,
    *,
    x: torch.Tensor,
    first_token: torch.Tensor,
    anchor: torch.Tensor,
    cand_ids: torch.Tensor,
    unary: torch.Tensor,
    scores: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    compile_only: bool,
) -> None:
    """The DFlash2 draft head of one request's draft steps, on every rank of
    the TP8 group at once (the launches exchange candidates).

    Args:
        head: From :func:`prepare_muse_spark_draft_head`.
        x: BF16 ``[steps, 8192]``: the draft's final-norm output of the
            block's rows 1 .. 7.
        first_token: int32 ``[1]``: the block's first token.
        anchor: int64 ``[1]`` out: that token clamped to the vocabulary
            (step 0's predecessor, for the walk).
        cand_ids: int64 ``[steps, 16]`` out: each step's candidates, by
            descending unary score (the lower token id first on a tie).
        unary: FP32 ``[steps, 16]`` out: their softcapped logits.
        scores: FP32 ``[steps, 16, 16]`` out: the selector's edge scores
            from the previous step's candidate ``p`` to this step's ``c``.
        scratch: int64 mailbox scratch of at least ``head.scratch_pairs`` on
            the Iris symmetric heap.
        step: int32 ``[1]`` counter that changes from launch to launch
            (CUDA graph safe).
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        compile_only: Only compile the launch for these operands (compile on
            every rank, synchronize the ranks, then launch).
    """
    head.decode(
        weights=head.weights,
        x=x,
        first_token=first_token,
        anchor=anchor,
        cand_ids=cand_ids,
        unary=unary,
        scores=scores,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        compile_only=compile_only,
    )


def prepare_muse_spark_draft_layer(
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    max_rows: int,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's layer of Muse Spark's DFlash2 draft for its
    monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.draft_layer.pack_draft_layer``).
        rank: The rank in the TP8 group.
        max_rows: The rows of one draft block forward (one request's block).
        override: Optional registered kernel name to force.

    Returns:
        The packed layer, or None when no monokernel serves ``max_rows``
        here.
    """
    try:
        prepare = _select("muse_spark_draft_layer_prepare", max_rows, override)
        decode = _select("muse_spark_draft_layer_decode", max_rows, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    weights, scratch_pairs, late_weights = prepare(
        tensors=tensors, rank=rank, max_rows=max_rows
    )
    return FusedDecodeLayer(
        decode=decode,
        weights=weights,
        scratch_pairs=scratch_pairs,
        late_weights=late_weights,
    )


def muse_spark_draft_layer_decode(
    layer: FusedDecodeLayer,
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    out: torch.Tensor | None,
    final_gain: torch.Tensor | None,
    normed: torch.Tensor | None,
    seq_len: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    layer_index: int,
    compile_only: bool,
) -> None:
    """One step of a packed DFlash2 draft layer for one request's block of
    rows, on every rank of the TP8 group at once (the launches exchange
    values).

    Args:
        layer: From :func:`prepare_muse_spark_draft_layer`.
        x: BF16 ``[rows, 8192]``: this rank's partial of the layer input.
        residual: BF16 ``[rows, 8192]``: the residual before the layer.
        residual_out: BF16 ``[rows, 8192]``: the rank stores its owned
            columns (``1024 rank ..``) of the residual after the attention
            sublayer; the other columns are left as they are.
        out: BF16 ``[rows, 8192]``: this rank's partial of the layer
            output; None for the draft's last layer, whose launch also runs
            the draft's final norm.
        final_gain: None, or (the last layer) FP32 ``[1024]``: the final
            norm's gain of the rank's owned columns.
        normed: None, or (the last layer) BF16 ``[rows, 8192]``: the final
            norm's output, the whole rows on every rank.
        seq_len: int32 ``[1]``: the request's length after the block.
        positions: int32 ``[rows]``: the block rows' positions.
        slots: int32 or int64 ``[rows]``: the block rows' cache slots.
        page_table: int32 page table of the request (64 tokens a page).
        k_cache, v_cache: BF16 ``[slots, 1, 128]``: the layer's cache.
        cos_sin: FP32 ``[positions, 128]`` rotary table, cos then sin.
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``
            on the Iris symmetric heap (the same offset on every rank).
        step: int32 ``[1]`` counter that changes once per draft forward
            (CUDA graph safe: the kernel reads it on device).
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        layer_index: Draft layer index, below 5.
        compile_only: Only compile the launch for these operands, without
            running it: compile on every rank, synchronize the ranks, then
            launch (a launch polls its peers for a few seconds before it
            traps).
    """
    layer.decode(
        weights=layer.weights,
        x=x,
        residual=residual,
        residual_out=residual_out,
        out=out,
        final_gain=final_gain,
        normed=normed,
        seq_len=seq_len,
        positions=positions,
        slots=slots,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        layer_index=layer_index,
        compile_only=compile_only,
    )


def select_qwen4_exp_mtp_front(max_rows: int, override: str | None = None) -> Any:
    """The kernel of :func:`qwen4_exp_mtp_front` for up to ``max_rows``
    rows, or None when no kernel serves them here."""
    signature = format_signature(hidden=dense_tensor_format(torch.bfloat16))
    try:
        return select_kernel(
            "monokernel",
            "qwen4_exp_mtp_front",
            signature,
            traits={"rows": max_rows},
            override=override,
        )
    except NoKernelFoundError:
        if override is not None:
            raise
        return None


def qwen4_exp_mtp_front(
    kernel: Any,
    input_ids: torch.Tensor,
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    norm_embedding: torch.Tensor,
    norm_hidden: torch.Tensor,
    fc_embedding: torch.Tensor,
    fc_hidden: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    """The Qwen3.8-Flash-Next MTP draft input of a decode step's rows, in one
    launch: ``fc_hidden(norm(hidden)) + fc_embedding(norm(embedding[ids]))``
    for each of the 4 hyper-connection streams.

    Args:
        kernel: From :func:`select_qwen4_exp_mtp_front`.
        input_ids: int ``[rows]`` tokens (``rows`` at most the selected
            ``max_rows``).
        embedding: BF16 ``[vocab, H]`` token embedding (unsharded).
        hidden: BF16 ``[rows, 4 H]`` captured hidden streams.
        norm_embedding, norm_hidden: Gemma RMSNorm weights ``[H]`` and
            ``[4 H]`` (the multiplier is ``1 + weight``); the hidden norm
            spans all 4 streams.
        fc_embedding, fc_hidden: BF16 ``[H, H]`` linear weights
            (``[out, in]``), applied to the embedding and to each stream.
        eps: The norms' epsilon.
        out: BF16 ``[rows, 4 H]`` result, stream major like ``hidden``.
    """
    kernel(
        input_ids,
        embedding,
        hidden,
        norm_embedding,
        norm_hidden,
        fc_embedding,
        fc_hidden,
        eps,
        out,
    )


# Register the backends (side-effect imports).
import tokenspeed_kernel.ops.monokernel.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.monokernel.triton  # noqa: E402,F401
