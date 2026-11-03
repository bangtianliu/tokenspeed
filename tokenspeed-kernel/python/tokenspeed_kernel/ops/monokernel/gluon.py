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

"""Shims that register the AMD Gluon monokernels."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.qwen4_exp import (
        layer as _qwen4_exp_layer,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.qwen4_exp import (
        weights as _qwen4_exp_weights,
    )
except ImportError:
    _qwen4_exp_layer = None

try:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        layer as _muse_spark_layer,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        weights as _muse_spark_weights,
    )
except ImportError:
    _muse_spark_layer = None

try:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        draft_context as _muse_spark_draft_context,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        draft_head as _muse_spark_draft_head,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        draft_layer as _muse_spark_draft_layer,
    )
except ImportError:
    _muse_spark_draft_context = None
    _muse_spark_draft_head = None
    _muse_spark_draft_layer = None


if _qwen4_exp_layer is not None:
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _SIGNATURES = format_signatures("h", "dense", {torch.bfloat16})

    def _pack_moe(t: dict[str, Any], dense_tiled: bool) -> Any:
        """Pop the MoE block's tensors from ``t`` and pack them, in the
        format that ``t["experts"]`` names (``layer.EXPERTS``)."""
        experts = t.pop("experts")
        if experts == "bf16":
            return _qwen4_exp_weights.pack_moe_bf16(
                router_w=t.pop("router_w"),
                shared_gate_w=t.pop("shared_gate_w"),
                w13=t.pop("w13"),
                w2=t.pop("w2"),
                shared_gu=t.pop("shared_gu"),
                shared_dn=t.pop("shared_dn"),
            )
        if experts != "mxfp4":
            raise ValueError(f"unknown experts format {experts!r}")
        return _qwen4_exp_weights.pack_moe(
            router_w=t.pop("router_w"),
            shared_gate_w=t.pop("shared_gate_w"),
            banks=t.pop("banks"),
            shared_gu=t.pop("shared_gu"),
            shared_dn=t.pop("shared_dn"),
            dense_tiled=dense_tiled,
        )

    _TRAITS = {"rows": frozenset(range(1, _qwen4_exp_layer.MAX_ROWS + 1))}

    @register_kernel(
        "monokernel",
        "qwen4_exp_delta_layer_prepare",
        name="gluon_qwen4_exp_delta_layer_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_delta_layer_prepare_gfx950(
        *, tensors: dict[str, Any], dense_tiled: bool, max_rows: int
    ) -> tuple[Any, int, torch.Tensor]:
        t = dict(tensors)
        ple = t.pop("ple", None)
        weights = _qwen4_exp_weights.pack_delta_layer(
            attn_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("attn_mixer")),
            mlp_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("mlp_mixer")),
            moe=_pack_moe(t, dense_tiled),
            ple=(None if ple is None else _qwen4_exp_weights.pack_ple(**ple)),
            **t,
        )
        # Size the scratch for the widest launch. Fewer rows use a prefix of it.
        plan = _qwen4_exp_layer.delta_plan(
            ple is not None, max_rows, 1, _qwen4_exp_layer.experts_format(weights)
        )
        return weights, plan.scratch_pairs, _qwen4_exp_layer.late_weights(weights)

    @register_kernel(
        "monokernel",
        "qwen4_exp_delta_layer_decode",
        name="gluon_qwen4_exp_delta_layer_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_delta_layer_gfx950(
        *,
        weights: Any,
        h: torch.Tensor,
        h_out: torch.Tensor,
        scratch: torch.Tensor,
        step: torch.Tensor,
        layer_index: int,
        ple: dict[str, torch.Tensor] | None,
        warm: torch.Tensor | None,
        **state: Any,
    ) -> None:
        _qwen4_exp_layer.launch_gluon_qwen4_exp_delta_layer_gfx950(
            h,
            h_out,
            weights,
            scratch=scratch,
            step=step,
            layer=layer_index,
            trace_buf=None,
            ple_step=None if ple is None else _qwen4_exp_layer.PleStep(**ple),
            warm=warm,
            **state,
        )

    @register_kernel(
        "monokernel",
        "qwen4_exp_qsa_layer_prepare",
        name="gluon_qwen4_exp_qsa_layer_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_qsa_layer_prepare_gfx950(
        *, tensors: dict[str, Any], dense_tiled: bool, max_rows: int
    ) -> tuple[Any, int, torch.Tensor]:
        t = dict(tensors)
        final_mixer = t.pop("final_mixer", None)
        weights = _qwen4_exp_weights.pack_qsa_layer(
            attn_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("attn_mixer")),
            mlp_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("mlp_mixer")),
            moe=_pack_moe(t, dense_tiled),
            final_mixer=(
                None
                if final_mixer is None
                else _qwen4_exp_weights.pack_mixer(*final_mixer)
            ),
            **t,
        )
        plan = _qwen4_exp_layer.qsa_plan(
            final_mixer is not None,
            max_rows,
            1,
            _qwen4_exp_layer.experts_format(weights),
            False,
        )
        return weights, plan.scratch_pairs, _qwen4_exp_layer.late_weights(weights)

    @register_kernel(
        "monokernel",
        "qwen4_exp_qsa_layer_decode",
        name="gluon_qwen4_exp_qsa_layer_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_qsa_layer_gfx950(
        *,
        weights: Any,
        h: torch.Tensor,
        h_out: torch.Tensor,
        scratch: torch.Tensor,
        step: torch.Tensor,
        layer_index: int,
        **caches: Any,
    ) -> None:
        _qwen4_exp_layer.launch_gluon_qwen4_exp_qsa_layer_gfx950(
            h,
            h_out,
            weights,
            scratch=scratch,
            step=step,
            layer=layer_index,
            trace_buf=None,
            **caches,
        )


if _muse_spark_layer is not None:
    _MUSE_GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    # The rows of one request per launch: decode, or a verify of up to 7
    # draft tokens.
    _MUSE_TRAITS = {"rows": frozenset(range(1, _muse_spark_layer.MAX_ROWS + 1))}

    @register_kernel(
        "monokernel",
        "muse_spark_layer_prepare",
        name="gluon_muse_spark_layer_prepare_gfx950",
        solution="gluon",
        capability=_MUSE_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_MUSE_TRAITS,
    )
    def gluon_muse_spark_layer_prepare_gfx950(
        *, tensors: dict[str, Any], max_rows: int
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_weights.pack_layer(**tensors)
        # One scratch serves launches of every row count up to max_rows.
        scratch_pairs = max(
            # The scratch's size depends on the rows, not on how they chain
            # or on the warm stage (which holds no region).
            _muse_spark_layer.layer_plan(weights.full, rows, rows, True).scratch_pairs
            for rows in range(1, max_rows + 1)
        )
        return weights, scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_layer_decode",
        name="gluon_muse_spark_layer_gfx950",
        solution="gluon",
        capability=_MUSE_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_MUSE_TRAITS,
    )
    def gluon_muse_spark_layer_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        layer_index: int,
        chain: int,
        tap: torch.Tensor | None,
        tap_slot: int | None,
        warm_next: Any,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _muse_spark_layer.launch_gluon_muse_spark_layer_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            layer=layer_index,
            chain=chain,
            tap=tap,
            tap_slot=tap_slot,
            warm_next=warm_next,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )

    @register_kernel(
        "monokernel",
        "muse_spark_experts_prepare",
        name="gluon_muse_spark_experts_prepare_gfx950",
        solution="gluon",
        capability=_MUSE_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_MUSE_TRAITS,
    )
    def gluon_muse_spark_experts_prepare_gfx950(
        *, tensors: dict[str, Any]
    ) -> tuple[torch.Tensor, ...]:
        return _muse_spark_weights.prepare_experts(**tensors)


if _muse_spark_draft_layer is not None:
    _DRAFT_GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    # One request's block of 8 rows per launch.
    _DRAFT_TRAITS = {"rows": frozenset({_muse_spark_draft_layer.MAX_ROWS})}

    @register_kernel(
        "monokernel",
        "muse_spark_draft_layer_prepare",
        name="gluon_muse_spark_draft_layer_prepare_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_DRAFT_TRAITS,
    )
    def gluon_muse_spark_draft_layer_prepare_gfx950(
        *, tensors: dict[str, torch.Tensor], rank: int, max_rows: int
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_draft_layer.pack_draft_layer(tensors, rank)
        # The last layer's plan (with the final norm) is the larger.
        plan = _muse_spark_draft_layer.layer_plan(max_rows, True)
        return weights, plan.scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_draft_layer_decode",
        name="gluon_muse_spark_draft_layer_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_DRAFT_TRAITS,
    )
    def gluon_muse_spark_draft_layer_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        layer_index: int,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _muse_spark_draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            layer=layer_index,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )

    # One request's draft steps (the block's rows 1 .. 7) per launch.
    _HEAD_TRAITS = {"rows": frozenset({_muse_spark_draft_head.STEPS})}

    @register_kernel(
        "monokernel",
        "muse_spark_draft_head_prepare",
        name="gluon_muse_spark_draft_head_prepare_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_HEAD_TRAITS,
    )
    def gluon_muse_spark_draft_head_prepare_gfx950(
        *,
        tensors: dict[str, torch.Tensor],
        rank: int,
        max_rows: int,
        scale: float,
        softcap: float,
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_draft_head.pack_draft_head(tensors, rank, scale, softcap)
        return weights, _muse_spark_draft_head.head_plan(max_rows).scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_draft_head_decode",
        name="gluon_muse_spark_draft_head_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_HEAD_TRAITS,
    )
    def gluon_muse_spark_draft_head_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _muse_spark_draft_head.launch_gluon_muse_spark_draft_head_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )

    # One request's verify rows per launch.
    _CONTEXT_TRAITS = {"rows": frozenset({_muse_spark_draft_context.ROWS})}

    @register_kernel(
        "monokernel",
        "muse_spark_draft_context_prepare",
        name="gluon_muse_spark_draft_context_prepare_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_CONTEXT_TRAITS,
    )
    def gluon_muse_spark_draft_context_prepare_gfx950(
        *, tensors: dict[str, torch.Tensor], rank: int, max_rows: int
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_draft_context.pack_draft_context(tensors, rank)
        plan = _muse_spark_draft_context.context_plan(max_rows)
        return weights, plan.scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_draft_context_decode",
        name="gluon_muse_spark_draft_context_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_CONTEXT_TRAITS,
    )
    def gluon_muse_spark_draft_context_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        compile_only: bool,
        **operands: Any,
    ) -> None:
        _muse_spark_draft_context.launch_gluon_muse_spark_draft_context_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )
