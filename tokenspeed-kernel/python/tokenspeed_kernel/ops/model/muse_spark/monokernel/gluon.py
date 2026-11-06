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

"""Shims that register the AMD Gluon Muse Spark monokernels."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

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
            # The scratch's size depends on the rows, not on how they chain,
            # on the warm stage or on qnext (every plan lays out the same
            # regions).
            _muse_spark_layer.layer_plan(
                weights.full,
                rows,
                rows,
                True,
                True,
                False,
                weights.bf16_experts,
                False,  # check_parity, off in serving (see the launch)
            ).scratch_pairs
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
        whole_rows: bool,
        tap: torch.Tensor | None,
        tap_slot: int | None,
        warm_next: Any,
        proj_in: torch.Tensor | None,
        proj_out: torch.Tensor | None,
        stuck_rec: torch.Tensor | None,
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
            whole_rows=whole_rows,
            tap=tap,
            tap_slot=tap_slot,
            warm_next=warm_next,
            proj_in=proj_in,
            proj_out=proj_out,
            trace_buf=None,
            stuck_rec=stuck_rec,
            # Serving may order two same-parity launches by a regular
            # layer's collectives in between (a layer that is not fused).
            check_parity=False,
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
    # The blocks of one or two requests (8 rows each) per launch.
    _DRAFT_TRAITS = {
        "rows": frozenset(
            range(
                _muse_spark_draft_layer.BLOCK_ROWS,
                _muse_spark_draft_layer.MAX_ROWS + 1,
                _muse_spark_draft_layer.BLOCK_ROWS,
            )
        )
    }

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

    # The draft steps (each block's rows 1 .. 7) of one or two requests per
    # launch.
    _HEAD_TRAITS = {
        "rows": frozenset(
            _muse_spark_draft_head.STEPS * n
            for n in range(1, _muse_spark_draft_head.MAX_REQUESTS + 1)
        )
    }

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
        "muse_spark_draft_head_scratch",
        name="gluon_muse_spark_draft_head_scratch_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_HEAD_TRAITS,
    )
    def gluon_muse_spark_draft_head_scratch_gfx950(*, max_rows: int) -> int:
        return _muse_spark_draft_head.head_plan(max_rows).scratch_pairs

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

    # The verify rows of one or two requests per launch.
    _CONTEXT_TRAITS = {
        "rows": frozenset(
            _muse_spark_draft_context.ROWS * n
            for n in range(1, _muse_spark_draft_context.MAX_REQUESTS + 1)
        )
    }

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
