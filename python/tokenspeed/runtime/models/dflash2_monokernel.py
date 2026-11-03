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

"""DFlash2 draft monokernel seam (Muse Spark's draft on TP8).

With ``--decode-monokernel``, a draft block forward of one request runs each
of the draft's layers as one launch of
``tokenspeed_kernel.ops.monokernel.muse_spark_draft_layer_decode`` on every
rank of the TP8 group. The draft model touches this module in three places:

* after loading, :func:`enable_draft_monokernels` packs the layers into
  ``model.monokernels`` (:class:`DFlash2Monokernels`), or leaves it None
  with the reasons logged;
* ``prepare_communication_runtime`` calls :meth:`DFlash2Monokernels.prepare`,
  which allocates the mailbox scratch on the Iris symmetric heap;
* the layer loop asks :meth:`DFlash2Monokernels.plan` for the forward's
  launches and runs them in place of the layers when it gets some.

The launches hand each other this rank's partial of the layer output and its
owned residual columns (``1024 rank ..``), so the layers are served all
together or not at all. The final norm keeps the regular path: it reads the
whole residual rows, gathered after the last launch.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.monokernel import (
    FusedDecodeLayer,
    muse_spark_draft_layer_decode,
    prepare_muse_spark_draft_layer,
)

from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

WORLD = 8
HIDDEN = 8192
OWNED = HIDDEN // WORLD
# One request's draft block.
BLOCK_ROWS = 8
PAGE_SIZE = 64
# The layers' window_left: every block row attends the request's last 2048
# keys (the block's 8 among them).
WINDOW = 2047

# The draft architecture the kernel bakes in; another one keeps the regular
# path.
KERNEL_SHAPE = {
    "hidden_size": HIDDEN,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
}
KERNEL_DFLASH = {
    "block_size": BLOCK_ROWS,
    "conv_kernel_size": 2,
    "conv_group_size": 16,
}
RMS_NORM_EPS = 1e-5

NOT_TP8 = "attention and the dense layers are not one TP8 group"
OTHER_SHAPES = "draft dimensions other than the kernel's"
OTHER_ATTENTION = (
    "an attention other than the kernel's (non-causal GQA, NeoX RoPE, "
    f"window_left {WINDOW}, RMS eps {RMS_NORM_EPS})"
)
NOT_BF16 = "a projection is quantized or a weight is not BF16"
NO_KERNEL = "no draft monokernel for this platform"


def _config_value(config, key: str):
    nested = getattr(config, "dflash_config", None) or {}
    return nested.get(key, getattr(config, key, None))


def model_reasons(model: torch.nn.Module) -> list[str]:
    """Why the layers of ``model`` (a ``DFlash2DraftModel``) cannot have
    draft monokernels; empty when they can."""
    config, mapping = model.config, model.mapping
    reasons = []
    if not (
        mapping.attn.tp_size == WORLD
        and mapping.dense.tp_size == WORLD
        and mapping.attn.tp_group == mapping.dense.tp_group
        and mapping.pp_size == 1
    ):
        reasons.append(NOT_TP8)
    if any(getattr(config, k, None) != v for k, v in KERNEL_SHAPE.items()) or any(
        _config_value(config, k) != v for k, v in KERNEL_DFLASH.items()
    ):
        reasons.append(OTHER_SHAPES)
    if str(_config_value(config, "attention_mode") or "gqa").lower() != "gqa" or any(
        not _attention_matches(layer) for layer in model.layers
    ):
        reasons.append(OTHER_ATTENTION)
    if any(not _bf16(layer) for layer in model.layers):
        reasons.append(NOT_BF16)
    return reasons


def _attention_matches(layer: torch.nn.Module) -> bool:
    attn = layer.self_attn
    norms = (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        attn.q_norm,
        attn.k_norm,
    )
    return (
        attn.attn.sliding_window_size == WINDOW
        and attn.rotary_emb.is_neox_style
        and attn.rotary_emb.cos_sin_cache.dtype == torch.float32
        and all(abs(n.variance_epsilon - RMS_NORM_EPS) < 1e-12 for n in norms)
    )


def _bf16(layer: torch.nn.Module) -> bool:
    linears = (
        layer.self_attn.qkv_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_up_proj,
        layer.mlp.down_proj,
    )
    return all(
        isinstance(linear.quant_method, UnquantizedLinearMethod)
        and linear.weight.dtype == torch.bfloat16
        for linear in linears
    )


def layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A ``DFlash2DecoderLayer``'s weights by the roles of
    :func:`prepare_muse_spark_draft_layer` (this rank's shards of the
    sharded projections, as loaded)."""
    attn, mlp = layer.self_attn, layer.mlp
    return {
        "input_norm": layer.input_layernorm.weight.data,
        "post_norm": layer.post_attention_layernorm.weight.data,
        "attn_coef": layer.attention_conv.kernel_projection.weight.data,
        "mlp_coef": layer.mlp_conv.kernel_projection.weight.data,
        "attn_base": layer.attention_conv.base_kernel.data,
        "mlp_base": layer.mlp_conv.base_kernel.data,
        "qkv": attn.qkv_proj.weight.data,
        "q_norm": attn.q_norm.weight.data,
        "k_norm": attn.k_norm.weight.data,
        "o_proj": attn.o_proj.weight.data,
        "gate_up": mlp.gate_up_proj.weight.data,
        "down": mlp.down_proj.weight.data,
    }


def takes_monokernels(rows: int, ctx: ForwardContext) -> bool:
    """Whether a draft forward of ``rows`` token rows takes the monokernels:
    the block of one request. Several requests keep the regular path."""
    return ctx.forward_mode.is_decode() and ctx.bs == 1 and rows == BLOCK_ROWS


class DraftMonokernelSteps:
    """The draft launches' shared state: the mailbox scratch on the Iris
    symmetric heap, the heap bases of the TP8 group and the step counter.

    The draft keeps a scratch of its own: its tags (``step * 64 + layer +
    1``) would repeat the target's, whose step counter moves at a rate of
    its own."""

    def __init__(self, group: dist.ProcessGroup, rank: int, scratch_pairs: int) -> None:
        from tokenspeed_kernel.ops.communication._iris import (
            MIN_HEAP_BYTES,
            IrisGroup,
            get_or_create_context,
        )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        context = get_or_create_context(heap_size=MIN_HEAP_BYTES)
        iris_group = IrisGroup(context, group, rank, device)
        # Nothing clears the scratch: the tags make old pairs stale.
        self.scratch = context.zeros((scratch_pairs,), dtype=torch.int64)
        self.bases = iris_group.bases.to(device)
        # The launches' tag base: it moves once per forward, inside the graph.
        self.step = torch.zeros(1, dtype=torch.int32, device=device)
        iris_group.finish_prepare()

    def advance(self) -> None:
        """Give the next forward's launches a new tag (graph safe)."""
        self.step.add_(1)


class DFlash2Monokernels:
    """The draft monokernels of one DFlash2 draft model, packed after loading.

    Attributes:
        layers: The draft's decoder layers.
        fused: The packed launch of each layer, in layer order.
        group: The TP8 device group the final residual is gathered over.
        steps: The mailbox scratch, the heap bases and the step counter, from
            :meth:`prepare` (None before it).
        cpu_group: The TP8 group's CPU (gloo) process group, from
            :meth:`prepare` (None before it).
        compiled: Whether every rank compiled the launches.
    """

    def __init__(self, model: torch.nn.Module, fused: list[FusedDecodeLayer]) -> None:
        self.layers = model.layers
        self.fused = fused
        self.group = model.mapping.attn.tp_group
        self.steps: DraftMonokernelSteps | None = None
        self.cpu_group: dist.ProcessGroup | None = None
        self.compiled = False

    def prepare(
        self, group: dist.ProcessGroup, cpu_group: dist.ProcessGroup, rank: int
    ) -> None:
        """Allocate the shared state on the Iris heap of ``group`` (the TP8
        device process group; ``cpu_group`` its CPU twin, ``rank`` this
        process's rank in it). Call once, before graph capture."""
        if self.steps is None:
            scratch_pairs = max(f.scratch_pairs for f in self.fused)
            self.steps = DraftMonokernelSteps(group, rank, scratch_pairs)
            self.cpu_group = cpu_group

    def serves(self, rows: int, ctx: ForwardContext) -> bool:
        """Whether this forward runs the launches (:meth:`forward`) rather
        than the layers."""
        return self.steps is not None and takes_monokernels(rows, ctx)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run every layer's launch on the block, as the layer loop would.

        Args:
            positions: The block rows' positions.
            hidden_states: This rank's partial of the first layer's input.
            residual: The residual entering the first layer.
            ctx: The draft forward's context.

        Returns:
            ``(hidden_states, residual)`` after the last layer: this rank's
            partial of its output and the whole residual rows.
        """
        steps = self.steps
        rank = self.fused[0].weights.rank
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        backend = ctx.attn_backend
        positions = positions.to(torch.int32)
        x, res = hidden_states, residual
        for index, (layer, fused) in enumerate(zip(self.layers, self.fused)):
            attn = layer.self_attn.attn
            view = backend.group_view(attn.group_id, ctx.bs)
            if view.kernel_page_size != PAGE_SIZE:
                raise RuntimeError(
                    f"DFlash2 draft monokernel: KV page size "
                    f"{view.kernel_page_size}, the kernel reads {PAGE_SIZE}"
                )
            k_cache, v_cache = ctx.token_to_kv_pool.get_kv_buffer(attn.layer_id)
            leaf = backend.leaves[attn.group_id]
            out = torch.empty_like(x)
            res_out = torch.empty_like(res)
            operands = dict(
                x=x,
                residual=res,
                residual_out=res_out,
                out=out,
                # Every block row's length: the request's after the block.
                seq_len=leaf.decode_seq_lens_buffer[:1],
                positions=positions,
                slots=backend.write_locations(attn, ctx.forward_mode)[:BLOCK_ROWS],
                page_table=view.page_table[0],
                k_cache=k_cache.view(-1, 1, k_cache.shape[-1]),
                v_cache=v_cache.view(-1, 1, v_cache.shape[-1]),
                cos_sin=layer.self_attn.rotary_emb.cos_sin_cache,
                scratch=steps.scratch,
                step=steps.step,
                peer_bases=steps.bases,
                layer_index=index,
            )
            if not self.compiled:
                # A launch traps when its peers' values do not come within a
                # few seconds, and a first launch compiles for longer on some
                # ranks than on others: every rank compiles, then the ranks
                # meet. Every layer shares one binary.
                muse_spark_draft_layer_decode(fused, compile_only=True, **operands)
                dist.barrier(group=self.cpu_group)
                self.compiled = True
            muse_spark_draft_layer_decode(fused, compile_only=False, **operands)
            x, res = out, res_out
        steps.advance()
        # The launches store only the owned residual columns.
        return x, all_gather(res[:, own].contiguous(), self.group, dim=-1)


def enable_draft_monokernels(model: torch.nn.Module) -> None:
    """Pack ``model``'s (a ``DFlash2DraftModel``) draft monokernels into
    ``model.monokernels`` when ``--decode-monokernel`` is set, or log why
    the draft keeps the regular path."""
    if not global_server_args_dict["decode_monokernel"]:
        return
    name = type(model).__name__
    reasons = model_reasons(model)
    fused: list[FusedDecodeLayer] = []
    if not reasons:
        rank = model.mapping.attn.tp_rank
        for layer in model.layers:
            prepared = prepare_muse_spark_draft_layer(
                tensors=layer_tensors(layer), rank=rank, max_rows=BLOCK_ROWS
            )
            if prepared is None:
                reasons.append(NO_KERNEL)
                break
            fused.append(prepared)
    if reasons:
        logger.warning(
            f"Decode monokernel: the {name} draft keeps the regular path "
            f"({'; '.join(reasons)})"
        )
        return
    model.monokernels = DFlash2Monokernels(model, fused)
    logger.info(f"Decode monokernel: {len(fused)} of {len(fused)} {name} layers")


__all__ = [
    "BLOCK_ROWS",
    "DFlash2Monokernels",
    "DraftMonokernelSteps",
    "enable_draft_monokernels",
    "layer_tensors",
    "model_reasons",
    "takes_monokernels",
]
