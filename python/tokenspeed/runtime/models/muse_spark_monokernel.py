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

"""Muse Spark decode monokernel seam.

With ``--decode-monokernel``, each eligible decoder layer of a one-row decode
step runs as one persistent launch per TP8 rank from
``tokenspeed_kernel.ops.monokernel``. The model touches this module in four
places:

* after loading, :func:`enable_decode_monokernels` packs the layers into
  ``model.monokernels`` (:class:`MuseSparkMonokernels`);
* before cache sizing and graph capture, :meth:`MuseSparkMonokernels.prepare`
  allocates the mailbox scratch on the Iris symmetric heap;
* each forward asks :meth:`MuseSparkMonokernels.plan` once for a
  :class:`MonokernelStep`, or None for the regular path;
* the layer loop runs ``step.layer(i)`` in place of layer ``i`` where it is
  not None, and calls ``step.finish()`` after the loop.

The launch boundary: rank ``r`` keeps hidden columns ``1024 r ..`` of the
FP32 residual and the whole BF16 attention input of the next layer (normed by
that layer's input norm). The regular layers hand over the whole residual
row. So the first launch of a run of fused layers reads its slice of that
row, and the last one gathers the 8 slices back into a whole row for the
regular layer after it. Layers 0 and 61 keep the regular path.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.monokernel import (
    FusedDecodeLayer,
    muse_spark_layer_decode,
    prepare_muse_spark_layer,
)

from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.models.decode_monokernel import coverage
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

# One token row per launch: decode steps of one request.
MAX_ROWS = 1
WORLD = 8
HIDDEN = 8192
OWNED = HIDDEN // WORLD
PAGE_SIZE = 64
WINDOW = 2048

# Values the kernel bakes in; a checkpoint with other values keeps the
# regular path.
KERNEL_CONSTANTS = {
    "qk_scale_factor": 5.473006248474121,
    "output_multiplier": 0.17677669529663687,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
}

# The routed experts' kernels whose weights the monokernel reads in place:
# the gfx950 gluon MXFP4 kernels with the ``gluon_mxfp4_gfx950_moe_weights``
# layout.
EXPERT_KERNELS = frozenset(
    {
        "gluon_mxfp4_a4w4_ep_deferred_precomputed_moe_apply",
        "gluon_mxfp4_a8w4_situ_precomputed_moe_apply",
        "gluon_mxfp4_dynamic_moe_apply",
        "gluon_mxfp4_precomputed_moe_apply",
        "gluon_mxfp4_moe_apply",
    }
)

# Why a layer has no monokernel. The startup report groups layers by these.
BOUNDARY_LAYER = "the first and last layers hand over to the embedding and LM head"
NOT_TP8 = "attention and MoE are not one TP8 group with EP1"
OTHER_SHAPES = "model dimensions other than the kernel's"
OTHER_CONSTANTS = "model constants other than the kernel's"
NOT_BF16 = "a dense projection or norm gain is not BF16"
NOT_SUPPORTED_EXPERTS = "the routed experts do not run a gluon MXFP4 kernel"
NO_KERNEL = "no monokernel for this platform"


def _gain(norm: torch.nn.Module) -> torch.Tensor | None:
    """The FP32 multiplier ``bf16(w + offset)`` of a weighted Muse RMS norm
    (the regular path adds the offset to the BF16 weight), else None."""
    weight = norm.weight
    if weight is None or weight.dtype != torch.bfloat16:
        return None
    return (weight.data + norm.weight_offset).to(torch.bfloat16).float()


def _bf16(proj: torch.nn.Module) -> torch.Tensor | None:
    """The weight of an unquantized BF16 projection, else None."""
    if not isinstance(proj.quant_method, UnquantizedLinearMethod):
        return None
    weight = proj.weight.data
    return weight if weight.dtype == torch.bfloat16 else None


def _experts(experts: torch.nn.Module) -> tuple[torch.Tensor, ...] | None:
    """``(w13, w13 scales, w2, w2 scales)`` of the regular gluon MXFP4 MoE's
    prepared weights, else None."""
    if experts.plan["apply_kernel_name"] not in EXPERT_KERNELS:
        return None
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
    )

    return (
        _extract_gluon_raw_w(experts.w13_weight_triton_tensor),
        _extract_gluon_raw_s(experts.w13_precision_config.b_mx_scale),
        _extract_gluon_raw_w(experts.w2_weight_triton_tensor),
        _extract_gluon_raw_s(experts.w2_precision_config.b_mx_scale),
    )


def model_reasons(config, mapping) -> list[str]:
    """Why no layer of a Muse Spark backbone with ``config`` (its
    ``MuseSparkConfig``) and ``mapping`` can have a monokernel."""
    reasons = []
    if not (
        mapping.attn.tp_size == WORLD
        and mapping.moe.tp_size == WORLD
        and mapping.moe.ep_size == 1
        and mapping.attn.tp_group == mapping.moe.tp_ep_group
        and mapping.pp_size == 1
    ):
        reasons.append(NOT_TP8)
    if (
        config.hidden_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_local_experts,
        config.num_experts_per_tok,
        config.moe_hidden_size,
    ) != (HIDDEN, 64, 128, 16, 256, 8, 4096):
        reasons.append(OTHER_SHAPES)
    if any(
        abs(float(getattr(config, name)) - value) > 1e-6 * max(1.0, abs(value))
        for name, value in KERNEL_CONSTANTS.items()
    ):
        reasons.append(OTHER_CONSTANTS)
    return reasons


def prepare_layer(model: torch.nn.Module, index: int) -> FusedDecodeLayer | list[str]:
    """Pack decoder layer ``index`` of ``model`` for the monokernel, or say
    why not (a list of reasons)."""
    layers = model.layers
    if index == 0 or index == len(layers) - 1:
        return [BOUNDARY_LAYER]
    layer = layers[index]
    attn, mlp = layer.self_attn, layer.mlp
    window = model.config.sliding_window[index]
    if window not in (None, WINDOW):
        return [OTHER_SHAPES]
    dense = [
        _bf16(p)
        for p in (attn.qkv_proj, attn.o_proj, mlp.pre_expert_proj, mlp.post_expert_proj)
    ]
    gains = [
        _gain(n)
        for n in (
            layer.pre_feedforward_layernorm,
            layer.post_feedforward_layernorm,
            layers[index + 1].input_layernorm,
            mlp.pre_expert_norm,
            mlp.post_expert_norm,
        )
    ]
    if any(t is None for t in dense + gains):
        return [NOT_BF16]
    experts = _experts(mlp.experts) if mlp.quantized_experts else None
    if experts is None:
        return [NOT_SUPPORTED_EXPERTS]
    _, o, pool, post = dense
    # The fused projection may hold every rank's rows (rank major, for large
    # batches); the kernel reads this rank's q | k | v | gate rows.
    qkvg = attn.qkv_proj._local_weight()
    try:
        fused = _pack(model, index, window, qkvg, o, pool, post, gains, experts)
    except ValueError as error:
        # The packer names every tensor the kernel does not expect.
        return [f"{OTHER_SHAPES} ({error})"]
    return [NO_KERNEL] if fused is None else fused


def _pack(model, index, window, qkvg, o, pool, post, gains, experts):
    """:func:`prepare_muse_spark_layer` of layer ``index``'s tensors."""
    layer = model.layers[index]
    mlp = layer.mlp
    pre_gain, post_gain, in_gain, pre_exp_gain, pen_gain = gains
    alpha1, beta1 = layer.post_attention_residual_gate._gate_coefficients()
    alpha2, beta2 = layer.post_feedforward_residual_gate._gate_coefficients()
    return prepare_muse_spark_layer(
        tensors={
            "rank": model.mapping.attn.tp_rank,
            "full": window is None,
            "qkvg": qkvg,
            "o": o,
            "alpha1": alpha1,
            "beta1": beta1,
            "pre_gain": pre_gain,
            "router": mlp.gate.weight.data,
            "router_bias": mlp.gate.e_score_correction_bias.data,
            "pool": pool,
            "experts": experts,
            "pre_exp_gain": pre_exp_gain,
            "pen_gain": pen_gain,
            "post": post,
            "post_gain": post_gain,
            "alpha2": alpha2,
            "beta2": beta2,
            "in_gain": in_gain,
        }
    )


def takes_monokernels(rows: int, ctx: ForwardContext) -> bool:
    """Whether a forward of ``rows`` token rows takes the monokernels: a
    decode step of one request with one token. Prefill, several requests and
    speculative verify rows keep the regular path."""
    return ctx.forward_mode.is_decode() and ctx.bs == 1 and rows == MAX_ROWS


class MonokernelSteps:
    """Model-wide monokernel state: the mailbox scratch on the Iris symmetric
    heap, the heap bases of the TP8 group and the step counter."""

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


class MuseSparkMonokernels:
    """The decode monokernels of one Muse Spark backbone, packed after loading.

    Attributes:
        layers: The backbone's decoder layers.
        fused: The packed launch of each served layer, by layer index.
        unserved: The reasons of each layer without a launch, by layer index.
            Those layers keep the regular path.
        steps: The mailbox scratch, the heap bases and the step counter, from
            :meth:`prepare` (None before it).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.layers = model.layers
        self.group = model.mapping.attn.tp_group
        self.fused: dict[int, FusedDecodeLayer] = {}
        self.unserved: dict[int, list[str]] = {}
        common = model_reasons(model.config, model.mapping)
        for index in range(len(self.layers)):
            prepared = common or prepare_layer(model, index)
            if isinstance(prepared, FusedDecodeLayer):
                self.fused[index] = prepared
            else:
                self.unserved[index] = prepared
        self.steps: MonokernelSteps | None = None

    def prepare(self, group: dist.ProcessGroup, rank: int) -> None:
        """Allocate the shared state on the Iris heap of ``group`` (the TP8
        device process group, ``rank`` this process's rank in it). Call once,
        before graph capture."""
        if self.steps is None:
            scratch_pairs = max(f.scratch_pairs for f in self.fused.values())
            self.steps = MonokernelSteps(group, rank, scratch_pairs)

    def plan(self, rows: int, ctx: ForwardContext) -> MonokernelStep | None:
        """This forward's step of ``rows`` token rows, or None when the
        monokernels do not serve it (the regular path): a decode step of one
        request with one token."""
        if self.steps is None or not takes_monokernels(rows, ctx):
            return None
        return MonokernelStep(self)


class MonokernelStep:
    """One forward that the monokernels serve."""

    def __init__(self, monokernels: MuseSparkMonokernels) -> None:
        self._monokernels = monokernels
        # The owned residual columns that the last launch wrote, while the
        # next layer is fused too.
        self._owned: torch.Tensor | None = None

    def layer(self, index: int) -> Callable | None:
        """The launch that replaces decoder layer ``index``, with the layer
        loop's arguments, or None when the layer keeps the regular path."""
        if index not in self._monokernels.fused:
            return None
        return functools.partial(self._run, index)

    def finish(self) -> None:
        """End the forward: give the next forward's launches a new tag."""
        self._monokernels.steps.advance()

    def _run(
        self,
        index: int,
        *,
        hidden_states: torch.Tensor,
        attention_input: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run layer ``index``'s launch. Returns ``(residual, next attention
        input)`` as the regular layer does: the whole residual row only when
        the next layer keeps the regular path, else the input row unchanged
        (the next launch reads the owned columns this one wrote)."""
        monokernels = self._monokernels
        layer, fused, steps = (
            monokernels.layers[index],
            monokernels.fused[index],
            monokernels.steps,
        )
        rank = fused.weights.rank
        if self._owned is None:
            h = hidden_states[0, rank * OWNED : (rank + 1) * OWNED]
        else:
            h = self._owned
        attn = layer.self_attn
        group_id = attn.attn.group_id
        view = ctx.attn_backend.group_view(group_id, ctx.bs)
        if view.kernel_page_size != PAGE_SIZE:
            raise RuntimeError(
                f"Muse Spark monokernel: KV page size {view.kernel_page_size}, "
                f"the kernel reads {PAGE_SIZE}"
            )
        k_cache, v_cache = ctx.token_to_kv_pool.get_kv_buffer(layer.comm_manager.layer_id)
        leaf = ctx.attn_backend.leaves[group_id]
        slot = ctx.attn_backend.write_locations(attn.attn, ctx.forward_mode)
        cos_sin = (
            attn.rotary_emb.cos_sin_cache
            if attn.rotary_emb is not None
            else attn.nope_cos_sin_cache
        )
        h_out = h.new_empty(OWNED)
        a_out = attention_input.new_empty(1, HIDDEN)
        muse_spark_layer_decode(
            fused,
            h=h,
            a=attention_input[0],
            h_out=h_out,
            a_out=a_out[0],
            seq_len=leaf.forward_decode_metadata.seq_lens[:1],
            slot=slot[:1],
            page_table=view.page_table[0],
            k_cache=k_cache.view(-1, 2, 64),
            v_cache=v_cache.view(-1, 2, 64),
            cos_sin=cos_sin,
            scratch=steps.scratch,
            step=steps.step,
            peer_bases=steps.bases,
            layer_index=index,
        )
        if index + 1 in monokernels.fused:
            self._owned = h_out
            return hidden_states, a_out
        # The next layer keeps the regular path: it reads the whole row.
        self._owned = None
        return all_gather(h_out[None], monokernels.group, dim=-1), a_out


def enable_decode_monokernels(model: torch.nn.Module) -> None:
    """Pack ``model``'s (a ``MuseSparkModel``) decode monokernels into
    ``model.monokernels`` when ``--decode-monokernel`` is set, and report the
    layers that keep the regular path, with the reasons.

    Raises:
        ValueError: No layer has a monokernel.
    """
    if not global_server_args_dict["decode_monokernel"]:
        return
    monokernels = MuseSparkMonokernels(model)
    name = type(model).__name__
    warning = coverage(
        name, len(monokernels.layers), monokernels.unserved, required=True
    )
    if warning is not None:
        logger.warning(warning)
    model.monokernels = monokernels
    logger.info(
        f"Decode monokernel: {len(monokernels.fused)} of {len(monokernels.layers)} "
        f"{name} layers"
    )


__all__ = [
    "MAX_ROWS",
    "MonokernelStep",
    "MonokernelSteps",
    "MuseSparkMonokernels",
    "enable_decode_monokernels",
    "model_reasons",
    "prepare_layer",
    "takes_monokernels",
]
