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
* after a verify of one request, the drafter runs
  :meth:`DFlash2Monokernels.write_context` in place of its context write
  (``fc`` and ``hidden_norm`` of the target's taps, then every layer's K and
  V of the verify rows into the draft cache);
* the layer loop asks :meth:`DFlash2Monokernels.serves` whether the forward
  takes the launches and runs :meth:`DFlash2Monokernels.forward` in place of
  the layers when it does;
* once the drafter is wired to the target's LM head,
  :meth:`DFlash2Monokernels.enable_head` packs the draft head (the
  vocabulary-parallel top 16 of each draft step and the candidate selector's
  scores) as one more launch, :class:`DraftHeadMonokernel`, which the
  drafter runs in place of its top-k and selector for one request. It keeps
  a tiled copy of the rank's LM-head shard (414 MB a GPU): the tiles stream
  faster than the row-major weight the target's verify reads.

The launches hand each other this rank's partial of the layer output and its
owned residual columns (``1024 rank ..``), so the layers are served all
together or not at all. The last launch also runs the draft's final norm and
stores its output, the whole rows on every rank.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.monokernel import (
    FusedDecodeLayer,
    muse_spark_draft_context_decode,
    muse_spark_draft_head_decode,
    muse_spark_draft_layer_decode,
    prepare_muse_spark_draft_context,
    prepare_muse_spark_draft_head,
    prepare_muse_spark_draft_layer,
)

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.logits_processor import (
    should_apply_lm_head_quant_method,
)
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
# The draft head the kernel bakes in.
VOCAB = 202048
SHARD = VOCAB // WORLD
SELECTOR_RANK = 256
TOPK = 16
STEPS = BLOCK_ROWS - 1  # the block's rows 1 .. 7 draft a token each
CONTEXT_ROWS = 8  # one request's verify rows
NO_KERNEL = "no draft monokernel for this platform"
NOT_BF16_KV = "the KV cache is not BF16 (--kv-cache-dtype): the kernels read BF16 keys"


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
    if (
        str(_config_value(config, "attention_mode") or "gqa").lower() != "gqa"
        or any(not _attention_matches(layer) for layer in model.layers)
        # The last launch runs the final norm.
        or abs(model.norm.variance_epsilon - RMS_NORM_EPS) >= 1e-12
    ):
        reasons.append(OTHER_ATTENTION)
    if any(not _bf16(layer) for layer in model.layers):
        reasons.append(NOT_BF16)
    # "auto" keeps the model's BF16.
    if global_server_args_dict["kv_cache_dtype"] not in ("auto", "bfloat16", "bf16"):
        reasons.append(NOT_BF16_KV)
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


def mailbox_scratch(
    group: dist.ProcessGroup, rank: int, scratch_pairs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scratch_pairs`` mailbox pairs on the Iris symmetric heap of
    ``group`` (``rank`` this process's rank in it), and the group's heap
    bases. Nothing clears the scratch: the tags make old pairs stale."""
    from tokenspeed_kernel.ops.communication._iris import (
        MIN_HEAP_BYTES,
        IrisGroup,
        get_or_create_context,
    )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    context = get_or_create_context(heap_size=MIN_HEAP_BYTES)
    iris_group = IrisGroup(context, group, rank, device)
    scratch = context.zeros((scratch_pairs,), dtype=torch.int64)
    bases = iris_group.bases.to(device)
    iris_group.finish_prepare()
    return scratch, bases


class DraftMonokernelSteps:
    """The draft launches' shared state: the mailbox scratch on the Iris
    symmetric heap, the heap bases of the TP8 group and the step counter.

    The draft keeps a scratch of its own: its tags (``step * 64 + layer +
    1``) would repeat the target's, whose step counter moves at a rate of
    its own."""

    def __init__(self, group: dist.ProcessGroup, rank: int, scratch_pairs: int) -> None:
        self.scratch, self.bases = mailbox_scratch(group, rank, scratch_pairs)
        # The launches' tag base: it moves once per forward, inside the graph.
        self.step = torch.zeros(1, dtype=torch.int32, device=self.scratch.device)

    def advance(self) -> None:
        """Give the next forward's launches a new tag (graph safe)."""
        self.step.add_(1)


class DFlash2Monokernels:
    """The draft monokernels of one DFlash2 draft model, packed after loading.

    Attributes:
        layers: The draft's decoder layers.
        fused: The packed launch of each layer, in layer order.
        context: The packed context write, or None when it keeps the regular
            path.
        context_scratch: Its mailbox scratch, from :meth:`prepare`.
        context_pending: Whether a context write ran whose block forward
            has not run yet (the write reads the layers' step counter, which
            the block forward advances).
        final_gain: FP32 ``[1024]``: the final norm's gain of the rank's
            owned columns (the last launch runs the final norm).
        steps: The mailbox scratch, the heap bases and the step counter, from
            :meth:`prepare` (None before it).
        cpu_group: The TP8 group's CPU (gloo) process group, from
            :meth:`prepare` (None before it).
        device_group, rank: The TP8 device group and this process's rank in
            it, from :meth:`prepare` (None before it).
        compiled: Which binaries every rank compiled: the last layer's
            (True) and the others' (False).
        block_pending: Whether the layers' launches ran a block whose head
            has not run yet (the head reads their step counter).
        head: The draft head's launch, from :meth:`enable_head` (None
            before it, or when the head keeps the regular path).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        fused: list[FusedDecodeLayer],
        context: FusedDecodeLayer | None,
    ) -> None:
        self.layers = model.layers
        self.fused = fused
        self.context = context
        self.context_scratch: torch.Tensor | None = None
        self.context_pending = False
        rank = fused[0].weights.rank
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        self.final_gain = model.norm.weight.data[own].float().contiguous()
        self.steps: DraftMonokernelSteps | None = None
        self.device_group: dist.ProcessGroup | None = None
        self.cpu_group: dist.ProcessGroup | None = None
        self.rank: int | None = None
        self.compiled: set[bool] = set()
        self.block_pending = False
        self.context_compiled = False
        self.head: DraftHeadMonokernel | None = None

    def prepare(
        self, group: dist.ProcessGroup, cpu_group: dist.ProcessGroup, rank: int
    ) -> None:
        """Allocate the shared state on the Iris heap of ``group`` (the TP8
        device process group; ``cpu_group`` its CPU twin, ``rank`` this
        process's rank in it). Call once, before graph capture."""
        if self.steps is None:
            scratch_pairs = max(f.scratch_pairs for f in self.fused)
            self.steps = DraftMonokernelSteps(group, rank, scratch_pairs)
            if self.context is not None:
                self.context_scratch, _ = mailbox_scratch(
                    group, rank, self.context.scratch_pairs
                )
            self.device_group = group
            self.cpu_group = cpu_group
            self.rank = rank

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
    ) -> torch.Tensor:
        """Run every layer's launch on the block, as the layer loop and the
        final norm would.

        Args:
            positions: The block rows' positions.
            hidden_states: This rank's partial of the first layer's input.
            residual: The residual entering the first layer.
            ctx: The draft forward's context.

        Returns:
            The final norm's output, BF16 ``[rows, 8192]``.
        """
        steps = self.steps
        backend = ctx.attn_backend
        last = len(self.fused) - 1
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
                out=None if index == last else out,
                final_gain=self.final_gain if index == last else None,
                normed=out if index == last else None,
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
            if (index == last) not in self.compiled:
                # A launch traps when its peers' values do not come within a
                # few seconds, and a first launch compiles for longer on some
                # ranks than on others: every rank compiles, then the ranks
                # meet. The layers before the last share one binary.
                muse_spark_draft_layer_decode(fused, compile_only=True, **operands)
                dist.barrier(group=self.cpu_group)
                self.compiled.add(index == last)
            muse_spark_draft_layer_decode(fused, compile_only=False, **operands)
            x, res = out, res_out
        steps.advance()
        self.block_pending = True
        self.context_pending = False
        return x

    def serves_context(self, rows: int) -> bool:
        """Whether a context write of ``rows`` verify rows (decode rows only)
        takes :meth:`write_context`: one request's."""
        return self.context_scratch is not None and rows == CONTEXT_ROWS

    def write_context(
        self,
        taps: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        token_to_kv_pool,
    ) -> None:
        """The context write of one request's verify rows as one launch.

        Args:
            taps: BF16 ``[8, 40960]``: the target's tapped hidden states.
            positions: ``[8]``: the rows' positions.
            slots: ``[8]``: the rows' draft cache slots.
            token_to_kv_pool: The draft's KV pool.
        """
        if self.context_pending:
            raise RuntimeError(
                "a second draft context write before the block forward would "
                "repeat its launch's tag"
            )
        caches = [
            token_to_kv_pool.get_kv_buffer(layer.self_attn.attn.layer_id)
            for layer in self.layers
        ]
        operands = dict(
            taps=taps.contiguous(),
            positions=positions,
            slots=slots,
            k_caches=tuple(k.view(-1, 1, k.shape[-1]) for k, _ in caches),
            v_caches=tuple(v.view(-1, 1, v.shape[-1]) for _, v in caches),
            cos_sin=self.layers[0].self_attn.rotary_emb.cos_sin_cache,
            scratch=self.context_scratch,
            step=self.steps.step,
            peer_bases=self.steps.bases,
        )
        if not self.context_compiled:
            muse_spark_draft_context_decode(self.context, compile_only=True, **operands)
            dist.barrier(group=self.cpu_group)
            self.context_compiled = True
        muse_spark_draft_context_decode(self.context, compile_only=False, **operands)
        self.context_pending = True

    def run_head(
        self, hidden_states: torch.Tensor, first_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The draft head's launch (:class:`DraftHeadMonokernel`) after the
        layers' launches of the same forward."""
        if not self.block_pending:
            raise RuntimeError(
                "the draft head monokernel runs after the draft layers' launches"
            )
        self.block_pending = False
        return self.head(hidden_states, first_token)

    def enable_head(
        self,
        lm_head: torch.nn.Module,
        selector: torch.nn.Module,
        *,
        top_k_exact: bool,
        scale: float,
        softcap: float,
    ) -> None:
        """Pack the draft head once the drafter holds the target's LM head
        (``lm_head``, vocabulary-parallel) and the draft's candidate
        ``selector``, after :meth:`prepare`. ``top_k_exact``: whether the
        shard-local top-k is exact for the head's shard geometry
        (``VocabParallelTopK.enabled``). Logs why the head keeps the regular
        path otherwise."""
        reasons = head_reasons(lm_head, selector, top_k_exact, softcap)
        fused = None
        if not reasons:
            fused = prepare_muse_spark_draft_head(
                tensors={
                    "lm_head": lm_head.weight.data[:SHARD],
                    "hidden_proj": selector.hidden_projection.weight.data,
                    "pred_codebook": selector.predecessor_codebook.data,
                    "succ_codebook": selector.successor_codebook.data,
                },
                rank=self.rank,
                max_rows=STEPS,
                scale=scale,
                softcap=softcap,
            )
            if fused is None:
                reasons.append(NO_KERNEL)
        if reasons:
            logger.warning(
                f"Decode monokernel: the DFlash2 draft head keeps the regular path "
                f"({'; '.join(reasons)})"
            )
            return
        self.head = DraftHeadMonokernel(
            fused, self.device_group, self.cpu_group, self.rank, self.steps.step
        )
        logger.info("Decode monokernel: the DFlash2 draft head (top 16 and selector)")


HEAD_NOT_SHARDED = "the LM head is quantized or not an exact 8-way vocabulary shard"
HEAD_OTHER_SHAPES = "an LM head or selector of other dimensions than the kernel's"
HEAD_NO_SOFTCAP = "no final logit softcap"


def head_reasons(
    lm_head: torch.nn.Module,
    selector: torch.nn.Module,
    top_k_exact: bool,
    softcap: float,
) -> list[str]:
    """Why the draft head (``lm_head``, ``selector``) cannot have its
    monokernel; empty when it can."""
    reasons = []
    weight = lm_head.weight
    if (
        not top_k_exact
        or should_apply_lm_head_quant_method(lm_head, lm_head.quant_method)
        or weight.dtype != torch.bfloat16
        or weight.shape[0] < SHARD
        or weight.shape[1] != HIDDEN
    ):
        reasons.append(HEAD_NOT_SHARDED)
    if (
        selector.top_k != TOPK
        or tuple(selector.predecessor_codebook.shape) != (VOCAB, SELECTOR_RANK)
        or tuple(selector.hidden_projection.weight.shape) != (SELECTOR_RANK, HIDDEN)
        or not isinstance(
            selector.hidden_projection.quant_method, UnquantizedLinearMethod
        )
    ):
        reasons.append(HEAD_OTHER_SHAPES)
    if not softcap > 0:
        reasons.append(HEAD_NO_SOFTCAP)
    return reasons


def takes_head(rows: int) -> bool:
    """Whether a draft head over ``rows`` draft rows (``bs * 7``) takes the
    monokernel: one request's. Several requests keep the regular path."""
    return rows == STEPS


class DraftHeadMonokernel:
    """The draft head's launch: its packed weights, its own mailbox scratch,
    and the TP8 group's CPU group for the compile-ahead.

    It keeps no step counter of its own: it reads the draft layers', which
    their launches advance once a block forward, before the head runs (the
    head has its own scratch, so its tags never meet the layers'). The head
    therefore runs only after the layers' launches of the same forward
    (:meth:`DFlash2Monokernels.forward` marks the block, the head takes the
    mark); a head without them would repeat the last launch's tag.

    Attributes:
        fused: The packed head.
        scratch, bases: Its mailbox scratch and the group's heap bases.
        step: The draft layers' step counter.
        cpu_group: The TP8 group's CPU (gloo) process group.
        compiled: Whether every rank compiled the launch.
    """

    def __init__(
        self,
        fused: FusedDecodeLayer,
        group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        rank: int,
        step: torch.Tensor,
    ) -> None:
        self.fused = fused
        self.scratch, self.bases = mailbox_scratch(group, rank, fused.scratch_pairs)
        self.step = step
        self.cpu_group = cpu_group
        self.compiled = False

    def __call__(
        self, hidden_states: torch.Tensor, first_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The candidates and edge scores of one request's draft steps.

        Args:
            hidden_states: BF16 ``[7, 8192]``: the draft's final-norm output
                of the block's rows 1 .. 7.
            first_token: int32 ``[1]``: the block's first token.

        Returns:
            ``(candidate_ids, scores, anchor)``: int64 ``[7, 16]`` and FP32
            ``[7, 16, 16]``, as ``VocabParallelTopK`` and
            ``CandidateSelector`` give them, and the first token clamped to
            the vocabulary (int64 ``[1]``).
        """
        device = hidden_states.device
        operands = dict(
            x=hidden_states.contiguous(),
            first_token=first_token,
            anchor=torch.empty(1, dtype=torch.int64, device=device),
            cand_ids=torch.empty(STEPS, TOPK, dtype=torch.int64, device=device),
            unary=torch.empty(STEPS, TOPK, dtype=torch.float32, device=device),
            scores=torch.empty(STEPS, TOPK, TOPK, dtype=torch.float32, device=device),
            scratch=self.scratch,
            step=self.step,
            peer_bases=self.bases,
        )
        if not self.compiled:
            muse_spark_draft_head_decode(self.fused, compile_only=True, **operands)
            dist.barrier(group=self.cpu_group)
            self.compiled = True
        muse_spark_draft_head_decode(self.fused, compile_only=False, **operands)
        return operands["cand_ids"], operands["scores"], operands["anchor"]


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
    context = None
    reasons = context_reasons(model)
    if not reasons:
        context = prepare_muse_spark_draft_context(
            tensors=context_tensors(model), rank=rank, max_rows=CONTEXT_ROWS
        )
        if context is None:
            reasons.append(NO_KERNEL)
    if reasons:
        logger.warning(
            f"Decode monokernel: the {name} context write keeps the regular path "
            f"({'; '.join(reasons)})"
        )
    model.monokernels = DFlash2Monokernels(model, fused, context)
    logger.info(
        f"Decode monokernel: {len(fused)} of {len(fused)} {name} layers"
        + ("" if context is None else ", and the context write")
    )


CONTEXT_OTHER = "an fc or hidden_norm other than the kernel's (5 taps, BF16)"


def context_reasons(model: torch.nn.Module) -> list[str]:
    """Why the context write of ``model`` (whose layers have monokernels)
    cannot have its monokernel; empty when it can."""
    fc, norm = model.fc, model.hidden_norm
    if (
        not isinstance(fc.quant_method, UnquantizedLinearMethod)
        or fc.weight.dtype != torch.bfloat16
        or tuple(fc.weight.shape) != (HIDDEN, 5 * HIDDEN)
        or norm.weight.dtype != torch.bfloat16
        or abs(norm.variance_epsilon - RMS_NORM_EPS) >= 1e-12
    ):
        return [CONTEXT_OTHER]
    return []


def context_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The context write's weights by the roles of
    :func:`prepare_muse_spark_draft_context`."""
    tensors = {"fc": model.fc.weight.data, "hidden_norm": model.hidden_norm.weight.data}
    for i, layer in enumerate(model.layers):
        tensors[f"qkv_{i}"] = layer.self_attn.qkv_proj.weight.data
        tensors[f"k_norm_{i}"] = layer.self_attn.k_norm.weight.data
    return tensors


__all__ = [
    "BLOCK_ROWS",
    "DFlash2Monokernels",
    "DraftHeadMonokernel",
    "DraftMonokernelSteps",
    "enable_draft_monokernels",
    "head_reasons",
    "mailbox_scratch",
    "layer_tensors",
    "model_reasons",
    "takes_head",
    "takes_monokernels",
]
