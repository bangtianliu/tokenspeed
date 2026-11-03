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

"""The DFlash2 draft monokernel seam: which draft forwards take the kernels,
and which drafts the kernels serve. CPU only (the drafts are built on the meta
device)."""

from __future__ import annotations

import os
import sys

import pytest
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel  # noqa: E402
from tokenspeed.runtime.models.dflash2_monokernel import (  # noqa: E402
    BLOCK_ROWS,
    NOT_BF16_KV,
    NOT_TP8,
    OTHER_ATTENTION,
    OTHER_SHAPES,
    model_reasons,
    takes_monokernels,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402


def _ctx(mode: ForwardMode, bs: int, rows: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(0, 0, bs, BLOCK_ROWS),
    )


def test_the_block_of_one_request_takes_the_kernels() -> None:
    """A launch computes one request's block of 8 rows; several requests
    keep the regular path."""
    assert takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.DECODE, 1, BLOCK_ROWS))
    assert not takes_monokernels(2 * BLOCK_ROWS, _ctx(ForwardMode.DECODE, 2, 16))
    assert not takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.EXTEND, 1, 8))
    assert not takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.IDLE, 1, 8))


def _config(**overrides) -> PretrainedConfig:
    """Muse Spark Open's DFlash2 draft."""
    layers = overrides.pop("num_hidden_layers", 5)
    config = {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "is_causal": False,
        "layer_types": ["sliding_attention"] * layers,
        "max_position_embeddings": 4096,
        "num_attention_heads": 64,
        "num_hidden_layers": layers,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 500000, "rope_type": "default"},
        "rope_theta": 500000,
        "sliding_window": 2048,
        "vocab_size": 1024,
        "dflash_config": {
            "attention_mode": "gqa",
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "final_logit_softcapping": 20,
            "mask_token_id": 2,
            "output_multiplier": 0.1767766952966369,
            "selector_rank": 16,
            "selector_top_k": 16,
            "target_layer_ids": [1, 16, 30, 44, 59],
        },
    }
    config.update(overrides)
    return PretrainedConfig(**config)


def _mapping(tp: int) -> Mapping:
    return Mapping(
        rank=0,
        world_size=tp,
        attn_tp_size=tp,
        attn_dp_size=1,
        dense_tp_size=tp,
        dense_dp_size=1,
        moe_tp_size=tp,
        moe_ep_size=1,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=tp,
        linear_attn_tp_size=tp,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=tp,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def _draft(config: PretrainedConfig, tp: int = 8) -> DFlash2DraftModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return DFlash2DraftModel(config, _mapping(tp))
    finally:
        torch.set_default_dtype(previous)


def test_muse_spark_opens_draft_at_tp8_has_monokernels() -> None:
    assert model_reasons(_draft(_config())) == []


def test_a_tp4_draft_keeps_the_regular_path() -> None:
    assert model_reasons(_draft(_config(), tp=4)) == [NOT_TP8]


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"num_hidden_layers": 4}, OTHER_SHAPES),
        ({"intermediate_size": 14336}, OTHER_SHAPES),
        # A 4096-token window, and a full-attention layer.
        ({"sliding_window": 4096}, OTHER_ATTENTION),
        (
            {"layer_types": ["sliding_attention"] * 4 + ["full_attention"]},
            OTHER_ATTENTION,
        ),
        ({"rms_norm_eps": 1e-6}, OTHER_ATTENTION),
    ],
)
def test_other_drafts_keep_the_regular_path(overrides, reason) -> None:
    assert reason in model_reasons(_draft(_config(**overrides)))


@pytest.mark.parametrize(
    "drafter_attention_backend, reasons", [(None, []), ("trtllm", [NOT_BF16_KV])]
)
def test_an_fp8_server_keeps_the_drafts_cache_bf16_unless_trtllm_attends(
    monkeypatch, drafter_attention_backend, reasons
) -> None:
    """The kernels read and write BF16 keys and values. Under an FP8
    --kv-cache-dtype the DFLASH draft's own cache stays BF16, except when it
    attends with trtllm."""
    monkeypatch.setitem(global_server_args_dict, "kv_cache_dtype", "fp8_e4m3")
    monkeypatch.setitem(global_server_args_dict, "speculative_algorithm", "DFLASH")
    monkeypatch.setitem(
        global_server_args_dict, "drafter_attention_backend", drafter_attention_backend
    )
    assert model_reasons(_draft(_config())) == reasons
