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

"""The Muse Spark decode monokernel seam: which forwards take the kernels,
and which models the kernels serve. CPU only."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.muse_spark_config import (  # noqa: E402
    MuseSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.models.muse_spark_monokernel import (  # noqa: E402
    NOT_TP8,
    OTHER_CONSTANTS,
    OTHER_SHAPES,
    model_reasons,
    takes_monokernels,
)


def _ctx(mode: ForwardMode, bs: int, rows: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=bs if mode.is_extend() else 0,
        input_num_tokens=rows,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(0, 0, bs, 1),
    )


def test_one_token_decode_steps_of_one_request_take_the_kernels() -> None:
    """A launch computes one token row: prefill, several requests and
    speculative verify rows keep the regular path."""
    assert takes_monokernels(1, _ctx(ForwardMode.DECODE, 1, 1))
    assert not takes_monokernels(2, _ctx(ForwardMode.DECODE, 2, 2))
    assert not takes_monokernels(4, _ctx(ForwardMode.DECODE, 1, 4))
    assert not takes_monokernels(1, _ctx(ForwardMode.EXTEND, 1, 1))
    assert not takes_monokernels(1, _ctx(ForwardMode.IDLE, 1, 1))


def _text_config(**overrides) -> dict:
    """Muse Spark Open's decoder (four of its 62 layers)."""
    layer_types = ["sliding_attention", "full_attention"] * 2
    config = {
        "vocab_size": 202048,
        "hidden_size": 8192,
        "head_dim": 64,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "num_hidden_layers": len(layer_types),
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 5.4730064863875,
        "output_multiplier": 0.17677669529663687,
        "residual_gate_temperature": 0.3,
        "sliding_window": 2048,
        "layer_types": layer_types,
        "layer_rope_theta": [
            0.0 if t == "full_attention" else 500000.0 for t in layer_types
        ],
        "mlp_layer_types": ["sparse"] * len(layer_types),
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "moe_hidden_size": 4096,
        "moe_intermediate_size": 4096,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "routed_scaling_factor": 1.0,
    }
    config.update(overrides)
    return config


def _config(**overrides) -> MuseSparkConfig:
    return MuseSparkConfig(
        text_config=_text_config(**overrides),
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "layer_types": ["full_attention"],
            "layer_norm_eps": 1e-5,
            "merge_size": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "max_position_embeddings": 4,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
        },
        audio_config={},
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=128,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
    )


def _mapping(moe_tp: int, moe_ep: int) -> Mapping:
    return Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        dense_tp_size=8,
        dense_dp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=8,
        linear_attn_tp_size=8,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=8,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def test_muse_spark_open_at_tp8_has_monokernels() -> None:
    assert model_reasons(_config(), _mapping(8, 1)) == []


def test_expert_parallel_moe_keeps_the_regular_path() -> None:
    """A launch computes every expert's 1/8 slice: experts split across
    ranks have no monokernel."""
    assert model_reasons(_config(), _mapping(4, 2)) == [NOT_TP8]


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"output_multiplier": 0.125}, OTHER_CONSTANTS),
        ({"qk_scale_factor": 4.0}, OTHER_CONSTANTS),
        ({"num_local_experts": 128}, OTHER_SHAPES),
        ({"moe_hidden_size": 2048}, OTHER_SHAPES),
    ],
)
def test_other_baked_in_values_keep_the_regular_path(override, reason) -> None:
    """The kernel bakes Muse Spark Open's dimensions and constants in: a
    checkpoint with other values has no monokernel."""
    assert model_reasons(_config(**override), _mapping(8, 1)) == [reason]
