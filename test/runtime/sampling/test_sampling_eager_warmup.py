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

"""The Triton sampling backend's startup warmup compiles every eager variant:
after it, requests with any top_k / top_p, one token or a verify window a
request, with or without draft distributions (speculative sampling), at any
batch size, compile nothing on the serving path."""

from __future__ import annotations

import importlib.util
import inspect
import os
import pkgutil

import pytest
import torch

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB = 202048  # a served vocabulary: the routes for large vocabularies
POOL = 16
MAX_BS = 16
WIDTH = 8  # a speculative verify window
CANDIDATES = 16  # the DFlash2 draft head's candidates a step


def _compile_helper():
    path = os.path.join(_TEST_DIR, "..", "..", "..", "tokenspeed-kernel", "test")
    spec = importlib.util.spec_from_file_location(
        "sampling_warmup_kernel_test_utils", os.path.join(path, "utils.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _sampling_kernels():
    """Every Triton kernel of the sampling kernel package."""
    import tokenspeed_kernel.ops.sampling.triton as pkg

    kernels = []
    for info in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for _, obj in inspect.getmembers(mod):
            if hasattr(obj, "_do_compile") and obj not in kernels:
                kernels.append(obj)
    return kernels


def _backend():
    """A speculative-sampling backend (it verifies against draft q when the
    batch carries one), and the draft q table as RuntimeStates allocates it."""
    from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend

    backend = TritonSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=MAX_BS,
            max_draft_tokens_per_req=WIDTH,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
        )
    )
    draft_q = DraftTreeQ.zeros(POOL + 1, WIDTH - 1, CANDIDATES, WIDTH, "cuda")
    return backend, draft_q


def _step(backend, bs, width, temperature, top_k, top_p, draft_q, dtype):
    """A request batch's prepare_step and its sample (one token a request)
    or verify (a window of ``width``), as serving runs them eagerly."""
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    rids = [f"req-{bs}-{width}-{top_k}-{i}" for i in range(bs)]
    sps = []
    for rid in rids:
        sp = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)
        sp.resolve_seed(rid)
        sp.normalize(None)
        sps.append(sp)
    # Slots past the warmup's first ones: any slot a request holds.
    slots = list(range(POOL - bs, POOL))
    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=slots,
        sampling_params_list=sps,
        num_tokens_per_req=width,
    )
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor(slots, dtype=torch.int64, device="cuda"),
        valid_cache_lengths=torch.arange(
            50, 50 + POOL + 1, dtype=torch.int32, device="cuda"
        ),
        vocab_size=VOCAB,
        device="cuda",
        draft_q=draft_q,
    )
    logits = torch.randn(bs * width, VOCAB, dtype=torch.float32, device="cuda")
    logits = logits.to(dtype)  # a BF16 LM head hands the sampler BF16 logits
    out = LogitsProcessorOutput(next_token_logits=logits)
    if width == 1:
        tokens, _ = backend.sample(out, info)
    else:
        candidates = torch.randint(0, VOCAB, (bs, width), dtype=torch.int32)
        tokens, _ = backend.verify(out, info, candidates.cuda(), tree=None)
    assert ((tokens >= 0) & (tokens < VOCAB)).all()


@requires_cuda
def test_warmed_backend_compiles_nothing_while_serving():
    helper = _compile_helper()
    backend, draft_q = _backend()
    backend.warmup_eager(draft_q)
    torch.cuda.synchronize()
    with helper.assert_no_triton_compile(*_sampling_kernels()):
        for temperature, top_k, top_p in (
            (0.0, -1, 1.0),  # greedy: top_k 1 after normalize, TOPK_PAD 64
            (0.8, -1, 1.0),
            (0.8, -1, 0.95),
            (0.8, 20, 0.95),  # TOPK_PAD 64
            (0.8, 100, 0.95),  # TOPK_PAD 128
        ):
            for bs, width, q, dtype in (
                (3, 1, None, torch.float32),
                (2, 1, None, torch.bfloat16),
                (1, WIDTH, None, torch.float32),
                (5, WIDTH, None, torch.bfloat16),  # between the ends: odd rows
                (MAX_BS, WIDTH, None, torch.float32),  # QRITA verify (128 rows)
                (5, WIDTH, draft_q, torch.float32),  # rejection against draft q
                (MAX_BS, WIDTH, draft_q, torch.bfloat16),
            ):
                _step(backend, bs, width, temperature, top_k, top_p, q, dtype)
        torch.cuda.synchronize()
    # The warmup reads the draft q table only: it still holds no proposal.
    assert (draft_q.slot == -1).all() and not draft_q.q.any()
    # A first real request on a slot the warmup used resets the slot's pools
    # (its rid differs), so it never samples with the warmup's parameters.
    sp = SamplingParams(temperature=0.7, top_k=7, top_p=0.9)
    sp.resolve_seed("first-request")
    sp.normalize(None)
    backend.prepare_step(
        request_ids=["first-request"],
        request_pool_indices=[0],
        sampling_params_list=[sp],
        num_tokens_per_req=1,
    )
    assert int(backend.speculative_sampling_pools().top_k[0]) == 7
