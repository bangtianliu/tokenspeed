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

"""The startup dummy extends reach every row bucket a later chunk can land in.

CPU only: the sizes are host arithmetic.
"""

import os
import sys

import pytest

from tokenspeed.runtime.execution.model_executor import warmup_extend_sizes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# This suite is CPU-only, but the runtime suite is consumed by AMD CI too.
register_cuda_ci(est_time=1, suite="runtime-1gpu")


@pytest.mark.parametrize("budget", [8192, 16384, 4096, 192])
def test_the_extends_cover_every_power_of_two_row_bucket_and_an_odd_count(budget):
    sizes = warmup_extend_sizes(budget)
    assert sizes[0] == budget
    # An extend in every power-of-two row bucket from 8 tokens up to the
    # budget's (kernels bucket rows by the next power of two).
    buckets = {1 << (t - 1).bit_length() for t in sizes}
    bucket = 8
    while bucket <= budget:
        assert bucket in buckets, (bucket, sizes)
        bucket *= 2
    # A short extend inside the small-row routes (at most 16 rows) whose rows
    # are not a multiple of 16: a separate GEMM specialization.
    assert any(t <= 16 and t % 16 for t in sizes)
    assert all(t <= budget for t in sizes) and len(sizes) == len(set(sizes))


def test_a_budget_of_eight_tokens_runs_only_itself():
    assert warmup_extend_sizes(8) == [8]
