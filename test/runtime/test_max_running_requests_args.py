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

"""``--max-running-requests``: the scheduler's running bound, below the
``--max-num-seqs`` that sizes the device side."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)

from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def test_unset_runs_as_many_requests_as_max_num_seqs():
    assert ServerArgs(model="x", max_num_seqs=64).max_running_requests == 64


def test_a_lower_cap_keeps_max_num_seqs():
    args = ServerArgs(model="x", max_num_seqs=64, max_running_requests=1)
    assert (args.max_num_seqs, args.max_running_requests) == (64, 1)


def test_cli_sets_the_cap():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = ServerArgs.from_cli_args(
        parser.parse_args(
            ["--model", "x", "--max-num-seqs", "16", "--max-running-requests", "2"]
        )
    )
    assert (args.max_num_seqs, args.max_running_requests) == (16, 2)


@pytest.mark.parametrize("cap", [0, 65])
def test_refuses_a_cap_outside_one_to_max_num_seqs(cap: int):
    with pytest.raises(ValueError, match="--max-running-requests"):
        ServerArgs(model="x", max_num_seqs=64, max_running_requests=cap)


def test_refuses_fewer_requests_than_attention_dp_ranks():
    with pytest.raises(ValueError, match="attention-DP size"):
        ServerArgs(
            model="x",
            world_size=2,
            data_parallel_size=2,
            max_num_seqs=8,
            max_running_requests=1,
        )
