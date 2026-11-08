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

"""The Iris column all-gather concatenates every rank's feature shard."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def test_large_tp8_shards_take_two_subgroups_per_program():
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        column_gather_num_warps,
    )

    assert column_gather_num_warps(256 * 896 * 2, 8) == 1
    assert column_gather_num_warps(848 * 288 * 2, 8) == 1
    assert column_gather_num_warps(848 * 512 * 2, 8) == 2
    assert column_gather_num_warps(8192 * 896 * 2, 8) == 2
    assert column_gather_num_warps(8192 * 896 * 2, 4) == 1


@pytest.mark.parametrize("local_hidden_size", [512, 1024])
def test_warmup_rows_cover_every_subgroup_count(local_hidden_size):
    # Muse Spark's gathered projections: 4096 and 8192 columns over 8 ranks.
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        column_gather_num_warps,
        column_gather_warmup_rows,
    )

    rows = column_gather_warmup_rows(local_hidden_size, 8, 2, 65)
    assert all(r >= 65 for r in rows)
    warps = {column_gather_num_warps(r * local_hidden_size * 2, 8) for r in rows}
    assert warps == {1, 2}
    # A long extend's rows take one of them.
    assert column_gather_num_warps(8192 * local_hidden_size * 2, 8) in warps


def _column_gather_warmup_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        IrisAllGather,
        iris_pull_allgather_columns,
    )
    from tokenspeed_kernel.ops.communication.iris import create_iris_state
    from tokenspeed_kernel.ops.tuning import kernel_warmup
    from utils import assert_no_triton_compile

    state = create_iris_state(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 7168,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
    )
    gather = IrisAllGather(state.producer)
    state.peers.finish_prepare()

    def run(rows: int, width: int) -> None:
        shard = torch.full((rows, width), float(rank), device=device).to(torch.bfloat16)
        result = gather.run(shard)
        expected = (
            torch.arange(8, device=device).repeat_interleave(width).to(torch.bfloat16)
        )
        assert torch.equal(result, expected.expand(rows, -1))
        torch.cuda.synchronize()
        dist.barrier()

    widths = (512, 1024)  # Muse Spark's 4096 and 8192 columns over 8 ranks
    # One gather a width inside kernel_warmup: the op compiles its other
    # variants itself (column_gather_warmup_rows).
    with kernel_warmup(enabled=True):
        for width in widths:
            run(65, width)
    # Both sides of the 768 KiB switch, and a long extend's (8192 rows: >= 8 MiB
    # a rank), compile nothing more.
    with assert_no_triton_compile(iris_pull_allgather_columns):
        for width in widths:
            for rows in (65, 100, 383, 384, 767, 768, 1000, 4096, 8192):
                run(rows, width)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_warmup_rows_compile_every_column_gather_a_long_extend_takes():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_column_gather_warmup_worker, args=(port,), nprocs=8, join=True)


def _column_gather_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris.all_gather import IrisAllGather
    from tokenspeed_kernel.ops.communication.iris import create_iris_state

    state = create_iris_state(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 7168,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
    )
    gather = IrisAllGather(state.producer)
    state.peers.finish_prepare()
    gen = torch.Generator(device=device)
    # One and two subgroups per program, including a partial last tile.
    for rows, width in (
        (4, 896),
        (256, 896),
        (848, 896),
        (8192, 896),
        (848, 288),
        (2047, 512),
    ):
        gen.manual_seed(rows * width + rank)
        shard = torch.randn((rows, width), device=device, generator=gen).to(
            torch.bfloat16
        )
        parts = [torch.empty_like(shard) for _ in range(8)]
        dist.all_gather(parts, shard, group=group)
        expected = torch.cat(parts, dim=1)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 5:
            torch.cuda._sleep(2_000_000)
        result = gather.run(shard)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.cuda.synchronize()
        dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_column_gather_matches_the_concatenated_shards():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_column_gather_worker, args=(port,), nprocs=8, join=True)
