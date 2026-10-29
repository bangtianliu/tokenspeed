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

"""Time one Muse Spark TP8 layer launch on 1, 2, 4 or 8 GPUs.

Each GPU runs the TP8 kernel of one rank with the TP8 shapes of that rank.
The kernel keeps RANK and WORLD at 8. Physical GPU p runs logical rank p, and
logical ranks ``--gpus`` to 7 are absent:

* The heap bases of the absent ranks point at a local ghost buffer, so puts
  to an absent rank land there.
* Before each launch, a fill kernel writes the segment of each absent rank in
  every peer region with the tag of this launch, as if that rank had already
  put. All peer regions are rank-major with equal segments. The payloads are
  the segment of rank 0, so sums, norms and routing see realistic values.
* The present ranks still wait for the real puts of each other, over the
  links when ``--gpus`` is more than 1.

The numbers are for iteration, not for claims: 1 GPU has no link latency,
and the absent ranks never arrive late. Confirm a kept change on 8 GPUs.

Run from the repository root (each line is one example):

    PYTHONPATH=tokenspeed-kernel/test HIP_VISIBLE_DEVICES=1 \\
        python tokenspeed-kernel/test/amd/gfx950/ops/monokernel/muse_spark_layer_bench.py --gpus 1
    PYTHONPATH=tokenspeed-kernel/test HIP_VISIBLE_DEVICES=0,1,2,3 \\
        python tokenspeed-kernel/test/amd/gfx950/ops/monokernel/muse_spark_layer_bench.py --gpus 4 \\
        --trace-out /tmp/trace --out-save /tmp/new.pt

The first line of the output is the time per launch. The table after it is the
trace of rank 0: for each stage, the first and last time (us, relative to the
first event) at which a wave starts, has its inputs, has computed, and ends.
``--out-save`` stores outputs of a fixed launch, so two builds can be checked
for bit identity with ``--compare``.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import muse_spark_reference as ref  # noqa: E402
import test_muse_spark_layer_gfx950 as layer_test  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import tl, triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import plan as plan_mod  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import layer  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

TP, HIDDEN, OWNED, POOL_OWNED, EXPERTS = 8, 8192, 1024, 512, 16
# Launches per graph: one per layer copy, so the timings average over them.
LAUNCHES = 40
# The trace buffer: [CTA, 4 points per stage, wave] in 10 ns ticks.
TRACE_SHAPE, TICK_US = (256, 128, 8), 0.01
POINTS = ("start", "ready", "computed", "end")
SAVED_REGIONS = ("lg", "lg2", "ss1", "ss2", "hp", "bq")


@triton.jit(do_not_specialize=["lay", "n"])
def _fill_absent_ranks(
    scratch, idx, bits, step, lay, n, SLOTS: tl.constexpr, BLOCK: tl.constexpr
):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tag = ((tl.load(step).to(tl.int64) * SLOTS + lay + 1) & 0xFFFFFFFF) << 32
    tl.store(
        scratch + tl.load(idx + o, mask=m), tl.load(bits + o, mask=m) | tag, mask=m
    )


def _peer_slices(plan, parity: int):
    """(offset, segment) of copy ``parity`` of each peer region."""
    sizes = dict(plan.regions)
    out = []
    for r in plan.peers:
        assert sizes[r] % TP == 0, (r, sizes[r])
        out.append((plan.offset(r) + parity * plan.peer_copy(r), sizes[r] // TP))
    return out


def absent_index(plan, parity: int, gpus: int) -> torch.Tensor:
    """Pair indices of the absent ranks' segments in copy ``parity``."""
    return torch.cat(
        [
            torch.arange(off + gpus * seg, off + TP * seg)
            for off, seg in _peer_slices(plan, parity)
        ]
    )


def rank0_payloads(plan, parity: int, gpus: int, scratch) -> torch.Tensor:
    """Payloads (low 32 bits) of rank 0's segments, once per absent rank."""
    return (
        torch.cat(
            [
                scratch[off : off + seg].repeat(TP - gpus)
                for off, seg in _peer_slices(plan, parity)
            ]
        )
        & 0xFFFFFFFF
    )


def _operands(rank: int, length: int, dev):
    sh = layer_test._shared(42, length)
    w = layer_test._rank_weights(rank, length)
    w13, w13s, w2, w2s = ref.preprocessed_experts(rank, EXPERTS, dev)
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    own_pool = slice(rank * POOL_OWNED, (rank + 1) * POOL_OWNED)
    last = length - 1
    slot = int(sh["table"][last // 64]) * 64 + last % 64
    h_next = torch.empty(OWNED, device=dev)
    a_out = torch.empty(HIDDEN, dtype=torch.bfloat16, device=dev)
    attn_ops = (
        bf16_tiles(w["w_qkvg"]).to(dev),
        sh["a_in"].to(dev),
        torch.full((1,), length, dtype=torch.int32, device=dev),
        torch.full((1,), slot, dtype=torch.int32, device=dev),
        sh["table"].to(dev),
        w["k_cache"].to(dev),
        w["v_cache"].to(dev),
        ref.rope_table(length + 1).to(dev),
        bf16_tiles(w["w_o"]).to(dev),
        sh["h"][own].to(dev),
        sh["alpha1"][own].to(dev),
        sh["beta1"][own].to(dev),
        sh["pre_gain"][own].to(dev),
    )
    moe_ops = (
        sh["router"][:, own].contiguous().to(dev),
        sh["bias"].to(dev),
        bf16_tiles(w["w_pool"]).to(dev),
        w13,
        w13s,
        w13.stride(0),
        w13s.stride(0),
        w13s.stride(1),
        w13s.stride(2),
        w2,
        w2s,
        w2.stride(0),
        w2s.stride(0),
        w2s.stride(1),
        w2s.stride(2),
        sh["pre_exp_gain"].to(dev),
        sh["pen_gain"][own_pool].to(dev),
        bf16_tiles(w["w_post"]).to(dev),
        sh["post_gain"][own].to(dev),
        sh["alpha2"][own].to(dev),
        sh["beta2"][own].to(dev),
        sh["in_gain"][own].to(dev),
        h_next,
        a_out,
    )
    return attn_ops, moe_ops, h_next, a_out


def _median_us(graph, replays: int) -> float:
    """Median time of one replay, per launch. One disturbed replay does not move it."""
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(replays + 1)]
    ev[0].record()
    for i in range(replays):
        graph.replay()
        ev[i + 1].record()
    torch.cuda.synchronize()
    times = sorted(ev[i].elapsed_time(ev[i + 1]) for i in range(replays))
    dist.barrier()
    return times[replays // 2] * 1e3 / LAUNCHES


def print_trace(t: torch.Tensor, names: list[str]) -> None:
    t = t.double()
    base = t[t > 0].min()
    for i, name in enumerate(names):
        cells = []
        for pt, label in enumerate(POINTS):
            v = t[:, 4 * i + pt, :]
            v = v[v > 0]
            if v.numel():
                lo = (v.min() - base).item() * TICK_US
                hi = (v.max() - base).item() * TICK_US
                cells.append(f"{label} {lo:5.1f}-{hi:5.1f}")
        print(f"  {name:7s} " + " | ".join(cells))


def worker(rank: int, port: int, args) -> None:
    gpus = args.gpus
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=gpus,
        timeout=timedelta(seconds=900),
    )
    dev = torch.device(f"cuda:{rank}")
    ctx = get_or_create_context(heap_size=256 << 20)
    group = IrisGroup(ctx, dist.group.WORLD, rank, dev)
    plan = layer.layer_plan(args.full)
    scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
    ghost = torch.zeros_like(scratch)
    bases = torch.empty(TP, dtype=torch.int64, device=dev)
    bases[:gpus] = group.bases.to(dev)
    bases[gpus:] = int(group.bases[rank]) + (ghost.data_ptr() - scratch.data_ptr())
    attn_ops, moe_ops, h_next, a_out = _operands(rank, args.length, dev)
    step = torch.full((1,), 100, dtype=torch.int32, device=dev)
    idx = [absent_index(plan, p, gpus).to(dev) for p in (0, 1)]
    bits = [torch.zeros(len(i), dtype=torch.int64, device=dev) for i in idx]

    def fill(lay: int) -> None:
        # The tag of this launch on the absent ranks' slots of copy ``lay & 1``.
        if gpus == TP:
            return
        n = idx[lay & 1].numel()
        _fill_absent_ranks[(triton.cdiv(n, 1024),)](
            scratch,
            idx[lay & 1],
            bits[lay & 1],
            step,
            lay,
            n,
            plan_mod.LAYER_SLOTS,
            1024,
        )

    # 512 MiB: more than L2 and the 256 MB MALL, so a cold launch reads HBM.
    flush = torch.zeros(1 << 29, dtype=torch.int8, device=dev)
    flush_sink = torch.zeros(1, dtype=torch.int64, device=dev)

    def do_flush() -> None:
        if args.flush == "read":
            torch.sum(flush.view(torch.int64), dim=0, keepdim=True, out=flush_sink)
        else:
            flush.zero_()

    trace = torch.zeros(TRACE_SHAPE, dtype=torch.int64, device=dev)
    group.finish_prepare()

    def launch(lay: int, trace_buf=None) -> None:
        fill(lay)
        layer.launch_layer(
            scratch, step, lay, bases, attn_ops, moe_ops, trace_buf, rank, args.full
        )

    def run(cold: bool) -> None:
        step.add_(1)
        for lay in range(1, LAUNCHES + 1):
            if cold:
                do_flush()
            launch(lay)

    stream = torch.cuda.Stream()
    graphs = {}
    with torch.cuda.stream(stream):
        if gpus < TP:
            # Values that depend on other ranks' data (the reduced o_proj, its
            # norms) come from launches that read the previous copies. Repeat
            # until both copies hold consistent payloads, or old and new builds
            # give different outputs for no reason.
            for _ in range(args.settle):
                for lay in (1, 2):
                    step.add_(1)
                    launch(lay)
                    torch.cuda.synchronize()
                snapshot = scratch.clone()
                for p in (0, 1):
                    bits[p].copy_(rank0_payloads(plan, p, gpus, snapshot))
        run(False)
        torch.cuda.synchronize()
        for name, cold in (("warm", False), ("cold", True)):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                run(cold)
            graphs[name] = g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            step.add_(1)
            for lay in range(1, LAUNCHES + 1):
                do_flush()
        graphs["flush"] = g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            step.add_(1)
            for lay in range(1, LAUNCHES + 1):
                fill(lay)
        graphs["fill"] = g
    torch.cuda.synchronize()
    dist.barrier()
    us = {name: _median_us(g, args.replays) for name, g in graphs.items()}
    if rank == 0:
        f = us["fill"] if gpus < TP else 0.0
        kind = "full" if args.full else "sliding"
        print(
            f"{gpus} GPU {kind} layer at {args.length} tokens: "
            f"warm {us['warm'] - f:.1f} us, cold {us['cold'] - us['flush'] - f:.1f} us "
            f"per launch (graph; flush {us['flush']:.1f}, fill {f:.1f} us subtracted)"
        )
    if args.out_save:
        dist.barrier()
        torch.cuda.synchronize()
        step.fill_(500)
        launch(1)
        torch.cuda.synchronize()
        if rank == 0:
            o = plan.offset("route")
            route = (scratch[o : o + 16] & 0xFFFFFFFF).to(torch.int32).cpu()
            regions = {}
            for r, n in plan.regions:
                if r in SAVED_REGIONS:
                    # Launch 1 writes copy 1 of a peer region.
                    off = plan.offset(r) + (plan.peer_copy(r) if r in plan.peers else 0)
                    regions[r] = (scratch[off : off + n] & 0xFFFFFFFF).to(torch.int32)
            torch.save(
                {
                    "h_next": h_next.cpu(),
                    "a_out": a_out.cpu(),
                    "ids": route[:8].clone(),
                    "weights": route[8:].view(torch.float32).clone(),
                    "regions": {r: v.cpu() for r, v in regions.items()},
                },
                args.out_save,
            )
    dist.barrier()
    do_flush()
    torch.cuda.synchronize()
    dist.barrier()
    step.add_(1)
    launch(1, trace)
    torch.cuda.synchronize()
    t = trace.cpu()
    names = [s.name for s in plan.stages]
    if args.trace_out:
        torch.save({"t": t, "names": names}, f"{args.trace_out}.{rank}")
    if rank == 0:
        print_trace(t, names)
    dist.barrier()
    dist.destroy_process_group()


def compare(a_path: str, b_path: str) -> bool:
    a, b = torch.load(a_path), torch.load(b_path)
    same = all(torch.equal(a[k], b[k]) for k in ("h_next", "a_out", "ids", "weights"))
    return same and all(
        torch.equal(a["regions"][r], b["regions"][r]) for r in a["regions"]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), help="present ranks")
    p.add_argument(
        "--flush",
        choices=("read", "write"),
        default="read",
        help="how a cold launch evicts the caches before it",
    )
    p.add_argument("--length", type=int, default=8700, help="tokens in the KV cache")
    p.add_argument("--full", action="store_true", help="a full-attention layer")
    p.add_argument("--replays", type=int, default=15)
    p.add_argument(
        "--settle",
        type=int,
        default=16,
        help="passes that make the absent ranks' copies consistent",
    )
    p.add_argument("--trace-out", help="save each rank's trace to PREFIX.<rank>")
    p.add_argument("--out-save", help="save rank 0's outputs of a fixed launch")
    p.add_argument(
        "--view", nargs="+", metavar="TRACE", help="print saved traces and exit"
    )
    p.add_argument(
        "--compare", nargs=2, metavar="OUT", help="check two --out-save files"
    )
    args = p.parse_args()
    if args.view:
        for path in args.view:
            d = torch.load(path)
            print(f"== {path}")
            print_trace(d["t"], d["names"])
        return
    if args.compare:
        print(f"outputs identical: {compare(*args.compare)}")
        return
    if args.gpus is None:
        p.error("--gpus is required to time a layer")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=args.gpus, join=True)


if __name__ == "__main__":
    main()
