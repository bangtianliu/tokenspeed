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

"""Stuck-poll records of Muse Spark layer launches, decoded under the plans
the launches ran (``monokernel.stuck``).

The serving seam describes its launches in a sidecar next to a file-backed
record buffer (``<file>.json``): each served layer's kind and the ``(rows,
chain)`` of every launch it compiled. Then, after a run that trapped::

    python -m tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.stuck_report FILE...
"""

from __future__ import annotations

import json
import sys

import torch
from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.layer import layer_plan


def plans_of(layers: dict[int, dict], launches: list[tuple[int, int]]):
    """``plan_of(layer)`` for :func:`stuck.decode`: the plan of each
    compiled ``(rows, chain)`` launch of layer ``layer`` (its kind from
    ``layers``: ``{"full": bool, "bf16_experts": bool}``). Every variant of
    a row count lays the scratch out alike, so one variant serves."""

    def plan_of(layer: int):
        kind = layers.get(layer)
        if kind is None:
            return []
        return [
            layer_plan(
                kind["full"],
                rows,
                chain,
                False,
                True,
                False,
                kind["bf16_experts"],
                False,
            )
            for rows, chain in launches
        ]

    return plan_of


def describe(t: torch.Tensor, layers: dict[int, dict], launches) -> list[str]:
    """The records of buffer ``t`` decoded under the launches' plans."""
    return stuck.decode(t, plans_of(layers, [tuple(x) for x in launches]))


def describe_file(path: str) -> list[str]:
    """The records of a file-backed buffer, with its sidecar's plans."""
    with open(f"{path}.json") as f:
        side = json.load(f)
    layers = {int(k): v for k, v in side["layers"].items()}
    return describe(stuck.read_file(path), layers, side["launches"])


def main(paths: list[str]) -> int:
    for path in paths:
        for line in describe_file(path) or [f"{path}: no stuck poll"]:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
