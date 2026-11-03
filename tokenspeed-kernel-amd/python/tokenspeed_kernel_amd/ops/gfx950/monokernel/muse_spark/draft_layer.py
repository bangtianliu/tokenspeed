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

"""A whole DFlash2 draft layer of Muse Spark as one launch, one TP8 rank.

The attention half (``draft_attn``) and then the MLP half (``draft_mlp``) in
one plan. Two hand-offs change, since a CTA never reads another CTA's store of
the same launch:

* the attention half's last stage (afin) puts its partial into the owner
  ranks' ``red`` slots itself, in place of the MLP half's scat;
* the residual after the input norm stays in the registers of the CTA that
  formed it: isum1 and sum1 have the same tasks on the same CTAs, and sum1
  takes isum1's residual (``draft_mlp.sum_norm_regs``). Only the residual
  after the attention sublayer is stored, as the layer's output with the MLP
  half's partial.
"""

from __future__ import annotations

from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    draft_attn,
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage

PEERS = draft_attn.PEERS + draft_mlp.PEERS


def regions(splits: int) -> tuple[tuple[str, int], ...]:
    return draft_attn.regions(splits) + draft_mlp.regions()


def stages(bases: dict[str, int], splits: int, rows: int) -> tuple[Stage, ...]:
    """The layer's stages of ``rows`` token rows (one block), placed by
    ``bases`` (both halves' names; ``isum`` and ``sum`` must match)."""
    if bases["isum"] != bases["sum"]:
        raise ValueError(
            "isum and sum must share CTAs: sum takes isum's residual from registers"
        )
    attn = [
        (
            Stage(st.name, st.tasks, st.base, reads=st.reads, writes=("red",))
            if st.name == "afin"
            else st
        )
        for st in draft_attn.stages(
            {k: v for k, v in bases.items() if k in _ATTN_NAMES}, splits, rows
        )
    ]
    mlp_bases = {"scat": 0, **{k: v for k, v in bases.items() if k in _MLP_NAMES}}
    mlp = [st for st in draft_mlp.stages(mlp_bases, rows) if st.name != "scat"]
    return tuple(attn + mlp)


_ATTN_NAMES = (
    "iscat",
    "isum",
    "acoef",
    "aconv0",
    "qkv",
    "head",
    "attn",
    "merge",
    "oproj",
    "afin",
)
_MLP_NAMES = ("sum", "coef", "conv0", "gu", "act", "down", "fin")
