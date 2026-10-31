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

"""The startup report shared by the models' decode monokernel seams: which
layers keep the regular path, and why."""

from __future__ import annotations


def unserved_report(unserved: dict[int, list[str]]) -> str:
    """``unserved`` (layer index -> reasons) grouped by reason, for a log or an
    error: ``"reason: layers 0, 2; other reason: layer 5"``."""
    layers_by_reason: dict[str, list[int]] = {}
    for index, reasons in sorted(unserved.items()):
        for reason in reasons:
            layers_by_reason.setdefault(reason, []).append(index)
    return "; ".join(
        f"{reason}: layer{'s' if len(layers) > 1 else ''} "
        f"{', '.join(str(i) for i in layers)}"
        for reason, layers in layers_by_reason.items()
    )


def coverage(
    name: str, layers: int, unserved: dict[int, list[str]], *, required: bool
) -> str | None:
    """The startup report of a backbone's monokernel coverage.

    Args:
        name: The backbone's class name.
        layers: Its number of decoder layers.
        unserved: The reasons of each layer without a monokernel.
        required: Whether the backbone must have a monokernel layer.

    Returns:
        A warning that names the layers on the regular path and why, None when
        every layer has a monokernel.

    Raises:
        ValueError: ``required`` and no layer has a monokernel.
    """
    if not unserved:
        return None
    report = unserved_report(unserved)
    if len(unserved) == layers and required:
        raise ValueError(
            f"--decode-monokernel: no layer of {name} has a monokernel "
            f"({report}). Serve without --decode-monokernel."
        )
    served = layers - len(unserved)
    return (
        f"Decode monokernel: {served} of {layers} {name} layers; the rest keep "
        f"the regular path ({report})"
    )


__all__ = ["coverage", "unserved_report"]
