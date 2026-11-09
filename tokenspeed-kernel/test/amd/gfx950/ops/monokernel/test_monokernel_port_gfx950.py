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

"""The rendezvous port of a multi-process monokernel run: an explicit
``MASTER_PORT`` wins, so concurrent runs on one host can be kept apart."""

from __future__ import annotations

import socket

from monokernel_testing import rendezvous_port


def test_rendezvous_port_takes_master_port(monkeypatch):
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("MASTER_PORT", str(port))
    assert rendezvous_port() == port


def test_rendezvous_port_without_master_port_is_free(monkeypatch):
    monkeypatch.delenv("MASTER_PORT", raising=False)
    port = rendezvous_port()
    # The port it picks is free: a server socket can listen on it.
    with socket.socket() as s:
        s.bind(("localhost", port))
        s.listen()
