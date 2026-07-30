"""Opt-in `live` tests for the real network/subprocess paths CI can't run.

Skipped by default (see conftest.py); run locally with::

    pytest tests/test_live.py --run-live

Two genuinely-live paths are covered, both self-contained (a spawned subprocess
and a localhost HTTP server — no external services, no credentials):

- **live MCP ingestion** — ``load_mcp_tools_live`` connects over stdio to a real
  ``mcp`` server (``tests/fixtures/live_mcp_server.py``), lists its tools, and
  adapts them to OKT concepts. Requires the optional ``mcp`` extra.
- **live HTTP dispatch** — an ``interface: http`` tool is dispatched through the
  serving layer to a real ``http.server`` on localhost, proving the phase-3
  dispatch path makes an actual HTTP round-trip.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

LIVE_MCP_SERVER = Path(__file__).parent / "fixtures" / "live_mcp_server.py"


# ---------------------------------------------------------------------------
# live MCP ingestion: real stdio server <-> load_mcp_tools_live
# ---------------------------------------------------------------------------


def test_load_mcp_tools_live_roundtrip():
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")

    from mcp.client.stdio import StdioServerParameters

    from okts.adapters.mcp import load_mcp_tools_live
    from okts.core.validator import validate_concept

    params = StdioServerParameters(command=sys.executable, args=[str(LIVE_MCP_SERVER)])
    concepts = anyio.run(load_mcp_tools_live, params, "test-live")

    by_id = {c.id: c for c in concepts}
    assert {"test-live.echo", "test-live.add"} <= set(by_id)

    echo = by_id["test-live.echo"]
    assert echo.interface.value == "mcp"
    assert echo.target == "test-live"           # server name -> target
    assert echo.input_schema["type"] == "object"
    assert "text" in echo.input_schema["properties"]

    # everything adapted off a live server is OKF-conformant
    for concept in concepts:
        assert validate_concept(concept) == []


# ---------------------------------------------------------------------------
# live HTTP dispatch: OKTSService.call_tool -> HttpDispatcher -> localhost
# ---------------------------------------------------------------------------


class _EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body or b"{}")
        out = json.dumps({"ok": True, "received": payload}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):  # silence the default stderr logging
        pass


@pytest.fixture
def local_http_server():
    server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_live_http_dispatch_makes_real_request(local_http_server):
    import urllib.request

    from okts.core.model import Bundle, Interface, OKTConcept
    from okts.serve.dispatch import DispatcherRegistry, HttpDispatcher
    from okts.serve.mcp_server import NaiveFallbackRetriever
    from okts.serve.service import OKTSService

    def urllib_client(concept, args, credential):
        # a real HTTP client callable, exactly what a caller wires into
        # HttpDispatcher (the serving layer never embeds one itself).
        req = urllib.request.Request(
            local_http_server + "/echo",
            data=json.dumps(args).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    concept = OKTConcept(
        id="demo.create_thing",
        title="Create Thing",
        description="Create a thing over HTTP.",
        input_schema={
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        },
        interface=Interface.HTTP,
        target="demo-api",
    )
    bundle = Bundle()
    bundle.add(concept)

    dispatcher = DispatcherRegistry()
    dispatcher.register(Interface.HTTP, HttpDispatcher(targets={"demo-api": urllib_client}))
    service = OKTSService(bundle, NaiveFallbackRetriever(), dispatcher)

    result = service.call_tool("demo.create_thing", {"name": "widget"})
    assert result == {"ok": True, "received": {"name": "widget"}}
