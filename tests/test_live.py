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


def test_live_mcp_dispatch_via_acall_tool():
    """End-to-end async dispatch: OKTSService.acall_tool -> McpDispatcher ->
    a real MCP ClientSession -> the running server actually executes the tool.

    This is the path the sync `call_tool` could never drive (a live session's
    `call_tool` is a coroutine); it works now because dispatch has an async
    path and MCP tools are adapted with `invocation: async`.
    """
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from okts.adapters.mcp import mcp_tools_to_okt
    from okts.core.model import Bundle
    from okts.serve.dispatch import DispatcherRegistry, McpDispatcher
    from okts.serve.mcp_server import NaiveFallbackRetriever
    from okts.serve.service import OKTSService

    params = StdioServerParameters(command=sys.executable, args=[str(LIVE_MCP_SERVER)])

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = (await session.list_tools()).tools
                raw = [t.model_dump(by_alias=True, exclude_none=True) for t in tools]
                concepts = mcp_tools_to_okt(raw, server="test-live")

                bundle = Bundle()
                for c in concepts:
                    bundle.add(c)

                # the live ClientSession drops straight in as the MCP client:
                # McpDispatcher strips the "test-live." id prefix to the bare
                # tool name the server knows ("add") — no wrapper needed.
                dispatcher = DispatcherRegistry()
                dispatcher.register("mcp", McpDispatcher(targets={"test-live": session}))
                service = OKTSService(bundle, NaiveFallbackRetriever(), dispatcher)

                return await service.acall_tool("test-live.add", {"a": 2, "b": 3})

    result = anyio.run(scenario)
    # FastMCP returns the sum in the result's text content
    text = "".join(getattr(part, "text", "") for part in result.content)
    assert "5" in text


# ---------------------------------------------------------------------------
# live config pipeline: config (connection spec) -> build -> wire -> dispatch
# ---------------------------------------------------------------------------


def test_live_build_from_config_ingests_mcp_server():
    """A config carrying an mcp CONNECTION spec (command/args, no offline tools)
    builds a bundle by connecting to the real server and listing its tools —
    closing the gap where `servers: [names]` couldn't build (review #2)."""
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")

    from okts.build import abuild_bundle_from_config, config_needs_live
    from okts.config.loader import config_from_dict

    config = config_from_dict({
        "sources": [{
            "interface": "mcp",
            "servers": {"test-live": {"command": sys.executable, "args": [str(LIVE_MCP_SERVER)]}},
        }],
    })
    assert config_needs_live(config) is True

    bundle = anyio.run(abuild_bundle_from_config, config)
    ids = {c.id for c in bundle}
    assert {"test-live.echo", "test-live.add"} <= ids
    assert bundle.hierarchy, "the built bundle should be auto-linked"


def test_live_open_dispatcher_end_to_end_call_tool():
    """Full serve-time path (review #5): build from a connection-spec config, wire
    the live dispatcher, and drive acall_tool -> the running server executes."""
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")

    from okts.build import abuild_bundle_from_config
    from okts.config.loader import config_from_dict
    from okts.serve.mcp_server import NaiveFallbackRetriever
    from okts.serve.service import OKTSService
    from okts.serve.wiring import open_dispatcher

    config = config_from_dict({
        "sources": [{
            "interface": "mcp",
            "servers": {"test-live": {"command": sys.executable, "args": [str(LIVE_MCP_SERVER)]}},
        }],
    })

    async def scenario():
        bundle = await abuild_bundle_from_config(config)
        async with open_dispatcher(config) as dispatcher:
            service = OKTSService(bundle, NaiveFallbackRetriever(), dispatcher)
            return await service.acall_tool("test-live.add", {"a": 2, "b": 3})

    result = anyio.run(scenario)
    text = "".join(getattr(part, "text", "") for part in result.content)
    assert "5" in text


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
