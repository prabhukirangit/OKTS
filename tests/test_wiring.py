"""Serve-time dispatcher wiring (okts/serve/wiring.py) — the offline slice.

Covers wiring `interface: function` + `module:` sources into a live
`FunctionDispatcher` so `call_tool` actually executes the module's callables.
The MCP-session slice needs a real subprocess and lives in tests/test_live.py
(run with --run-live).
"""

from __future__ import annotations

import anyio

from okts.build import build_bundle_from_config
from okts.config.loader import config_from_dict
from okts.serve.mcp_server import NaiveFallbackRetriever
from okts.serve.service import OKTSService
from okts.serve.wiring import open_dispatcher


def test_open_dispatcher_wires_function_module_for_real_dispatch(tmp_path):
    (tmp_path / "tools.py").write_text(
        "def add(a: float, b: float) -> float:\n"
        "    'Add two numbers.'\n"
        "    return a + b\n"
        "def shout(text: str) -> str:\n"
        "    'Uppercase text.'\n"
        "    return text.upper()\n"
    )
    config = config_from_dict({"sources": [{"interface": "function", "module": "./tools.py"}]})
    bundle = build_bundle_from_config(config, base_dir=tmp_path)

    async def scenario():
        async with open_dispatcher(config, base_dir=tmp_path) as dispatcher:
            service = OKTSService(bundle, NaiveFallbackRetriever(), dispatcher)
            summed = await service.acall_tool("add", {"a": 2, "b": 3})
            shouted = await service.acall_tool("shout", {"text": "hi"})
            return summed, shouted

    summed, shouted = anyio.run(scenario)
    assert summed == 5
    assert shouted == "HI"


def test_open_dispatcher_no_backends_leaves_registry_empty(tmp_path):
    # a config with only offline mcp tools (no live command, no module fns) wires
    # nothing, so the registry declines dispatch rather than guessing.
    config = config_from_dict(
        {"sources": [{"interface": "mcp", "servers": {"s": {"tools": [
            {"name": "x", "description": "d", "inputSchema": {"type": "object", "properties": {}}}
        ]}}}]}
    )
    bundle = build_bundle_from_config(config)

    async def scenario():
        async with open_dispatcher(config) as dispatcher:
            return dispatcher.supports(bundle.get("s.x"))

    assert anyio.run(scenario) is False
