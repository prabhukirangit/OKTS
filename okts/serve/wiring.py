"""Serve-time wiring: a config → a live :class:`DispatcherRegistry`.

The build pipeline (``okts.build``) produces the static OKT descriptor bundle.
Serving it so ``call_tool`` actually reaches the real tools needs *live* backends
— open MCP sessions, in-process function callables — and that connection
lifecycle belongs to the serving process, not the (offline, deterministic) build.

:func:`open_dispatcher` is an async context manager that, from a
``tools.config.yaml``, connects to each configured live MCP server and registers
its ``ClientSession`` in an :class:`~okts.serve.dispatch.McpDispatcher`, and
registers ``interface: function`` ``module:`` callables in a
:class:`~okts.serve.dispatch.FunctionDispatcher`. Sessions are held open for the
lifetime of the ``async with`` block and closed on exit (via ``AsyncExitStack``).

Credentials are resolved inside the dispatchers from the environment
(``EnvSecretsProvider``) and never surface here (invariant #4). ``http`` /
``search`` / ``agent`` interfaces still need caller-supplied clients — they can't
be fully specified from config — so they're left for the caller to register.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from okts.build import _mcp_server_is_live, load_module_callables
from okts.config.loader import Config
from okts.core.model import Interface
from okts.serve.dispatch import DispatcherRegistry, FunctionDispatcher, McpDispatcher

import logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def _connect_mcp(cfg: dict[str, Any]) -> AsyncIterator[Any]:
    """Open and initialize a live stdio MCP ``ClientSession`` from a server spec."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=cfg["command"], args=list(cfg.get("args") or []))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _function_targets(config: Config, base_dir: Path | None) -> dict[str, Any]:
    """Callables for ``interface: function`` + ``module:`` sources, keyed by the
    same name the adapter used as ``id``/``target`` (``fn.__name__``)."""
    targets: dict[str, Any] = {}
    for source in config.sources:
        if source.interface != "function":
            continue
        module_path = source.options.get("module")
        if not module_path:
            continue
        for fn in load_module_callables(
            module_path, names=source.options.get("functions"), base_dir=base_dir
        ):
            targets[fn.__name__] = fn
    return targets


@asynccontextmanager
async def open_dispatcher(
    config: Config, *, base_dir: Path | None = None
) -> AsyncIterator[DispatcherRegistry]:
    """Yield a :class:`DispatcherRegistry` wired to the config's live backends.

    Registers an :class:`McpDispatcher` (one live session per connection-spec mcp
    server) and a :class:`FunctionDispatcher` (module callables). Sessions stay
    open for the ``async with`` block. Interfaces with nothing wired simply have
    no dispatcher registered — ``call_tool`` on them raises ``NotConfiguredError``
    (the safe default), exactly as before.
    """
    registry = DispatcherRegistry()
    async with AsyncExitStack() as stack:
        mcp_targets: dict[str, Any] = {}
        for source in config.sources:
            if source.interface != "mcp":
                continue
            servers = source.options.get("servers")
            if not isinstance(servers, dict):
                continue
            for name, cfg in servers.items():
                cfg = cfg or {}
                if _mcp_server_is_live(cfg):
                    session = await stack.enter_async_context(_connect_mcp(cfg))
                    mcp_targets[name] = session
                    log.info("serve: connected live mcp session for target %r", name)

        if mcp_targets:
            registry.register(Interface.MCP, McpDispatcher(targets=mcp_targets))

        fn_targets = _function_targets(config, base_dir)
        if fn_targets:
            registry.register(Interface.FUNCTION, FunctionDispatcher(targets=fn_targets))
            log.info("serve: registered %d function target(s)", len(fn_targets))

        yield registry
