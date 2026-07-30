"""Layer 4 / integration mode 1 — MCP server entrypoint.

Serves the three meta-tools (``search_tools``, ``load_tool``, ``call_tool``)
over the Model Context Protocol so any MCP client can drop in one ``okts``
server entry in place of N raw upstream server entries. This is the
``okts`` console script (see ``pyproject.toml``'s ``[project.scripts]``).

The ``mcp`` package is an optional extra (``pip install okts[serve]``). This
module MUST import cleanly without it installed — the import is guarded and
only raises when ``main()`` actually tries to run without the dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from okts.config.loader import Config, load_config
from okts.core.bundle_io import load_bundle
from okts.core.model import Bundle
from okts.core.protocols import Dispatcher, Retriever, SearchHit
from okts.serve.dispatch import DispatcherRegistry
from okts.serve.service import ArgumentValidationError, OKTSService, ToolNotFoundError

try:  # pragma: no cover - exercised only when the optional 'serve' extra is installed
    import mcp.types as mcp_types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    _MCP_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    mcp_types = None  # type: ignore[assignment]
    Server = None  # type: ignore[assignment]
    stdio_server = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = exc


class NaiveFallbackRetriever:
    """Minimal, dependency-free ``Retriever`` (see ``okts.core.protocols``).

    Ranks candidates by naive case-insensitive term-overlap counting over
    ``description + tags + body``. It exists purely so ``okts`` can run
    standalone with zero extra dependencies and no network/keys; it never
    imports ``okts.index`` (invariant: serve depends only on the protocol,
    not the concrete retriever). Swap in the real hybrid retriever by passing
    ``retriever=`` to :func:`build_service`.
    """

    def __init__(self) -> None:
        self._bundle: Bundle | None = None

    def index(self, bundle: Bundle) -> None:
        self._bundle = bundle

    def search(self, query: str, k: int = 5, **opts: Any) -> list[SearchHit]:
        if self._bundle is None:
            return []
        terms = [t for t in query.lower().split() if t]
        hits: list[SearchHit] = []
        for concept in self._bundle:
            text = concept.match_text().lower()
            score = sum(text.count(t) for t in terms) if terms else 0
            if terms and score == 0:
                continue
            hits.append(
                SearchHit(
                    id=concept.id,
                    title=concept.title,
                    description=concept.description,
                    score=float(score),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def build_service(
    *,
    config_path: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    retriever: Retriever | None = None,
    dispatcher: Dispatcher | None = None,
) -> OKTSService:
    """Assemble an :class:`OKTSService` from config + a built OKT bundle.

    ``retriever``/``dispatcher`` default to safe, dependency-free fallbacks:
    :class:`NaiveFallbackRetriever` and an empty :class:`DispatcherRegistry`
    (which raises a clear "not configured" error from ``call_tool`` until
    real dispatchers are registered, rather than crashing or silently
    no-opping). Inject the real retriever/dispatcher — wired by the
    coordinator once layers 1-3 land — to get real ranking and live dispatch.
    """
    config: Config | None = load_config(config_path) if config_path is not None else None

    resolved_bundle_dir = bundle_dir or (config.bundle_dir if config else "./okt-bundle")
    bundle = load_bundle(resolved_bundle_dir)

    return OKTSService(
        bundle=bundle,
        retriever=retriever or NaiveFallbackRetriever(),
        dispatcher=dispatcher or DispatcherRegistry(),
    )


def _require_mcp() -> None:
    if _MCP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "the 'mcp' package is required to run the OKTS MCP server. "
            "Install it with `pip install okts[serve]`."
        ) from _MCP_IMPORT_ERROR


def _build_mcp_server(service: OKTSService) -> Any:
    """Wire the three meta-tools into an ``mcp.server.Server``. Requires ``mcp``."""
    _require_mcp()
    server = Server("okts")

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="search_tools",
                description=(
                    "Search the OKTS tool catalog for candidates matching a query. "
                    "Returns lightweight refs (id, title, description) -- never schemas."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "description": "max results, default 5"},
                    },
                },
            ),
            mcp_types.Tool(
                name="load_tool",
                description="Load the structured input_schema + side_effects for one tool id.",
                inputSchema={
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "string"}},
                },
            ),
            mcp_types.Tool(
                name="call_tool",
                description="Validate args against a tool's input_schema, then dispatch the call.",
                inputSchema={
                    "type": "object",
                    "required": ["id", "args"],
                    "properties": {
                        "id": {"type": "string"},
                        "args": {"type": "object"},
                    },
                },
            ),
        ]

    @server.call_tool()
    async def _dispatch(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        try:
            if name == "search_tools":
                result: Any = service.search_tools(arguments["query"], k=arguments.get("k", 5))
            elif name == "load_tool":
                result = service.load_tool(arguments["id"])
            elif name == "call_tool":
                result = service.call_tool(arguments["id"], arguments.get("args") or {})
            else:
                raise ToolNotFoundError(f"unknown meta-tool: {name!r}")
        except (ToolNotFoundError, ArgumentValidationError) as exc:
            result = {"error": str(exc)}
        return [mcp_types.TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def main(argv: list[str] | None = None) -> None:
    """Console-script entrypoint: ``okts`` (see ``pyproject.toml``)."""
    _require_mcp()

    parser = argparse.ArgumentParser(
        prog="okts", description="Serve the three OKTS meta-tools over MCP (stdio)."
    )
    parser.add_argument("--config", default=None, help="path to tools.config.yaml")
    parser.add_argument("--bundle-dir", default=None, help="path to the built OKT bundle directory")
    args = parser.parse_args(argv)

    service = build_service(config_path=args.config, bundle_dir=args.bundle_dir)
    server = _build_mcp_server(service)

    import anyio

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_run)


if __name__ == "__main__":  # pragma: no cover
    main()
