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
from okts.serve.service import (
    ArgumentValidationError,
    DispatchNotSupportedError,
    OKTSService,
    PolicyDenied,
    ToolNotFoundError,
)

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
        default_k=config.retrieval.k if config else 5,
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
                        "k": {
                            "type": "integer",
                            "description": f"max results (defaults to {service.default_k})",
                            "default": service.default_k,
                        },
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
                # omit k -> service applies its configured default_k
                result: Any = service.search_tools(arguments["query"], k=arguments.get("k"))
            elif name == "load_tool":
                result = service.load_tool(arguments["id"])
            elif name == "call_tool":
                # this handler already runs inside the MCP server's event loop,
                # so use the async path — a live MCP/agent/HTTP target dispatches
                # to a coroutine that must be awaited (invocation: async).
                result = await service.acall_tool(arguments["id"], arguments.get("args") or {})
            else:
                raise ToolNotFoundError(f"unknown meta-tool: {name!r}")
        except (
            ToolNotFoundError,
            ArgumentValidationError,
            DispatchNotSupportedError,
            PolicyDenied,
        ) as exc:
            # Return a structured, safe error to the agent rather than letting the
            # exception crash the tool handler. Messages carry ids/reasons only —
            # never credentials (invariant #4).
            result = {"error": str(exc), "error_type": type(exc).__name__}
        return [mcp_types.TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def _default_retriever(retrieval_cfg: Any) -> Retriever:
    """Prefer the real graph-aware retriever; fall back to the dependency-free
    naive one when the index layer (numpy) isn't installed, so ``okts[serve]``
    runs without the ``dense`` extra."""
    try:
        from okts.build import make_retriever

        return make_retriever(retrieval_cfg)
    except Exception:  # pragma: no cover - only when numpy/index is unavailable
        return NaiveFallbackRetriever()


async def _serve(args: Any) -> None:
    """Async serve loop: obtain the bundle (prebuilt dir or built from config),
    open the live dispatcher wired from config, then run the MCP server with the
    sessions held open for its lifetime."""
    from contextlib import nullcontext

    from okts.build import abuild_bundle_from_config, build_bundle_from_config, config_needs_live
    from okts.config.loader import RetrievalConfig, load_config
    from okts.serve.wiring import open_dispatcher

    config: Config | None = load_config(args.config) if args.config else None
    base_dir = Path(args.config).parent if args.config else None
    retrieval_cfg = config.retrieval if config is not None else RetrievalConfig()

    # bundle: a prebuilt dir wins (build once with okts-build, serve many times);
    # otherwise build from config now (live-ingesting mcp servers if needed).
    if args.bundle_dir:
        bundle = load_bundle(args.bundle_dir)
    elif config is not None:
        bundle = (
            await abuild_bundle_from_config(config, base_dir=base_dir)
            if config_needs_live(config)
            else build_bundle_from_config(config, base_dir=base_dir)
        )
    else:
        bundle = load_bundle("./okt-bundle")

    # dispatcher: live backends wired from config (mcp sessions + module fns),
    # held open for the server's lifetime. No config -> empty registry.
    dispatcher_ctx = (
        open_dispatcher(config, base_dir=base_dir)
        if config is not None
        else nullcontext(DispatcherRegistry())
    )
    async with dispatcher_ctx as dispatcher:
        service = OKTSService(
            bundle=bundle,
            retriever=_default_retriever(retrieval_cfg),
            dispatcher=dispatcher,
            default_k=retrieval_cfg.k,
        )
        server = _build_mcp_server(service)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    """Console-script entrypoint: ``okts`` (see ``pyproject.toml``).

    Serves the three meta-tools over MCP. Provide ``--bundle-dir`` to serve a
    bundle already built by ``okts-build``, or ``--config`` to build (and, for
    mcp servers with a connection spec, live-ingest) on startup. Either way, live
    dispatch backends from the config are wired so ``call_tool`` actually reaches
    the real tools.
    """
    _require_mcp()

    parser = argparse.ArgumentParser(
        prog="okts", description="Serve the three OKTS meta-tools over MCP (stdio)."
    )
    parser.add_argument("--config", default=None, help="path to tools.config.yaml")
    parser.add_argument("--bundle-dir", default=None, help="path to a built OKT bundle directory")
    args = parser.parse_args(argv)

    import anyio

    anyio.run(_serve, args)


if __name__ == "__main__":  # pragma: no cover
    main()
