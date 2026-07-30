"""Phase 2 — the end-to-end wiring that turns a ``tools.config.yaml`` into a
running :class:`~okts.serve.service.OKTSService`.

This is the ONE module allowed to reach across every layer at once::

    config -> adapters (layer 1) -> enrich (layer 1½) -> bundle (layer 2)
           -> retriever (layer 3) -> service (layer 4)

It deliberately lives at the package top level, NOT inside ``okts.serve``:
the serving layer depends only on the ``Retriever``/``Dispatcher`` protocols
(invariant — see ``okts/serve/service.py``), so it must never import
``okts.index``. This module is where the concrete ``GraphAwareRetriever`` is
finally bolted onto the service, keeping that dependency out of the serving
layer proper.

Everything here runs OFFLINE and deterministically: adapters consume
already-parsed source payloads (inline in the config, or read from a local
file), enrichment is the deterministic :class:`OfflineEnricher`, and no source
requires the network or credentials to BUILD a bundle. Live MCP ingestion
(``okts.adapters.mcp.load_mcp_tools_live``) is an optional, separate path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from okts.adapters.agent import agents_to_okt
from okts.adapters.function import function_schemas_to_okt
from okts.adapters.mcp import mcp_tools_to_okt
from okts.adapters.openapi import openapi_to_okt
from okts.adapters.search import search_endpoints_to_okt
from okts.config.loader import Config, RetrievalConfig, Source, load_config
from okts.core.bundle_io import save_bundle
from okts.core.model import Bundle, OKTConcept
from okts.core.protocols import Dispatcher, Retriever
from okts.core.validator import validate_bundle
from okts.enrich.enricher import OfflineEnricher, enrich_bundle

__all__ = [
    "make_retriever",
    "concepts_from_source",
    "build_bundle_from_config",
    "build_service",
]


class BuildError(RuntimeError):
    """Raised when a config source can't be turned into concepts offline."""


# ---------------------------------------------------------------------------
# layer 3 factory: RetrievalConfig -> a concrete Retriever
# ---------------------------------------------------------------------------


def make_retriever(retrieval: RetrievalConfig | None = None) -> Retriever:
    """Build the production retriever from a :class:`RetrievalConfig`.

    Always the :class:`~okts.index.retriever.GraphAwareRetriever` (the project's
    actual contribution) — ``mode``/``graph_expand``/``hierarchy_prefilter`` from
    config select which signals are active. The flat-BM25 baseline is an
    eval-only comparator (see ``okts/eval``), never what production serves, so
    it is intentionally not reachable from here.
    """
    rc = retrieval or RetrievalConfig()
    from okts.index.retriever import GraphAwareRetriever  # lazy: keep numpy optional

    return GraphAwareRetriever(
        mode=rc.mode,
        graph_expand=rc.graph_expand,
        hierarchy_prefilter=rc.hierarchy_prefilter,
    )


# ---------------------------------------------------------------------------
# layer 1: config source -> OKT concepts (offline)
# ---------------------------------------------------------------------------


def _read_spec_file(path: str | Path) -> Any:
    """Load a local ``.yaml``/``.yml``/``.json`` spec file (for openapi etc.)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def concepts_from_source(source: Source, *, base_dir: Path | None = None) -> list[OKTConcept]:
    """Turn ONE config ``source`` into OKT concepts using the offline adapters.

    Every source may carry its payload inline (so a config is self-contained and
    testable) or point at a local file:

    - ``mcp``       — ``servers: {<name>: {tools: [<tools/list items>]}}`` OR
                      ``server`` + ``tools: [...]``. Live network ingestion is a
                      separate opt-in path (``load_mcp_tools_live``), never
                      triggered by a plain build.
    - ``function``  — ``schemas: [<function-schema dicts>]``.
    - ``http``      — ``openapi: <path>`` (file) OR ``spec: {<openapi doc>}``.
    - ``agent``     — ``cards: [<agent card dicts>]``.
    - ``search``    — ``endpoints: [<search endpoint specs>]``.
    """
    opts = source.options
    interface = source.interface
    base_dir = base_dir or Path.cwd()

    if interface == "mcp":
        servers = opts.get("servers")
        concepts: list[OKTConcept] = []
        # dict form: {server_name: {tools: [...], auth: ...}}
        if isinstance(servers, dict):
            for server, cfg in servers.items():
                cfg = cfg or {}
                tools = cfg.get("tools")
                if tools is None:
                    raise BuildError(
                        f"mcp server {server!r} has no offline 'tools' payload; a plain "
                        f"build is offline — use load_mcp_tools_live() to ingest a live server"
                    )
                concepts.extend(
                    mcp_tools_to_okt(tools, server=server, auth=cfg.get("auth"))
                )
            return concepts
        # flat form: server + tools inline
        server = opts.get("server")
        tools = opts.get("tools")
        if server and tools is not None:
            return mcp_tools_to_okt(tools, server=server, auth=opts.get("auth"))
        raise BuildError(
            "mcp source needs either servers:{name:{tools:[...]}} or server:+tools:[...] "
            "to build offline"
        )

    if interface == "function":
        schemas = opts.get("schemas")
        if schemas is None:
            raise BuildError("function source needs a 'schemas' list to build offline")
        return function_schemas_to_okt(schemas)

    if interface == "http":
        spec = opts.get("spec")
        if spec is None and opts.get("openapi"):
            spec = _read_spec_file(base_dir / opts["openapi"])
        if spec is None:
            raise BuildError("http source needs 'openapi' (a file path) or inline 'spec'")
        return openapi_to_okt(spec, auth=opts.get("auth"))

    if interface == "agent":
        cards = opts.get("cards")
        if cards is None:
            raise BuildError("agent source needs a 'cards' list to build offline")
        return agents_to_okt(cards)

    if interface == "search":
        endpoints = opts.get("endpoints")
        if endpoints is None:
            raise BuildError("search source needs an 'endpoints' list to build offline")
        return search_endpoints_to_okt(endpoints)

    raise BuildError(f"unknown source interface: {interface!r}")


# ---------------------------------------------------------------------------
# layers 1 -> 2: config -> enriched, validated bundle
# ---------------------------------------------------------------------------


def build_bundle_from_config(
    config: Config,
    *,
    base_dir: Path | None = None,
    enrich: bool = True,
    validate: bool = True,
    save_to: str | Path | None = None,
) -> Bundle:
    """Run every source through its adapter, enrich, validate, and (optionally)
    persist the resulting OKT bundle.

    Returns the in-memory :class:`Bundle`. If ``save_to`` is given (or falls
    back to ``config.bundle_dir``... only when explicitly passed), the bundle is
    also written to disk via :func:`~okts.core.bundle_io.save_bundle` so
    ``okts`` can serve it later without rebuilding.

    Raises :class:`BuildError` if a source can't be built offline, or
    ``ValueError`` (from the conformance validator) if ``validate`` is on and the
    assembled bundle isn't OKF-conformant.
    """
    bundle = Bundle()
    for source in config.sources:
        for concept in concepts_from_source(source, base_dir=base_dir):
            bundle.add(concept)

    if enrich:
        bundle = enrich_bundle(bundle, OfflineEnricher())

    if validate:
        problems = validate_bundle(bundle, check_edges=True)
        if problems:
            raise BuildError(
                "built bundle failed OKF conformance validation:\n  - "
                + "\n  - ".join(problems)
            )

    if save_to is not None:
        save_bundle(bundle, save_to)

    return bundle


# ---------------------------------------------------------------------------
# layers 3 -> 4: bundle + retriever + dispatcher -> a running service
# ---------------------------------------------------------------------------


def build_service(
    *,
    config: Config | None = None,
    config_path: str | Path | None = None,
    bundle: Bundle | None = None,
    retriever: Optional[Retriever] = None,
    dispatcher: Optional[Dispatcher] = None,
) -> "OKTSService":
    """Assemble a fully-wired :class:`~okts.serve.service.OKTSService`.

    The production counterpart to ``okts.serve.mcp_server.build_service`` (which
    defaults to the dependency-free ``NaiveFallbackRetriever``): this one injects
    the real :func:`make_retriever` (GraphAwareRetriever) by default, so callers
    that import from ``okts.build`` get graph-aware retrieval out of the box.

    Provide EITHER a ready ``bundle`` or a ``config``/``config_path`` to build one
    from sources. ``retriever`` defaults to ``make_retriever(config.retrieval)``;
    ``dispatcher`` defaults to a :class:`~okts.serve.dispatch.MockDispatcher`-backed
    registry so ``call_tool`` succeeds offline (swap in real dispatchers for live
    calls). Credentials, when wired, stay inside the dispatcher (invariant #4).
    """
    from okts.serve.dispatch import DispatcherRegistry
    from okts.serve.service import OKTSService

    if config is None and config_path is not None:
        config = load_config(config_path)

    if bundle is None:
        if config is None:
            raise BuildError(
                "build_service needs either a prebuilt bundle= or a config=/config_path= "
                "to build one from sources"
            )
        base_dir = Path(config_path).parent if config_path is not None else None
        bundle = build_bundle_from_config(config, base_dir=base_dir)

    retrieval_cfg = config.retrieval if config is not None else RetrievalConfig()
    return OKTSService(
        bundle=bundle,
        retriever=retriever or make_retriever(retrieval_cfg),
        dispatcher=dispatcher or DispatcherRegistry.mock_all(),
    )
