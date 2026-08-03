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
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from okts.adapters.agent import agents_to_okt
from okts.adapters.function import function_from_callable, function_schemas_to_okt
from okts.adapters.mcp import mcp_tools_to_okt
from okts.adapters.openapi import openapi_to_okt
from okts.adapters.search import search_endpoints_to_okt
from okts.config.loader import Config, RetrievalConfig, Source, load_config
from okts.core.bundle_io import save_bundle
from okts.core.model import Bundle, OKTConcept
from okts.core.protocols import Dispatcher, Retriever
from okts.core.validator import validate_bundle
from okts.enrich.autolink import autolink
from okts.enrich.enricher import OfflineEnricher, enrich_bundle

__all__ = [
    "make_retriever",
    "concepts_from_source",
    "aconcepts_from_source",
    "load_module_callables",
    "build_bundle_from_config",
    "abuild_bundle_from_config",
    "config_needs_live",
    "build_service",
    "main",
]

log = logging.getLogger(__name__)


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


def load_module_callables(
    module_path: str | Path,
    *,
    names: list[str] | None = None,
    base_dir: Path | None = None,
) -> list[Any]:
    """Import a local ``.py`` module and return the callables to adapt.

    Selection: the explicit ``names`` list when given, else every public
    (non-underscore) function DEFINED in that module (imported names are
    skipped). Used by the ``interface: function`` + ``module:`` config form for
    both adapting (here) and dispatch wiring (``okts.serve.wiring``), so the same
    callables back the concepts and their dispatch targets.
    """
    import importlib.util
    import inspect as _inspect

    path = Path(module_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        raise BuildError(f"function source module not found: {path}")

    spec = importlib.util.spec_from_file_location(f"okts_user_module_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise BuildError(f"could not import function module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if names:
        out: list[Any] = []
        for name in names:
            fn = getattr(module, name, None)
            if not callable(fn):
                raise BuildError(f"function {name!r} not found (or not callable) in {path}")
            out.append(fn)
        return out
    # default: public functions defined in THIS module (not imported ones)
    return [
        fn
        for name, fn in _inspect.getmembers(module, _inspect.isfunction)
        if not name.startswith("_") and getattr(fn, "__module__", None) == module.__name__
    ]


def concepts_from_source(source: Source, *, base_dir: Path | None = None) -> list[OKTConcept]:
    """Turn ONE config ``source`` into OKT concepts using the offline adapters.

    Every source may carry its payload inline (so a config is self-contained and
    testable) or point at a local file:

    - ``mcp``       — ``servers: {<name>: {tools: [<tools/list items>]}}`` OR
                      ``server`` + ``tools: [...]``. Live network ingestion is a
                      separate opt-in path (``load_mcp_tools_live``), never
                      triggered by a plain build.
    - ``function``  — ``schemas: [<function-schema dicts>]`` OR
                      ``module: <path.py>`` (+ optional ``functions: [names]``).
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
        if schemas is not None:
            return function_schemas_to_okt(schemas)
        module_path = opts.get("module")
        if module_path is not None:
            callables = load_module_callables(
                module_path, names=opts.get("functions"), base_dir=base_dir
            )
            return [
                function_from_callable(fn, id=fn.__name__, target=fn.__name__)
                for fn in callables
            ]
        raise BuildError(
            "function source needs a 'schemas' list or a 'module' path (with "
            "optional 'functions: [names]') to build"
        )

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
    link: bool = True,
    validate: bool = True,
    save_to: str | Path | None = None,
) -> Bundle:
    """Run every source through its adapter, enrich, auto-link, validate, and
    (optionally) persist the resulting OKT bundle.

    The full layer-1 → layer-2 pipeline from CLAUDE.md: adapters → enrichment →
    **structural auto-link** → conformance validation. The auto-link pass
    (``link=True``, default) is what derives the ``index.md`` category hierarchy
    and the ``alternatives`` edges from the flat adapted concepts — the exact
    signals the graph-aware retriever (hierarchy prefilter + graph expansion)
    exploits. Real sources (MCP ``tools/list``, function/OpenAPI schemas) carry
    neither, so WITHOUT this pass a served bundle degrades to plain hybrid
    ranking. It only ADDS structure and is query-independent, so it never biases
    retrieval (see ``okts/enrich/autolink.py``).

    Returns the in-memory :class:`Bundle`. If ``save_to`` is given the bundle is
    also written to disk via :func:`~okts.core.bundle_io.save_bundle` (including
    an ``index.md`` for the derived hierarchy) so ``okts`` can serve it later
    without rebuilding.

    Raises :class:`BuildError` if a source can't be built offline, or if
    ``validate`` is on and the assembled bundle isn't OKF-conformant.
    """
    log.info("build: assembling bundle from %d source(s)", len(config.sources))
    bundle = Bundle()
    for source in config.sources:
        for concept in concepts_from_source(source, base_dir=base_dir):
            bundle.add(concept)
    return _finalize_bundle(bundle, enrich=enrich, link=link, validate=validate, save_to=save_to)


def _finalize_bundle(
    bundle: Bundle,
    *,
    enrich: bool,
    link: bool,
    validate: bool,
    save_to: str | Path | None,
) -> Bundle:
    """Shared layer-1½→2 tail: enrich → auto-link → validate → (persist).

    Split out so both the offline :func:`build_bundle_from_config` and the async
    live path :func:`abuild_bundle_from_config` run the identical pipeline after
    their (sync vs. live) adapter stage."""
    log.info("build: %d concepts adapted from sources", sum(1 for _ in bundle))

    if enrich:
        bundle = enrich_bundle(bundle, OfflineEnricher())

    if link:
        bundle = autolink(bundle)
        log.info(
            "build: auto-linked %d categories + alternatives edges", len(bundle.hierarchy)
        )

    if validate:
        problems = validate_bundle(bundle, check_edges=True)
        if problems:
            log.warning("build: bundle failed OKF conformance (%d problems)", len(problems))
            raise BuildError(
                "built bundle failed OKF conformance validation:\n  - "
                + "\n  - ".join(problems)
            )
        log.info("build: bundle passed OKF conformance validation")

    if save_to is not None:
        save_bundle(bundle, save_to)
        log.info("build: bundle saved to %s", save_to)

    return bundle


# ---------------------------------------------------------------------------
# live ingestion: connect to real MCP servers, list + adapt their tools
# ---------------------------------------------------------------------------


def _mcp_server_is_live(cfg: dict[str, Any]) -> bool:
    """A server entry is 'live' when it gives connection details (``command``)
    and no offline ``tools`` payload."""
    return bool(cfg) and cfg.get("tools") is None and bool(cfg.get("command"))


def config_needs_live(config: Config) -> bool:
    """True if any source requires a live network connection to build (an mcp
    server with a connection spec rather than an inline ``tools`` payload)."""
    for source in config.sources:
        if source.interface != "mcp":
            continue
        servers = source.options.get("servers")
        if isinstance(servers, dict) and any(_mcp_server_is_live(c or {}) for c in servers.values()):
            return True
    return False


async def _ingest_live_mcp_server(name: str, cfg: dict[str, Any]) -> list[OKTConcept]:
    from okts.adapters.mcp import load_mcp_tools_live

    try:
        from mcp.client.stdio import StdioServerParameters
    except ImportError as exc:  # pragma: no cover - requires the optional extra
        raise BuildError(
            f"mcp server {name!r} is configured for live connection but the 'mcp' "
            f"package is not installed (pip install okts[serve])"
        ) from exc

    params = StdioServerParameters(command=cfg["command"], args=list(cfg.get("args") or []))
    log.info("build: live-ingesting mcp server %r via %s", name, cfg["command"])
    return await load_mcp_tools_live(params, server=name, auth=cfg.get("auth"))


async def aconcepts_from_source(source: Source, *, base_dir: Path | None = None) -> list[OKTConcept]:
    """Async source→concepts that ALSO handles live mcp servers (connection
    specs) via :func:`~okts.adapters.mcp.load_mcp_tools_live`. Every other source
    (and offline mcp ``tools:``) delegates to the sync :func:`concepts_from_source`."""
    if source.interface == "mcp":
        servers = source.options.get("servers")
        if isinstance(servers, dict) and any(
            _mcp_server_is_live(c or {}) for c in servers.values()
        ):
            concepts: list[OKTConcept] = []
            for name, cfg in servers.items():
                cfg = cfg or {}
                if _mcp_server_is_live(cfg):
                    concepts.extend(await _ingest_live_mcp_server(name, cfg))
                elif cfg.get("tools") is not None:
                    concepts.extend(mcp_tools_to_okt(cfg["tools"], server=name, auth=cfg.get("auth")))
                else:
                    raise BuildError(
                        f"mcp server {name!r} needs either a 'command' (live) or a "
                        f"'tools' payload (offline)"
                    )
            return concepts
    return concepts_from_source(source, base_dir=base_dir)


async def abuild_bundle_from_config(
    config: Config,
    *,
    base_dir: Path | None = None,
    enrich: bool = True,
    link: bool = True,
    validate: bool = True,
    save_to: str | Path | None = None,
) -> Bundle:
    """Async counterpart to :func:`build_bundle_from_config` that connects to any
    live mcp servers (connection specs) to ingest their tools, then runs the same
    enrich → auto-link → validate → persist pipeline."""
    log.info("build: assembling bundle from %d source(s) [live]", len(config.sources))
    bundle = Bundle()
    for source in config.sources:
        for concept in await aconcepts_from_source(source, base_dir=base_dir):
            bundle.add(concept)
    return _finalize_bundle(bundle, enrich=enrich, link=link, validate=validate, save_to=save_to)


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
        default_k=retrieval_cfg.k,
    )


# ---------------------------------------------------------------------------
# CLI: `okts-build` — config -> built, saved bundle (the missing middle step)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Console entry (`okts-build`): build an OKT bundle from a config and save it.

    Runs the full pipeline (adapters → enrich → auto-link → validate) and writes
    the bundle to ``--out`` (default: the config's ``bundle_dir``) so ``okts``
    can serve it. Automatically uses the live-ingest path when the config has any
    mcp server with a connection spec (``command``); otherwise builds offline.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="okts-build",
        description="Build an OKT bundle from a tools.config.yaml and save it to disk.",
    )
    parser.add_argument("--config", required=True, help="path to tools.config.yaml")
    parser.add_argument("--out", default=None, help="output bundle dir (default: config.bundle_dir)")
    parser.add_argument("--no-enrich", action="store_true", help="skip body enrichment")
    parser.add_argument("--no-link", action="store_true", help="skip structural auto-linking")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    out_dir = args.out or config.bundle_dir
    opts = dict(
        base_dir=base_dir,
        enrich=not args.no_enrich,
        link=not args.no_link,
        validate=True,
        save_to=out_dir,
    )

    if config_needs_live(config):
        import anyio

        async def _run() -> Bundle:
            return await abuild_bundle_from_config(config, **opts)

        bundle = anyio.run(_run)
    else:
        bundle = build_bundle_from_config(config, **opts)

    print(
        f"okts-build: {sum(1 for _ in bundle)} tools, {len(bundle.hierarchy)} categories "
        f"-> {out_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
