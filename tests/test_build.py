"""Phase 2 end-to-end wiring tests for ``okts.build``.

Exercises the full pipeline the way a real user hits it:
``tools.config.yaml`` -> adapters -> enrich -> bundle -> GraphAwareRetriever
-> OKTSService -> the three meta-tools. Everything runs offline; the repo's
own ``tools.config.yaml`` is the fixture, so this test also guards that the
shipped example config stays buildable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from okts.build import (
    BuildError,
    build_bundle_from_config,
    build_service,
    concepts_from_source,
    make_retriever,
)
from okts.config.loader import RetrievalConfig, config_from_dict, load_config
from okts.core.model import Interface
from okts.core.validator import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "tools.config.yaml"


# A single config that exercises EVERY source interface OKTS advertises
# (MCP · functions · agents · HTTP/OpenAPI · search APIs), each with an inline
# offline payload, so the whole build pipeline is covered for all five — not
# just the mcp+function pair the shipped example config uses.
ALL_SOURCES_CONFIG = {
    "sources": [
        {
            "interface": "mcp",
            "servers": {
                "github-mcp": {
                    "auth": "github_oauth",
                    "tools": [
                        {
                            "name": "create_issue",
                            "description": "Open a new issue in a GitHub repository.",
                            "inputSchema": {
                                "type": "object",
                                "required": ["repo", "title"],
                                "properties": {"repo": {"type": "string"}, "title": {"type": "string"}},
                            },
                        }
                    ],
                }
            },
        },
        {
            "interface": "function",
            "schemas": [
                {
                    "name": "math.add",
                    "description": "Add two numbers.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    },
                }
            ],
        },
        {
            "interface": "http",
            "spec": {
                "paths": {
                    "/v1/charges": {
                        "post": {
                            "operationId": "createCharge",
                            "summary": "Create a charge",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "required": ["amount"],
                                            "properties": {"amount": {"type": "integer"}},
                                        }
                                    }
                                }
                            },
                        }
                    }
                }
            },
        },
        {
            "interface": "agent",
            "cards": [
                {
                    "name": "research-agent",
                    "description": "Delegate deep research to a specialized sub-agent.",
                    "input_schema": {
                        "type": "object",
                        "required": ["topic"],
                        "properties": {"topic": {"type": "string"}},
                    },
                }
            ],
        },
        {
            "interface": "search",
            "endpoints": [
                {
                    "name": "web_search",
                    "description": "Search the public web.",
                    "url": "https://search.example.com/v1/search",
                    "query_params": [
                        {"name": "q", "type": "string", "required": True, "description": "query text"}
                    ],
                }
            ],
        },
    ]
}


def test_make_retriever_is_graph_aware_and_reflects_config():
    from okts.index.retriever import GraphAwareRetriever

    r = make_retriever(RetrievalConfig(mode="bm25", graph_expand=False, hierarchy_prefilter=False))
    assert isinstance(r, GraphAwareRetriever)
    assert r.mode == "bm25"
    assert r.graph_expand is False
    assert r.hierarchy_prefilter is False


def test_shipped_example_config_builds_a_conformant_bundle():
    config = load_config(EXAMPLE_CONFIG)
    bundle = build_bundle_from_config(config, base_dir=EXAMPLE_CONFIG.parent)

    ids = {c.id for c in bundle}
    assert {"github-mcp.create_issue", "github-mcp.list_issues", "math.add"} <= ids
    # enrichment ran: bodies got fattened with the deterministic offline lines
    assert "Use this to" in bundle.get("github-mcp.create_issue").body
    # and the whole thing is OKF-conformant
    assert validate_bundle(bundle, check_edges=True) == []


def test_end_to_end_search_load_call(tmp_path):
    config = load_config(EXAMPLE_CONFIG)
    service = build_service(config=config)

    # phase 1: search returns lightweight refs, never schemas
    hits = service.search_tools("open a new issue", k=5)
    assert hits, "expected at least one search hit"
    assert all(set(h) == {"id", "title", "description"} for h in hits)
    assert any(h["id"] == "github-mcp.create_issue" for h in hits)

    # phase 2: load injects exactly one structured input_schema
    view = service.load_tool("github-mcp.create_issue")
    assert view["input_schema"]["type"] == "object"
    assert "repo" in view["input_schema"]["properties"]

    # phase 3: call validates + dispatches (MockDispatcher, offline)
    result = service.call_tool("github-mcp.create_issue", {"repo": "o/n", "title": "hi"})
    assert result["id"] == "github-mcp.create_issue"
    assert result["args"]["title"] == "hi"


def test_build_service_from_path_persists_and_reloads(tmp_path):
    # building from a config path resolves relative source files against the
    # config's own directory, and the built bundle round-trips through disk.
    config = load_config(EXAMPLE_CONFIG)
    out_dir = tmp_path / "okt-bundle"
    bundle = build_bundle_from_config(
        config, base_dir=EXAMPLE_CONFIG.parent, save_to=out_dir
    )
    # concepts are persisted one-file-per-tool (no index.md here: the adapters
    # don't assign an index.md hierarchy, and save_bundle only writes one when
    # there is a hierarchy to write)
    assert (out_dir / "github-mcp.create_issue.md").exists()

    from okts.core.bundle_io import load_bundle

    reloaded = load_bundle(out_dir)
    assert {c.id for c in reloaded} == {c.id for c in bundle}


def test_mcp_source_without_offline_tools_raises_clear_error():
    from okts.config.loader import Source

    src = Source(interface="mcp", options={"servers": ["github-mcp"]})
    # a live server name with no inline tools can't be built offline
    with pytest.raises(BuildError, match="offline"):
        # dict path not taken (servers is a list) -> falls through to flat form
        concepts_from_source(src)


# ---------------------------------------------------------------------------
# every advertised source interface builds through the pipeline, not just
# the mcp+function pair the shipped example config happens to use.
# ---------------------------------------------------------------------------


def test_every_source_interface_builds_concepts():
    config = config_from_dict(ALL_SOURCES_CONFIG)
    by_interface = {}
    for source in config.sources:
        concepts = concepts_from_source(source)
        assert concepts, f"{source.interface} source produced no concepts"
        for c in concepts:
            by_interface.setdefault(c.interface, []).append(c.id)

    # all five interfaces are represented
    assert set(by_interface) == {
        Interface.MCP,
        Interface.FUNCTION,
        Interface.HTTP,
        Interface.AGENT,
        Interface.SEARCH,
    }


def test_all_sources_build_a_conformant_bundle_and_serve():
    config = config_from_dict(ALL_SOURCES_CONFIG)
    bundle = build_bundle_from_config(config)

    # one tool per source made it in, and the mixed-interface bundle is conformant
    assert {c.interface for c in bundle} == set(Interface)
    assert validate_bundle(bundle, check_edges=True) == []

    # the whole thing serves: search/load/call works regardless of interface
    # (DispatcherRegistry.mock_all backs call_tool offline for every interface)
    service = build_service(config=config)

    # phase 3 dispatch succeeds for a non-mcp interface too (the search API)
    result = service.call_tool("web_search", {"q": "hello"})
    assert result["id"] == "web_search"

    # and for the HTTP/OpenAPI-derived tool
    load = service.load_tool("createCharge")
    assert load["input_schema"]["type"] == "object"
