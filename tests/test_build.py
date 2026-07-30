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
from okts.config.loader import RetrievalConfig, load_config
from okts.core.validator import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "tools.config.yaml"


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
