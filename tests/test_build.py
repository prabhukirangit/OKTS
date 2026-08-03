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


def test_retrieval_k_config_sets_served_default_top_k():
    # `retrieval.k` in config becomes the default number of refs search_tools
    # returns across the unified multi-source corpus; a per-call k overrides.
    cfg = dict(ALL_SOURCES_CONFIG)
    cfg["retrieval"] = {"mode": "hybrid", "graph_expand": True, "k": 3}
    config = config_from_dict(cfg)

    service = build_service(config=config)
    assert service.default_k == 3

    # a query with no k uses the configured default (<= 3 refs)
    refs = service.search_tools("create an issue in a repository")
    assert 0 < len(refs) <= 3

    # an explicit k still overrides the config default
    assert len(service.search_tools("create an issue in a repository", k=1)) == 1


def test_retrieval_k_defaults_to_five_when_unset():
    config = config_from_dict(ALL_SOURCES_CONFIG)  # no retrieval block
    service = build_service(config=config)
    assert service.default_k == 5


def test_build_auto_links_hierarchy_and_alternatives():
    # the production build MUST derive the graph/hierarchy signals the
    # graph-aware retriever relies on — real MCP tools/list carries neither, so
    # without auto-linking the differentiator would be inert (review finding #1).
    config = config_from_dict({
        "sources": [{
            "interface": "mcp",
            "servers": {"github-mcp": {"tools": [
                {"name": "create_issue", "description": "Open a new issue.",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "list_issues", "description": "List issues in a repo.",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "update_issue", "description": "Edit an existing issue.",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}},
        }],
    })
    bundle = build_bundle_from_config(config)

    # hierarchy derived (github/issue category groups the three issue tools)
    assert bundle.hierarchy, "build should derive an index.md hierarchy"
    # same-<server>/<resource> tools linked as mutual alternatives
    create = bundle.get("github-mcp.create_issue")
    assert create.alternatives, "issue tools should be cross-linked as alternatives"
    assert any("issue" in a for a in create.alternatives)


def test_build_can_disable_linking():
    config = config_from_dict({
        "sources": [{
            "interface": "mcp",
            "servers": {"s": {"tools": [
                {"name": "a_thing", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
                {"name": "b_thing", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
            ]}},
        }],
    })
    bundle = build_bundle_from_config(config, link=False)
    assert not bundle.hierarchy
    assert not bundle.get("s.a_thing").alternatives


def test_load_module_callables_selects_public_defined_functions(tmp_path):
    from okts.build import load_module_callables

    (tmp_path / "m.py").write_text(
        "from math import sqrt  # imported -> excluded\n"
        "def alpha(x):\n    return x\n"
        "def _hidden(x):\n    return x\n"
        "def beta(y):\n    return y\n"
    )
    names = sorted(f.__name__ for f in load_module_callables(tmp_path / "m.py"))
    assert names == ["alpha", "beta"]  # sqrt (imported) and _hidden (private) excluded

    only = load_module_callables(tmp_path / "m.py", names=["beta"])
    assert [f.__name__ for f in only] == ["beta"]


def test_function_module_source_builds_callable_concepts(tmp_path):
    (tmp_path / "tools.py").write_text(
        "def add(a: float, b: float) -> float:\n    'Add two numbers.'\n    return a + b\n"
    )
    config = config_from_dict({"sources": [{"interface": "function", "module": "./tools.py"}]})
    bundle = build_bundle_from_config(config, base_dir=tmp_path)
    concept = bundle.get("add")
    assert concept is not None and concept.interface is Interface.FUNCTION
    assert {"a", "b"} <= set(concept.input_schema["properties"])


def test_okts_build_cli_writes_a_served_bundle(tmp_path):
    from okts.build import main
    from okts.core.bundle_io import load_bundle

    (tmp_path / "tools.py").write_text("def ping() -> str:\n    'Ping.'\n    return 'pong'\n")
    (tmp_path / "cfg.yaml").write_text(
        "sources:\n  - interface: function\n    module: ./tools.py\nbundle_dir: ./out\n"
    )
    rc = main(["--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "out")])
    assert rc == 0
    reloaded = load_bundle(tmp_path / "out")
    assert "ping" in {c.id for c in reloaded}


def test_config_needs_live_detects_connection_specs():
    from okts.build import config_needs_live

    offline = config_from_dict({"sources": [{"interface": "mcp", "servers": {"s": {"tools": []}}}]})
    live = config_from_dict(
        {"sources": [{"interface": "mcp", "servers": {"s": {"command": "python", "args": ["x.py"]}}}]}
    )
    assert config_needs_live(offline) is False
    assert config_needs_live(live) is True


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
    # concepts are persisted one-file-per-tool, plus an index.md for the
    # hierarchy that the auto-link pass derives during the build
    assert (out_dir / "github-mcp.create_issue.md").exists()
    assert (out_dir / "index.md").exists()

    from okts.core.bundle_io import load_bundle

    reloaded = load_bundle(out_dir)
    assert {c.id for c in reloaded} == {c.id for c in bundle}
    assert reloaded.hierarchy, "derived hierarchy should round-trip through index.md"


def test_save_bundle_neutralizes_path_traversal_ids(tmp_path):
    # A hostile/edge concept id (e.g. from an untrusted MCP server's tool name)
    # must NEVER escape the bundle directory when written to disk.
    from okts.core.bundle_io import load_bundle, save_bundle
    from okts.core.model import Bundle, Interface, OKTConcept

    hostile = OKTConcept(
        id="../../../../etc/passwd",
        title="Hostile",
        description="tries to escape the bundle dir",
        input_schema={"type": "object", "properties": {}},
        interface=Interface.MCP,
        target="evil",
    )
    out_dir = tmp_path / "bundle"
    b = Bundle()
    b.add(hostile)
    save_bundle(b, out_dir)

    # nothing was written outside the bundle directory: the file is a flat name
    # inside out_dir with NO path separators (so it can't traverse), even though
    # harmless leftover dots may remain in the slug.
    written = list(out_dir.iterdir())
    assert len(written) == 1
    assert written[0].parent.resolve() == out_dir.resolve()
    assert "/" not in written[0].name and "\\" not in written[0].name
    assert not (tmp_path.parent / "etc" / "passwd").exists()

    # the authoritative id survives inside the file, so load round-trips it
    reloaded = load_bundle(out_dir)
    assert {c.id for c in reloaded} == {"../../../../etc/passwd"}


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
