"""Layer 1 adapter tests: source dict -> OKTConcept(s), all offline.

Every emitted concept must pass ``okts.core.validator.validate_concept`` with
zero problems — that's the hard requirement each adapter is held to.
"""

from __future__ import annotations

import pytest

from okts.core.model import Interface, SideEffects
from okts.core.validator import validate_concept

from okts.adapters.mcp import mcp_tools_to_okt
from okts.adapters.function import (
    function_from_callable,
    function_schema_to_okt,
    function_schemas_to_okt,
    python_signature_to_schema,
)
from okts.adapters.openapi import openapi_to_okt
from okts.adapters.agent import agent_to_okt, agents_to_okt
from okts.adapters.search import search_endpoint_to_okt, search_endpoints_to_okt


# ---------------------------------------------------------------------------
# MCP adapter
# ---------------------------------------------------------------------------

MCP_TOOLS_LIST = {
    "tools": [
        {
            "name": "create_issue",
            "description": "Open a new issue in a GitHub repository.",
            "inputSchema": {
                "type": "object",
                "required": ["repo", "title"],
                "properties": {
                    "repo": {"type": "string", "description": "owner/name"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
            "annotations": {"destructiveHint": False},
        },
        {
            "name": "list_issues",
            "description": "List issues in a repository.",
            "inputSchema": {
                "type": "object",
                "properties": {"repo": {"type": "string"}},
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "delete_issue",
            "description": "Permanently delete an issue.",
            "inputSchema": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
            },
            "annotations": {"destructiveHint": True},
        },
        {
            "name": "no_annotations_tool",
            "description": "Does something without hints.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]
}


def test_mcp_tools_to_okt_maps_fields():
    concepts = mcp_tools_to_okt(MCP_TOOLS_LIST, server="github-mcp")
    assert len(concepts) == 4
    by_id = {c.id: c for c in concepts}

    create = by_id["github-mcp.create_issue"]
    assert create.interface == Interface.MCP
    assert create.target == "github-mcp"
    assert create.description == "Open a new issue in a GitHub repository."
    assert create.input_schema["required"] == ["repo", "title"]


def test_mcp_accepts_bare_list_not_just_wrapped_dict():
    concepts = mcp_tools_to_okt(MCP_TOOLS_LIST["tools"], server="github-mcp")
    assert len(concepts) == 4
    assert {c.id for c in concepts} == {c.id for c in mcp_tools_to_okt(MCP_TOOLS_LIST, server="github-mcp")}


def test_mcp_annotation_side_effects_mapping():
    concepts = mcp_tools_to_okt(MCP_TOOLS_LIST, server="github-mcp")
    by_id = {c.id: c for c in concepts}

    assert by_id["github-mcp.list_issues"].side_effects == SideEffects.READ
    assert by_id["github-mcp.delete_issue"].side_effects == SideEffects.DESTRUCTIVE
    # destructiveHint explicitly False, no readOnlyHint -> default write
    assert by_id["github-mcp.create_issue"].side_effects == SideEffects.WRITE
    # no annotations at all -> default write
    assert by_id["github-mcp.no_annotations_tool"].side_effects == SideEffects.WRITE


def test_mcp_readonly_wins_if_both_hints_set():
    tools = [
        {
            "name": "weird",
            "description": "Weird tool with contradictory hints.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True, "destructiveHint": True},
        }
    ]
    concepts = mcp_tools_to_okt(tools, server="srv")
    assert concepts[0].side_effects == SideEffects.READ


def test_mcp_concepts_pass_validator():
    concepts = mcp_tools_to_okt(MCP_TOOLS_LIST, server="github-mcp")
    for c in concepts:
        assert validate_concept(c) == []


def test_mcp_synthesizes_title_and_description_when_missing():
    tools = [{"name": "weird_tool_name", "inputSchema": {"type": "object", "properties": {}}}]
    concepts = mcp_tools_to_okt(tools, server="srv")
    c = concepts[0]
    assert c.title == "Weird Tool Name"
    assert c.description  # non-empty, synthesized
    assert c.id == "srv.weird_tool_name"
    assert validate_concept(c) == []


def test_mcp_missing_input_schema_defaults_to_empty_object():
    tools = [{"name": "bare"}]
    concepts = mcp_tools_to_okt(tools, server="srv")
    assert concepts[0].input_schema == {"type": "object", "properties": {}}
    assert validate_concept(concepts[0]) == []


def test_mcp_missing_name_raises():
    with pytest.raises(ValueError):
        mcp_tools_to_okt([{"description": "no name"}], server="srv")


def test_mcp_module_imports_cleanly_regardless_of_mcp_sdk():
    # importing the adapter module must never require the optional `mcp`
    # package; the live-connect helper is only exercised if it's installed.
    from okts.adapters import mcp as mcp_adapter

    assert hasattr(mcp_adapter, "mcp_tools_to_okt")
    assert hasattr(mcp_adapter, "load_mcp_tools_live")


# ---------------------------------------------------------------------------
# function adapter
# ---------------------------------------------------------------------------


def sample_func(user_id: str, limit: int = 10) -> list[str]:
    """Fetch items for a user.

    More details go here.
    """
    return []


def test_function_from_callable_introspects_signature():
    c = function_from_callable(sample_func, target="mypkg.sample_func")
    assert c.interface == Interface.FUNCTION
    assert c.target == "mypkg.sample_func"
    assert c.input_schema["properties"]["user_id"] == {"type": "string"}
    assert c.input_schema["properties"]["limit"] == {"type": "integer"}
    assert c.input_schema["required"] == ["user_id"]
    assert c.description == "Fetch items for a user."
    assert "More details" in c.body
    assert validate_concept(c) == []


def test_function_from_callable_defaults_target_to_dotted_path():
    c = function_from_callable(sample_func)
    assert c.target is not None
    assert c.target.endswith("sample_func")


def test_python_signature_to_schema_skips_self_and_var_args():
    class Foo:
        def bar(self, x: int, *args, **kwargs) -> None:
            ...

    schema = python_signature_to_schema(Foo.bar)
    assert set(schema["properties"]) == {"x"}
    assert schema["required"] == ["x"]


def test_function_schema_to_okt_bare_shape():
    schema = {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    c = function_schema_to_okt(schema, target="weather.get_weather")
    assert c.id == "get_weather"
    assert c.interface == Interface.FUNCTION
    assert c.input_schema == schema["parameters"]
    assert validate_concept(c) == []


def test_function_schema_to_okt_openai_wrapped_shape():
    schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    c = function_schema_to_okt(schema)
    assert c.id == "get_weather"
    assert c.target == "get_weather"
    assert validate_concept(c) == []


def test_function_schema_to_okt_missing_name_raises():
    with pytest.raises(ValueError):
        function_schema_to_okt({"description": "no name"})


def test_function_schemas_to_okt_batch():
    schemas = [
        {"name": "a", "description": "does a", "parameters": {"type": "object", "properties": {}}},
        {"name": "b", "description": "does b", "parameters": {"type": "object", "properties": {}}},
    ]
    concepts = function_schemas_to_okt(schemas)
    assert [c.id for c in concepts] == ["a", "b"]
    for c in concepts:
        assert validate_concept(c) == []


# ---------------------------------------------------------------------------
# openapi adapter
# ---------------------------------------------------------------------------

OPENAPI_SPEC = {
    "paths": {
        "/v1/charges": {
            "post": {
                "operationId": "createCharge",
                "summary": "Create a charge",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "amount": {"type": "integer"},
                                    "currency": {"type": "string"},
                                },
                                "required": ["amount", "currency"],
                            }
                        }
                    },
                },
                "security": [{"apiKeyAuth": []}],
            },
            "get": {
                "operationId": "listCharges",
                "parameters": [
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}, "required": False}
                ],
            },
        },
        "/v1/charges/{id}": {
            "delete": {
                # no operationId -> must be synthesized
                "parameters": [
                    {"name": "id", "in": "path", "schema": {"type": "string"}, "required": True}
                ],
            }
        },
    },
    "components": {"securitySchemes": {"apiKeyAuth": {"type": "apiKey"}}},
}


def test_openapi_to_okt_maps_operation_fields():
    concepts = openapi_to_okt(OPENAPI_SPEC)
    by_id = {c.id: c for c in concepts}

    create = by_id["createCharge"]
    assert create.interface == Interface.HTTP
    assert create.target == "POST /v1/charges"
    assert "amount" in create.input_schema["properties"]
    assert "currency" in create.input_schema["properties"]
    assert set(create.input_schema.get("required", [])) == {"amount", "currency"}
    assert create.auth == "apiKeyAuth"
    assert create.side_effects == SideEffects.WRITE


def test_openapi_get_is_read_and_merges_query_params():
    concepts = openapi_to_okt(OPENAPI_SPEC)
    by_id = {c.id: c for c in concepts}
    listing = by_id["listCharges"]
    assert listing.side_effects == SideEffects.READ
    assert listing.target == "GET /v1/charges"
    assert "limit" in listing.input_schema["properties"]


def test_openapi_synthesizes_operation_id_and_delete_is_destructive():
    concepts = openapi_to_okt(OPENAPI_SPEC)
    deletes = [c for c in concepts if c.target == "DELETE /v1/charges/{id}"]
    assert len(deletes) == 1
    d = deletes[0]
    assert d.side_effects == SideEffects.DESTRUCTIVE
    assert d.id  # synthesized, non-empty
    assert "id" in d.input_schema["properties"]


def test_openapi_concepts_all_pass_validator():
    for c in openapi_to_okt(OPENAPI_SPEC):
        assert validate_concept(c) == []


# ---------------------------------------------------------------------------
# agent adapter
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "research-agent",
    "description": "Delegates deep research tasks to a specialized sub-agent.",
    "prompt": "You are a research assistant. Given a topic, produce a cited summary.",
    "input_schema": {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
    "url": "https://agents.example.com/research",
    "tags": ["research", "delegate"],
}


def test_agent_to_okt_maps_fields():
    c = agent_to_okt(AGENT_CARD)
    assert c.interface == Interface.AGENT
    assert c.target == "https://agents.example.com/research"
    assert c.input_schema == AGENT_CARD["input_schema"]
    assert "cited summary" in c.body
    assert validate_concept(c) == []


def test_agent_to_okt_defaults_when_sparse():
    c = agent_to_okt({"name": "minimal-agent"})
    assert c.id == "minimal_agent"
    assert c.description  # synthesized
    assert c.input_schema["type"] == "object"
    assert "input" in c.input_schema["properties"]
    assert validate_concept(c) == []


def test_agent_to_okt_missing_name_raises():
    with pytest.raises(ValueError):
        agent_to_okt({"description": "no name"})


def test_agents_to_okt_batch():
    cards = [{"name": "a", "description": "agent a"}, {"name": "b", "description": "agent b"}]
    concepts = agents_to_okt(cards)
    assert [c.id for c in concepts] == ["a", "b"]
    for c in concepts:
        assert validate_concept(c) == []


# ---------------------------------------------------------------------------
# search adapter
# ---------------------------------------------------------------------------

SEARCH_SPEC = {
    "name": "web_search",
    "description": "Search the public web.",
    "url": "https://search.example.com/v1/search",
    "query_params": [
        {"name": "q", "type": "string", "required": True, "description": "query text"},
        {"name": "limit", "type": "integer", "required": False},
    ],
}


def test_search_endpoint_to_okt_maps_fields():
    c = search_endpoint_to_okt(SEARCH_SPEC)
    assert c.interface == Interface.SEARCH
    assert c.side_effects == SideEffects.READ
    assert c.target == "https://search.example.com/v1/search"
    assert c.input_schema["required"] == ["q"]
    assert c.input_schema["properties"]["q"]["description"] == "query text"
    assert validate_concept(c) == []


def test_search_endpoint_to_okt_missing_name_raises():
    with pytest.raises(ValueError):
        search_endpoint_to_okt({"description": "no name"})


def test_search_endpoints_to_okt_batch():
    specs = [
        {"name": "a", "description": "search a", "query_params": []},
        {"name": "b", "description": "search b", "query_params": []},
    ]
    concepts = search_endpoints_to_okt(specs)
    assert [c.id for c in concepts] == ["a", "b"]
    for c in concepts:
        assert validate_concept(c) == []
