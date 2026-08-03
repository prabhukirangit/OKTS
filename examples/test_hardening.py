"""Tests for the hardening examples (lazy targets + context hygiene).

Not part of the default suite (``testpaths = ["tests"]``); run explicitly::

    pytest examples/test_hardening.py

context_hygiene's LangChain adapter needs the example extras; that test skips if
LangChain isn't installed. The lazy-target test needs no extras.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def test_lazy_targets_connect_once():
    import lazy_targets as ex

    service, connects = ex._demo_service()
    assert connects == {"mcp": 0, "http": 0}  # nothing connects at construction

    for i in range(3):
        service.call_tool("calc.add", {"a": i, "b": 1})
    for i in range(2):
        service.call_tool("demo.create", {"name": f"t{i}"})

    assert connects == {"mcp": 1, "http": 1}  # each upstream connected exactly once


def test_lazy_connection_factory_caches_and_resets():
    import lazy_targets as ex

    calls = {"n": 0}

    def connect():
        calls["n"] += 1
        return object()

    factory = ex.LazyConnectionFactory(connect)
    assert not factory.connected
    a = factory.get()
    b = factory.get()
    assert a is b and calls["n"] == 1 and factory.connected
    factory.reset()
    assert not factory.connected
    c = factory.get()
    assert c is not a and calls["n"] == 2


def test_schema_for_id_detects_marker():
    import context_hygiene as ex
    from okts.serve.service import SCHEMA_MARKER_KEY, SCHEMA_MARKER_KIND

    payload = {"input_schema": {}, SCHEMA_MARKER_KEY: {"kind": SCHEMA_MARKER_KIND, "for_id": "x.y"}}
    assert ex.schema_for_id(payload) == "x.y"
    assert ex.schema_for_id(json.dumps(payload)) == "x.y"       # JSON string form
    assert ex.schema_for_id({"input_schema": {}}) is None       # no marker
    assert ex.schema_for_id("not json") is None


def test_spent_schema_indices_only_after_call():
    import context_hygiene as ex

    # records: (loaded_id, called_id) tuples for the generic core
    records = [
        ("a.tool", None),   # loaded schema for a.tool
        (None, "a.tool"),   # ...then called -> spent
        ("b.tool", None),   # loaded schema for b.tool, never called -> kept
    ]
    spent = ex.spent_schema_indices(
        records,
        get_loaded_id=lambda r: r[0],
        get_called_id=lambda r: r[1],
    )
    assert spent == {0}


def test_scrub_langchain_history_tombstones_spent_schema():
    pytest.importorskip("langchain_core")
    import context_hygiene as ex
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from okts.serve.service import SCHEMA_MARKER_KEY, SCHEMA_MARKER_KIND

    schema = json.dumps({
        "id": "gh.create",
        "input_schema": {"type": "object", "properties": {"repo": {"type": "string"}}},
        SCHEMA_MARKER_KEY: {"kind": SCHEMA_MARKER_KIND, "for_id": "gh.create"},
    })
    history = [
        HumanMessage(content="go"),
        ToolMessage(content=schema, name="load_tool", tool_call_id="t1"),
        AIMessage(content="", tool_calls=[{"name": "call_tool", "args": {"id": "gh.create"}, "id": "t2"}]),
        ToolMessage(content='{"ok": true}', name="call_tool", tool_call_id="t2"),
    ]

    scrubbed = ex.scrub_langchain_history(history, mode="tombstone")
    assert len(scrubbed) == len(history)                       # structure preserved
    assert "input_schema" not in str(scrubbed[1].content)      # schema evicted
    assert "evicted" in str(scrubbed[1].content)

    dropped = ex.scrub_langchain_history(history, mode="drop")
    assert len(dropped) == len(history) - 1                    # schema message removed


def test_scrub_keeps_unspent_schema():
    pytest.importorskip("langchain_core")
    import context_hygiene as ex
    from langchain_core.messages import HumanMessage, ToolMessage
    from okts.serve.service import SCHEMA_MARKER_KEY, SCHEMA_MARKER_KIND

    schema = json.dumps({
        "input_schema": {"type": "object"},
        SCHEMA_MARKER_KEY: {"kind": SCHEMA_MARKER_KIND, "for_id": "gh.create"},
    })
    # loaded but never called -> must be kept verbatim
    history = [HumanMessage(content="go"), ToolMessage(content=schema, name="load_tool", tool_call_id="t1")]
    scrubbed = ex.scrub_langchain_history(history)
    assert "input_schema" in str(scrubbed[1].content)
