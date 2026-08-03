"""Example — context hygiene: evicting spent ``load_tool`` schemas from history.

Progressive disclosure loads exactly ONE tool schema per ``load_tool`` — but
across a multi-turn conversation those schemas pile up in the message history
and slowly re-introduce the bloat OKTS exists to remove. OKTS is
framework-agnostic and does not own your history, so it can't evict them for
you. What it *does* do is make each loaded schema **self-identifying**:
``OKTSService.load_tool`` stamps every payload with a marker
(``{"_okts": {"kind": "schema-instance", "for_id": <id>}}`` — see
``okts/serve/service.py``).

This file is the reference scrubber that uses that marker: once a tool's
``call_tool`` has been issued, its earlier loaded schema is *spent* and can be
dropped (or tombstoned) from history. The core is framework-agnostic; a
LangChain ``BaseMessage`` adapter is included.

Run it::

    python examples/context_hygiene.py
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from okts.serve.service import SCHEMA_MARKER_KEY, SCHEMA_MARKER_KIND


# ---------------------------------------------------------------------------
# framework-agnostic core
# ---------------------------------------------------------------------------


def schema_for_id(payload: Any) -> Optional[str]:
    """If ``payload`` is a ``load_tool`` result (a dict, or JSON string of one)
    carrying the OKTS schema marker, return the tool id it loaded — else None."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return None
    if isinstance(payload, dict):
        marker = payload.get(SCHEMA_MARKER_KEY)
        if isinstance(marker, dict) and marker.get("kind") == SCHEMA_MARKER_KIND:
            return marker.get("for_id")
    return None


def spent_schema_indices(
    records: list[Any],
    *,
    get_loaded_id: Callable[[Any], Optional[str]],
    get_called_id: Callable[[Any], Optional[str]],
) -> set[int]:
    """Indices of records that are loaded schemas whose tool was later *called*.

    ``get_loaded_id(record)`` -> the id a record's loaded schema is for (or None);
    ``get_called_id(record)`` -> the id a record invokes via ``call_tool`` (or None).
    A schema is spent once a ``call_tool`` for the same id has appeared anywhere
    in the history — at that point the contract has served its purpose.
    """
    called: set[str] = set()
    loaded: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        lid = get_loaded_id(record)
        if lid is not None:
            loaded.setdefault(lid, []).append(i)
        cid = get_called_id(record)
        if cid is not None:
            called.add(cid)
    spent: set[int] = set()
    for tool_id, idxs in loaded.items():
        if tool_id in called:
            spent.update(idxs)
    return spent


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------


def scrub_langchain_history(messages: list[Any], *, mode: str = "tombstone") -> list[Any]:
    """Return a new message list with spent ``load_tool`` schemas cleaned up.

    ``mode="tombstone"`` (default, safest) keeps each message in place but
    replaces the spent schema's content with a tiny placeholder — so the
    tool-call/tool-response pairing frameworks expect stays intact while the
    heavy schema text leaves the context. ``mode="drop"`` removes the schema
    ``ToolMessage`` entirely (smaller, but can orphan its request).
    """
    from langchain_core.messages import ToolMessage

    def get_loaded_id(msg: Any) -> Optional[str]:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "load_tool":
            return schema_for_id(msg.content)
        return None

    def get_called_id(msg: Any) -> Optional[str]:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "call_tool":
                return (tc.get("args") or {}).get("id")
        return None

    spent = spent_schema_indices(messages, get_loaded_id=get_loaded_id, get_called_id=get_called_id)
    if not spent:
        return list(messages)

    out: list[Any] = []
    for i, msg in enumerate(messages):
        if i not in spent:
            out.append(msg)
            continue
        if mode == "drop":
            continue
        # tombstone: preserve the ToolMessage (and its tool_call_id) but drop the schema
        for_id = schema_for_id(msg.content)
        out.append(
            ToolMessage(
                content=f"[okts: schema for {for_id!r} evicted after call]",
                name=msg.name,
                tool_call_id=getattr(msg, "tool_call_id", ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# offline demo
# ---------------------------------------------------------------------------


def main() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    schema_payload = json.dumps({
        "id": "github.create_issue",
        "input_schema": {"type": "object", "properties": {"repo": {"type": "string"}}},
        "side_effects": "write",
        SCHEMA_MARKER_KEY: {"kind": SCHEMA_MARKER_KIND, "for_id": "github.create_issue"},
    })
    history = [
        HumanMessage(content="open an issue"),
        AIMessage(content="", tool_calls=[{"name": "load_tool", "args": {"id": "github.create_issue"}, "id": "t1"}]),
        ToolMessage(content=schema_payload, name="load_tool", tool_call_id="t1"),
        AIMessage(content="", tool_calls=[{"name": "call_tool", "args": {"id": "github.create_issue", "arguments": {}}, "id": "t2"}]),
        ToolMessage(content='{"ok": true}', name="call_tool", tool_call_id="t2"),
    ]

    scrubbed = scrub_langchain_history(history, mode="tombstone")
    print("-- before --")
    for m in history:
        print(f"  {type(m).__name__:<13} {getattr(m,'name',None)}: {str(m.content)[:60]}")
    print("-- after (schema evicted, structure intact) --")
    for m in scrubbed:
        print(f"  {type(m).__name__:<13} {getattr(m,'name',None)}: {str(m.content)[:60]}")

    assert "input_schema" not in str(scrubbed[2].content), "spent schema should be gone"
    assert len(scrubbed) == len(history), "tombstone mode keeps message structure"
    print("\nOK — the spent load_tool schema was evicted once call_tool ran.")


if __name__ == "__main__":
    main()
