"""Layer 1 adapter: MCP ``tools/list`` -> OKT concepts.

This is the primary/reference adapter (CLAUDE.md build order #1). It is pure
and OFFLINE: :func:`mcp_tools_to_okt` takes an already-parsed ``tools/list``
response (a ``{"tools": [...]}`` dict or a bare list of tool dicts) and never
touches the network. A thin, optional live-connect helper is provided for
convenience but is guarded behind a try-import of the ``mcp`` package so this
module always imports cleanly even when that optional dependency is absent.

Mapping (CLAUDE.md "Adapters (layer 1)"):

- ``name``            -> ``id`` (namespaced ``<server>.<name>``)
- ``description``     -> ``description``
- ``inputSchema``     -> ``input_schema``
- server name         -> ``target``
- fixed               -> ``interface: mcp``
- ``annotations.readOnlyHint``    -> ``side_effects: read``
- ``annotations.destructiveHint`` -> ``side_effects: destructive``
- otherwise           -> ``side_effects: write`` (safe default)
"""

from __future__ import annotations

import re
from typing import Any

from okts.core.model import Interface, OKTConcept, SideEffects

__all__ = ["mcp_tools_to_okt", "load_mcp_tools_live"]


def _synthesize_title(concept_id: str) -> str:
    """Derive a human title from an id like ``github.create_issue``."""
    tail = concept_id.rsplit(".", 1)[-1]
    words = [w for w in re.split(r"[_\-]+", tail) if w]
    return " ".join(w.capitalize() for w in words) or concept_id


def _side_effects_from_annotations(annotations: dict[str, Any] | None) -> SideEffects:
    """Map MCP tool ``annotations`` to a :class:`SideEffects` value.

    ``readOnlyHint`` wins if both hints are somehow set (it is the stronger
    safety signal); ``destructiveHint`` is checked next; unannotated tools
    default to ``write`` (the safe assumption per CLAUDE.md).
    """
    if not annotations:
        return SideEffects.WRITE
    if annotations.get("readOnlyHint") is True:
        return SideEffects.READ
    if annotations.get("destructiveHint") is True:
        return SideEffects.DESTRUCTIVE
    return SideEffects.WRITE


def mcp_tools_to_okt(
    tools_list: list[dict[str, Any]] | dict[str, Any],
    server: str,
    *,
    auth: str | None = None,
    timestamp: str | None = None,
) -> list[OKTConcept]:
    """Convert a parsed MCP ``tools/list`` response into OKT concepts.

    ``tools_list`` may be the raw ``{"tools": [...]}`` response dict or an
    already-unwrapped list of tool dicts. ``server`` is the MCP server name;
    it becomes the ``target`` and namespaces each concept id
    (``<server>.<tool name>``).
    """
    tools = tools_list.get("tools", []) if isinstance(tools_list, dict) else tools_list

    concepts: list[OKTConcept] = []
    for tool in tools:
        name = tool.get("name")
        if not name:
            raise ValueError(f"MCP tool is missing required 'name': {tool!r}")

        concept_id = f"{server}.{name}"
        annotations = tool.get("annotations") or {}
        title = tool.get("title") or annotations.get("title") or _synthesize_title(concept_id)
        description = tool.get("description") or f"Call {name} via the {server} MCP server."
        input_schema = tool.get("inputSchema") or {"type": "object", "properties": {}}

        concepts.append(
            OKTConcept(
                id=concept_id,
                title=title,
                description=description,
                tags=[server, "mcp"],
                input_schema=input_schema,
                interface=Interface.MCP,
                target=server,
                auth=auth,
                side_effects=_side_effects_from_annotations(annotations),
                timestamp=timestamp,
            )
        )
    return concepts


# --- optional live-connect helper -------------------------------------------------
# Guarded behind a try-import so importing this module never requires the
# optional `mcp` package (pip install okts[serve]). Not used by the offline
# adapter path or by CI tests.
try:  # pragma: no cover - exercised only when `mcp` is installed
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _MCP_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MCP_SDK_AVAILABLE = False


async def load_mcp_tools_live(
    server_params: "StdioServerParameters",
    server: str,
    **kwargs: Any,
) -> list[OKTConcept]:
    """Connect to a live MCP server over stdio, list its tools, and adapt them.

    Convenience wrapper around :func:`mcp_tools_to_okt` for callers who do have
    the optional ``mcp`` SDK installed. Raises :class:`ImportError` if it is
    not. Never called by the offline adapter path or by the test suite.
    """
    if not _MCP_SDK_AVAILABLE:
        raise ImportError(
            "the optional 'mcp' package is required for live MCP connections; "
            "install it via `pip install okts[serve]`"
        )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = [
                t.model_dump(by_alias=True, exclude_none=True) for t in result.tools
            ]
            return mcp_tools_to_okt(tools, server=server, **kwargs)
