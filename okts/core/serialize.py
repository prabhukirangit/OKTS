"""Markdown <-> OKTConcept (de)serialization.

An OKT file is YAML frontmatter fenced by ``---`` lines, followed by a markdown
body. Round-tripping is lossless: unknown frontmatter keys are preserved on
``OKTConcept.extra`` and re-emitted.
"""

from __future__ import annotations

from typing import Any

import yaml

from okts.core.model import Cost, Interface, Invocation, OKTConcept, SideEffects

# Frontmatter keys we map onto typed fields. Anything else lands in ``extra``.
_KNOWN_KEYS = {
    "type",
    "id",
    "title",
    "description",
    "tags",
    "input_schema",
    "output_schema",
    "interface",
    "target",
    "auth",
    "side_effects",
    "invocation",
    "cost",
    "alternatives",
    "prerequisites",
    "composes_with",
    "timestamp",
    "version",
}


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split an OKT markdown document into (frontmatter dict, body str)."""
    stripped = text.lstrip("﻿")  # tolerate BOM
    if not stripped.startswith("---"):
        # No frontmatter: whole thing is body.
        return {}, text.strip()
    # Find the closing fence.
    lines = stripped.splitlines()
    # lines[0] is the opening '---'
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("unterminated YAML frontmatter (missing closing '---')")
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :]).strip()
    data = yaml.safe_load(fm_text) or {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter did not parse to a mapping")
    return data, body


def concept_from_markdown(text: str) -> OKTConcept:
    """Parse an OKT markdown document into an :class:`OKTConcept`.

    This does no conformance checking — call ``validate_concept`` for that. It is
    tolerant so a malformed file can still be loaded and then reported on.
    """
    fm, body = split_frontmatter(text)

    def _enum(cls, value, default):
        if value is None:
            return default
        try:
            return cls(value)
        except ValueError:
            # Unknown enum value: keep the raw string via a permissive member.
            return value

    extra = {k: v for k, v in fm.items() if k not in _KNOWN_KEYS}

    return OKTConcept(
        id=fm.get("id", ""),
        title=fm.get("title", ""),
        description=fm.get("description", "") or "",
        tags=list(fm.get("tags") or []),
        input_schema=fm.get("input_schema") or {},
        output_schema=fm.get("output_schema"),
        interface=_enum(Interface, fm.get("interface"), Interface.FUNCTION),
        target=fm.get("target"),
        auth=fm.get("auth"),
        side_effects=_enum(SideEffects, fm.get("side_effects"), SideEffects.WRITE),
        invocation=_enum(Invocation, fm.get("invocation"), Invocation.SYNC),
        cost=Cost.from_frontmatter(fm.get("cost")),
        alternatives=list(fm.get("alternatives") or []),
        prerequisites=list(fm.get("prerequisites") or []),
        composes_with=list(fm.get("composes_with") or []),
        type=fm.get("type", "tool"),
        timestamp=fm.get("timestamp"),
        version=fm.get("version"),
        body=body,
        extra=extra,
    )


def _interface_value(v: Any) -> Any:
    return v.value if isinstance(v, Interface) else v


def _side_effects_value(v: Any) -> Any:
    return v.value if isinstance(v, SideEffects) else v


def _invocation_value(v: Any) -> Any:
    return v.value if isinstance(v, Invocation) else v


# The field groups map one-to-one onto the three meta-tools / runtime phases, so
# we emit each under a ``#`` comment header naming the consuming phase. Comments
# are ignored by ``yaml.safe_load`` on read, so grouping stays lossless.
_GROUP_LABELS = {
    "identity": "identity",
    "match": "match — ranked by search_tools (phase 1); never sent at call time",
    "call": "call — loaded by load_tool (phase 2); the calling contract",
    "route": "route — used by call_tool to dispatch (phase 3)",
    "edges": "graph edges — expanded during search_tools",
    "okf": "OKF standard",
    "extension": "extension — preserved for lossless round-trip",
}


def _dump_group(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()


def concept_to_markdown(concept: OKTConcept) -> str:
    """Serialize an :class:`OKTConcept` back to an OKT markdown document.

    Frontmatter is emitted in the spec's field groups (identity, match, call,
    route, graph edges, OKF standard), each under a ``#`` comment header naming
    the runtime phase / meta-tool that consumes it — so an opened file is
    self-documenting. The comments are ignored on read, so round-tripping stays
    lossless.
    """
    identity = {"type": concept.type, "id": concept.id, "title": concept.title}

    match: dict[str, Any] = {"description": concept.description}
    if concept.tags:
        match["tags"] = concept.tags

    call: dict[str, Any] = {"input_schema": concept.input_schema}
    if concept.output_schema is not None:
        call["output_schema"] = concept.output_schema

    route: dict[str, Any] = {"interface": _interface_value(concept.interface)}
    if concept.target is not None:
        route["target"] = concept.target
    if concept.auth is not None:
        route["auth"] = concept.auth
    route["side_effects"] = _side_effects_value(concept.side_effects)
    # Emit `invocation` only when it departs from the `sync` default, so the
    # common case stays clean; round-trip is still lossless (a missing key
    # parses back to SYNC).
    invocation = _invocation_value(concept.invocation)
    if invocation != Invocation.SYNC.value:
        route["invocation"] = invocation
    if concept.cost is not None:
        cost_fm = concept.cost.to_frontmatter()
        if cost_fm:
            route["cost"] = cost_fm

    edges: dict[str, Any] = {}
    if concept.alternatives:
        edges["alternatives"] = concept.alternatives
    if concept.prerequisites:
        edges["prerequisites"] = concept.prerequisites
    if concept.composes_with:
        edges["composes_with"] = concept.composes_with

    okf: dict[str, Any] = {}
    if concept.timestamp is not None:
        okf["timestamp"] = concept.timestamp
    if concept.version is not None:
        okf["version"] = concept.version

    # Unknown keys preserved for lossless round-trip / portability. Exclude any a
    # known group already emits so nothing is duplicated.
    extension = {k: v for k, v in concept.extra.items() if k not in _KNOWN_KEYS}

    groups = [
        ("identity", identity),
        ("match", match),
        ("call", call),
        ("route", route),
        ("edges", edges),
        ("okf", okf),
        ("extension", extension),
    ]
    yaml_text = "\n\n".join(
        f"# {_GROUP_LABELS[name]}\n{_dump_group(data)}" for name, data in groups if data
    )

    body = concept.body.strip()
    return f"---\n{yaml_text}\n---\n\n{body}\n" if body else f"---\n{yaml_text}\n---\n"
