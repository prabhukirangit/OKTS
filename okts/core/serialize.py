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


def concept_to_markdown(concept: OKTConcept) -> str:
    """Serialize an :class:`OKTConcept` back to an OKT markdown document.

    Field order follows the spec's grouping (identity, match, call, route, edges,
    OKF standard) so diffs stay readable.
    """
    fm: dict[str, Any] = {
        "type": concept.type,
        "id": concept.id,
        "title": concept.title,
        "description": concept.description,
    }
    if concept.tags:
        fm["tags"] = concept.tags
    fm["input_schema"] = concept.input_schema
    if concept.output_schema is not None:
        fm["output_schema"] = concept.output_schema
    fm["interface"] = _interface_value(concept.interface)
    if concept.target is not None:
        fm["target"] = concept.target
    if concept.auth is not None:
        fm["auth"] = concept.auth
    fm["side_effects"] = _side_effects_value(concept.side_effects)
    # Emit `invocation` only when it departs from the `sync` default, so the
    # common case stays clean; round-trip is still lossless (a missing key
    # parses back to SYNC).
    invocation = _invocation_value(concept.invocation)
    if invocation != Invocation.SYNC.value:
        fm["invocation"] = invocation
    if concept.cost is not None:
        cost_fm = concept.cost.to_frontmatter()
        if cost_fm:
            fm["cost"] = cost_fm
    if concept.alternatives:
        fm["alternatives"] = concept.alternatives
    if concept.prerequisites:
        fm["prerequisites"] = concept.prerequisites
    if concept.composes_with:
        fm["composes_with"] = concept.composes_with
    if concept.timestamp is not None:
        fm["timestamp"] = concept.timestamp
    if concept.version is not None:
        fm["version"] = concept.version
    # Preserve unknown keys for lossless round-trip / portability.
    for k, v in concept.extra.items():
        fm.setdefault(k, v)

    yaml_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    body = concept.body.strip()
    return f"---\n{yaml_text}\n---\n\n{body}\n" if body else f"---\n{yaml_text}\n---\n"
