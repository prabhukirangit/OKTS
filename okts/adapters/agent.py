"""Layer 1 adapter: sub-agent card/dict -> OKT concept.

A "sub-agent" here is any delegate agent exposed as a single callable unit
(an A2A-style agent card, or any bespoke ``{"name": ..., "prompt": ...}``
dict). The agent's prompt/instructions become the retrieval ``body``; its
declared input contract becomes ``input_schema``; ``interface: agent``.
"""

from __future__ import annotations

import re
from typing import Any

from okts.core.model import Interface, Invocation, OKTConcept, SideEffects

__all__ = ["agent_to_okt", "agents_to_okt"]

_DEFAULT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "Free-form instruction / task description for the sub-agent.",
        }
    },
    "required": ["input"],
}


def _synthesize_title(concept_id: str) -> str:
    tail = concept_id.rsplit(".", 1)[-1]
    words = [w for w in re.split(r"[_\-]+", tail) if w]
    return " ".join(w.capitalize() for w in words) or concept_id


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "agent"


def _coerce_side_effects(value: Any) -> SideEffects:
    if isinstance(value, SideEffects):
        return value
    if isinstance(value, str):
        try:
            return SideEffects(value)
        except ValueError:
            pass
    return SideEffects.WRITE


def _coerce_invocation(value: Any) -> Invocation:
    if isinstance(value, Invocation):
        return value
    if isinstance(value, str):
        try:
            return Invocation(value)
        except ValueError:
            pass
    return Invocation.SYNC  # the wired agent callable decides; runtime auto-awaits


def _schema_from_skills(skills: Any) -> dict[str, Any] | None:
    """Best-effort input schema derived from an A2A-style ``skills`` list.

    A2A agent cards describe capabilities as ``skills`` rather than a single
    JSON Schema; we fold their declared example params (if any) into a loose
    object schema. Returns ``None`` when nothing usable is present, so the
    caller can fall back to the generic default.
    """
    if not isinstance(skills, list) or not skills:
        return None
    properties: dict[str, Any] = {}
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        params = skill.get("parameters") or skill.get("input_schema")
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            properties.update(params["properties"])
    if not properties:
        return None
    return {"type": "object", "properties": properties}


def agent_to_okt(
    card: dict[str, Any],
    *,
    id: str | None = None,
    target: str | None = None,
    auth: str | None = None,
) -> OKTConcept:
    """Convert one sub-agent card/dict into an :class:`OKTConcept`.

    Recognized keys (all optional except a name): ``name``/``id``,
    ``title``, ``description``, ``prompt``/``instructions``/``system_prompt``,
    ``input_schema``/``input_contract``, ``skills`` (A2A-style), ``url``/
    ``endpoint`` (-> ``target``), ``auth``, ``tags``, ``side_effects``.
    """
    name = card.get("name") or card.get("id")
    if not name:
        raise ValueError(f"agent card is missing required 'name'/'id': {card!r}")

    concept_id = id or card.get("id") or _slugify(name)
    description = card.get("description") or f"Delegate a task to the {name} sub-agent."
    prompt = (
        card.get("prompt")
        or card.get("instructions")
        or card.get("system_prompt")
        or ""
    )
    input_schema = (
        card.get("input_schema")
        or card.get("input_contract")
        or _schema_from_skills(card.get("skills"))
        or dict(_DEFAULT_INPUT_SCHEMA)
    )

    return OKTConcept(
        id=concept_id,
        title=card.get("title") or _synthesize_title(concept_id),
        description=description,
        tags=list(card.get("tags") or []),
        input_schema=input_schema,
        interface=Interface.AGENT,
        target=target or card.get("url") or card.get("endpoint") or concept_id,
        auth=auth or card.get("auth"),
        side_effects=_coerce_side_effects(card.get("side_effects")),
        invocation=_coerce_invocation(card.get("invocation")),
        body=prompt.strip(),
    )


def agents_to_okt(cards: list[dict[str, Any]], **kwargs: Any) -> list[OKTConcept]:
    """Convert a list of sub-agent cards into OKT concepts."""
    return [agent_to_okt(card, **kwargs) for card in cards]
