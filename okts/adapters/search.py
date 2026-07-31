"""Layer 1 adapter: search endpoint spec -> OKT concept.

A "search endpoint" is any query-in/results-out source (a web search API, a
vector store query endpoint, an internal document search service, ...). Its
query parameters become ``input_schema``; ``interface: search``;
``side_effects`` is always ``read`` (search never mutates state).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from okts.core.model import Interface, Invocation, OKTConcept, SideEffects

__all__ = ["search_endpoint_to_okt", "search_endpoints_to_okt"]

log = logging.getLogger(__name__)


def _coerce_invocation(value: Any) -> Invocation:
    if isinstance(value, Invocation):
        return value
    if isinstance(value, str):
        try:
            return Invocation(value)
        except ValueError:
            pass
    return Invocation.SYNC  # the wired search client decides; runtime auto-awaits


def _synthesize_title(concept_id: str) -> str:
    tail = concept_id.rsplit(".", 1)[-1]
    words = [w for w in re.split(r"[_\-]+", tail) if w]
    return " ".join(w.capitalize() for w in words) or concept_id


def _build_query_schema(params: Any) -> dict[str, Any]:
    """Build an object schema from either a list of param dicts
    (``[{"name": "q", "type": "string", "required": true}, ...]``) or an
    already-shaped mapping of ``{name: json_schema}``."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    if isinstance(params, dict):
        for name, p in params.items():
            properties[name] = dict(p) if isinstance(p, dict) else {"type": "string"}
    else:
        for param in params or []:
            if not isinstance(param, dict) or "name" not in param:
                continue
            pname = param["name"]
            pschema: dict[str, Any] = {"type": param.get("type", "string")}
            if param.get("description"):
                pschema["description"] = param["description"]
            if "enum" in param:
                pschema["enum"] = param["enum"]
            if "default" in param:
                pschema["default"] = param["default"]
            properties[pname] = pschema
            if param.get("required"):
                required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def search_endpoint_to_okt(
    spec: dict[str, Any],
    *,
    id: str | None = None,
    target: str | None = None,
    auth: str | None = None,
) -> OKTConcept:
    """Convert one search endpoint spec dict into an :class:`OKTConcept`.

    Recognized keys: ``name``/``id``, ``title``, ``description``,
    ``query_params``/``parameters`` (list-of-dicts or name->schema mapping),
    ``url``/``endpoint`` (-> ``target``), ``auth``, ``tags``.
    """
    name = spec.get("name") or spec.get("id")
    if not name:
        raise ValueError(f"search endpoint spec is missing required 'name'/'id': {spec!r}")

    concept_id = id or spec.get("id") or name
    description = spec.get("description") or f"Search via {name}."
    input_schema = _build_query_schema(spec.get("query_params") or spec.get("parameters"))

    return OKTConcept(
        id=concept_id,
        title=spec.get("title") or _synthesize_title(concept_id),
        description=description,
        tags=list(spec.get("tags") or ["search"]),
        input_schema=input_schema,
        interface=Interface.SEARCH,
        target=target or spec.get("url") or spec.get("endpoint") or concept_id,
        auth=auth or spec.get("auth"),
        side_effects=SideEffects.READ,
        invocation=_coerce_invocation(spec.get("invocation")),
    )


def search_endpoints_to_okt(specs: list[dict[str, Any]], **kwargs: Any) -> list[OKTConcept]:
    """Convert a list of search endpoint specs into OKT concepts."""
    concepts = [search_endpoint_to_okt(spec, **kwargs) for spec in specs]
    log.info("search adapter: %d endpoints -> concepts", len(concepts))
    return concepts
