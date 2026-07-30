"""Layer 1 adapter: Python function schema / live callable -> OKT concepts.

Two entry points:

- :func:`function_schema_to_okt` — a declarative function-calling schema (the
  common ``{"name": ..., "description": ..., "parameters": {...}}`` shape used
  by OpenAI/Anthropic-style tool definitions, optionally wrapped in
  ``{"type": "function", "function": {...}}``) -> :class:`OKTConcept`.
- :func:`function_from_callable` — introspects a *live* Python callable's
  signature + docstring into a JSON Schema and builds the concept from that,
  no schema dict required.

Mapping (CLAUDE.md "Adapters (layer 1)"): ``function.name`` -> ``id``,
``parameters`` -> ``input_schema``, ``interface: function``, dotted callable
path -> ``target``.
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import Any, Callable

from okts.core.model import Interface, OKTConcept, SideEffects

__all__ = [
    "function_schema_to_okt",
    "function_schemas_to_okt",
    "function_from_callable",
    "python_signature_to_schema",
]


def _synthesize_title(concept_id: str) -> str:
    tail = concept_id.rsplit(".", 1)[-1]
    words = [w for w in re.split(r"[_\-]+", tail) if w]
    return " ".join(w.capitalize() for w in words) or concept_id


def _coerce_side_effects(value: Any) -> SideEffects:
    if isinstance(value, SideEffects):
        return value
    if isinstance(value, str):
        try:
            return SideEffects(value)
        except ValueError:
            pass
    return SideEffects.WRITE


def function_schema_to_okt(
    schema: dict[str, Any],
    *,
    id: str | None = None,
    target: str | None = None,
    auth: str | None = None,
    side_effects: Any = None,
    tags: list[str] | None = None,
) -> OKTConcept:
    """Convert one function-calling schema dict into an :class:`OKTConcept`.

    Accepts either the bare ``{"name": ..., "parameters": {...}}`` shape or
    the OpenAI-style wrapper ``{"type": "function", "function": {...}}`` —
    the wrapper is unwrapped automatically.
    """
    spec = schema.get("function", schema) if isinstance(schema.get("function"), dict) else schema
    name = spec.get("name")
    if not name:
        raise ValueError(f"function schema is missing required 'name': {schema!r}")

    concept_id = id or name
    description = spec.get("description") or f"Call the {name} function."
    parameters = spec.get("parameters") or {"type": "object", "properties": {}}
    dotted_target = target or spec.get("target") or name

    return OKTConcept(
        id=concept_id,
        title=spec.get("title") or _synthesize_title(concept_id),
        description=description,
        tags=list(tags or spec.get("tags") or []),
        input_schema=parameters,
        interface=Interface.FUNCTION,
        target=dotted_target,
        auth=auth,
        side_effects=_coerce_side_effects(side_effects if side_effects is not None else spec.get("side_effects")),
    )


def function_schemas_to_okt(schemas: list[dict[str, Any]], **kwargs: Any) -> list[OKTConcept]:
    """Convert a list of function-calling schema dicts into OKT concepts."""
    return [function_schema_to_okt(schema, **kwargs) for schema in schemas]


# --- live callable introspection ---------------------------------------------------

_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Best-effort JSON Schema for one type annotation. Unknown/absent types
    degrade to ``{}`` (any type) rather than guessing wrong."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return {}

    origin = typing.get_origin(annotation)

    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _annotation_to_schema(args[0])
        return {"anyOf": [_annotation_to_schema(a) for a in args]}

    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(annotation)
        item_schema = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    return {}


def python_signature_to_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Introspect a live Python callable's signature into a JSON Schema.

    Parameters without a default become ``required``. ``self``/``cls`` and
    ``*args``/``**kwargs`` are skipped.
    """
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        properties[name] = _annotation_to_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def function_from_callable(
    func: Callable[..., Any],
    *,
    id: str | None = None,
    description: str | None = None,
    target: str | None = None,
    auth: str | None = None,
    side_effects: Any = None,
    tags: list[str] | None = None,
) -> OKTConcept:
    """Build an :class:`OKTConcept` from a live Python callable.

    The input schema is introspected from the signature via
    :func:`python_signature_to_schema`; the first line of the docstring (if
    any) becomes ``description`` unless overridden; the full docstring seeds
    the body. ``target`` defaults to the callable's dotted module path.
    """
    name = getattr(func, "__name__", None)
    if not name:
        raise ValueError(f"callable {func!r} has no __name__; pass id= explicitly")

    concept_id = id or name
    doc = (inspect.getdoc(func) or "").strip()
    desc = description or (doc.splitlines()[0].strip() if doc else f"Call the {name} function.")
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", name)
    dotted_target = target or (f"{module}.{qualname}" if module else qualname)

    return OKTConcept(
        id=concept_id,
        title=_synthesize_title(concept_id),
        description=desc,
        tags=list(tags or []),
        input_schema=python_signature_to_schema(func),
        interface=Interface.FUNCTION,
        target=dotted_target,
        auth=auth,
        side_effects=_coerce_side_effects(side_effects),
        body=doc,
    )
