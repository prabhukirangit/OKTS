"""OKF conformance validation for OKT concepts and bundles.

The bar (CLAUDE.md invariant #5): the required-minimum frontmatter is small —
``type``, ``id``, ``title``, ``description``, ``input_schema``, ``interface`` —
and everything else degrades gracefully. We validate structure, not taste:

- the six required keys are present and non-empty
- ``type == "tool"`` (the OKF discriminator)
- ``input_schema`` is either an inline JSON-Schema object or a ``{resource: ...}``
  pointer, and stays STRUCTURED (invariant #2 — never prose)
- ``interface`` / ``side_effects`` / ``invocation`` are known enum values
- graph edges resolve within the bundle (bundle-level check)

Every emitted bundle must pass this before it is served or committed.
"""

from __future__ import annotations

from typing import Any

from okts.core.model import (
    Bundle,
    Interface,
    Invocation,
    OKTConcept,
    REQUIRED_MINIMUM,
    SideEffects,
)


class ConformanceError(Exception):
    """Raised when a concept or bundle violates the OKF/OKT conformance rules.

    Carries a flat list of human-readable problem strings.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _schema_is_structured(schema: Any) -> bool:
    """input_schema must be a mapping: an inline JSON Schema or a resource ref."""
    if not isinstance(schema, dict):
        return False
    if "resource" in schema:
        return isinstance(schema["resource"], str) and bool(schema["resource"])
    # Inline JSON Schema: require at least a ``type`` or ``properties`` key so we
    # don't accept an empty {} as a valid contract.
    return "type" in schema or "properties" in schema


def validate_concept(concept: OKTConcept) -> list[str]:
    """Return a list of conformance problems for one concept (empty == valid)."""
    problems: list[str] = []

    # Required-minimum presence.
    values = {
        "type": concept.type,
        "id": concept.id,
        "title": concept.title,
        "description": concept.description,
        "input_schema": concept.input_schema,
        "interface": concept.interface,
    }
    for key in REQUIRED_MINIMUM:
        v = values[key]
        if v is None or (isinstance(v, str) and not v.strip()):
            problems.append(f"[{concept.id or '?'}] missing required field: {key}")

    # OKF discriminator.
    if concept.type != "tool":
        problems.append(f"[{concept.id or '?'}] type must be 'tool', got {concept.type!r}")

    # input_schema stays structured (invariant #2).
    if concept.input_schema is not None and not _schema_is_structured(concept.input_schema):
        problems.append(
            f"[{concept.id or '?'}] input_schema must be a structured JSON Schema "
            f"object or a {{resource: ...}} pointer, not prose/empty"
        )

    # Known enum values.
    if not isinstance(concept.interface, Interface):
        problems.append(
            f"[{concept.id or '?'}] interface {concept.interface!r} is not one of "
            f"{[i.value for i in Interface]}"
        )
    if not isinstance(concept.side_effects, SideEffects):
        problems.append(
            f"[{concept.id or '?'}] side_effects {concept.side_effects!r} is not one of "
            f"{[s.value for s in SideEffects]}"
        )
    # invocation is optional (defaults to sync); if set it must be a known value.
    if not isinstance(concept.invocation, Invocation):
        problems.append(
            f"[{concept.id or '?'}] invocation {concept.invocation!r} is not one of "
            f"{[i.value for i in Invocation]}"
        )

    return problems


def validate_bundle(bundle: Bundle, *, check_edges: bool = True) -> list[str]:
    """Validate every concept plus bundle-level graph integrity.

    ``check_edges`` verifies that ``alternatives``/``prerequisites``/
    ``composes_with`` references resolve to concepts present in the bundle.
    """
    problems: list[str] = []
    for concept in bundle:
        problems.extend(validate_concept(concept))

    if check_edges:
        for concept in bundle:
            for ref in concept.neighbors():
                if bundle.resolve_edge(ref) is None:
                    problems.append(
                        f"[{concept.id}] dangling graph edge -> {ref!r} "
                        f"(no matching concept in bundle)"
                    )
    return problems


def assert_conformant(bundle: Bundle, *, check_edges: bool = True) -> None:
    """Raise :class:`ConformanceError` if the bundle has any problems."""
    problems = validate_bundle(bundle, check_edges=check_edges)
    if problems:
        raise ConformanceError(problems)
