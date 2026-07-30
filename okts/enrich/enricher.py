"""Layer 1½ enrichment: fatten a concept's retrieval body.

Implements the ``Enricher`` protocol (``okts.core.protocols.Enricher``):
``enrich(concept, bundle=None) -> OKTConcept``.

Two implementations:

- :class:`OfflineEnricher` — DETERMINISTIC, no network/keys. Mechanically
  derives synonyms / when-to / when-not-to / prerequisites / composes-with /
  gotchas from the concept's own tags, id, description, side_effects, and
  graph edges (resolved against ``bundle`` when one is supplied). This is the
  enricher CI runs; it must always be available and always produce the same
  output for the same input.
- :class:`LLMEnricher` — a scaffold for calling out to an LLM to generate
  richer prose. It is guarded so importing this module, and constructing an
  ``LLMEnricher``, never requires any SDK or API key: the caller injects a
  plain ``call_fn: str -> str`` (or supplies a ``fallback`` enricher, e.g.
  ``OfflineEnricher()``, used when no ``call_fn`` is configured or the call
  fails). Nothing in this module ever imports a provider SDK.

:func:`enrich_bundle` applies an enricher across every concept in a bundle
and returns a new, expanded :class:`Bundle`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from okts.core.model import Bundle, OKTConcept, SideEffects

__all__ = ["OfflineEnricher", "LLMEnricher", "enrich_bundle"]

_SIDE_EFFECT_GOTCHAS: dict[SideEffects, str] = {
    SideEffects.READ: "This is read-only and safe to call speculatively; it does not modify state.",
    SideEffects.WRITE: "This action modifies state — double-check inputs before calling.",
    SideEffects.DESTRUCTIVE: "This action is destructive and may be irreversible — confirm before calling.",
}


def _lower_first(text: str) -> str:
    text = text.strip()
    return text[:1].lower() + text[1:] if text else text


def _synonyms_line(concept: OKTConcept) -> str:
    """Mechanical synonym list from the id's tail words + tags, deduped and
    order-preserving so the result is deterministic."""
    words: list[str] = []
    tail = concept.id.rsplit(".", 1)[-1] if concept.id else ""
    words.extend(w for w in re.split(r"[_\-]+", tail) if w)
    words.extend(concept.tags)

    seen: set[str] = set()
    uniq: list[str] = []
    for w in words:
        key = w.lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(w)

    if not uniq:
        return ""
    return "Synonyms: " + ", ".join(uniq) + "."


def _when_to_line(concept: OKTConcept) -> str:
    desc = concept.description.strip().rstrip(".")
    if not desc:
        return ""
    return f"Use this to {_lower_first(desc)}."


def _when_not_lines(concept: OKTConcept, bundle: Optional[Bundle]) -> list[str]:
    lines: list[str] = []
    for ref in concept.alternatives:
        resolved_id = bundle.resolve_edge(ref) if bundle is not None else None
        alt = bundle.get(resolved_id) if (bundle is not None and resolved_id) else None
        if alt is not None:
            alt_desc = alt.description.strip().rstrip(".")
            if alt_desc:
                lines.append(
                    f"Do not use this when you actually need to {_lower_first(alt_desc)} "
                    f"— use `{alt.id}` instead."
                )
            else:
                lines.append(f"Consider `{alt.id}` instead if this isn't the right fit.")
        else:
            label = resolved_id or ref
            lines.append(f"Consider `{label}` as an alternative if this isn't the right fit.")
    return lines


def _prereq_line(concept: OKTConcept) -> str:
    if not concept.prerequisites:
        return ""
    labels = ", ".join(f"`{p}`" for p in concept.prerequisites)
    return f"Prerequisite: you typically need {labels} first to get the required inputs."


def _composes_line(concept: OKTConcept) -> str:
    if not concept.composes_with:
        return ""
    labels = ", ".join(f"`{c}`" for c in concept.composes_with)
    return f"Composes with: {labels}."


def _gotcha_line(concept: OKTConcept) -> str:
    gotcha = _SIDE_EFFECT_GOTCHAS.get(concept.side_effects)
    return f"Gotcha: {gotcha}" if gotcha else ""


@dataclass
class OfflineEnricher:
    """Deterministic, offline body enricher. No network, no keys, safe for CI.

    ``enrich`` never mutates its input; it returns a new :class:`OKTConcept`
    (via ``dataclasses.replace``) whose ``body`` is the original body plus
    mechanically-derived synonyms / when-to / when-not-to / prerequisite /
    composes-with / gotcha lines. Because every derived line is a pure
    function of the concept's own fields (plus the bundle's graph, which is
    itself static input), calling ``enrich`` twice on the same
    ``(concept, bundle)`` always yields byte-identical output.
    """

    def enrich(self, concept: OKTConcept, bundle: Optional[Bundle] = None) -> OKTConcept:
        sections: list[str] = []
        existing = concept.body.strip()
        if existing:
            sections.append(existing)

        for line in (
            _synonyms_line(concept),
            _when_to_line(concept),
            *_when_not_lines(concept, bundle),
            _prereq_line(concept),
            _composes_line(concept),
            _gotcha_line(concept),
        ):
            if line:
                sections.append(line)

        return replace(concept, body="\n\n".join(sections))


_DEFAULT_SYSTEM_PROMPT = (
    "You expand a tool's retrieval text with synonyms, when-to-use, "
    "when-not-to-use, and gotchas. Be concise and concrete. Never invent "
    "parameters that are not in the tool's schema."
)


@dataclass
class LLMEnricher:
    """Scaffold for LLM-backed body enrichment.

    Guarded by construction, not by import: this module never imports a
    provider SDK, so ``from okts.enrich.enricher import LLMEnricher`` always
    succeeds with no network/keys. An ``LLMEnricher`` instance is only
    *usable* once the caller supplies ``call_fn`` (a plain ``str -> str``
    callable that sends ``_build_prompt(concept)`` to whatever LLM the caller
    has configured) and/or a ``fallback`` enricher (typically
    ``OfflineEnricher()``) to use when ``call_fn`` is absent or raises.

    Calling ``.enrich()`` with neither ``call_fn`` nor ``fallback`` configured
    raises ``RuntimeError`` — enrichment is never silently skipped.
    """

    call_fn: Optional[Callable[[str], str]] = None
    fallback: Optional[Any] = None  # typically an OfflineEnricher()
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT

    def _build_prompt(self, concept: OKTConcept) -> str:
        return (
            f"{self.system_prompt}\n\n"
            f"Tool id: {concept.id}\n"
            f"Title: {concept.title}\n"
            f"Description: {concept.description}\n"
            f"Tags: {', '.join(concept.tags)}\n"
            f"Side effects: {concept.side_effects.value}\n"
            f"Existing body:\n{concept.body}\n"
        )

    def enrich(self, concept: OKTConcept, bundle: Optional[Bundle] = None) -> OKTConcept:
        if self.call_fn is None:
            if self.fallback is not None:
                return self.fallback.enrich(concept, bundle)
            raise RuntimeError(
                "LLMEnricher has no call_fn configured and no fallback; pass a "
                "call_fn=<str->str callable> or fallback=OfflineEnricher() for "
                "offline/CI use"
            )

        try:
            generated = self.call_fn(self._build_prompt(concept))
        except Exception:
            if self.fallback is not None:
                return self.fallback.enrich(concept, bundle)
            raise

        generated = (generated or "").strip()
        existing = concept.body.strip()
        body = f"{existing}\n\n{generated}".strip() if existing else generated
        return replace(concept, body=body)


def enrich_bundle(bundle: Bundle, enricher: Any) -> Bundle:
    """Apply ``enricher.enrich`` to every concept in ``bundle``.

    Returns a NEW :class:`Bundle` (same hierarchy, expanded concepts); the
    input bundle is left untouched and is what graph-edge lookups (e.g.
    ``alternatives``) are resolved against during enrichment.
    """
    out = Bundle(hierarchy=dict(bundle.hierarchy))
    for concept in bundle:
        out.add(enricher.enrich(concept, bundle))
    return out
