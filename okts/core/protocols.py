"""Structural interfaces shared across layers.

These Protocols are the seams that let the layers be built independently:

- ``Adapter``    (layer 1)  a source -> list[OKTConcept] pure-ish function
- ``Enricher``   (layer 1½) fattens a concept body with synonyms/gotchas
- ``Retriever``  (layer 3)  ranks concepts for a query (phase 1)
- ``Dispatcher`` (layer 4)  routes a validated call to the real source (phase 3)

The serving layer depends only on ``Retriever`` + ``Dispatcher`` (not concrete
classes), so retrieval and dispatch can be swapped without touching serve/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from okts.core.model import Bundle, OKTConcept


@dataclass
class SearchHit:
    """One phase-1 result. ``score`` is opaque; ``via`` records why it surfaced
    (``"rank"`` for direct match, or ``"alternatives"``/``"composes_with"``/
    ``"prerequisites"`` when pulled in by graph expansion)."""

    id: str
    title: str
    description: str
    score: float
    via: str = "rank"
    source_id: str | None = None  # the concept that expanded to this one, if any

    def to_ref(self) -> dict[str, Any]:
        """The lightweight ref returned to the agent (no schema)."""
        return {"id": self.id, "title": self.title, "description": self.description}


@runtime_checkable
class Adapter(Protocol):
    """Layer 1: normalize a source into OKT concepts. Keep it mechanical; leave
    body enrichment to an ``Enricher``."""

    interface: str  # one of Interface values, e.g. "mcp"

    def load(self, source: Any) -> list[OKTConcept]:
        ...


@runtime_checkable
class Enricher(Protocol):
    """Layer 1½: expand a concept's body with synonyms / when-to / gotchas.

    Implementations may call an LLM, but MUST offer a deterministic offline path
    so bundles build in CI without network/keys."""

    def enrich(self, concept: OKTConcept, bundle: Bundle | None = None) -> OKTConcept:
        ...


@runtime_checkable
class Retriever(Protocol):
    """Layer 3 / phase 1: rank concepts for a query.

    ``index(bundle)`` is called once; ``search`` is called per query and returns
    ranked ``SearchHit``s (lightweight refs, never schemas)."""

    def index(self, bundle: Bundle) -> None:
        ...

    def search(self, query: str, k: int = 5, **opts: Any) -> list[SearchHit]:
        ...


@runtime_checkable
class Dispatcher(Protocol):
    """Layer 4 / phase 3: route a validated call to the real source.

    Credentials are applied HERE, inside OKTS, and never enter agent context
    (invariant #4).

    ``dispatch`` is the synchronous path. A dispatcher whose target is a
    coroutine (a live MCP session, an async function/agent/HTTP client) SHOULD
    also provide ``adispatch`` so ``OKTSService.acall_tool`` can await it
    natively; it is optional, and the service falls back to awaiting whatever
    ``dispatch`` returns when it is missing."""

    def dispatch(self, concept: OKTConcept, args: dict[str, Any]) -> Any:
        ...

    def supports(self, concept: OKTConcept) -> bool:
        ...

    # Optional async path: a dispatcher MAY also define
    #   async def adispatch(self, concept, args) -> Any
    # It is deliberately NOT declared as a Protocol member — doing so would make
    # the ``@runtime_checkable`` structural check require it, and sync-only
    # dispatchers would stop satisfying ``Dispatcher``. ``OKTSService.acall_tool``
    # duck-types it via ``getattr`` and falls back to awaiting ``dispatch``'s
    # result when it is absent.
