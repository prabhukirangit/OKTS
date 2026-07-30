"""Layer 3 — retrieval/index: rank OKT concepts for a query (phase 1).

Implements the ``Retriever`` protocol. Contains BM25, dense, hybrid fusion, the
``index.md`` hierarchy prefilter, and graph expansion over ``alternatives`` /
``composes_with`` / ``prerequisites``.

The INNOVATION lives here: hybrid ranking itself is standard IR; the novelty is
doing it over layer-2's cross-linked graph + category hierarchy to disambiguate
near-duplicate tools. Populated by Phase 1B.
"""
