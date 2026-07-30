"""Layer 1 — adapters: normalize each source type into OKT concepts.

Each adapter is a (mostly) pure function ``source -> list[OKTConcept]`` and must
always emit the required-minimum frontmatter (``type``, ``id``, ``title``,
``description``, ``input_schema``, ``interface``). Body enrichment is a separate
pass (see ``okts.enrich``).

Populated by Phase 1A. Keep adapters mechanical; do not smuggle ranking or
dispatch logic in here.
"""
