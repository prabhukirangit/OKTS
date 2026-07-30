"""Layer 3 / phase 1 — graph expansion over the concept graph.

Given the top-ranked hits from BM25/dense/hybrid, pull in each hit's graph
neighbors (``alternatives`` / ``composes_with`` / ``prerequisites``, resolved
through ``bundle.resolve_edge``) as additional ``SearchHit``s. This is the
other half of the project's actual contribution: near-duplicate tools (the
three issue-reading tools in the fixture bundle, cross-linked via
``alternatives``) get stitched together, so once a query lands on ANY sibling
the close alternatives still surface for a second look, and a tool's usual
next steps (``composes_with``/``prerequisites``) surface alongside it.
"""

from __future__ import annotations

from okts.core.model import Bundle
from okts.core.protocols import SearchHit

# How much a graph-expanded hit's score is damped relative to the hit it was
# pulled in from. Alternatives are near-duplicates of the seed hit, so they
# stay close to it; composes_with/prerequisites are a genuinely different
# tool the agent would reach for next, so they're damped further.
EDGE_DAMPING: dict[str, float] = {
    "alternatives": 0.85,
    "composes_with": 0.6,
    "prerequisites": 0.6,
}

_DEFAULT_DAMPING = 0.5


def expand(
    bundle: Bundle,
    hits: list[SearchHit],
    *,
    max_per_hit: int = 3,
    damping: dict[str, float] | None = None,
) -> list[SearchHit]:
    """Return NEW ``SearchHit``s reachable from ``hits`` via graph edges.

    Never duplicates an id already present in ``hits``, nor one already added
    earlier in this same expansion pass (so a neighbor shared by two seed
    hits is only added once, from whichever seed reaches it first). Each
    returned hit's ``via`` names the edge type it was pulled in through
    (``"alternatives"`` / ``"composes_with"`` / ``"prerequisites"``) and
    ``source_id`` names the seed hit it expanded from. Sorted by score
    descending.
    """
    damping = damping or EDGE_DAMPING
    seen = {hit.id for hit in hits}
    expanded: list[SearchHit] = []

    for hit in hits:
        concept = bundle.get(hit.id)
        if concept is None:
            continue
        edge_groups = (
            ("alternatives", concept.alternatives),
            ("composes_with", concept.composes_with),
            ("prerequisites", concept.prerequisites),
        )
        added = 0
        for edge_type, refs in edge_groups:
            if added >= max_per_hit:
                break
            for ref in refs:
                if added >= max_per_hit:
                    break
                neighbor_id = bundle.resolve_edge(ref)
                if neighbor_id is None or neighbor_id in seen:
                    continue
                neighbor = bundle.get(neighbor_id)
                if neighbor is None:
                    continue
                expanded.append(
                    SearchHit(
                        id=neighbor.id,
                        title=neighbor.title,
                        description=neighbor.description,
                        score=hit.score * damping.get(edge_type, _DEFAULT_DAMPING),
                        via=edge_type,
                        source_id=hit.id,
                    )
                )
                seen.add(neighbor_id)
                added += 1

    expanded.sort(key=lambda h: h.score, reverse=True)
    return expanded
