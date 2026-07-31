"""Layer 1½ — structural auto-linking: derive layer-2 graph + hierarchy from a
FLAT adapted bundle.

Real sources (MCP ``tools/list``, OpenAPI, function schemas) carry no
``alternatives`` edges and no ``index.md`` category tree — those are exactly the
signals the graph-aware retriever exploits (see ``okts/index/hierarchy.py`` and
``okts/index/graph.py``). So when OKTS ingests real tools, layer 2 has to be
*derived*, not read off the source. That derivation is this module.

Two hard rules keep it honest as the thing that powers the graph-aware win:

1. **Query-independent.** Nothing here ever sees the eval queries. It derives
   structure from the concepts' own ids/names/descriptions only, so the same
   corpus always yields the same graph — it cannot be tuned to a query set.
2. **Fair to the baseline.** It only ADDS structure to a bundle. The flat-BM25
   baseline is handed the identical enriched+linked bundle; it simply ignores
   the hierarchy and edges. The graph-aware retriever's advantage is therefore
   purely "does using this derived structure help ranking", nothing else.

Heuristics (deliberately simple and mechanical):

- **hierarchy** — group concepts by ``<server>/<resource>`` where ``server`` is
  the id namespace (``github`` in ``github.create_issue``) and ``resource`` is
  the tool name's primary noun (its last ``_``-token, plural-stemmed:
  ``list_issues`` -> ``issue``). Category *paths* thus contain real words the
  hierarchy prefilter matches a query against (``postgres/query``,
  ``kubernetes/pod``).
- **alternatives** — tools sharing a ``<server>/<resource>`` group are mutual
  near-duplicates (the confusable cluster: ``create/update/list/search`` on the
  same object), so each is linked to the others as ``alternatives``. Optionally
  gated by a token-overlap floor so unrelated same-object tools aren't linked.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from okts.core.model import Bundle, OKTConcept
from okts.index.bm25 import tokenize

__all__ = ["autolink", "derive_hierarchy", "derive_edges", "server_of", "resource_of"]

log = logging.getLogger(__name__)

# Cap how many siblings one tool links to, so a large same-object group (e.g. a
# dozen github issue tools) doesn't produce a fully-connected blob. Deterministic
# (sorted) selection keeps output reproducible.
_MAX_ALTERNATIVES = 5

# Generic object-less tail tokens that make poor resource nouns on their own;
# fall back to an earlier token when the last one is one of these.
_WEAK_RESOURCE_TOKENS = frozenset(
    {"by", "id", "all", "one", "many", "list", "get", "for", "to", "from", "with"}
)


def _stem(token: str) -> str:
    """Trivial plural stem (strip one trailing 's'), matching the hierarchy
    prefilter's stemmer so derived category paths line up with query tokens."""
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def server_of(concept_id: str) -> str:
    """Namespace of an id: ``github`` for ``github.create_issue``."""
    return concept_id.split(".", 1)[0] if "." in concept_id else concept_id


def resource_of(concept_id: str) -> str:
    """Primary resource noun of a tool: the tool name's last meaningful
    ``_``-token, plural-stemmed. ``github.list_issues`` -> ``issue``,
    ``kubernetes.get_pod_logs`` -> ``log``, ``postgres.run_query`` -> ``query``.
    """
    name = concept_id.split(".", 1)[1] if "." in concept_id else concept_id
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    # walk from the end to the first non-weak token
    for tok in reversed(parts):
        if tok not in _WEAK_RESOURCE_TOKENS:
            return _stem(tok)
    return _stem(parts[-1])


def _group_key(concept: OKTConcept) -> tuple[str, str]:
    return server_of(concept.id), resource_of(concept.id)


def derive_hierarchy(bundle: Bundle) -> dict[str, list[str]]:
    """Group every concept under a ``"<server>/<resource>"`` category path.

    Deterministic: categories and their members are sorted. This is exactly the
    shape ``index.md`` / ``bundle.hierarchy`` expects (path -> list of ids).
    """
    groups: dict[str, list[str]] = {}
    for concept in bundle:
        server, resource = _group_key(concept)
        groups.setdefault(f"{server}/{resource}", []).append(concept.id)
    return {path: sorted(ids) for path, ids in sorted(groups.items())}


def derive_edges(
    bundle: Bundle, *, min_overlap: float = 0.0
) -> dict[str, list[str]]:
    """Compute ``alternatives`` for each concept id.

    Members of the same ``<server>/<resource>`` group are candidate near-
    duplicates. ``min_overlap`` (Jaccard over description+name tokens) optionally
    filters out same-object tools that are otherwise unrelated; the default 0.0
    links all same-group siblings (the confusable cluster). Result is sorted and
    capped at :data:`_MAX_ALTERNATIVES` per concept for reproducibility.
    """
    # bucket ids by group
    buckets: dict[tuple[str, str], list[str]] = {}
    for concept in bundle:
        buckets.setdefault(_group_key(concept), []).append(concept.id)

    def _tokens(cid: str) -> set[str]:
        c = bundle.get(cid)
        if c is None:
            return set()
        name = cid.split(".", 1)[1] if "." in cid else cid
        return set(tokenize(f"{name} {c.description}"))

    edges: dict[str, list[str]] = {}
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for cid in members:
            others = [m for m in members if m != cid]
            if min_overlap > 0.0:
                ct = _tokens(cid)
                scored = []
                for other in others:
                    ot = _tokens(other)
                    union = ct | ot
                    jac = len(ct & ot) / len(union) if union else 0.0
                    if jac >= min_overlap:
                        scored.append((jac, other))
                others = [o for _, o in sorted(scored, key=lambda x: (-x[0], x[1]))]
            if others:
                edges[cid] = others[:_MAX_ALTERNATIVES]
    return edges


def autolink(bundle: Bundle, *, min_overlap: float = 0.0) -> Bundle:
    """Return a NEW bundle with a derived hierarchy and ``alternatives`` edges.

    Pure and deterministic: the input bundle is untouched, and the same input
    always yields byte-identical structure. Existing ``alternatives`` on a
    concept are preserved and unioned with the derived ones (so a source that
    *did* declare edges keeps them).
    """
    hierarchy = derive_hierarchy(bundle)
    edges = derive_edges(bundle, min_overlap=min_overlap)

    out = Bundle(hierarchy=hierarchy)
    for concept in bundle:
        derived = edges.get(concept.id, [])
        if derived:
            existing = list(concept.alternatives)
            merged = existing + [d for d in derived if d not in existing]
            log.debug("autolink: %r -> alternatives %s", concept.id, merged)
            out.add(replace(concept, alternatives=merged))
        else:
            out.add(replace(concept))
    log.info(
        "autolink: derived %d categories and alternatives edges for %d concepts",
        len(hierarchy), len(edges),
    )
    return out
