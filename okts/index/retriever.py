"""Layer 3 / phase 1 — the two ``Retriever`` implementations.

``FlatBM25Retriever`` is the BASELINE the eval harness (``okts/eval/``)
measures against: BM25 only, no hierarchy prefilter, no graph expansion.

``GraphAwareRetriever`` is the project's actual contribution: hybrid
(BM25 + dense) ranking, scoped/boosted by the ``index.md`` hierarchy, then
expanded over the concept graph. Hybrid ranking itself is standard IR — the
novelty is doing it over layer-2's cross-linked graph + category hierarchy to
disambiguate near-duplicate tools (see CLAUDE.md "Architecture").

Both classes implement ``okts.core.protocols.Retriever``:
``index(bundle)`` once, then ``search(query, k) -> list[SearchHit]`` per
query. Results are always lightweight ``SearchHit``s — never schemas
(invariant: phase 1 returns refs, phase 2 loads the contract).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from okts.core.model import Bundle, OKTConcept
from okts.core.protocols import SearchHit
from okts.index.bm25 import BM25Index
from okts.index.dense import DenseIndex, EmbedFn
from okts.index.graph import expand as graph_expand_fn
from okts.index.hierarchy import HierarchyPrefilter
from okts.index.hybrid import fuse

log = logging.getLogger(__name__)


class FlatBM25Retriever:
    """Baseline retriever: BM25 over ``match_text()`` only.

    No hierarchy prefilter, no graph expansion — every hit's ``via`` is
    ``"rank"``. This is deliberately the simplest thing that could plausibly
    work, so the eval harness has a clean baseline to beat.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._bm25 = BM25Index(k1=k1, b=b)
        self._bundle: Optional[Bundle] = None

    def index(self, bundle: Bundle) -> None:
        self._bundle = bundle
        self._bm25.fit({c.id: c.match_text() for c in bundle})

    def search(self, query: str, k: int = 5, **opts: Any) -> list[SearchHit]:
        if self._bundle is None:
            raise RuntimeError("FlatBM25Retriever.search() called before index()")
        scores = self._bm25.score(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        hits: list[SearchHit] = []
        for concept_id, score in ranked:
            if score <= 0.0:
                continue  # no term overlap at all: not a match, don't pad results with it
            concept = self._bundle.get(concept_id)
            if concept is None:
                continue
            hits.append(
                SearchHit(
                    id=concept.id,
                    title=concept.title,
                    description=concept.description,
                    score=score,
                    via="rank",
                )
            )
            if len(hits) >= k:
                break
        return hits


def _dense_doc_text(concept: OKTConcept) -> str:
    """Text fed to the dense embedder for one concept.

    Deliberately narrower than ``match_text()`` (which BM25 indexes as-is,
    per spec): description and tags only, each repeated so they dominate the
    embedding, and the body dropped entirely. A tool's own body routinely
    name-drops its siblings by id ("don't confuse with list_issues", "use
    find_issues_by_label instead") to steer a *human* reader away from the
    near-duplicate — but as raw text that cross-reference is lexically
    indistinguishable from an endorsement, and it is exactly what confuses
    BM25 on the near-duplicate collision cases (e.g. the three
    issue-reading tools). Anchoring the dense signal on the concept's own
    description/tags keeps it describing what the tool actually IS rather
    than what it merely mentions in passing.
    """
    tags_text = " ".join(concept.tags)
    return "\n".join([concept.description, concept.description, tags_text, tags_text])


class GraphAwareRetriever:
    """The full retriever: hybrid rank + hierarchy prefilter + graph expansion.

    ``mode`` selects the ranking signal(s): ``"bm25"``, ``"dense"``, or
    ``"hybrid"`` (both, combined via ``fusion``: ``"weighted_sum"`` or
    ``"rrf"``, see ``okts.index.hybrid``). ``hierarchy_prefilter`` and
    ``graph_expand`` are independently toggleable so the eval harness can
    ablate each signal. Every constructor default can be overridden per-call
    via ``search(..., **opts)`` using the same keyword names.
    """

    def __init__(
        self,
        *,
        mode: str = "hybrid",
        fusion: str = "weighted_sum",
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        hierarchy_prefilter: bool = True,
        hierarchy_boost: float = 0.15,
        hierarchy_top_categories: int = 2,
        hierarchy_restrict: bool = False,
        graph_expand: bool = True,
        graph_max_per_hit: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
        dense_dim: int = 256,
        embed_fn: Optional[EmbedFn] = None,
    ):
        self.mode = mode
        self.fusion = fusion
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.hierarchy_prefilter = hierarchy_prefilter
        self.hierarchy_boost = hierarchy_boost
        self.hierarchy_top_categories = hierarchy_top_categories
        self.hierarchy_restrict = hierarchy_restrict
        self.graph_expand = graph_expand
        self.graph_max_per_hit = graph_max_per_hit

        self._bm25 = BM25Index(k1=k1, b=b)
        self._dense = DenseIndex(dim=dense_dim, embed_fn=embed_fn)
        self._bundle: Optional[Bundle] = None
        self._hierarchy: Optional[HierarchyPrefilter] = None

    def index(self, bundle: Bundle) -> None:
        self._bundle = bundle
        if self.mode in ("bm25", "hybrid"):
            self._bm25.fit({c.id: c.match_text() for c in bundle})
        if self.mode in ("dense", "hybrid"):
            self._dense.fit({c.id: _dense_doc_text(c) for c in bundle})
        self._hierarchy = HierarchyPrefilter(bundle)
        n_concepts = sum(1 for _ in bundle)
        log.info(
            "indexed %d concepts (mode=%s, hierarchy=%s, graph_expand=%s, %d categories)",
            n_concepts, self.mode, self.hierarchy_prefilter, self.graph_expand,
            len(bundle.hierarchy),
        )

    def search(self, query: str, k: int = 5, **opts: Any) -> list[SearchHit]:
        if self._bundle is None or self._hierarchy is None:
            raise RuntimeError("GraphAwareRetriever.search() called before index()")

        mode = opts.get("mode", self.mode)
        fusion = opts.get("fusion", self.fusion)
        bm25_weight = opts.get("bm25_weight", self.bm25_weight)
        dense_weight = opts.get("dense_weight", self.dense_weight)
        rrf_k = opts.get("rrf_k", self.rrf_k)
        use_hierarchy = opts.get("hierarchy_prefilter", self.hierarchy_prefilter)
        use_graph = opts.get("graph_expand", self.graph_expand)
        hierarchy_boost = opts.get("hierarchy_boost", self.hierarchy_boost)
        hierarchy_top_categories = opts.get(
            "hierarchy_top_categories", self.hierarchy_top_categories
        )
        hierarchy_restrict = opts.get("hierarchy_restrict", self.hierarchy_restrict)
        graph_max_per_hit = opts.get("graph_max_per_hit", self.graph_max_per_hit)

        log.debug(
            "search q=%r k=%d mode=%s fusion=%s (bm25=%.2f/dense=%.2f) hierarchy=%s graph=%s",
            query, k, mode, fusion, bm25_weight, dense_weight, use_hierarchy, use_graph,
        )

        bm25_scores = self._bm25.score(query) if mode in ("bm25", "hybrid") else {}
        dense_scores = self._dense.score(query) if mode in ("dense", "hybrid") else {}

        if mode == "bm25":
            fused = dict(bm25_scores)
        elif mode == "dense":
            fused = dict(dense_scores)
        elif mode == "hybrid":
            fused = fuse(
                bm25_scores,
                dense_scores,
                method=fusion,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                rrf_k=rrf_k,
            )
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        if use_hierarchy:
            fused = self._hierarchy.boost(
                query,
                fused,
                top_categories=hierarchy_top_categories,
                boost=hierarchy_boost,
                restrict=hierarchy_restrict,
            )
            if log.isEnabledFor(logging.DEBUG):
                cat_scores = self._hierarchy.score_categories(query)
                top = sorted(cat_scores.items(), key=lambda kv: kv[1], reverse=True)[
                    :hierarchy_top_categories
                ]
                log.debug(
                    "hierarchy matched %d categories; top-%d boosted (+%.2f): %s",
                    len(cat_scores), hierarchy_top_categories, hierarchy_boost, top,
                )

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        rank_hits: list[SearchHit] = []
        for concept_id, score in ranked:
            if mode == "bm25" and score <= 0.0:
                continue  # no term overlap at all under pure-BM25 semantics
            concept = self._bundle.get(concept_id)
            if concept is None:
                continue
            rank_hits.append(
                SearchHit(
                    id=concept.id,
                    title=concept.title,
                    description=concept.description,
                    score=score,
                    via="rank",
                )
            )
            if len(rank_hits) >= k:
                break

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "ranked %d hits: %s",
                len(rank_hits),
                [(h.id, round(h.score, 4)) for h in rank_hits],
            )

        if not use_graph or not rank_hits:
            return rank_hits[:k]

        graph_hits = graph_expand_fn(self._bundle, rank_hits, max_per_hit=graph_max_per_hit)
        if not graph_hits:
            log.debug("graph expansion added no siblings")
            return rank_hits[:k]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "graph expansion surfaced %d siblings: %s",
                len(graph_hits),
                [(h.id, h.via, round(h.score, 4)) for h in graph_hits],
            )

        # k is the caller's TOTAL budget, not just the rank budget: an agent
        # asking search_tools(k=5) must not get 15 refs dumped into context
        # (each ref is real tokens — see okts/eval/tokens.py). So graph
        # siblings COMPETE WITHIN k on their damped scores rather than extend
        # past it. Rank hits win ties for a shared id (a tool surfaced on its
        # own merits keeps via="rank"); the merged set is re-sorted by score
        # and truncated to k. Because every sibling is damped strictly below
        # the seed it came from, the top-ranked hit is always a rank hit, so
        # accuracy@1 is unchanged while the token cost drops to the flat
        # budget — a strong sibling only ever displaces a WEAKER rank hit in
        # the tail of the window.
        merged: dict[str, SearchHit] = {}
        for hit in rank_hits:
            merged[hit.id] = hit
        for hit in graph_hits:
            merged.setdefault(hit.id, hit)
        ordered = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        result = ordered[:k]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "returning %d hits within k=%d budget: %s",
                len(result), k, [(h.id, h.via) for h in result],
            )
        return result
