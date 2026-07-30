"""Layer 3 / phase 1 — hierarchy prefilter over the ``index.md`` category tree.

This is half of the project's actual contribution (the other half is graph
expansion in ``graph.py``): score ``bundle.hierarchy`` category paths against
the query, then use the best-matching categories to scope/boost the ranked
candidate set before the final sort. A category's members are treated as
*siblings* — this prefilter narrows down *which category* is relevant to the
query; it deliberately does NOT try to rank between siblings within a
category (that discrimination is BM25/dense/graph's job).

Degrades to a no-op whenever the signal isn't there: an empty
``bundle.hierarchy``, or a query that matches no category at all, both leave
scores untouched — the graph-aware retriever never regresses below flat BM25
because of this module.
"""

from __future__ import annotations

import re

from okts.core.model import Bundle
from okts.index.bm25 import tokenize

_PATH_SPLIT_RE = re.compile(r"[/_\-]+")


def _stem(token: str) -> str:
    """Trivial plural stemmer (strip one trailing 's') so "issues" ~ "issue".

    Category paths ("github/issues") and queries ("get all issues...") don't
    reliably agree on singular/plural; this is the cheapest fix that doesn't
    require a real stemmer dependency.
    """
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _path_tokens(path: str) -> set[str]:
    return {_stem(t) for t in _PATH_SPLIT_RE.split(path.lower()) if t}


class HierarchyPrefilter:
    """Scores ``bundle.hierarchy`` categories against a query and uses the
    best matches to scope/boost a candidate ``{doc_id: score}`` ranking."""

    def __init__(self, bundle: Bundle):
        self._hierarchy: dict[str, list[str]] = dict(bundle.hierarchy or {})
        self._category_members: dict[str, list[str]] = {}
        for path, members in self._hierarchy.items():
            resolved: list[str] = []
            for ref in members:
                cid = bundle.resolve_edge(ref)
                if cid is None and ref in bundle.concepts:
                    cid = ref
                if cid is not None:
                    resolved.append(cid)
            self._category_members[path] = resolved

    def is_empty(self) -> bool:
        """True when the bundle carries no hierarchy at all — callers should
        skip straight to plain ranking rather than calling ``boost``."""
        return not self._hierarchy

    def score_categories(self, query: str) -> dict[str, float]:
        """Overlap-based relevance of each category path to the query,
        normalized to ``[0, 1]`` by category-path token count. Categories
        with zero overlap are omitted entirely."""
        q_tokens = {_stem(t) for t in tokenize(query)}
        scores: dict[str, float] = {}
        if not q_tokens:
            return scores
        for path in self._hierarchy:
            cat_tokens = _path_tokens(path)
            if not cat_tokens:
                continue
            overlap = len(q_tokens & cat_tokens)
            if overlap:
                scores[path] = overlap / len(cat_tokens)
        return scores

    def boost(
        self,
        query: str,
        scores: dict[str, float],
        *,
        top_categories: int = 2,
        boost: float = 0.15,
        restrict: bool = False,
    ) -> dict[str, float]:
        """Adjust a ``{doc_id: score}`` ranking using the hierarchy signal.

        With ``restrict=False`` (default), concepts in the best-matching
        categories get an additive ``boost``; concepts outside are untouched
        — nothing is ever removed. With ``restrict=True`` the candidate set
        is narrowed to just those categories' members (a harder prefilter).

        No-op (returns ``scores`` unchanged) if the bundle has no hierarchy,
        or if no category matches the query at all.
        """
        if self.is_empty():
            return dict(scores)
        cat_scores = self.score_categories(query)
        if not cat_scores:
            return dict(scores)
        top = sorted(cat_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_categories]
        in_scope: set[str] = set()
        for path, _ in top:
            in_scope.update(self._category_members.get(path, []))
        if not in_scope:
            return dict(scores)
        if restrict:
            return {doc_id: s for doc_id, s in scores.items() if doc_id in in_scope}
        adjusted = dict(scores)
        for doc_id in in_scope:
            if doc_id in adjusted:
                adjusted[doc_id] = adjusted[doc_id] + boost
        return adjusted
