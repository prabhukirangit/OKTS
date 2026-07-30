"""Layer 3 / phase 1 — fuse BM25 + dense scores into one ranking.

Hybrid ranking is standard IR; two fusion strategies are supported and
selectable via ``method`` so the eval harness can ablate them:

- ``"weighted_sum"`` — min-max normalize each score dict to ``[0, 1]``, then
  take a weighted sum. Cheap, intuitive to tune, but sensitive to the raw
  score distributions.
- ``"rrf"`` (Reciprocal Rank Fusion) — combine using each list's *rank*, not
  its raw score, so it is robust to BM25 and dense scores living on very
  different scales (Cormack, Clarke & Buettcher, 2009).
"""

from __future__ import annotations

from typing import Literal

FusionMethod = Literal["weighted_sum", "rrf"]


def normalize_minmax(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score dict to ``[0, 1]``.

    An empty or all-equal input normalizes to all ``0.0`` (rather than
    dividing by zero) so a degenerate signal contributes nothing to a fused
    sum instead of raising.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    spread = hi - lo
    if spread <= 1e-12:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (v - lo) / spread for doc_id, v in scores.items()}


def reciprocal_rank_fusion(
    rankings: list[dict[str, float]], *, rrf_k: int = 60
) -> dict[str, float]:
    """Fuse N score dicts by rank rather than raw score (RRF).

    ``rrf_k`` is the standard damping constant (60 in the original paper);
    larger values flatten the influence of top ranks.
    """
    fused: dict[str, float] = {}
    for scores in rankings:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def fuse(
    bm25_scores: dict[str, float],
    dense_scores: dict[str, float],
    *,
    method: FusionMethod = "weighted_sum",
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Combine BM25 and dense score dicts into one ``{doc_id: fused_score}``.

    Either side may be an empty dict (e.g. a pure-BM25 or pure-dense mode
    only ever computed one of the two); the missing side then contributes
    nothing to the fused score.
    """
    if method == "rrf":
        rankings = [s for s in (bm25_scores, dense_scores) if s]
        return reciprocal_rank_fusion(rankings, rrf_k=rrf_k)
    if method == "weighted_sum":
        norm_bm25 = normalize_minmax(bm25_scores)
        norm_dense = normalize_minmax(dense_scores)
        all_ids = set(norm_bm25) | set(norm_dense)
        return {
            doc_id: bm25_weight * norm_bm25.get(doc_id, 0.0)
            + dense_weight * norm_dense.get(doc_id, 0.0)
            for doc_id in all_ids
        }
    raise ValueError(f"unknown fusion method: {method!r}")
