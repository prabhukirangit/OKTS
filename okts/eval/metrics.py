"""Retrieval metrics over the labeled query set.

Two tiers of pure functions, both operating on plain ``(ranked_ids, expected,
distractors)`` data -- no I/O, no dependency on a concrete ``Retriever`` -- so
they're trivial to unit test and reusable by any harness:

- **per-case** (``hit_at_1``, ``hit_at_k``, ``reciprocal_rank``,
  ``avoids_collision``): evaluate a single query's ranking.
- **aggregate** (``accuracy_at_1``, ``accuracy_at_k``,
  ``mean_reciprocal_rank``, ``collision_avoidance_rate``): reduce a list of
  ``(ranked_ids, expected, distractors)`` cases to one rate in ``[0, 1]``.

``collision_avoidance_rate`` is the metric that isolates the graph/hierarchy
signal: it measures how often the correct tool outranks the near-duplicate
tools listed as its ``distractors``, which is the specific failure mode the
OKT graph edges (``alternatives`` / ``composes_with`` / ``prerequisites``) are
meant to fix (see CLAUDE.md "Testing / eval expectations").
"""

from __future__ import annotations

# One case: (ranked_ids returned by search, the correct tool id, its listed
# near-duplicate distractor ids).
Case = tuple[list[str], str, list[str]]


def hit_at_1(ranked_ids: list[str], expected: str) -> bool:
    """True iff ``expected`` is the top-ranked id."""
    return bool(ranked_ids) and ranked_ids[0] == expected


def hit_at_k(ranked_ids: list[str], expected: str, k: int) -> bool:
    """True iff ``expected`` appears anywhere in the top ``k`` ranked ids."""
    return expected in ranked_ids[:k]


def reciprocal_rank(ranked_ids: list[str], expected: str) -> float:
    """``1 / rank`` of ``expected`` in ``ranked_ids`` (1-indexed), or ``0.0``
    if it never appears."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid == expected:
            return 1.0 / i
    return 0.0


def avoids_collision(ranked_ids: list[str], expected: str, distractors: list[str]) -> bool:
    """True iff ``expected`` outranks every listed distractor that appears in
    ``ranked_ids`` -- i.e. the retriever didn't let a near-duplicate tool win.

    If ``expected`` is absent from ``ranked_ids`` the case only counts as
    avoided when no distractor beat it there either (nothing surfaced ahead of
    it); if no distractors are listed/present there is nothing to collide
    with, so the case counts as avoided.
    """
    if expected not in ranked_ids:
        return not any(d in ranked_ids for d in distractors)
    expected_rank = ranked_ids.index(expected)
    return all(
        ranked_ids.index(d) >= expected_rank for d in distractors if d in ranked_ids
    )


def accuracy_at_1(cases: list[Case]) -> float:
    """Fraction of ``cases`` where the correct tool is ranked first."""
    if not cases:
        return 0.0
    return sum(hit_at_1(ranked, expected) for ranked, expected, _ in cases) / len(cases)


def accuracy_at_k(cases: list[Case], k: int = 5) -> float:
    """Fraction of ``cases`` where the correct tool appears in the top ``k``."""
    if not cases:
        return 0.0
    return sum(hit_at_k(ranked, expected, k) for ranked, expected, _ in cases) / len(cases)


def mean_reciprocal_rank(cases: list[Case]) -> float:
    """Mean of ``1/rank`` of the correct tool across ``cases``."""
    if not cases:
        return 0.0
    return sum(reciprocal_rank(ranked, expected) for ranked, expected, _ in cases) / len(cases)


def collision_avoidance_rate(cases: list[Case]) -> float:
    """Fraction of ``cases`` where the correct tool outranked its distractors."""
    if not cases:
        return 0.0
    return sum(
        avoids_collision(ranked, expected, distractors) for ranked, expected, distractors in cases
    ) / len(cases)
