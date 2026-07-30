"""Run the labeled query set through any ``Retriever`` and produce a report.

Depends only on the ``Retriever`` protocol + ``SearchHit`` from ``okts.core``
-- never a concrete retriever class -- so :func:`run_eval` works unmodified
against the flat-BM25 baseline, the graph-aware retriever, or any future
implementation. Concrete retrievers are wired up by callers (see
``okts/eval/run.py``), not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from okts.core.model import Bundle
from okts.core.protocols import Retriever, SearchHit

from okts.eval.metrics import (
    Case,
    accuracy_at_1,
    accuracy_at_k,
    collision_avoidance_rate,
    mean_reciprocal_rank,
)
from okts.eval.tokens import okts_query_cost, raw_tools_cost


@dataclass
class EvalCase:
    """One labeled query, as loaded from ``eval/queries.yaml``."""

    query: str
    expected: str
    distractors: list[str] = field(default_factory=list)


@dataclass
class CaseReport:
    """Per-query outcome: what the retriever returned and what it cost."""

    query: str
    expected: str
    distractors: list[str]
    ranked_ids: list[str]
    tokens: int


@dataclass
class EvalReport:
    """Aggregate result of running one retriever over the labeled set.

    Always carries BOTH the selection-accuracy metrics AND the token-cost
    numbers side by side (CLAUDE.md "Testing / eval expectations": never
    report one without the other).
    """

    retriever_name: str
    num_cases: int
    accuracy_at_1: float
    accuracy_at_k: float
    k: int
    mrr: float
    collision_avoidance: float
    raw_tools_tokens: int
    avg_okts_tokens: float
    token_reduction_pct: float
    cases: list[CaseReport] = field(default_factory=list)


def run_eval(
    bundle: Bundle,
    retriever: Retriever,
    cases: list[EvalCase] | list[dict[str, Any]],
    k: int = 5,
) -> EvalReport:
    """Index ``retriever`` on ``bundle``, run every case through ``search``,
    and return an :class:`EvalReport` with both accuracy and token cost.

    ``retriever`` need only satisfy the ``Retriever`` protocol (``index`` +
    ``search``) -- this function has no import-time dependency on any
    concrete implementation, so it works against a test stub just as well as
    ``okts.index.retriever.FlatBM25Retriever`` / ``GraphAwareRetriever``.
    ``cases`` may be :class:`EvalCase` instances or plain dicts with
    ``query``/``expected``/``distractors`` keys (e.g. straight off
    ``yaml.safe_load``).
    """
    parsed_cases = [c if isinstance(c, EvalCase) else EvalCase(**c) for c in cases]

    retriever.index(bundle)

    raw_cost = raw_tools_cost(bundle)

    case_reports: list[CaseReport] = []
    metric_cases: list[Case] = []
    total_okts_tokens = 0

    for case in parsed_cases:
        hits: list[SearchHit] = retriever.search(case.query, k=k)
        ranked_ids = [hit.id for hit in hits]
        tokens = okts_query_cost(bundle, hits)
        total_okts_tokens += tokens

        case_reports.append(
            CaseReport(
                query=case.query,
                expected=case.expected,
                distractors=case.distractors,
                ranked_ids=ranked_ids,
                tokens=tokens,
            )
        )
        metric_cases.append((ranked_ids, case.expected, case.distractors))

    n = len(parsed_cases)
    avg_okts_tokens = total_okts_tokens / n if n else 0.0
    reduction = 100.0 * (1 - avg_okts_tokens / raw_cost) if raw_cost else 0.0

    return EvalReport(
        retriever_name=type(retriever).__name__,
        num_cases=n,
        accuracy_at_1=accuracy_at_1(metric_cases),
        accuracy_at_k=accuracy_at_k(metric_cases, k=k),
        k=k,
        mrr=mean_reciprocal_rank(metric_cases),
        collision_avoidance=collision_avoidance_rate(metric_cases),
        raw_tools_tokens=raw_cost,
        avg_okts_tokens=avg_okts_tokens,
        token_reduction_pct=reduction,
        cases=case_reports,
    )
