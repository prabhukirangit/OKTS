"""Eval-harness tests: metrics, token cost, and the run_eval() harness.

The harness/metrics tests use a tiny deterministic stub retriever defined
in-test -- NOT okts.index, which is being built in parallel. run_eval() only
depends on the Retriever protocol (index + search), so any object with those
two methods satisfies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from okts.core.protocols import SearchHit
from okts.eval.harness import EvalCase, EvalReport, run_eval
from okts.eval.metrics import (
    accuracy_at_1,
    accuracy_at_k,
    avoids_collision,
    collision_avoidance_rate,
    hit_at_1,
    hit_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
)
from okts.eval.tokens import estimate_tokens, okts_query_cost, raw_tools_cost

# ---------------------------------------------------------------------------
# metrics.py -- per-case functions
# ---------------------------------------------------------------------------


def test_hit_at_1_true_when_top_ranked():
    assert hit_at_1(["a", "b", "c"], "a") is True


def test_hit_at_1_false_when_not_top():
    assert hit_at_1(["b", "a", "c"], "a") is False


def test_hit_at_1_false_when_ranking_empty():
    assert hit_at_1([], "a") is False


def test_hit_at_k():
    assert hit_at_k(["b", "c", "a"], "a", k=3) is True
    assert hit_at_k(["b", "c", "a"], "a", k=2) is False


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], "a") == 1.0
    assert reciprocal_rank(["b", "a"], "a") == 0.5
    assert reciprocal_rank(["b", "c"], "a") == 0.0


def test_avoids_collision_true_when_expected_outranks_distractor():
    assert avoids_collision(["a", "b"], "a", ["b"]) is True


def test_avoids_collision_false_when_distractor_wins():
    assert avoids_collision(["b", "a"], "a", ["b"]) is False


def test_avoids_collision_true_when_nothing_present():
    assert avoids_collision(["z"], "a", ["b"]) is True


def test_avoids_collision_false_when_expected_absent_but_distractor_present():
    assert avoids_collision(["b"], "a", ["b"]) is False


# ---------------------------------------------------------------------------
# metrics.py -- aggregate functions, on hand-built rankings
# ---------------------------------------------------------------------------


def test_aggregate_metrics_on_hand_built_rankings():
    # 3 hand-built cases: 2 correct@1, 1 correct only at rank 2 with a
    # distractor that beats it (a collision).
    cases = [
        (["x", "y"], "x", ["y"]),  # hit@1, no collision
        (["y", "x"], "x", ["y"]),  # miss@1, hit@2, collision (y beats x)
        (["z", "w"], "z", []),  # hit@1, no distractors listed
    ]

    assert accuracy_at_1(cases) == pytest.approx(2 / 3)
    assert accuracy_at_k(cases, k=2) == pytest.approx(1.0)
    assert mean_reciprocal_rank(cases) == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert collision_avoidance_rate(cases) == pytest.approx(2 / 3)


def test_aggregate_metrics_empty_is_zero():
    assert accuracy_at_1([]) == 0.0
    assert accuracy_at_k([], k=5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
    assert collision_avoidance_rate([]) == 0.0


# ---------------------------------------------------------------------------
# tokens.py
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_tokens_grows_with_length():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_raw_tools_cost_sums_all_concepts(bundle):
    total = raw_tools_cost(bundle)
    assert total > 0
    single = estimate_tokens(bundle.get("github.create_issue").description)
    assert total > single  # corpus cost dwarfs any one concept's description


def test_okts_per_query_cost_much_cheaper_than_raw_tools(bundle):
    """The core claim this harness exists to prove: OKTS's per-query cost is
    a small fraction of the raw-tools "hand the agent every schema" baseline
    (CLAUDE.md targets ~85% reduction at production corpus sizes; the fixture
    bundle only has 11 concepts, so the margin here is smaller but still
    large and unambiguously positive)."""
    hits = [
        SearchHit(id=c.id, title=c.title, description=c.description, score=1.0)
        for c in list(bundle)[:5]
    ]
    raw = raw_tools_cost(bundle)
    per_query = okts_query_cost(bundle, hits)

    assert per_query < raw
    reduction = 100.0 * (1 - per_query / raw)
    assert reduction > 40.0  # comfortably large and positive even on 11 tools


def test_reduction_approaches_target_as_corpus_grows(bundle):
    """The ~85% reduction target is a property of corpus SIZE: raw-tools cost
    grows linearly with N while OKTS's per-query cost stays ~constant (meta
    schemas + k refs + 1 loaded schema). Simulate a larger corpus by
    replicating the fixture's per-concept cost N times to demonstrate the
    trend the harness is built to measure."""
    per_concept_avg = raw_tools_cost(bundle) / len(bundle)
    hits = [
        SearchHit(id=c.id, title=c.title, description=c.description, score=1.0)
        for c in list(bundle)[:5]
    ]
    per_query = okts_query_cost(bundle, hits)

    simulated_raw_cost_at_300_tools = per_concept_avg * 300
    reduction = 100.0 * (1 - per_query / simulated_raw_cost_at_300_tools)
    assert reduction > 85.0


def test_okts_query_cost_empty_hits_is_just_meta_tool_cost(bundle):
    from okts.eval.tokens import META_TOOL_SCHEMAS_TOKENS

    assert okts_query_cost(bundle, []) == META_TOOL_SCHEMAS_TOKENS


# ---------------------------------------------------------------------------
# harness.py -- stub retriever, no dependency on okts.index
# ---------------------------------------------------------------------------


@dataclass
class _StubRetriever:
    """Deterministic in-test retriever satisfying the Retriever protocol
    (index + search). Returns a fixed ranking per query so the harness can be
    exercised without okts.index, which is being built in parallel."""

    rankings: dict[str, list[str]]
    indexed: bool = False

    def index(self, bundle) -> None:
        self.indexed = True
        self._bundle = bundle

    def search(self, query: str, k: int = 5, **opts) -> list[SearchHit]:
        ids = self.rankings.get(query, [])[:k]
        concepts = {c.id: c for c in self._bundle}
        return [
            SearchHit(
                id=cid,
                title=concepts[cid].title if cid in concepts else cid,
                description=concepts[cid].description if cid in concepts else "",
                score=1.0 / (i + 1),
            )
            for i, cid in enumerate(ids)
        ]


def test_run_eval_returns_well_formed_report(bundle):
    cases = [
        EvalCase(
            query="q1", expected="github.create_issue", distractors=["github.update_issue"]
        ),
        EvalCase(
            query="q2", expected="github.update_issue", distractors=["github.create_issue"]
        ),
    ]
    retriever = _StubRetriever(
        rankings={
            "q1": ["github.create_issue", "github.update_issue"],
            "q2": ["github.create_issue", "github.update_issue"],  # miss@1 for q2
        }
    )

    report = run_eval(bundle, retriever, cases, k=5)

    assert isinstance(report, EvalReport)
    assert retriever.indexed is True
    assert report.retriever_name == "_StubRetriever"
    assert report.num_cases == 2
    assert report.k == 5
    assert report.accuracy_at_1 == pytest.approx(0.5)
    assert report.accuracy_at_k == pytest.approx(1.0)  # both ids present within k
    assert 0.0 <= report.mrr <= 1.0
    assert 0.0 <= report.collision_avoidance <= 1.0
    assert report.raw_tools_tokens > 0
    assert report.avg_okts_tokens > 0
    assert report.avg_okts_tokens < report.raw_tools_tokens
    assert report.token_reduction_pct > 0.0
    assert len(report.cases) == 2
    assert report.cases[0].ranked_ids == ["github.create_issue", "github.update_issue"]
    assert report.cases[0].tokens > 0


def test_run_eval_accepts_plain_dict_cases(bundle):
    """run_eval should accept plain dicts too (as loaded from queries.yaml
    via yaml.safe_load), not just EvalCase instances."""
    retriever = _StubRetriever(rankings={"q": ["github.get_repo"]})
    report = run_eval(
        bundle,
        retriever,
        [{"query": "q", "expected": "github.get_repo", "distractors": []}],
    )
    assert report.accuracy_at_1 == 1.0


def test_run_eval_on_full_labeled_queries_yaml(bundle):
    """Sanity check against the real eval/queries.yaml with a stub that
    always ranks the expected id first -- should score perfectly."""
    import yaml

    queries_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "queries.yaml"
    )
    data = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    raw_cases = data["cases"]

    rankings = {
        c["query"]: [c["expected"], *c.get("distractors", [])] for c in raw_cases
    }
    retriever = _StubRetriever(rankings=rankings)

    report = run_eval(bundle, retriever, raw_cases, k=5)

    assert report.num_cases == len(raw_cases)
    assert report.accuracy_at_1 == 1.0
    assert report.collision_avoidance == 1.0
