"""Large-corpus validation of the two CLAUDE.md claims (see eval/corpus/).

Guards, on the ~150-tool / 20-server canned corpus:
- it builds via the ordinary offline pipeline and is OKF-conformant,
- the auto-linker is deterministic and query-independent,
- ~85% token reduction is reached at full corpus size (the large-corpus claim),
- graph/hierarchy-aware retrieval beats flat BM25 on accuracy at comparable cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from okts.core.validator import validate_bundle
from okts.enrich.autolink import autolink, derive_edges, derive_hierarchy
from okts.eval.corpus import (
    DEFAULT_QUERIES,
    _flat,
    _load_cases,
    build_corpus_bundle,
)
from okts.eval.harness import run_eval

CORPUS_DIR = Path(__file__).resolve().parents[1] / "eval" / "corpus"


@pytest.fixture(scope="module")
def corpus():
    return build_corpus_bundle(CORPUS_DIR)


@pytest.fixture(scope="module")
def cases():
    return _load_cases(DEFAULT_QUERIES)


def test_corpus_is_large_and_conformant(corpus):
    n = sum(1 for _ in corpus)
    # a genuine large-corpus test, not the 11-tool unit fixture
    assert n >= 140, f"corpus only has {n} tools"
    servers = {c.id.split(".", 1)[0] for c in corpus}
    assert len(servers) >= 18
    assert validate_bundle(corpus, check_edges=True) == []
    # the auto-linker produced a hierarchy to prefilter over
    assert corpus.hierarchy


def test_every_labeled_expected_id_exists(corpus, cases):
    ids = {c.id for c in corpus}
    missing = sorted({c.expected for c in cases} - ids)
    assert not missing, f"queries reference unknown tool ids: {missing}"


def test_autolink_is_deterministic_and_structural():
    # same input -> byte-identical derived structure (it never sees queries)
    flat_a = _flat(CORPUS_DIR)
    flat_b = _flat(CORPUS_DIR)
    assert derive_hierarchy(flat_a) == derive_hierarchy(flat_b)
    assert derive_edges(flat_a) == derive_edges(flat_b)

    linked = autolink(flat_a)
    # edges only ever point at same-namespace siblings (within-server clusters)
    for concept in linked:
        server = concept.id.split(".", 1)[0]
        for ref in concept.alternatives:
            assert ref.split(".", 1)[0] == server


def test_token_reduction_exceeds_85pct_at_full_corpus(corpus, cases):
    from okts.index.retriever import GraphAwareRetriever

    report = run_eval(corpus, GraphAwareRetriever(), cases, k=5)
    # the headline large-corpus claim
    assert report.token_reduction_pct > 85.0, report.token_reduction_pct


def test_graph_aware_beats_flat_on_accuracy_at_comparable_cost(corpus, cases):
    from okts.index.retriever import FlatBM25Retriever, GraphAwareRetriever

    flat = run_eval(corpus, FlatBM25Retriever(), cases, k=5)
    graph = run_eval(corpus, GraphAwareRetriever(), cases, k=5)

    # accuracy rises (or at least holds) on every headline metric...
    assert graph.accuracy_at_1 >= flat.accuracy_at_1
    assert graph.collision_avoidance >= flat.collision_avoidance
    assert graph.mrr >= flat.mrr
    # ...at comparable token cost (within 5% of the flat baseline)
    assert graph.avg_okts_tokens <= flat.avg_okts_tokens * 1.05


def test_graph_aware_strictly_wins_somewhere(corpus, cases):
    # not just >=: on this collision-rich corpus the graph/hierarchy signal
    # must actually improve top-1 or collision-avoidance, else it earns nothing.
    from okts.index.retriever import FlatBM25Retriever, GraphAwareRetriever

    flat = run_eval(corpus, FlatBM25Retriever(), cases, k=5)
    graph = run_eval(corpus, GraphAwareRetriever(), cases, k=5)
    assert (
        graph.accuracy_at_1 > flat.accuracy_at_1
        or graph.collision_avoidance > flat.collision_avoidance
    )
