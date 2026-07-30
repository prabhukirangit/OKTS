"""Layer 3 (retrieval/index) tests: BM25, dense, hybrid fusion, hierarchy
prefilter, graph expansion, and the two ``Retriever`` implementations.

The headline test (``test_graph_aware_beats_or_matches_flat_on_collisions``)
is the actual value proposition of this project: the graph/hierarchy signal
disambiguating near-duplicate tools, not hybrid ranking per se (see
CLAUDE.md "Architecture").
"""

from __future__ import annotations

import numpy as np
import pytest

from okts.core.model import Bundle, Interface, OKTConcept
from okts.core.protocols import Retriever, SearchHit
from okts.index.bm25 import BM25Index, tokenize
from okts.index.dense import DenseIndex, hashing_embed
from okts.index.graph import expand as graph_expand
from okts.index.hierarchy import HierarchyPrefilter
from okts.index.hybrid import fuse, normalize_minmax, reciprocal_rank_fusion
from okts.index.retriever import FlatBM25Retriever, GraphAwareRetriever

# ---------------------------------------------------------------------------
# The five collision / disambiguation queries this project is built to solve.
# ---------------------------------------------------------------------------
COLLISION_QUERIES: list[tuple[str, str]] = [
    ("open a new bug ticket in a repo", "github.create_issue"),
    ("get all issues tagged with a particular label", "github.find_issues_by_label"),
    ("search issues by free text across repos", "github.search_issues"),
    ("refund a payment", "stripe.create_refund"),
    ("post a message to a channel", "slack.send_message"),
]


def _accuracy_at_1(retriever, queries) -> float:
    correct = 0
    for query, expected_id in queries:
        hits = retriever.search(query, k=5)
        if hits and hits[0].id == expected_id:
            correct += 1
    return correct / len(queries)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Open a NEW issue, please!") == ["open", "a", "new", "issue", "please"]
    assert tokenize("create_issue's alt-name") == ["create", "issue", "s", "alt", "name"]


def test_bm25_empty_index_returns_empty_scores():
    idx = BM25Index()
    assert idx.score("anything") == {}


def test_bm25_empty_query_returns_all_zero():
    idx = BM25Index()
    idx.fit({"a": "hello world", "b": "goodbye world"})
    scores = idx.score("")
    assert scores == {"a": 0.0, "b": 0.0}


def test_bm25_ranks_exact_term_match_above_no_overlap():
    idx = BM25Index()
    idx.fit(
        {
            "a": "refund a payment reverse a charge",
            "b": "send a message to a channel",
            "c": "totally unrelated filler text",
        }
    )
    scores = idx.score("refund a payment")
    assert scores["a"] > scores["b"] > 0
    assert scores["c"] == 0.0


def test_bm25_favors_higher_term_frequency():
    idx = BM25Index()
    idx.fit({"a": "label label label issue", "b": "label issue issue issue issue"})
    # doc "a" repeats the query term "label" more -> should score at least as high
    scores = idx.score("label")
    assert scores["a"] >= scores["b"]


def test_bm25_over_fixture_match_text(bundle):
    idx = BM25Index()
    idx.fit({c.id: c.match_text() for c in bundle})
    scores = idx.score("refund a payment")
    assert scores["stripe.create_refund"] > 0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    assert ranked[0][0] == "stripe.create_refund"


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


def test_hashing_embed_is_deterministic():
    v1 = hashing_embed("post a message to a channel")
    v2 = hashing_embed("post a message to a channel")
    assert np.array_equal(v1, v2)


def test_hashing_embed_is_l2_normalized():
    v = hashing_embed("some reasonably long piece of text to embed")
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_hashing_embed_empty_text_is_zero_vector():
    v = hashing_embed("")
    assert np.allclose(v, 0.0)


def test_dense_index_empty_returns_empty_scores():
    idx = DenseIndex()
    assert idx.score("hello") == {}


def test_dense_index_cosine_similarity_prefers_similar_text():
    idx = DenseIndex()
    idx.fit(
        {
            "a": "refund a payment reverse a charge",
            "b": "send a chat message to a channel",
        }
    )
    scores = idx.score("refund a payment")
    assert scores["a"] > scores["b"]
    # cosine similarity is bounded
    for s in scores.values():
        assert -1.0001 <= s <= 1.0001


def test_dense_index_accepts_injected_embed_fn():
    # A trivial custom embedding: one-hot on text length parity. Proves the
    # DEFAULT is not hardwired into DenseIndex — a real model can be swapped in.
    def embed_fn(text: str) -> np.ndarray:
        v = np.zeros(2)
        v[len(text) % 2] = 1.0
        return v

    idx = DenseIndex(dim=2, embed_fn=embed_fn)
    idx.fit({"even": "ab", "odd": "abc"})
    scores = idx.score("xy")  # len 2 -> even
    assert scores["even"] == 1.0
    assert scores["odd"] == 0.0


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------


def test_normalize_minmax_scales_to_unit_range():
    normed = normalize_minmax({"a": 5.0, "b": 10.0, "c": 0.0})
    assert normed["a"] == pytest.approx(0.5)
    assert normed["b"] == pytest.approx(1.0)
    assert normed["c"] == pytest.approx(0.0)


def test_normalize_minmax_all_equal_is_all_zero():
    assert normalize_minmax({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}


def test_normalize_minmax_empty():
    assert normalize_minmax({}) == {}


def test_reciprocal_rank_fusion_rewards_consistent_top_rank():
    rrf = reciprocal_rank_fusion(
        [{"x": 10, "y": 1}, {"x": 5, "y": 0.5}], rrf_k=60
    )
    # x is rank-1 in both lists; y is rank-2 in both -> x must win
    assert rrf["x"] > rrf["y"]


def test_fuse_weighted_sum_respects_weights():
    bm25 = {"a": 1.0, "b": 0.0}
    dense = {"a": 0.0, "b": 1.0}
    fused_bm25_heavy = fuse(bm25, dense, method="weighted_sum", bm25_weight=1.0, dense_weight=0.0)
    fused_dense_heavy = fuse(bm25, dense, method="weighted_sum", bm25_weight=0.0, dense_weight=1.0)
    assert fused_bm25_heavy["a"] > fused_bm25_heavy["b"]
    assert fused_dense_heavy["b"] > fused_dense_heavy["a"]


def test_fuse_rrf_method():
    bm25 = {"a": 3.0, "b": 1.0}
    dense = {"a": 1.0, "b": 3.0}
    fused = fuse(bm25, dense, method="rrf")
    # symmetric ranks -> tie
    assert fused["a"] == pytest.approx(fused["b"])


def test_fuse_unknown_method_raises():
    with pytest.raises(ValueError):
        fuse({"a": 1.0}, {"a": 1.0}, method="bogus")  # type: ignore[arg-type]


def test_fuse_handles_one_sided_input():
    fused = fuse({"a": 1.0, "b": 2.0}, {}, method="weighted_sum")
    assert set(fused) == {"a", "b"}


# ---------------------------------------------------------------------------
# Hierarchy prefilter
# ---------------------------------------------------------------------------


def test_hierarchy_prefilter_not_empty_on_fixture(bundle):
    hp = HierarchyPrefilter(bundle)
    assert not hp.is_empty()


def test_hierarchy_prefilter_empty_bundle_is_noop():
    hp = HierarchyPrefilter(Bundle())
    assert hp.is_empty()
    scores = {"x": 1.0, "y": 2.0}
    assert hp.boost("anything", scores) == scores


def test_hierarchy_prefilter_no_matching_category_is_noop(bundle):
    hp = HierarchyPrefilter(bundle)
    scores = {c.id: 1.0 for c in bundle}
    # gibberish query matches no category path token
    boosted = hp.boost("zzz qqq xyzzy plugh", dict(scores))
    assert boosted == scores


def test_hierarchy_prefilter_boosts_matching_category_members(bundle):
    hp = HierarchyPrefilter(bundle)
    scores = {c.id: 0.0 for c in bundle}
    boosted = hp.boost("github issues", dict(scores), boost=0.2)
    # every github/issues member should be boosted above baseline
    for cid in ("github.create_issue", "github.list_issues", "github.find_issues_by_label"):
        assert boosted[cid] > scores[cid]
    # slack/stripe members should be untouched
    assert boosted["slack.send_message"] == 0.0
    assert boosted["stripe.create_refund"] == 0.0


def test_hierarchy_prefilter_restrict_narrows_candidate_set(bundle):
    hp = HierarchyPrefilter(bundle)
    scores = {c.id: 1.0 for c in bundle}
    restricted = hp.boost("stripe payments", dict(scores), restrict=True)
    assert set(restricted) == {"stripe.create_charge", "stripe.create_refund"}


def test_hierarchy_score_categories_prefers_more_specific_overlap(bundle):
    hp = HierarchyPrefilter(bundle)
    scores = hp.score_categories("github issues label")
    assert scores["github/issues"] > 0
    assert "slack/chat" not in scores


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------


def test_graph_expand_surfaces_alternatives(bundle):
    seed = SearchHit(
        id="github.find_issues_by_label",
        title="Find GitHub Issues by Label",
        description="d",
        score=1.0,
        via="rank",
    )
    expanded = graph_expand(bundle, [seed])
    ids = {h.id for h in expanded}
    assert {"github.list_issues", "github.search_issues"} <= ids
    for h in expanded:
        assert h.via == "alternatives"
        assert h.source_id == "github.find_issues_by_label"
        assert 0 < h.score < seed.score  # damped, never amplified


def test_graph_expand_covers_all_edge_types(bundle):
    seed = SearchHit(
        id="github.create_issue", title="t", description="d", score=1.0, via="rank"
    )
    expanded = graph_expand(bundle, [seed], max_per_hit=10)
    via_types = {h.via for h in expanded}
    # create_issue has alternatives, composes_with, and prerequisites edges
    assert via_types == {"alternatives", "composes_with", "prerequisites"}


def test_graph_expand_never_duplicates_seed_hits(bundle):
    seeds = [
        SearchHit(id="github.list_issues", title="t", description="d", score=1.0, via="rank"),
        SearchHit(
            id="github.find_issues_by_label", title="t", description="d", score=0.9, via="rank"
        ),
    ]
    expanded = graph_expand(bundle, seeds)
    expanded_ids = [h.id for h in expanded]
    seed_ids = {h.id for h in seeds}
    assert not (set(expanded_ids) & seed_ids)
    # no id appears twice even though both seeds point at each other/search_issues
    assert len(expanded_ids) == len(set(expanded_ids))


def test_graph_expand_respects_max_per_hit(bundle):
    seed = SearchHit(
        id="github.create_issue", title="t", description="d", score=1.0, via="rank"
    )
    expanded = graph_expand(bundle, [seed], max_per_hit=1)
    assert len(expanded) == 1


def test_graph_expand_no_edges_returns_empty(bundle):
    # slack.send_message has only an "alternatives" edge to update_message;
    # a concept with genuinely no matching bundle member expands to nothing.
    b = Bundle()
    b.add(
        OKTConcept(
            id="solo.tool",
            title="Solo",
            description="d",
            input_schema={"type": "object"},
            interface=Interface.FUNCTION,
        )
    )
    seed = SearchHit(id="solo.tool", title="t", description="d", score=1.0, via="rank")
    assert graph_expand(b, [seed]) == []


# ---------------------------------------------------------------------------
# Retriever protocol conformance
# ---------------------------------------------------------------------------


def test_flat_retriever_implements_protocol():
    assert isinstance(FlatBM25Retriever(), Retriever)


def test_graph_aware_retriever_implements_protocol():
    assert isinstance(GraphAwareRetriever(), Retriever)


def test_flat_retriever_requires_index_before_search():
    r = FlatBM25Retriever()
    with pytest.raises(RuntimeError):
        r.search("anything")


def test_graph_aware_retriever_requires_index_before_search():
    r = GraphAwareRetriever()
    with pytest.raises(RuntimeError):
        r.search("anything")


def test_flat_retriever_returns_search_hits_no_schema(bundle):
    r = FlatBM25Retriever()
    r.index(bundle)
    hits = r.search("refund a payment", k=3)
    assert hits
    for h in hits:
        assert isinstance(h, SearchHit)
        assert h.via == "rank"
        ref = h.to_ref()
        assert set(ref) == {"id", "title", "description"}


def test_flat_retriever_never_uses_hierarchy_or_graph_via(bundle):
    r = FlatBM25Retriever()
    r.index(bundle)
    for query, _ in COLLISION_QUERIES:
        for hit in r.search(query, k=11):
            assert hit.via == "rank"
            assert hit.source_id is None


def test_flat_retriever_respects_k(bundle):
    r = FlatBM25Retriever()
    r.index(bundle)
    hits = r.search("issue repository label read write", k=2)
    assert len(hits) <= 2


def test_graph_aware_retriever_returns_search_hits_no_schema(bundle):
    r = GraphAwareRetriever()
    r.index(bundle)
    hits = r.search("refund a payment", k=3)
    assert hits
    for h in hits:
        assert isinstance(h, SearchHit)
        ref = h.to_ref()
        assert set(ref) == {"id", "title", "description"}


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_graph_aware_retriever_modes_run(bundle, mode):
    r = GraphAwareRetriever(mode=mode, graph_expand=False, hierarchy_prefilter=False)
    r.index(bundle)
    hits = r.search("post a message to a channel", k=3)
    assert hits
    assert hits[0].id == "slack.send_message"


def test_graph_aware_retriever_unknown_mode_raises(bundle):
    r = GraphAwareRetriever()
    r.index(bundle)
    with pytest.raises(ValueError):
        r.search("hello", mode="bogus")


@pytest.mark.parametrize("fusion", ["weighted_sum", "rrf"])
def test_graph_aware_retriever_fusion_methods_run(bundle, fusion):
    r = GraphAwareRetriever(mode="hybrid", fusion=fusion)
    r.index(bundle)
    hits = r.search("refund a payment", k=3)
    assert hits[0].id == "stripe.create_refund"


def test_graph_aware_retriever_graph_expand_toggle(bundle):
    r_on = GraphAwareRetriever(graph_expand=True)
    r_off = GraphAwareRetriever(graph_expand=False)
    r_on.index(bundle)
    r_off.index(bundle)
    # A query whose top hit (create_issue) has graph neighbors that are NOT
    # themselves strong lexical matches, so expansion is the only way they
    # surface. k must leave room past the seed for a sibling to compete in.
    query = "create a new issue"
    hits_on = r_on.search(query, k=3)
    hits_off = r_off.search(query, k=3)
    assert any(h.via != "rank" for h in hits_on)
    assert all(h.via == "rank" for h in hits_off)
    # base ranking is unaffected by whether expansion runs afterward: the
    # top-ranked hit is a seed either way (siblings are damped below it)
    assert hits_on[0].id == hits_off[0].id
    # k is the total budget: expansion competes WITHIN it, never past it
    assert len(hits_on) <= 3


def test_graph_aware_retriever_hierarchy_toggle_changes_scores(bundle):
    r_on = GraphAwareRetriever(hierarchy_prefilter=True, graph_expand=False)
    r_off = GraphAwareRetriever(hierarchy_prefilter=False, graph_expand=False)
    r_on.index(bundle)
    r_off.index(bundle)
    query = "github issues"
    on_scores = {h.id: h.score for h in r_on.search(query, k=11)}
    off_scores = {h.id: h.score for h in r_off.search(query, k=11)}
    # hierarchy boost should raise github/issues members' scores vs no-boost
    assert on_scores["github.create_issue"] > off_scores["github.create_issue"]


def test_graph_aware_retriever_per_call_opts_override_constructor(bundle):
    r = GraphAwareRetriever(graph_expand=True, hierarchy_prefilter=True)
    r.index(bundle)
    hits = r.search(
        "get all issues tagged with a particular label",
        k=1,
        graph_expand=False,
        hierarchy_prefilter=False,
    )
    assert all(h.via == "rank" for h in hits)


def test_graph_aware_retriever_degrades_gracefully_with_no_hierarchy():
    b = Bundle()
    b.add(
        OKTConcept(
            id="a.b",
            title="Thing",
            description="does a thing",
            input_schema={"type": "object"},
            interface=Interface.FUNCTION,
        )
    )
    r = GraphAwareRetriever()
    r.index(b)  # bundle.hierarchy is empty
    hits = r.search("thing", k=5)
    assert hits and hits[0].id == "a.b"


# ---------------------------------------------------------------------------
# The value proposition: graph-aware disambiguates near-duplicates at least
# as well as flat BM25, and strictly better accuracy@1 across the collision
# set on this fixture.
# ---------------------------------------------------------------------------


def test_graph_aware_ranks_collision_tool_at_least_as_high_as_flat(bundle):
    flat = FlatBM25Retriever()
    flat.index(bundle)
    graph = GraphAwareRetriever()
    graph.index(bundle)

    query, expected_id = "get all issues tagged with a particular label", "github.find_issues_by_label"

    flat_hits = flat.search(query, k=len(bundle))
    graph_hits = [h for h in graph.search(query, k=len(bundle)) if h.via == "rank"]

    flat_ids = [h.id for h in flat_hits]
    graph_ids = [h.id for h in graph_hits]

    flat_rank = flat_ids.index(expected_id) if expected_id in flat_ids else len(flat_ids)
    graph_rank = graph_ids.index(expected_id) if expected_id in graph_ids else len(graph_ids)

    # graph-aware must not rank the correct tool WORSE than flat BM25 does
    assert graph_rank <= flat_rank
    # and on this fixture it actually nails top-1
    assert graph_hits[0].id == expected_id


def test_graph_expansion_surfaces_graph_neighbors_within_budget(bundle):
    graph = GraphAwareRetriever()
    graph.index(bundle)
    # k is the caller's TOTAL budget (an agent asking for k refs must not get
    # 3x that dumped into context). Graph siblings compete for the tail slots
    # of the window on their damped scores rather than extending past k.
    hits = graph.search("create a new issue", k=3)
    assert len(hits) <= 3

    rank_ids = {h.id for h in hits if h.via == "rank"}
    expanded = [h for h in hits if h.via != "rank"]

    # the seed ranks on its own merits...
    assert "github.create_issue" in rank_ids
    # ...and a graph neighbor that is NOT a strong lexical match on its own
    # (get_repo, a prerequisite of create_issue) is pulled into the window via
    # the graph — the whole point of expansion.
    assert expanded, "expected at least one graph-expanded neighbor in the window"
    for h in expanded:
        assert h.via in ("alternatives", "composes_with", "prerequisites")
        # every expanded hit points back at the seed it was reached from...
        assert h.source_id in rank_ids
        # ...and is damped strictly below that seed, so it can never outrank it
        seed = next(x for x in hits if x.id == h.source_id)
        assert h.score < seed.score


def test_graph_aware_accuracy_at_1_beats_or_matches_flat_on_collision_set(bundle):
    flat = FlatBM25Retriever()
    flat.index(bundle)
    graph = GraphAwareRetriever()
    graph.index(bundle)

    flat_acc = _accuracy_at_1(flat, COLLISION_QUERIES)
    graph_acc = _accuracy_at_1(graph, COLLISION_QUERIES)

    assert graph_acc >= flat_acc
    # on this fixture the graph-aware retriever gets every collision query right
    assert graph_acc == 1.0


@pytest.mark.parametrize("query,expected_id", COLLISION_QUERIES)
def test_graph_aware_top1_per_collision_query(bundle, query, expected_id):
    graph = GraphAwareRetriever()
    graph.index(bundle)
    hits = graph.search(query, k=3)
    assert hits[0].id == expected_id
