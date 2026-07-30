"""Layer 1½ enrichment tests.

``OfflineEnricher`` is the one that must run in CI: deterministic, offline,
no keys. ``LLMEnricher`` is a guarded scaffold — importing/constructing it
must never require network/keys, and it must degrade cleanly (raise, or fall
back) when unconfigured.
"""

from __future__ import annotations

import pytest

from okts.core.model import Bundle, Interface, OKTConcept, SideEffects
from okts.core.validator import validate_concept

from okts.enrich.enricher import LLMEnricher, OfflineEnricher, enrich_bundle


# ---------------------------------------------------------------------------
# OfflineEnricher
# ---------------------------------------------------------------------------


def test_offline_enricher_lengthens_body(bundle):
    concept = bundle.get("github.create_issue")
    enriched = OfflineEnricher().enrich(concept, bundle)
    assert len(enriched.body) > len(concept.body)


def test_offline_enricher_deterministic(bundle):
    concept = bundle.get("github.create_issue")
    e1 = OfflineEnricher().enrich(concept, bundle)
    e2 = OfflineEnricher().enrich(concept, bundle)
    assert e1.body == e2.body


def test_offline_enricher_does_not_mutate_input(bundle):
    concept = bundle.get("github.create_issue")
    original_body = concept.body
    OfflineEnricher().enrich(concept, bundle)
    assert concept.body == original_body


def test_offline_enricher_preserves_original_body_text(bundle):
    concept = bundle.get("github.create_issue")
    enriched = OfflineEnricher().enrich(concept, bundle)
    assert concept.body.strip() in enriched.body


def test_offline_enricher_mentions_resolved_alternative(bundle):
    concept = bundle.get("github.create_issue")  # alternatives: [./github.update_issue.md]
    enriched = OfflineEnricher().enrich(concept, bundle)
    assert "github.update_issue" in enriched.body


def test_offline_enricher_mentions_prerequisites_and_composes(bundle):
    concept = bundle.get("github.create_issue")  # prerequisites/composes_with set in fixture
    enriched = OfflineEnricher().enrich(concept, bundle)
    for prereq in concept.prerequisites:
        assert prereq in enriched.body
    for comp in concept.composes_with:
        assert comp in enriched.body


def test_offline_enricher_gotcha_reflects_side_effects():
    destructive = OKTConcept(
        id="x.delete_thing",
        title="Delete Thing",
        description="Delete a thing permanently.",
        input_schema={"type": "object", "properties": {}},
        interface=Interface.FUNCTION,
        side_effects=SideEffects.DESTRUCTIVE,
    )
    read_only = OKTConcept(
        id="x.get_thing",
        title="Get Thing",
        description="Fetch a thing.",
        input_schema={"type": "object", "properties": {}},
        interface=Interface.FUNCTION,
        side_effects=SideEffects.READ,
    )
    d_body = OfflineEnricher().enrich(destructive).body.lower()
    r_body = OfflineEnricher().enrich(read_only).body.lower()
    assert "irreversible" in d_body or "destructive" in d_body
    assert "read-only" in r_body


def test_offline_enricher_works_without_bundle():
    c = OKTConcept(
        id="x.y",
        title="Y",
        description="Do the y thing.",
        input_schema={"type": "object", "properties": {}},
        interface=Interface.FUNCTION,
        alternatives=["some.other"],
    )
    enriched = OfflineEnricher().enrich(c, bundle=None)
    assert len(enriched.body) > 0
    assert "some.other" in enriched.body


def test_offline_enricher_output_still_conformant(bundle):
    for concept in bundle:
        enriched = OfflineEnricher().enrich(concept, bundle)
        assert validate_concept(enriched) == []


def test_offline_enricher_synonyms_include_tags_and_id_words():
    c = OKTConcept(
        id="slack.send_message",
        title="Send Message",
        description="Post a message to a channel.",
        tags=["slack", "chat", "notify"],
        input_schema={"type": "object", "properties": {}},
        interface=Interface.FUNCTION,
    )
    enriched = OfflineEnricher().enrich(c)
    assert "send" in enriched.body.lower()
    assert "notify" in enriched.body.lower()


# ---------------------------------------------------------------------------
# enrich_bundle
# ---------------------------------------------------------------------------


def test_enrich_bundle_expands_every_concept(bundle):
    out = enrich_bundle(bundle, OfflineEnricher())
    assert len(out) == len(bundle)
    assert set(out.concepts) == set(bundle.concepts)
    for cid, enriched in out.concepts.items():
        original = bundle.get(cid)
        assert len(enriched.body) >= len(original.body)
        assert validate_concept(enriched) == []


def test_enrich_bundle_preserves_hierarchy(bundle):
    out = enrich_bundle(bundle, OfflineEnricher())
    assert out.hierarchy == bundle.hierarchy


def test_enrich_bundle_does_not_mutate_input(bundle):
    original_bodies = {c.id: c.body for c in bundle}
    enrich_bundle(bundle, OfflineEnricher())
    for c in bundle:
        assert c.body == original_bodies[c.id]


# ---------------------------------------------------------------------------
# LLMEnricher (guarded scaffold)
# ---------------------------------------------------------------------------


def test_llm_enricher_imports_and_constructs_without_sdk_or_keys():
    # module-level import already happened at top of file; constructing with
    # no args must not touch network or require any provider SDK/env var.
    enricher = LLMEnricher()
    assert enricher.call_fn is None


def test_llm_enricher_unconfigured_raises():
    c = OKTConcept(
        id="x.y", title="Y", description="Do y.",
        input_schema={"type": "object", "properties": {}}, interface=Interface.FUNCTION,
    )
    with pytest.raises(RuntimeError):
        LLMEnricher().enrich(c)


def test_llm_enricher_falls_back_to_offline_when_unconfigured():
    c = OKTConcept(
        id="x.y", title="Y", description="Do y.",
        input_schema={"type": "object", "properties": {}}, interface=Interface.FUNCTION,
    )
    enricher = LLMEnricher(fallback=OfflineEnricher())
    enriched = enricher.enrich(c)
    assert len(enriched.body) > 0
    assert validate_concept(enriched) == []


def test_llm_enricher_falls_back_when_call_fn_raises():
    def boom(prompt: str) -> str:
        raise RuntimeError("simulated LLM failure")

    c = OKTConcept(
        id="x.y", title="Y", description="Do y.",
        input_schema={"type": "object", "properties": {}}, interface=Interface.FUNCTION,
    )
    enricher = LLMEnricher(call_fn=boom, fallback=OfflineEnricher())
    enriched = enricher.enrich(c)
    assert len(enriched.body) > 0


def test_llm_enricher_uses_call_fn_when_configured():
    def fake_llm(prompt: str) -> str:
        assert "Tool id: x.y" in prompt
        return "Synonyms: foo, bar. Gotcha: careful with foo."

    c = OKTConcept(
        id="x.y", title="Y", description="Do y.",
        input_schema={"type": "object", "properties": {}}, interface=Interface.FUNCTION,
    )
    enriched = LLMEnricher(call_fn=fake_llm).enrich(c)
    assert "foo, bar" in enriched.body


def test_llm_enricher_no_fallback_reraises_on_call_fn_error():
    def boom(prompt: str) -> str:
        raise ValueError("no fallback configured")

    c = OKTConcept(
        id="x.y", title="Y", description="Do y.",
        input_schema={"type": "object", "properties": {}}, interface=Interface.FUNCTION,
    )
    with pytest.raises(ValueError):
        LLMEnricher(call_fn=boom).enrich(c)
