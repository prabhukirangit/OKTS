"""Phase-0 core tests: model, (de)serialization round-trip, validator, config."""

import textwrap

import pytest

from okts.core.model import Bundle, Interface, OKTConcept, SideEffects
from okts.core.serialize import concept_from_markdown, concept_to_markdown
from okts.core.validator import validate_concept, validate_bundle


def test_fixture_bundle_loads(bundle):
    assert len(bundle) == 11
    assert bundle.get("github.create_issue") is not None
    assert bundle.hierarchy["github/issues"]


def test_fixture_bundle_is_conformant(bundle):
    problems = validate_bundle(bundle, check_edges=True)
    assert problems == [], problems


def test_roundtrip_preserves_fields(bundle):
    original = bundle.get("github.create_issue")
    text = concept_to_markdown(original)
    reparsed = concept_from_markdown(text)
    assert reparsed.id == original.id
    assert reparsed.interface == Interface.MCP
    assert reparsed.side_effects == SideEffects.WRITE
    assert reparsed.input_schema == original.input_schema
    assert reparsed.alternatives == original.alternatives
    assert "422" in reparsed.body  # body preserved


def test_unknown_frontmatter_survives_roundtrip():
    md = textwrap.dedent(
        """\
        ---
        type: tool
        id: x.y
        title: Y
        description: does y
        input_schema: {type: object}
        interface: function
        custom_key: keep-me
        ---
        body
        """
    )
    c = concept_from_markdown(md)
    assert c.extra["custom_key"] == "keep-me"
    assert "custom_key: keep-me" in concept_to_markdown(c)


def test_validator_flags_missing_required():
    c = OKTConcept(id="", title="", input_schema={})
    problems = validate_concept(c)
    assert any("id" in p for p in problems)
    assert any("title" in p for p in problems)
    assert any("input_schema" in p for p in problems)


def test_validator_rejects_unstructured_schema():
    c = OKTConcept(
        id="a.b", title="B", description="d",
        input_schema={"note": "just prose, no type/properties"},
        interface=Interface.FUNCTION,
    )
    problems = validate_concept(c)
    assert any("structured" in p for p in problems)


def test_validator_accepts_resource_schema():
    c = OKTConcept(
        id="a.b", title="B", description="d",
        input_schema={"resource": "./schema.json"},
        interface=Interface.FUNCTION,
    )
    assert validate_concept(c) == []


def test_bundle_rejects_duplicate_ids():
    b = Bundle()
    b.add(OKTConcept(id="a.b", title="B", input_schema={"type": "object"}))
    with pytest.raises(ValueError):
        b.add(OKTConcept(id="a.b", title="B2", input_schema={"type": "object"}))


def test_dangling_edge_detected():
    b = Bundle()
    b.add(OKTConcept(
        id="a.b", title="B", description="d",
        input_schema={"type": "object"}, interface=Interface.FUNCTION,
        alternatives=["./does_not_exist.md"],
    ))
    problems = validate_bundle(b, check_edges=True)
    assert any("dangling" in p for p in problems)


def test_edge_resolution_by_path_and_id(bundle):
    # create_issue.alternatives references ./github.update_issue.md
    assert bundle.resolve_edge("./github.update_issue.md") == "github.update_issue"
    assert bundle.resolve_edge("github.update_issue") == "github.update_issue"


def test_match_ref_has_no_schema(bundle):
    ref = bundle.get("github.create_issue").match_ref()
    assert set(ref) == {"id", "title", "description"}
    assert "input_schema" not in ref
