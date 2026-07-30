"""The OKT concept and bundle data model.

An **OKT concept** is one callable tool described as one markdown file with YAML
frontmatter. The frontmatter fields split into three consumption groups that map
onto the three runtime phases:

- **match** (ranked in phase 1, never sent at call time): ``description``, ``tags``
- **call** (loaded on demand in phase 2): ``input_schema``, ``output_schema``
- **route** (used in phase 3 dispatch): ``interface``, ``target``, ``auth``,
  ``side_effects``, ``cost``

Plus identity (``type``, ``id``, ``title``), graph edges (``alternatives``,
``prerequisites``, ``composes_with``) used for phase-1 graph expansion, and the
OKF standard fields (``timestamp``, ``version``).

The markdown **body** is retrieval text (synonyms, when-to/when-not, gotchas). It
is embedded and ranked; it is NEVER sent at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Union

# Required-minimum frontmatter — adapters must ALWAYS emit these six.
# (invariant #5 in CLAUDE.md). Everything else degrades gracefully.
REQUIRED_MINIMUM: tuple[str, ...] = (
    "type",
    "id",
    "title",
    "description",
    "input_schema",
    "interface",
)


class Interface(str, Enum):
    """Phase-3 dispatch interface — how ``call_tool`` reaches the real source."""

    MCP = "mcp"
    FUNCTION = "function"
    HTTP = "http"
    AGENT = "agent"
    SEARCH = "search"


class SideEffects(str, Enum):
    """Effect class of invoking a tool. Default to WRITE when unknown (safe)."""

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


# input_schema may be an inline JSON Schema dict OR a resource pointer
# ``{"resource": "./schema.json"}``. We keep it as a raw dict either way; the
# validator distinguishes the two forms.
JsonSchema = dict[str, Any]


@dataclass
class Cost:
    """Optional cost hints used only as rank tie-breaks."""

    latency_ms: Optional[int] = None
    dollars: Optional[float] = None

    def to_frontmatter(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.latency_ms is not None:
            out["latency_ms"] = self.latency_ms
        if self.dollars is not None:
            out["dollars"] = self.dollars
        return out

    @classmethod
    def from_frontmatter(cls, raw: Optional[dict[str, Any]]) -> Optional["Cost"]:
        if not raw:
            return None
        return cls(latency_ms=raw.get("latency_ms"), dollars=raw.get("dollars"))


@dataclass
class OKTConcept:
    """One callable tool, described in the OKT format.

    Construct directly, or via ``concept_from_markdown``. Serialize via
    ``concept_to_markdown``. Validate via ``validate_concept``.
    """

    # --- identity (required) ---
    id: str
    title: str
    # --- match half (ranked; description required, tags recommended) ---
    description: str = ""
    tags: list[str] = field(default_factory=list)
    # --- call half (input_schema required; output_schema optional) ---
    input_schema: JsonSchema = field(default_factory=dict)
    output_schema: Optional[JsonSchema] = None
    # --- route half (interface required) ---
    interface: Interface = Interface.FUNCTION
    target: Optional[str] = None
    auth: Optional[str] = None
    side_effects: SideEffects = SideEffects.WRITE
    cost: Optional[Cost] = None
    # --- graph edges (phase-1 expansion) ---
    alternatives: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    composes_with: list[str] = field(default_factory=list)
    # --- OKF standard ---
    type: str = "tool"
    timestamp: Optional[str] = None
    version: Optional[str] = None
    # --- retrieval text (the markdown body; ranked, never sent at call time) ---
    body: str = ""
    # --- provenance / extension: unknown frontmatter keys are preserved here so
    # bundles round-trip losslessly and stay portable to other OKF consumers. ---
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- lightweight views used by each phase ----

    def match_ref(self) -> dict[str, Any]:
        """Phase-1 result shape — lightweight ref, NO schema."""
        return {"id": self.id, "title": self.title, "description": self.description}

    def call_view(self) -> dict[str, Any]:
        """Phase-2 payload — the structured schema + effect class."""
        view: dict[str, Any] = {
            "id": self.id,
            "input_schema": self.input_schema,
            "side_effects": self.side_effects.value,
        }
        if self.output_schema is not None:
            view["output_schema"] = self.output_schema
        return view

    def match_text(self) -> str:
        """Concatenated text that phase-1 ranks over: description + tags + body."""
        return "\n".join(
            part for part in (self.description, " ".join(self.tags), self.body) if part
        )

    def neighbors(self) -> list[str]:
        """All outbound graph edges (used by graph-expansion)."""
        return [*self.alternatives, *self.prerequisites, *self.composes_with]


@dataclass
class Bundle:
    """A collection of OKT concepts cross-linked into a graph.

    ``hierarchy`` is the ``index.md`` category tree (path -> list of child paths
    or concept ids) used by the retrieval hierarchy prefilter. Edge references in
    concepts may be relative file paths (``./update_issue.md``) or bare ids; the
    bundle resolves both.
    """

    concepts: dict[str, OKTConcept] = field(default_factory=dict)
    hierarchy: dict[str, list[str]] = field(default_factory=dict)

    def add(self, concept: OKTConcept) -> None:
        if concept.id in self.concepts:
            raise ValueError(f"duplicate concept id: {concept.id!r}")
        self.concepts[concept.id] = concept

    def get(self, concept_id: str) -> Optional[OKTConcept]:
        return self.concepts.get(concept_id)

    def __len__(self) -> int:
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts.values())

    def resolve_edge(self, ref: str) -> Optional[str]:
        """Resolve a graph-edge reference (path or id) to a concept id in bundle."""
        if ref in self.concepts:
            return ref
        # strip ./ and .md, take the stem, try to match by id suffix
        stem = ref.rsplit("/", 1)[-1]
        if stem.endswith(".md"):
            stem = stem[:-3]
        if stem in self.concepts:
            return stem
        for cid in self.concepts:
            if cid.rsplit(".", 1)[-1] == stem:
                return cid
        return None
