"""Token cost estimation for the OKTS retrieval pipeline.

Compares two ways an agent can learn about tools:

- **raw-tools baseline** (:func:`raw_tools_cost`): every concept's
  ``input_schema`` + ``description`` serialized up front, as if the agent had
  been handed all N tool definitions -- the naive "give the agent every tool"
  pattern this project replaces.
- **OKTS per-query cost** (:func:`okts_query_cost`): the constant cost of the
  three meta-tool schemas (``search_tools`` / ``load_tool`` / ``call_tool``),
  plus the ``k`` lightweight search refs returned by phase 1, plus the ONE
  ``input_schema`` loaded in phase 2 for the tool actually chosen.

This is the pair of numbers that demonstrates the ~85% token reduction target
in ``CLAUDE.md``.

``estimate_tokens`` prefers ``tiktoken`` (cl100k_base) when importable, and
falls back to a deterministic chars/4 heuristic otherwise, so the estimator
works fully offline with zero required dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from okts.core.model import Bundle, OKTConcept
from okts.core.protocols import SearchHit

try:  # pragma: no cover - exercised only when tiktoken is installed
    import tiktoken

    _ENC: Any = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - exercised only when tiktoken is absent
    _ENC = None


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Uses ``tiktoken``'s ``cl100k_base`` encoding when importable; otherwise
    falls back to a deterministic ``ceil(len(text) / 4)`` heuristic, the
    standard rule-of-thumb for English text under BPE tokenizers. Either path
    is deterministic given the same input.
    """
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, -(-len(text) // 4))  # ceil division, zero deps


def concept_schema_tokens(concept: OKTConcept) -> int:
    """Tokens for one concept's ``input_schema`` + ``description``, serialized
    the way a raw function-calling API (OpenAI-style / MCP ``tools/list``)
    would send it to an agent up front."""
    payload = {
        "name": concept.id,
        "description": concept.description,
        "parameters": concept.input_schema,
    }
    return estimate_tokens(json.dumps(payload, sort_keys=True))


def raw_tools_cost(bundle: Bundle) -> int:
    """The naive baseline: tokens if ALL N concepts' schemas were loaded up
    front, as most function-calling agents do today (invariant this project
    exists to fix -- see CLAUDE.md "The problem it solves")."""
    return sum(concept_schema_tokens(c) for c in bundle)


# ---------------------------------------------------------------------------
# The three meta-tool schemas the agent sees ONCE per session/turn, regardless
# of corpus size. Computed from realistic OpenAI-style function definitions
# (not a bare magic number) so the constant stays grounded and deterministic.
# ---------------------------------------------------------------------------

_META_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_tools",
        "description": (
            "Search the tool corpus for tools relevant to a natural-language "
            "query. Returns lightweight refs (id, title, description) ranked "
            "by relevance -- never full schemas."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the task",
                },
                "k": {
                    "type": "integer",
                    "description": "Max number of results to return",
                    "default": 5,
                },
            },
        },
    },
    {
        "name": "load_tool",
        "description": (
            "Load the full calling contract (structured input_schema + "
            "side_effects) for one tool id returned by search_tools. Only "
            "the chosen tool's schema is loaded into context."
        ),
        "parameters": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "The tool id to load"},
            },
        },
    },
    {
        "name": "call_tool",
        "description": (
            "Validate args against the loaded input_schema and dispatch the "
            "call to the real upstream source (MCP server, HTTP API, "
            "function, agent, or search endpoint). Credentials are applied "
            "inside OKTS and never enter agent context."
        ),
        "parameters": {
            "type": "object",
            "required": ["id", "args"],
            "properties": {
                "id": {"type": "string", "description": "The tool id to call"},
                "args": {
                    "type": "object",
                    "description": "Arguments matching the loaded input_schema",
                },
            },
        },
    },
]

# The constant per-query fixed cost of the three meta-tool schemas.
META_TOOL_SCHEMAS_TOKENS: int = sum(
    estimate_tokens(json.dumps(schema, sort_keys=True)) for schema in _META_TOOL_SCHEMAS
)


def search_ref_tokens(hit: SearchHit) -> int:
    """Tokens for one lightweight phase-1 search ref (id/title/description,
    no schema -- see ``SearchHit.to_ref``)."""
    return estimate_tokens(json.dumps(hit.to_ref(), sort_keys=True))


def okts_query_cost(
    bundle: Bundle, hits: list[SearchHit], loaded_id: str | None = None
) -> int:
    """Tokens the agent actually sees for one query under OKTS:

    the constant meta-tool schemas (:data:`META_TOOL_SCHEMAS_TOKENS`) + the
    ``k`` lightweight search refs returned by ``search_tools`` + the ONE
    ``input_schema`` loaded via ``load_tool`` for the concept the agent picks
    (defaults to the top-ranked hit, i.e. the common case).
    """
    cost = META_TOOL_SCHEMAS_TOKENS
    cost += sum(search_ref_tokens(h) for h in hits)
    target_id = loaded_id if loaded_id is not None else (hits[0].id if hits else None)
    if target_id is not None:
        concept = bundle.get(target_id)
        if concept is not None:
            cost += concept_schema_tokens(concept)
    return cost


@dataclass
class TokenComparison:
    """Aggregate token comparison for a batch of queries.

    ``raw_tools_tokens`` is paid once (the whole corpus loaded up front);
    ``okts_tokens_total`` / ``okts_tokens_avg`` are paid per query. Reduction
    is measured as OKTS's average per-query cost against the one-time
    raw-tools cost -- this is the number CLAUDE.md's "~85% reduction" target
    refers to.
    """

    raw_tools_tokens: int
    okts_tokens_total: int
    num_queries: int

    @property
    def okts_tokens_avg(self) -> float:
        return self.okts_tokens_total / self.num_queries if self.num_queries else 0.0

    @property
    def reduction_pct(self) -> float:
        if self.raw_tools_tokens == 0:
            return 0.0
        return 100.0 * (1 - self.okts_tokens_avg / self.raw_tools_tokens)
