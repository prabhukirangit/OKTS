"""Example 1 — a LangGraph ReAct agent over ~20 MCP servers (~148 tools),
with and without the OKTS wrapper.

The catalog is the project's benchmark corpus (``eval/corpus/*.tools.json``:
github, gitlab, filesystem, playwright, postgres, kubernetes, aws, ...) adapted
through the ordinary offline pipeline into ~148 OKT tools.

Two agents, same task:

- **WITHOUT OKTS** — every one of the ~148 tools is bound to the agent
  directly (``raw_langchain_tools``). The agent carries all ~148 schemas in
  context every turn, and a real LLM has to pick the right one out of the pile.
- **WITH OKTS** — the agent is bound to just the 3 meta-tools
  (``okts_langchain_tools``). It ``search_tools`` -> ``load_tool`` ->
  ``call_tool``; OKTS's graph-aware retriever does the selection, and only the
  chosen tool's schema is ever loaded.

Run it::

    python examples/langgraph_mcp_corpus.py

Everything is offline/deterministic (see ``examples/_common.py``): the token
numbers are real (``okts.eval.tokens``); the agent is driven by a scripted
model so it runs with no API key. Point it at a real LLM by setting
``OKTS_EXAMPLE_REAL_LLM=1`` — the wiring is identical.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    _sanitize,
    _synthesize_args,
    compare_tokens,
    make_llm,
    okts_langchain_tools,
    print_comparison,
    raw_langchain_tools,
    run_agent,
)

from okts.eval.corpus import build_corpus_bundle  # noqa: E402
from okts.serve.dispatch import DispatcherRegistry  # noqa: E402
from okts.serve.service import OKTSService  # noqa: E402

TASK = "open a new issue in a github repository"
EXPECTED_ID = "github.create_issue"


def build_service() -> OKTSService:
    """The whole OKTS setup: corpus bundle + real graph-aware retriever + a
    (mock, offline) dispatcher so ``call_tool`` succeeds without live servers."""
    from okts.index.retriever import GraphAwareRetriever

    bundle = build_corpus_bundle()
    return OKTSService(bundle, GraphAwareRetriever(), DispatcherRegistry.mock_all())


def run(task: str = TASK, expected_id: str = EXPECTED_ID, k: int = 3):
    service = build_service()
    bundle = service.bundle

    # --- WITH OKTS: bind 3 meta-tools, let the agent search -> load -> call ---
    with_run = run_agent(
        make_llm("meta", task=task, k=k),
        okts_langchain_tools(service),
        task,
    )

    # --- WITHOUT OKTS: bind all ~148 tools; agent calls the target directly ---
    # (offline the scripted model calls the known target; a real LLM would have
    # to select it out of the full ~148-schema list.)
    concept = bundle.get(expected_id)
    without_run = run_agent(
        make_llm(
            "direct",
            direct_target=_sanitize(expected_id),
            direct_args=_synthesize_args(concept.input_schema),
        ),
        raw_langchain_tools(bundle),
        task,
    )

    cmp = compare_tokens(bundle, service, task, k=k)
    return cmp, without_run, with_run


def main() -> None:
    cmp, without_run, with_run = run()
    print_comparison("Example 1 — 20 MCP servers (~148 tools)", cmp, without_run, with_run)

    # sanity checks (also asserted in examples/test_examples.py)
    assert with_run.called("search_tools"), "OKTS agent should search"
    assert with_run.called("call_tool"), "OKTS agent should reach call_tool"
    assert without_run.called(_sanitize(EXPECTED_ID)), "raw agent should hit the target tool"
    assert cmp.reduction_pct > 80, f"expected >80% reduction, got {cmp.reduction_pct:.1f}%"
    print("\nOK — both agents reached the tool; OKTS cut per-query tool tokens by "
          f"{cmp.reduction_pct:.1f}%.")


if __name__ == "__main__":
    main()
