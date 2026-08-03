"""Example 2 — a LangGraph agent over a HETEROGENEOUS catalog (Python
functions + sub-agents + a search endpoint), with and without the OKTS wrapper.

Where Example 1 is about *scale* (one source type, ~148 tools, big token win),
this one is about *heterogeneity*: the catalog mixes three source kinds, and
the point is that the agent talks to all of them through the **same three
meta-tools**. ``call_tool`` here dispatches to REAL backends — a plain function
runs, a sub-agent callable runs, a search endpoint runs — via
``FunctionDispatcher`` / ``AgentDispatcher`` / ``SearchDispatcher``.

Honest note on tokens: with only a handful of tools the 3 meta-tool schemas
cost about as much as binding the tools directly — the ~85% reduction is a
*large-corpus* property (see Example 1). The win demonstrated here is uniform
dispatch + progressive disclosure across source types, not raw token savings.

Run it::

    python examples/langgraph_mixed_sources.py
"""

from __future__ import annotations

import json
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

from okts.adapters.agent import agent_to_okt  # noqa: E402
from okts.adapters.function import function_from_callable  # noqa: E402
from okts.adapters.search import search_endpoints_to_okt  # noqa: E402
from okts.core.model import Bundle, Interface  # noqa: E402
from okts.enrich.autolink import autolink  # noqa: E402
from okts.enrich.enricher import OfflineEnricher, enrich_bundle  # noqa: E402
from okts.serve.dispatch import (  # noqa: E402
    AgentDispatcher,
    DispatcherRegistry,
    FunctionDispatcher,
    SearchDispatcher,
)
from okts.serve.service import OKTSService  # noqa: E402


# --- real backends: plain functions ---------------------------------------

def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


# --- real backends: sub-agents (callable that takes an args dict) ----------

def run_summarizer(args: dict) -> dict:
    doc = str(args.get("document", ""))
    return {"summary": f"[summary of {len(doc)} chars]"}


def run_researcher(args: dict) -> dict:
    return {"findings": [f"finding about {args.get('topic')!r}"]}


# --- real backend: search endpoint -----------------------------------------

def run_web_search(args: dict) -> dict:
    return {"results": [f"hit for {args.get('q')!r}"]}


SUMMARIZER_CARD = {
    "name": "summarizer",
    "description": "Summarize a long document into a short paragraph.",
    "input_schema": {
        "type": "object",
        "required": ["document"],
        "properties": {"document": {"type": "string"}},
    },
}
RESEARCHER_CARD = {
    "name": "researcher",
    "description": "Research a topic on the web and return key findings.",
    "input_schema": {
        "type": "object",
        "required": ["topic"],
        "properties": {"topic": {"type": "string"}},
    },
}
WEB_SEARCH_SPEC = {
    "name": "web_search",
    "description": "Search the public web for a query string.",
    "url": "https://search.example.com/v1/search",
    "query_params": [{"name": "q", "type": "string", "required": True}],
}


def build_service() -> OKTSService:
    """Adapt three source kinds into one bundle, wire a real dispatcher per
    interface, and serve them behind the same 3 meta-tools."""
    concepts = [
        function_from_callable(add, target="add"),
        function_from_callable(multiply, target="multiply"),
        function_from_callable(word_count, target="word_count"),
        agent_to_okt(SUMMARIZER_CARD, target="summarizer"),
        agent_to_okt(RESEARCHER_CARD, target="researcher"),
        *search_endpoints_to_okt([WEB_SEARCH_SPEC], target="web_search"),
    ]
    flat = Bundle()
    for c in concepts:
        flat.add(c)
    bundle = autolink(enrich_bundle(flat, OfflineEnricher()))

    from okts.index.retriever import GraphAwareRetriever

    registry = DispatcherRegistry()
    registry.register(
        Interface.FUNCTION,
        FunctionDispatcher(targets={"add": add, "multiply": multiply, "word_count": word_count}),
    )
    registry.register(
        Interface.AGENT,
        AgentDispatcher(targets={"summarizer": run_summarizer, "researcher": run_researcher}),
    )
    registry.register(Interface.SEARCH, SearchDispatcher(targets={"web_search": run_web_search}))

    return OKTSService(bundle, GraphAwareRetriever(), registry)


def call_through_wrapper(service: OKTSService, task: str, k: int = 3):
    """Drive one task through the 3 meta-tools and return the AgentRun."""
    return run_agent(make_llm("meta", task=task, k=k), okts_langchain_tools(service), task)


def run(task: str = "add two numbers together", expected_id: str = "add", k: int = 3):
    service = build_service()
    bundle = service.bundle

    with_run = call_through_wrapper(service, task, k=k)

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
    return service, cmp, without_run, with_run


def main() -> None:
    service, cmp, without_run, with_run = run()
    print_comparison(
        "Example 2 — functions + sub-agents + search (heterogeneous)",
        cmp,
        without_run,
        with_run,
    )
    print(f"  (small catalog: reduction here is {cmp.reduction_pct:.1f}% — the token "
          "win is a large-corpus property, see Example 1)")

    # the real payoff here: the SAME 3 meta-tools dispatch to different source
    # kinds — a function AND a sub-agent both execute for real.
    fn_run = with_run  # already ran the function task
    agent_run = call_through_wrapper(service, "summarize a long document")
    print("\n-- one wrapper, many source kinds (real dispatch) --")
    print(f"  function  task -> {fn_run.tool_calls} -> {fn_run.final_text}")
    print(f"  sub-agent task -> {agent_run.tool_calls} -> {agent_run.final_text}")

    assert with_run.called("search_tools") and with_run.called("call_tool")
    assert "2" in fn_run.final_text, "add(1,1) should really execute to 2"
    assert "summary" in agent_run.final_text, "sub-agent should really dispatch"
    assert without_run.called(_sanitize("add"))
    print("\nOK — functions and sub-agents both dispatched through the same 3 meta-tools.")


if __name__ == "__main__":
    main()
