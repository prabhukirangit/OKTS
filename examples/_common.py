"""Shared helpers for the LangGraph + OKTS examples.

Everything here is **offline and deterministic** so the examples double as
tests (no API key, no network). Two pieces do the heavy lifting:

- :class:`ScriptedLLM` — a tiny ``BaseChatModel`` that emits tool calls from a
  fixed script instead of a real LLM, so ``create_react_agent`` runs end to end
  with zero credentials. In ``mode="meta"`` it drives the OKTS loop
  (``search_tools`` -> ``load_tool`` -> ``call_tool``); the *tool selection*
  inside ``search_tools`` is done by OKTS's real retriever, so that part is a
  faithful demo — only the orchestration is scripted. In ``mode="direct"`` it
  calls one named tool directly (the tool a real LLM would have had to pick out
  of the full list), because offline we have no model to choose among N raw
  schemas. Swap in a real chat model (see ``make_llm``) to make both sides fully
  LLM-driven.
- the tool wrappers — :func:`okts_langchain_tools` (the 3 meta-tools, the OKTS
  "stitch") and :func:`raw_langchain_tools` (every upstream tool bound directly,
  the "before" picture).

Token accounting reuses ``okts.eval.tokens`` so the numbers are the same ones
the project's benchmark reports.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from okts.core.model import Bundle
from okts.eval.tokens import (
    META_TOOL_SCHEMAS_TOKENS,
    okts_query_cost,
    raw_tools_cost,
)
from okts.serve.service import OKTSService


# ---------------------------------------------------------------------------
# 1. The OKTS "stitch": wrap the three meta-tools as LangChain tools
# ---------------------------------------------------------------------------


def okts_langchain_tools(service: OKTSService) -> list[StructuredTool]:
    """The whole integration: turn an :class:`OKTSService` into the 3 LangChain
    tools you bind to your agent *instead of* N upstream tools.

    This is all "adding OKTS to a LangGraph agent" takes — build the service
    (bundle + retriever + dispatcher), wrap it here, and bind these three.
    """

    def search_tools(query: str, k: int = 5) -> str:
        """Search the tool catalog for tools relevant to a task. Returns
        lightweight refs (id, title, description) — never full schemas."""
        return json.dumps(service.search_tools(query, k=k))

    def load_tool(id: str) -> str:
        """Load the structured input_schema (+ side_effects) for one tool id
        returned by search_tools."""
        return json.dumps(service.load_tool(id))

    def call_tool(id: str, arguments: dict) -> str:
        """Validate arguments against the loaded schema and dispatch the call.

        NOTE: OKTS's own ``call_tool`` names this parameter ``args``; it is
        renamed to ``arguments`` here because LangChain reserves ``args`` on a
        tool signature and raises ``unexpected keyword argument 'v__args'``
        otherwise. This rename is the one real gotcha when binding OKTS to
        LangChain — the value is forwarded unchanged to ``service.call_tool``.
        """
        return json.dumps(service.call_tool(id, arguments), default=str)

    return [
        StructuredTool.from_function(search_tools),
        StructuredTool.from_function(load_tool),
        StructuredTool.from_function(call_tool),
    ]


def okts_langchain_tools_async(service: OKTSService) -> list[StructuredTool]:
    """Async variant of :func:`okts_langchain_tools`.

    ``call_tool`` awaits ``service.acall_tool``, so it can dispatch to an
    **async** target — e.g. a live MCP ``ClientSession`` (``invocation: async``)
    — from inside a running event loop (a LangGraph ``astream`` run). Use this
    set when any served tool dispatches to a coroutine; the sync set would raise
    from within the loop.
    """

    async def search_tools(query: str, k: int = 5) -> str:
        """Search the tool catalog for tools relevant to a task. Returns
        lightweight refs (id, title, description) — never full schemas."""
        return json.dumps(service.search_tools(query, k=k))

    async def load_tool(id: str) -> str:
        """Load the structured input_schema (+ side_effects) for one tool id."""
        return json.dumps(service.load_tool(id))

    async def call_tool(id: str, arguments: dict) -> str:
        """Validate arguments against the loaded schema and dispatch the call.
        (Param is ``arguments`` not ``args`` — see okts_langchain_tools.)"""
        return json.dumps(await service.acall_tool(id, arguments), default=str)

    return [
        StructuredTool.from_function(coroutine=search_tools, name="search_tools",
                                     description=search_tools.__doc__),
        StructuredTool.from_function(coroutine=load_tool, name="load_tool",
                                     description=load_tool.__doc__),
        StructuredTool.from_function(coroutine=call_tool, name="call_tool",
                                     description=call_tool.__doc__),
    ]


# ---------------------------------------------------------------------------
# 2. The "before" picture: bind every upstream tool directly
# ---------------------------------------------------------------------------


def _sanitize(concept_id: str) -> str:
    """Function-calling APIs disallow ``.``/``-`` in tool names; map an OKT id
    (``github.create_issue``) to a legal name (``github__create_issue``)."""
    return concept_id.replace(".", "__").replace("-", "_")


def raw_langchain_tools(bundle: Bundle) -> list[StructuredTool]:
    """Bind EVERY concept in the bundle as its own LangChain tool — the naive
    "give the agent all N tools" setup OKTS replaces. Each tool is a no-op stub
    that echoes its call (the point here is the *schema* the agent must carry,
    not the execution)."""
    tools: list[StructuredTool] = []
    for concept in bundle:
        def _run(_id: str = concept.id, **kwargs: Any) -> str:
            return json.dumps({"ok": _id, "args": kwargs}, default=str)

        args_schema = (
            concept.input_schema
            if isinstance(concept.input_schema, dict) and concept.input_schema.get("properties")
            else None
        )
        tools.append(
            StructuredTool.from_function(
                func=_run,
                name=_sanitize(concept.id),
                description=concept.description,
                args_schema=args_schema,
            )
        )
    return tools


# ---------------------------------------------------------------------------
# 3. Deterministic, no-API-key chat model to drive the agent offline
# ---------------------------------------------------------------------------


def _placeholder(prop: dict) -> Any:
    return {
        "string": "example",
        "integer": 1,
        "number": 1,
        "boolean": True,
        "array": [],
        "object": {},
    }.get((prop or {}).get("type"), "example")


def _synthesize_args(schema: Any) -> dict:
    """Fill a schema's required fields with type-appropriate placeholders so a
    scripted ``call_tool`` passes OKTS's arg validation and actually dispatches."""
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties") or {}
    return {key: _placeholder(props.get(key, {})) for key in (schema.get("required") or [])}


class ScriptedLLM(BaseChatModel):
    """A no-LLM chat model that emits tool calls from the conversation so far.

    ``mode="meta"`` drives OKTS's search->load->call loop; ``mode="direct"``
    calls ``direct_target`` once with ``direct_args``. See the module docstring.
    """

    mode: str = "meta"
    task: str = ""
    k: int = 3
    direct_target: str = ""
    direct_args: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "okts-scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedLLM":
        # create_react_agent binds the tool list here; the script doesn't need
        # it (it already knows the names), so just accept and return self.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]

        if not isinstance(last, ToolMessage):
            # first turn: kick off the loop
            if self.mode == "direct":
                return _tool_call(self.direct_target, self.direct_args or {})
            return _tool_call("search_tools", {"query": self.task, "k": self.k})

        name, content = last.name, str(last.content)
        if name == "search_tools":
            first_id = json.loads(content)[0]["id"]
            return _tool_call("load_tool", {"id": first_id})
        if name == "load_tool":
            view = json.loads(content)
            return _tool_call(
                "call_tool",
                {"id": self._picked_id(messages), "arguments": _synthesize_args(view.get("input_schema"))},
            )
        # a call_tool result (or a direct tool result): we're done
        return _final(f"done: {content[:120]}")

    @staticmethod
    def _picked_id(messages: list[BaseMessage]) -> str:
        for m in messages:
            if isinstance(m, ToolMessage) and m.name == "search_tools":
                return json.loads(str(m.content))[0]["id"]
        return ""


def _tool_call(name: str, args: dict) -> ChatResult:
    msg = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])
    return ChatResult(generations=[ChatGeneration(message=msg)])


def _final(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def make_llm(mode: str, **kwargs: Any) -> BaseChatModel:
    """Return a chat model to drive an example.

    Default is the offline :class:`ScriptedLLM`. Set ``OKTS_EXAMPLE_REAL_LLM=1``
    (and have ``langchain-openai`` + ``OPENAI_API_KEY``, or edit this function
    for your provider) to run the examples against a real model instead — the
    wiring is identical; only the model changes.
    """
    if os.environ.get("OKTS_EXAMPLE_REAL_LLM") == "1":  # pragma: no cover - opt-in
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("OKTS_EXAMPLE_MODEL", "gpt-4o-mini"), temperature=0)
    return ScriptedLLM(mode=mode, **kwargs)


# ---------------------------------------------------------------------------
# 4. Run an agent + collect which tools it called
# ---------------------------------------------------------------------------


@dataclass
class AgentRun:
    tool_calls: list[str]      # tool names, in order
    final_text: str
    n_messages: int

    def called(self, name: str) -> bool:
        return name in self.tool_calls


def run_agent(llm: BaseChatModel, tools: list[Any], task: str) -> AgentRun:
    """Build a LangGraph ReAct agent over ``tools`` and run one task."""
    agent = create_react_agent(llm, tools)
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    msgs = result["messages"]
    calls: list[str] = []
    final = ""
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            calls.append(tc["name"])
        if isinstance(m, AIMessage) and m.content:
            final = str(m.content)
    return AgentRun(tool_calls=calls, final_text=final, n_messages=len(msgs))


# ---------------------------------------------------------------------------
# 5. Token comparison (reuses okts.eval.tokens — the benchmark's own numbers)
# ---------------------------------------------------------------------------


@dataclass
class TokenComparison:
    n_tools: int
    raw_tools_tokens: int          # all N schemas bound up front (per turn)
    okts_meta_tokens: int          # the 3 meta-tool schemas (fixed)
    okts_per_query_tokens: int     # meta + k refs + 1 loaded schema
    query: str

    @property
    def reduction_pct(self) -> float:
        if not self.raw_tools_tokens:
            return 0.0
        return 100.0 * (1 - self.okts_per_query_tokens / self.raw_tools_tokens)


def compare_tokens(bundle: Bundle, service: OKTSService, query: str, k: int = 5) -> TokenComparison:
    hits = service.retriever.search(query, k=k)
    loaded = hits[0].id if hits else None
    return TokenComparison(
        n_tools=sum(1 for _ in bundle),
        raw_tools_tokens=raw_tools_cost(bundle),
        okts_meta_tokens=META_TOOL_SCHEMAS_TOKENS,
        okts_per_query_tokens=okts_query_cost(bundle, hits, loaded_id=loaded),
        query=query,
    )


def print_comparison(title: str, cmp: TokenComparison, without: AgentRun, with_: AgentRun) -> None:
    print(f"\n=== {title} ===")
    print(f"upstream tools in catalog: {cmp.n_tools}")
    print("\n-- tool-schema tokens the agent must carry --")
    print(f"  WITHOUT OKTS (bind all {cmp.n_tools}): {cmp.raw_tools_tokens:>7} tokens (every turn)")
    print(f"  WITH OKTS (bind 3 meta-tools):    {cmp.okts_meta_tokens:>7} tokens fixed")
    print(f"  WITH OKTS, per query (meta+refs+1 schema): {cmp.okts_per_query_tokens:>7} tokens")
    print(f"  reduction (per-query vs bind-all): {cmp.reduction_pct:>5.1f}%")
    print("\n-- agent runs --")
    print(f"  WITHOUT OKTS: called {without.tool_calls}")
    print(f"  WITH OKTS:    called {with_.tool_calls}")
