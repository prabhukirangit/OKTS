"""Example — the SAME agent, WITH OKTS.

Counterpart to ``company_analysis_basic.py``. Identical LangGraph ReAct graph
(imported from that file) and identical task, but instead of binding the live
MCP tool + two native tools directly, they are ingested into ONE OKT bundle and
served behind the three meta-tools. The agent now ``search_tools`` ->
``load_tool`` -> ``call_tool`` and only ever carries three schemas, no matter how
many upstream tools exist.

What this shows that the ~148-tool corpus example doesn't: a HETEROGENEOUS,
partly-LIVE catalog (an async MCP session + sync Python functions) unified
behind one interface, dispatched for real — the MCP tool over a live
``ClientSession`` (``invocation: async``), the functions in-process.

Requirements: the ``mcp`` extra is needed for the live server; the LLM is
pluggable. With a real model::

    pip install -e ".[examples,serve]" langchain-openai
    export OPENAI_API_KEY=... OKTS_EXAMPLE_REAL_LLM=1
    python examples/company_analysis_okts.py

Without a key it runs a scripted single-task loop (offline) so you can see the
search->load->call flow and the dispatch working end to end. The full wiring is
also covered offline by ``examples/test_company_analysis.py``.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import compare_tokens, okts_langchain_tools_async  # noqa: E402
from company_analysis_basic import NATIVE_TOOLS, build_agent_graph  # noqa: E402

from okts.adapters.function import function_from_callable  # noqa: E402
from okts.adapters.mcp import mcp_tools_to_okt  # noqa: E402
from okts.core.model import Bundle, Interface  # noqa: E402
from okts.enrich.autolink import autolink  # noqa: E402
from okts.enrich.enricher import OfflineEnricher, enrich_bundle  # noqa: E402
from okts.serve.dispatch import DispatcherRegistry, FunctionDispatcher, McpDispatcher  # noqa: E402
from okts.serve.service import OKTSService  # noqa: E402

MCP_SERVER = pathlib.Path(__file__).resolve().parent / "mcp_company_db.py"
SERVER_NAME = "company_db"


class _TextMcpTarget:
    """MCP dispatch target that returns the tool result's plain text.

    ``McpDispatcher`` calls ``target.call_tool(name, args)``; a live
    ``ClientSession`` returns a rich ``CallToolResult``. We unwrap it to text so
    the value handed back to the agent is clean prose, not a serialized object.
    """

    def __init__(self, session) -> None:
        self.session = session

    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments)
        return "".join(getattr(part, "text", "") for part in result.content)


async def build_service(session) -> OKTSService:
    """Ingest the live MCP server's tools + the native functions into one OKT
    bundle and wire a real dispatcher per source kind."""
    listed = await session.list_tools()
    raw = [t.model_dump(by_alias=True, exclude_none=True) for t in listed.tools]

    concepts = list(mcp_tools_to_okt(raw, server=SERVER_NAME))
    for lc_tool in NATIVE_TOOLS:
        fn = lc_tool.func  # the plain callable behind the @tool wrapper
        concepts.append(function_from_callable(fn, id=fn.__name__, target=fn.__name__))

    flat = Bundle()
    for concept in concepts:
        flat.add(concept)
    bundle = autolink(enrich_bundle(flat, OfflineEnricher()))

    from okts.index.retriever import GraphAwareRetriever

    registry = DispatcherRegistry()
    registry.register(Interface.MCP, McpDispatcher(targets={SERVER_NAME: _TextMcpTarget(session)}))
    registry.register(
        Interface.FUNCTION,
        FunctionDispatcher(targets={t.func.__name__: t.func for t in NATIVE_TOOLS}),
    )
    return OKTSService(bundle, GraphAwareRetriever(), registry)


def _print_token_comparison(service: OKTSService) -> None:
    cmp = compare_tokens(service.bundle, service, "internal metrics and ARR for Acme Corp", k=3)
    print("\n-- tool-schema tokens the agent carries --")
    print(f"  WITHOUT OKTS (bind all {cmp.n_tools}): {cmp.raw_tools_tokens:>6} tokens/turn")
    print(f"  WITH OKTS (3 meta-tools):          {cmp.okts_meta_tokens:>6} tokens fixed")
    print(f"  WITH OKTS, per query:              {cmp.okts_per_query_tokens:>6} tokens")
    print(f"  (small catalog -> reduction {cmp.reduction_pct:.0f}%; the token win is a "
          "large-corpus property, see langgraph_mcp_corpus.py)")


async def main() -> None:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            service = await build_service(session)
            tools = okts_langchain_tools_async(service)

            print(f"OKTS serves {sum(1 for _ in service.bundle)} upstream tools behind "
                  f"{len(tools)} meta-tools: {[t.name for t in tools]}")
            _print_token_comparison(service)

            if os.environ.get("OKTS_EXAMPLE_REAL_LLM") == "1":
                from langchain_core.messages import HumanMessage
                from langchain_openai import ChatOpenAI
                from company_analysis_basic import ANALYSIS_PROMPT, print_stream_event

                app = build_agent_graph(ChatOpenAI(model="gpt-4o", temperature=0), tools)
                print("\nUser Request:\n" + ANALYSIS_PROMPT + "\n" + "-" * 50)
                inputs = {"messages": [HumanMessage(content=ANALYSIS_PROMPT)]}
                async for chunk in app.astream(inputs, stream_mode="values"):
                    print_stream_event(chunk["messages"][-1])
            else:
                # offline (no key): walk the three meta-tools by hand over the
                # REAL service — search, load, then dispatch to the live MCP tool.
                print("\n(offline walk-through — set OKTS_EXAMPLE_REAL_LLM=1 for the full "
                      "GPT-4o multi-step analysis)")
                refs = service.search_tools("confidential internal metrics and ARR for a company", k=3)
                print(f"  search_tools -> {[r['id'] for r in refs]}")
                chosen = refs[0]["id"]
                view = service.load_tool(chosen)
                print(f"  load_tool({chosen!r}) -> input_schema keys: {list(view['input_schema'].get('properties', {}))}")
                result = await service.acall_tool(chosen, {"company_name": "Acme Corp"})
                print(f"  call_tool -> {result!r}")


if __name__ == "__main__":
    asyncio.run(main())
