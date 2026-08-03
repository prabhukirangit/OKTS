"""Example — the BASIC agent (no OKTS): a LangGraph ReAct agent with every tool
bound directly.

This is the conventional setup OKTS replaces. A GPT-4o ReAct agent is handed
three tools at once — a live MCP tool (``get_internal_metrics`` from
``mcp_company_db.py``) plus two native Python tools — and every tool's schema
sits in the model's context on every turn. Its OKTS counterpart is
``company_analysis_okts.py``: same graph, same task, but the agent sees only the
three meta-tools and pulls each real schema on demand.

Requirements (real LLM + live MCP, so NOT offline):

    pip install -e ".[examples]" langchain-openai langchain-mcp-adapters
    export OPENAI_API_KEY=...

    python examples/company_analysis_basic.py

Only ``build_agent_graph`` and the native tools import at module load (they need
just langgraph + langchain-core); the OpenAI / MCP-adapter imports live inside
``main`` so ``company_analysis_okts.py`` can reuse the identical graph without
those heavier deps.
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# ==========================================
# 1. Native Python Tools & Mock API Tools
# ==========================================


@tool
def calculate_growth_projection(current_val: float, growth_rate: float, years: int) -> str:
    """Calculates future value based on compound growth rate.

    Args:
        current_val: Current revenue/ARR value
        growth_rate: Annual growth percentage (e.g., 0.15 for 15%)
        years: Projection horizon in years
    """
    future_val = current_val * ((1 + growth_rate) ** years)
    return f"Projected Value in {years} years at {growth_rate * 100}% growth: ${future_val:,.2f}"


@tool
def search_market_trends(query: str) -> str:
    """Searches web news and market reports for sector trends."""
    # Mocking external API output (e.g. Tavily / Serper)
    if "acquisition" in query.lower() or "market" in query.lower():
        return "Market Report 2026: Consolidation is accelerating in mid-cap AI software providers."
    return f"Search result for '{query}': High demand for enterprise intelligence automation."


NATIVE_TOOLS = [calculate_growth_projection, search_market_trends]


# ==========================================
# 2. Build LangGraph ReAct Workflow
# ==========================================


def build_agent_graph(llm, tools):
    """Constructs the classic ReAct agent state graph.

    Shared verbatim by the OKTS example — the ONLY thing that changes between the
    two is the ``tools`` list handed in here (N real tools vs. 3 meta-tools).
    """
    model_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        """Reasoning node: decide whether to invoke a tool or respond."""
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        """Conditional router: inspect the last message for tool-call requests."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


ANALYSIS_PROMPT = (
    "Run a market analysis on 'Acme Corp'. "
    "1. Fetch their internal metrics and current ARR. "
    "2. Project what their ARR will be in 3 years assuming a 12% growth rate. "
    "3. Search market trends for relevant context and summarize your findings."
)


def print_stream_event(latest_msg) -> None:
    if latest_msg.type == "ai" and latest_msg.tool_calls:
        for tc in latest_msg.tool_calls:
            print(f"[Agent Decision] call {tc['name']} with args: {tc['args']}")
    elif latest_msg.type == "tool":
        print(f"[Tool Output] ({latest_msg.name}): {latest_msg.content}\n")
    elif latest_msg.type == "ai" and not latest_msg.tool_calls:
        print("[Final Response]:")
        print(latest_msg.content)


# ==========================================
# 3. Main Execution Loop
# ==========================================


async def main():
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    server_script = os.path.join(os.path.dirname(__file__), "mcp_company_db.py")

    async with MultiServerMCPClient(
        {
            "company_db": {
                "command": "python",
                "args": [server_script],
                "transport": "stdio",
            }
        }
    ) as mcp_client:
        mcp_tools = await mcp_client.get_tools()
        all_tools = [*NATIVE_TOOLS, *mcp_tools]

        print(f"Loaded {len(all_tools)} tool(s): {[t.name for t in all_tools]}\n")
        app = build_agent_graph(llm, all_tools)

        print(f"User Request:\n{ANALYSIS_PROMPT}\n" + "-" * 50)
        inputs = {"messages": [HumanMessage(content=ANALYSIS_PROMPT)]}
        async for chunk in app.astream(inputs, stream_mode="values"):
            print_stream_event(chunk["messages"][-1])


if __name__ == "__main__":
    asyncio.run(main())
