"""Offline test for the company-analysis OKTS example.

No API key needed: it spawns the real ``mcp_company_db.py`` stdio server as a
subprocess, wraps it (plus the native functions) into an OKT bundle, and drives
the three meta-tools — proving the OKTS wrapper unifies a LIVE async MCP source
and sync Python functions behind one interface and dispatches both for real.

Run: ``pytest examples/test_company_analysis.py`` (skips without the deps).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("anyio")
pytest.importorskip("mcp")
pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def test_okts_wraps_live_mcp_and_functions_end_to_end():
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    import company_analysis_okts as ex

    params = StdioServerParameters(command=sys.executable, args=[str(ex.MCP_SERVER)])

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                service = await ex.build_service(session)

                ids = {c.id for c in service.bundle}
                # one catalog, three sources unified: 1 live MCP tool + 2 functions
                assert ids == {
                    "company_db.get_internal_metrics",
                    "calculate_growth_projection",
                    "search_market_trends",
                }

                # phase 1: retrieval finds the right tool for the task
                refs = service.search_tools("confidential internal metrics and ARR", k=3)
                assert refs[0]["id"] == "company_db.get_internal_metrics"

                # phase 3: live MCP dispatch returns the real confidential data
                mcp_result = await service.acall_tool(
                    "company_db.get_internal_metrics", {"company_name": "Acme Corp"}
                )
                # phase 3: in-process function dispatch really computes
                fn_result = await service.acall_tool(
                    "calculate_growth_projection",
                    {"current_val": 45.0, "growth_rate": 0.12, "years": 3},
                )
                return mcp_result, fn_result

    mcp_result, fn_result = anyio.run(scenario)
    assert "45M" in mcp_result and "Acme Corp" in mcp_result
    assert "Projected Value" in fn_result


def test_async_meta_tools_dispatch_through_langchain_tool():
    """The async LangChain meta-tool wrapper (used by the LangGraph agent) drives
    a real dispatch via ``ainvoke`` — no LLM, no key."""
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from _common import okts_langchain_tools_async
    import company_analysis_okts as ex

    params = StdioServerParameters(command=sys.executable, args=[str(ex.MCP_SERVER)])

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                service = await ex.build_service(session)
                tools = {t.name: t for t in okts_langchain_tools_async(service)}
                assert set(tools) == {"search_tools", "load_tool", "call_tool"}
                return await tools["call_tool"].ainvoke(
                    {"id": "company_db.get_internal_metrics", "arguments": {"company_name": "Beta Co"}}
                )

    result = anyio.run(scenario)
    assert "12M" in result  # Beta Co's ARR, dispatched live through the wrapper
