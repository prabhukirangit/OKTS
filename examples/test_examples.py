"""Tests for the LangGraph + OKTS examples — the "run tests on both" part.

Not part of the default suite (``pyproject.toml`` sets ``testpaths = ["tests"]``);
run them explicitly::

    pytest examples/test_examples.py

They need the example extras (``pip install -e ".[examples]"``); the whole
module is skipped if LangGraph/LangChain aren't installed, and everything runs
offline (scripted model, no API key).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def test_mcp_corpus_with_and_without_wrapper():
    import langgraph_mcp_corpus as ex

    cmp, without_run, with_run = ex.run()

    # WITH OKTS: the agent went search -> load -> call over the 3 meta-tools
    assert with_run.called("search_tools")
    assert with_run.called("load_tool")
    assert with_run.called("call_tool")

    # WITHOUT OKTS: the agent hit the (correct) upstream tool directly
    assert without_run.called(ex._sanitize(ex.EXPECTED_ID))

    # the catalog really is the ~148-tool corpus, and the wrapper is a big win
    assert cmp.n_tools > 100
    assert cmp.reduction_pct > 80


def test_mixed_sources_dispatch_through_one_wrapper():
    import langgraph_mixed_sources as ex

    service, cmp, without_run, with_run = ex.run()

    # function task really executed through call_tool: add(1, 1) -> 2
    assert with_run.called("search_tools") and with_run.called("call_tool")
    assert "2" in with_run.final_text

    # the SAME 3 meta-tools also dispatch to a sub-agent for real
    agent_run = ex.call_through_wrapper(service, "summarize a long document")
    assert agent_run.called("call_tool")
    assert "summary" in agent_run.final_text

    # without the wrapper, the direct call still works
    assert without_run.called(ex._sanitize("add"))
