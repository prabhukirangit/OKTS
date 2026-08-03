# OKTS × LangGraph examples

Two runnable examples showing how to **stitch OKTS into a LangGraph ReAct
agent**, and what changes when you do — measured with/without the wrapper.

| file | catalog | what it shows |
|------|---------|---------------|
| [`langgraph_mcp_corpus.py`](langgraph_mcp_corpus.py) | ~20 MCP servers, **~148 tools** (the benchmark corpus) | the token win at scale (~95% fewer tool-schema tokens per query) |
| [`langgraph_mixed_sources.py`](langgraph_mixed_sources.py) | functions + sub-agents + a search endpoint | one wrapper, **real dispatch** to three different source kinds |
| [`lazy_targets.py`](lazy_targets.py) | — | connect-on-first-`call_tool` pattern so proxying 20+ servers doesn't open every connection at startup |
| [`context_hygiene.py`](context_hygiene.py) | — | a scrubber that evicts a spent `load_tool` schema from history once its `call_tool` runs (using OKTS's schema marker) |

Run the hardening examples directly, or test them with `pytest examples/test_hardening.py`:

```bash
python examples/lazy_targets.py
python examples/context_hygiene.py
```

## The stitch (this is the whole integration)

An existing LangGraph agent binds N tools. To add OKTS you build one service and
bind **3 tools instead of N** — nothing else about the graph changes:

```python
from okts.serve.service import OKTSService
from langgraph.prebuilt import create_react_agent
from _common import okts_langchain_tools   # the wrapper in this folder

service = OKTSService(bundle, retriever, dispatcher)   # your OKT bundle
tools = okts_langchain_tools(service)                  # -> [search_tools, load_tool, call_tool]

agent = create_react_agent(model, tools)               # 3 tools, not N
```

The agent's loop becomes: `search_tools(task)` → `load_tool(id)` →
`call_tool(id, args)`. OKTS's retriever does the tool selection and only the
chosen tool's schema is ever loaded into context.

### The one gotcha

LangChain reserves `args` as a tool parameter name, so binding OKTS's
`call_tool(id, args)` verbatim fails with `unexpected keyword argument
'v__args'`. The wrapper renames the parameter to `arguments` and forwards it
unchanged — see `okts_langchain_tools` in [`_common.py`](_common.py).

## Running

```bash
pip install -e ".[examples]"          # langgraph + langchain-core

python examples/langgraph_mcp_corpus.py
python examples/langgraph_mixed_sources.py

pytest examples/test_examples.py      # asserts both paths, with & without wrapper
```

Everything runs **offline and deterministically** — no API key. The agent is
driven by a small scripted chat model (`ScriptedLLM` in `_common.py`) so the
examples double as tests; token numbers come from `okts.eval.tokens` (the same
estimator the benchmark uses). Tool *selection* on the OKTS side is done by the
real graph-aware retriever, so that part is genuine, not scripted.

To run against a real LLM instead, set `OKTS_EXAMPLE_REAL_LLM=1` (with
`langchain-openai` + `OPENAI_API_KEY`, or edit `make_llm` for your provider) —
the wiring is identical; only the model changes.

## What the numbers say

Example 1 (148 tools) prints roughly:

```
WITHOUT OKTS (bind all 148): 9041 tokens (every turn)
WITH OKTS, per query:         488 tokens   -> ~94.6% reduction
```

Example 2 (6 tools) prints a *negative* reduction — with a tiny catalog the 3
meta-tool schemas cost about as much as binding the tools directly. That's the
honest large-corpus caveat: the token win scales with catalog size (Example 1).
Example 2's point is different — the same three meta-tools dispatch to a
function **and** a sub-agent **and** a search endpoint, all executing for real.
