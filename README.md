# OKTS — Open Knowledge Tool Search

**Give your agent 300 tools. It only ever sees 3.**

OKTS is a source-agnostic wrapper that ingests tools from anywhere — MCP servers, plain function schemas, sub-agents, OpenAPI/HTTP endpoints, search APIs — describes each one as a portable [OKT](#the-okt-format) markdown file, indexes them with graph-aware retrieval, and exposes a stable three-tool interface to any agent. The agent searches for the tool it needs, loads that one schema on demand, and calls it. Nothing else touches the context window.

> **OKT** (Open Knowledge Tools) is the *format* — an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog) profile for describing a callable tool.
> **OKTS** (Open Knowledge Tool Search) is the *runtime* that produces, ranks, and serves it.

---

## The problem

Load 50 tools into an agent and it gets *worse*, not better. Tool schemas eat 30–50% of the context window before the task even starts. Attention dilutes across definitions the agent won't use. Near-duplicate tools (`search_issues`, `list_issues`, `find_issues_by_label`) blur together and the model picks wrong or hallucinates parameters. The system prompt gets starved for room.

The fix is **progressive disclosure**: stop loading tool schemas until the agent actually needs one. OKTS does that — and does it over a portable, git-versioned descriptor rather than an ephemeral in-memory index.

## How it works

The agent sees exactly three tools, forever:

```
search_tools(query, k=5)   → [{id, title, description}]      # rank, don't load
load_tool(id)              → { input_schema, side_effects }  # load one schema on demand
call_tool(id, args)        → result                          # validate + dispatch
```

Three phases, three field groups per tool:

1. **search** — ranks on the human-readable description + body + tags, applies a category-hierarchy prefilter, and graph-expands to surface alternatives. Returns lightweight refs, **not** schemas.
2. **load** — injects the one structured `input_schema` the agent chose.
3. **call** — validates args against that schema and dispatches to the real source. Credentials stay inside OKTS and never enter the agent's context.

## Install

```bash
npm install okts        # Node / TypeScript
pip install okts        # Python
```

Point it at your sources with a config:

```yaml
# tools.config.yaml
sources:
  - interface: mcp
    servers: [github-mcp, slack-mcp, linear-mcp, postgres-mcp]
  - interface: http
    openapi: ./specs/stripe.yaml
  - interface: function
    module: ./my_local_tools.py
retrieval: { mode: hybrid, graph_expand: true }
```

Then drop it in front of any agent. As an MCP server, replace all your raw server entries with one:

```json
{ "mcpServers": { "okts": { "command": "okts", "args": ["--config", "tools.config.yaml"] } } }
```

Any MCP client — Claude Desktop, Claude Code, Cursor, your own loop — now sees 3 tools instead of 300, with zero agent code changes. Not on MCP? Use OKTS as a library (register the three methods as native functions) or as an HTTP sidecar. Same contract.

## How OKTS is different

Progressive disclosure for tools is a crowded space. Most solutions solve one slice of it. OKTS differs on four axes at once:

| | Progressive disclosure | Source-agnostic | Agent/client-agnostic | Portable, editable descriptors | Graph + hierarchy-aware retrieval |
|---|:---:|:---:|:---:|:---:|:---:|
| **OKTS** | ✅ | ✅ MCP · functions · agents · HTTP · search | ✅ | ✅ OKT markdown, git-versioned | ✅ hybrid + graph + hierarchy |
| Native tool-search *(e.g. Anthropic Tool Search)* | ✅ | ➖ registered tools only | ❌ vendor-tied | ❌ | ❌ flat BM25 / regex |
| Schema-compression proxy *(e.g. mcp-compressor)* | ❌ compresses, still loads all | ❌ MCP-only | ✅ | ❌ | — |
| MCP aggregator gateway *(e.g. MetaMCP, ContextForge)* | ➖ some | ❌ MCP-only | ✅ | ❌ | ➖ usually flat |
| Code-execution mode *(e.g. Code Mode)* | ✅ | ➖ partial | ✅ | ❌ | — |

**What the alternatives do well** — and why OKTS still differs:

- **Native tool-search** (Anthropic's Tool Search Tool) is the zero-setup option if you live entirely on one vendor's API, and it already delivers large context savings. But it's tied to that client and ranks flat over independent tool definitions. OKTS is client-agnostic and ranks over a cross-linked graph.
- **Compression proxies** are the least disruptive drop-in — they shrink descriptions without changing how tools are called. But they still carry *every* tool in context; there's a floor. OKTS loads schemas on demand, so cost scales with tools *used*, not tools *available*.
- **Aggregator gateways** are excellent at federation, governance, and RBAC across many MCP servers. But they're MCP-bound and their tool-finding is usually flat keyword search. OKTS spans non-MCP sources and treats retrieval quality as the core problem, not an afterthought.
- **Code-execution mode** achieves the highest raw token reduction by exposing tools as a typed SDK. But it demands a code sandbox and a model strong at writing code. OKTS keeps normal tool-calling ergonomics.

The two things unique to OKTS:

1. **Portable descriptors.** Every tool is a plain OKT markdown file with YAML frontmatter — human-readable, hand-editable, diff-able, and version-controlled in git. Your tool corpus survives the runtime that created it. Everyone else's index is ephemeral or proprietary.
2. **Graph-aware retrieval.** Hybrid ranking (BM25 + dense) is standard IR — that part isn't the innovation. The innovation is doing it over OKT's cross-links and category hierarchy: when a query hits a tool, OKTS surfaces its `alternatives` and `composes_with` neighbors so the agent disambiguates near-duplicates instead of guessing. That directly attacks the tool-collision failure mode.

## The OKT format

One markdown file per tool. Frontmatter splits into **match** (ranked), **call** (loaded on demand), and **route** (dispatch) groups:

```yaml
---
type: tool
id: github.create_issue
title: Create GitHub Issue
description: Open a new issue in a GitHub repository.
tags: [github, issues, create, write]
input_schema:
  type: object
  required: [repo, title]
  properties:
    repo:   { type: string, description: "owner/name" }
    title:  { type: string }
    labels: { type: array, items: { type: string } }
interface: mcp
target: github-mcp
side_effects: write
alternatives:  [./update_issue.md, ./list_issues.md]
composes_with: [./add_labels.md]
---

Use this to **create** a brand-new issue. Do not use it to comment on or edit
an existing one — that's `update_issue`. Gotcha: labels must already exist in
the repo or the call fails.
```

The body is retrieval text (synonyms, when-to/when-not, gotchas) — indexed, never sent at call time. `input_schema` is the authoritative calling contract — structured, loaded only in phase 2. Bundles are validated against the OKF conformance spec, so they stay portable.

## Architecture

```
Sources        MCP · functions · sub-agents · HTTP/OpenAPI · search APIs
   │
1. Adapters    normalize each source into an OKT concept   (source → okt)
2. Descriptor  the OKT bundle: one file per tool, cross-linked into a graph
3. Retrieval   hybrid rank + hierarchy prefilter + graph expansion
4. Serving     search_tools · load_tool · call_tool         (all the agent sees)
```

## Roadmap

- [ ] MCP → OKT adapter + LLM enrichment pass
- [ ] Eval harness (flat BM25 baseline vs graph-aware retriever)
- [ ] Serving layer as an MCP server (mode 1)
- [ ] Remaining adapters: function, OpenAPI, agent, search
- [ ] Library + HTTP sidecar integration modes
- [ ] Live index refresh on `toolListChanged`

## Contributing

OKTS targets the real [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog) spec, not a private convention — bundles should be portable to any OKF consumer. Retrieval changes must ship with eval numbers (token cost **and** tool-selection accuracy); we don't merge ranking changes on vibes.

## License

_TBD — add before first release._
