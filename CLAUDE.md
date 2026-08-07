# CLAUDE.md

Project memory for coding agents working in this repo. Read this before writing code.

## What this project is

Two things live here, and keeping them distinct matters:

- **OKT (Open Knowledge Tools)** — a *format*. It is an OKF (Google's Open Knowledge Format) profile that describes a single callable tool as one markdown file with YAML frontmatter (`type: tool`). OKT is to tools what OKF is to organizational knowledge. It is a spec, not a program.
- **OKTS (Open Knowledge Tool Search)** — the *runtime*. A single, reusable, source-agnostic wrapper that ingests tools from many sources, emits an OKT bundle, indexes it, and exposes three stable meta-tools to any agent. OKTS is what ships as a package (`okts` on npm and PyPI).

One-line framing: **OKT is the descriptor the project produces; OKTS is the runtime that produces, ranks, and serves it.**

Tagline: *"OKT — open knowledge tools; OKTS — tool search and discovery for any agent."*

## The problem it solves

Agents given dozens of tools degrade: tool schemas eat 30–50% of the context window, attention dilutes, near-duplicate tools collide, and the system prompt gets starved. OKTS fixes this with **progressive disclosure** — the agent never sees N tool schemas up front. It sees three meta-tools and pulls the exact schema it needs, only when it needs it.

## Non-negotiable invariants

These are load-bearing. Do not "improve" them away.

1. **The public surface is exactly three tools, forever:** `search_tools`, `load_tool`, `call_tool`. Adding the 400th upstream tool must not change this contract. This is the agnostic/scalable property.
2. **`input_schema` stays structured, always.** It is the authoritative calling contract, loaded in phase 2. Never let the agent reconstruct a schema from prose. Prose is the signpost; the schema is the contract.
3. **The markdown body is retrieval text, not a contract.** It carries synonyms, when-to/when-not, and gotchas. It is embedded and ranked; it is never sent at call time.
4. **Credentials stay inside OKTS.** Upstream secrets (tokens, keys) live in the wrapper and are used at dispatch. They must never enter the agent's context.
5. **Required-minimum frontmatter is small:** `type`, `id`, `title`, `description`, `input_schema`, `interface`. Everything else degrades gracefully. Adapters must always emit these six.

## Architecture (four layers)

Data flows top to bottom; the agent only ever touches the bottom.

```
Sources        MCP servers · function schemas · sub-agents · search/HTTP APIs
   │  (adapters read from these)
1. Adapters    normalize each source type into an OKT concept  (pure fn: source -> okt)
2. Descriptor  the OKT bundle: one markdown file per tool, cross-linked into a graph
3. Retrieval   hybrid rank (BM25 + dense) + hierarchy prefilter + graph expansion
4. Serving     search_tools · load_tool · call_tool  (this is all the agent sees)
```

The novel part of this project is **layer 3 done over layer 2's graph** — hierarchy- and graph-aware retrieval over a cross-linked tool corpus. Hybrid ranking itself is standard IR; do not describe it as the innovation. The innovation is: (a) a source-agnostic OKT descriptor, and (b) exploiting OKT's `index.md` hierarchy and cross-links during retrieval to disambiguate near-duplicate tools.

**Dense signal default (`okts/index/dense.py`).** "Dense" in the hybrid does *not* mean a neural embedding by default. The shipped default is a deterministic **hashing embedding** (hashed bag-of-words + char n-grams → 256-dim, L2-normalized, cosine) — fully offline, deterministic, `numpy`-only, so CI has no network/keys. That signal is lexical/morphological (`issue`↔`issues`), **not semantic** (`refund`↔`reverse a payment`). `DenseIndex` accepts an injectable `embed_fn`, so a real embedder (OpenAI, sentence-transformers, local model) drops in for semantic retrieval without changing anything else.

**What the ablation actually shows (do not restate the old guess).** On the 148-tool corpus at k=5, the accuracy win is carried by **hybrid fusion**, not by the structural signals and not by dense semantics: BM25-only 92.7%/95.1% (acc@1/collision), dense-only 92.7%/100%, BM25+hierarchy 92.7%/95.1%, BM25+graph 92.7%/95.1%, hybrid-no-structure **97.6%/100%**, hybrid+hierarchy+graph 97.6%/100%. Neither signal reaches 97.6% alone. The hierarchy and graph are accuracy-neutral here — 60 of the 91 derived categories are singletons, and graph expansion is designed not to move top-1 (siblings are damped below their seed). Their role is holding the return count fixed and surfacing alternatives. Regenerate with `python paper/make_results.py` before repeating any of these numbers.

## The OKT format

One file per tool. Frontmatter splits into three consumption groups matching the three runtime phases: **match** (ranked in phase 1), **call** (loaded in phase 2), **route** (used in phase 3).

```yaml
---
# identity
type: tool                      # required — OKF discriminator
id: github.create_issue         # required — stable, namespaced, unique
title: Create GitHub Issue      # required

# match — ranked by search_tools (phase 1); never sent at call time
description: Open a new issue in a GitHub repository.   # required, one line
tags: [github, issues, create, write]                   # recommended

# call — loaded by load_tool (phase 2); the calling contract
input_schema:                   # required — inline JSON Schema OR { resource: ./schema.json }
  type: object
  required: [repo, title]
  properties:
    repo:   { type: string, description: "owner/name" }
    title:  { type: string }
    labels: { type: array, items: { type: string } }
output_schema: { resource: ./create_issue.out.json }    # optional

# route — used by call_tool to dispatch (phase 3)
interface: mcp                  # required — mcp | function | http | agent | search
target: github-mcp              # server name / module path / URL, per interface
auth: github_oauth              # optional
side_effects: write             # recommended — read | write | destructive
invocation: async               # optional — sync | async (default sync); how the target is called
cost: { latency_ms: 400 }       # optional, rank tie-breaks

# graph edges — expanded during search_tools
alternatives:   [./update_issue.md, ./list_issues.md]
prerequisites:  [./get_repo.md]
composes_with:  [./add_labels.md]

# OKF standard
timestamp: 2026-07-30T00:00:00Z
version: "1.0"
---

Use this to **create** a brand-new issue. Do not use it to comment on or edit
an existing one — that's `update_issue`. Synonyms: file a ticket, open a bug.
Gotcha: labels must already exist in the repo or the call 422s.
```

Every emitted bundle must pass an OKF conformance validator. Target the real OKF spec, not a private convention, so bundles stay portable.

## Runtime phases (what loads when)

- **Phase 1 — search.** `search_tools(query, k)` ranks OKT concepts on `description` + `tags` + body, applies the `index.md` hierarchy prefilter, and graph-expands to surface `alternatives`/`composes_with`. Returns lightweight refs (`id`, `title`, `description`) — **not** schemas.
- **Phase 2 — load.** `load_tool(id)` injects the structured `input_schema` (+ `side_effects`) into context. Only the chosen tool's schema loads. Each payload carries an additive **eviction marker** (`{"_okts": {"kind": "schema-instance", "for_id": id}}`) so a context-hygiene scrubber can drop a spent schema from history once its `call_tool` runs — OKTS can't force eviction (it doesn't own the agent's history), so it makes the schema self-identifying instead. Reference scrubber: `examples/context_hygiene.py`.
- **Phase 3 — call.** `call_tool(id, args)` (sync) / `acall_tool(id, args)` (async) validate `args` against the loaded schema, run any injected **pre-dispatch policies**, then dispatch via `interface` + `target` to the real source. Credentials applied here, inside OKTS. Security is done by **policy-based gating, not argument sanitization** (SSRF/SQLi are downstream-tool properties; sanitizing at the proxy is false security). Inject `PreDispatchPolicy` gates via `OKTSService(policies=[...])`; they run at the single choke point (`_prepare_call`) for both sync and async paths and can allow/mutate/deny (raise `PolicyDenied`). Shipped in `okts/serve/policy.py`: `SideEffectPolicy` (enforces `side_effects` — read-only or host-`confirmed` for write/destructive), `RateLimitPolicy`, `ArgRedactionPolicy`, `DomainAllowlistPolicy` (egress scoping). The confirmation/policy `scope` is a **host** arg to `call_tool(..., scope=...)`, never agent args — an agent can't self-authorize. Policies default empty (no behavior change). A tool's optional `invocation` field (`sync`|`async`, default `sync`) declares whether the target is a coroutine — adapters derive it (MCP → `async` since a live `ClientSession.call_tool` is a coroutine; `function_from_callable` → introspects `async def`). Dispatch is robust regardless: `Dispatcher` may provide an optional `adispatch`, the async `acall_tool` awaits async targets natively, and the sync `call_tool` bridges an awaitable when no event loop is running (and errors clearly if called from inside one — use `acall_tool` there). The three-tool public surface is unchanged; `acall_tool` is the async variant of the same phase-3 tool, not a fourth.

## Adapters (layer 1)

Each is a pure function `source -> OKT concept`. Keep them mechanical; a separate LLM enrichment pass fattens the body afterward.

- **MCP -> OKT:** `tools/list`; map `name`->`id`, `inputSchema`->`input_schema`, server->`target`, `interface: mcp`. Map annotations (`readOnlyHint`/`destructiveHint`) to `side_effects`; default `write` + flag if absent.
- **Function schema -> OKT:** `function.name`->`id`, `parameters`->`input_schema`, `interface: function`, callable path->`target`.
- **OpenAPI/REST -> OKT:** one concept per `operationId`; merge params + `requestBody`->`input_schema`; `path`+`method`->`target`; `interface: http`; `auth` from `securitySchemes`.
- **Framework tool -> OKT** (LangChain etc.): `args_schema`->`input_schema`, `interface: function`.
- **Sub-agent -> OKT:** agent card/prompt->body, input contract->`input_schema`, `interface: agent`.
- **Search endpoint -> OKT:** query params->`input_schema`, `interface: search`.

Then: an **enrichment pass** (LLM) expands each body with synonyms/gotchas (this is what lifts retrieval quality), followed by a **structural auto-link pass**, then the **conformance validator**.

**Structural auto-linker (`okts/enrich/autolink.py`).** Real sources hand you flat concepts — `tools/list` (and function/OpenAPI schemas) carry no `alternatives` edges and no `index.md` hierarchy, which are exactly what layer 3 exploits. So OKTS *derives* layer 2 from the flat bundle with a **query-independent, structural** pass: group concepts under `"<server>/<resource>"` categories (namespace + the tool name's primary noun) for the hierarchy, and link same-`<server>/<resource>` tools as mutual `alternatives`. It only *adds* structure and never sees any query, so a flat-vs-graph eval is fair (the baseline runs on the same bundle and ignores the derived signals). A source that already declares edges keeps them (union, not overwrite).

## Integration modes (how an existing agent consumes OKTS)

The agent is coded once against three tools; the tool universe behind them is swappable.

1. **MCP server (default drop-in, zero agent code change).** OKTS runs as one MCP server exposing only the three meta-tools. User replaces N raw server entries in their client config with one `okts` entry. Works with any MCP client.
2. **Library / in-process SDK.** For non-MCP agents: import OKTS, register `search_tools`/`load_tool`/`call_tool` as native functions in the framework's tool list.
3. **Sidecar HTTP service.** For polyglot stacks: run OKTS as a service; agent hits three endpoints.

Optional upgrade for mode 1: if the client supports mid-session tool registration (`defer_loading` style), `load_tool` can register the *native* tool so the agent emits a native call instead of routing through `call_tool`. Assumes that client capability — generic `call_tool` dispatch is the safe default.

The agent's system prompt needs one orientation line: *"to use a tool, `search_tools` for it, `load_tool` to see its parameters, then `call_tool`."*

## Proposed repo layout

```
okts/
  core/           # bundle model, OKT concept types, validator
  adapters/       # mcp.py, function.py, openapi.py, agent.py, search.py
  enrich/         # body enrichment (offline + LLM) + structural auto-linker
  index/          # bm25, dense (hashing default, injectable embed_fn), hybrid, graph-expand, hierarchy
  serve/          # mcp_server, sdk, http_sidecar — the three meta-tools
  config/         # tools.config.yaml loader
  build.py        # end-to-end wiring: config -> adapters -> enrich -> index -> serve
  eval/           # flat-BM25 vs graph-aware harness + large-corpus benchmark (eval/corpus/)
```

Config is the entry point users touch:

```yaml
# tools.config.yaml
sources:
  - interface: mcp
    servers:                            # per-server CONNECTION spec (command/args)
      github-mcp: { command: npx, args: ["-y", "@modelcontextprotocol/server-github"] }
      calc:       { tools: [ ... ] }    # OR an inline offline payload for testing
  - interface: http
    openapi: ./specs/stripe.yaml
  - interface: function
    module: ./my_local_tools.py         # public functions become tools (+ optional `functions: [names]`)
retrieval: { mode: hybrid, graph_expand: true, k: 8 }
bundle_dir: ./okt-bundle
```

**End-to-end pipeline (all wired):** `okts-build --config tools.config.yaml` (`okts.build:main`) runs adapters → enrich → **auto-link** → validate and saves to `bundle_dir`; for mcp servers with a `command` spec it live-connects and ingests (`aconcepts_from_source` → `load_mcp_tools_live`). Then `okts --bundle-dir ./okt-bundle` serves the three meta-tools, and `okts.serve.wiring.open_dispatcher` re-connects the live MCP sessions + registers `module:` callables so `call_tool` dispatches for real (`okts --config …` builds+serves in one shot). The served retriever is graph-aware when `numpy` is present, else the naive fallback. `http`/`search`/`agent` dispatch still needs caller-supplied clients.

All sources merge into one ranked corpus; `search_tools` searches across every source at once and returns the global top-`k`. `retrieval.k` is threaded into `OKTSService(default_k=...)` and becomes the default result count when a `search_tools` call omits `k` (per-call `k` overrides). Defaults to 5 when unset.

## Build order

1. **MCP -> OKT adapter + enrichment pass.** Biggest test corpus, clearest before/after token numbers.
2. **Eval harness** (flat BM25 baseline vs graph-aware retriever). Stand this up early so every change is measured. Baselines to beat: token reduction in the ~85% range and selection accuracy that *rises* (not falls) vs the raw-tools setup.
3. **Serving layer as an MCP server** (mode 1 skeleton: three tools + config loader + dispatch table).
4. Remaining adapters, then modes 2 and 3.

## Testing / eval expectations

The defensible claim is: the OKT graph-aware retriever beats flat BM25 *specifically because of the graph/hierarchy signal*, at comparable or better token cost. Every retrieval change must run against `eval/` and report both token cost and tool-selection accuracy. Do not merge retrieval changes without eval numbers.

## Logging / debuggability

Every stage logs under the `okts.*` logger hierarchy via the stdlib `logging`
module, so a user can trace a build or a query end to end across all four
layers. Following library convention, the package attaches a `NullHandler` and
configures nothing else — **silent by default**, never touching the root
logger. Users opt in with `okts.enable_debug_logging()` (all stages, to stderr),
`okts.enable_debug_logging("okts.index")` (just retrieval), or their own
`logging.basicConfig`.

Levels are consistent across stages: **DEBUG** = per-item tracing (each concept
adapted, the query + ranked hits + scores + hierarchy matches + graph
expansions, the sync/async dispatch path taken); **INFO** = stage milestones
(N tools adapted per source, N concepts enriched, bundle validated, service
ready with N tools); **WARNING** = graceful degradations (malformed source
input skipped, missing dispatcher backend/credential, LLM-enrich fallback).
Loggers map to modules: `okts.adapters.*`, `okts.enrich.*`, `okts.index.retriever`,
`okts.serve.service`, `okts.serve.dispatch`. **Credential values are never
logged** (invariant #4) — dispatch logs only that a credential resolved, by name.

## Naming note

- **Format = OKT (Open Knowledge Tools).** Spec/concept name; no package needed.
- **Runtime = OKTS (Open Knowledge Tool Search).** Ships as `okts` — confirmed available on npm and PyPI. GitHub `@okts` org handle unverified; confirm before claiming.
- `okt` as a package name is taken (npm + PyPI) and collides visually with Okta/Okteto — do not use it for the installable artifact.
- "Search" is the public handle (maps to `search_tools`); "discovery" lives in the tagline to cover the full find-then-reveal loop.
