# Large-corpus eval fixtures

Canned MCP `tools/list` responses for ~20 widely-used MCP servers, used by
`python -m okts.eval.corpus` to validate the two CLAUDE.md claims at scale:

1. **~85% token reduction** — a large-corpus property (fixed 3-meta-tool
   overhead ÷ a large raw corpus), not visible on the 11-tool unit fixture.
2. **graph/hierarchy-aware retrieval beats flat BM25** on selection accuracy.

## Provenance — read this

These files are **hand-authored approximations** of each server's published
tool surface (tool names + realistic, deliberately compact input schemas).
They are **NOT** live captures from running servers, and schemas are trimmed
for readability, so exact tool counts and parameter lists will differ from any
given server version. They are representative enough to exercise retrieval at a
realistic corpus size (~150 tools across 20 servers) with the near-duplicate
collisions that make tool selection hard (e.g. `postgres.run_query` vs
`supabase.run_sql` vs `gcp.run_bigquery_query`; `github.create_issue` vs
`gitlab.create_issue` vs `linear.create_issue`; `docker.get_container_logs` vs
`kubernetes.get_pod_logs` vs `vercel.get_deployment_logs`).

To ingest a *real* server instead, use
`okts.adapters.mcp.load_mcp_tools_live` against a live stdio MCP server — the
adapter path is identical; only the source of the `tools/list` payload differs.

## How they're consumed

Each `<server>.tools.json` is read by `okts.eval.corpus.load_corpus_bundle`,
which runs the standard offline pipeline with no special-casing:

```
tools/list JSON
  -> okts.adapters.mcp.mcp_tools_to_okt   (flat OKT concepts, one per tool)
  -> okts.enrich.enricher.OfflineEnricher (deterministic body enrichment)
  -> okts.enrich.autolink.autolink        (derive hierarchy + alternatives)
  -> okts.core.validator.validate_bundle  (OKF conformance)
```

The auto-linker (`okts/enrich/autolink.py`) is **query-independent** and only
*adds* structure; the flat-BM25 baseline is measured on the identical bundle
and simply ignores the hierarchy/edges. See that module's docstring for why the
comparison stays fair.
