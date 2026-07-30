"""Large-corpus eval: validate the ~85% token-reduction claim + the graph-aware
accuracy lift on a realistic ~150-tool corpus adapted from ~20 real MCP servers.

Run it::

    python -m okts.eval.corpus [--corpus DIR] [--queries PATH] [--k N]

Pipeline (no special-casing — the ordinary offline path):

    eval/corpus/*.tools.json
      -> mcp_tools_to_okt   (flat OKT concepts)
      -> OfflineEnricher    (deterministic body enrichment)
      -> autolink           (derive index.md hierarchy + alternatives edges)
      -> validate_bundle    (OKF conformance)

Then: a full-size FlatBM25 vs GraphAware comparison (accuracy + token cost via
``okts.eval.harness``), plus a token-reduction sweep over growing corpus sizes
that shows reduction climbing past 85% as the corpus grows — the whole point of
"85% is a large-corpus property".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from okts.adapters.mcp import mcp_tools_to_okt
from okts.core.model import Bundle
from okts.core.validator import validate_bundle
from okts.enrich.autolink import autolink
from okts.enrich.enricher import OfflineEnricher, enrich_bundle

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = _REPO_ROOT / "eval" / "corpus"
DEFAULT_QUERIES = _REPO_ROOT / "eval" / "corpus_queries.yaml"

# Corpus sizes for the reduction sweep (clamped to the real corpus size).
_SWEEP_SIZES = (11, 25, 50, 75, 100, 150, 250)


def _server_name(path: Path) -> str:
    """`eval/corpus/chrome-devtools.tools.json` -> `chrome-devtools`."""
    return path.name[: -len(".tools.json")] if path.name.endswith(".tools.json") else path.stem


def load_corpus_concepts(corpus_dir: Path = DEFAULT_CORPUS) -> list:
    """Adapt every ``*.tools.json`` file into flat OKT concepts (pre-link)."""
    concepts = []
    for path in sorted(corpus_dir.glob("*.tools.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        concepts.extend(mcp_tools_to_okt(data, server=_server_name(path)))
    return concepts


def build_corpus_bundle(
    corpus_dir: Path = DEFAULT_CORPUS, *, validate: bool = True
) -> Bundle:
    """Full offline pipeline -> enriched, auto-linked, conformant Bundle."""
    flat = Bundle()
    for concept in load_corpus_concepts(corpus_dir):
        flat.add(concept)
    enriched = enrich_bundle(flat, OfflineEnricher())
    linked = autolink(enriched)
    if validate:
        problems = validate_bundle(linked, check_edges=True)
        if problems:
            raise ValueError(
                "corpus bundle failed OKF conformance:\n  - " + "\n  - ".join(problems)
            )
    return linked


def _subbundle(concepts: list, n: int) -> Bundle:
    """Auto-linked Bundle over the first ``n`` enriched concepts (for the sweep)."""
    flat = Bundle()
    for concept in concepts[:n]:
        flat.add(concept)
    return autolink(flat)


def _load_cases(path: Path):
    from okts.eval.harness import EvalCase

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        EvalCase(
            query=c["query"],
            expected=c["expected"],
            distractors=list(c.get("distractors") or []),
        )
        for c in data.get("cases", [])
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        from okts.index.retriever import FlatBM25Retriever, GraphAwareRetriever
    except ImportError as exc:  # pragma: no cover
        print(f"index layer unavailable ({exc}); cannot run corpus eval.")
        return 1

    from okts.eval.harness import run_eval

    # --- build corpus (once): flat -> enrich -> link ---
    enriched = enrich_bundle(_flat(args.corpus), OfflineEnricher())
    enriched_concepts = list(enriched)  # for the size sweep below
    bundle = autolink(enriched)
    problems = validate_bundle(bundle, check_edges=True)

    servers = sorted({c.id.split(".", 1)[0] for c in bundle})
    print("== OKTS large-corpus eval ==")
    print(f"  corpus dir:      {args.corpus}")
    print(f"  servers:         {len(servers)}  ({', '.join(servers)})")
    print(f"  tools (concepts):{sum(1 for _ in bundle)}")
    print(f"  categories:      {len(bundle.hierarchy)}")
    print(f"  OKF-conformant:  {'yes' if not problems else 'NO -> ' + str(problems)}")

    cases = _load_cases(args.queries)
    print(f"  labeled queries: {len(cases)}")

    # --- full-size accuracy + token comparison ---
    reports = []
    for name, retriever in (
        ("FlatBM25Retriever", FlatBM25Retriever()),
        ("GraphAwareRetriever", GraphAwareRetriever()),
    ):
        report = run_eval(bundle, retriever, cases, k=args.k)
        report.retriever_name = name
        reports.append(report)

    header = (
        f"{'retriever':<22}{'acc@1':>8}{'acc@k':>8}{'MRR':>8}"
        f"{'coll-avoid':>12}{'avg tok/qry':>14}{'reduction':>12}"
    )
    print("\n-- full corpus: flat BM25 vs graph-aware --")
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.retriever_name:<22}{r.accuracy_at_1:>8.1%}{r.accuracy_at_k:>8.1%}"
            f"{r.mrr:>8.3f}{r.collision_avoidance:>12.1%}"
            f"{r.avg_okts_tokens:>14.1f}{r.token_reduction_pct:>11.1f}%"
        )

    # --- token-reduction sweep over growing corpus size ---
    print("\n-- token reduction vs corpus size (graph-aware) --")
    sweep_header = f"{'N tools':>8}{'raw tokens':>14}{'avg okts tok':>16}{'reduction':>12}"
    print(sweep_header)
    print("-" * len(sweep_header))
    total = sum(1 for _ in bundle)
    sizes = [n for n in _SWEEP_SIZES if n < total] + [total]
    for n in sizes:
        sub = _subbundle(enriched_concepts, n)
        rep = run_eval(sub, GraphAwareRetriever(), cases, k=args.k)
        print(
            f"{n:>8}{rep.raw_tools_tokens:>14}{rep.avg_okts_tokens:>16.1f}"
            f"{rep.token_reduction_pct:>11.1f}%"
        )

    return 0


def _flat(corpus_dir: Path) -> Bundle:
    flat = Bundle()
    for concept in load_corpus_concepts(corpus_dir):
        flat.add(concept)
    return flat


if __name__ == "__main__":
    sys.exit(main())
