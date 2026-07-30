"""CLI: compare the flat-BM25 baseline vs the graph-aware retriever on the
labeled query set, reporting both selection accuracy and token cost.

Usage::

    python -m okts.eval.run [--bundle PATH] [--queries PATH] [--k N]

``okts.index`` (layer 3) is imported lazily inside :func:`main`, not at module
scope, so this file stays importable -- and usable the moment the index layer
lands -- even while layer 3 is still being built in parallel. If the import
fails, a clear message is printed instead of a crash/traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from okts.core.bundle_io import load_bundle
from okts.eval.harness import EvalCase, EvalReport, run_eval

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = _REPO_ROOT / "tests" / "fixtures" / "bundle"
DEFAULT_QUERIES = _REPO_ROOT / "eval" / "queries.yaml"


def _load_cases(path: Path) -> list[EvalCase]:
    """Parse ``eval/queries.yaml``-shaped YAML into :class:`EvalCase` list."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        EvalCase(
            query=c["query"],
            expected=c["expected"],
            distractors=list(c.get("distractors") or []),
        )
        for c in data.get("cases", [])
    ]


def _print_report(report: EvalReport) -> None:
    print(f"\n== {report.retriever_name} ==")
    print(f"  cases:                {report.num_cases}")
    print(f"  accuracy@1:           {report.accuracy_at_1:.1%}")
    print(f"  accuracy@{report.k}:           {report.accuracy_at_k:.1%}")
    print(f"  MRR:                  {report.mrr:.3f}")
    print(f"  collision-avoidance:  {report.collision_avoidance:.1%}")
    print(f"  raw-tools tokens:     {report.raw_tools_tokens}")
    print(f"  avg OKTS tokens/qry:  {report.avg_okts_tokens:.1f}")
    print(f"  token reduction:      {report.token_reduction_pct:.1f}%")


def _print_comparison(reports: list[EvalReport]) -> None:
    header = (
        f"{'retriever':<22}{'acc@1':>8}{'acc@k':>8}{'MRR':>8}"
        f"{'coll-avoid':>12}{'avg tok/qry':>14}{'reduction':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.retriever_name:<22}"
            f"{r.accuracy_at_1:>8.1%}"
            f"{r.accuracy_at_k:>8.1%}"
            f"{r.mrr:>8.3f}"
            f"{r.collision_avoidance:>12.1%}"
            f"{r.avg_okts_tokens:>14.1f}"
            f"{r.token_reduction_pct:>11.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point: load fixtures, run both retrievers, print a comparison."""
    parser = argparse.ArgumentParser(
        description="OKTS retrieval eval: flat BM25 baseline vs graph-aware retriever"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        from okts.index.retriever import FlatBM25Retriever, GraphAwareRetriever
    except ImportError as exc:
        print(
            "index layer not available yet -- `okts.index.retriever` could not "
            f"be imported ({exc}). This CLI will work as soon as layer 3 "
            "(FlatBM25Retriever / GraphAwareRetriever) lands; the harness "
            "itself (okts/eval/harness.py) is already usable against any "
            "Retriever-protocol stub in the meantime."
        )
        return 1

    bundle = load_bundle(args.bundle)
    cases = _load_cases(args.queries)

    reports: list[EvalReport] = []
    for name, retriever in (
        ("FlatBM25Retriever", FlatBM25Retriever()),
        ("GraphAwareRetriever", GraphAwareRetriever()),
    ):
        report = run_eval(bundle, retriever, cases, k=args.k)
        report.retriever_name = name
        reports.append(report)
        _print_report(report)

    _print_comparison(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
