#!/usr/bin/env python
"""Run the OKTS retrieval evaluation and emit LaTeX result fragments.

This is the single source of truth for every number in the paper: it builds the
large corpus, runs the baseline / full / ablation retrievers and the
token-reduction sweep through the *same* ``okts.eval`` harness the test suite
uses, and writes ``paper/results/*.tex`` that ``main.tex`` ``\\input``s. Re-run it
to regenerate the numbers; never hand-edit the fragments.

    python paper/make_results.py            # writes paper/results/*.tex

No new evaluation logic lives here — it only *composes* existing pieces
(``build_corpus_bundle``, ``run_eval``, the ``Retriever`` implementations) so the
paper's figures cannot drift from the code's behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(REPO_ROOT))  # make `okts` importable when run from anywhere

from okts.enrich.autolink import _MAX_ALTERNATIVES  # noqa: E402
from okts.eval.corpus import DEFAULT_CORPUS, DEFAULT_QUERIES, build_corpus_bundle, _subbundle  # noqa: E402
from okts.eval.harness import EvalCase, run_eval  # noqa: E402
from okts.eval.tokens import META_TOOL_SCHEMAS_TOKENS, _ENC  # noqa: E402
from okts.index.dense import DEFAULT_DIM, _NGRAM_SIZES, _NGRAM_WEIGHT  # noqa: E402
from okts.index.graph import EDGE_DAMPING  # noqa: E402
from okts.index.retriever import FlatBM25Retriever, GraphAwareRetriever  # noqa: E402

K = 5
SWEEP_SIZES = (11, 25, 50, 75, 100, 125, 148)


def load_cases(path: Path) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        EvalCase(query=c["query"], expected=c["expected"], distractors=list(c.get("distractors") or []))
        for c in data.get("cases", [])
    ]


def ga(**kw) -> GraphAwareRetriever:
    return GraphAwareRetriever(**kw)


def corpus_stats(bundle, cases) -> dict[str, str]:
    """Structural facts about the corpus/query set that the paper asserts in prose.

    Generated here rather than typed into ``main.tex`` so a corpus change can
    never leave a sentence in the paper stale (the failure mode this whole file
    exists to prevent).
    """
    ids = set(bundle.concepts)
    cat_sizes = sorted(len(v) for v in bundle.hierarchy.values())
    linked = [c for c in bundle if c.alternatives]
    n_alt_edges = sum(len(c.alternatives) for c in bundle)
    n_other_edges = sum(len(c.composes_with) + len(c.prerequisites) for c in bundle)

    distractors = [(c.expected, d) for c in cases for d in c.distractors]
    cross = sum(1 for exp, d in distractors if d.split(".", 1)[0] != exp.split(".", 1)[0])
    resolvable = sum(1 for _, d in distractors if d in ids)
    cases_with_distractor = sum(
        1 for c in cases if any(d in ids for d in c.distractors)
    )
    mid = len(cat_sizes) // 2
    median = (cat_sizes[mid] if len(cat_sizes) % 2 else (cat_sizes[mid - 1] + cat_sizes[mid]) / 2)

    d = GraphAwareRetriever()  # read the shipped defaults off the class itself
    return {
        # --- corpus structure ---
        "numSingletonCategories": str(sum(1 for s in cat_sizes if s == 1)),
        "medianCategorySize": f"{median:g}",
        "maxCategorySize": str(cat_sizes[-1]),
        "numLinkedConcepts": str(len(linked)),
        "numAltEdges": str(n_alt_edges),
        "numOtherEdges": str(n_other_edges),
        # --- query set structure ---
        "numDistractors": str(len(distractors)),
        "numCrossServerDistractors": str(cross),
        "numResolvableDistractors": str(resolvable),
        "numCasesWithDistractor": str(cases_with_distractor),
        # --- shipped retriever hyperparameters ---
        "denseDim": str(DEFAULT_DIM),
        "ngramSizes": ",".join(str(n) for n in _NGRAM_SIZES),
        "ngramWeight": f"{_NGRAM_WEIGHT:g}",
        "bmWeight": f"{d.bm25_weight:g}",
        "denseWeight": f"{d.dense_weight:g}",
        "bmKOne": f"{d._bm25.k1:g}",
        "bmB": f"{d._bm25.b:g}",
        "hierBoost": f"{d.hierarchy_boost:g}",
        "hierTopCats": str(d.hierarchy_top_categories),
        "graphMaxPerHit": str(d.graph_max_per_hit),
        "maxAlternatives": str(_MAX_ALTERNATIVES),
        "altDamping": f"{EDGE_DAMPING['alternatives']:g}",
        "otherDamping": f"{EDGE_DAMPING['composes_with']:g}",
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    bundle = build_corpus_bundle(DEFAULT_CORPUS)
    cases = load_cases(DEFAULT_QUERIES)
    n_tools = sum(1 for _ in bundle)
    n_servers = len({c.id.split(".", 1)[0] for c in bundle})
    n_cats = len(bundle.hierarchy)
    stats = corpus_stats(bundle, cases)
    enriched_concepts = list(bundle)  # already enriched+linked; good enough for the sweep

    # ---- headline: baseline vs full graph-aware -----------------------------
    flat = run_eval(bundle, FlatBM25Retriever(), cases, k=K)
    full = run_eval(bundle, ga(mode="hybrid", hierarchy_prefilter=True, graph_expand=True), cases, k=K)

    # ---- ablation: isolate each structural signal ---------------------------
    ablations = [
        ("BM25 only (baseline)", FlatBM25Retriever()),
        ("Dense only (hashing)", ga(mode="dense", hierarchy_prefilter=False, graph_expand=False)),
        ("Hybrid, no structure", ga(mode="hybrid", hierarchy_prefilter=False, graph_expand=False)),
        ("BM25 + hierarchy", ga(mode="bm25", hierarchy_prefilter=True, graph_expand=False)),
        ("BM25 + graph", ga(mode="bm25", hierarchy_prefilter=False, graph_expand=True)),
        ("Hybrid + hierarchy + graph", ga(mode="hybrid", hierarchy_prefilter=True, graph_expand=True)),
    ]
    ablation_reports = [(name, run_eval(bundle, r, cases, k=K)) for name, r in ablations]

    # ---- token-reduction sweep over growing corpus size ---------------------
    sweep = []
    sizes = [n for n in SWEEP_SIZES if n < n_tools] + [n_tools]
    for n in sorted(set(sizes)):
        sub = _subbundle(enriched_concepts, n)
        rep = run_eval(sub, ga(mode="hybrid", hierarchy_prefilter=True, graph_expand=True), cases, k=K)
        sweep.append((n, rep.raw_tools_tokens, rep.avg_okts_tokens, rep.token_reduction_pct))

    tokenizer = "tiktoken cl100k_base" if _ENC is not None else "chars/4 heuristic"

    # Prose in the Results section names specific ablation rows; bind them to
    # macros so the narrative cannot drift from the table above it.
    by_name = dict(ablation_reports)
    stats.update(
        denseAccOne=_pct(by_name["Dense only (hashing)"].accuracy_at_1),
        denseColl=_pct(by_name["Dense only (hashing)"].collision_avoidance),
        hybridBareAccOne=_pct(by_name["Hybrid, no structure"].accuracy_at_1),
        hybridBareColl=_pct(by_name["Hybrid, no structure"].collision_avoidance),
        bmHierAccOne=_pct(by_name["BM25 + hierarchy"].accuracy_at_1),
        bmGraphAccOne=_pct(by_name["BM25 + graph"].accuracy_at_1),
    )

    _write_macros(n_tools, n_servers, n_cats, len(cases), flat, full, sweep, tokenizer, stats)
    _write_main_table(flat, full)
    _write_ablation_table(ablation_reports)
    _write_sweep(sweep)
    _write_console(bundle, n_tools, n_servers, n_cats, len(cases), flat, full, ablation_reports, sweep, tokenizer)

    print(f"wrote LaTeX fragments to {RESULTS}")
    print(f"  tokenizer: {tokenizer}")
    print(f"  corpus: {n_tools} tools / {n_servers} servers / {n_cats} categories / {len(cases)} queries")
    print(f"  flat  acc@1={flat.accuracy_at_1:.1%} coll={flat.collision_avoidance:.1%}")
    print(f"  graph acc@1={full.accuracy_at_1:.1%} coll={full.collision_avoidance:.1%} reduction={full.token_reduction_pct:.1f}%")
    return 0


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _write_macros(n_tools, n_servers, n_cats, n_q, flat, full, sweep, tokenizer, stats) -> None:
    red75 = next((r for n, _, _, r in sweep if n == 75), None)
    lines = [
        "% AUTO-GENERATED by paper/make_results.py -- do not edit by hand.",
        f"\\newcommand{{\\numTools}}{{{n_tools}}}",
        f"\\newcommand{{\\numServers}}{{{n_servers}}}",
        f"\\newcommand{{\\numCategories}}{{{n_cats}}}",
        f"\\newcommand{{\\numQueries}}{{{n_q}}}",
        f"\\newcommand{{\\metaToolTokens}}{{{META_TOOL_SCHEMAS_TOKENS}}}",
        f"\\newcommand{{\\tokenizerName}}{{{tokenizer.replace('_', chr(92) + '_')}}}",
        f"\\newcommand{{\\flatAccOne}}{{{_pct(flat.accuracy_at_1)}}}",
        f"\\newcommand{{\\graphAccOne}}{{{_pct(full.accuracy_at_1)}}}",
        f"\\newcommand{{\\flatAccK}}{{{_pct(flat.accuracy_at_k)}}}",
        f"\\newcommand{{\\graphAccK}}{{{_pct(full.accuracy_at_k)}}}",
        f"\\newcommand{{\\flatMRR}}{{{flat.mrr:.3f}}}",
        f"\\newcommand{{\\graphMRR}}{{{full.mrr:.3f}}}",
        f"\\newcommand{{\\flatColl}}{{{_pct(flat.collision_avoidance)}}}",
        f"\\newcommand{{\\graphColl}}{{{_pct(full.collision_avoidance)}}}",
        f"\\newcommand{{\\avgTok}}{{{full.avg_okts_tokens:.0f}}}",
        f"\\newcommand{{\\rawTokFull}}{{{full.raw_tools_tokens}}}",
        f"\\newcommand{{\\reductionFull}}{{{full.token_reduction_pct:.1f}}}",
        f"\\newcommand{{\\kval}}{{{K}}}",
    ]
    if red75 is not None:
        lines.append(f"\\newcommand{{\\reductionSeventyFive}}{{{red75:.1f}}}")
    lines.append("% corpus/query structure + shipped retriever hyperparameters")
    for name, value in stats.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (RESULTS / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(name, r) -> str:
    return (
        f"{name} & {_pct(r.accuracy_at_1)} & {_pct(r.accuracy_at_k)} & {r.mrr:.3f} "
        f"& {_pct(r.collision_avoidance)} & {r.avg_okts_tokens:.0f} \\\\"
    )


def _write_main_table(flat, full) -> None:
    body = "\n".join([
        "% AUTO-GENERATED by paper/make_results.py",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Retriever & acc@1 & acc@$k$ & MRR & coll.\\ avoid. & avg tok/qry \\\\",
        "\\midrule",
        _row("Flat BM25 (baseline)", flat),
        "\\textbf{Graph-aware (ours)} & \\textbf{" + _pct(full.accuracy_at_1) + "} & "
        + _pct(full.accuracy_at_k) + " & \\textbf{" + f"{full.mrr:.3f}" + "} & \\textbf{"
        + _pct(full.collision_avoidance) + "} & " + f"{full.avg_okts_tokens:.0f}" + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (RESULTS / "main_table.tex").write_text(body + "\n", encoding="utf-8")


def _write_ablation_table(reports) -> None:
    rows = [_row(name, r) for name, r in reports]
    body = "\n".join([
        "% AUTO-GENERATED by paper/make_results.py",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Configuration & acc@1 & acc@$k$ & MRR & coll.\\ avoid. & avg tok/qry \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (RESULTS / "ablation_table.tex").write_text(body + "\n", encoding="utf-8")


def _write_sweep(sweep) -> None:
    coords = " ".join(f"({n},{red:.1f})" for n, _, _, red in sweep)
    raw_coords = " ".join(f"({n},{raw})" for n, raw, _, _ in sweep)
    okts_coords = " ".join(f"({n},{okts:.0f})" for n, _, okts, _ in sweep)
    (RESULTS / "sweep_reduction.tex").write_text(
        "% AUTO-GENERATED by paper/make_results.py\n\\addplot coordinates {" + coords + "};\n",
        encoding="utf-8",
    )
    (RESULTS / "sweep_tokens_raw.tex").write_text(
        "\\addplot coordinates {" + raw_coords + "};\n", encoding="utf-8"
    )
    (RESULTS / "sweep_tokens_okts.tex").write_text(
        "\\addplot coordinates {" + okts_coords + "};\n", encoding="utf-8"
    )


def _write_console(bundle, n_tools, n_servers, n_cats, n_q, flat, full, ablations, sweep, tokenizer) -> None:
    """A human-readable capture of the run, for the reproducibility appendix."""
    lines = [
        "OKTS retrieval evaluation -- raw output",
        "=" * 60,
        f"tokenizer:  {tokenizer}",
        f"corpus:     {n_tools} tools, {n_servers} servers, {n_cats} categories",
        f"queries:    {n_q} labeled cases   k={K}",
        "",
        f"{'retriever':<30}{'acc@1':>7}{'acc@k':>7}{'MRR':>7}{'coll':>7}{'tok':>8}",
        "-" * 66,
    ]
    for name, r in [("Flat BM25 (baseline)", flat), ("Graph-aware (full)", full)]:
        lines.append(f"{name:<30}{r.accuracy_at_1:>6.1%}{r.accuracy_at_k:>7.1%}{r.mrr:>7.3f}{r.collision_avoidance:>7.1%}{r.avg_okts_tokens:>8.0f}")
    lines += ["", "ablation:", "-" * 66]
    for name, r in ablations:
        lines.append(f"{name:<30}{r.accuracy_at_1:>6.1%}{r.accuracy_at_k:>7.1%}{r.mrr:>7.3f}{r.collision_avoidance:>7.1%}{r.avg_okts_tokens:>8.0f}")
    lines += ["", "token reduction sweep:", f"{'N':>6}{'raw':>10}{'okts':>10}{'reduction':>12}", "-" * 38]
    for n, raw, okts, red in sweep:
        lines.append(f"{n:>6}{raw:>10}{okts:>10.0f}{red:>11.1f}%")
    (RESULTS / "console.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
