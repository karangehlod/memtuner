#!/usr/bin/env python3
"""MemTuner report generator.

Scans every data/output/study_* CSV, aggregates results across all datasets,
and writes three artefacts:

  data/output/master_results.csv           — all cells, composite math columns
  data/output/COMPOSITE_SCORE_FORMULA.md  — formula reference doc
  data/output/reports_data.js             — JSON data consumed by both HTML dashboards

Run manually:
    python scripts/generate_reports.py

Auto-triggered by study_runner.py at the end of every benchmark run.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Load config — YAML + .env overrides.  cfg is the single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import cfg  # noqa: E402

# Module-level aliases so the rest of the file stays readable
DATASET_MAP: dict[int, str] = cfg.datasets.query_count_to_name
COMPOSITE_W: dict[str, float] = cfg.composite.weights


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_cells(output_dir: Path) -> list[dict]:
    cells: list[dict] = []
    for study_dir in sorted(output_dir.iterdir()):
        if not study_dir.is_dir() or not study_dir.name.startswith("study_"):
            continue
        for csv_path in study_dir.glob("*_grid.csv"):
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("success", "True").lower() != "true":
                            continue
                        nq = int(row.get("total_queries", 0))
                        row["dataset_name"]  = DATASET_MAP.get(nq, f"unknown_{nq}")
                        row["source_study"]  = study_dir.name
                        cells.append(row)
            except Exception as exc:
                print(f"  [warn] {csv_path.name}: {exc}", file=sys.stderr)
    return cells


# ── Composite math ────────────────────────────────────────────────────────────

def _gate(r: float) -> float:
    return 1.0 if r >= cfg.composite.recall_gate else 0.0

def composite(r: float, p: float, m: float, t: float) -> float:
    g = _gate(r)
    return g * (COMPOSITE_W["recall"] * r + COMPOSITE_W["precision"] * p
                + COMPOSITE_W["mrr"] * m + COMPOSITE_W["temporal"] * t)


# ── Master CSV ────────────────────────────────────────────────────────────────

MATH_COLS = [
    "composite_score_computed",  # verified recalculation
    "w_recall",                  # 0.40 × R@10 × gate
    "w_precision",               # 0.25 × P@10 × gate
    "w_mrr",                     # 0.20 × MRR × gate
    "w_temporal",                # 0.15 × TA × gate
    "recall_gate",               # 1 if R@10 ≥ 0.01 else 0
    "composite_formula",         # human-readable breakdown string
]

def write_master_csv(cells: list[dict], out_path: Path) -> None:
    if not cells:
        print("  [skip] No cells found — master CSV not written.", file=sys.stderr)
        return

    base_cols = [k for k in cells[0] if k not in MATH_COLS]
    all_cols  = ["dataset_name", "source_study"] + [
        c for c in base_cols if c not in ("dataset_name", "source_study")
    ] + MATH_COLS

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # Formula header (comment rows — parsers that skip '#' lines still work)
        for line in [
            "# ============================================================",
            "# MemTuner — Master Results",
            f"# Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Total cells: {len(cells)}",
            "# ============================================================",
            "# COMPOSITE SCORE FORMULA",
            "#   composite = gate × (w_R×R@10 + w_P×P@10 + w_M×MRR + w_T×TA)",
            "#   gate      = 1  if R@10 ≥ 0.01  else  0",
            "#   w_R=0.40  w_P=0.25  w_M=0.20  w_T=0.15  (sum=1.00)",
            "#",
            "#   w_recall    = 0.40 × Recall@10   × gate",
            "#   w_precision = 0.25 × Precision@10 × gate",
            "#   w_mrr       = 0.20 × MRR          × gate",
            "#   w_temporal  = 0.15 × TemporalAcc  × gate",
            "# ============================================================",
        ]:
            f.write(line + "\n")

        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()

        for cell in cells:
            r = float(cell.get("recall_at_k",       0))
            p = float(cell.get("precision_at_k",    0))
            m = float(cell.get("mrr",               0))
            t = float(cell.get("temporal_accuracy", 0))
            g = _gate(r)
            c = composite(r, p, m, t)

            cell["recall_gate"]               = f"{g:.1f}"
            cell["w_recall"]                  = f"{COMPOSITE_W['recall']    * r * g:.6f}"
            cell["w_precision"]               = f"{COMPOSITE_W['precision'] * p * g:.6f}"
            cell["w_mrr"]                     = f"{COMPOSITE_W['mrr']       * m * g:.6f}"
            cell["w_temporal"]                = f"{COMPOSITE_W['temporal']  * t * g:.6f}"
            cell["composite_score_computed"]  = f"{c:.6f}"
            cell["composite_formula"] = (
                f"gate={g:.0f} × (0.40×{r:.4f} + 0.25×{p:.4f} + 0.20×{m:.4f} + 0.15×{t:.4f}) = {c:.6f}"
            )
            writer.writerow(cell)

    print(f"  master_results.csv  → {out_path.name}  ({len(cells)} cells)")


# ── Formula doc ───────────────────────────────────────────────────────────────

def write_formula_doc(out_path: Path, cells: list[dict] | None = None) -> None:
    """Update the Per-Dataset Results section of the formula reference doc.

    The static formula content (sections 1–6, 8–9) is preserved exactly as written.
    Only section 7 (latest benchmark results) is regenerated from current cells.
    If the file doesn't exist yet, a minimal stub is written first.
    """
    SECTION_MARKER = "## 7. Per-Dataset Benchmark Results (latest run)"

    # Read existing content (preserve all static formula sections)
    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        # Trim everything from section 7 onwards so we can replace it
        cut_idx = existing.find(SECTION_MARKER)
        base    = existing[:cut_idx].rstrip() + "\n\n" if cut_idx != -1 else existing.rstrip() + "\n\n"
    else:
        # File doesn't exist — start with a minimal header; full content is in the repo
        base = "# MemTuner — Complete Formula Reference\n\nSee README.md § Mathematical Foundations.\n\n"

    if not cells:
        out_path.write_text(base, encoding="utf-8")
        print(f"  COMPOSITE_SCORE_FORMULA.md  → {out_path.name}")
        return

    # Build fresh section 7 from current cells
    DATASET_ORDER = cfg.datasets.display_order
    by_ds: dict[str, list] = defaultdict(list)
    for c in cells:
        by_ds[c["dataset_name"]].append(c)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        SECTION_MARKER,
        "",
        f"_Auto-updated by `generate_reports.py` — {ts} — {len(cells)} cells_",
        "",
        "### Best configurations",
        "",
        "| Dataset | Strategy | Embedding | BM25 w | Decay | Recall@10 | Composite |",
        "|---------|----------|-----------|--------|-------|-----------|-----------|",
    ]

    results = []
    for ds in DATASET_ORDER:
        rows = by_ds.get(ds)
        if not rows:
            continue
        best = max(rows, key=lambda r: composite(
            float(r.get("recall_at_k", 0)), float(r.get("precision_at_k", 0)),
            float(r.get("mrr", 0)), float(r.get("temporal_accuracy", 0))
        ))
        r10  = float(best.get("recall_at_k",    0))
        p10  = float(best.get("precision_at_k", 0))
        mrr  = float(best.get("mrr",            0))
        ta   = float(best.get("temporal_accuracy", 0))
        comp = composite(r10, p10, mrr, ta)
        strat  = best.get("retrieval_strategy", "")
        embed  = best.get("embedding_model", "").split("/")[-1]
        decay  = best.get("decay_policy", "")
        bm25w  = float(best.get("bm25_weight", 0))
        lines.append(f"| **{ds}** | {strat} | {embed} | {bm25w:.2f} | {decay} | **{r10:.4f}** | {comp:.4f} |")
        results.append((ds, strat, embed, bm25w, decay, r10, p10, mrr, ta, comp, len(rows)))

    lines += ["", "### Composite breakdown per dataset (best config)", ""]
    for ds, strat, embed, bm25w, decay, r10, p10, mrr, ta, comp, n in results:
        gate = 1.0 if r10 >= cfg.composite.recall_gate else 0.0
        lines.append(
            f"```\n{ds:<14} = {gate:.0f} × "
            f"(0.40×{r10:.4f} + 0.25×{p10:.4f} + 0.20×{mrr:.4f} + 0.15×{ta:.4f})"
            f" = {comp:.4f}\n```"
        )

    lines += [
        "",
        "### Dataset cell counts",
        "",
        "| Dataset | Cells benchmarked | Best strategy | Best Recall@10 |",
        "|---------|------------------|---------------|----------------|",
    ]
    for ds, strat, embed, bm25w, decay, r10, p10, mrr, ta, comp, n in results:
        lines.append(f"| {ds} | {n} | {strat} | {r10:.4f} |")

    out_path.write_text(base + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  COMPOSITE_SCORE_FORMULA.md  → {out_path.name}  (section 7 refreshed)")


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _agg_strategy(rows: list[dict]) -> list[dict]:
    by_strat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_strat[r["retrieval_strategy"]].append(r)
    result = []
    for s, rs in by_strat.items():
        r10s = [float(x.get("recall_at_k", 0))    for x in rs]
        p1s  = [float(x.get("precision_at_1", 0)) for x in rs]
        mrrs = [float(x.get("mrr", 0))            for x in rs]
        comps= [composite(float(x.get("recall_at_k",0)), float(x.get("precision_at_k",0)),
                          float(x.get("mrr",0)), float(x.get("temporal_accuracy",0))) for x in rs]
        result.append({
            "name": s, "n": len(rs),
            "avg": round(_avg(r10s), 4), "best": round(max(r10s), 4),
            "p1":  round(_avg(p1s), 4), "mrr": round(_avg(mrrs), 4),
            "avg_comp": round(_avg(comps), 4),
        })
    return sorted(result, key=lambda x: -x["avg"])

def _agg_decay(rows: list[dict], strategy: str | None = None) -> list[dict]:
    subset = [r for r in rows if strategy is None or r["retrieval_strategy"] == strategy]
    by_decay: dict[str, list] = defaultdict(list)
    for r in subset:
        by_decay[r["decay_policy"]].append(r)
    result = []
    for d, rs in by_decay.items():
        r10s = [float(x.get("recall_at_k", 0)) for x in rs]
        p1s  = [float(x.get("precision_at_1", 0)) for x in rs]
        mrrs = [float(x.get("mrr", 0)) for x in rs]
        result.append({"name": d, "n": len(rs),
                        "r10": round(_avg(r10s), 4),
                        "p1":  round(_avg(p1s), 4),
                        "mrr": round(_avg(mrrs), 4)})
    return sorted(result, key=lambda x: -x["r10"])

def _best_config(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get("composite_score", 0)))
    return {
        "strategy":  best.get("retrieval_strategy", ""),
        "embedding": best.get("embedding_model", "").split("/")[-1],
        "backend":   best.get("embedding_backend", ""),
        "decay":     best.get("decay_policy", ""),
        "lam":       float(best.get("lambda", 0)),
        "bm25w":     float(best.get("bm25_weight", 0)),
        "recall10":  round(float(best.get("recall_at_k", 0)), 4),
        "composite": round(float(best.get("composite_score", 0)), 4),
    }


# ── reports_data.js ───────────────────────────────────────────────────────────

def build_reports_data(cells: list[dict]) -> dict:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_ds[c["dataset_name"]].append(c)

    ds_order = ["SQuAD", "CoQA", "LoCoMo", "LongMemEval", "Synthetic"]

    # ── global strategy ranking (all datasets) ────────────────────────────────
    global_strat = _agg_strategy(cells)

    # ── global embedding ranking ──────────────────────────────────────────────
    by_embed: dict[str, list] = defaultdict(list)
    for c in cells:
        em = c.get("embedding_model", "")
        if em and em != "none":
            by_embed[em].append(float(c.get("recall_at_k", 0)))
    embed_ranking = sorted(
        [{"name": k.split("/")[-1], "full": k, "n": len(v), "avg": round(_avg(v), 4)}
         for k, v in by_embed.items()],
        key=lambda x: -x["avg"]
    )[:8]

    # ── global decay ranking ──────────────────────────────────────────────────
    decay_ranking = _agg_decay(cells)

    # ── per-dataset records ───────────────────────────────────────────────────
    datasets = []
    for ds_name in ds_order:
        rows = by_ds.get(ds_name, [])
        if not rows:
            continue

        strats = _agg_strategy(rows)
        winner = strats[0] if strats else {}

        # Determine best decay strategy (use top strategy by avg recall)
        top_strat_name = winner.get("name", "hybrid")
        decay_rows = _agg_decay(rows, strategy=top_strat_name)
        if not decay_rows:
            decay_rows = _agg_decay(rows)

        multi_relevant = any(
            float(r.get("precision_at_1", 0)) > float(r.get("recall_at_k", 0)) + 0.05
            for r in rows[:20]
        )

        datasets.append({
            "id":             ds_name.lower().replace(" ", "_"),
            "label":          ds_name,
            "cells":          len(rows),
            "multiRelevant":  multi_relevant,
            "winner":         top_strat_name,
            "winnerAvg":      winner.get("avg", 0),
            "winnerBest":     winner.get("best", 0),
            "strategies":     strats,
            "decay":          decay_rows,
            "bestConfig":     _best_config(rows),
        })

    # ── per-dataset recall chart (for main dashboard) ─────────────────────────
    ds_recall = [
        {"name": d["label"], "strat": d["winner"],
         "recall": d["winnerAvg"], "best": d["winnerBest"]}
        for d in datasets
    ]

    # ── overall best ──────────────────────────────────────────────────────────
    overall_best = _best_config(cells)

    return {
        "generatedAt":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "totalCells":    len(cells),
        "compositeWeights": COMPOSITE_W,
        "compositeFormula": (
            cfg.composite.formula_str
        ),
        "overallBest":   overall_best,
        "strategyRanking": global_strat,
        "embeddingRanking": embed_ranking,
        "decayRanking":  decay_ranking,
        "dsRecall":      ds_recall,
        "datasets":      datasets,
    }


def write_reports_data_js(data: dict, out_path: Path) -> None:
    ts  = data["generatedAt"]
    n   = data["totalCells"]
    js  = (
        f"/* AUTO-GENERATED by scripts/generate_reports.py — do not edit manually */\n"
        f"/* Last updated: {ts} | {n} cells | {len(data['datasets'])} datasets */\n"
        f"const REPORT_DATA = {json.dumps(data, indent=2)};\n"
    )
    out_path.write_text(js, encoding="utf-8")
    print(f"  reports_data.js          → {out_path.name}  ({n} cells, {len(data['datasets'])} datasets)")


# ── Entry point ───────────────────────────────────────────────────────────────

def generate(project_root: Path | None = None) -> None:
    # project_root param kept for backward compat with study_runner hook
    output_dir = cfg.reporting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nMemTuner Report Generator")
    print(f"  Scanning {output_dir} ...")

    cells = load_all_cells(output_dir)
    if not cells:
        print("  No cells found. Run a benchmark first.", file=sys.stderr)
        return

    ds_counts = defaultdict(int)
    for c in cells:
        ds_counts[c["dataset_name"]] += 1
    print(f"  Loaded {len(cells)} cells:")
    for ds, n in sorted(ds_counts.items(), key=lambda x: -x[1]):
        print(f"    {ds:<16} {n:>4} cells")

    write_master_csv(cells,   output_dir / "master_results.csv")
    write_formula_doc(        output_dir / "COMPOSITE_SCORE_FORMULA.md", cells)

    data = build_reports_data(cells)
    write_reports_data_js(data, output_dir / "reports_data.js")

    # Generate PNG plots and recommendations doc
    try:
        import importlib.util
        plot_script = Path(__file__).parent / "plot_benchmark.py"
        spec = importlib.util.spec_from_file_location("plot_benchmark", plot_script)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)                  # type: ignore[union-attr]
        mod.generate_plots(root)
    except Exception as exc:
        print(f"  [warn] Plot generation skipped: {exc}", file=sys.stderr)

    print(f"\n  Reports ready in {output_dir}/")
    print(f"    Open dashboard.html or dashboard_per_dataset.html to view.\n")


if __name__ == "__main__":
    generate()
