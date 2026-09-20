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
from config import cfg

# Module-level aliases so the rest of the file stays readable
DATASET_MAP: dict[int, str] = cfg.datasets.query_count_to_name
COMPOSITE_W: dict[str, float] = cfg.composite.weights

# Strategies whose results actually depend on the embedding model: semantic and
# hybrid embed with it, colbert uses it as the token encoder, adaptive forwards
# it to its embeddings sub-strategy. Cells for other strategies (recency, bm25,
# bm25l, llm_rerank — BM25 fetch + CrossEncoder rerank, no embedding involved)
# may still carry an embedding_model tag from the sweep config, but attributing
# their scores to that embedding would corrupt the embedding ranking.
EMBEDDING_STRATEGIES = {"semantic", "hybrid", "colbert", "adaptive"}

# A dataset is "saturated" when most configs hit near-perfect recall: it can no
# longer discriminate between strategies, so it is excluded from the global
# rankings (its per-dataset section is kept, flagged `saturated: true`).
SATURATION_RECALL   = 0.99
SATURATION_FRACTION = 0.5
SATURATION_MIN_N    = 5

# Minimum cells a strategy needs before it can be declared a dataset "winner".
# A single lucky run must not outrank a strategy averaged over dozens of cells.
WINNER_MIN_N = 3


# ── Data loading ──────────────────────────────────────────────────────────────

def _resolve_dataset_name(row: dict) -> str:
    """Canonical dataset name for one grid row.

    Preference order:
      1. query-count map — measurement-derived and canonical;
      2. the row's explicit dataset_name column (newer grid CSVs record it),
         canonicalized by case-insensitive containment so a gold-file stem
         like 'longmemeval_oracle_gold' groups with 'LongMemEval';
      3. unknown_<nq> (backfilled per study later when possible).
    """
    try:
        nq = int(float(row.get("total_queries") or 0))
    except (TypeError, ValueError):
        nq = 0
    if nq in DATASET_MAP:
        return DATASET_MAP[nq]

    explicit = (row.get("dataset_name") or "").strip()
    if explicit and not explicit.startswith("unknown"):
        low = explicit.lower()
        for canon in DATASET_MAP.values():
            if canon.lower() in low or low in canon.lower():
                return canon
        return explicit

    return f"unknown_{nq}"


def _backfill_unknown_datasets(rows: list[dict], group_label: str) -> None:
    """Assign a dataset name to rows that couldn't be identified by query count.

    Each study runs exactly one dataset, so when a study's identifiable rows all
    belong to a single dataset, its unidentifiable rows (e.g. phase-3 fast-sweep
    cells that recorded total_queries=0) must belong to it too.
    """
    unknown = [r for r in rows if r["dataset_name"].startswith("unknown")]
    if not unknown:
        return
    known = {r["dataset_name"] for r in rows if not r["dataset_name"].startswith("unknown")}
    if len(known) == 1:
        name = known.pop()
        for r in unknown:
            r["dataset_name"] = name
        print(f"  [fix] {group_label}: {len(unknown)} cells lacked a query count → "
              f"assigned '{name}' (a study runs a single dataset)")
    else:
        print(f"  [warn] {group_label}: {len(unknown)} cells have no identifiable dataset "
              f"(candidates: {sorted(known) or 'none'}) — excluded from rankings",
              file=sys.stderr)


def _dedup_fast_sweep(rows: list[dict], group_label: str) -> list[dict]:
    """Collapse duplicated phase-3 fast-sweep measurements.

    Fast-sweep cells produced before the per-type indexing fix recorded one
    full-pool measurement once per memory type, and again wherever the broad
    and fine weight grids overlapped. Rows with identical config AND identical
    metrics are the same measurement; keeping the copies triple-counts hybrid
    in every average. Genuinely distinct runs always differ in some metric,
    so only provable duplicates are dropped.
    """
    seen: dict[tuple, dict] = {}
    kept: list[dict] = []
    dropped = 0
    for r in rows:
        if r.get("study_phase") == "phase3_hybrid_weight":
            key = (r.get("source_study"), r.get("retrieval_strategy"),
                   r.get("embedding_model"), r.get("bm25_weight"),
                   r.get("decay_policy"), r.get("lambda"),
                   r.get("recall_at_k"), r.get("precision_at_k"),
                   r.get("mrr"), r.get("ndcg"))
            if key in seen:
                # Identical metrics under a DIFFERENT memory type ⇒ the old
                # full-pool bug: relabel the kept row so the per-type tag
                # doesn't misattribute it. Same memory type (e.g. broad/fine
                # grids sharing a weight) is just a recompute — keep the label.
                if seen[key].get("memory_type") != r.get("memory_type"):
                    seen[key]["memory_type"] = "all"
                dropped += 1
                continue
            seen[key] = r
        kept.append(r)
    if dropped:
        print(f"  [fix] {group_label}: dropped {dropped} duplicated phase-3 "
              f"fast-sweep rows (same measurement recorded twice — legacy "
              f"per-memory-type copies or broad/fine grid overlap)")
    return kept


def load_all_cells(output_dir: Path) -> list[dict]:
    cells: list[dict] = []
    for study_dir in sorted(output_dir.iterdir()):
        if not study_dir.is_dir() or not study_dir.name.startswith("study_"):
            continue
        study_cells: list[dict] = []
        for csv_path in study_dir.glob("*_grid.csv"):
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("success", "True").lower() != "true":
                            continue
                        row["dataset_name"]  = _resolve_dataset_name(row)
                        row["source_study"]  = study_dir.name
                        study_cells.append(row)
            except Exception as exc:
                print(f"  [warn] {csv_path.name}: {exc}", file=sys.stderr)
        _backfill_unknown_datasets(study_cells, study_dir.name)
        cells.extend(_dedup_fast_sweep(study_cells, study_dir.name))
    return cells


def _warn_if_dropping_datasets(master_path: Path, new_datasets: set[str]) -> None:
    """A directory scan only sees the study_* dirs on THIS machine. If the
    existing master covers datasets the scan can't see (e.g. it was rebuilt
    from another machine's results via --from-master), overwriting it silently
    discards them — warn loudly so the loss is a choice, not an accident."""
    if not master_path.exists():
        return
    try:
        with open(master_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            old = {row.get("dataset_name", "") for row in reader}
    except Exception:
        return
    missing = sorted(d for d in old
                     if d and not d.startswith("unknown") and d not in new_datasets)
    if missing:
        print(f"  [WARNING] The existing master_results.csv covers dataset(s) this scan "
              f"did not find: {', '.join(missing)}. Overwriting will DROP them. If that "
              f"master was built from another machine's results, regenerate with "
              f"--from-master instead.", file=sys.stderr)


def load_cells_from_master(master_path: Path) -> list[dict]:
    """Load cells from an existing master_results.csv (e.g. produced on another
    machine) so its reports can be rebuilt locally with corrected attribution.

    Comment lines (#) are skipped; unknown dataset names are re-derived from the
    query count and then backfilled per source study.
    """
    from collections import defaultdict as _dd
    groups: dict[str, list[dict]] = _dd(list)
    with open(master_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            if (row.get("success") or "True").lower() != "true":
                continue
            row["dataset_name"] = _resolve_dataset_name(row)
            row["source_study"] = row.get("source_study") or "master"
            groups[row["source_study"]].append(row)
    cells: list[dict] = []
    for study, rows in sorted(groups.items()):
        _backfill_unknown_datasets(rows, study)
        cells.extend(_dedup_fast_sweep(rows, study))
    return cells


# ── Composite math ────────────────────────────────────────────────────────────

def _gate(r: float) -> float:
    return 1.0 if r >= cfg.composite.recall_gate else 0.0

def composite(r: float, p: float, m: float, t: float) -> float:
    """Composite score matching MatrixRunResult.composite_score().

    For non-temporal datasets t=0.0, so we renormalize by the active weight
    sum (0.85) to keep the composite in [0,1]. Without renormalization the
    max achievable composite would be 0.85 instead of 1.0, making the
    master_results.csv 17.6% lower than the leaderboard rankings.
    """
    g = _gate(r)
    raw = g * (COMPOSITE_W["recall"] * r + COMPOSITE_W["precision"] * p
               + COMPOSITE_W["mrr"] * m + COMPOSITE_W["temporal"] * t)
    # Renormalize: exclude the temporal weight when temporal_accuracy == 0
    active_w = (COMPOSITE_W["recall"] + COMPOSITE_W["precision"] + COMPOSITE_W["mrr"]
                + (COMPOSITE_W["temporal"] if t > 0 else 0.0))
    return raw / active_w if active_w > 0 else 0.0


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
            "#   composite = gate × (w_R×R@10 + w_P×P@10 + w_M×MRR + w_T×TA) / active_weight_sum",
            "#   active_weight_sum = 0.85 when TA=0 (non-temporal), 1.00 when TA>0",
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

            active_w = (COMPOSITE_W["recall"] + COMPOSITE_W["precision"]
                        + COMPOSITE_W["mrr"]
                        + (COMPOSITE_W["temporal"] if t > 0 else 0.0))
            cell["recall_gate"]               = f"{g:.1f}"
            cell["w_recall"]                  = f"{COMPOSITE_W['recall']    * r * g:.6f}"
            cell["w_precision"]               = f"{COMPOSITE_W['precision'] * p * g:.6f}"
            cell["w_mrr"]                     = f"{COMPOSITE_W['mrr']       * m * g:.6f}"
            cell["w_temporal"]                = f"{COMPOSITE_W['temporal']  * t * g:.6f}"
            cell["composite_score_computed"]  = f"{c:.6f}"
            # The divisor must appear in the string, otherwise the printed
            # arithmetic doesn't reproduce the result for TA=0 rows
            cell["composite_formula"] = (
                f"gate={g:.0f} × (0.40×{r:.4f} + 0.25×{p:.4f} + 0.20×{m:.4f} + 0.15×{t:.4f})"
                f" / {active_w:.2f} = {c:.6f}"
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
        rows = [r for r in rows if not _is_fullpool(r)] or rows
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
    for ds, _strat, _embed, _bm25w, _decay, r10, p10, mrr, ta, comp, _n in results:
        gate = 1.0 if r10 >= cfg.composite.recall_gate else 0.0
        active_w = (COMPOSITE_W["recall"] + COMPOSITE_W["precision"] + COMPOSITE_W["mrr"]
                    + (COMPOSITE_W["temporal"] if ta > 0 else 0.0))
        lines.append(
            f"```\n{ds:<14} = {gate:.0f} × "
            f"(0.40×{r10:.4f} + 0.25×{p10:.4f} + 0.20×{mrr:.4f} + 0.15×{ta:.4f})"
            f" / {active_w:.2f} = {comp:.4f}\n```"
        )

    lines += [
        "",
        "### Dataset cell counts",
        "",
        "| Dataset | Cells benchmarked | Best strategy | Best Recall@10 |",
        "|---------|------------------|---------------|----------------|",
    ]
    for ds, _strat, _embed, _bm25w, _decay, r10, _p10, _mrr, _ta, _comp, _n in results:
        lines.append(f"| {ds} | {_n} | {_strat} | {r10:.4f} |")

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
    # Fast-sweep cells never vary decay (always 'none') and were measured under
    # a different condition, so counting them inflates the 'none' bucket against
    # the per-store decay-sweep cells — compare only comparable measurements.
    subset = [r for r in subset if r.get("study_phase") != "phase3_hybrid_weight"]
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

def _is_saturated(rows: list[dict]) -> bool:
    if len(rows) < SATURATION_MIN_N:
        return False
    near_perfect = sum(1 for r in rows if float(r.get("recall_at_k", 0)) >= SATURATION_RECALL)
    return near_perfect / len(rows) >= SATURATION_FRACTION


def _is_fullpool(row: dict) -> bool:
    """Cells measured over the full memory pool instead of one per-type store.

    Pre-fix fast-sweep cells (identifiable by their missing query counts, or by
    the 'all' label the de-dup assigns) retrieved from all memory types at once
    — a different, easier condition that cannot be scored against per-store
    cells."""
    if row.get("memory_type") == "all":
        return True
    try:
        nq = int(float(row.get("total_queries") or 0))
    except (TypeError, ValueError):
        nq = 0
    return row.get("study_phase") == "phase3_hybrid_weight" and nq == 0


def _is_ranking_comparable(row: dict) -> bool:
    """Cells usable when ranking strategies or embeddings against each other.

    Tuning sweeps measure ONE strategy across many deliberately sub-optimal
    variants; averaging them into a head-to-head ranking is unfair in both
    directions (phase 3 sweeps only hybrid, phase 4 sweeps mostly-degraded
    decay settings of the winning strategy). Rankings therefore use the
    comparison phases plus phase-4 cells at default (no-decay) settings, and
    never full-pool cells."""
    if _is_fullpool(row):
        return False
    phase = row.get("study_phase", "")
    if phase == "phase3_hybrid_weight":
        return False
    return not (phase.startswith("phase4") and row.get("decay_policy") != "none")


def build_reports_data(cells: list[dict]) -> dict:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_ds[c["dataset_name"]].append(c)

    # Config display order first, then any datasets present but not listed there.
    ds_order = [d for d in cfg.datasets.display_order if d in by_ds]
    ds_order += sorted(d for d in by_ds
                       if d not in ds_order and not d.startswith("unknown"))

    # Datasets that can no longer discriminate between configs are kept in the
    # per-dataset section (flagged) but excluded from every global ranking —
    # otherwise a trivially-easy dataset dictates the "overall best" config.
    saturated = {ds for ds, rows in by_ds.items() if _is_saturated(rows)}
    unattributed = {ds for ds in by_ds if ds.startswith("unknown")}
    for ds in sorted(saturated):
        print(f"  [warn] {ds}: {len(by_ds[ds])} cells, "
              f"≥{SATURATION_FRACTION:.0%} at recall ≥ {SATURATION_RECALL} — "
              f"saturated, excluded from global rankings", file=sys.stderr)

    ranked_cells = [c for c in cells
                    if c["dataset_name"] not in saturated
                    and c["dataset_name"] not in unattributed]
    # Head-to-head rankings additionally require condition-comparable cells
    # (per-store, not tuning-sweep variants) — see _is_ranking_comparable.
    comparable_cells = [c for c in ranked_cells if _is_ranking_comparable(c)]

    # ── global strategy ranking (non-saturated, attributed datasets) ──────────
    global_strat = _agg_strategy(comparable_cells)

    # ── global embedding ranking ──────────────────────────────────────────────
    # Only cells whose strategy actually consults the embedding model count.
    by_embed: dict[str, list] = defaultdict(list)
    for c in comparable_cells:
        if c.get("retrieval_strategy") not in EMBEDDING_STRATEGIES:
            continue
        em = c.get("embedding_model", "")
        if em and em != "none":
            by_embed[em].append(float(c.get("recall_at_k", 0)))
    embed_ranking = sorted(
        [{"name": k.split("/")[-1], "full": k, "n": len(v), "avg": round(_avg(v), 4)}
         for k, v in by_embed.items()],
        key=lambda x: -x["avg"]
    )[:8]

    # ── global decay ranking ──────────────────────────────────────────────────
    decay_ranking = _agg_decay(ranked_cells)

    # ── per-dataset records ───────────────────────────────────────────────────
    datasets = []
    for ds_name in ds_order:
        rows = by_ds.get(ds_name, [])
        if not rows:
            continue

        cmp_rows = [r for r in rows if _is_ranking_comparable(r)] or rows
        strats = _agg_strategy(cmp_rows)
        # Winner needs a minimum sample size — n=1 flukes must not outrank a
        # strategy averaged over dozens of cells. Fall back if nothing qualifies.
        eligible = [s for s in strats if s["n"] >= WINNER_MIN_N] or strats
        winner = eligible[0] if eligible else {}

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
            "saturated":      ds_name in saturated,
            "winner":         top_strat_name,
            "winnerAvg":      winner.get("avg", 0),
            "winnerBest":     winner.get("best", 0),
            "strategies":     strats,
            "decay":          decay_rows,
            # Best config may come from any per-store cell (incl. tuned sweep
            # winners) but never from a full-pool cell — the recommendation
            # must be reproducible under the standard condition.
            "bestConfig":     _best_config([r for r in rows if not _is_fullpool(r)] or rows),
        })

    # ── per-dataset recall chart (for main dashboard) ─────────────────────────
    ds_recall = [
        {"name": d["label"], "strat": d["winner"],
         "recall": d["winnerAvg"], "best": d["winnerBest"]}
        for d in datasets if not d["saturated"]
    ]

    # ── overall best (never from a saturated/unattributed dataset or a
    #    full-pool cell — must be reproducible under the standard condition) ───
    overall_best = _best_config([c for c in ranked_cells if not _is_fullpool(c)])

    return {
        "generatedAt":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "totalCells":    len(cells),
        "rankedCells":   len(ranked_cells),
        "excludedFromGlobal": {
            "saturated":    sorted(saturated),
            "unattributed": sorted(unattributed),
        },
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

def generate(project_root: Path | None = None, skip_plots: bool = False,
             master_csv: Path | None = None) -> None:
    # project_root param kept for backward compat with study_runner hook
    output_dir = cfg.reporting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nMemTuner Report Generator")
    if master_csv:
        print(f"  Rebuilding from {master_csv} ...")
        cells = load_cells_from_master(master_csv)
    else:
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

    if not master_csv:
        _warn_if_dropping_datasets(output_dir / "master_results.csv", set(ds_counts))

    write_master_csv(cells,   output_dir / "master_results.csv")
    write_formula_doc(        output_dir / "COMPOSITE_SCORE_FORMULA.md", cells)

    data = build_reports_data(cells)
    write_reports_data_js(data, output_dir / "reports_data.js")

    # Generate PNG plots and recommendations doc
    if skip_plots:
        print("  [--no-plots] Skipping PNG generation.")
    else:
        try:
            import importlib.util
            _proj_root = project_root or Path(__file__).resolve().parent.parent
            plot_script = Path(__file__).parent / "plot_benchmark.py"
            spec = importlib.util.spec_from_file_location("plot_benchmark", plot_script)
            mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)                  # type: ignore[union-attr]
            mod.generate_plots(_proj_root, master_csv=master_csv)
        except Exception as exc:
            print(f"  [warn] Plot generation failed: {exc}", file=sys.stderr)
            print("         Run with --no-plots to skip on headless machines.", file=sys.stderr)

    print(f"\n  Reports ready in {output_dir}/")
    print("    Open dashboard.html or dashboard_per_dataset.html to view.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate MemTuner report artefacts.")
    parser.add_argument("--from-master", type=Path, default=None, metavar="CSV",
                        help="Rebuild reports from an existing master_results.csv "
                             "(e.g. produced on another machine) instead of scanning "
                             "study_* directories.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation.")
    args = parser.parse_args()
    generate(skip_plots=args.no_plots, master_csv=args.from_master)
