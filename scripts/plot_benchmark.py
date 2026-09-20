#!/usr/bin/env python3
"""MemTuner — Publication-quality plot generator.

Generates PNG figures for every dataset and cross-dataset comparisons:

  data/output/plots/
    {DS}_01_strategy_comparison.png   — avg vs best recall per strategy
    {DS}_02_decay_sweep.png           — recall & MRR across decay policies
    {DS}_03_recall_curve.png          — P@1 vs R@10 per strategy (measured points only)
    {DS}_04_bm25_sweep.png            — recall vs BM25 weight (hybrid)
    {DS}_05_memory_type.png           — composite & recall per memory type
    {DS}_06_best_config_card.png      — recommended configuration card
    cross_01_strategy_bars.png        — strategy × dataset recall bars
    cross_02_decay_lines.png          — decay × dataset lines
    cross_03_composite_breakdown.png  — stacked composite weight bars
    cross_04_best_configs.png         — best config per dataset summary

Run manually:
    python scripts/plot_benchmark.py
    python scripts/plot_benchmark.py --dpi 300   # for print/publication

Auto-called by generate_reports.py after every benchmark run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display needed
# Load config — YAML + .env overrides
import sys as _sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import cfg

# ── Runtime constants resolved from config ─────────────────────────────────────
C               = cfg.colors.palette           # name → hex
STRATEGY_COLORS = cfg.colors.strategy_map      # strategy → hex
DECAY_COLORS    = cfg.colors.decay_map         # decay → hex
MEMORY_COLORS   = {m: cfg.colors.for_memory(m) for m in ("episodic", "semantic", "preference")}

DATASET_ORDER   = cfg.datasets.display_order
DATASET_MAP     = cfg.datasets.query_count_to_name
COMPOSITE_W     = cfg.composite.weights

# ── Style ─────────────────────────────────────────────────────────────────────

def _setup_style(dpi: int | None = None) -> None:
    dpi = dpi or cfg.reporting.plot_dpi
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,
        "axes.linewidth":       0.6,
        "axes.grid":            True,
        "grid.color":           "#e1e0d9",
        "grid.linewidth":       0.5,
        "grid.alpha":           0.8,
        "legend.fontsize":      8,
        "legend.framealpha":    0.85,
        "legend.edgecolor":     "#e1e0d9",
        "figure.dpi":           dpi,
        "savefig.dpi":          dpi,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
    })

def _formula_footer(fig, extra: str = "") -> None:
    """Composite formula as a figure footer — never collides with legends/bars."""
    text = cfg.composite.formula_str + (f"     {extra}" if extra else "")
    fig.text(0.5, -0.015, text, ha="center", va="top", fontsize=6.5, color=C["gray"])

def _remove_stale(out_dir: Path, filename: str) -> None:
    """A skipped plot must not leave a previous run's PNG behind pretending to
    be current — delete any stale file occupying this slot."""
    stale = out_dir / filename
    if stale.exists():
        stale.unlink()
        print(f"  −  removed stale {filename} (not applicable to current data)")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  {path.relative_to(path.parent.parent)}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(output_dir: Path, master_csv: Path | None = None) -> dict[str, list[dict]]:
    """Return {dataset_name: [row_dicts]} for all successful cells.

    Delegates to generate_reports' loaders so dataset attribution (query-count
    map + per-study backfill of missing counts) and fast-sweep de-duplication
    stay identical between the plots and the report artefacts.
    """
    from generate_reports import load_all_cells, load_cells_from_master
    cells = load_cells_from_master(master_csv) if master_csv else load_all_cells(output_dir)
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in cells:
        if not row["dataset_name"].startswith("unknown"):
            by_ds[row["dataset_name"]].append(row)
    return by_ds


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def _active_weight(ta: float) -> float:
    """Renormalization divisor: the temporal weight only counts when TA > 0."""
    return (COMPOSITE_W["recall"] + COMPOSITE_W["precision"] + COMPOSITE_W["mrr"]
            + (COMPOSITE_W["temporal"] if ta > 0 else 0.0))


def _composite(r: dict) -> float:
    """Composite score — must match MatrixRunResult.composite_score() and
    generate_reports.composite() (renormalized by the active weight sum), so
    every artefact reports the same number for the same cell."""
    rc = _f(r, "recall_at_k")
    if rc < cfg.composite.recall_gate:
        return 0.0
    ta  = _f(r, "temporal_accuracy")
    raw = (COMPOSITE_W["recall"]    * rc
         + COMPOSITE_W["precision"] * _f(r, "precision_at_k")
         + COMPOSITE_W["mrr"]       * _f(r, "mrr")
         + COMPOSITE_W["temporal"]  * ta)
    return raw / _active_weight(ta)


def _agg(rows: list[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(_f(r, "recall_at_k"))
    return out


def _ranking_comparable(rows: list[dict]) -> list[dict]:
    """Cells usable in head-to-head strategy/embedding rankings (per-store,
    not tuning-sweep variants) — same rule as generate_reports. Falls back to
    all rows when nothing qualifies so a plot is never silently empty."""
    from generate_reports import _is_ranking_comparable
    return [r for r in rows if _is_ranking_comparable(r)]


def _store_only(rows: list[dict]) -> list[dict]:
    """Rows measured per-store; full-pool fast-sweep cells are excluded so a
    'best config' is always reproducible under the standard condition."""
    from generate_reports import _is_fullpool
    return [r for r in rows if not _is_fullpool(r)] or rows


def _decay_comparable(rows: list[dict]) -> list[dict]:
    """Rows usable in a decay comparison. Fast-sweep cells never vary decay
    (always 'none') and were measured under a different condition, so they
    would inflate the 'none' bucket against per-store decay-sweep cells."""
    return [r for r in rows if r.get("study_phase") != "phase3_hybrid_weight"]


def _final_test_rows(rows: list[dict]) -> list[dict]:
    """Return final, untouched evaluations; tuning cells never qualify."""
    return [r for r in rows if r.get("study_phase") == "final_test"]


def _equal_replication_decay(rows: list[dict]) -> dict[str, list[dict]]:
    """Return decay groups only when every policy has equal repeated evidence."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in _decay_comparable(rows):
        groups[row["decay_policy"]].append(row)
    if len(groups) < 2:
        return {}
    counts = {len(group) for group in groups.values()}
    return dict(groups) if len(counts) == 1 and next(iter(counts)) >= 2 else {}


def _saturated(rows: list[dict]) -> bool:
    """Same rule as generate_reports: when most configs hit near-perfect recall,
    the dataset cannot discriminate between configurations."""
    from generate_reports import _is_saturated
    return _is_saturated(rows)


def _ds_label(ds_name: str, rows: list[dict]) -> str:
    """Dataset display name, marked when the benchmark is saturated."""
    return f"{ds_name} (saturated)" if _saturated(rows) else ds_name


_PER_DATASET_SLOTS = (
    "01_strategy_comparison", "02_decay_sweep", "03_recall_curve",
    "04_bm25_sweep", "05_memory_type", "06_best_config_card",
)


def _skip_if_saturated(ds_name: str, rows: list[dict], out_dir: Path) -> bool:
    """Saturated datasets get NO per-dataset charts: their numbers do not
    measure retrieval quality, so publishing charts of them (even watermarked)
    invites misquotation. A missing chart is the established convention for
    "deliberately not applicable" — stale copies from earlier runs are removed.
    Cross-dataset charts still show the dataset, hatched and labeled
    "(saturated)". The flag is data-driven: re-running the dataset with fixed
    ground truth restores its charts automatically."""
    if not _saturated(rows):
        return False
    print(f"  [skip] {ds_name}: saturated dataset — per-dataset charts not generated")
    for slot in _PER_DATASET_SLOTS:
        _remove_stale(out_dir, f"{ds_name}_{slot}.png")
    return True


# ── 1. Strategy comparison (avg vs best) ─────────────────────────────────────

def plot_strategy_comparison(ds_name: str, rows: list[dict], out_dir: Path, dpi: int) -> None:
    by_strat: dict[str, list] = defaultdict(list)
    for r in _ranking_comparable(rows):
        by_strat[r["retrieval_strategy"]].append(r)

    if not by_strat:
        _remove_stale(out_dir, f"{ds_name}_01_strategy_comparison.png")
        return

    records = []
    for s, rs in by_strat.items():
        r10s = [_f(r, "recall_at_k") for r in rs]
        p1s  = [_f(r, "precision_at_1") for r in rs]
        mrrs = [_f(r, "mrr") for r in rs]
        records.append(dict(name=s, avg=np.mean(r10s), std=np.std(r10s, ddof=1) if len(r10s) > 1 else 0.0,
                    p1=np.mean(p1s), mrr=np.mean(mrrs), n=len(rs)))
    records.sort(key=lambda x: -x["avg"])

    names = [r["name"] for r in records]
    avgs  = [r["avg"]  for r in records]
    stds  = [r["std"]  for r in records]
    p1s   = [r["p1"]   for r in records]
    y     = np.arange(len(names))
    h     = 0.28

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(names) + 1.2)))
    ax.set_title(f"{ds_name} — Head-to-head strategy comparison", pad=8)

    ax.barh(y + h/2, avgs, h, xerr=stds,
            color=[STRATEGY_COLORS.get(n, C["blue"]) for n in names],
            error_kw={"capsize": 3, "elinewidth": 0.8}, label="Mean Recall@10 ± 1 SD")
    # P@1 bar (below)
    ax.barh(y - h/2, p1s,   h,   color=C["orange"], alpha=0.75, label="Avg Precision@1")

    for index, record in enumerate(records):
        ax.text(max(record["avg"] + record["std"], record["p1"]) + 0.01, index + h / 2,
                f"n={record['n']}", va="center", fontsize=7, color=C["gray"])

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Score")
    ax.set_xlim(0, min(1.0, max(avg + std for avg, std in zip(avgs, stds)) * 1.22))
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", alpha=0)

    legend_patches = [
        mpatches.Patch(color=C["blue"], label="Mean Recall@10 ± 1 SD"),
        mpatches.Patch(color=C["orange"], alpha=0.75, label="Avg Precision@1"),
    ]
    ax.legend(handles=legend_patches, loc="center left",
              bbox_to_anchor=(1.02, 0.12), fontsize=7.5)

    fig.tight_layout()
    _save(fig, out_dir / f"{ds_name}_01_strategy_comparison.png")


# ── 2. Decay policy sweep ─────────────────────────────────────────────────────

def plot_decay_sweep(ds_name: str, rows: list[dict], out_dir: Path, dpi: int) -> None:
    # Use the best strategy (by avg recall)
    by_strat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_strat[r["retrieval_strategy"]].append(r)
    top_strat = max(by_strat, key=lambda s: np.mean([_f(r, "recall_at_k") for r in by_strat[s]]))
    by_decay = _equal_replication_decay(by_strat[top_strat])
    if not by_decay:
        _remove_stale(out_dir, f"{ds_name}_02_decay_sweep.png")
        print(f"  [skip] {ds_name}: decay policies lack equal repeated measurements")
        return

    records = []
    for d, rs in by_decay.items():
        r10s = [_f(r, "recall_at_k")    for r in rs]
        mrrs = [_f(r, "mrr")            for r in rs]
        p1s  = [_f(r, "precision_at_1") for r in rs]
        records.append(dict(name=d, r10=np.mean(r10s), mrr=np.mean(mrrs), p1=np.mean(p1s), n=len(rs)))
    records.sort(key=lambda x: -x["r10"])

    names = [r["name"] for r in records]
    r10s  = [r["r10"]  for r in records]
    mrrs  = [r["mrr"]  for r in records]
    y = np.arange(len(names))
    h = 0.35

    fig, ax = plt.subplots(figsize=(6.5, max(2.8, 0.55 * len(names) + 1.4)))
    ax.set_title(f"{ds_name} — Decay policy impact ({top_strat} strategy)", pad=8)

    colors = [DECAY_COLORS.get(n, C["gray"]) for n in names]
    ax.barh(y + h/2, r10s, h, color=colors, label="Avg Recall@10")
    ax.barh(y - h/2, mrrs, h, color=colors, alpha=0.5, label="Avg MRR")

    ax.text(r10s[0] + 0.005, y[0] + h/2, f"{r10s[0]:.4f}", va="center",
            fontsize=7.5, fontweight="bold", color=colors[0])

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Score")
    # Both bar families must fit — MRR often exceeds recall
    ax.set_xlim(0, min(1.0, max(max(r10s), max(mrrs)) * 1.25))
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", alpha=0)

    legend_patches = [
        mpatches.Patch(color=C["gray"], label="Recall@10 (solid)"),
        mpatches.Patch(color=C["gray"], alpha=0.5, label="MRR (faded)"),
    ]
    ax.legend(handles=legend_patches, loc="center left",
              bbox_to_anchor=(1.02, 0.12), fontsize=7.5)

    n_note = ", ".join(f"{r['name']}(n={r['n']})" for r in records)
    ax.set_xlabel(f"Score  —  runs: {n_note}", fontsize=7.5)

    fig.tight_layout()
    _save(fig, out_dir / f"{ds_name}_02_decay_sweep.png")


# ── 3. Recall@k curve ─────────────────────────────────────────────────────────

def plot_recall_curve(ds_name: str, rows: list[dict], out_dir: Path, dpi: int) -> None:
    by_strat: dict[str, list] = defaultdict(list)
    for r in _ranking_comparable(rows):
        by_strat[r["retrieval_strategy"]].append(r)
    top4 = sorted(by_strat, key=lambda s: -np.mean([_f(r, "recall_at_k") for r in by_strat[s]]))[:4]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.set_title(f"{ds_name} — Ranking quality  (measured at k=1 and k=10)", pad=8)

    ks = [1, 10]
    colors = [STRATEGY_COLORS.get(s, C["gray"]) for s in top4]

    for s, col in zip(top4, colors):
        rs = by_strat[s]
        r1  = np.mean([_f(r, "precision_at_1") for r in rs])
        r10 = np.mean([_f(r, "recall_at_k")    for r in rs])
        # Dashed connector: the two points are different metrics (P@1 vs R@10)
        # at measured cutoffs only — no interpolated points in between.
        ax.plot(ks, [r1, r10], "o--", color=col, linewidth=1.5, markersize=6,
                label=f"{s}  (P@1={r1:.3f}, R@10={r10:.3f})")

    ax.set_xticks(ks)
    ax.set_xticklabels(["k=1\n(Precision@1)", "k=10\n(Recall@10)"])
    ax.set_xlim(0, 11)
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    # Legend outside the axes: with only two x positions the data can occupy
    # any corner (P@1 above or below R@10), so no inside placement is safe
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)

    multi = any(_f(r, "precision_at_1") > _f(r, "recall_at_k") + 0.05 for r in rows[:20])
    if multi:
        # Below the tick labels — any inside placement collides with data
        # whenever the top scores approach the axis ceiling
        ax.annotate("⚠ Multi-relevant: P@1 may exceed R@10", xy=(0.5, -0.24),
                    xycoords="axes fraction", ha="center", va="top",
                    fontsize=7, color=C["orange"],
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff8f0", ec=C["orange"], alpha=0.8))

    fig.tight_layout()
    _save(fig, out_dir / f"{ds_name}_03_recall_curve.png")


# ── 4. BM25 weight sweep ──────────────────────────────────────────────────────

def plot_bm25_sweep(ds_name: str, rows: list[dict], out_dir: Path, dpi: int) -> None:
    hybrid = [r for r in rows if r["retrieval_strategy"] == "hybrid"]
    if not hybrid:
        _remove_stale(out_dir, f"{ds_name}_04_bm25_sweep.png")
        return

    # Only the dedicated weight-sweep cells are comparable point-to-point:
    # hybrid cells from other phases run per-store retrieval at one default
    # weight, and mixing them in fakes a dip at that weight.
    sweep = [r for r in hybrid if r.get("study_phase") == "phase3_hybrid_weight"]
    mixed_phases = not sweep
    if sweep:
        hybrid = sweep

    by_w: dict[float, list] = defaultdict(list)
    for r in hybrid:
        w = round(_f(r, "bm25_weight"), 2)
        by_w[w].append(_f(r, "recall_at_k"))

    if len(by_w) < 3:
        _remove_stale(out_dir, f"{ds_name}_04_bm25_sweep.png")
        return

    ws   = sorted(by_w)
    avgs = [np.mean(by_w[w]) for w in ws]
    ns   = [len(by_w[w]) for w in ws]
    peak_idx = int(np.argmax(avgs))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_title(f"{ds_name} — BM25 weight sweep  (hybrid strategy)", pad=8)

    # Area fill
    ax.fill_between(ws, avgs, alpha=0.10, color=C["blue"])
    ax.plot(ws, avgs, "o-", color=C["blue"], linewidth=1.8, markersize=5, label="Avg Recall@10")

    # Highlight peak
    ax.plot(ws[peak_idx], avgs[peak_idx], "o", color=C["orange"], markersize=8, zorder=5,
            label=f"Optimal BM25w={ws[peak_idx]:.2f}  R@10={avgs[peak_idx]:.3f}")
    # Keep the annotation clear of the marker and inside the axes even when
    # the peak sits near either end of the weight range
    on_right = ws[peak_idx] > 0.7
    ax.annotate(f"BM25w={ws[peak_idx]:.2f}\nR@10={avgs[peak_idx]:.3f}",
                xy=(ws[peak_idx], avgs[peak_idx]),
                xytext=(ws[peak_idx] + (-0.06 if on_right else 0.06),
                        avgs[peak_idx] * 0.80),
                ha="right" if on_right else "left",
                fontsize=7.5, color=C["orange"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1))

    n_lo, n_hi = min(ns), max(ns)
    n_note = f"n={n_lo}" if n_lo == n_hi else f"n={n_lo}–{n_hi}"
    src_note = ("mixed study phases — points may not be comparable"
                if mixed_phases else "phase-3 weight sweep")
    ax.set_xlabel(f"BM25 weight  (semantic = 1 – BM25 weight)   —   {src_note}, {n_note} runs/point")
    ax.set_ylabel("Avg Recall@10")
    ax.set_xlim(min(ws) - 0.02, max(ws) + 0.02)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y")

    fig.tight_layout()
    _save(fig, out_dir / f"{ds_name}_04_bm25_sweep.png")


# ── 5. Memory type breakdown ──────────────────────────────────────────────────

def plot_memory_type(ds_name: str, rows: list[dict], out_dir: Path, dpi: int) -> None:
    by_mem: dict[str, list] = defaultdict(list)
    for r in rows:
        by_mem[r["memory_type"]].append(r)

    if len(by_mem) < 2:
        _remove_stale(out_dir, f"{ds_name}_05_memory_type.png")
        return

    records = []
    for m, rs in by_mem.items():
        r10s  = [_f(r, "recall_at_k") for r in rs]
        comps = [_composite(r) for r in rs]
        p1s   = [_f(r, "precision_at_1") for r in rs]
        records.append(dict(name=m, r10=np.mean(r10s), comp=np.mean(comps),
                            p1=np.mean(p1s), n=len(rs)))
    records.sort(key=lambda x: -x["comp"])

    names = [r["name"] for r in records]
    comps = [r["comp"] for r in records]
    r10s  = [r["r10"]  for r in records]
    p1s   = [r["p1"]   for r in records]
    y = np.arange(len(names))
    h = 0.28

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.6 * len(names) + 1.2)))
    ax.set_title(f"{ds_name} — Memory type comparison", pad=8)

    colors = [MEMORY_COLORS.get(n, C["gray"]) for n in names]
    ax.barh(y + h/2, comps, h, color=colors, label="Composite score")
    ax.barh(y - h/2, r10s,  h, color=colors, alpha=0.45, label="Recall@10")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (n={r['n']})" for n, r in zip(names, records)])
    ax.set_xlabel("Score")
    ax.set_xlim(0, max(max(comps), max(r10s)) * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", alpha=0)

    legend_patches = [
        mpatches.Patch(color=C["gray"],          label="Composite score (solid)"),
        mpatches.Patch(color=C["gray"], alpha=0.45, label="Recall@10 (faded)"),
    ]
    ax.legend(handles=legend_patches, loc="center left",
              bbox_to_anchor=(1.02, 0.12), fontsize=7.5)

    fig.tight_layout()
    extra = ("'all' = full-pool phase-3 sweep cells (not per-store)"
             if "all" in names else "")
    _formula_footer(fig, extra)
    _save(fig, out_dir / f"{ds_name}_05_memory_type.png")


# ── 6. Best-config recommendation card ───────────────────────────────────────

def plot_best_config_card(ds_name: str, rows: list[dict], out_dir: Path, dpi: int) -> None:
    final_rows = _final_test_rows(rows)
    best = max(final_rows, key=_composite) if final_rows else max(_store_only(rows), key=_composite)
    comp = _composite(best)

    r10  = _f(best, "recall_at_k")
    p10  = _f(best, "precision_at_k")
    mrr  = _f(best, "mrr")
    ta   = _f(best, "temporal_accuracy")
    gate = 1.0 if r10 >= cfg.composite.recall_gate else 0.0

    # Components share the composite's renormalization so the stacked
    # segments sum exactly to the total shown
    aw  = _active_weight(ta)
    w_r = COMPOSITE_W["recall"]    * r10 * gate / aw
    w_p = COMPOSITE_W["precision"] * p10 * gate / aw
    w_m = COMPOSITE_W["mrr"]       * mrr  * gate / aw
    w_t = COMPOSITE_W["temporal"]  * ta   * gate / aw

    strat   = best.get("retrieval_strategy", "")
    embed   = best.get("embedding_model", "").split("/")[-1]
    backend = best.get("embedding_backend", "")
    decay   = best.get("decay_policy", "")
    lam     = _f(best, "lambda")
    bm25w   = _f(best, "bm25_weight")
    mem     = best.get("memory_type", "")
    phase   = best.get("study_phase", "")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    evidence_label = "Final-test result" if final_rows else "Tuning-only candidate — not deployable"
    fig.suptitle(f"{ds_name} — {evidence_label}",
                 fontsize=11, fontweight="bold", y=1.01)

    # LEFT: Configuration details
    ax_cfg = axes[0]
    ax_cfg.axis("off")
    lines = [
        ("Retrieval strategy", strat,   STRATEGY_COLORS.get(strat, C["blue"])),
        ("Embedding model",    embed,   C["blue"]),
        ("Backend",            backend, C["gray"]),
        ("BM25 weight",        f"{bm25w:.2f}  →  semantic {1-bm25w:.2f}", C["orange"]),
        ("Decay policy",       f"{decay}  (λ={lam:.4f})", DECAY_COLORS.get(decay, C["gray"])),
        ("Memory type",        mem,     MEMORY_COLORS.get(mem, C["gray"])),
        ("Study phase",        phase,   C["gray"]),
    ]
    for i, (label, value, color) in enumerate(lines):
        y_pos = 0.92 - i * 0.135
        ax_cfg.text(0.0, y_pos, label + ":",
                    transform=ax_cfg.transAxes, fontsize=8.5, color="#52514e", va="top")
        ax_cfg.text(0.42, y_pos, value,
                    transform=ax_cfg.transAxes, fontsize=8.5, color=color,
                    fontweight="bold" if i <= 1 else "normal", va="top")

    ax_cfg.text(0.0, -0.04,
                f"Composite score: {comp:.4f}   Recall@10: {r10:.4f}   MRR: {mrr:.4f}",
                transform=ax_cfg.transAxes, fontsize=8, color=C["blue"], fontweight="bold")
    if not final_rows:
        ax_cfg.text(0.0, -0.38,
                    "No untouched final-test row was found. Re-run with --final-test-holdout-fraction.",
                    transform=ax_cfg.transAxes, fontsize=7, color="#A33A00", fontweight="bold")

    gate_note = (f"gate = 1.0 (R@10 ≥ {cfg.composite.recall_gate})" if gate == 1.0
                 else f"gate = 0.0 (R@10 < {cfg.composite.recall_gate})")
    # Two lines, left panel only — one long line runs under the right panel's legend
    ax_cfg.text(0.0, -0.16, cfg.composite.formula_str,
                transform=ax_cfg.transAxes, fontsize=6.5, color=C["gray"])
    ax_cfg.text(0.0, -0.26, f"Σw_active = {aw:.2f}    {gate_note}",
                transform=ax_cfg.transAxes, fontsize=6.5, color=C["gray"])

    # RIGHT: Composite score breakdown (stacked horizontal bar)
    ax_bar = axes[1]
    ax_bar.set_title(f"Composite score breakdown  (total = {comp:.4f})", fontsize=9)

    contributions = [
        (f"{COMPOSITE_W['recall']} × R@10 = {w_r:.3f}",   w_r, C["blue"]),
        (f"{COMPOSITE_W['precision']} × P@10 = {w_p:.3f}", w_p, C["orange"]),
        (f"{COMPOSITE_W['mrr']} × MRR  = {w_m:.3f}",       w_m, C["aqua"]),
        (f"{COMPOSITE_W['temporal']} × TA   = {w_t:.3f}",  w_t, C["yellow"]),
    ]

    left = 0.0
    for _label, val, col in contributions:
        ax_bar.barh(0, val, 0.4, left=left, color=col)
        # Segments too narrow for an in-bar label get their value from the legend
        if val > comp * 0.12:
            ax_bar.text(left + val / 2, 0, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left += val

    ax_bar.set_xlim(0, comp * 1.18 if comp > 0 else 1.0)
    ax_bar.set_yticks([])
    ax_bar.grid(axis="x")

    legend_patches = [mpatches.Patch(color=c, label=lbl) for lbl, _, c in contributions]
    ax_bar.legend(handles=legend_patches, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.10), ncol=2)

    fig.tight_layout()
    _save(fig, out_dir / f"{ds_name}_06_best_config_card.png")


# ── Cross-dataset: strategy grouped bar chart ─────────────────────────────────

def plot_cross_strategy_heatmap(by_ds: dict[str, list], out_dir: Path, dpi: int) -> None:
    """Grouped horizontal bar chart: strategy × dataset avg Recall@10."""
    ds_present = [d for d in DATASET_ORDER if d in by_ds]
    strategies = ["hybrid", "semantic", "colbert", "adaptive", "bm25", "bm25l", "recency"]

    # Avg recall per (strategy, dataset)
    recall: dict[str, dict[str, float]] = {s: {} for s in strategies}
    for ds in ds_present:
        by_s: dict[str, list] = defaultdict(list)
        for r in _ranking_comparable(by_ds[ds]):
            by_s[r["retrieval_strategy"]].append(_f(r, "recall_at_k"))
        for s in strategies:
            if s in by_s:
                recall[s][ds] = float(np.mean(by_s[s]))

    # Only strategies with any data, sorted best→worst by overall avg.
    # Saturated datasets (every strategy near-perfect) are excluded from the
    # ordering so they cannot inflate a strategy's rank; their bars stay
    # visible but hatched.
    active = [s for s in strategies if recall[s]]
    sat_ds = {ds for ds in ds_present if _saturated(by_ds[ds])}

    def _rank_key(s: str) -> float:
        vals = [v for ds, v in recall[s].items() if ds not in sat_ds]
        vals = vals or list(recall[s].values())
        return -float(np.mean(vals))

    sorted_strats = sorted(active, key=_rank_key)

    n_strat = len(sorted_strats)
    n_ds    = len(ds_present)
    bar_h   = 0.70 / n_ds
    y       = np.arange(n_strat)
    ds_cols = [C["blue"], C["orange"], C["aqua"], C["yellow"],
               C.get("magenta", "#e87ba4"), C["gray"], "#4a3aa7"][:n_ds]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.60 * n_strat + 1.5)))
    ax.set_title("Cross-dataset — Strategy comparison  (Avg Recall@10 per dataset)", pad=8)

    for di, (ds, col) in enumerate(zip(ds_present, ds_cols)):
        offsets = y + (di - (n_ds - 1) / 2.0) * bar_h
        vals    = [recall[s].get(ds, 0.0) for s in sorted_strats]
        sat     = _saturated(by_ds[ds])
        ax.barh(offsets, vals, bar_h * 0.88, color=col,
                label=_ds_label(ds, by_ds[ds]),
                alpha=0.35 if sat else 0.85,
                hatch="//" if sat else None)
        # Annotate best value for each dataset's top strategy
        best_i = int(np.argmax(vals))
        if vals[best_i] > 0:
            ax.text(vals[best_i] + 0.003, offsets[best_i],
                    f"{vals[best_i]:.3f}", va="center", fontsize=6.5, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_strats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Recall@10")
    ax.set_xlim(0, min(1.0, max(
        recall[s].get(ds, 0) for s in sorted_strats for ds in ds_present) * 1.25))
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    # Legend outside the axes — bars can reach the right edge
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="x")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    _save(fig, out_dir / "cross_01_strategy_bars.png")


# ── Cross-dataset: decay policy line chart ────────────────────────────────────

def plot_cross_decay_heatmap(by_ds: dict[str, list], out_dir: Path, dpi: int) -> None:
    """Grouped bars for balanced decay measurements across categorical datasets."""
    ds_present  = [d for d in DATASET_ORDER if d in by_ds]
    decay_order = ["linear", "logarithmic", "tiered", "exponential", "none"]

    # For each non-saturated dataset, use its top head-to-head strategy only
    # when every decay policy has the same number of repeated measurements.
    data: dict[str, dict[str, tuple[float, float, int]]] = {}
    for ds in ds_present:
        if _saturated(by_ds[ds]):
            continue
        by_s: dict[str, list] = defaultdict(list)
        for r in _ranking_comparable(by_ds[ds]):
            by_s[r["retrieval_strategy"]].append(r)
        if not by_s:
            continue
        top_s = max(by_s, key=lambda s: np.mean([_f(r, "recall_at_k") for r in by_s[s]]))
        # Decay buckets need the strategy's FULL rows — by_s holds only the
        # ranking-comparable subset, which excludes the decay-swept cells.
        top_s_all = [r for r in by_ds[ds] if r["retrieval_strategy"] == top_s]
        by_d = _equal_replication_decay(top_s_all)
        if not by_d:
            print(f"  [skip] {ds}: cross-dataset decay comparison lacks balanced replication")
            continue
        data[ds] = {
            decay: (
                float(np.mean([_f(row, "recall_at_k") for row in decay_rows])),
                float(np.std([_f(row, "recall_at_k") for row in decay_rows], ddof=1)),
                len(decay_rows),
            )
            for decay, decay_rows in by_d.items()
        }

    valid_datasets = [ds for ds in ds_present if ds in data]
    active_decays = [decay for decay in decay_order if any(decay in data[ds] for ds in valid_datasets)]
    if not valid_datasets or not active_decays:
        _remove_stale(out_dir, "cross_02_decay_lines.png")
        return

    decay_cols = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["gray"]]
    x = np.arange(len(valid_datasets))
    width = 0.72 / len(active_decays)

    fig, ax = plt.subplots(figsize=(max(7.5, len(valid_datasets) * 2.2), 4.2))
    ax.set_title("Cross-dataset — Balanced decay comparison (top strategy per dataset)", pad=8)

    for index, (decay, color) in enumerate(zip(active_decays, decay_cols)):
        values = [data[ds].get(decay, (np.nan, np.nan, 0)) for ds in valid_datasets]
        means = [value[0] for value in values]
        stds = [value[1] for value in values]
        offset = (index - (len(active_decays) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=color,
               alpha=0.85, label=decay)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n(n={next(iter(data[ds].values()))[2]})" for ds in valid_datasets],
                       rotation=15, ha="right")
    ax.set_ylabel("Avg Recall@10")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, loc="upper right", title="Decay policy", title_fontsize=7.5)
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    _save(fig, out_dir / "cross_02_decay_lines.png")


# ── Cross-dataset: composite breakdown ────────────────────────────────────────

def plot_composite_breakdown(by_ds: dict[str, list], out_dir: Path, dpi: int) -> None:
    ds_present = [d for d in DATASET_ORDER if d in by_ds]

    data: dict[str, list[float]] = {k: [] for k in ["w_recall","w_prec","w_mrr","w_temp"]}
    best_comps: list[float] = []

    for ds in ds_present:
        best = max(_store_only(by_ds[ds]), key=_composite)
        r10  = _f(best, "recall_at_k")
        p10  = _f(best, "precision_at_k")
        mrr  = _f(best, "mrr")
        ta   = _f(best, "temporal_accuracy")
        gate = 1.0 if r10 >= cfg.composite.recall_gate else 0.0
        aw   = _active_weight(ta)  # segments must sum to the renormalized total
        data["w_recall"].append(COMPOSITE_W["recall"]    * r10 * gate / aw)
        data["w_prec"  ].append(COMPOSITE_W["precision"] * p10 * gate / aw)
        data["w_mrr"   ].append(COMPOSITE_W["mrr"]       * mrr  * gate / aw)
        data["w_temp"  ].append(COMPOSITE_W["temporal"]  * ta   * gate / aw)
        best_comps.append(_composite(best))

    x = np.arange(len(ds_present))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Tuning-candidate composite score breakdown per dataset\n" +
                 cfg.composite.formula_str, pad=8)

    bottoms = np.zeros(len(ds_present))
    labels_colors = [
        (f"{COMPOSITE_W['recall']} × Recall@10",    data["w_recall"], C["blue"]),
        (f"{COMPOSITE_W['precision']} × Precision@10", data["w_prec"],  C["orange"]),
        (f"{COMPOSITE_W['mrr']} × MRR",             data["w_mrr"],    C["aqua"]),
        (f"{COMPOSITE_W['temporal']} × Temporal",   data["w_temp"],   C["yellow"]),
    ]
    for _label, vals, col in labels_colors:
        ax.bar(x, vals, 0.55, bottom=bottoms, color=col, label=_label)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.02:
                ax.text(xi, b + v/2, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms += np.array(vals)

    for xi, c in enumerate(best_comps):
        ax.text(xi, c + 0.012, f"{c:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#0b0b0b")

    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(ds, by_ds[ds]) for ds in ds_present], fontsize=8)
    ax.set_ylabel("Composite score (stacked components)")
    ax.set_ylim(0, max(best_comps) * 1.25)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    if any(_saturated(by_ds[ds]) for ds in ds_present):
        fig.text(0.5, -0.015,
                 "Saturated datasets reach near-perfect recall for most configs — "
                 "their scores do not reflect retrieval quality.",
                 ha="center", va="top", fontsize=6.5, color="#d9534f")
    _save(fig, out_dir / "cross_03_composite_breakdown.png")


# ── Cross-dataset: best-config summary table ──────────────────────────────────

def plot_best_configs_summary(by_ds: dict[str, list], out_dir: Path, dpi: int) -> None:
    ds_present = [d for d in DATASET_ORDER if d in by_ds]

    rows_out = []
    for ds in ds_present:
        best  = max(_store_only(by_ds[ds]), key=_composite)
        comp  = _composite(best)
        r10   = _f(best, "recall_at_k")
        strat = best.get("retrieval_strategy", "")
        embed = best.get("embedding_model", "").split("/")[-1]
        decay = best.get("decay_policy", "")
        bm25w = _f(best, "bm25_weight")
        mem   = best.get("memory_type", "")
        rows_out.append([_ds_label(ds, by_ds[ds]), strat, embed, f"{bm25w:.2f}",
                         decay, mem, f"{r10:.4f}", f"{comp:.4f}"])

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.52 * len(rows_out) + 1.8)))
    ax.axis("off")
    ax.set_title("Final-Test Configuration Results Per Dataset\n"
                 "(tuning candidates shown where final test is absent)\n" +
                 cfg.composite.formula_str,
                 fontsize=10, pad=10)

    cols = ["Dataset", "Strategy", "Embedding", "BM25 w", "Decay", "Memory type",
            "Recall@10", "Composite"]
    table = ax.table(
        cellText=rows_out, colLabels=cols,
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(range(len(cols)))

    # Header styling
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor("#2a78d6")
        cell.set_text_props(color="white", fontweight="bold")

    # Row striping + color the winning strategy
    strat_col_idx = cols.index("Strategy")
    for i, row in enumerate(rows_out):
        bg = "#f9f9f7" if i % 2 == 0 else "#f0efec"
        strat = row[strat_col_idx]
        for j in range(len(cols)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            if j == strat_col_idx:
                scol = STRATEGY_COLORS.get(strat, C["gray"])
                cell.set_text_props(color=scol, fontweight="bold")
            if j == len(cols) - 1:  # Composite
                cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    if any(_saturated(by_ds[ds]) for ds in ds_present):
        fig.text(0.5, 0.16,
                 "⚠ Rows marked (saturated): nearly every config reaches perfect recall on "
                 "that dataset — the row is not a meaningful recommendation.",
                 ha="center", va="top", fontsize=7.5, color="#d9534f")
    _save(fig, out_dir / "cross_04_best_configs.png")


# ── Recommendations markdown ──────────────────────────────────────────────────

def write_recommendations(by_ds: dict[str, list], out_path: Path) -> None:
    lines = [
        "# MemTuner — Per-Dataset Configuration Recommendations",
        "",
        "Generated by `scripts/plot_benchmark.py`. A deployable recommendation requires an **untouched final-test**",
        "measurement. When no final test is present, this document shows a tuning candidate only.",
        "",
        "## Composite score formula",
        "",
        "```",
        cfg.composite.formula_str,
        "```",
        "",
        "Weights rationale: Recall (40%) is the primary signal — missing a relevant memory is the worst",
        "failure. Precision (25%) controls context window waste. MRR (20%) rewards early ranking of relevant",
        "content. Temporal accuracy (15%) penalises answers from the wrong time window.",
        "",
        "---",
        "",
    ]

    for ds_name in DATASET_ORDER:
        rows = by_ds.get(ds_name)
        if not rows:
            continue

        # Saturated dataset: most configs hit near-perfect recall, so its
        # "best config" carries no signal — same rule as generate_reports.py.
        near_perfect = sum(1 for r in rows if _f(r, "recall_at_k") >= 0.99)
        saturated = len(rows) >= 5 and near_perfect / len(rows) >= 0.5

        final_rows = _final_test_rows(rows)
        best  = max(final_rows, key=_composite) if final_rows else max(_store_only(rows), key=_composite)
        comp  = _composite(best)
        r10   = _f(best, "recall_at_k")
        p10   = _f(best, "precision_at_k")
        mrr   = _f(best, "mrr")
        ta    = _f(best, "temporal_accuracy")
        strat = best.get("retrieval_strategy", "")
        embed = best.get("embedding_model", "").split("/")[-1]
        backend = best.get("embedding_backend", "")
        decay = best.get("decay_policy", "")
        lam   = _f(best, "lambda")
        bm25w = _f(best, "bm25_weight")
        sem_w = round(1 - bm25w, 2)
        mem   = best.get("memory_type", "")
        phase = best.get("study_phase", "")

        # Strategy ranking for this dataset (condition-comparable cells only)
        by_s: dict[str, list] = defaultdict(list)
        for r in _ranking_comparable(rows):
            by_s[r["retrieval_strategy"]].append(r)
        strat_rank = sorted(by_s, key=lambda s: -np.mean([_f(r, "recall_at_k") for r in by_s[s]]))

        # Memory type ranking
        by_m: dict[str, list] = defaultdict(list)
        for r in rows:
            by_m[r["memory_type"]].append(r)
        mem_rank = sorted(by_m, key=lambda m: -np.mean([_composite(r) for r in by_m[m]]))

        # Decay ranking (top strategy only, comparable cells only).
        # Drawn from ALL of the strategy's rows — by_s holds only the
        # ranking-comparable subset, which excludes the decay-swept cells.
        top_s_all = [r for r in rows if r["retrieval_strategy"] == strat_rank[0]]
        top_s_rows = _decay_comparable(top_s_all or rows)
        by_d: dict[str, list] = defaultdict(list)
        for r in top_s_rows:
            by_d[r["decay_policy"]].append(r)
        decay_rank = sorted(by_d, key=lambda d: -np.mean([_f(r, "recall_at_k") for r in by_d[d]]))

        lines += [
            f"## {ds_name}",
            "",
        ]
        if not final_rows:
            lines += [
                "> ⚠️ **Tuning-only candidate — not deployable.** No untouched final-test row was found. "
                "Re-run with `--final-test-holdout-fraction` before treating this selection as a recommendation.",
                "",
            ]
        if saturated:
            lines += [
                f"> ⚠️ **Saturated dataset — do not use for tuning.** "
                f"{near_perfect}/{len(rows)} configs reach Recall@10 ≥ 0.99, so this dataset "
                f"cannot discriminate between strategies. Check its ground-truth labels: when "
                f"nearly every candidate memory counts as relevant, retrieval is trivial. "
                f"Excluded from global rankings in reports_data.js.",
                "",
            ]
        lines += [
            f"+ BM25 weight={bm25w:.2f} (semantic={sem_w:.2f}) + `{mem}` memory",
            f"**{'Final-test result' if final_rows else 'Best tuning candidate'}**: `{strat}` + `{embed}` ({backend}) + `{decay}` decay (λ={lam:.4f})",
            f"+ BM25 weight={bm25w:.2f} (semantic={sem_w:.2f}) + `{mem}` memory",
            f"+ BM25 weight={bm25w:.2f} (semantic={sem_w:.2f}) + `{mem}` memory",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Recall@10 | {r10:.4f} |",
            f"| Precision@10 | {p10:.4f} |",
            f"| MRR | {mrr:.4f} |",
            f"| Temporal Accuracy | {ta:.4f} |",
            f"| **Composite score** | **{comp:.4f}** |",
            f"| Study phase | {phase} |",
            f"| Total cells benchmarked | {len(rows)} |",
            "",
            f"**Composite breakdown**: `gate={1 if r10 >= cfg.composite.recall_gate else 0} × ("
            f"{COMPOSITE_W['recall']}×{r10:.4f} + {COMPOSITE_W['precision']}×{p10:.4f}"
            f" + {COMPOSITE_W['mrr']}×{mrr:.4f} + {COMPOSITE_W['temporal']}×{ta:.4f})"
            f" / {_active_weight(ta):.2f} = {comp:.4f}`",
            "",
            "**Strategy ranking** (avg Recall@10): "
            + " > ".join(f"`{s}` ({np.mean([_f(r,'recall_at_k') for r in by_s[s]]):.3f})"
                         for s in strat_rank),
            "",
            "**Best memory type** (avg composite): "
            + " > ".join(f"`{m}` ({np.mean([_composite(r) for r in by_m[m]]):.3f})"
                         for m in mem_rank),
            "",
            f"**Decay ranking** (avg Recall@10, {strat_rank[0]} strategy): "
            + " > ".join(f"`{d}` ({np.mean([_f(r,'recall_at_k') for r in by_d[d]]):.3f})"
                         for d in decay_rank),
            "",
            "---",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  DATASET_RECOMMENDATIONS.md  → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_plots(project_root: Path | None = None, dpi: int = 150,
                   master_csv: Path | None = None) -> None:
    root       = project_root or Path(__file__).resolve().parent.parent
    output_dir = root / "data" / "output"
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _setup_style(dpi)

    print(f"\nMemTuner Plot Generator  (dpi={dpi})")
    print(f"  Output: {plots_dir}")
    print("  Loading data...")

    by_ds = load_all(output_dir, master_csv=master_csv)
    if not by_ds:
        print("  No data found.", file=sys.stderr)
        return

    total = sum(len(v) for v in by_ds.values())
    print(f"  Loaded {total} cells across {len(by_ds)} datasets\n")

    # Per-dataset plots
    for ds_name in DATASET_ORDER:
        rows = by_ds.get(ds_name)
        if not rows:
            continue
        if _skip_if_saturated(ds_name, rows, plots_dir):
            continue
        print(f"  [{ds_name}]  ({len(rows)} cells)")
        plot_strategy_comparison(ds_name, rows, plots_dir, dpi)
        plot_decay_sweep        (ds_name, rows, plots_dir, dpi)
        plot_recall_curve       (ds_name, rows, plots_dir, dpi)
        plot_bm25_sweep         (ds_name, rows, plots_dir, dpi)
        plot_memory_type        (ds_name, rows, plots_dir, dpi)
        plot_best_config_card   (ds_name, rows, plots_dir, dpi)

    # Cross-dataset plots
    print("\n  [Cross-dataset]")
    plot_cross_strategy_heatmap (by_ds, plots_dir, dpi)
    plot_cross_decay_heatmap    (by_ds, plots_dir, dpi)
    plot_composite_breakdown    (by_ds, plots_dir, dpi)
    plot_best_configs_summary   (by_ds, plots_dir, dpi)

    # Recommendations doc
    write_recommendations(by_ds, output_dir / "DATASET_RECOMMENDATIONS.md")

    plot_files = sorted(plots_dir.glob("*.png"))
    print(f"\n  {len(plot_files)} PNG files written to {plots_dir}/")


def load_single_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Load a single grid CSV and return the same {dataset: [rows]} structure.

    Useful when you want to plot just one run:
        python scripts/plot_benchmark.py --csv data/output/study_x/locomo_*.csv

    Applies the same corrections as the multi-run path (dataset backfill for
    rows missing query counts, fast-sweep de-duplication) so a single-CSV plot
    can never disagree with the full report.
    """
    from generate_reports import (
        _backfill_unknown_datasets,
        _dedup_fast_sweep,
        _resolve_dataset_name,
    )
    rows: list[dict] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("success", "True").lower() != "true":
                    continue
                row["dataset_name"] = _resolve_dataset_name(row)
                row["source_study"] = csv_path.parent.name
                rows.append(row)
    except Exception as exc:
        print(f"  [warn] {csv_path.name}: {exc}", file=sys.stderr)
        return {}
    _backfill_unknown_datasets(rows, csv_path.name)
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in _dedup_fast_sweep(rows, csv_path.name):
        if row["dataset_name"].startswith("unknown"):
            print(f"  [warn] {csv_path.name}: dropping cell with unidentifiable dataset",
                  file=sys.stderr)
            continue
        by_ds[row["dataset_name"]].append(row)
    return by_ds


def main() -> None:
    """CLI entry point — used by `memtuner plots` and `python scripts/plot_benchmark.py`."""
    parser = argparse.ArgumentParser(
        description="MemTuner plot generator — reads directly from study grid CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot everything in data/output/ (all datasets, all runs)
  python scripts/plot_benchmark.py

  # Publication-quality (300 DPI)
  python scripts/plot_benchmark.py --dpi 300

  # Plot a single run CSV
  python scripts/plot_benchmark.py --csv data/output/study_x/locomo_1977q_p1-baselines_*_grid.csv

  # Custom output directory
  python scripts/plot_benchmark.py --out my_plots/

Install dependencies:
  pip install -e .          # installs matplotlib, pandas, numpy from pyproject.toml
  pip install matplotlib numpy pandas   # or directly
""",
    )
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI — 150=screen (default), 300=print/publication")
    parser.add_argument("--root", type=str, default=None,
                        help="Project root directory (default: parent of scripts/)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Plot a single grid CSV instead of all runs")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for PNG files (default: data/output/plots/)")
    parser.add_argument("--from-master", type=str, default=None, metavar="CSV",
                        help="Plot from an existing master_results.csv instead of "
                             "scanning study_* directories")
    args = parser.parse_args()
    _setup_style(args.dpi)

    if args.csv:
        # Single-CSV mode: plot just one run
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: {csv_path} not found", file=sys.stderr)
            sys.exit(1)
        by_ds = load_single_csv(csv_path)
        out_dir = Path(args.out) if args.out else csv_path.parent / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nMemTuner Plot Generator — single CSV mode")
        print(f"  CSV:    {csv_path.name}")
        print(f"  Output: {out_dir}\n")
        for ds_name, rows in by_ds.items():
            if _skip_if_saturated(ds_name, rows, out_dir):
                continue
            print(f"  [{ds_name}]  ({len(rows)} cells)")
            plot_strategy_comparison(ds_name, rows, out_dir, args.dpi)
            plot_decay_sweep        (ds_name, rows, out_dir, args.dpi)
            plot_recall_curve       (ds_name, rows, out_dir, args.dpi)
            plot_bm25_sweep         (ds_name, rows, out_dir, args.dpi)
            plot_memory_type        (ds_name, rows, out_dir, args.dpi)
            plot_best_config_card   (ds_name, rows, out_dir, args.dpi)
        print(f"\n  Done — plots in {out_dir}/")
    else:
        # All-runs mode
        root    = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
        out_dir = Path(args.out) if args.out else root / "data" / "output" / "plots"
        master  = Path(args.from_master) if args.from_master else None
        generate_plots(project_root=root, dpi=args.dpi, master_csv=master)
        if args.out:
            print(f"  (use --out to redirect to {args.out})")


if __name__ == "__main__":
    main()
