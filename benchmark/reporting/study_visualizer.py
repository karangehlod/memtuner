"""Publication-quality study visualizer.

Generates all chart panels from study results at research-publication standard:
  - Colorblind-safe palette (Wong 2011, 8-color)
  - 300 DPI output, tight layout, consistent typography
  - Error bars (±1 std) wherever multiple runs exist per group
  - All axes labeled, all values annotated, all legends complete
  - Per-dataset breakdown panel for cross-dataset comparison

Seven chart sections:
  1. BM25 Baseline        — strategy × decay: recall/precision bars + noise
  2. Embedding Comparison — model × memory type: grouped bars + recall/latency tradeoff
  3. Hybrid Weight Sweep  — bm25_weight line: recall & MRR vs weight per memory type
  4. Reranker Comparison  — grouped bars: recall/MRR/latency with lift annotations
  5. Decay × Lambda       — dual heatmap: Recall@K and MRR vs (policy × lambda)
  6. Per-Dataset Summary  — cross-dataset strategy and embedding comparison
  7. Leaderboard          — horizontal bar chart: top-15 configs, all metrics
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, gridspec, np
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualizations. "
            "Install: pip install matplotlib"
        ) from e


# ── Wong (2011) colorblind-safe 8-color palette ──────────────────────────────
# Works for deuteranopia, protanopia, tritanopia.
# Order: blue, orange, green, red, purple, brown, pink, grey
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red-orange
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
]

# Typography constants
_TITLE_SIZE  = 12
_LABEL_SIZE  = 10
_TICK_SIZE   = 9
_ANNOT_SIZE  = 8
_LEGEND_SIZE = 8
_DPI         = 300   # publication minimum
_DPI_REPORT  = 200   # composite report (large file otherwise)


def _fig_width_for_n(n: int, min_w: float = 8.0, per_item: float = 1.2) -> float:
    """Return a figure width that gives each x-axis item enough space."""
    return max(min_w, n * per_item)


def _apply_xticklabels(ax, labels, max_chars: int = 14, rotation: int = 40) -> None:
    """Set x-tick labels with smart truncation and rotation to prevent overlap."""
    short = [lb[:max_chars] + "…" if len(lb) > max_chars else lb for lb in labels]
    ax.set_xticklabels(short, rotation=rotation, ha="right", fontsize=_TICK_SIZE)


def _style_ax(ax, title="", xlabel="", ylabel="", ylim=None):
    """Apply consistent publication styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=_TICK_SIZE)


def _annotate_bars(ax, bars, fmt="{:.3f}", offset_frac=0.01, fontsize=None,
                   threshold: float = 0.0):
    """Annotate bar tops — only bars above `threshold` to avoid clutter."""
    fs = fontsize or _ANNOT_SIZE
    for bar in bars:
        h = bar.get_height()
        if h > max(threshold, 0.001):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset_frac,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fs, fontweight="bold",
            )


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    s = (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5
    return m, s


def _grouped_bar(ax, np, groups, categories, values_by_group_cat,
                 std_by_group_cat=None, palette=None, bar_width=None):
    """Draw grouped bar chart. Returns list of bar containers."""
    pal = palette or _PALETTE
    n_cat = len(categories)
    n_grp = len(groups)
    if bar_width is None:
        bar_width = min(0.8 / max(n_cat, 1), 0.25)
    x = np.arange(n_grp)
    containers = []
    for ci, cat in enumerate(categories):
        offset = (ci - (n_cat - 1) / 2) * bar_width
        vals = [values_by_group_cat.get((g, cat), 0.0) for g in groups]
        errs = None
        if std_by_group_cat:
            errs = [std_by_group_cat.get((g, cat), 0.0) for g in groups]
        bars = ax.bar(
            x + offset, vals, bar_width * 0.88,
            label=cat, color=pal[ci % len(pal)], alpha=0.88,
            **({}  if errs is None else
               {"yerr": errs, "capsize": 3,
                "error_kw": {"elinewidth": 1, "ecolor": "black", "alpha": 0.6}}),
        )
        containers.append(bars)
    ax.set_xticks(x)
    return containers


class StudyVisualizer:
    """Generates publication-quality study report from StudyRunResult objects."""

    def __init__(self, results: list, output_dir: Path):
        self._results = [r for r in results if r.success]
        self._all = results
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    # ─── Public API ──────────────────────────────────────────────────────────

    def generate_all(self) -> dict[str, str]:
        """Generate all charts. Returns dict of section → file path."""
        plt, gridspec, np = _require_matplotlib()
        _set_rcparams(plt)
        paths = {}
        paths["bm25_baseline"]          = self._plot_bm25_baseline(plt, np)
        paths["embedding_comparison"]    = self._plot_embedding_comparison(plt, np)
        paths["hybrid_weight"]           = self._plot_hybrid_weight(plt, np)
        paths["reranker_comparison"]     = self._plot_reranker_comparison(plt, np)
        paths["decay_heatmap"]           = self._plot_decay_heatmap(plt, np)
        paths["decay_curves"]            = self._plot_decay_curves(plt, np)
        paths["per_dataset"]             = self._plot_per_dataset(plt, np)
        paths["leaderboard"]             = self._plot_leaderboard(plt, np)
        paths["composite_breakdown"]     = self._plot_composite_breakdown(plt, np)
        paths["parameter_sensitivity"]   = self._plot_parameter_sensitivity(plt, np)
        paths["phase_progression"]       = self._plot_phase_progression(plt, np)
        paths["noise_quality"]           = self._plot_noise_quality(plt, np)
        paths["efficiency"]              = self._plot_efficiency(plt, np)
        paths["resource_usage"]          = self._plot_resource_usage(plt, np)
        paths["recall_at_k_variation"]   = self._plot_recall_k_variation(plt, np)
        paths["ci_comparison"]           = self._plot_ci_comparison(plt, np)
        paths["full_report"]             = self._plot_full_report(plt, gridspec, np)
        plt.close("all")
        return paths

    @classmethod
    def plot_cross_dataset_heatmap(
        cls,
        per_dataset_summaries: dict[str, dict],
        output_path: Path | str,
    ) -> str:
        """Cross-dataset strategy heatmap — shows visually which strategy wins where.

        Rows = datasets, columns = strategies, cell = best Recall@K for that
        (dataset, strategy) combination. ★ marks the winner per dataset.
        Produced once after all datasets have run; not part of per-dataset charts.

        Args:
            per_dataset_summaries: Maps dataset_name → StudyAggregator.study_summary().
            output_path: Where to write the PNG.

        Returns:
            The output path as a string.
        """
        plt, _, np = _require_matplotlib()
        _set_rcparams(plt)

        # Collect all strategies and datasets present
        all_strats: list[str] = []
        _seen: set[str] = set()
        strategy_order = ["recency", "bm25", "embeddings", "hybrid",
                          "api_embeddings", "llm_rerank"]
        for s in strategy_order:
            if any(s in {r["retrieval_strategy"] for r in
                         sum.get("retrieval_strategy_ranking", [])}
                   for sum in per_dataset_summaries.values()) and s not in _seen:
                all_strats.append(s)
                _seen.add(s)

        if not all_strats:
            return ""

        datasets = sorted(per_dataset_summaries.keys())
        matrix = np.zeros((len(datasets), len(all_strats)))

        for i, ds in enumerate(datasets):
            summary = per_dataset_summaries[ds]
            recall_by_strat = {
                r["retrieval_strategy"]: r["avg_recall"]
                for r in summary.get("retrieval_strategy_ranking", [])
            }
            for j, strat in enumerate(all_strats):
                matrix[i, j] = recall_by_strat.get(strat, 0.0)

        fig, ax = plt.subplots(figsize=(max(10, len(all_strats) * 2), max(5, len(datasets) * 1.4)))
        fig.suptitle(
            "Cross-Dataset Strategy Performance\n"
            "Recall@K — ★ = best strategy per dataset",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        vmin = float(matrix[matrix > 0].min()) if (matrix > 0).any() else 0.0
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=min(1.0, vmin + 0.6),
                       aspect="auto")

        ax.set_xticks(range(len(all_strats)))
        ax.set_yticks(range(len(datasets)))
        ax.set_xticklabels(all_strats, rotation=35, ha="right", fontsize=_TICK_SIZE)
        ax.set_yticklabels(datasets, fontsize=_TICK_SIZE)
        ax.set_xlabel("Retrieval Strategy", fontsize=_LABEL_SIZE)
        ax.set_ylabel("Dataset", fontsize=_LABEL_SIZE)

        for i, _ds in enumerate(datasets):
            row_best = float(matrix[i].max())
            for j in range(len(all_strats)):
                val = float(matrix[i, j])
                if val == 0:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=_ANNOT_SIZE, color="grey")
                else:
                    star = " ★" if val == row_best and val > 0 else ""
                    bg = (val - vmin) / max(0.6, row_best - vmin)
                    tc = "white" if bg > 0.65 else "black"
                    ax.text(j, i, f"{val:.3f}{star}", ha="center", va="center",
                            fontsize=_ANNOT_SIZE, color=tc, fontweight="bold" if star else "normal")

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Recall@K", fontsize=_LABEL_SIZE - 1)
        cbar.ax.tick_params(labelsize=_TICK_SIZE - 1)

        # Annotate with dataset character from profiles
        try:
            from benchmark.gold.dataset_profiles import get_profile
            for i, ds in enumerate(datasets):
                p = get_profile(ds)
                if p:
                    ax.annotate(
                        f"  ← {p.character}",
                        xy=(len(all_strats) - 0.45, i),
                        fontsize=_ANNOT_SIZE - 1, va="center", color="dimgrey",
                        annotation_clip=False,
                    )
        except ImportError:
            pass

        plt.tight_layout()
        path = str(output_path)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 1: BM25 Baseline ──────────────────────────────────────────────

    def _plot_bm25_baseline(self, plt, np) -> str:
        bm25 = [r for r in self._results if r.retrieval_strategy == "bm25"]
        if not bm25:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Phase 1 — BM25 Baseline",
            fontsize=_TITLE_SIZE + 1, fontweight="bold"
        )

        # ── Left: grouped bars — memory type × decay policy ──────────────────
        mem_types   = sorted({r.memory_type for r in bm25})
        decay_keys  = sorted({f"{r.decay_policy} λ={r.lambda_value:.3f}" for r in bm25})
        vals_map, std_map = {}, {}
        for r in bm25:
            dk = f"{r.decay_policy} λ={r.lambda_value:.3f}"
            vals_map.setdefault((r.memory_type, dk), []).append(r.recall_at_k)
        for k, v in vals_map.items():
            m, s = _mean_std(v)
            vals_map[k] = m
            std_map[k] = s

        _grouped_bar(
            axes[0], np, mem_types, decay_keys,
            {(mt, dk): vals_map.get((mt, dk), 0.0) for mt in mem_types for dk in decay_keys},
            {(mt, dk): std_map.get((mt, dk), 0.0)  for mt in mem_types for dk in decay_keys},
        )
        axes[0].legend(loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall@K by Memory Type × Decay Config",
                  "Memory Type", "Recall@K", ylim=(0, 1.0))

        # ── Right: Recall vs Precision scatter ───────────────────────────────
        for i, mem in enumerate(mem_types):
            pts = [(r.recall_at_k, r.precision_at_k) for r in bm25 if r.memory_type == mem]
            if pts:
                xs, ys = zip(*pts)
                axes[1].scatter(xs, ys, label=mem, color=_PALETTE[i], s=55, alpha=0.85,
                                edgecolors="white", linewidths=0.4, zorder=3)

        # Diagonal reference line (perfect P=R)
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="P=R")
        axes[1].legend(loc="lower right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1], "BM25 Recall–Precision Trade-off",
                  "Recall@K", "Precision@K", ylim=(0, 1.0))
        axes[1].set_xlim(0, 1.0)

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase1_bm25_baseline.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 2: Embedding Model Comparison ─────────────────────────────────

    def _plot_embedding_comparison(self, plt, np) -> str:
        # Phase 2 uses strategy names: semantic, colbert, adaptive, embeddings, api_embeddings.
        # Filter by phase tag first (most reliable); fall back to strategy name for
        # runs where phase tags were not set.
        _phase2_tags = {"phase2_embedding_comparison"}
        embed = [r for r in self._results
                 if getattr(r, "study_phase", "") in _phase2_tags
                 and r.embedding_model]
        if not embed:
            embed = [r for r in self._results
                     if r.retrieval_strategy in ("embeddings", "api_embeddings", "semantic")
                     and r.embedding_model]
        if not embed:
            return ""

        models    = sorted({r.embedding_model for r in embed})
        short     = [m.split("/")[-1] for m in models]
        mem_types = sorted({r.memory_type for r in embed})

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Phase 2 — Embedding Model Comparison",
                     fontsize=_TITLE_SIZE + 1, fontweight="bold")

        # ── Left: grouped bars per model × memory_type with error bars ────────
        vals_map, std_map = {}, {}
        for r in embed:
            vals_map.setdefault((r.embedding_model, r.memory_type), []).append(r.recall_at_k)
        for k, v in vals_map.items():
            m, s = _mean_std(v)
            vals_map[k] = m
            std_map[k] = s

        _grouped_bar(
            axes[0], np, models, mem_types,
            {(m, mt): vals_map.get((m, mt), 0.0) for m in models for mt in mem_types},
            {(m, mt): std_map.get((m, mt), 0.0)  for m in models for mt in mem_types},
        )
        axes[0].set_xticklabels(short, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[0].legend(title="Memory Type", loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall@K by Embedding Model × Memory Type",
                  "Embedding Model", "Recall@K", ylim=(0, 1.0))

        # ── Right: Recall vs Latency P50 — Pareto frontier ───────────────────
        by_model_recall = defaultdict(list)
        by_model_lat    = defaultdict(list)
        for r in embed:
            by_model_recall[r.embedding_model].append(r.recall_at_k)
            by_model_lat[r.embedding_model].append(r.latency_p50_ms)

        recall_means = {m: _mean_std(by_model_recall[m])[0] for m in models}
        lat_means    = {m: _mean_std(by_model_lat[m])[0]    for m in models}

        ax2 = axes[1]
        # Markers: use different shapes so models are distinguishable even if colours
        # look similar on grayscale prints.
        _markers = ["o", "s", "^", "D", "P", "*", "X", "v"]

        for i, model in enumerate(models):
            lx, ry = lat_means[model], recall_means[model]
            ax2.scatter(lx, ry, s=140, marker=_markers[i % len(_markers)],
                        color=_PALETTE[i % len(_PALETTE)], zorder=4, linewidths=1,
                        edgecolors="white")

        # Draw Pareto frontier line — sort by latency, keep points where recall
        # monotonically increases (no other point has both lower latency AND higher recall)
        pts = sorted([(lat_means[m], recall_means[m], i, m) for i, m in enumerate(models)],
                     key=lambda t: t[0])
        pareto, best_r = [], 0.0
        for lx, ry, idx, model in pts:
            if ry >= best_r:
                pareto.append((lx, ry, idx, model))
                best_r = ry
        if len(pareto) >= 2:
            px, py = zip(*[(p[0], p[1]) for p in pareto])
            ax2.step(px, py, where="post", color="red", linewidth=1.5,
                     linestyle="--", alpha=0.6, zorder=2, label="Pareto frontier")

        # Smart label placement: use quadrant-based offsets so labels don't pile up.
        # Place label above-right, below-right, above-left, below-left alternately.
        offsets = [(8, 6), (8, -12), (-8, 6), (-8, -12), (8, 14), (8, -20)]
        for i, model in enumerate(models):
            lx, ry = lat_means[model], recall_means[model]
            ox, oy = offsets[i % len(offsets)]
            is_pareto = any(p[3] == model for p in pareto)
            weight = "bold" if is_pareto else "normal"
            colour = _PALETTE[i % len(_PALETTE)]
            ax2.annotate(
                short[i],
                (lx, ry),
                textcoords="offset points", xytext=(ox, oy),
                fontsize=_ANNOT_SIZE, color=colour, fontweight=weight,
                arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.8, "alpha": 0.5},
            )

        # Adaptive axes — zoom to actual data range with 15% padding
        all_lx = [lat_means[m] for m in models if lat_means[m] > 0]
        all_ry = [recall_means[m] for m in models]
        if all_lx and all_ry:
            x_pad = (max(all_lx) - min(all_lx)) * 0.2 or max(all_lx) * 0.3
            y_pad = (max(all_ry) - min(all_ry)) * 0.15 or 0.03
            ax2.set_xlim(max(0, min(all_lx) - x_pad), max(all_lx) + x_pad)
            ax2.set_ylim(max(0, min(all_ry) - y_pad), min(1, max(all_ry) + y_pad))

        # "Better" direction arrow in lower-left corner
        ax2.annotate("Better ↗\n(faster + accurate)",
                     xy=(0.05, 0.08), xycoords="axes fraction",
                     fontsize=_ANNOT_SIZE - 1, color="green", alpha=0.75,
                     fontweight="bold")

        ax2.legend(loc="lower right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(ax2, "Recall@K vs Latency P50 — Pareto Frontier",
                  "Latency P50 (ms)", "Recall@K")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase2_embedding_comparison.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 3: Hybrid Weight Sweep ────────────────────────────────────────

    def _plot_hybrid_weight(self, plt, np) -> str:
        hybrid = [r for r in self._results if r.retrieval_strategy == "hybrid"]
        if not hybrid:
            return ""

        mem_types = sorted({r.memory_type for r in hybrid})
        weights   = sorted({r.bm25_weight for r in hybrid})

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "Phase 3 — Hybrid Strategy BM25 Weight Sweep\n"
            "BM25 weight 0 = pure semantic  ·  1 = pure BM25",
            fontsize=_TITLE_SIZE + 1, fontweight="bold"
        )

        for ax_idx, (metric, ylabel) in enumerate([
            ("recall_at_k", "Recall@K"),
            ("mrr",         "MRR"),
        ]):
            ax = axes[ax_idx]
            for i, mem in enumerate(mem_types):
                pts = []
                for w in weights:
                    vals = [getattr(r, metric) for r in hybrid
                            if r.memory_type == mem and abs(r.bm25_weight - w) < 1e-9]
                    if vals:
                        m, s = _mean_std(vals)
                        pts.append((w, m, s))
                if pts:
                    ws, ms, _ = zip(*pts)
                    ax.plot(ws, ms, "o-", label=mem, color=_PALETTE[i],
                            linewidth=2, markersize=8)
                    # Shading removed: ±std mixes episodic/preference ranges (misleading with 1 seed)
            # Vertical reference lines at sweep points
            for w in weights:
                ax.axvline(w, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
            ax.legend(title="Memory Type", loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
            _style_ax(ax, f"{ylabel} vs BM25 Weight", "BM25 Weight", ylabel,
                      ylim=(0, 1.0))
            ax.set_xlim(-0.05, 1.05)

        # ── Right: Composite score vs BM25 weight (macro-avg across memory types) ──
        ax3 = axes[2]
        comp_pts = []
        for w in weights:
            vals = [r.composite_score() for r in hybrid if abs(r.bm25_weight - w) < 1e-9]
            if vals:
                m, s = _mean_std(vals)
                comp_pts.append((w, m, s))
        if comp_pts:
            ws, ms, _ = zip(*comp_pts)
            ax3.plot(ws, ms, "s-", color=_PALETTE[3], linewidth=2.5, markersize=9,
                     label="Composite")
            # Shading removed: single seed, ±std not meaningful
            # Mark best weight
            best_w, best_m = max(zip(ws, ms), key=lambda t: t[1])
            ax3.axvline(best_w, color="red", linestyle="--", linewidth=1.5,
                        label=f"Best: {best_w:.2f}")
            ax3.annotate(f"★ {best_m:.4f}", xy=(best_w, best_m),
                         xytext=(best_w + 0.05, best_m - 0.015),
                         fontsize=_ANNOT_SIZE, color="red", fontweight="bold")
        for w in weights:
            ax3.axvline(w, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
        ax3.legend(loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(ax3, "Composite Score vs BM25 Weight",
                  "BM25 Weight", "Composite Score", ylim=(0, 1.0))
        ax3.set_xlim(-0.05, 1.05)

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase3_hybrid_weight.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 4: Reranker Comparison ────────────────────────────────────────

    def _plot_reranker_comparison(self, plt, np) -> str:
        rerank = [r for r in self._results if r.study_phase == "phase_reranker_comparison"]
        if not rerank:
            return ""

        rerankers  = sorted({r.reranker_model for r in rerank})
        short_names = [rr.split("/")[-1] if rr != "none" else "none\n(n-gram baseline)" for rr in rerankers]
        metrics    = ["recall_at_k", "precision_at_k", "mrr"]
        mlabels    = ["Recall@K", "Precision@K", "MRR"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Phase 4 — Reranker Comparison",
                     fontsize=_TITLE_SIZE + 1, fontweight="bold")

        # ── Left: grouped bars — metric × reranker ────────────────────────────
        vals_map, std_map = {}, {}
        for r in rerank:
            for m in metrics:
                vals_map.setdefault((r.reranker_model, m), []).append(getattr(r, m))
        for k, v in vals_map.items():
            mn, s = _mean_std(v)
            vals_map[k] = mn
            std_map[k] = s

        _grouped_bar(
            axes[0], np, rerankers, metrics,
            {(rr, m): vals_map.get((rr, m), 0.0) for rr in rerankers for m in metrics},
            {(rr, m): std_map.get((rr, m), 0.0)  for rr in rerankers for m in metrics},
            palette=_PALETTE,
        )
        axes[0].set_xticklabels(short_names, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[0].legend(labels=mlabels, loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall / Precision@K / MRR by Reranker",
                  "Reranker", "Score", ylim=(0, 1.0))

        # Lift annotation: delta vs baseline (none)
        baseline_recall = vals_map.get(("none", "recall_at_k"), 0.0)
        for xi, rr in enumerate(rerankers):
            rv = vals_map.get((rr, "recall_at_k"), 0.0)
            delta = rv - baseline_recall
            if abs(delta) > 0.001:
                colour = "green" if delta > 0 else "red"
                axes[0].annotate(
                    f"{delta:+.3f}",
                    xy=(xi, rv + 0.03),
                    ha="center", fontsize=_ANNOT_SIZE, color=colour, fontweight="bold",
                )

        # ── Right: Latency P50 / P90 / P99 comparison ───────────────────────
        # Tail latency matters for rerankers: CrossEncoder has long P99 due to batch processing.
        lat_percs = [
            ("latency_p50_ms", "P50", _PALETTE[3]),
            ("latency_p90_ms", "P90", _PALETTE[0]),
            ("latency_p99_ms", "P99", _PALETTE[2]),
        ]
        x = np.arange(len(rerankers))
        w = 0.25
        for pi, (field, plabel, colour) in enumerate(lat_percs):
            vals = [
                _mean_std([getattr(r, field) for r in rerank if r.reranker_model == rr])[0]
                for rr in rerankers
            ]
            bars = axes[1].bar(x + (pi - 1) * w, vals, w * 0.88,
                               label=plabel, color=colour, alpha=0.88)
            if pi == 0:
                _annotate_bars(axes[1], bars, fmt="{:.0f}ms", fontsize=_ANNOT_SIZE - 1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(short_names, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[1].legend(title="Percentile", loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1], "Query Latency P50 / P90 / P99 by Reranker", "Reranker", "Latency (ms)")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase4_reranker_comparison.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 5: Decay × Lambda Heatmap ─────────────────────────────────────

    def _plot_decay_heatmap(self, plt, np) -> str:
        decay_r = [r for r in self._results if r.study_phase == "phase_decay_sweep"]
        if not decay_r:
            decay_r = self._results

        policies = sorted({r.decay_policy for r in decay_r if r.decay_policy != "none"})
        lambdas  = sorted({r.lambda_value  for r in decay_r if r.lambda_value > 0})

        if not policies or not lambdas:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Phase 5 — Decay Policy × Lambda Sweep",
                     fontsize=_TITLE_SIZE + 1, fontweight="bold")

        for ax_idx, (metric, metric_label) in enumerate([
            ("recall_at_k", "Recall@K"),
            ("mrr",         "MRR"),
        ]):
            ax = axes[ax_idx]
            grid = np.zeros((len(policies), len(lambdas)))
            for pi, policy in enumerate(policies):
                for li, lam in enumerate(lambdas):
                    vals = [
                        getattr(r, metric) for r in decay_r
                        if r.decay_policy == policy and abs(r.lambda_value - lam) < 1e-9
                    ]
                    if vals:
                        grid[pi, li] = sum(vals) / len(vals)

            vmin = 0.0
            vmax = max(float(grid.max()), 0.01)
            im = ax.imshow(grid, aspect="auto", cmap="YlGn",
                           vmin=vmin, vmax=vmax + 0.02 * vmax)

            ax.set_xticks(np.arange(len(lambdas)))
            ax.set_xticklabels([f"{lam_val:.3f}" for lam_val in lambdas],
                               rotation=40, ha="right", fontsize=_TICK_SIZE)
            ax.set_yticks(np.arange(len(policies)))
            ax.set_yticklabels(policies, fontsize=_TICK_SIZE)
            ax.set_xlabel("Lambda (λ)", fontsize=_LABEL_SIZE)
            ax.set_ylabel("Decay Policy", fontsize=_LABEL_SIZE)
            ax.set_title(f"{metric_label} — Decay × Lambda",
                         fontsize=_TITLE_SIZE, fontweight="bold")

            # Cell annotations
            for pi in range(len(policies)):
                for li in range(len(lambdas)):
                    v = grid[pi, li]
                    text_colour = "white" if v > 0.65 * vmax else "black"
                    ax.text(li, pi, f"{v:.3f}", ha="center", va="center",
                            fontsize=_ANNOT_SIZE, color=text_colour, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.ax.tick_params(labelsize=_TICK_SIZE - 1)
            cbar.set_label(metric_label, fontsize=_LABEL_SIZE - 1)

            # Mark best cell with a star
            best_idx = np.unravel_index(np.argmax(grid), grid.shape)
            ax.add_patch(plt.Rectangle(
                (best_idx[1] - 0.47, best_idx[0] - 0.47), 0.94, 0.94,
                fill=False, edgecolor="red", linewidth=2.5, zorder=5,
            ))
            ax.text(best_idx[1], best_idx[0] - 0.35, "★",
                    ha="center", va="bottom", color="red", fontsize=10, zorder=6)

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase5_decay_heatmap.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 6: Per-Dataset Strategy Comparison ────────────────────────────

    def _plot_per_dataset(self, plt, np) -> str:
        """Cross-dataset comparison: best strategy and best embedding per dataset."""
        if not self._results:
            return ""

        # Group by dataset — inferred from run_id prefix (set by from_csv source_run_id)
        by_ds: dict[str, list] = defaultdict(list)
        for r in self._results:
            ds = r.run_id.split(":")[0] if ":" in r.run_id else "default"
            by_ds[ds].append(r)

        if len(by_ds) <= 1:
            return ""  # skip single-dataset runs — no comparison to show

        datasets = sorted(by_ds.keys())
        strategies = sorted({r.retrieval_strategy for r in self._results})
        models     = sorted({r.embedding_model for r in self._results
                              if r.retrieval_strategy in ("embeddings", "api_embeddings", "semantic", "colbert", "adaptive")})

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(
            "Per-Dataset Strategy & Embedding Comparison\n"
            "Same configuration performs differently across datasets — use per-dataset recommendations",
            fontsize=_TITLE_SIZE, fontweight="bold",
        )

        # ── Left: strategy recall per dataset ────────────────────────────────
        vals_map = {}
        for ds, results in by_ds.items():
            for strat in strategies:
                vs = [r.recall_at_k for r in results if r.retrieval_strategy == strat]
                if vs:
                    vals_map[(ds, strat)] = _mean_std(vs)[0]

        _grouped_bar(
            axes[0], np, datasets, strategies,
            {(ds, s): vals_map.get((ds, s), 0.0) for ds in datasets for s in strategies},
        )
        axes[0].set_xticklabels(
            [d[:20] for d in datasets], rotation=35, ha="right", fontsize=_TICK_SIZE
        )
        axes[0].legend(title="Strategy", loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall@K by Strategy — Per Dataset",
                  "Dataset", "Recall@K", ylim=(0, 1.0))

        # ── Right: embedding model recall per dataset ─────────────────────────
        emb_map = {}
        for ds, results in by_ds.items():
            for model in models:
                vs = [r.recall_at_k for r in results
                      if r.embedding_model == model
                      and r.retrieval_strategy in ("embeddings", "api_embeddings", "semantic", "colbert", "adaptive")]
                if vs:
                    emb_map[(ds, model)] = _mean_std(vs)[0]

        short_models = [m.split("/")[-1] for m in models]
        _grouped_bar(
            axes[1], np, datasets, models,
            {(ds, m): emb_map.get((ds, m), 0.0) for ds in datasets for m in models},
        )
        axes[1].set_xticklabels(
            [d[:20] for d in datasets], rotation=35, ha="right", fontsize=_TICK_SIZE
        )
        # Relabel legend with short model names
        handles, _ = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, short_models, title="Embedding Model",
                       fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1], "Recall@K by Embedding Model — Per Dataset",
                  "Dataset", "Recall@K", ylim=(0, 1.0))

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase6_per_dataset.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Leaderboard ─────────────────────────────────────────────────────────

    def _plot_leaderboard(self, plt, np) -> str:
        if not self._results:
            return ""

        ranked = sorted(self._results, key=lambda r: r.composite_score(), reverse=True)[:10]

        labels = []
        for r in ranked:
            embed = r.embedding_model.split("/")[-1][:12]
            strat = r.retrieval_strategy[:10]
            labels.append(
                f"{strat} | {embed}\n"
                f"λ={r.lambda_value:.3f}  bm25w={r.bm25_weight:.1f}  {r.memory_type[:4]}"
            )

        # Composite score weights (must match scheduler.py COMPOSITE_WEIGHTS)
        _W = {"recall": 0.40, "precision": 0.25, "mrr": 0.20, "temporal": 0.15}
        composite = [r.composite_score() for r in ranked]

        fig, axes = plt.subplots(1, 3, figsize=(24, max(7, len(ranked) * 0.8)))
        fig.suptitle("Top-15 Configurations — Leaderboard",
                     fontsize=_TITLE_SIZE + 1, fontweight="bold")

        y = np.arange(len(ranked))

        # ── Panel 1: Composite score horizontal bars ──────────────────────────
        axes[0].barh(y, composite,
                     color=[_PALETTE[i % len(_PALETTE)] for i in range(len(ranked))],
                     alpha=0.88, height=0.72)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=_ANNOT_SIZE - 1)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Composite Score  (0.40×R + 0.25×P + 0.20×MRR + 0.15×T)",
                            fontsize=_LABEL_SIZE - 1)
        axes[0].set_xlim(0, 1.05)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axes[0].set_axisbelow(True)
        axes[0].set_title("Composite Score (Top 15)", fontsize=_TITLE_SIZE, fontweight="bold")
        for xi, score in zip(y, composite):
            axes[0].text(score + 0.008, xi, f"{score:.3f}",
                         va="center", fontsize=_ANNOT_SIZE, fontweight="bold")

        # ── Panel 2: Composite score breakdown — stacked horizontal bars ──────
        # Shows the 4 weighted components so you can see WHICH metric drives the score.
        components = [
            ("recall_at_k",       _W["recall"],    "Recall×0.40",    _PALETTE[0]),
            ("precision_at_k",    _W["precision"], "Precision×0.25", _PALETTE[1]),
            ("mrr",               _W["mrr"],       "MRR×0.20",       _PALETTE[2]),
            ("temporal_accuracy", _W["temporal"],  "Temporal×0.15",  _PALETTE[4]),
        ]
        lefts = np.zeros(len(ranked))
        for field, weight, clabel, colour in components:
            widths = np.array([getattr(r, field) * weight for r in ranked])
            axes[1].barh(y, widths, left=lefts, height=0.72,
                         label=clabel, color=colour, alpha=0.88)
            lefts += widths
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([""] * len(ranked))  # y-labels on panel 1 already
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Weighted Component Value", fontsize=_LABEL_SIZE)
        axes[1].set_xlim(0, 1.05)
        axes[1].legend(fontsize=_LEGEND_SIZE, loc="lower right", framealpha=0.9)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axes[1].set_axisbelow(True)
        axes[1].set_title("Composite Breakdown — Weighted Components",
                          fontsize=_TITLE_SIZE, fontweight="bold")

        # ── Panel 3: Latency P50 / P90 / P99 per configuration ───────────────
        # Tail latency (P99) reveals configurations that look good on recall but are slow.
        lat_fields = [
            ("latency_p50_ms", "P50", _PALETTE[3]),
            ("latency_p90_ms", "P90", _PALETTE[5]),
            ("latency_p99_ms", "P99", _PALETTE[7]),
        ]
        w_lat = 0.22
        for pi, (field, plabel, colour) in enumerate(lat_fields):
            vals = np.array([getattr(r, field, 0.0) for r in ranked])
            axes[2].barh(y + (pi - 1) * w_lat, vals, w_lat * 0.88,
                         label=plabel, color=colour, alpha=0.88)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels([""] * len(ranked))
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Latency (ms)", fontsize=_LABEL_SIZE)
        axes[2].legend(title="Percentile", fontsize=_LEGEND_SIZE, framealpha=0.9)
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)
        axes[2].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axes[2].set_axisbelow(True)
        axes[2].set_title("Query Latency — P50 / P90 / P99",
                          fontsize=_TITLE_SIZE, fontweight="bold")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase7_leaderboard.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase Progression — Did each phase improve recall? ──────────────────

    def _plot_phase_progression(self, plt, np) -> str:
        """Step chart: best recall and composite after each study phase.

        Answers the most important question: 'Was it worth running all 5 phases?'
        Shows Phase 1 → 2 → 3 → 4 → 5 progression of the best config seen so far.
        """
        _phase_order = [
            ("phase1_baselines",             "Phase 1\nBM25 Baseline"),
            ("phase2_embedding_comparison",  "Phase 2\nEmbedding"),
            ("phase3_hybrid_broad",          "Phase 3 (broad)\nHybrid Weight"),
            ("phase3_hybrid_fine",           "Phase 3 (fine)\nHybrid Weight"),
            ("phase4_decay_broad",           "Phase 4 (broad)\nDecay"),
            ("phase4_decay_fine",            "Phase 4 (fine)\nDecay"),
            ("phase4b_archival_floor",       "Phase 4b\nArchival Floor"),
        ]

        # Collect best recall and composite seen after each phase
        seen_results: list = []
        phase_labels, phase_recall, phase_composite = [], [], []
        phase_p50, phase_p99 = [], []

        for phase_tag, label in _phase_order:
            phase_cells = [r for r in self._results
                           if getattr(r, "study_phase", "") == phase_tag]
            if not phase_cells:
                continue
            seen_results.extend(phase_cells)
            best = max(seen_results, key=lambda r: r.composite_score())
            phase_labels.append(label)
            phase_recall.append(best.recall_at_k)
            phase_composite.append(best.composite_score())
            phase_p50.append(best.latency_p50_ms)
            phase_p99.append(best.latency_p99_ms)

        if len(phase_labels) < 2:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Phase Progression — Was Each Phase Worth Running?\n"
            "Best config seen so far, updated after each phase completes",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        x = np.arange(len(phase_labels))

        # ── Left: Recall + composite step chart ──────────────────────────────
        axes[0].plot(x, phase_recall, "o-", color=_PALETTE[0], linewidth=2.5,
                     markersize=10, label="Best Recall@K")
        axes[0].plot(x, phase_composite, "s--", color=_PALETTE[3], linewidth=2,
                     markersize=8, label="Best Composite Score")
        for xi, (r, _c) in enumerate(zip(phase_recall, phase_composite)):
            axes[0].annotate(f"{r:.3f}", xy=(xi, r), xytext=(0, 10),
                             textcoords="offset points", ha="center",
                             fontsize=_ANNOT_SIZE, color=_PALETTE[0], fontweight="bold")
        # Shade improvement between phases
        for i in range(1, len(x)):
            delta = phase_recall[i] - phase_recall[i - 1]
            if delta > 0.001:
                axes[0].annotate(f"+{delta:.3f}",
                                 xy=((x[i - 1] + x[i]) / 2, (phase_recall[i - 1] + phase_recall[i]) / 2),
                                 fontsize=_ANNOT_SIZE, color="green", ha="center",
                                 fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(phase_labels, fontsize=_TICK_SIZE)
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(loc="lower right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall@K & Composite — Best Config After Each Phase",
                  "Study Phase", "Score")

        # ── Right: Latency P50/P99 of best config at each phase ──────────────
        w = 0.3
        axes[1].bar(x - w/2, phase_p50, w * 0.92, label="Latency P50",
                    color=_PALETTE[0], alpha=0.88)
        axes[1].bar(x + w/2, phase_p99, w * 0.92, label="Latency P99",
                    color=_PALETTE[3], alpha=0.88)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(phase_labels, fontsize=_TICK_SIZE)
        axes[1].legend(loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1], "Best Config Latency P50 / P99 After Each Phase",
                  "Study Phase", "Latency (ms)")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "phase_progression.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Noise, Quality Metrics: Contamination, P@1, NDCG, F1 ───────────────

    def _plot_noise_quality(self, plt, np) -> str:
        """3-panel quality chart showing the full picture beyond a single recall number.

        Panel 1 — Recall per memory type (episodic / preference separately).
          The macro-average hides that episodic recall can be 0.71 while preference
          is 0.08 — they average to 0.38 which looks mediocre. Showing them
          separately reveals the true picture.

        Panel 2 — Contamination rate with expected baseline at K=10.
          Contamination = fraction of returned results that are NOT relevant.
          At K=10 with ~1 gold memory per query, contamination ≥ 0.90 is
          mathematically expected and does NOT mean the system is broken.
          The reference line shows the minimum contamination achievable at K=10.

        Panel 3 — Precision@1, NDCG, F1 per strategy.
          P@1 = is the TOP result correct? (most important for single-answer chatbots)
          NDCG = position-discounted ranking quality
          F1   = harmonic mean of recall and precision
        """
        if not self._results:
            return ""

        _decay_tags = {"phase4_decay_broad", "phase4_decay_fine",
                       "phase4_decay_sweep", "phase4b_archival_floor"}
        cells = [r for r in self._results
                 if getattr(r, "study_phase", "general") not in _decay_tags]
        strategies = sorted({r.retrieval_strategy for r in cells},
                             key=lambda s: -_mean_std(
                                 [r.recall_at_k for r in cells
                                  if r.retrieval_strategy == s and r.memory_type == "episodic"]
                             )[0])  # sort by episodic recall desc

        fw = _fig_width_for_n(len(strategies), 16, 1.5)
        fig, axes = plt.subplots(1, 3, figsize=(fw, 5))
        fig.suptitle(
            "Quality Beyond Recall — Per Memory Type, Contamination Context, Ranking Quality\n"
            "The macro-average recall hides the per-type picture",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        x = np.arange(len(strategies))

        # ── Panel 1: Recall per memory type — episodic and preference separately ──
        mem_types = sorted({r.memory_type for r in cells})
        w_mt = 0.75 / max(len(mem_types), 1)
        for mi, mem in enumerate(mem_types):
            vals = [_mean_std([r.recall_at_k for r in cells
                               if r.retrieval_strategy == s and r.memory_type == mem])[0]
                    for s in strategies]
            offset = (mi - (len(mem_types) - 1) / 2) * w_mt
            axes[0].bar(x + offset, vals, w_mt * 0.88,
                               label=mem, color=_PALETTE[mi], alpha=0.88)
            # Annotate only the episodic bars (the more informative ones)
            if mem == "episodic":
                for xi, v in enumerate(vals):
                    if v > 0.05:
                        axes[0].text(xi + offset, v + 0.01, f"{v:.2f}",
                                     ha="center", fontsize=_ANNOT_SIZE - 1,
                                     fontweight="bold", color=_PALETTE[mi])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[0].legend(title="Memory Type", loc="upper right",
                       fontsize=_LEGEND_SIZE, framealpha=0.9)
        axes[0].text(0.02, 0.02,
                     "Macro-avg mixes episodic & preference —\nread each type separately",
                     transform=axes[0].transAxes, fontsize=_ANNOT_SIZE - 1,
                     va="bottom", color="grey", style="italic")
        axes[0].autoscale(axis="y", tight=False)
        _yl, _yh = axes[0].get_ylim()
        axes[0].set_ylim(0, min(1.0, _yh + 0.05))
        _style_ax(axes[0], "Recall@K per Memory Type (episodic vs preference)",
                  "Strategy", "Recall@K")

        # ── Panel 2: Contamination curve K=1 → 3 → 5 → 10 per strategy ──────────
        # contamination(k) = 1 - precision(k) = 1 - recall(k) / k
        # At k=1: exact (= 1 - precision@1).  At k=K: exact (= contamination_rate).
        # Intermediate k values are estimated using the logarithmic recall curve.
        # Formula: recall(k) ≈ p1 + (r10 - p1) × log(k)/log(K)
        #          contamination(k) = 1 - recall(k) / k
        import math as _math
        import os as _os
        k_cfg = int(_os.environ.get("BENCHMARK_RECALL_K", "10"))
        k_points = [1, 2, 3, 5, k_cfg]
        x_k = np.arange(len(k_points))

        for i, s in enumerate(strategies):
            sub = [r for r in cells if r.retrieval_strategy == s]
            if not sub:
                continue
            p1_s  = _mean_std([r.precision_at_1    for r in sub])[0]  # recall@K=1
            r10_s = _mean_std([r.recall_at_k        for r in sub])[0]  # recall@K=k_cfg
            c10_s = _mean_std([r.contamination_rate for r in sub])[0]  # exact @K=k_cfg

            # Build contamination curve
            c_curve = []
            for kv in k_points:
                if kv == 1:
                    c_curve.append(1.0 - p1_s)  # exact
                elif kv >= k_cfg:
                    c_curve.append(c10_s)         # exact
                else:
                    # Estimate recall(k) via log interpolation
                    frac = _math.log(kv) / _math.log(k_cfg) if k_cfg > 1 else 1.0
                    recall_k = p1_s + (r10_s - p1_s) * frac
                    c_curve.append(max(0.0, min(1.0, 1.0 - recall_k / kv)))

            colour = _PALETTE[i % len(_PALETTE)]
            axes[1].plot(x_k, c_curve, "o-", label=s, color=colour,
                         linewidth=2.5, markersize=8)
            # Annotate K=1 and K=k_cfg endpoints
            axes[1].text(0, c_curve[0] - 0.018, f"{c_curve[0]:.2f}",
                         ha="center", fontsize=_ANNOT_SIZE - 1, color=colour,
                         fontweight="bold")
            axes[1].text(len(k_points) - 1, c_curve[-1] + 0.008, f"{c_curve[-1]:.2f}",
                         ha="center", fontsize=_ANNOT_SIZE - 1, color=colour,
                         fontweight="bold")

        # Expected floor at K=k_cfg: (K-1)/K for 1 gold per query
        axes[1].axhline((k_cfg - 1) / k_cfg, color="grey", linestyle="--",
                        linewidth=1.5, alpha=0.75,
                        label=f"Min possible @K={k_cfg}\n(1 gold per query)")
        axes[1].set_xticks(x_k)
        axes[1].set_xticklabels([f"K={kv}" for kv in k_points], fontsize=_TICK_SIZE)
        axes[1].legend(loc="lower right", fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
        axes[1].text(0.02, 0.98,
                     "Falling slope = strategy ranks correct\nresult near rank 1",
                     transform=axes[1].transAxes, fontsize=_ANNOT_SIZE - 1,
                     va="top", color="grey", style="italic")
        axes[1].set_ylim(0, 1.05)
        _style_ax(axes[1],
                  "Contamination Rate vs K  (lower = less noise in top-K)",
                  "K  (number of results returned)", "Contamination (fraction not relevant)")

        # ── Panel 3: P@1 + NDCG + F1 ─────────────────────────────────────────
        p1_vals   = [_mean_std([r.precision_at_1 for r in cells if r.retrieval_strategy == s])[0] for s in strategies]
        ndcg_vals = [_mean_std([r.ndcg           for r in cells if r.retrieval_strategy == s])[0] for s in strategies]
        f1_vals   = []
        for s in strategies:
            sub = [r for r in cells if r.retrieval_strategy == s]
            avg_r = _mean_std([r.recall_at_k    for r in sub])[0]
            avg_p = _mean_std([r.precision_at_k for r in sub])[0]
            f1_vals.append((2 * avg_r * avg_p / (avg_r + avg_p)) if (avg_r + avg_p) > 0 else 0.0)

        w3 = 0.22
        for i, (vals, label, colour) in enumerate([
            (p1_vals,   "P@1 (top-1 correct?)", _PALETTE[4]),
            (ndcg_vals, "NDCG (rank quality)",  _PALETTE[2]),
            (f1_vals,   "F1 (balance)",         _PALETTE[1]),
        ]):
            axes[2].bar(x + (i - 1) * w3, vals, w3 * 0.92,
                                label=label, color=colour, alpha=0.88)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(strategies, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[2].legend(fontsize=_LEGEND_SIZE - 1, framealpha=0.9, loc="upper right")
        axes[2].autoscale(axis="y", tight=False)
        _yl2, _yh2 = axes[2].get_ylim()
        axes[2].set_ylim(0, min(1.0, _yh2 + 0.05))
        axes[2].text(0.02, 0.98, "P@1 = chatbot top-1 accuracy",
                     transform=axes[2].transAxes, fontsize=_ANNOT_SIZE - 1,
                     va="top", color="grey", style="italic")
        _style_ax(axes[2], "Ranking Quality — P@1 / NDCG / F1",
                  "Strategy", "Score")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "noise_quality.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Efficiency: Recall-per-ms + Cost vs Recall Pareto ───────────────────

    def _plot_efficiency(self, plt, np) -> str:
        """Two efficiency views for deployment decision-making.

        Panel 1 — Recall-per-ms (efficiency ratio) per strategy.
          BM25 = ~0.05 recall/ms  vs  embedding = ~0.002 recall/ms
          Shows which strategy gives the most recall for the latency budget.

        Panel 2 — Cost vs Recall Pareto scatter.
          X = total_cost per cell  Y = recall_at_k
          Pareto frontier: no other config has both higher recall AND lower cost.
          Essential for API-backed or cloud-deployed systems.
        """
        if not self._results:
            return ""

        _decay_tags = {"phase4_decay_broad", "phase4_decay_fine",
                       "phase4_decay_sweep", "phase4b_archival_floor"}
        cells = [r for r in self._results
                 if getattr(r, "study_phase", "general") not in _decay_tags]
        strategies = sorted({r.retrieval_strategy for r in cells})

        # Extra width for the outside legend on the cost scatter panel
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(
            "Efficiency — Recall per ms & Cost vs Recall Pareto\n"
            "Which strategy gives the most recall for your latency and cost budget?",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        # ── Panel 1: Recall-per-ms efficiency ratio ───────────────────────────
        eff_vals = []
        for s in strategies:
            sub = [r for r in cells if r.retrieval_strategy == s]
            avg_r = _mean_std([r.recall_at_k for r in sub])[0]
            avg_lat = _mean_std([r.latency_p50_ms for r in sub])[0]
            eff_vals.append(avg_r / max(avg_lat, 0.001))

        x = np.arange(len(strategies))
        bars = axes[0].bar(x, eff_vals,
                           color=[_PALETTE[i % len(_PALETTE)] for i in range(len(strategies))],
                           alpha=0.88)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=35, ha="right", fontsize=_TICK_SIZE)
        _annotate_bars(axes[0], bars, fmt="{:.4f}")
        axes[0].text(0.02, 0.98, "Higher = more recall per millisecond of latency",
                     transform=axes[0].transAxes, fontsize=_ANNOT_SIZE - 1,
                     va="top", color="grey", style="italic")
        _style_ax(axes[0], "Efficiency — Recall@K / Latency P50 (ms)",
                  "Strategy", "Recall@K per ms")

        # ── Panel 2: Cost vs Recall Pareto scatter ────────────────────────────
        for i, s in enumerate(strategies):
            sub = [r for r in cells if r.retrieval_strategy == s]
            costs   = [r.total_cost    for r in sub]
            recalls = [r.recall_at_k   for r in sub]
            if recalls:
                axes[1].scatter(costs, recalls, label=s, s=30, alpha=0.7,
                                color=_PALETTE[i % len(_PALETTE)], edgecolors="none")
                # Mark mean
                mc, mr = _mean_std(costs)[0], _mean_std(recalls)[0]
                axes[1].scatter([mc], [mr], s=120, marker="D",
                                color=_PALETTE[i % len(_PALETTE)], edgecolors="black",
                                linewidths=1, zorder=5)
                axes[1].annotate(s, xy=(mc, mr), xytext=(4, 4),
                                 textcoords="offset points", fontsize=_ANNOT_SIZE - 1,
                                 fontweight="bold", color=_PALETTE[i % len(_PALETTE)])

        axes[1].set_xlabel("Total Cost per Cell (USD)", fontsize=_LABEL_SIZE)
        axes[1].set_ylabel("Recall@K", fontsize=_LABEL_SIZE)
        # Adaptive recall range — if all costs are 0 (local models) the scatter
        # collapses to a vertical line; adaptive y shows the recall spread clearly.
        axes[1].autoscale(axis="y", tight=False)
        _yl, _yh = axes[1].get_ylim()
        axes[1].set_ylim(max(0, _yl - 0.02), min(1.0, _yh + 0.02))
        # Legend outside plot area so it never covers data points.
        axes[1].legend(fontsize=_LEGEND_SIZE, framealpha=0.9,
                       loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axes[1].yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axes[1].set_axisbelow(True)
        axes[1].set_title("Cost vs Recall (◆ = per-strategy mean)",
                          fontsize=_TITLE_SIZE, fontweight="bold")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "efficiency.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Recall@K Variation — How K affects retrieval quality ────────────────

    def _plot_recall_k_variation(self, plt, np) -> str:
        """Two panels showing how the choice of K (top-N returned results) affects quality.

        Panel 1 — Recall@K curve per strategy.
          Uses the available metrics to reconstruct the recall-at-K curve:
            K=1  → precision_at_1   (fraction of queries where top-1 result is correct)
            K=10 → recall_at_k      (fraction of gold memories in top-10)
          MRR gives the expected rank of the first hit: estimated K for 50% recall ≈ 1/MRR.
          Shows how quickly each strategy "finds" the right answer as K grows.

        Panel 2 — Precision@K vs Recall@K tradeoff.
          Lower K = higher precision, lower recall.
          Higher K = lower precision, higher recall.
          Shows the operating point for each strategy.
        """
        if not self._results:
            return ""

        _decay_tags = {"phase4_decay_broad", "phase4_decay_fine",
                       "phase4_decay_sweep", "phase4b_archival_floor"}
        cells = [r for r in self._results
                 if getattr(r, "study_phase", "general") not in _decay_tags]
        strategies = sorted({r.retrieval_strategy for r in cells},
                             key=lambda s: -_mean_std([r.recall_at_k for r in cells
                                                        if r.retrieval_strategy == s])[0])

        fig, axes = plt.subplots(1, 2, figsize=(_fig_width_for_n(len(strategies), 13, 1.2), 5))
        fig.suptitle(
            "How K Affects Retrieval Quality — Top-1 vs Top-3 vs Top-5 vs Top-10\n"
            "P@1 = is the top result correct?  Recall@K = are the right memories in the top K?",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        # ── Panel 1: Recall@K curve estimated from P@1, MRR, and Recall@K ────
        k_values = [1, 2, 3, 5, 10]
        x = np.arange(len(k_values))

        for i, s in enumerate(strategies):
            sub = [r for r in cells if r.retrieval_strategy == s]
            if not sub:
                continue
            p1   = _mean_std([r.precision_at_1  for r in sub])[0]  # recall@1
            r10  = _mean_std([r.recall_at_k     for r in sub])[0]  # recall@10
            _mrr  = _mean_std([r.mrr             for r in sub])[0]

            # Build estimated recall curve using a logarithmic growth model:
            # recall(k) ≈ recall@1 + (recall@10 - recall@1) × log(k) / log(10)
            # This matches the typical sub-linear growth of recall with K.
            # At K=1: recall(1) = p1  (exact match)
            # At K=10: recall(10) = r10  (exact match)
            # The MRR informs the "steepness" — high MRR means fast early gains.
            if r10 > p1:
                est_curve = []
                for k in k_values:
                    import math
                    if k == 1:
                        est_curve.append(p1)
                    elif k >= 10:
                        est_curve.append(r10)
                    else:
                        # Logarithmic interpolation
                        frac = math.log(k) / math.log(10)
                        est_curve.append(p1 + (r10 - p1) * frac)
            else:
                est_curve = [p1] + [r10] * (len(k_values) - 1)

            colour = _PALETTE[i % len(_PALETTE)]
            axes[0].plot(x, est_curve, "o-", label=s, color=colour,
                         linewidth=2.5, markersize=8)
            # Annotate K=1 and K=10 endpoints
            axes[0].text(0, est_curve[0] + 0.008, f"{est_curve[0]:.3f}",
                         ha="center", fontsize=_ANNOT_SIZE - 1, color=colour)
            axes[0].text(len(k_values) - 1, est_curve[-1] + 0.008,
                         f"{est_curve[-1]:.3f}",
                         ha="center", fontsize=_ANNOT_SIZE - 1, color=colour)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"K={k}" for k in k_values], fontsize=_TICK_SIZE)
        axes[0].legend(loc="upper left", fontsize=_LEGEND_SIZE, framealpha=0.9)
        axes[0].autoscale(axis="y", tight=False)
        _yl, _yh = axes[0].get_ylim()
        axes[0].set_ylim(max(0, _yl - 0.02), min(1.0, _yh + 0.02))
        _style_ax(axes[0], "Recall@K Curve — How Recall Grows as K Increases",
                  "K  (number of results returned)", "Recall@K")
        axes[0].text(0.98, 0.05,
                     "Steeper rise = strategy ranks\ngold memories earlier",
                     transform=axes[0].transAxes, ha="right", fontsize=_ANNOT_SIZE - 1,
                     color="grey", style="italic")

        # ── Panel 2: Precision@K vs Recall@K operating point per strategy ────
        # Each strategy shows one point: (recall@K, precision@K) at configured K.
        # The K=1 operating point (precision@1, recall@1) is a second point per strategy.
        _markers = ["o", "s", "^", "D", "P", "*", "X"]
        for i, s in enumerate(strategies):
            sub = [r for r in cells if r.retrieval_strategy == s]
            if not sub:
                continue
            r10  = _mean_std([r.recall_at_k     for r in sub])[0]
            p10  = _mean_std([r.precision_at_k  for r in sub])[0]
            p1   = _mean_std([r.precision_at_1  for r in sub])[0]
            colour = _PALETTE[i % len(_PALETTE)]
            mk = _markers[i % len(_markers)]

            # K=10 point
            axes[1].scatter(r10, p10, s=130, marker=mk, color=colour,
                            zorder=4, edgecolors="white", linewidths=1)
            axes[1].annotate(f"{s}\nK=10", (r10, p10),
                             xytext=(5, 5), textcoords="offset points",
                             fontsize=_ANNOT_SIZE - 1, color=colour)

            # K=1 point (open marker)
            axes[1].scatter(p1, p1, s=70, marker=mk, color=colour,
                            zorder=3, edgecolors=colour, linewidths=1.5,
                            facecolors="white", alpha=0.9)

            # Connect K=1 to K=10 with a thin line
            axes[1].plot([p1, r10], [p1, p10], "-",
                         color=colour, linewidth=0.8, alpha=0.4)

        # Reference diagonal: precision = recall / K
        axes[1].text(0.98, 0.98, "● = K=10   ○ = K=1",
                     transform=axes[1].transAxes, ha="right", va="top",
                     fontsize=_ANNOT_SIZE - 1, color="grey")
        axes[1].autoscale(tight=False)
        _style_ax(axes[1],
                  "Precision@K vs Recall@K  (K=1 open, K=10 filled)",
                  "Recall@K", "Precision@K")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axes[1].yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axes[1].set_axisbelow(True)

        plt.tight_layout(pad=2.0)
        path = str(self._out / "recall_k_variation.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Resource Usage: RAM, CPU, Duration ──────────────────────────────────

    def _plot_resource_usage(self, plt, np) -> str:
        """System resource consumption per strategy.

        Shows peak RAM, average CPU, and wall-clock duration — so you can
        see whether a high-recall strategy is affordable on your hardware.
        Embedding strategies use 3-5× more RAM than BM25; this chart makes
        that tradeoff explicit.

        Uses horizontal bars so strategy names are fully readable regardless
        of how many strategies are in the run.
        """
        if not self._results:
            return ""

        # Sort by peak RAM so the most resource-intensive strategy is at the top
        strategies_all = sorted({r.retrieval_strategy for r in self._results})
        ram_vals = {s: _mean_std([r.peak_ram_mb for r in self._results
                                  if r.retrieval_strategy == s])[0] for s in strategies_all}
        strategies = sorted(strategies_all, key=lambda s: ram_vals[s], reverse=True)

        n = len(strategies)
        row_h = max(0.55, 4.5 / max(n, 1))  # each strategy row gets breathing room
        fig_h = max(5, n * row_h + 1.5)
        fig, axes = plt.subplots(1, 3, figsize=(16, fig_h))
        fig.suptitle(
            "System Resource Usage per Strategy  (sorted by peak RAM)\n"
            "Can your hardware afford the best-recall strategy?",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        y = np.arange(n)

        for ax, field, xlabel, panel_title, colour in [
            (axes[0], "peak_ram_mb",     "Peak RAM (MB)",      "Peak RAM Usage",      _PALETTE[0]),
            (axes[1], "avg_cpu_percent", "Avg CPU %",          "Average CPU Load",    _PALETTE[2]),
            (axes[2], "duration_seconds","Duration (s)",       "Wall-Clock Duration", _PALETTE[4]),
        ]:
            vals = [_mean_std([getattr(r, field, 0.0) for r in self._results
                               if r.retrieval_strategy == s])[0] for s in strategies]
            ax.barh(y, vals, color=colour, alpha=0.88, height=0.65)
            # Annotate value at end of each bar
            for yi, v in zip(y, vals):
                if v > 0.001:
                    ax.text(v + max(vals) * 0.01, yi, f"{v:.1f}",
                            va="center", fontsize=_ANNOT_SIZE, fontweight="bold")
            ax.set_yticks(y)
            ax.set_yticklabels(strategies, fontsize=_TICK_SIZE)
            ax.invert_yaxis()  # highest RAM at top
            ax.set_xlabel(xlabel, fontsize=_LABEL_SIZE)
            ax.set_title(panel_title, fontsize=_TITLE_SIZE, fontweight="bold", pad=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout(pad=2.0)
        path = str(self._out / "resource_usage.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Bootstrap CI Comparison with Significance Markers ───────────────────

    def _plot_ci_comparison(self, plt, np) -> str:
        """Strategy comparison with 95% bootstrap CI error bars.

        The leaderboard shows composite scores but hides uncertainty.
        This chart shows mean ± CI for each strategy — strategies whose
        CIs overlap are NOT statistically distinguishable. A '★' marks
        pairs where CIs don't overlap (p < 0.05 approx.).
        Also shows: correct query count annotation (e.g. '1468/1977').
        """
        if not self._results:
            return ""

        _decay_tags = {"phase4_decay_broad", "phase4_decay_fine",
                       "phase4_decay_sweep", "phase4b_archival_floor"}
        cells = [r for r in self._results
                 if getattr(r, "study_phase", "general") not in _decay_tags]
        strategies = sorted({r.retrieval_strategy for r in cells},
                             key=lambda s: -_mean_std([r.recall_at_k for r in cells
                                                        if r.retrieval_strategy == s])[0])

        fig, axes = plt.subplots(1, 2, figsize=(_fig_width_for_n(len(strategies), 12, 1.5), 5))
        fig.suptitle(
            "Statistical Comparison — Recall@K with 95% Bootstrap CI\n"
            "★ = CIs do not overlap (p < 0.05 approx.) · Overlap = not statistically different",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        x = np.arange(len(strategies))

        # ── Panel 1: Recall with CI error bars ───────────────────────────────
        means, ci_lows, ci_highs, total_q, correct_q = [], [], [], [], []
        for s in strategies:
            sub = [r for r in cells if r.retrieval_strategy == s]
            vals = [r.recall_at_k for r in sub]
            m, _ = _mean_std(vals)
            n = len(vals)
            # Simple bootstrap CI (fast approximation)
            if n >= 2:
                rng = np.random.default_rng(42)
                arr = np.array(vals)
                boots = rng.choice(arr, size=(500, n), replace=True).mean(axis=1)
                boots.sort()
                ci_lo = float(boots[int(0.025 * 500)])
                ci_hi = float(boots[int(0.975 * 500) - 1])
            else:
                ci_lo = ci_hi = m
            means.append(m)
            ci_lows.append(m - ci_lo)
            ci_highs.append(ci_hi - m)
            total_q.append(sum(r.total_queries for r in sub))
            correct_q.append(sum(r.correct_recalls for r in sub))

        axes[0].bar(x, means,
                    color=[_PALETTE[i % len(_PALETTE)] for i in range(len(strategies))],
                    alpha=0.88,
                    yerr=[ci_lows, ci_highs], capsize=6,
                    error_kw={"elinewidth": 2, "ecolor": "black", "alpha": 0.7})
        # Significance markers and query count annotations
        for xi, (_m, _lo, _hi, tq, cq) in enumerate(zip(means, ci_lows, ci_highs, total_q, correct_q)):
            # Correct/total annotation
            if tq > 0:
                axes[0].text(xi, -0.04, f"{cq}/{tq}",
                             ha="center", fontsize=_ANNOT_SIZE - 1, color="grey",
                             transform=axes[0].get_xaxis_transform())
        # Mark significance between adjacent ranked strategies
        for i in range(len(strategies) - 1):
            ci_high_i  = means[i]   + ci_highs[i]
            ci_low_i1  = means[i+1] - ci_lows[i+1]
            if means[i] > means[i+1] and ci_low_i1 > ci_high_i:
                mid_x = (x[i] + x[i + 1]) / 2
                axes[0].text(mid_x, max(means[i], means[i+1]) + 0.03, "★",
                             ha="center", fontsize=12, color="green", fontweight="bold")

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strategies, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[0].set_ylim(0, 1.05)
        axes[0].text(0.5, -0.12, "Number below bar = correct/total queries across all cells",
                     ha="center", transform=axes[0].transAxes,
                     fontsize=_ANNOT_SIZE - 1, color="grey", style="italic")
        _style_ax(axes[0], "Recall@K with 95% Bootstrap CI",
                  "Strategy", "Recall@K")

        # ── Panel 2: Latency mean vs P50 gap (outlier detector) ──────────────
        for i, s in enumerate(strategies):
            sub = [r for r in cells if r.retrieval_strategy == s]
            p50_m  = _mean_std([r.latency_p50_ms  for r in sub])[0]
            mean_m = _mean_std([r.latency_mean_ms for r in sub])[0]
            p99_m  = _mean_std([r.latency_p99_ms  for r in sub])[0]
            colour = _PALETTE[i % len(_PALETTE)]
            axes[1].plot([i, i, i], [p50_m, mean_m, p99_m],
                         "_", markersize=18, color=colour, markeredgewidth=2)
            axes[1].plot([i - 0.2, i + 0.2], [p50_m, p50_m], "-",
                         color=colour, linewidth=2.5, label=s if i == 0 else "")
            axes[1].plot([i - 0.15, i + 0.15], [mean_m, mean_m], "--",
                         color=colour, linewidth=1.5, alpha=0.7)
            axes[1].plot([i - 0.1, i + 0.1], [p99_m, p99_m], ":",
                         color=colour, linewidth=1.5, alpha=0.5)
            if mean_m > p50_m * 2:
                axes[1].annotate("⚠ outliers", xy=(i, mean_m),
                                 xytext=(8, 4), textcoords="offset points",
                                 fontsize=_ANNOT_SIZE - 1, color="red")

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategies, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[1].text(0.02, 0.98,
                     "─ P50   - - mean   ··· P99\n⚠ = mean >> P50 (outlier queries)",
                     transform=axes[1].transAxes, fontsize=_ANNOT_SIZE - 1,
                     va="top", color="grey", family="monospace")
        _style_ax(axes[1], "Latency Distribution — P50 / Mean / P99",
                  "Strategy", "Latency (ms)")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "ci_comparison.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Decay Curves — Lambda × Policy Line Plots ───────────────────────────

    def _plot_decay_curves(self, plt, np) -> str:
        """Recall@K and MRR as continuous curves vs lambda, one line per policy.

        More readable than the heatmap for seeing the shape of the recall-lambda
        relationship: where does it peak? where does it collapse?
        """
        decay_r = [r for r in self._results if r.lambda_value > 0]
        if not decay_r:
            return ""

        policies = sorted({r.decay_policy for r in decay_r})
        lambdas  = sorted({r.lambda_value for r in decay_r})
        if len(lambdas) < 2:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Decay Policy — Recall@K & MRR vs Lambda (λ)\n"
            "How does recall change as you tune the forgetting rate?",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        # No-decay baseline for reference
        no_decay = [r for r in self._results if r.lambda_value == 0.0
                    and r.retrieval_strategy not in ("recency",)]

        for ax_idx, (metric, ylabel) in enumerate([
            ("recall_at_k", "Recall@K"),
            ("mrr",         "MRR"),
        ]):
            ax = axes[ax_idx]

            # Baseline reference line — no shading, just a clean dashed line
            if no_decay:
                base_m = _mean_std([getattr(r, metric) for r in no_decay])[0]
                ax.axhline(base_m, color="grey", linestyle="--", linewidth=1.8,
                           label=f"No-decay baseline ({base_m:.3f})", alpha=0.85)

            for i, policy in enumerate(policies):
                pts = []
                for lam in lambdas:
                    vals = [getattr(r, metric) for r in decay_r
                            if r.decay_policy == policy and abs(r.lambda_value - lam) < 1e-9]
                    if vals:
                        m = sum(vals) / len(vals)  # mean only — no shading (single seed)
                        pts.append((lam, m))
                if len(pts) < 2:
                    continue
                xs, ms = zip(*pts)
                ax.semilogx(xs, ms, "o-", label=policy, color=_PALETTE[i],
                            linewidth=2.5, markersize=8)
                # Annotate best λ value
                best_lam, best_m = max(zip(xs, ms), key=lambda t: t[1])
                ax.annotate(f"★ {best_lam:.4f}",
                            xy=(best_lam, best_m),
                            xytext=(best_lam * 1.3, best_m + 0.003),
                            fontsize=_ANNOT_SIZE - 1, color=_PALETTE[i],
                            fontweight="bold",
                            arrowprops={"arrowstyle": "->", "color": _PALETTE[i], "lw": 1})

            ax.legend(loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9, ncol=2)
            # Adaptive y-axis: zoom to actual data range so the curve
            # fills the plot. If values are 0.2–0.4 a 0–1 scale wastes
            # 60% of chart space on empty whitespace.
            ax.autoscale(axis="y", tight=False)
            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(max(0.0, y_lo - 0.02), min(1.0, y_hi + 0.02))
            _style_ax(ax, f"{ylabel} vs Lambda — Per Decay Policy",
                      "Lambda (λ)  [log scale]", ylabel)
            ax.set_xlim(min(lambdas) * 0.7, max(lambdas) * 1.5)

        plt.tight_layout(pad=2.0)
        path = str(self._out / "decay_curves.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Parameter Sensitivity Dashboard ─────────────────────────────────────

    def _plot_parameter_sensitivity(self, plt, np) -> str:
        """4-panel dashboard: how does changing each parameter move recall?

        Panel 1 — BM25 weight sweep (Recall@K episodic + composite, dual axis)
        Panel 2 — Embedding model lift over BM25 baseline (delta bars)
        Panel 3 — Decay lambda curves (recall vs lambda per policy, log x-axis)
        Panel 4 — Strategy comparison (recall + MRR + latency P50 per strategy)

        This is the "what-if" view — change a knob, see what moves.
        """
        if not self._results:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle(
            "Parameter Sensitivity — How Each Knob Moves Recall & Composite\n"
            "Change a parameter on the x-axis, read the metric change on the y-axis",
            fontsize=_TITLE_SIZE + 2, fontweight="bold",
        )

        # ── Panel 1: BM25 weight → Recall (episodic only, most meaningful) ──
        ax1 = axes[0, 0]
        hybrid = [r for r in self._results
                  if r.retrieval_strategy == "hybrid" and r.memory_type == "episodic"]
        if not hybrid:
            hybrid = [r for r in self._results if r.retrieval_strategy == "hybrid"]
        weights = sorted({r.bm25_weight for r in hybrid})
        if len(weights) >= 2:
            r_pts, c_pts = [], []
            for w in weights:
                vals = [r for r in hybrid if abs(r.bm25_weight - w) < 1e-9]
                if vals:
                    rm, rs = _mean_std([r.recall_at_k for r in vals])
                    cm, _  = _mean_std([r.composite_score() for r in vals])
                    r_pts.append((w, rm, rs))
                    c_pts.append((w, cm))
            if r_pts:
                ws, rms, _ = zip(*r_pts)
                ax1.plot(ws, rms, "o-", color=_PALETTE[0], linewidth=2.5,
                         markersize=9, label="Recall@K (episodic)")
                # Shading removed: single seed, ±std not meaningful
                # Best weight marker
                best_w = weights[list(rms).index(max(rms))]
                ax1.axvline(best_w, color="red", linestyle="--", linewidth=1.5,
                            label=f"Best weight: {best_w:.2f}", alpha=0.8)
            if c_pts:
                ax1c = ax1.twinx()
                cws, cms = zip(*c_pts)
                ax1c.plot(cws, cms, "s--", color=_PALETTE[3], linewidth=1.5,
                          markersize=6, label="Composite", alpha=0.85)
                ax1c.set_ylabel("Composite Score", color=_PALETTE[3],
                                fontsize=_LABEL_SIZE - 1)
                ax1c.tick_params(axis="y", colors=_PALETTE[3], labelsize=_TICK_SIZE - 1)
                ax1c.set_ylim(0, 1.0)
            ax1.legend(loc="upper left", fontsize=_LEGEND_SIZE, framealpha=0.9)
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(0, 1.0)
        ax1.axvline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.4)
        ax1.text(0.02, 0.98, "← pure semantic", transform=ax1.transAxes,
                 fontsize=_ANNOT_SIZE - 1, va="top", color="grey")
        ax1.text(0.98, 0.98, "pure BM25 →", transform=ax1.transAxes,
                 fontsize=_ANNOT_SIZE - 1, va="top", ha="right", color="grey")
        _style_ax(ax1, "BM25 Weight → Recall@K & Composite",
                  "BM25 Weight  (0=semantic, 1=keyword)", "Recall@K")

        # ── Panel 2: Embedding model → recall lift over BM25 baseline ──────
        ax2 = axes[0, 1]
        bm25_baseline = _mean_std([r.recall_at_k for r in self._results
                                   if r.retrieval_strategy == "bm25"])[0]
        embed_r = [r for r in self._results
                   if r.retrieval_strategy in ("embeddings", "semantic", "api_embeddings")]
        if embed_r and bm25_baseline > 0:
            model_recalls: dict[str, list] = {}
            model_lats: dict[str, list] = {}
            for r in embed_r:
                m = r.embedding_model.split("/")[-1][:18]
                model_recalls.setdefault(m, []).append(r.recall_at_k)
                model_lats.setdefault(m, []).append(r.latency_p50_ms)
            models_sorted = sorted(model_recalls.keys(),
                                   key=lambda m: _mean_std(model_recalls[m])[0],
                                   reverse=True)
            x = np.arange(len(models_sorted))
            deltas = [_mean_std(model_recalls[m])[0] - bm25_baseline
                      for m in models_sorted]
            stds   = [_mean_std(model_recalls[m])[1] for m in models_sorted]
            colours = [_PALETTE[0] if d >= 0 else _PALETTE[3] for d in deltas]
            ax2.bar(x, deltas, color=colours, alpha=0.88,
                           yerr=stds, capsize=4,
                           error_kw={"elinewidth": 1.2, "ecolor": "black"})
            ax2.axhline(0, color="grey", linewidth=1.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(models_sorted, rotation=35, ha="right",
                                fontsize=_TICK_SIZE)
            for xi, (d, _model) in enumerate(zip(deltas, models_sorted)):
                sign = "+" if d >= 0 else ""
                ax2.text(xi, d + (0.002 if d >= 0 else -0.005),
                         f"{sign}{d:.3f}", ha="center", va="bottom" if d >= 0 else "top",
                         fontsize=_ANNOT_SIZE, fontweight="bold")
            _style_ax(ax2,
                      f"Embedding Model → Recall Lift vs BM25 ({bm25_baseline:.3f})",
                      "Embedding Model", "Δ Recall@K vs BM25 baseline")
            ax2.text(0.98, 0.98, f"BM25 baseline = {bm25_baseline:.3f}",
                     transform=ax2.transAxes, ha="right", va="top",
                     fontsize=_ANNOT_SIZE, style="italic", color="grey")
        else:
            ax2.text(0.5, 0.5, "No embedding data", ha="center", va="center",
                     transform=ax2.transAxes)
            _style_ax(ax2, "Embedding Model Lift")

        # ── Panel 3: Lambda → Recall (log scale, per policy) ────────────────
        ax3 = axes[1, 0]
        decay_r = [r for r in self._results if r.lambda_value > 0]
        policies = sorted({r.decay_policy for r in decay_r})
        lambdas  = sorted({r.lambda_value for r in decay_r})
        if len(lambdas) >= 2 and policies:
            base_vals = [r.recall_at_k for r in self._results
                         if r.lambda_value == 0.0
                         and r.retrieval_strategy not in ("recency",)]
            if base_vals:
                bm, _bs = _mean_std(base_vals)
                ax3.axhline(bm, color="grey", linestyle="--", linewidth=1.5,
                            label=f"No-decay ({bm:.3f})", alpha=0.8)
            for i, policy in enumerate(policies):
                pts = [(lam, _mean_std([r.recall_at_k for r in decay_r
                        if r.decay_policy == policy and abs(r.lambda_value - lam) < 1e-9]))
                       for lam in lambdas]
                pts = [(lam, m, s) for lam, (m, s) in pts if m > 0]
                if len(pts) < 2:
                    continue
                xs, ms, _ = zip(*pts)
                ax3.semilogx(xs, ms, "o-", label=policy, color=_PALETTE[i],
                             linewidth=2.5, markersize=8)
                # Shading removed: single seed, ±std not meaningful
            ax3.legend(loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
            ax3.set_xlim(min(lambdas) * 0.7, max(lambdas) * 1.5)
            # Adaptive: zoom to actual range so curves fill the panel
            ax3.autoscale(axis="y", tight=False)
            _y_lo, _y_hi = ax3.get_ylim()
            ax3.set_ylim(max(0.0, _y_lo - 0.02), min(1.0, _y_hi + 0.02))
        _style_ax(ax3, "Lambda (λ) → Recall@K per Decay Policy",
                  "Lambda (λ)  [log scale — larger = faster forgetting]",
                  "Recall@K")
        ax3.text(0.02, 0.02,
                 "← low λ = slow decay    high λ = fast decay →",
                 transform=ax3.transAxes, fontsize=_ANNOT_SIZE - 1,
                 va="bottom", color="grey")

        # ── Panel 4: Strategy → Recall + MRR + P50 (grouped bars + twin) ───
        ax4 = axes[1, 1]
        _decay_tags = {"phase4_decay_broad", "phase4_decay_fine",
                       "phase4_decay_sweep", "phase4b_archival_floor"}
        strat_cells = [r for r in self._results
                       if getattr(r, "study_phase", "general") not in _decay_tags]
        strategies = sorted({r.retrieval_strategy for r in strat_cells})
        if strategies:
            x = np.arange(len(strategies))
            w = 0.28
            r_vals = [_mean_std([r.recall_at_k for r in strat_cells
                                 if r.retrieval_strategy == s])[0] for s in strategies]
            m_vals = [_mean_std([r.mrr for r in strat_cells
                                 if r.retrieval_strategy == s])[0] for s in strategies]
            ax4.bar(x - w/2, r_vals, w * 0.92,
                             label="Recall@K", color=_PALETTE[0], alpha=0.88)
            ax4.bar(x + w/2, m_vals, w * 0.92,
                             label="MRR", color=_PALETTE[2], alpha=0.88)
            # Latency P50 on twin axis
            lat_vals = [_mean_std([r.latency_p50_ms for r in strat_cells
                                   if r.retrieval_strategy == s])[0] for s in strategies]
            ax4b = ax4.twinx()
            ax4b.plot(x, lat_vals, "D--", color="darkorange",
                      linewidth=2, markersize=8, label="Latency P50")
            ax4b.set_ylabel("Latency P50 (ms)", color="darkorange",
                            fontsize=_LABEL_SIZE - 1)
            ax4b.tick_params(axis="y", colors="darkorange", labelsize=_TICK_SIZE - 1)
            # Annotate recall values
            for xi, rv in zip(x, r_vals):
                ax4.text(xi - w/2, rv + 0.008, f"{rv:.3f}",
                         ha="center", fontsize=_ANNOT_SIZE - 1, fontweight="bold")
            ax4.set_xticks(x)
            ax4.set_xticklabels(strategies, rotation=35, ha="right",
                                fontsize=_TICK_SIZE)
            # Combined legend
            handles1, labels1 = ax4.get_legend_handles_labels()
            handles2, labels2 = ax4b.get_legend_handles_labels()
            ax4.legend(handles1 + handles2, labels1 + labels2,
                       fontsize=_LEGEND_SIZE, framealpha=0.9, loc="upper right")
            ax4.set_ylim(0, 1.0)
        _style_ax(ax4, "Strategy → Recall@K & MRR & Latency P50",
                  "Retrieval Strategy", "Recall@K  /  MRR")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = str(self._out / "parameter_sensitivity.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Composite Score Breakdown & Latency Sensitivity ─────────────────────

    def _plot_composite_breakdown(self, plt, np) -> str:
        """Two-panel chart showing composite score variation across all configurations.

        Panel 1 — Scatter: composite score vs Recall@K for every cell.
          Each point is coloured by strategy. Shows how recall drives composite
          and which non-recall components lift or drag specific configs.

        Panel 2 — Latency P50/P90/P99 per strategy.
          Shows typical and tail latency so a config isn't chosen for recall
          alone without seeing its 99th-percentile cost.
        """
        if not self._results:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Composite Score Analysis — Variation, Recall Driver, Latency Tail",
            fontsize=_TITLE_SIZE + 1, fontweight="bold",
        )

        # ── Panel 1: Composite vs Recall@K scatter (coloured by strategy) ────
        strategies = sorted({r.retrieval_strategy for r in self._results})
        for i, strat in enumerate(strategies):
            pts = [(r.recall_at_k, r.composite_score()) for r in self._results
                   if r.retrieval_strategy == strat]
            if pts:
                xs, ys = zip(*pts)
                axes[0].scatter(xs, ys, label=strat, color=_PALETTE[i % len(_PALETTE)],
                                s=28, alpha=0.75, edgecolors="none")

        # Reference line: composite ≈ recall (if precision=MRR=temporal ≈ recall)
        # Adaptive axes: zoom to actual data range so points fill the chart
        all_x = [r.recall_at_k for r in self._results]
        all_y = [r.composite_score() for r in self._results]
        if all_x and all_y:
            pad_x = (max(all_x) - min(all_x)) * 0.1 or 0.05
            pad_y = (max(all_y) - min(all_y)) * 0.1 or 0.05
            axes[0].set_xlim(max(0, min(all_x) - pad_x), min(1, max(all_x) + pad_x))
            axes[0].set_ylim(max(0, min(all_y) - pad_y), min(1, max(all_y) + pad_y))
            # Reference line within the data window
            xmin, xmax = axes[0].get_xlim()
            axes[0].plot([xmin, xmax], [xmin, xmax], "--", color="grey",
                         linewidth=1, alpha=0.5, label="y = x")
        axes[0].legend(title="Strategy", loc="upper left", fontsize=_LEGEND_SIZE,
                       framealpha=0.9, markerscale=1.5)
        _style_ax(axes[0],
                  "Composite Score vs Recall@K  (each dot = one cell)",
                  "Recall@K", "Composite Score")

        # ── Panel 2: Latency P50 / P90 / P99 grouped bars per strategy ───────
        lat_percs = [
            ("latency_p50_ms", "P50", _PALETTE[0]),
            ("latency_p90_ms", "P90", _PALETTE[1]),
            ("latency_p99_ms", "P99", _PALETTE[3]),
        ]
        x = np.arange(len(strategies))
        w = 0.24
        for pi, (field, plabel, colour) in enumerate(lat_percs):
            vals = [
                _mean_std([getattr(r, field, 0.0) for r in self._results
                           if r.retrieval_strategy == strat])[0]
                for strat in strategies
            ]
            axes[1].bar(x + (pi - 1) * w, vals, w * 0.88,
                        label=plabel, color=colour, alpha=0.88)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategies, rotation=35, ha="right", fontsize=_TICK_SIZE)
        axes[1].legend(title="Percentile", loc="upper right", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1],
                  "Query Latency P50 / P90 / P99 by Strategy",
                  "Strategy", "Latency (ms)")

        plt.tight_layout(pad=2.0)
        path = str(self._out / "composite_breakdown.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Full Composite Report ────────────────────────────────────────────────

    def _plot_full_report(self, plt, gridspec, np) -> str:
        fig = plt.figure(figsize=(22, 26))
        fig.suptitle(
            "MemTuner — Comprehensive Study Report",
            fontsize=16, fontweight="bold", y=0.995
        )
        fig.text(
            0.5, 0.988,
            "BM25 Baseline  ·  Embedding Models  ·  Hybrid Weights + Composite  ·  "
            "Rerankers P50/P90/P99  ·  Decay Sweep  ·  Composite Breakdown  ·  Leaderboard",
            ha="center", fontsize=10, style="italic",
        )

        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.40)

        self._render_bm25_panel(        fig.add_subplot(gs[0, 0]), plt, np)
        self._render_embedding_panel(   fig.add_subplot(gs[0, 1]), plt, np)
        self._render_hybrid_panel(      fig.add_subplot(gs[1, 0]), plt, np)
        self._render_reranker_panel(    fig.add_subplot(gs[1, 1]), plt, np)
        self._render_decay_panel(       fig.add_subplot(gs[2, 0]), plt, np)
        self._render_per_dataset_panel( fig.add_subplot(gs[2, 1]), plt, np)
        # Leaderboard spans full bottom row
        ax_lb = fig.add_subplot(gs[3, :])
        self._render_leaderboard_panel(ax_lb, plt, np)

        path = str(self._out / "study_report.png")
        fig.savefig(path, dpi=_DPI_REPORT, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Inline panel renderers ───────────────────────────────────────────────

    def _render_bm25_panel(self, ax, plt, np):
        bm25 = [r for r in self._results if r.retrieval_strategy == "bm25"]
        if not bm25:
            ax.text(0.5, 0.5, "No BM25 data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "BM25 Baseline")
            return

        by_mem: dict[str, list] = defaultdict(list)
        for r in bm25:
            by_mem[r.memory_type].append(r.recall_at_k)
        mem_types = sorted(by_mem.keys())
        means = [_mean_std(by_mem[m])[0] for m in mem_types]
        bars  = ax.bar(mem_types, means, color=_PALETTE[:len(mem_types)], alpha=0.88)
        _annotate_bars(ax, bars)
        _style_ax(ax, "BM25 Baseline — Recall@K", "Memory Type", "Recall@K", ylim=(0, 1.0))

    def _render_embedding_panel(self, ax, plt, np):
        embed = [r for r in self._results
                 if r.retrieval_strategy in ("embeddings", "api_embeddings", "semantic", "colbert", "adaptive")]
        if not embed:
            ax.text(0.5, 0.5, "No embedding data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "Embedding Models")
            return

        models = sorted({r.embedding_model for r in embed})
        short  = [m.split("/")[-1][:14] for m in models]
        by_m_r = defaultdict(list)
        by_m_l = defaultdict(list)
        for r in embed:
            by_m_r[r.embedding_model].append(r.recall_at_k)
            by_m_l[r.embedding_model].append(r.latency_p50_ms)

        x     = np.arange(len(models))
        means = [_mean_std(by_m_r[m])[0] for m in models]
        ax.bar(x, means, color=_PALETTE[:len(models)], alpha=0.88)
        ax2 = ax.twinx()
        lats = [_mean_std(by_m_l[m])[0] for m in models]
        ax2.plot(x, lats, "D--", color="darkorange", linewidth=1.5, markersize=6)
        ax2.set_ylabel("Latency P50 (ms)", color="darkorange", fontsize=_LABEL_SIZE - 1)
        ax2.tick_params(axis="y", colors="darkorange", labelsize=_TICK_SIZE - 1)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=_TICK_SIZE - 1)
        _style_ax(ax, "Embedding: Recall@K & Latency", "Model", "Recall@K", ylim=(0, 1.0))

    def _render_hybrid_panel(self, ax, plt, np):
        hybrid = [r for r in self._results if r.retrieval_strategy == "hybrid"]
        if not hybrid:
            ax.text(0.5, 0.5, "No hybrid data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "Hybrid Weight Sweep")
            return

        mem_types = sorted({r.memory_type for r in hybrid})
        for i, mem in enumerate(mem_types):
            pts = sorted(
                [(r.bm25_weight, r.recall_at_k) for r in hybrid if r.memory_type == mem],
                key=lambda x: x[0],
            )
            if pts:
                ws, recalls = zip(*pts)
                ax.plot(ws, recalls, "o-", label=mem, color=_PALETTE[i],
                        linewidth=2, markersize=7)
        ax.legend(loc="upper right", fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
        _style_ax(ax, "Hybrid: Recall vs BM25 Weight", "BM25 Weight", "Recall@K")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(bottom=0)

    def _render_reranker_panel(self, ax, plt, np):
        rerank = [r for r in self._results if r.study_phase == "phase_reranker_comparison"]
        if not rerank:
            ax.text(0.5, 0.5, "No reranker data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "Reranker Comparison")
            return

        rerankers = sorted({r.reranker_model for r in rerank})
        short     = [rr.split("/")[-1][:13] if rr != "none" else "none" for rr in rerankers]
        x = np.arange(len(rerankers))
        w = 0.22
        for j, (metric, label, colour) in enumerate([
            ("recall_at_k",    "Recall",    _PALETTE[0]),
            ("precision_at_k", "Precision", _PALETTE[1]),
            ("mrr",            "MRR",       _PALETTE[2]),
        ]):
            vals = [
                _mean_std([getattr(r, metric) for r in rerank if r.reranker_model == rr])[0]
                for rr in rerankers
            ]
            ax.bar(x + (j - 1) * w, vals, w * 0.88, label=label, color=colour, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=_TICK_SIZE - 1)
        ax.legend(loc="upper right", fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
        _style_ax(ax, "Reranker: Recall / Precision / MRR", "Reranker", "Score", ylim=(0, 1.0))

    def _render_decay_panel(self, ax, plt, np):
        decay_r = [r for r in self._results if r.study_phase == "phase_decay_sweep" and r.lambda_value > 0]
        if not decay_r:
            decay_r = [r for r in self._results if r.lambda_value > 0]
        if not decay_r:
            ax.text(0.5, 0.5, "No decay data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "Decay Sweep")
            return

        policies = sorted({r.decay_policy for r in decay_r})
        lambdas  = sorted({r.lambda_value for r in decay_r})
        grid = np.zeros((len(policies), len(lambdas)))
        for pi, policy in enumerate(policies):
            for li, lam in enumerate(lambdas):
                vals = [r.recall_at_k for r in decay_r
                        if r.decay_policy == policy and abs(r.lambda_value - lam) < 1e-9]
                if vals:
                    grid[pi, li] = sum(vals) / len(vals)

        vmax = max(float(grid.max()), 0.01)
        ax.imshow(grid, aspect="auto", cmap="YlGn", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(lambdas)))
        ax.set_xticklabels([f"{lam_val:.3f}" for lam_val in lambdas], rotation=40, ha="right", fontsize=_TICK_SIZE - 1)
        ax.set_yticks(np.arange(len(policies)))
        ax.set_yticklabels(policies, fontsize=_TICK_SIZE - 1)
        ax.set_xlabel("Lambda (λ)", fontsize=_LABEL_SIZE - 1)
        ax.set_title("Recall@K — Decay × Lambda", fontsize=_TITLE_SIZE, fontweight="bold")
        for pi in range(len(policies)):
            for li in range(len(lambdas)):
                v = grid[pi, li]
                tc = "white" if v > 0.65 * vmax else "black"
                ax.text(li, pi, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)

    def _render_per_dataset_panel(self, ax, plt, np):
        by_ds: dict[str, list] = defaultdict(list)
        for r in self._results:
            ds = r.run_id.split(":")[0] if ":" in r.run_id else "default"
            by_ds[ds].append(r)

        if len(by_ds) <= 1:
            ax.text(0.5, 0.5, "Single dataset — no cross-dataset comparison",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            _style_ax(ax, "Per-Dataset Comparison")
            return

        datasets   = sorted(by_ds.keys())
        strategies = sorted({r.retrieval_strategy for r in self._results})
        x = np.arange(len(datasets))
        w = 0.8 / max(len(strategies), 1)
        for si, strat in enumerate(strategies):
            vals = [
                _mean_std([r.recall_at_k for r in by_ds[ds] if r.retrieval_strategy == strat])[0]
                for ds in datasets
            ]
            offset = (si - (len(strategies) - 1) / 2) * w
            ax.bar(x + offset, vals, w * 0.88, label=strat, color=_PALETTE[si % len(_PALETTE)], alpha=0.88)

        ax.set_xticks(x)
        ax.set_xticklabels([d[:16] for d in datasets], rotation=35, ha="right", fontsize=_TICK_SIZE - 1)
        ax.legend(title="Strategy", loc="upper right", fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
        _style_ax(ax, "Recall@K by Strategy — Per Dataset", "Dataset", "Recall@K", ylim=(0, 1.0))

    def _render_leaderboard_panel(self, ax, plt, np):
        """Leaderboard in the composite full-report panel.

        Splits the provided axes area into two side-by-side sub-plots:
        left = composite score bars, right = stacked component breakdown.
        Uses the figure's transAxes to create a divider line between them.
        """
        if not self._results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "Leaderboard")
            return

        ranked = sorted(self._results, key=lambda r: r.composite_score(), reverse=True)[:10]
        labels = [
            f"{r.retrieval_strategy[:8]}|{(r.embedding_model or '—').split('/')[-1][:10]}|"
            f"λ={r.lambda_value:.3f}|{r.memory_type[:4]}"
            for r in ranked
        ]
        composite = [r.composite_score() for r in ranked]
        y = np.arange(len(ranked))

        # ── Left half: composite score bars with P50 latency twin axis ───────
        ax.barh(y, composite,
                color=[_PALETTE[i % len(_PALETTE)] for i in range(len(ranked))],
                alpha=0.88, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_ANNOT_SIZE - 1)
        ax.invert_yaxis()
        ax.set_xlabel("Composite Score", fontsize=_LABEL_SIZE)
        ax.set_xlim(0, 1.05)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Leaderboard — Top-10  (composite = 0.40R + 0.25P + 0.20MRR + 0.15T)",
                     fontsize=_TITLE_SIZE, fontweight="bold")

        # Score annotation + P50 latency in brackets
        for xi, (score, r) in enumerate(zip(composite, ranked)):
            lat = getattr(r, "latency_p50_ms", 0.0)
            ax.text(score + 0.008, xi,
                    f"{score:.3f}  [{lat:.0f}ms P50]",
                    va="center", fontsize=_ANNOT_SIZE, fontweight="bold")


# ─── Global style ─────────────────────────────────────────────────────────────

def _set_rcparams(plt):
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize":     _TITLE_SIZE,
        "axes.labelsize":     _LABEL_SIZE,
        "xtick.labelsize":    _TICK_SIZE,
        "ytick.labelsize":    _TICK_SIZE,
        "legend.fontsize":    _LEGEND_SIZE,
        "figure.dpi":         _DPI,
        "savefig.dpi":        _DPI,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.5,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "patch.edgecolor":    "none",
    })
