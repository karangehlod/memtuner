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


def _annotate_bars(ax, bars, fmt="{:.3f}", offset_frac=0.01, fontsize=None):
    fs = fontsize or _ANNOT_SIZE
    for bar in bars:
        h = bar.get_height()
        if h > 0.001:
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
            yerr=errs, capsize=3 if errs else 0,
            error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6},
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
        paths["bm25_baseline"]        = self._plot_bm25_baseline(plt, np)
        paths["embedding_comparison"]  = self._plot_embedding_comparison(plt, np)
        paths["hybrid_weight"]         = self._plot_hybrid_weight(plt, np)
        paths["reranker_comparison"]   = self._plot_reranker_comparison(plt, np)
        paths["decay_heatmap"]         = self._plot_decay_heatmap(plt, np)
        paths["per_dataset"]           = self._plot_per_dataset(plt, np)
        paths["leaderboard"]           = self._plot_leaderboard(plt, np)
        paths["full_report"]           = self._plot_full_report(plt, gridspec, np)
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
        ax.set_xticklabels(all_strats, rotation=20, ha="right", fontsize=_TICK_SIZE)
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
        axes[0].legend(fontsize=_LEGEND_SIZE, framealpha=0.9)
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
        axes[1].legend(fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1], "BM25 Recall–Precision Trade-off",
                  "Recall@K", "Precision@K", ylim=(0, 1.0))
        axes[1].set_xlim(0, 1.0)

        plt.tight_layout()
        path = str(self._out / "phase1_bm25_baseline.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Phase 2: Embedding Model Comparison ─────────────────────────────────

    def _plot_embedding_comparison(self, plt, np) -> str:
        embed = [
            r for r in self._results
            if r.retrieval_strategy in ("embeddings", "api_embeddings")
        ]
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
        axes[0].set_xticklabels(short, rotation=25, ha="right", fontsize=_TICK_SIZE)
        axes[0].legend(title="Memory Type", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall@K by Embedding Model × Memory Type",
                  "Embedding Model", "Recall@K", ylim=(0, 1.0))

        # ── Right: Recall vs Latency scatter — Pareto frontier ────────────────
        by_model_recall = defaultdict(list)
        by_model_lat    = defaultdict(list)
        for r in embed:
            by_model_recall[r.embedding_model].append(r.recall_at_k)
            by_model_lat[r.embedding_model].append(r.latency_p50_ms)

        recall_means = {m: _mean_std(by_model_recall[m])[0] for m in models}
        lat_means    = {m: _mean_std(by_model_lat[m])[0]    for m in models}
        recall_stds  = {m: _mean_std(by_model_recall[m])[1] for m in models}
        lat_stds     = {m: _mean_std(by_model_lat[m])[1]    for m in models}

        ax2 = axes[1]
        for i, model in enumerate(models):
            ax2.errorbar(
                lat_means[model], recall_means[model],
                xerr=lat_stds[model], yerr=recall_stds[model],
                fmt="o", color=_PALETTE[i], markersize=10, capsize=4,
                label=short[i], linewidth=1.2, zorder=3,
            )
            ax2.annotate(
                short[i],
                (lat_means[model], recall_means[model]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=_ANNOT_SIZE - 1,
            )

        _style_ax(ax2, "Recall@K vs Query Latency P50 (Pareto Frontier)",
                  "Latency P50 (ms)", "Recall@K", ylim=(0, 1.0))
        ax2.legend(fontsize=_LEGEND_SIZE, framealpha=0.9)

        plt.tight_layout()
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

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
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
                    ws, ms, ss = zip(*pts)
                    ax.plot(ws, ms, "o-", label=mem, color=_PALETTE[i],
                            linewidth=2, markersize=8)
                    ax.fill_between(ws,
                                    [m - s for m, s in zip(ms, ss)],
                                    [m + s for m, s in zip(ms, ss)],
                                    color=_PALETTE[i], alpha=0.15)
            # Vertical reference lines at sweep points
            for w in weights:
                ax.axvline(w, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
            ax.legend(title="Memory Type", fontsize=_LEGEND_SIZE, framealpha=0.9)
            _style_ax(ax, f"{ylabel} vs BM25 Weight", "BM25 Weight", ylabel)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
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
        axes[0].set_xticklabels(short_names, rotation=20, ha="right", fontsize=_TICK_SIZE)
        axes[0].legend(labels=mlabels, fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall / Precision@K / MRR by Reranker",
                  "Reranker", "Score", ylim=(0, 1.0))

        # Lift annotation: delta vs baseline (none)
        baseline_recall = vals_map.get(("none", "recall_at_k"), 0.0)
        range(len(rerankers))
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

        # ── Right: Latency comparison ─────────────────────────────────────────
        lat_vals = []
        lat_stds = []
        for rr in rerankers:
            vs = [r.latency_p50_ms for r in rerank if r.reranker_model == rr]
            m, s = _mean_std(vs)
            lat_vals.append(m)
            lat_stds.append(s)

        x = np.arange(len(rerankers))
        bars = axes[1].bar(x, lat_vals, color=_PALETTE[3], alpha=0.88,
                           yerr=lat_stds, capsize=4,
                           error_kw={"elinewidth": 1.2, "ecolor": "black"})
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(short_names, rotation=20, ha="right", fontsize=_TICK_SIZE)
        _annotate_bars(axes[1], bars, fmt="{:.0f}ms")
        _style_ax(axes[1], "Query Latency P50 by Reranker", "Reranker", "Latency P50 (ms)")

        plt.tight_layout()
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

        plt.tight_layout()
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
                              if r.retrieval_strategy in ("embeddings", "api_embeddings")})

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
            [d[:20] for d in datasets], rotation=20, ha="right", fontsize=_TICK_SIZE
        )
        axes[0].legend(title="Strategy", fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[0], "Recall@K by Strategy — Per Dataset",
                  "Dataset", "Recall@K", ylim=(0, 1.0))

        # ── Right: embedding model recall per dataset ─────────────────────────
        emb_map = {}
        for ds, results in by_ds.items():
            for model in models:
                vs = [r.recall_at_k for r in results
                      if r.embedding_model == model
                      and r.retrieval_strategy in ("embeddings", "api_embeddings")]
                if vs:
                    emb_map[(ds, model)] = _mean_std(vs)[0]

        short_models = [m.split("/")[-1] for m in models]
        _grouped_bar(
            axes[1], np, datasets, models,
            {(ds, m): emb_map.get((ds, m), 0.0) for ds in datasets for m in models},
        )
        axes[1].set_xticklabels(
            [d[:20] for d in datasets], rotation=20, ha="right", fontsize=_TICK_SIZE
        )
        # Relabel legend with short model names
        handles, _ = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, short_models, title="Embedding Model",
                       fontsize=_LEGEND_SIZE, framealpha=0.9)
        _style_ax(axes[1], "Recall@K by Embedding Model — Per Dataset",
                  "Dataset", "Recall@K", ylim=(0, 1.0))

        plt.tight_layout()
        path = str(self._out / "phase6_per_dataset.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Leaderboard ─────────────────────────────────────────────────────────

    def _plot_leaderboard(self, plt, np) -> str:
        if not self._results:
            return ""

        ranked = sorted(self._results, key=lambda r: r.composite_score(), reverse=True)[:15]

        labels = []
        for r in ranked:
            embed = r.embedding_model.split("/")[-1][:12]
            strat = r.retrieval_strategy[:10]
            labels.append(
                f"{strat} | {embed}\n"
                f"λ={r.lambda_value:.3f}  bm25w={r.bm25_weight:.1f}  {r.memory_type[:4]}"
            )

        metrics      = ["recall_at_k", "precision_at_k", "mrr", "ndcg"]
        metric_labels = ["Recall@K", "Precision@K", "MRR", "NDCG"]
        composite     = [r.composite_score() for r in ranked]

        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle("Top-15 Configurations — Leaderboard",
                     fontsize=_TITLE_SIZE + 1, fontweight="bold")

        # ── Left: horizontal composite score bars ────────────────────────────
        y = np.arange(len(ranked))
        axes[0].barh(y, composite,
                     color=[_PALETTE[i % len(_PALETTE)] for i in range(len(ranked))],
                     alpha=0.88, height=0.72)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=_ANNOT_SIZE - 1)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Composite Score", fontsize=_LABEL_SIZE)
        axes[0].set_xlim(0, 1.05)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axes[0].set_axisbelow(True)
        axes[0].set_title("Composite Score (Top 15)", fontsize=_TITLE_SIZE, fontweight="bold")
        for xi, score in zip(y, composite):
            axes[0].text(score + 0.008, xi, f"{score:.3f}",
                         va="center", fontsize=_ANNOT_SIZE, fontweight="bold")

        # ── Right: full metric table as grouped horizontal bars ───────────────
        w = 0.18
        for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            vals = [getattr(r, metric) for r in ranked]
            offset = (mi - 1.5) * w
            axes[1].barh(y + offset, vals, w * 0.88,
                         label=mlabel, color=_PALETTE[mi], alpha=0.85)

        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=_ANNOT_SIZE - 1)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Score", fontsize=_LABEL_SIZE)
        axes[1].set_xlim(0, 1.05)
        axes[1].legend(fontsize=_LEGEND_SIZE, loc="lower right", framealpha=0.9)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axes[1].set_axisbelow(True)
        axes[1].set_title("Recall / Precision / MRR / NDCG", fontsize=_TITLE_SIZE, fontweight="bold")

        plt.tight_layout()
        path = str(self._out / "phase7_leaderboard.png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Full Composite Report ────────────────────────────────────────────────

    def _plot_full_report(self, plt, gridspec, np) -> str:
        fig = plt.figure(figsize=(22, 26))
        fig.suptitle(
            "Agentic Memory Benchmark — Comprehensive Study Report",
            fontsize=16, fontweight="bold", y=0.995
        )
        fig.text(
            0.5, 0.988,
            "BM25 Baseline  ·  Embedding Models  ·  Hybrid Weights  ·  "
            "Rerankers  ·  Decay Sweep  ·  Leaderboard",
            ha="center", fontsize=10, style="italic",
        )

        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.38)

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
        stds  = [_mean_std(by_mem[m])[1] for m in mem_types]
        bars  = ax.bar(mem_types, means, color=_PALETTE[:len(mem_types)], alpha=0.88,
                       yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})
        _annotate_bars(ax, bars)
        _style_ax(ax, "BM25 Baseline — Recall@K", "Memory Type", "Recall@K", ylim=(0, 1.0))

    def _render_embedding_panel(self, ax, plt, np):
        embed = [r for r in self._results
                 if r.retrieval_strategy in ("embeddings", "api_embeddings")]
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
        stds  = [_mean_std(by_m_r[m])[1] for m in models]
        ax.bar(x, means, color=_PALETTE[:len(models)], alpha=0.88,
                       yerr=stds, capsize=3, error_kw={"elinewidth": 1.0})
        ax2 = ax.twinx()
        lats = [_mean_std(by_m_l[m])[0] for m in models]
        ax2.plot(x, lats, "D--", color="darkorange", linewidth=1.5, markersize=6)
        ax2.set_ylabel("Latency P50 (ms)", color="darkorange", fontsize=_LABEL_SIZE - 1)
        ax2.tick_params(axis="y", colors="darkorange", labelsize=_TICK_SIZE - 1)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=25, ha="right", fontsize=_TICK_SIZE - 1)
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
        ax.legend(fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
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
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=_TICK_SIZE - 1)
        ax.legend(fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
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
        ax.set_xticklabels([d[:16] for d in datasets], rotation=20, ha="right", fontsize=_TICK_SIZE - 1)
        ax.legend(title="Strategy", fontsize=_LEGEND_SIZE - 1, framealpha=0.9)
        _style_ax(ax, "Recall@K by Strategy — Per Dataset", "Dataset", "Recall@K", ylim=(0, 1.0))

    def _render_leaderboard_panel(self, ax, plt, np):
        if not self._results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            _style_ax(ax, "Leaderboard")
            return

        ranked = sorted(self._results, key=lambda r: r.composite_score(), reverse=True)[:12]
        labels = [
            f"{r.retrieval_strategy[:8]}|{(r.embedding_model or '—').split('/')[-1][:10]}|"
            f"λ={r.lambda_value:.3f}|{r.memory_type[:4]}"
            for r in ranked
        ]
        composite = [r.composite_score() for r in ranked]
        y = np.arange(len(ranked))
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
        ax.set_title("Leaderboard — Top-12 Configurations by Composite Score",
                     fontsize=_TITLE_SIZE, fontweight="bold")
        for xi, score in zip(y, composite):
            ax.text(score + 0.008, xi, f"{score:.3f}", va="center",
                    fontsize=_ANNOT_SIZE, fontweight="bold")


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
