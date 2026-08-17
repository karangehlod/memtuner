"""Mathematical simulation plots for the benchmark paper.

Generates publication-quality figures of all mathematical functions
used in the benchmark — decay curves, composite score sensitivity,
archival floor behaviour, and hybrid fusion weights.

Run standalone:
    python -m benchmark.reporting.simulation_plots --output docs/figures/

Each function is rendered over its full parameter space so the reader
can immediately see what each configuration choice means numerically.
"""

from __future__ import annotations

import math
from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError as e:
        raise ImportError("pip install matplotlib") from e


_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
_DPI = 300
_TITLE  = 11
_LABEL  = 10
_TICK   = 9
_LEGEND = 9


def _style(ax, title="", xlabel="", ylabel="", ylim=None, xlim=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if title:   ax.set_title(title, fontsize=_TITLE, fontweight="bold", pad=5)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=_LABEL)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=_LABEL)
    if ylim:    ax.set_ylim(*ylim)
    if xlim:    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=_TICK)


def plot_decay_curves(output_dir: Path) -> str:
    """Figure 1 — Decay function families over t ∈ [0, 300] days.

    Shows all four decay formulas for representative λ values so a reader
    can immediately see the qualitative difference between policies.

    Equations:
        exponential:  f(t) = exp(-λt),            f(t) ≥ 0.65 for t ≥ 90
        linear:       f(t) = max(0, 1 - λt),      f(t) ≥ 0.65 for t ≥ 90
        logarithmic:  f(t) = 1 / (1 + λt),        f(t) ≥ 0.65 for t ≥ 90
        tiered:       f(t) = 1           for t ≤ 7
                             exp(-λ(t-7)) for 7 < t < 90
                             1            for t ≥ 90
    """
    plt, np = _require_matplotlib()
    t = np.linspace(0, 300, 600)
    archival_floor = 0.65
    archival_threshold = 90

    lam_values = [0.005, 0.01, 0.02, 0.05, 0.10]
    policies = ["exponential", "linear", "logarithmic", "tiered"]
    policy_titles = {
        "exponential":  "Exponential  f(t) = e^{-λt}",
        "linear":       "Linear  f(t) = max(0, 1 − λt)",
        "logarithmic":  "Logarithmic  f(t) = 1 / (1 + λt)",
        "tiered":       "Tiered  (working/episodic/archival)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Memory Decay Functions — All Policies × Lambda Values\n"
        "Dashed red line = archival floor (0.65) applied at t ≥ 90 days",
        fontsize=_TITLE, fontweight="bold",
    )

    for ax_idx, policy in enumerate(policies):
        ax = axes[ax_idx // 2][ax_idx % 2]
        for i, lam in enumerate(lam_values):
            vals = []
            for ti in t:
                if policy == "exponential":
                    raw = math.exp(-lam * ti)
                    v = max(archival_floor, raw) if ti >= archival_threshold else raw
                elif policy == "linear":
                    raw = max(0.0, 1.0 - lam * ti)
                    v = max(archival_floor, raw) if ti >= archival_threshold else raw
                elif policy == "logarithmic":
                    raw = 1.0 / (1.0 + lam * ti)
                    v = max(archival_floor, raw) if ti >= archival_threshold else raw
                else:  # tiered
                    if ti <= 7 or ti >= archival_threshold:
                        v = 1.0
                    else:
                        v = math.exp(-lam * (ti - 7))
                vals.append(v)
            half_life = math.log(2) / lam if lam > 0 else float("inf")
            label = f"λ={lam:.3f}  (t½={half_life:.0f}d)"
            ax.plot(t, vals, linewidth=1.8, color=_PALETTE[i], label=label, alpha=0.9)

        ax.axhline(archival_floor, color="crimson", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"Archival floor ({archival_floor})")
        ax.axvline(archival_threshold, color="grey", linestyle=":",
                   linewidth=1.0, alpha=0.5, label=f"Archival threshold ({archival_threshold}d)")
        ax.legend(fontsize=_LEGEND - 1, framealpha=0.9, loc="upper right")
        _style(ax, policy_titles[policy], "Memory age (days)", "Decay factor f(t)", ylim=(0, 1.05))

    plt.tight_layout()
    path = str(output_dir / "fig1_decay_curves.png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_composite_sensitivity(output_dir: Path) -> str:
    """Figure 2 — Composite score sensitivity to metric weights.

    Shows how the composite score changes as each component weight varies
    while holding the others at their actual benchmark values. This is the
    standard sensitivity analysis for weighted scoring in IR evaluation.

    Composite formula:
        C = recall_gate × (w_R × Recall@K
                         + w_P × Precision@K
                         + w_M × MRR
                         + w_T × TemporalAccuracy)

    where recall_gate = 0 if Recall@K < 0.01, else 1.
    Default weights: w_R=0.40, w_P=0.25, w_M=0.20, w_T=0.15
    """
    plt, np = _require_matplotlib()

    # Typical values from LoCoMo bge-base-en-v1.5 hybrid run
    baseline = {"recall": 0.657, "precision": 0.295, "mrr": 0.445, "temporal": 0.820}
    default_weights = {"recall": 0.40, "precision": 0.25, "mrr": 0.20, "temporal": 0.15}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Composite Score Sensitivity Analysis\n"
        "Baseline: bge-base-en-v1.5, hybrid, λ=0.01 (Recall=0.657, P=0.295, MRR=0.445, Temp=0.820)",
        fontsize=_TITLE, fontweight="bold",
    )

    # ── Left: one-at-a-time weight variation ─────────────────────────────────
    w_range = np.linspace(0, 1, 200)
    components = [
        ("recall",    "Recall@K weight (w_R)",    _PALETTE[0]),
        ("precision", "Precision@K weight (w_P)", _PALETTE[1]),
        ("mrr",       "MRR weight (w_M)",          _PALETTE[2]),
        ("temporal",  "Temporal weight (w_T)",     _PALETTE[3]),
    ]
    ax = axes[0]
    for comp, label, color in components:
        scores = []
        for w in w_range:
            # All other weights re-normalised so total = 1.0
            remaining = {k: v for k, v in default_weights.items() if k != comp}
            remaining_sum = sum(remaining.values())
            scale = (1.0 - w) / remaining_sum if remaining_sum > 0 else 0
            wts = {k: v * scale for k, v in remaining.items()}
            wts[comp] = w
            score = sum(wts[k] * baseline[k] for k in wts)
            scores.append(score)
        ax.plot(w_range, scores, linewidth=2, color=color, label=label, alpha=0.9)
        # Mark the default weight
        dw = default_weights[comp]
        ds = sum(default_weights[k] * baseline[k] for k in default_weights)
        ax.scatter([dw], [ds], color=color, s=80, zorder=5)

    ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=_LEGEND, framealpha=0.9)
    _style(ax, "One-at-a-time Sensitivity (other weights re-normalised)",
           "Weight value", "Composite Score", ylim=(0, 1.0))

    # ── Right: Recall@K vs composite for different MRR levels ────────────────
    ax2 = axes[1]
    recall_range = np.linspace(0, 1, 200)
    mrr_levels = [0.2, 0.4, 0.6, 0.8]
    for i, mrr_val in enumerate(mrr_levels):
        scores = []
        for r in recall_range:
            if r < 0.01:
                scores.append(0.0)
            else:
                scores.append(
                    0.40 * r
                    + 0.25 * baseline["precision"]
                    + 0.20 * mrr_val
                    + 0.15 * baseline["temporal"]
                )
        ax2.plot(recall_range, scores, linewidth=2, color=_PALETTE[i],
                 label=f"MRR={mrr_val:.1f}", alpha=0.9)

    ax2.axvline(0.01, color="crimson", linestyle="--", linewidth=1.0, alpha=0.7,
                label="Recall gate (< 0.01 → C=0)")
    ax2.legend(fontsize=_LEGEND, framealpha=0.9)
    _style(ax2, "Composite vs Recall@K — Varying MRR\n(P@K=0.295, Temporal=0.820)",
           "Recall@K", "Composite Score", ylim=(0, 1.0))

    plt.tight_layout()
    path = str(output_dir / "fig2_composite_sensitivity.png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hybrid_fusion(output_dir: Path) -> str:
    """Figure 3 — Reciprocal Rank Fusion and hybrid BM25/semantic weights.

    Shows how RRF combines ranked lists and how the bm25_weight parameter
    controls the BM25/semantic balance.

    RRF formula (Cormack et al. 2009):
        score_RRF(d) = Σ_i  weight_i / (k + rank_i(d))
        where k=60 (standard constant), weight_BM25 = w, weight_embed = 1-w

    This figure shows:
      Left:  RRF contribution vs rank position for different k constants
      Right: Final hybrid score vs BM25 weight for representative rank combos
    """
    plt, np = _require_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Hybrid Strategy — Reciprocal Rank Fusion (RRF)\n"
        "score_RRF(d) = w_BM25/(60 + rank_BM25) + w_embed/(60 + rank_embed)",
        fontsize=_TITLE, fontweight="bold",
    )

    # ── Left: RRF contribution vs rank, different k ───────────────────────────
    ranks = np.arange(1, 201)
    k_values = [30, 60, 120, 200]
    ax = axes[0]
    for i, k in enumerate(k_values):
        contributions = 1.0 / (k + ranks)
        ax.plot(ranks, contributions, linewidth=2, color=_PALETTE[i],
                label=f"k={k}", alpha=0.9)
    ax.axvline(60, color="grey", linestyle=":", linewidth=1.0, alpha=0.5, label="rank=60 reference")
    ax.legend(fontsize=_LEGEND, framealpha=0.9)
    _style(ax, "RRF Contribution vs Rank Position",
           "Rank position", "1 / (k + rank)", ylim=(0, None))

    # ── Right: hybrid score vs bm25_weight ───────────────────────────────────
    bm25_weights = np.linspace(0, 1, 200)
    k = 60
    # Representative (bm25_rank, embed_rank) pairs for a given document
    rank_pairs = [
        (1,   50,  "BM25 rank=1, embed rank=50  (keyword match)"),
        (50,   1,  "BM25 rank=50, embed rank=1  (semantic match)"),
        (5,    5,  "Both rank=5  (both systems agree)"),
        (20,  20,  "Both rank=20 (middle tier)"),
        (1,    1,  "Both rank=1  (perfect agreement)"),
    ]
    ax2 = axes[1]
    for i, (br, er, label) in enumerate(rank_pairs):
        scores = []
        for w in bm25_weights:
            score = w / (k + br) + (1 - w) / (k + er)
            scores.append(score)
        ax2.plot(bm25_weights, scores, linewidth=2, color=_PALETTE[i % len(_PALETTE)],
                 label=label[:35], alpha=0.9)
    ax2.axvline(0.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.5, label="w=0.5 (equal weight)")
    ax2.legend(fontsize=_LEGEND - 1, framealpha=0.9, loc="upper center")
    _style(ax2, "Hybrid Score vs BM25 Weight (k=60)",
           "BM25 weight (w)", "RRF score", ylim=(0, None))

    plt.tight_layout()
    path = str(output_dir / "fig3_hybrid_fusion.png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lambda_phase_space(output_dir: Path) -> str:
    """Figure 4 — 2D phase-space: λ × time → decay factor for all policies.

    A heat map that shows the full (λ, t) space — this is the mathematical
    equivalent of the MATLAB parameter sweep plot. Readers can read off
    "what decay factor does policy X produce at time t for parameter λ?"

    This is the figure equivalent of running the benchmark across all
    57 decay cells simultaneously as a continuous surface.
    """
    plt, np = _require_matplotlib()

    lambdas = np.linspace(0.001, 0.15, 100)
    times   = np.linspace(0, 200, 100)
    L, T    = np.meshgrid(lambdas, times)
    archival_floor     = 0.65
    archival_threshold = 90

    policies = ["exponential", "linear", "logarithmic", "tiered"]
    policy_eqs = {
        "exponential":  r"$f(t,\lambda) = e^{-\lambda t}$",
        "linear":       r"$f(t,\lambda) = \max(0,\ 1 - \lambda t)$",
        "logarithmic":  r"$f(t,\lambda) = \frac{1}{1+\lambda t}$",
        "tiered":       r"$f(t,\lambda) = \begin{cases}1 & t\leq 7\\ e^{-\lambda(t-7)} & 7<t<90\\ 1 & t\geq 90\end{cases}$",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Parameter Phase Space: λ × Time → Decay Factor f(t, λ)\n"
        "Contour lines at f = 0.2, 0.4, 0.65 (archival floor), 0.8",
        fontsize=_TITLE + 1, fontweight="bold",
    )

    # Vectorised computation
    def compute_grid(policy, L, T):
        G = np.zeros_like(L)
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                lam, t = float(L[i, j]), float(T[i, j])
                if policy == "exponential":
                    raw = math.exp(-lam * t)
                    G[i, j] = max(archival_floor, raw) if t >= archival_threshold else raw
                elif policy == "linear":
                    raw = max(0.0, 1.0 - lam * t)
                    G[i, j] = max(archival_floor, raw) if t >= archival_threshold else raw
                elif policy == "logarithmic":
                    raw = 1.0 / (1.0 + lam * t)
                    G[i, j] = max(archival_floor, raw) if t >= archival_threshold else raw
                else:  # tiered
                    if t <= 7 or t >= archival_threshold:
                        G[i, j] = 1.0
                    else:
                        G[i, j] = math.exp(-lam * (t - 7))
        return G

    for idx, policy in enumerate(policies):
        ax = axes[idx // 2][idx % 2]
        G = compute_grid(policy, L, T)

        im = ax.pcolormesh(lambdas, times, G, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label("Decay factor f(t, λ)", fontsize=_LABEL - 1)
        cbar.ax.tick_params(labelsize=_TICK - 1)

        # Contour lines at key values
        levels = [0.2, 0.4, archival_floor, 0.8]
        cs = ax.contour(lambdas, times, G, levels=levels,
                        colors=["white"], linewidths=1.2, alpha=0.85)
        ax.clabel(cs, fmt={0.2: "0.2", 0.4: "0.4", archival_floor: f"{archival_floor}", 0.8: "0.8"},
                  fontsize=_TICK - 1, inline=True)

        # Mark the benchmark λ grid points (LAMBDA_STEPS)
        lambda_steps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
        for ls in lambda_steps:
            if lambdas[0] <= ls <= lambdas[-1]:
                ax.axvline(ls, color="black", linestyle=":", linewidth=0.7, alpha=0.5)

        ax.axhline(archival_threshold, color="cyan", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Archival threshold ({archival_threshold}d)")
        ax.legend(fontsize=_LEGEND - 2, loc="upper right", framealpha=0.8)
        _style(ax, f"{policy.capitalize()}  {policy_eqs[policy]}",
               "Lambda (λ)", "Memory age (days)")

    plt.tight_layout()
    path = str(output_dir / "fig4_lambda_phase_space.png")
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_simulation_plots(output_dir: str = "docs/figures") -> dict[str, str]:
    """Generate all mathematical simulation figures for the paper."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    print("Generating simulation plots...")
    paths["decay_curves"]          = plot_decay_curves(out)
    print(f"  ✓ fig1_decay_curves.png")
    paths["composite_sensitivity"]  = plot_composite_sensitivity(out)
    print(f"  ✓ fig2_composite_sensitivity.png")
    paths["hybrid_fusion"]          = plot_hybrid_fusion(out)
    print(f"  ✓ fig3_hybrid_fusion.png")
    paths["lambda_phase_space"]     = plot_lambda_phase_space(out)
    print(f"  ✓ fig4_lambda_phase_space.png")
    print(f"\nAll figures written to: {out}/")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulation/mathematical figures for the paper")
    parser.add_argument("--output", default="docs/figures", help="Output directory (default: docs/figures)")
    args = parser.parse_args()
    generate_all_simulation_plots(args.output)
