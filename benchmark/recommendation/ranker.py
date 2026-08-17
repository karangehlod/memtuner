"""Recommendation engine — ranks configurations and explains why.

Given matrix results, the recommender answers:
  "For workload X, which memory type + retrieval strategy + decay policy is best?"

It:
1. Filters by minimum quality thresholds (reject configs with unacceptable FPR)
2. Ranks survivors by composite score
3. Explains the recommendation in plain language
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    """Minimum acceptable quality for a recommended configuration.

    FPR threshold is dataset-dependent. For sparse-evidence datasets like
    LoCoMo (avg 1.42 relevant per query, k=10), theoretical minimum FPR is
    ~0.86. The threshold should be set relative to the dataset's sparsity.
    Default 0.95 means: at least 5% of retrieved items must be relevant.
    """

    min_recall: float = 0.30  # Must recall at least 30% of expected memories
    max_noise_ratio: float = 0.95  # Must not exceed 95% noise in retrieval window
    min_temporal: float = 0.0  # Temporal accuracy (0 if dataset has no temporal signal)
    max_peak_ram_mb: float = 4096  # Must stay under 4 GB RAM


@dataclass
class Recommendation:
    """A ranked recommendation from the matrix results."""

    rank: int
    memory_type: str
    retrieval_strategy: str
    decay_policy: str
    lambda_value: float
    recall_at_k: float
    contamination_rate: float
    temporal_accuracy: float
    composite_score: float
    peak_ram_mb: float
    explanation: str
    meets_thresholds: bool


class MatrixRanker:
    """Ranks matrix results and selects the best configuration."""

    def __init__(self, thresholds: QualityThresholds | None = None):
        self._thresholds = thresholds or QualityThresholds()

    def rank(self, results: list) -> list[Recommendation]:
        """Rank all successful results.

        Args:
            results: List of MatrixRunResult objects.

        Returns:
            Ranked list of Recommendation objects (best first).
        """
        successful = [r for r in results if r.success]
        if not successful:
            return []

        ranked = sorted(successful, key=lambda r: r.composite_score(), reverse=True)

        recommendations = []
        for i, r in enumerate(ranked, 1):
            meets = self._meets_thresholds(r)
            recommendations.append(
                Recommendation(
                    rank=i,
                    memory_type=r.memory_type,
                    retrieval_strategy=r.retrieval_strategy,
                    decay_policy=r.decay_policy,
                    lambda_value=r.lambda_value,
                    recall_at_k=r.recall_at_k,
                    contamination_rate=r.contamination_rate,
                    temporal_accuracy=r.temporal_accuracy,
                    composite_score=r.composite_score(),
                    peak_ram_mb=r.peak_ram_mb,
                    explanation=self._explain(r, meets),
                    meets_thresholds=meets,
                )
            )
        return recommendations

    def best_production_config(self, results: list) -> Recommendation | None:
        """Return the top-ranked config that meets all quality thresholds."""
        for rec in self.rank(results):
            if rec.meets_thresholds:
                return rec
        return None

    def _meets_thresholds(self, r) -> bool:
        t = self._thresholds
        return (
            r.recall_at_k >= t.min_recall
            and r.contamination_rate <= t.max_noise_ratio
            and r.temporal_accuracy >= t.min_temporal
            and r.peak_ram_mb <= t.max_peak_ram_mb
        )

    def _explain(self, r, meets: bool) -> str:
        t = self._thresholds
        parts = []

        # Recall assessment
        if r.recall_at_k >= 0.70:
            parts.append(f"excellent recall ({r.recall_at_k:.1%})")
        elif r.recall_at_k >= t.min_recall:
            parts.append(f"acceptable recall ({r.recall_at_k:.1%})")
        else:
            parts.append(f"low recall ({r.recall_at_k:.1%} < {t.min_recall:.1%} threshold)")

        # Noise ratio assessment (fraction of irrelevant items in retrieval window)
        noise = r.contamination_rate
        if noise <= 0.50:
            parts.append(f"low noise ({noise:.1%} of retrieved items irrelevant)")
        elif noise <= t.max_noise_ratio:
            parts.append(f"acceptable noise ({noise:.1%})")
        else:
            parts.append(
                f"high noise ({noise:.1%} > {t.max_noise_ratio:.1%} threshold — "
                "dataset has sparse evidence per query"
            )

        # Decay effect
        if r.decay_policy == "none":
            parts.append("no decay applied (memories persist indefinitely)")
        elif r.decay_policy == "exponential" and r.lambda_value >= 0.10:
            parts.append(f"aggressive decay (λ={r.lambda_value:.2f}) reduces stale memory")
        elif r.decay_policy in ("exponential", "logarithmic", "linear"):
            parts.append(f"moderate decay (λ={r.lambda_value:.2f})")

        # Strategy note
        strat_notes = {
            "bm25": "low-cost sparse retrieval",
            "embeddings": "semantic similarity (higher resource usage)",
            "hybrid": "balanced sparse+dense retrieval",
            "pgvector": "DB-backed vector retrieval (requires PostgreSQL)",
            "llm_rerank": "LLM reranking (highest cost, optional)",
        }
        parts.append(strat_notes.get(r.retrieval_strategy, r.retrieval_strategy))

        verdict = "PASS — production viable" if meets else "FAIL — does not meet quality thresholds"
        return f"{verdict}. {'; '.join(parts)}."


class MatrixExplainer:
    """Explains differences between two configurations."""

    def compare(self, a, b) -> str:
        """Generate a plain-language comparison of two MatrixRunResults."""
        lines = []
        lines.append("Comparing:")
        lines.append(
            f"  A: {a.memory_type} × {a.retrieval_strategy} × {a.decay_policy}(λ={a.lambda_value:.2f})"
        )
        lines.append(
            f"  B: {b.memory_type} × {b.retrieval_strategy} × {b.decay_policy}(λ={b.lambda_value:.2f})"
        )
        lines.append("")

        winner = "A" if a.composite_score() >= b.composite_score() else "B"
        lines.append(f"  Overall winner: {winner}")
        lines.append("")

        # Metric-by-metric comparison
        metrics = [
            ("Recall@K", a.recall_at_k, b.recall_at_k, True),
            ("Noise", a.contamination_rate, b.contamination_rate, False),  # lower better
            ("Temporal", a.temporal_accuracy, b.temporal_accuracy, True),
            ("Composite", a.composite_score(), b.composite_score(), True),
            ("Peak RAM", a.peak_ram_mb, b.peak_ram_mb, False),  # lower better
        ]

        for name, va, vb, higher_better in metrics:
            delta = va - vb
            if higher_better:
                better = "A" if delta > 0.001 else ("B" if delta < -0.001 else "tie")
            else:
                better = "A" if delta < -0.001 else ("B" if delta > 0.001 else "tie")
            lines.append(f"  {name:15s}: A={va:.4f}  B={vb:.4f}  winner={better}")

        return "\n".join(lines)
