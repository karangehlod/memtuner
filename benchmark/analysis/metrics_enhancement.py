"""Enhanced metrics analysis and reporting.

Provides comprehensive analysis beyond basic IR metrics:
- Resource usage (CPU, memory, disk)
- Strategy efficiency scoring
- Per-model comparisons and ranking
- Decay optimization analysis
- Memory type efficiency
- Cost-benefit analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceMetrics:
    """Resource usage during benchmark execution."""

    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""  # Non-empty if tracking unavailable

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "avg_memory_mb": round(self.avg_memory_mb, 2),
            "total_disk_read_mb": round(self.total_disk_read_mb, 2),
            "total_disk_write_mb": round(self.total_disk_write_mb, 2),
            "duration_seconds": round(self.duration_seconds, 3),
            "error": self.error,
        }


@dataclass
class StrategyEfficiency:
    """Efficiency metrics for a retrieval strategy."""

    strategy: str
    recall: float
    latency_ms: float
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    recall_per_latency: float = 0.0  # Recall / (latency in seconds)
    recall_per_memory: float = 0.0  # Recall / memory_mb
    efficiency_score: float = 0.0  # Combined score (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy,
            "recall": round(self.recall, 4),
            "latency_ms": round(self.latency_ms, 2),
            "memory_mb": round(self.memory_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "recall_per_latency": round(self.recall_per_latency, 4),
            "recall_per_memory": round(self.recall_per_memory, 4),
            "efficiency_score": round(self.efficiency_score, 4),
        }


@dataclass
class EmbeddingModelComparison:
    """Comparison of embedding models."""

    model: str
    label: str = ""
    recall: float = 0.0
    precision: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "label": self.label,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "latency_ms": round(self.latency_ms, 2),
            "memory_mb": round(self.memory_mb, 2),
            "rank": self.rank,
        }


@dataclass
class RerankerComparison:
    """Comparison of reranker models."""

    model: str
    base_recall: float = 0.0
    reranked_recall: float = 0.0
    improvement_pp: float = 0.0  # Percentage points
    base_precision: float = 0.0
    reranked_precision: float = 0.0
    latency_overhead_ms: float = 0.0
    cost_per_query_usd: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "base_recall": round(self.base_recall, 4),
            "reranked_recall": round(self.reranked_recall, 4),
            "improvement_pp": round(self.improvement_pp, 2),
            "base_precision": round(self.base_precision, 4),
            "reranked_precision": round(self.reranked_precision, 4),
            "latency_overhead_ms": round(self.latency_overhead_ms, 2),
            "cost_per_query_usd": round(self.cost_per_query_usd, 6),
            "rank": self.rank,
        }


@dataclass
class DecayOptimization:
    """Analysis of decay sweep optimization."""

    best_lambda: float = 0.0
    best_threshold: float = 0.0
    best_recall: float = 0.0
    best_precision: float = 0.0
    improvement_over_baseline_pp: float = 0.0
    configurations_tested: int = 0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "best_lambda": round(self.best_lambda, 4),
            "best_threshold": round(self.best_threshold, 4),
            "best_recall": round(self.best_recall, 4),
            "best_precision": round(self.best_precision, 4),
            "improvement_over_baseline_pp": round(self.improvement_over_baseline_pp, 2),
            "configurations_tested": self.configurations_tested,
            "recommendations": self.recommendations,
        }


@dataclass
class MemoryTypeEfficiency:
    """Efficiency metrics per memory type."""

    module: str
    recall: float = 0.0
    memories_stored: int = 0
    estimated_size_mb: float = 0.0
    queries_served: int = 0
    efficiency_score: float = 0.0  # recall / estimated_size_mb
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "module": self.module,
            "recall": round(self.recall, 4),
            "memories_stored": self.memories_stored,
            "estimated_size_mb": round(self.estimated_size_mb, 2),
            "queries_served": self.queries_served,
            "efficiency_score": round(self.efficiency_score, 4),
            "rank": self.rank,
        }


@dataclass
class LatencyBreakdown:
    """Breakdown of latency by operation."""

    total_ms: float = 0.0
    retrieval_ms: float = 0.0  # Search/scoring time
    model_inference_ms: float = 0.0  # Embedding generation
    ranking_ms: float = 0.0  # BM25 or other scoring
    reranking_ms: float = 0.0  # LLM reranking
    overhead_ms: float = 0.0  # Other overhead

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_ms": round(self.total_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "model_inference_ms": round(self.model_inference_ms, 2),
            "ranking_ms": round(self.ranking_ms, 2),
            "reranking_ms": round(self.reranking_ms, 2),
            "overhead_ms": round(self.overhead_ms, 2),
        }


class MetricsEnhancer:
    """Enhances benchmark results with comprehensive analysis."""

    @staticmethod
    def calculate_strategy_efficiency(
        strategy: str,
        recall: float,
        latency_ms: float,
        memory_mb: float = 100.0,
        cpu_percent: float = 25.0,
    ) -> StrategyEfficiency:
        """Calculate efficiency metrics for a strategy.

        Efficiency combines recall achievement with resource consumption.

        Args:
            strategy: Strategy name
            recall: Recall@K (0-1)
            latency_ms: Latency per query in milliseconds
            memory_mb: Memory usage in MB
            cpu_percent: CPU utilization percentage

        Returns:
            StrategyEfficiency with calculated scores
        """
        latency_seconds = latency_ms / 1000.0
        recall_per_latency = recall / latency_seconds if latency_seconds > 0 else 0.0
        recall_per_memory = recall / memory_mb if memory_mb > 0 else 0.0

        # Efficiency score combines recall with latency (0-1 scale)
        # Normalize: perfect = 1.0 recall at <10ms
        latency_score = min(1.0, 10.0 / latency_ms) if latency_ms > 0 else 0.5
        efficiency_score = (recall * 0.7) + (latency_score * 0.3)

        return StrategyEfficiency(
            strategy=strategy,
            recall=recall,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            recall_per_latency=recall_per_latency,
            recall_per_memory=recall_per_memory,
            efficiency_score=efficiency_score,
        )

    @staticmethod
    def rank_strategies(efficiencies: list[StrategyEfficiency]) -> list[StrategyEfficiency]:
        """Rank strategies by efficiency score."""
        ranked = sorted(
            efficiencies,
            key=lambda x: (x.efficiency_score, x.recall),
            reverse=True,
        )
        for i, e in enumerate(ranked, 1):
            e.rank = i
        return ranked

    @staticmethod
    def analyze_decay_sweep(
        sweep_results: list[dict[str, Any]],
    ) -> DecayOptimization:
        """Analyze decay/pruning sweep to find optimal configuration.

        Args:
            sweep_results: List of {lambda, threshold, recall, precision}

        Returns:
            DecayOptimization with best config and recommendations
        """
        if not sweep_results:
            return DecayOptimization()

        baseline_recall = sweep_results[0].get("recall", 0.0)
        best_result = max(
            sweep_results,
            key=lambda x: x.get("recall", 0.0),
        )

        best_lambda = best_result.get("lambda", 0.0)
        best_threshold = best_result.get("threshold", 0.01)
        best_recall = best_result.get("recall", 0.0)
        best_precision = best_result.get("precision", 0.0)

        improvement_pp = (best_recall - baseline_recall) * 100

        recommendations = []
        if improvement_pp > 1.0:
            recommendations.append(
                f"Enable decay with λ={best_lambda} for +{improvement_pp:.1f}pp recall improvement"
            )
        elif improvement_pp > 0.1:
            recommendations.append(
                f"Marginal gain from decay ({improvement_pp:.2f}pp); may not justify complexity"
            )
        else:
            recommendations.append("Decay provides minimal benefit; keep λ=0")

        if best_threshold > 0.1:
            recommendations.append(
                "Consider lower threshold to retain more memories"
            )

        return DecayOptimization(
            best_lambda=best_lambda,
            best_threshold=best_threshold,
            best_recall=best_recall,
            best_precision=best_precision,
            improvement_over_baseline_pp=improvement_pp,
            configurations_tested=len(sweep_results),
            recommendations=recommendations,
        )

    @staticmethod
    def analyze_memory_efficiency(
        memory_types: list[dict[str, Any]],
        total_queries: int = 1977,
    ) -> list[MemoryTypeEfficiency]:
        """Analyze efficiency of each memory type.

        Args:
            memory_types: List of memory type results
            total_queries: Total queries for normalization

        Returns:
            Ranked list of MemoryTypeEfficiency
        """
        efficiencies = []

        for mem_type in memory_types:
            module = mem_type.get("module", "unknown")
            recall = mem_type.get("recall", 0.0)
            memories_stored = mem_type.get("memories_stored", 0)

            # Estimate size: ~1KB per memory on average
            estimated_size_mb = memories_stored * 0.001 if memories_stored > 0 else 0.001

            # Estimate queries served: proportional to recall
            queries_served = int(total_queries * recall) if recall > 0 else 0

            # Efficiency: recall per MB of storage
            efficiency_score = recall / estimated_size_mb if estimated_size_mb > 0 else 0.0

            efficiencies.append(
                MemoryTypeEfficiency(
                    module=module,
                    recall=recall,
                    memories_stored=memories_stored,
                    estimated_size_mb=estimated_size_mb,
                    queries_served=queries_served,
                    efficiency_score=efficiency_score,
                )
            )

        # Rank by efficiency
        efficiencies.sort(key=lambda x: x.efficiency_score, reverse=True)
        for i, e in enumerate(efficiencies, 1):
            e.rank = i

        return efficiencies

    @staticmethod
    def estimate_latency_breakdown(
        total_latency_ms: float,
        has_reranker: bool = False,
        has_embeddings: bool = False,
    ) -> LatencyBreakdown:
        """Estimate latency breakdown by operation.

        This is an estimation based on typical operation times.
        Better approach would be to instrument each stage.

        Args:
            total_latency_ms: Total latency in milliseconds
            has_reranker: Whether reranking is enabled
            has_embeddings: Whether embeddings are used

        Returns:
            LatencyBreakdown with estimated components
        """
        # Typical breakdown:
        # - BM25 search: 1-2ms
        # - Embedding generation: 20-50ms (if using)
        # - Ranking/scoring: 1-5ms
        # - Reranking: 10-100ms (if using)
        # - Overhead: 1-2ms

        if has_embeddings:
            embedding_ms = total_latency_ms * 0.6  # ~60% of time
            ranking_ms = total_latency_ms * 0.25  # ~25%
        else:
            embedding_ms = 0.0
            ranking_ms = total_latency_ms * 0.8

        reranking_ms = 0.0
        if has_reranker:
            reranking_ms = total_latency_ms * 0.4
            embedding_ms = total_latency_ms * 0.35
            ranking_ms = total_latency_ms * 0.15

        retrieval_ms = total_latency_ms - embedding_ms - ranking_ms - reranking_ms
        overhead_ms = max(0.0, total_latency_ms * 0.05)

        return LatencyBreakdown(
            total_ms=total_latency_ms,
            retrieval_ms=max(0.0, retrieval_ms),
            model_inference_ms=embedding_ms,
            ranking_ms=ranking_ms,
            reranking_ms=reranking_ms,
            overhead_ms=overhead_ms,
        )


def enhance_report(
    report: dict[str, Any],
    resource_report: Any | None = None,
) -> dict[str, Any]:
    """Enhance a benchmark report with comprehensive analysis.

    Args:
        report: Original benchmark report
        resource_report: Optional ResourceReport from ResourceTracker

    Returns:
        Enhanced report with additional metrics sections
    """
    enhanced = dict(report)

    # Add resource metrics
    if resource_report:
        enhanced["resource_summary"] = {
            "peak_cpu_percent": round(resource_report.peak_cpu_percent, 2),
            "avg_cpu_percent": round(resource_report.avg_cpu_percent, 2),
            "peak_memory_mb": round(resource_report.peak_ram_mb, 2),
            "avg_memory_mb": round(resource_report.avg_ram_mb, 2),
            "total_disk_read_mb": round(resource_report.total_disk_read_mb, 2),
            "total_disk_write_mb": round(resource_report.total_disk_write_mb, 2),
            "duration_seconds": round(resource_report.duration_seconds, 3),
            "platform": resource_report.platform,
        }
    else:
        enhanced["resource_summary"] = {"error": "Resource tracking not available"}

    # Add strategy ranking
    strategies = report.get("strategy_comparison", [])
    if strategies:
        efficiencies = [
            MetricsEnhancer.calculate_strategy_efficiency(
                strategy=s.get("strategy", "unknown"),
                recall=s.get("recall", 0.0),
                latency_ms=s.get("ms_per_query", 0.0),
            )
            for s in strategies
        ]
        ranked = MetricsEnhancer.rank_strategies(efficiencies)
        enhanced["strategy_ranking"] = [e.to_dict() for e in ranked]

    # Add decay analysis
    decay_sweep = report.get("decay_sweep", [])
    if decay_sweep:
        decay_analysis = MetricsEnhancer.analyze_decay_sweep(decay_sweep)
        enhanced["decay_optimization"] = decay_analysis.to_dict()

    # Add memory type efficiency
    memory_types = report.get("memory_type_comparison", [])
    if memory_types:
        total_queries = report.get("dataset", {}).get("queries", 1977)
        efficiency = MetricsEnhancer.analyze_memory_efficiency(
            memory_types,
            total_queries=total_queries,
        )
        enhanced["memory_type_efficiency"] = [e.to_dict() for e in efficiency]

    # Add latency breakdown
    if strategies:
        first_strategy = strategies[0]
        latency_ms = first_strategy.get("ms_per_query", 0.0)
        has_reranker = any("rerank" in s.get("strategy", "").lower() for s in strategies)
        has_embeddings = any(
            "embed" in s.get("strategy", "").lower() for s in strategies
        )
        breakdown = MetricsEnhancer.estimate_latency_breakdown(
            latency_ms,
            has_reranker=has_reranker,
            has_embeddings=has_embeddings,
        )
        enhanced["latency_breakdown"] = breakdown.to_dict()

    return enhanced
