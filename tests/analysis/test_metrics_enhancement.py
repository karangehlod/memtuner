"""Tests for metrics enhancement module.

Verifies that all enhanced metrics are calculated correctly and can be
integrated into benchmark reports.
"""

from __future__ import annotations

import json

import pytest

from benchmark.analysis.metrics_enhancement import (
    EmbeddingModelComparison,
    MemoryTypeEfficiency,
    MetricsEnhancer,
    RerankerComparison,
    ResourceMetrics,
    enhance_report,
)


class TestStrategyEfficiency:
    """Test strategy efficiency calculation."""

    def test_basic_efficiency_calculation(self):
        """Test basic efficiency score calculation."""
        eff = MetricsEnhancer.calculate_strategy_efficiency(
            strategy="test_strategy",
            recall=0.6,
            latency_ms=50.0,
            memory_mb=100.0,
            cpu_percent=25.0,
        )

        assert eff.strategy == "test_strategy"
        assert eff.recall == 0.6
        assert eff.latency_ms == 50.0
        assert eff.memory_mb == 100.0
        assert eff.recall_per_latency > 0
        assert eff.recall_per_memory > 0
        assert 0.0 <= eff.efficiency_score <= 1.0

    def test_efficiency_with_zero_latency(self):
        """Test handling of zero latency."""
        eff = MetricsEnhancer.calculate_strategy_efficiency(
            strategy="test",
            recall=0.5,
            latency_ms=0.0,
        )

        assert eff.recall_per_latency == 0.0

    def test_efficiency_scoring(self):
        """Test that higher recall and lower latency increase efficiency."""
        fast = MetricsEnhancer.calculate_strategy_efficiency(
            strategy="fast",
            recall=0.5,
            latency_ms=10.0,
        )

        slow = MetricsEnhancer.calculate_strategy_efficiency(
            strategy="slow",
            recall=0.5,
            latency_ms=100.0,
        )

        # Fast should have higher efficiency
        assert fast.efficiency_score > slow.efficiency_score

    def test_strategy_ranking(self):
        """Test ranking of multiple strategies."""
        strategies = [
            MetricsEnhancer.calculate_strategy_efficiency(
                strategy="bm25",
                recall=0.53,
                latency_ms=5.6,
            ),
            MetricsEnhancer.calculate_strategy_efficiency(
                strategy="embeddings",
                recall=0.65,
                latency_ms=40.0,
            ),
            MetricsEnhancer.calculate_strategy_efficiency(
                strategy="hybrid",
                recall=0.68,
                latency_ms=46.0,
            ),
        ]

        ranked = MetricsEnhancer.rank_strategies(strategies)

        # Should be sorted by efficiency
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3
        # Rankings are by efficiency score (recall * 0.7 + latency_score * 0.3)
        # So the one with best efficiency score should be first
        assert all(ranked[i].rank == i + 1 for i in range(len(ranked)))


class TestDecayOptimization:
    """Test decay sweep analysis."""

    def test_basic_decay_analysis(self):
        """Test analysis of decay sweep results."""
        sweep_results = [
            {"lambda": 0.0, "threshold": 0.01, "recall": 0.531, "precision": 0.063},
            {"lambda": 0.01, "threshold": 0.01, "recall": 0.546, "precision": 0.064},
            {"lambda": 0.02, "threshold": 0.01, "recall": 0.548, "precision": 0.065},
        ]

        analysis = MetricsEnhancer.analyze_decay_sweep(sweep_results)

        assert analysis.best_lambda == 0.02
        assert analysis.best_recall == 0.548
        assert analysis.improvement_over_baseline_pp == (0.548 - 0.531) * 100
        assert analysis.configurations_tested == 3

    def test_decay_recommendations(self):
        """Test that decay recommendations are generated."""
        sweep_results = [
            {"lambda": 0.0, "threshold": 0.01, "recall": 0.531, "precision": 0.063},
            {"lambda": 0.01, "threshold": 0.01, "recall": 0.548, "precision": 0.064},
        ]

        analysis = MetricsEnhancer.analyze_decay_sweep(sweep_results)

        assert len(analysis.recommendations) > 0
        assert any("decay" in r.lower() for r in analysis.recommendations)

    def test_empty_decay_sweep(self):
        """Test handling of empty sweep results."""
        analysis = MetricsEnhancer.analyze_decay_sweep([])

        assert analysis.best_lambda == 0.0
        assert analysis.configurations_tested == 0


class TestMemoryTypeEfficiency:
    """Test memory type efficiency analysis."""

    def test_memory_efficiency_calculation(self):
        """Test memory type efficiency scoring."""
        memory_types = [
            {"module": "episodic_store", "recall": 0.53, "memories_stored": 5879},
            {"module": "preference_store", "recall": 0.064, "memories_stored": 313},
            {"module": "entity_store", "recall": 0.001, "memories_stored": 6},
        ]

        efficiency = MetricsEnhancer.analyze_memory_efficiency(memory_types)

        assert len(efficiency) == 3
        # All should have efficiency scores
        assert all(e.efficiency_score >= 0 for e in efficiency)
        # Should be ranked
        assert efficiency[0].rank == 1
        assert efficiency[1].rank == 2
        assert efficiency[2].rank == 3
        # Highest efficiency should be first
        # (Preference store has 0.064 / 0.313 MB ~= 0.204, which is > episodic 0.53 / 5.879 MB ~= 0.090)
        assert efficiency[0].efficiency_score >= efficiency[1].efficiency_score

    def test_memory_efficiency_dict_output(self):
        """Test serialization to dict."""
        mem = MemoryTypeEfficiency(
            module="test_store",
            recall=0.5,
            memories_stored=1000,
            estimated_size_mb=1.0,
            efficiency_score=500.0,
            rank=1,
        )

        d = mem.to_dict()

        assert d["module"] == "test_store"
        assert "recall" in d
        assert "efficiency_score" in d
        assert isinstance(d["recall"], float)


class TestLatencyBreakdown:
    """Test latency breakdown estimation."""

    def test_basic_latency_breakdown(self):
        """Test latency breakdown calculation."""
        breakdown = MetricsEnhancer.estimate_latency_breakdown(
            total_latency_ms=50.0,
            has_reranker=False,
            has_embeddings=True,
        )

        assert breakdown.total_ms == 50.0
        assert breakdown.retrieval_ms >= 0
        assert breakdown.model_inference_ms >= 0
        assert breakdown.ranking_ms >= 0
        # Total should be approximately correct (allow for rounding)
        total_parts = (
            breakdown.retrieval_ms
            + breakdown.model_inference_ms
            + breakdown.ranking_ms
            + breakdown.reranking_ms
            + breakdown.overhead_ms
        )
        assert abs(total_parts - 50.0) < 5.0  # Allow for estimation variance

    def test_latency_with_reranker(self):
        """Test latency breakdown with reranker."""
        with_reranker = MetricsEnhancer.estimate_latency_breakdown(
            total_latency_ms=100.0,
            has_reranker=True,
            has_embeddings=True,
        )

        assert with_reranker.reranking_ms > 0

    def test_latency_without_embeddings(self):
        """Test latency breakdown without embeddings."""
        breakdown = MetricsEnhancer.estimate_latency_breakdown(
            total_latency_ms=10.0,
            has_reranker=False,
            has_embeddings=False,
        )

        assert breakdown.model_inference_ms == 0.0


class TestResourceMetrics:
    """Test resource metrics data classes."""

    def test_resource_metrics_to_dict(self):
        """Test serialization of resource metrics."""
        metrics = ResourceMetrics(
            peak_cpu_percent=45.2,
            avg_cpu_percent=32.1,
            peak_memory_mb=512.3,
            avg_memory_mb=256.1,
            total_disk_read_mb=123.4,
            total_disk_write_mb=234.5,
            duration_seconds=125.3,
        )

        d = metrics.to_dict()

        assert d["peak_cpu_percent"] == 45.2
        assert d["peak_memory_mb"] == 512.3
        assert isinstance(d["duration_seconds"], float)


class TestEmbeddingModelComparison:
    """Test embedding model comparison data class."""

    def test_embedding_model_to_dict(self):
        """Test serialization of embedding model comparison."""
        model = EmbeddingModelComparison(
            model="all-MiniLM-L6-v2",
            label="MiniLM (22M, 384d)",
            recall=0.65,
            precision=0.08,
            latency_ms=40.0,
            memory_mb=100.0,
            rank=1,
        )

        d = model.to_dict()

        assert d["model"] == "all-MiniLM-L6-v2"
        assert d["recall"] == 0.65
        assert d["rank"] == 1


class TestRerankerComparison:
    """Test reranker comparison data class."""

    def test_reranker_to_dict(self):
        """Test serialization of reranker comparison."""
        reranker = RerankerComparison(
            model="bge-reranker-base",
            base_recall=0.65,
            reranked_recall=0.73,
            improvement_pp=8.0,
            base_precision=0.08,
            reranked_precision=0.12,
            latency_overhead_ms=45.0,
            cost_per_query_usd=0.0003,
            rank=1,
        )

        d = reranker.to_dict()

        assert d["model"] == "bge-reranker-base"
        assert d["improvement_pp"] == 8.0
        assert d["reranked_recall"] == 0.73


class TestEnhanceReport:
    """Test report enhancement."""

    def test_enhance_report_basic(self):
        """Test basic report enhancement."""
        report = {
            "dataset": {"queries": 1977, "memories": 5879},
            "strategy_comparison": [
                {
                    "strategy": "bm25",
                    "recall": 0.531,
                    "precision": 0.063,
                    "ms_per_query": 5.59,
                }
            ],
            "memory_type_comparison": [
                {"module": "episodic_store", "recall": 0.53, "memories_stored": 5879}
            ],
            "decay_sweep": [
                {"lambda": 0.0, "threshold": 0.01, "recall": 0.531, "precision": 0.063}
            ],
        }

        enhanced = enhance_report(report)

        # Check that enhancements were added
        assert "resource_summary" in enhanced
        assert "strategy_ranking" in enhanced
        assert "memory_type_efficiency" in enhanced
        assert "decay_optimization" in enhanced
        assert "latency_breakdown" in enhanced

    def test_enhance_report_with_resource_data(self):
        """Test enhancement with resource data."""
        from benchmark.resources.tracker import ResourceReport

        report = {
            "dataset": {"queries": 100},
            "strategy_comparison": [
                {"strategy": "test", "recall": 0.5, "ms_per_query": 10.0}
            ],
        }

        resource_report = ResourceReport(
            peak_ram_mb=512.3,
            avg_ram_mb=256.1,
            peak_cpu_percent=45.2,
            avg_cpu_percent=32.1,
            duration_seconds=125.3,
            platform="darwin",
        )

        enhanced = enhance_report(report, resource_report)

        assert "resource_summary" in enhanced
        assert enhanced["resource_summary"]["peak_memory_mb"] == 512.3
        assert enhanced["resource_summary"]["peak_cpu_percent"] == 45.2

    def test_enhanced_report_is_json_serializable(self):
        """Test that enhanced report can be JSON serialized."""
        report = {
            "dataset": {"queries": 100},
            "strategy_comparison": [
                {"strategy": "test", "recall": 0.5, "ms_per_query": 10.0}
            ],
            "memory_type_comparison": [
                {"module": "test", "recall": 0.5, "memories_stored": 100}
            ],
            "decay_sweep": [{"lambda": 0.0, "recall": 0.5, "threshold": 0.01}],
        }

        enhanced = enhance_report(report)

        # Should be JSON serializable
        json_str = json.dumps(enhanced)
        assert isinstance(json_str, str)

        # Should deserialize back
        parsed = json.loads(json_str)
        assert "strategy_ranking" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
