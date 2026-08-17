"""Tests for retrieval benchmark orchestrator."""

import pytest

# Import all strategy adapters to register them
from benchmark.retrieval.strategies.bm25_adapter import BM25Adapter
from benchmark.retrieval.strategies.tfidf_adapter import TFIDFAdapter
from benchmark.retrieval.strategies.boolean_adapter import BooleanAdapter
from benchmark.retrieval.strategies.dense_vector_adapter import DenseVectorAdapter
from benchmark.retrieval.strategies.learned_dense_adapter import LearnedDenseAdapter
from benchmark.retrieval.strategies.ann_adapter import ANNAdapter
from benchmark.retrieval.strategies.quantized_adapter import QuantizedAdapter
from benchmark.retrieval.strategies.hybrid_fusion_adapter import HybridFusionAdapter
from benchmark.retrieval.strategies.cascading_adapter import CascadingAdapter
from benchmark.retrieval.strategies.retrieval_rerank_adapter import RetrievalRerankAdapter
from benchmark.retrieval.strategies.chainsearch_adapter import ChainSearchAdapter

from benchmark.retrieval.benchmark_orchestrator import RetrievalBenchmarkOrchestrator


@pytest.fixture
def sample_documents():
    """Sample documents for benchmarking."""
    return [
        {"id": "doc_1", "content": "The quick brown fox jumps over the lazy dog"},
        {"id": "doc_2", "content": "A fast red fox runs through the forest"},
        {"id": "doc_3", "content": "The lazy dog sleeps under the tree"},
        {"id": "doc_4", "content": "Quick brown animals jump high"},
        {"id": "doc_5", "content": "Dogs and foxes are animals"},
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for benchmarking."""
    return [
        "quick fox",
        "lazy dog",
        "brown fox jumps",
        "forest animals",
        "tree sleep",
    ]


class TestRetrievalBenchmarkOrchestrator:
    """Tests for retrieval benchmark orchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = RetrievalBenchmarkOrchestrator()
        assert orchestrator.registry is not None
        assert orchestrator.leaderboard_generator is not None
        assert orchestrator.results == {}

    def test_benchmark_single_strategy(self, sample_documents, sample_queries):
        """Test benchmarking a single strategy."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        result = orchestrator.benchmark_strategy(
            "bm25",
            sample_documents,
            sample_queries,
            "test_dataset",
        )

        assert result["status"] == "success"
        assert result["strategy"] == "bm25"
        assert result["recall_at_10"] > 0
        assert result["query_latency_ms"] > 0

    def test_benchmark_all_strategies(self, sample_documents, sample_queries):
        """Test benchmarking all strategies."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        result = orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
        )

        assert "strategies" in result
        assert len(result["strategies"]) > 0
        assert result["num_documents"] == len(sample_documents)
        assert result["num_queries"] == len(sample_queries)
        assert "total_elapsed_seconds" in result

    def test_benchmark_subset_strategies(self, sample_documents, sample_queries):
        """Test benchmarking subset of strategies."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        result = orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=["bm25", "tfidf"],
        )

        assert len(result["strategies"]) == 2
        assert "bm25" in result["strategies"]
        assert "tfidf" in result["strategies"]

    def test_get_leaderboard(self, sample_documents, sample_queries):
        """Test getting leaderboard."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=["bm25", "tfidf", "boolean"],
        )

        leaderboard = orchestrator.get_leaderboard("test_dataset")

        assert len(leaderboard) == 3
        assert all("rank" in entry for entry in leaderboard)
        assert all("strategy" in entry for entry in leaderboard)
        assert all("recall_at_10" in entry for entry in leaderboard)
        assert all("score" in entry for entry in leaderboard)

    def test_get_leaderboard_by_latency(self, sample_documents, sample_queries):
        """Test getting leaderboard sorted by latency."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=["bm25", "ann", "boolean"],
        )

        leaderboard = orchestrator.get_leaderboard("test_dataset", by="query_latency_ms")

        assert len(leaderboard) == 3
        # boolean should be fastest
        assert leaderboard[0]["strategy"] in ["ann", "boolean"]

    def test_get_summary(self, sample_documents, sample_queries):
        """Test getting summary."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=["bm25", "learned_dense"],
        )

        summary = orchestrator.get_summary()

        assert "summary" in summary
        assert "datasets" in summary
        assert "test_dataset" in summary["datasets"]
        assert "total_strategies" in summary

    def test_export_json(self, sample_documents, sample_queries):
        """Test exporting results as JSON."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=["bm25"],
        )

        json_str = orchestrator.export_results("test_dataset", format="json")

        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_export_csv(self, sample_documents, sample_queries):
        """Test exporting results as CSV."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=["bm25"],
        )

        csv_str = orchestrator.export_results("test_dataset", format="csv")

        assert isinstance(csv_str, str)
        assert len(csv_str) > 0
        assert "bm25" in csv_str

    def test_multiple_datasets(self, sample_documents, sample_queries):
        """Test benchmarking multiple datasets."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "dataset1",
            strategy_names=["bm25"],
        )

        orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "dataset2",
            strategy_names=["learned_dense"],
        )

        summary = orchestrator.get_summary()

        assert len(summary["datasets"]) == 2
        assert "dataset1" in summary["datasets"]
        assert "dataset2" in summary["datasets"]

    def test_failed_strategy(self, sample_documents, sample_queries):
        """Test handling of non-existent strategy."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        result = orchestrator.benchmark_strategy(
            "nonexistent_strategy",
            sample_documents,
            sample_queries,
            "test_dataset",
        )

        assert result["status"] == "failed"

    def test_all_strategies_from_registry(self, sample_documents, sample_queries):
        """Test benchmarking all strategies from registry."""
        orchestrator = RetrievalBenchmarkOrchestrator()

        all_strategies = orchestrator.registry.list_all()
        assert len(all_strategies) >= 11  # Should have at least 11 strategies

        result = orchestrator.benchmark_all_strategies(
            sample_documents,
            sample_queries,
            "test_dataset",
            strategy_names=all_strategies,
        )

        assert len(result["strategies"]) == len(all_strategies)
        assert all(s in result["strategies"] for s in all_strategies)
