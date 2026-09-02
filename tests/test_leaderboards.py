"""Tests for leaderboard generation."""

import json

import pytest

from benchmark.memory.adapters import MemoryMetrics
from benchmark.memory.leaderboards import LeaderboardEntry, LeaderboardGenerator


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return MemoryMetrics(
        recall_at_10=0.85,
        recall_at_100=0.92,
        mrr=0.88,
        ndcg=0.87,
        write_latency_ms=2.5,
        query_latency_ms=5.0,
        storage_bytes=1024 * 1024,  # 1MB
        success_rate=0.99,
        error_count=1,
        dataset_name="test_dataset",
        num_memories=100,
        num_queries=50,
        elapsed_seconds=10.5,
    )


@pytest.fixture
def populated_generator(sample_metrics):
    """Generator with sample data."""
    gen = LeaderboardGenerator()

    adapters = [
        "episodic_store",
        "semantic_store",
        "entity_store",
        "episodic_buffer",
        "scratchpad",
    ]
    datasets = ["locomo", "coqa", "squad"]

    for adapter in adapters:
        for dataset in datasets:
            # Vary metrics by adapter for realism
            metrics = MemoryMetrics(
                recall_at_10=sample_metrics.recall_at_10 * (0.8 + hash(f"{adapter}{dataset}") % 50 / 100),
                recall_at_100=sample_metrics.recall_at_100 * (0.8 + hash(f"{adapter}{dataset}") % 50 / 100),
                mrr=sample_metrics.mrr * (0.8 + hash(f"{adapter}{dataset}") % 50 / 100),
                ndcg=sample_metrics.ndcg * (0.8 + hash(f"{adapter}{dataset}") % 50 / 100),
                write_latency_ms=sample_metrics.write_latency_ms * (1.0 + hash(f"{adapter}") % 100 / 100),
                query_latency_ms=sample_metrics.query_latency_ms * (1.0 + hash(f"{adapter}") % 100 / 100),
                storage_bytes=sample_metrics.storage_bytes * (0.5 + hash(f"{adapter}") % 100 / 100),
                success_rate=sample_metrics.success_rate,
                error_count=sample_metrics.error_count,
                dataset_name=dataset,
                num_memories=sample_metrics.num_memories,
                num_queries=sample_metrics.num_queries,
                elapsed_seconds=sample_metrics.elapsed_seconds,
            )
            gen.add_result(metrics, adapter, dataset)

    return gen


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry."""

    def test_entry_creation(self):
        """Test creating a leaderboard entry."""
        entry = LeaderboardEntry(
            memory_adapter="episodic_store",
            dataset="test",
            accuracy_score=0.85,
            efficiency_score=0.75,
        )

        assert entry.memory_adapter == "episodic_store"
        assert entry.dataset == "test"
        assert entry.accuracy_score == 0.85

    def test_balanced_score(self):
        """Test balanced score computation."""
        entry = LeaderboardEntry(
            memory_adapter="test",
            dataset="test",
            accuracy_score=0.8,
            efficiency_score=0.6,
        )

        # 60% accuracy + 40% efficiency
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert entry.balanced_score() == expected


class TestLeaderboardGenerator:
    """Tests for LeaderboardGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = LeaderboardGenerator()

        assert len(gen.entries) == 0

    def test_add_result(self, sample_metrics):
        """Test adding results."""
        gen = LeaderboardGenerator()

        gen.add_result(sample_metrics, "episodic_store", "test_dataset")

        assert len(gen.entries) == 1
        assert gen.entries[0].memory_adapter == "episodic_store"

    def test_accuracy_leaderboard(self, populated_generator):
        """Test accuracy leaderboard generation."""
        leaderboard = populated_generator.accuracy_leaderboard()

        assert "title" in leaderboard
        assert "entries" in leaderboard
        assert len(leaderboard["entries"]) > 0

        # Check ranking
        scores = [e["accuracy_score"] for e in leaderboard["entries"]]
        assert scores == sorted(scores, reverse=True)

    def test_efficiency_leaderboard(self, populated_generator):
        """Test efficiency leaderboard generation."""
        leaderboard = populated_generator.efficiency_leaderboard()

        assert "title" in leaderboard
        assert "entries" in leaderboard
        assert len(leaderboard["entries"]) > 0

        # Check ranking
        scores = [e["efficiency_score"] for e in leaderboard["entries"]]
        assert scores == sorted(scores, reverse=True)

    def test_balanced_leaderboard(self, populated_generator):
        """Test balanced leaderboard generation."""
        leaderboard = populated_generator.balanced_leaderboard()

        assert "title" in leaderboard
        assert "entries" in leaderboard
        assert len(leaderboard["entries"]) > 0

        # Check ranking
        scores = [e["balanced_score"] for e in leaderboard["entries"]]
        assert scores == sorted(scores, reverse=True)

    def test_cross_dataset_analysis(self, populated_generator):
        """Test cross-dataset analysis."""
        analysis = populated_generator.cross_dataset_analysis()

        assert "specializations" in analysis
        assert "analysis_matrix" in analysis

        # Check specializations for each adapter
        for _adapter, spec in analysis["specializations"].items():
            assert "best_dataset" in spec
            assert "best_score" in spec
            assert "worst_dataset" in spec
            assert "worst_score" in spec
            assert "avg_score" in spec

    def test_summary_report(self, populated_generator):
        """Test summary report generation."""
        report = populated_generator.summary_report()

        assert isinstance(report, str)
        assert "ACCURACY LEADERBOARD" in report
        assert "EFFICIENCY LEADERBOARD" in report
        assert "BALANCED LEADERBOARD" in report
        assert "SPECIALIZATIONS" in report

    def test_to_json(self, populated_generator):
        """Test JSON export."""
        json_str = populated_generator.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)

        assert "accuracy_leaderboard" in data
        assert "efficiency_leaderboard" in data
        assert "balanced_leaderboard" in data
        assert "cross_dataset_analysis" in data

    def test_to_html(self, populated_generator):
        """Test HTML export."""
        html = populated_generator.to_html()

        assert isinstance(html, str)
        assert "<html>" in html
        assert "<table>" in html
        assert "Accuracy Leaderboard" in html
        assert "Efficiency Leaderboard" in html

    def test_empty_generator(self):
        """Test leaderboard generation on empty generator."""
        gen = LeaderboardGenerator()

        accuracy = gen.accuracy_leaderboard()
        assert accuracy == {}

        efficiency = gen.efficiency_leaderboard()
        assert efficiency == {}

        report = gen.summary_report()
        assert isinstance(report, str)


class TestLeaderboardRanking:
    """Tests for leaderboard ranking."""

    def test_accuracy_ranking_order(self):
        """Test that adapters are ranked by accuracy."""
        gen = LeaderboardGenerator()

        # Add results in non-sorted order
        adapters_scores = [
            ("adapter_a", 0.75),
            ("adapter_b", 0.95),
            ("adapter_c", 0.85),
        ]

        for adapter, score in adapters_scores:
            metrics = MemoryMetrics(
                recall_at_10=score,
                recall_at_100=score,
                mrr=score,
                dataset_name="test",
            )
            gen.add_result(metrics, adapter, "test")

        leaderboard = gen.accuracy_leaderboard()
        entries = leaderboard["entries"]

        # Should be sorted by score descending
        assert entries[0]["memory_adapter"] == "adapter_b"  # 0.95
        assert entries[1]["memory_adapter"] == "adapter_c"  # 0.85
        assert entries[2]["memory_adapter"] == "adapter_a"  # 0.75

    def test_efficiency_ranking_order(self):
        """Test that adapters are ranked by efficiency."""
        gen = LeaderboardGenerator()

        # Add results with different latencies
        adapters_latencies = [
            ("adapter_a", 10.0),  # Slow
            ("adapter_b", 1.0),   # Fast
            ("adapter_c", 5.0),   # Medium
        ]

        for adapter, latency in adapters_latencies:
            metrics = MemoryMetrics(
                write_latency_ms=latency / 2,
                query_latency_ms=latency / 2,
                dataset_name="test",
            )
            gen.add_result(metrics, adapter, "test")

        leaderboard = gen.efficiency_leaderboard()
        entries = leaderboard["entries"]

        # adapter_b should be first (lowest latency)
        assert entries[0]["memory_adapter"] == "adapter_b"
        # adapter_a should be last (highest latency)
        assert entries[-1]["memory_adapter"] == "adapter_a"


class TestLeaderboardMultiDataset:
    """Tests for multi-dataset leaderboard behavior."""

    def test_averages_across_datasets(self):
        """Test that leaderboards average across datasets."""
        gen = LeaderboardGenerator()

        # Same adapter on different datasets
        for dataset in ["dataset_a", "dataset_b", "dataset_c"]:
            metrics = MemoryMetrics(
                recall_at_10=0.8,
                recall_at_100=0.9,
                mrr=0.85,
                dataset_name=dataset,
            )
            gen.add_result(metrics, "episodic_store", dataset)

        leaderboard = gen.accuracy_leaderboard()

        # Should have one entry for episodic_store
        assert len(leaderboard["entries"]) == 1
        entry = leaderboard["entries"][0]

        assert entry["memory_adapter"] == "episodic_store"
        assert entry["num_datasets"] == 3
        # Accuracy should be average of all 3 datasets
        expected_accuracy = (0.8 + 0.9 + 0.85) / 3
        assert abs(entry["accuracy_score"] - expected_accuracy) < 0.001
