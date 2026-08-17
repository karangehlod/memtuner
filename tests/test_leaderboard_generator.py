"""Tests for leaderboard generator."""

import pytest
import json
from benchmark.retrieval.leaderboard_generator import LeaderboardGenerator, LeaderboardEntry
from benchmark.retrieval.strategies.base import RetrievalMetrics


@pytest.fixture
def sample_metrics():
    """Sample metrics for multiple strategies."""
    return [
        RetrievalMetrics(
            recall_at_10=0.85,
            recall_at_100=0.92,
            mrr=0.88,
            ndcg=0.87,
            precision_at_10=0.83,
            query_latency_ms=1.5,
            index_build_time_sec=2.0,
            index_size_bytes=200_000_000,
            success_rate=0.99,
            error_count=1,
            strategy_name="bm25",
            dataset_name="test_dataset",
            num_queries=100,
            num_documents=5000,
            elapsed_seconds=150.0,
        ),
        RetrievalMetrics(
            recall_at_10=0.89,
            recall_at_100=0.95,
            mrr=0.91,
            ndcg=0.90,
            precision_at_10=0.87,
            query_latency_ms=4.0,
            index_build_time_sec=5.0,
            index_size_bytes=1_000_000_000,
            success_rate=0.98,
            error_count=2,
            strategy_name="learned_dense",
            dataset_name="test_dataset",
            num_queries=100,
            num_documents=5000,
            elapsed_seconds=500.0,
        ),
        RetrievalMetrics(
            recall_at_10=0.82,
            recall_at_100=0.88,
            mrr=0.84,
            ndcg=0.83,
            precision_at_10=0.80,
            query_latency_ms=0.5,
            index_build_time_sec=1.0,
            index_size_bytes=150_000_000,
            success_rate=1.0,
            error_count=0,
            strategy_name="ann",
            dataset_name="test_dataset",
            num_queries=100,
            num_documents=5000,
            elapsed_seconds=100.0,
        ),
    ]


class TestLeaderboardGenerator:
    """Tests for leaderboard generator."""

    def test_initialization(self):
        """Test leaderboard generator initialization."""
        gen = LeaderboardGenerator()
        assert gen.results == {}
        assert gen.registry is not None

    def test_add_result(self, sample_metrics):
        """Test adding results to leaderboard."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        assert "test_dataset" in gen.results
        assert len(gen.results["test_dataset"]) == 3

    def test_generate_leaderboard_by_score(self, sample_metrics):
        """Test generating leaderboard sorted by score."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        leaderboard = gen.generate_leaderboard("test_dataset", by="score")

        assert len(leaderboard) == 3
        # Check that results are ranked
        assert all(e.rank > 0 for e in leaderboard)
        # Check that scores are in descending order
        scores = [e.score for e in leaderboard]
        assert scores == sorted(scores, reverse=True)

    def test_generate_leaderboard_by_recall(self, sample_metrics):
        """Test generating leaderboard sorted by recall."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        leaderboard = gen.generate_leaderboard("test_dataset", by="recall_at_10")

        assert len(leaderboard) == 3
        # learned_dense should be first (highest recall)
        assert leaderboard[0].strategy_name == "learned_dense"

    def test_generate_leaderboard_by_latency(self, sample_metrics):
        """Test generating leaderboard sorted by latency."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        leaderboard = gen.generate_leaderboard("test_dataset", by="query_latency_ms")

        assert len(leaderboard) == 3
        # ann should be first (lowest latency)
        assert leaderboard[0].strategy_name == "ann"

    def test_generate_all_leaderboards(self, sample_metrics):
        """Test generating all leaderboards."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("dataset1", metrics)
            modified = RetrievalMetrics(
                recall_at_10=metrics.recall_at_10 * 0.95,
                recall_at_100=metrics.recall_at_100,
                mrr=metrics.mrr,
                ndcg=metrics.ndcg,
                precision_at_10=metrics.precision_at_10,
                query_latency_ms=metrics.query_latency_ms * 1.1,
                index_build_time_sec=metrics.index_build_time_sec,
                index_size_bytes=metrics.index_size_bytes,
                success_rate=metrics.success_rate,
                error_count=metrics.error_count,
                strategy_name=metrics.strategy_name,
                dataset_name="dataset2",
                num_queries=metrics.num_queries,
                num_documents=metrics.num_documents,
                elapsed_seconds=metrics.elapsed_seconds,
            )
            gen.add_result("dataset2", modified)

        all_leaderboards = gen.generate_all_leaderboards()

        assert len(all_leaderboards) == 2
        assert "dataset1" in all_leaderboards
        assert "dataset2" in all_leaderboards
        assert len(all_leaderboards["dataset1"]) == 3
        assert len(all_leaderboards["dataset2"]) == 3

    def test_to_json(self, sample_metrics):
        """Test exporting leaderboard as JSON."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        json_str = gen.to_json("test_dataset")

        # Parse JSON to verify it's valid
        data = json.loads(json_str)
        assert len(data) == 3
        assert all("strategy_name" in entry for entry in data)
        assert all("rank" in entry for entry in data)

    def test_to_csv(self, sample_metrics):
        """Test exporting leaderboard as CSV."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        csv_str = gen.to_csv("test_dataset")

        lines = csv_str.split("\n")
        assert len(lines) == 4  # Header + 3 entries
        assert "rank,strategy_name" in lines[0]

    def test_summary(self, sample_metrics):
        """Test generating summary statistics."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        summary = gen.summary()

        assert "test_dataset" in summary
        assert summary["test_dataset"]["num_strategies"] == 3
        assert summary["test_dataset"]["num_queries"] == 100
        assert "best_overall" in summary["test_dataset"]
        assert "best_recall" in summary["test_dataset"]
        assert "best_speed" in summary["test_dataset"]

    def test_best_strategy_selection(self, sample_metrics):
        """Test that best strategy is correctly identified."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        leaderboard = gen.generate_leaderboard("test_dataset", by="score")

        # First entry should be the best
        assert leaderboard[0].rank == 1
        assert leaderboard[0].score > leaderboard[1].score

    def test_empty_leaderboard(self):
        """Test handling of empty leaderboard."""
        gen = LeaderboardGenerator()

        leaderboard = gen.generate_leaderboard("nonexistent_dataset")
        assert leaderboard == []

        json_str = gen.to_json("nonexistent_dataset")
        assert json_str == "[]"

        csv_str = gen.to_csv("nonexistent_dataset")
        assert csv_str == ""

    def test_composite_score_calculation(self, sample_metrics):
        """Test that composite scores are calculated reasonably."""
        gen = LeaderboardGenerator()

        for metrics in sample_metrics:
            gen.add_result("test_dataset", metrics)

        leaderboard = gen.generate_leaderboard("test_dataset", by="score")

        # Scores should be between 0 and 1
        for entry in leaderboard:
            assert 0.0 <= entry.score <= 1.0

        # All scores should be positive
        assert all(e.score > 0.0 for e in leaderboard)
