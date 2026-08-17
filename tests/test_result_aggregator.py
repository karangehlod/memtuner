"""Comprehensive tests for ResultAggregator."""

import pytest
import numpy as np
from benchmark.memory.distributed.result_aggregator import (
    ResultAggregator,
    AggregatedResult,
    AggregatedMetrics,
)


@pytest.fixture
def aggregator():
    """Create aggregator instance."""
    return ResultAggregator()


@pytest.fixture
def sample_results():
    """Create sample query results."""
    return [
        {"query": "q1", "success": True, "time_ms": 10.5, "metrics": {"score": 0.95}},
        {"query": "q2", "success": True, "time_ms": 12.3, "metrics": {"score": 0.87}},
        {"query": "q3", "success": True, "time_ms": 11.2, "metrics": {"score": 0.91}},
    ]


class TestAggregatorInitialization:
    """Test aggregator initialization."""

    def test_initialization(self, aggregator):
        """Test aggregator initializes correctly."""
        assert aggregator is not None
        assert len(aggregator.results_buffer) == 0

    def test_buffer_operations(self, aggregator, sample_results):
        """Test buffer operations."""
        for result in sample_results:
            aggregator.add_result(result)

        assert len(aggregator.get_buffered_results()) == 3
        aggregator.clear_buffer()
        assert len(aggregator.get_buffered_results()) == 0


class TestQueryResultAggregation:
    """Test query result aggregation."""

    def test_aggregate_single_result(self, aggregator):
        """Test aggregating single result."""
        results = [{"success": True, "time_ms": 10.0}]
        aggregated = aggregator.aggregate_query_results(results)

        assert aggregated.total_results == 1
        assert aggregated.successful_results == 1
        assert aggregated.failed_results == 0

    def test_aggregate_multiple_results(self, aggregator, sample_results):
        """Test aggregating multiple results."""
        aggregated = aggregator.aggregate_query_results(sample_results)

        assert aggregated.total_results == 3
        assert aggregated.successful_results == 3
        assert aggregated.failed_results == 0

    def test_aggregate_with_failures(self, aggregator):
        """Test aggregation with some failures."""
        results = [
            {"success": True, "time_ms": 10.0},
            {"success": False, "time_ms": 5.0},
            {"success": True, "time_ms": 12.0},
        ]
        aggregated = aggregator.aggregate_query_results(results)

        assert aggregated.total_results == 3
        assert aggregated.successful_results == 2
        assert aggregated.failed_results == 1

    def test_empty_result_raises_error(self, aggregator):
        """Test empty result list raises error."""
        with pytest.raises(ValueError, match="empty"):
            aggregator.aggregate_query_results([])


class TestMetricsAggregation:
    """Test metrics aggregation."""

    def test_aggregate_single_metric(self, aggregator):
        """Test aggregating single metric."""
        metrics_list = [
            {"throughput": 100.0},
            {"throughput": 120.0},
            {"throughput": 110.0},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        assert "throughput" in aggregated
        assert aggregated["throughput"].mean > 0
        assert aggregated["throughput"].count == 3

    def test_aggregate_multiple_metrics(self, aggregator):
        """Test aggregating multiple metrics."""
        metrics_list = [
            {"throughput": 100.0, "latency": 10.5},
            {"throughput": 120.0, "latency": 11.2},
            {"throughput": 110.0, "latency": 10.8},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        assert "throughput" in aggregated
        assert "latency" in aggregated
        assert len(aggregated) == 2

    def test_empty_metrics_list(self, aggregator):
        """Test empty metrics list."""
        aggregated = aggregator.aggregate_metrics([])
        assert aggregated == {}

    def test_metrics_with_nan_values(self, aggregator):
        """Test metrics containing NaN values."""
        metrics_list = [
            {"score": 0.95},
            {"score": np.nan},
            {"score": 0.87},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        assert "score" in aggregated
        # NaN should be excluded
        assert aggregated["score"].count == 2

    def test_metrics_with_inf_values(self, aggregator):
        """Test metrics containing inf values."""
        metrics_list = [
            {"rate": 10.0},
            {"rate": np.inf},
            {"rate": 12.0},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        assert "rate" in aggregated
        # inf should be excluded
        assert aggregated["rate"].count == 2


class TestStatisticsComputation:
    """Test statistics computation."""

    def test_compute_mean(self, aggregator):
        """Test mean computation."""
        results = [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
        ]
        stats = aggregator.compute_aggregate_statistics(results)

        assert "value" in stats
        assert stats["value"]["mean"] == 20.0

    def test_compute_median(self, aggregator):
        """Test median computation."""
        results = [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
        ]
        stats = aggregator.compute_aggregate_statistics(results)

        assert stats["value"]["median"] == 20.0

    def test_compute_std_dev(self, aggregator):
        """Test standard deviation computation."""
        results = [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
        ]
        stats = aggregator.compute_aggregate_statistics(results)

        assert stats["value"]["std_dev"] > 0

    def test_compute_min_max(self, aggregator):
        """Test min/max computation."""
        results = [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
        ]
        stats = aggregator.compute_aggregate_statistics(results)

        assert stats["value"]["min"] == 10.0
        assert stats["value"]["max"] == 30.0

    def test_statistics_with_single_value(self, aggregator):
        """Test statistics with single value."""
        results = [{"value": 42.0}]
        stats = aggregator.compute_aggregate_statistics(results)

        assert stats["value"]["mean"] == 42.0
        assert stats["value"]["std_dev"] == 0.0

    def test_empty_results_statistics(self, aggregator):
        """Test statistics with empty results."""
        stats = aggregator.compute_aggregate_statistics([])
        assert isinstance(stats, dict)


class TestPercentileComputation:
    """Test percentile computation."""

    def test_percentile_50(self, aggregator, sample_results):
        """Test 50th percentile (median)."""
        aggregated = aggregator.aggregate_query_results(sample_results)

        assert 50 in aggregated.percentile_latencies
        p50 = aggregated.percentile_latencies[50]
        assert p50 > 0

    def test_percentile_95(self, aggregator, sample_results):
        """Test 95th percentile."""
        aggregated = aggregator.aggregate_query_results(sample_results)

        assert 95 in aggregated.percentile_latencies
        p95 = aggregated.percentile_latencies[95]
        assert p95 >= aggregated.percentile_latencies[50]

    def test_percentile_99(self, aggregator, sample_results):
        """Test 99th percentile."""
        aggregated = aggregator.aggregate_query_results(sample_results)

        assert 99 in aggregated.percentile_latencies
        p99 = aggregated.percentile_latencies[99]
        assert p99 >= aggregated.percentile_latencies[95]

    def test_percentiles_monotonic(self, aggregator):
        """Test percentiles are monotonically increasing."""
        results = [{"success": True, "time_ms": float(i)} for i in range(1, 101)]
        aggregated = aggregator.aggregate_query_results(results)

        p50 = aggregated.percentile_latencies[50]
        p95 = aggregated.percentile_latencies[95]
        p99 = aggregated.percentile_latencies[99]

        assert p50 <= p95 <= p99


class TestConsistencyScoring:
    """Test consistency score computation."""

    def test_consistency_perfect_results(self, aggregator):
        """Test consistency with all successful results."""
        results = [
            {"success": True, "time_ms": 10.0},
            {"success": True, "time_ms": 10.1},
            {"success": True, "time_ms": 10.2},
        ]
        aggregated = aggregator.aggregate_query_results(results)

        # All successful and low variance = high consistency
        assert aggregated.consistency_score > 0.8

    def test_consistency_with_failures(self, aggregator):
        """Test consistency with failures."""
        results = [
            {"success": True, "time_ms": 10.0},
            {"success": False},
            {"success": True, "time_ms": 10.0},
        ]
        aggregated = aggregator.aggregate_query_results(results)

        # 66% success rate = lower consistency
        assert aggregated.consistency_score < 1.0
        assert aggregated.consistency_score > 0.5

    def test_consistency_high_variance(self, aggregator):
        """Test consistency with high latency variance."""
        results = [
            {"success": True, "time_ms": 1.0},
            {"success": True, "time_ms": 100.0},
            {"success": True, "time_ms": 50.0},
        ]
        aggregated = aggregator.aggregate_query_results(results)

        # High variance = lower consistency
        assert 0.0 <= aggregated.consistency_score <= 1.0

    def test_consistency_range(self, aggregator):
        """Test consistency score is in valid range."""
        results = [{"success": True, "time_ms": 10.0}]
        aggregated = aggregator.aggregate_query_results(results)

        assert 0.0 <= aggregated.consistency_score <= 1.0


class TestNullNaNHandling:
    """Test null and NaN value handling."""

    def test_null_metrics_excluded(self, aggregator):
        """Test null metrics are excluded."""
        metrics_list = [
            {"score": 0.95},
            {"score": None},
            {"score": 0.87},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        # Should skip the None value
        assert "score" in aggregated
        assert aggregated["score"].count >= 1

    def test_nan_in_statistics(self, aggregator):
        """Test NaN values in statistics."""
        results = [
            {"value": 10.0},
            {"value": np.nan},
            {"value": 20.0},
        ]
        stats = aggregator.compute_aggregate_statistics(results)

        # NaN should be excluded
        assert stats["value"]["count"] == 2

    def test_inf_in_statistics(self, aggregator):
        """Test inf values in statistics."""
        results = [
            {"value": 10.0},
            {"value": np.inf},
            {"value": 20.0},
        ]
        stats = aggregator.compute_aggregate_statistics(results)

        # inf should be excluded
        assert stats["value"]["count"] == 2


class TestDeduplication:
    """Test duplicate result handling."""

    def test_duplicate_results_counted(self, aggregator):
        """Test duplicate results are counted."""
        results = [
            {"query": "q1", "success": True, "time_ms": 10.0},
            {"query": "q1", "success": True, "time_ms": 10.0},
            {"query": "q2", "success": True, "time_ms": 12.0},
        ]
        aggregated = aggregator.aggregate_query_results(results)

        # Both duplicates should be counted
        assert aggregated.total_results == 3

    def test_duplicate_metrics_aggregated(self, aggregator):
        """Test duplicate metrics are aggregated together."""
        metrics_list = [
            {"throughput": 100.0},
            {"throughput": 100.0},
            {"throughput": 100.0},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        assert aggregated["throughput"].mean == 100.0
        assert aggregated["throughput"].std_dev == 0.0


class TestLargeResultSets:
    """Test aggregation with large result sets."""

    def test_large_result_set_1000(self, aggregator):
        """Test aggregating 1000 results."""
        results = [
            {"success": True, "time_ms": 10.0 + (i % 5)}
            for i in range(1000)
        ]
        aggregated = aggregator.aggregate_query_results(results)

        assert aggregated.total_results == 1000
        assert aggregated.successful_results == 1000

    def test_large_metrics_aggregation(self, aggregator):
        """Test aggregating large metric sets."""
        metrics_list = [
            {"metric": 100.0 + (i % 20), "score": 0.5 + (i % 50) / 100.0}
            for i in range(500)
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        assert len(aggregated) >= 1
        assert aggregated["metric"].count == 500

    def test_percentile_accuracy_large_set(self, aggregator):
        """Test percentile accuracy on large dataset."""
        # Create dataset with known percentiles
        results = [{"success": True, "time_ms": float(i)} for i in range(1, 1001)]
        aggregated = aggregator.aggregate_query_results(results)

        # For uniform 1-1000 distribution:
        # p50 ≈ 500, p95 ≈ 950, p99 ≈ 990
        assert 400 < aggregated.percentile_latencies[50] < 600
        assert 900 < aggregated.percentile_latencies[95] < 1000
        assert 950 < aggregated.percentile_latencies[99] <= 1000


class TestConsistencyVerification:
    """Test consistency verification."""

    def test_consistency_with_all_metrics(self, aggregator):
        """Test consistency includes all metric types."""
        results = [
            {
                "success": True,
                "time_ms": 10.0,
                "metrics": {"throughput": 100.0},
            },
            {
                "success": True,
                "time_ms": 11.0,
                "metrics": {"throughput": 105.0},
            },
        ]
        aggregated = aggregator.aggregate_query_results(results)

        assert len(aggregated.aggregated_metrics) > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_single_result(self, aggregator):
        """Test aggregating single result."""
        results = [{"success": True, "time_ms": 10.0}]
        aggregated = aggregator.aggregate_query_results(results)

        assert aggregated.total_results == 1

    def test_all_failures(self, aggregator):
        """Test aggregating all failures."""
        results = [
            {"success": False},
            {"success": False},
            {"success": False},
        ]
        aggregated = aggregator.aggregate_query_results(results)

        assert aggregated.failed_results == 3
        assert aggregated.successful_results == 0

    def test_mixed_metric_types(self, aggregator):
        """Test aggregating mixed metric types."""
        metrics_list = [
            {"int_metric": 10, "float_metric": 10.5, "str_metric": "skip"},
            {"int_metric": 20, "float_metric": 20.5, "str_metric": "skip"},
        ]
        aggregated = aggregator.aggregate_metrics(metrics_list)

        # String metrics should be skipped
        assert "int_metric" in aggregated
        assert "float_metric" in aggregated


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_aggregation(self, aggregator):
        """Test complete aggregation workflow."""
        # Generate results
        results = [
            {
                "query": f"q{i}",
                "success": True,
                "time_ms": 10.0 + i,
                "metrics": {"score": 0.8 + (i % 20) / 100.0},
            }
            for i in range(50)
        ]

        # Aggregate
        aggregated = aggregator.aggregate_query_results(results)

        # Verify
        assert aggregated.total_results == 50
        assert aggregated.successful_results == 50
        assert len(aggregated.percentile_latencies) == 3
        assert len(aggregated.aggregated_metrics) > 0
        assert 0.0 <= aggregated.consistency_score <= 1.0

    def test_multi_stage_aggregation(self, aggregator):
        """Test aggregating already-aggregated results."""
        # First stage: per-worker aggregation
        worker_1_results = [{"time_ms": float(i)} for i in range(10)]
        worker_2_results = [{"time_ms": float(i + 10)} for i in range(10)]

        # Both mark success
        for r in worker_1_results + worker_2_results:
            r["success"] = True

        # Aggregate worker 1
        agg1 = aggregator.aggregate_query_results(worker_1_results)
        # Aggregate worker 2
        agg2 = aggregator.aggregate_query_results(worker_2_results)

        # Verify both aggregations
        assert agg1.successful_results == 10
        assert agg2.successful_results == 10
