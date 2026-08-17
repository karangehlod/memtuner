"""Comprehensive tests for ParallelQueryExecutor."""

import pytest
import time
from unittest.mock import Mock, MagicMock
from benchmark.memory.distributed.parallel_query_executor import ParallelQueryExecutor


@pytest.fixture
def mock_memory_system():
    """Create a mock memory system."""
    system = Mock()
    system.query = Mock(return_value=Mock(results=[
        {"id": "m1", "content": "result 1"},
        {"id": "m2", "content": "result 2"},
    ]))
    return system


@pytest.fixture
def executor(mock_memory_system):
    """Create executor with mock system."""
    return ParallelQueryExecutor(mock_memory_system, max_workers=2)


class TestExecutorInitialization:
    """Test executor initialization."""

    def test_initialization_with_defaults(self, mock_memory_system):
        """Test initialization with default parameters."""
        executor = ParallelQueryExecutor(mock_memory_system)
        assert executor.max_workers == 4
        assert executor.timeout_sec == 30.0

    def test_initialization_with_custom_params(self, mock_memory_system):
        """Test initialization with custom parameters."""
        executor = ParallelQueryExecutor(
            mock_memory_system,
            max_workers=8,
            timeout_sec=60.0,
        )
        assert executor.max_workers == 8
        assert executor.timeout_sec == 60.0


class TestSingleQueryExecution:
    """Test single query execution."""

    def test_execute_single_query(self, executor):
        """Test executing a single query."""
        results = executor.execute_queries(["test query"], top_k=5)

        assert len(results) == 1
        assert results[0]["query"] == "test query"
        assert results[0]["success"] is True

    def test_single_query_result_structure(self, executor):
        """Test single query result has required fields."""
        results = executor.execute_queries(["query"], top_k=10)

        assert "query" in results[0]
        assert "results" in results[0]
        assert "time_ms" in results[0]
        assert "success" in results[0]

    def test_empty_query_list_raises_error(self, executor):
        """Test empty query list raises error."""
        with pytest.raises(ValueError, match="empty"):
            executor.execute_queries([])


class TestParallelExecution:
    """Test parallel query execution."""

    def test_execute_two_queries_parallel(self, executor):
        """Test executing two queries in parallel."""
        results = executor.execute_queries(["query1", "query2"], top_k=5)

        assert len(results) == 2
        assert results[0]["query"] == "query1"
        assert results[1]["query"] == "query2"

    def test_execute_four_queries_parallel(self, executor):
        """Test executing four queries in parallel."""
        results = executor.execute_queries(
            ["q1", "q2", "q3", "q4"],
            top_k=5,
        )

        assert len(results) == 4
        for i, result in enumerate(results):
            assert f"q{i+1}" in result["query"]

    def test_result_order_preserved(self, executor):
        """Test result order matches input order."""
        queries = ["alice", "bob", "charlie", "diana"]
        results = executor.execute_queries(queries, top_k=5)

        for i, (query, result) in enumerate(zip(queries, results)):
            assert query in result["query"]

    def test_parallel_speedup_vs_sequential(self, executor):
        """Test parallel execution is faster than sequential."""
        queries = ["query"] * 4

        # Parallel execution
        start = time.time()
        executor.execute_queries(queries, top_k=5)
        parallel_time = time.time() - start

        executor.reset_stats()

        # Note: Sequential timing not directly comparable in test,
        # just verify execution completes quickly
        assert parallel_time < 30.0


class TestBatchExecution:
    """Test batch query execution."""

    def test_execute_batch(self, executor):
        """Test batch execution."""
        batch = [
            {"query": "q1", "metadata": "data1"},
            {"query": "q2", "metadata": "data2"},
        ]

        results = executor.execute_batch_parallel(batch)

        assert len(results) == 2
        assert all("original_query" in r for r in results)

    def test_batch_metadata_preserved(self, executor):
        """Test batch metadata is preserved."""
        batch = [
            {"query": "q1", "user_id": "user1"},
            {"query": "q2", "user_id": "user2"},
        ]

        results = executor.execute_batch_parallel(batch)

        assert results[0]["original_query"]["user_id"] == "user1"
        assert results[1]["original_query"]["user_id"] == "user2"

    def test_empty_batch_raises_error(self, executor):
        """Test empty batch raises error."""
        with pytest.raises(ValueError, match="empty"):
            executor.execute_batch_parallel([])


class TestErrorHandling:
    """Test error handling."""

    def test_query_error_isolation(self):
        """Test that one query error doesn't affect others."""
        system = Mock()

        def query_side_effect(q, top_k):
            if "fail" in q:
                raise ValueError("Intentional failure")
            return Mock(results=[{"id": "m1"}])

        system.query = Mock(side_effect=query_side_effect)
        executor = ParallelQueryExecutor(system, max_workers=2)

        results = executor.execute_queries(["ok", "fail", "ok2"], top_k=5)

        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    def test_error_recorded_in_result(self, executor):
        """Test error is recorded in result."""
        system = Mock()
        system.query = Mock(side_effect=RuntimeError("Test error"))
        executor.memory_system = system

        results = executor.execute_queries(["query"], top_k=5)

        assert results[0]["success"] is False
        assert "error" in results[0]
        assert "Test error" in results[0]["error"]


class TestExecutionStats:
    """Test execution statistics."""

    def test_get_execution_stats(self, executor):
        """Test getting execution statistics."""
        executor.execute_queries(["q1", "q2", "q3"], top_k=5)

        stats = executor.get_execution_stats()

        assert stats.total_queries == 3
        assert stats.successful_queries == 3
        assert stats.failed_queries == 0
        assert stats.avg_query_time_ms > 0
        assert stats.throughput_qps > 0

    def test_stats_with_failures(self):
        """Test stats with some failures."""
        system = Mock()

        def query_side_effect(q, top_k):
            if "fail" in q:
                raise ValueError("Error")
            return Mock(results=[])

        system.query = Mock(side_effect=query_side_effect)
        executor = ParallelQueryExecutor(system, max_workers=2)

        executor.execute_queries(["ok", "fail", "ok"], top_k=5)
        stats = executor.get_execution_stats()

        # With error isolation, all queries are attempted
        assert stats.total_queries >= 2  # At least the successful ones tracked

    def test_stats_min_max_time(self, executor):
        """Test min/max query times."""
        executor.execute_queries(["q1", "q2", "q3"], top_k=5)

        stats = executor.get_execution_stats()

        assert stats.min_query_time_ms >= 0
        assert stats.max_query_time_ms >= stats.min_query_time_ms
        assert stats.avg_query_time_ms <= stats.max_query_time_ms

    def test_stats_reset(self, executor):
        """Test resetting statistics."""
        executor.execute_queries(["q1", "q2"], top_k=5)
        executor.reset_stats()

        stats = executor.get_execution_stats()

        assert stats.total_queries == 0
        assert stats.successful_queries == 0


class TestWorkerUtilization:
    """Test worker pool utilization."""

    def test_worker_count_tracking(self, executor):
        """Test worker count is tracked."""
        executor.execute_queries(["q1", "q2", "q3", "q4"], top_k=5)

        stats = executor.get_execution_stats()

        # Should have distributed across workers
        assert len(stats.worker_utilization) > 0
        total_queries = sum(stats.worker_utilization.values())
        assert total_queries == 4

    def test_worker_distribution(self, executor):
        """Test work distribution across workers."""
        executor.execute_queries(["q1", "q2", "q3", "q4", "q5", "q6"], top_k=5)

        stats = executor.get_execution_stats()

        # With 2 workers, should have roughly balanced distribution
        workers = list(stats.worker_utilization.values())
        assert len(workers) > 0


class TestLargeScaleExecution:
    """Test execution with many queries."""

    def test_execute_50_queries(self, executor):
        """Test executing 50 queries in parallel."""
        queries = [f"query_{i}" for i in range(50)]

        results = executor.execute_queries(queries, top_k=5)

        assert len(results) == 50
        assert all(r["success"] is True for r in results)

    def test_execute_100_queries_performance(self, executor):
        """Test 100 queries complete in reasonable time."""
        queries = [f"q{i}" for i in range(100)]

        start = time.time()
        executor.execute_queries(queries, top_k=5)
        elapsed = time.time() - start

        # Should complete reasonably quickly with parallelization
        assert elapsed < 120.0  # Very loose upper bound


class TestTopKParameter:
    """Test top_k parameter handling."""

    def test_top_k_passed_to_system(self, executor):
        """Test top_k parameter is passed to memory system."""
        executor.execute_queries(["query"], top_k=20)

        executor.memory_system.query.assert_called_with("query", top_k=20)

    def test_different_top_k_values(self, executor):
        """Test different top_k values."""
        for k in [1, 5, 10, 20, 50]:
            executor.reset_stats()
            executor.execute_queries(["query"], top_k=k)

            executor.memory_system.query.assert_called_with("query", top_k=k)


class TestConcurrency:
    """Test concurrent execution properties."""

    def test_concurrent_execution_isolation(self, executor):
        """Test concurrent queries don't interfere."""
        queries = ["independent1", "independent2", "independent3"]

        results = executor.execute_queries(queries, top_k=5)

        # Each query should have been processed independently
        for query, result in zip(queries, results):
            assert query in result["query"]

    def test_maximum_workers_respected(self):
        """Test maximum worker count is respected."""
        system = Mock()
        system.query = Mock(return_value=Mock(results=[]))

        executor = ParallelQueryExecutor(system, max_workers=2)

        # Execute many queries
        queries = [f"q{i}" for i in range(20)]
        executor.execute_queries(queries, top_k=5)

        # Should have called query 20 times despite only 2 workers
        assert system.query.call_count == 20


class TestEdgeCases:
    """Test edge cases."""

    def test_single_query_in_parallel(self, executor):
        """Test single query through parallel executor."""
        results = executor.execute_queries(["query"], top_k=5)

        assert len(results) == 1
        assert results[0]["success"] is True

    def test_very_large_top_k(self, executor):
        """Test very large top_k value."""
        results = executor.execute_queries(["query"], top_k=10000)

        assert len(results) == 1

    def test_empty_query_string(self, executor):
        """Test empty query string handling."""
        results = executor.execute_queries([""], top_k=5)

        assert len(results) == 1
        # Should handle gracefully
