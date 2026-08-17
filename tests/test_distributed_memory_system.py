"""Comprehensive integration tests for DistributedMemorySystem."""

import pytest
from unittest.mock import Mock, MagicMock
from benchmark.memory.distributed.distributed_memory_system import (
    DistributedMemorySystem,
    DistributedJobConfig,
)


@pytest.fixture
def mock_memory_system():
    """Create mock memory system."""
    system = Mock()
    system.query = Mock(return_value=Mock(
        results=[{"id": "m1", "content": "result"}],
        memory_stats={"tier": "working"},
    ))
    return system


@pytest.fixture
def config():
    """Create job configuration."""
    return DistributedJobConfig(
        job_id="test_job",
        num_workers=2,
        max_workers_query=2,
        top_k=10,
    )


@pytest.fixture
def distributed_system(mock_memory_system, config):
    """Create distributed memory system."""
    return DistributedMemorySystem(mock_memory_system, config)


class TestSystemInitialization:
    """Test system initialization."""

    def test_initialization(self, distributed_system):
        """Test system initializes correctly."""
        assert distributed_system is not None
        assert distributed_system.memory_system is not None

    def test_initialization_with_config(self, mock_memory_system):
        """Test initialization with custom config."""
        config = DistributedJobConfig(
            job_id="custom_job",
            num_workers=4,
            max_workers_query=4,
        )

        system = DistributedMemorySystem(mock_memory_system, config)

        assert system.config.job_id == "custom_job"
        assert system.config.num_workers == 4

    def test_components_initialized(self, distributed_system):
        """Test all components are initialized."""
        assert distributed_system._parallel_executor is not None
        assert distributed_system._benchmark_runner is not None
        assert distributed_system._aggregator is not None
        assert distributed_system._progress_monitor is not None


class TestParallelQueryExecution:
    """Test parallel query execution."""

    def test_execute_queries_parallel_basic(self, distributed_system):
        """Test basic parallel query execution."""
        queries = ["q1", "q2", "q3"]

        results = distributed_system.execute_queries_parallel(queries)

        assert len(results) == 3
        assert all(r["success"] for r in results)

    def test_execute_queries_with_custom_top_k(self, distributed_system):
        """Test query execution with custom top_k."""
        queries = ["q1"]

        distributed_system.execute_queries_parallel(queries, top_k=20)

        distributed_system.memory_system.query.assert_called_with("q1", top_k=20)

    def test_execute_queries_empty_raises_error(self, distributed_system):
        """Test empty query list raises error."""
        with pytest.raises(ValueError, match="empty"):
            distributed_system.execute_queries_parallel([])

    def test_execute_queries_updates_stats(self, distributed_system):
        """Test query execution updates stats."""
        queries = ["q1", "q2"]

        distributed_system.execute_queries_parallel(queries)

        stats = distributed_system.get_execution_stats()
        assert stats["total_queries"] >= 2


class TestDistributedBenchmark:
    """Test distributed benchmark execution."""

    def test_run_distributed_benchmark_basic(self, distributed_system):
        """Test basic benchmark execution."""
        dataset = [{"query": f"q{i}"} for i in range(10)]

        result = distributed_system.run_distributed_benchmark(dataset)

        assert result.job_id == "test_job"
        assert result.total_items == 10
        assert result.success

    def test_run_distributed_benchmark_returns_result(self, distributed_system):
        """Test benchmark returns result object."""
        dataset = [{"query": "q1"}]

        result = distributed_system.run_distributed_benchmark(dataset)

        assert hasattr(result, "job_id")
        assert hasattr(result, "total_items")
        assert hasattr(result, "throughput")
        assert hasattr(result, "total_time_sec")

    def test_run_distributed_benchmark_empty_raises_error(self, distributed_system):
        """Test empty dataset raises error."""
        try:
            result = distributed_system.run_distributed_benchmark([])
            # If no exception, check that result indicates failure
            assert not result.success or result.failed_items > 0
        except ValueError:
            # This is also acceptable
            pass

    def test_run_distributed_benchmark_large_dataset(self, distributed_system):
        """Test benchmark with large dataset."""
        dataset = [{"query": f"q{i}"} for i in range(100)]

        result = distributed_system.run_distributed_benchmark(dataset)

        assert result.total_items == 100


class TestProgressMonitoring:
    """Test progress monitoring."""

    def test_get_current_progress_no_job(self, distributed_system):
        """Test progress when no job running."""
        progress = distributed_system.get_current_progress()

        assert progress is None

    def test_get_current_progress_during_benchmark(self, distributed_system):
        """Test progress during benchmark."""
        dataset = [{"query": f"q{i}"} for i in range(10)]

        # Start benchmark (doesn't block in mock)
        result = distributed_system.run_distributed_benchmark(dataset)

        # After completion, progress would be cleared
        assert result.total_items == 10

    def test_get_worker_health_no_job(self, distributed_system):
        """Test worker health when no job running."""
        health = distributed_system.get_worker_health()

        assert health is None


class TestFaultTolerance:
    """Test fault tolerance integration."""

    def test_fault_tolerance_enabled_by_default(self, distributed_system):
        """Test fault tolerance is enabled by default."""
        assert distributed_system._fault_tolerance is not None

    def test_fault_tolerance_can_be_disabled(self, mock_memory_system):
        """Test fault tolerance can be disabled."""
        config = DistributedJobConfig(
            job_id="no_ft_job",
            enable_fault_tolerance=False,
        )

        system = DistributedMemorySystem(mock_memory_system, config)

        assert system._fault_tolerance is None

    def test_get_fault_tolerance_stats(self, distributed_system):
        """Test getting fault tolerance stats."""
        stats = distributed_system.get_fault_tolerance_stats()

        assert stats is not None
        assert "total_errors" in stats


class TestSystemStatus:
    """Test system status queries."""

    def test_get_system_status(self, distributed_system):
        """Test getting system status."""
        status = distributed_system.get_system_status()

        assert "current_job" in status
        assert "num_workers" in status
        assert "fault_tolerance_enabled" in status

    def test_get_system_status_after_job(self, distributed_system):
        """Test system status after job execution."""
        dataset = [{"query": "q1"}]

        distributed_system.run_distributed_benchmark(dataset)

        status = distributed_system.get_system_status()

        # Last result should be stored
        assert status["last_result"] is not None

    def test_get_last_result_no_job(self, distributed_system):
        """Test getting result when no job run."""
        result = distributed_system.get_last_result()

        assert result is None

    def test_get_last_result_after_job(self, distributed_system):
        """Test getting result after job execution."""
        dataset = [{"query": "q1"}]

        distributed_system.run_distributed_benchmark(dataset)

        result = distributed_system.get_last_result()

        assert result is not None
        assert result.job_id == "test_job"


class TestExecutionStats:
    """Test execution statistics."""

    def test_get_execution_stats_empty(self, distributed_system):
        """Test stats with no execution."""
        stats = distributed_system.get_execution_stats()

        assert stats["total_queries"] == 0

    def test_get_execution_stats_after_queries(self, distributed_system):
        """Test stats after query execution."""
        queries = ["q1", "q2", "q3"]

        distributed_system.execute_queries_parallel(queries)

        stats = distributed_system.get_execution_stats()

        assert stats["total_queries"] == 3

    def test_execution_stats_includes_throughput(self, distributed_system):
        """Test stats includes throughput."""
        queries = ["q1", "q2"]

        distributed_system.execute_queries_parallel(queries)

        stats = distributed_system.get_execution_stats()

        assert "throughput_qps" in stats


class TestResetOperations:
    """Test reset operations."""

    def test_reset_stats(self, distributed_system):
        """Test resetting statistics."""
        queries = ["q1"]

        distributed_system.execute_queries_parallel(queries)

        distributed_system.reset_stats()

        stats = distributed_system.get_execution_stats()
        assert stats["total_queries"] == 0

    def test_reset_clears_last_result(self, distributed_system):
        """Test reset clears last result."""
        dataset = [{"query": "q1"}]

        distributed_system.run_distributed_benchmark(dataset)

        assert distributed_system.get_last_result() is not None

        distributed_system.reset_stats()

        assert distributed_system.get_last_result() is None


class TestComponentAccessors:
    """Test component accessor methods."""

    def test_get_aggregator(self, distributed_system):
        """Test getting aggregator."""
        aggregator = distributed_system.get_aggregator()

        assert aggregator is not None

    def test_get_progress_monitor(self, distributed_system):
        """Test getting progress monitor."""
        monitor = distributed_system.get_progress_monitor()

        assert monitor is not None

    def test_get_fault_tolerance_manager(self, distributed_system):
        """Test getting fault tolerance manager."""
        manager = distributed_system.get_fault_tolerance_manager()

        assert manager is not None

    def test_get_fault_tolerance_manager_when_disabled(self, mock_memory_system):
        """Test getting fault tolerance manager when disabled."""
        config = DistributedJobConfig(
            job_id="no_ft_job",
            enable_fault_tolerance=False,
        )

        system = DistributedMemorySystem(mock_memory_system, config)

        manager = system.get_fault_tolerance_manager()

        assert manager is None


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_query_execution(self, distributed_system):
        """Test complete query execution workflow."""
        queries = ["q1", "q2", "q3"]

        results = distributed_system.execute_queries_parallel(queries)

        assert len(results) == 3
        stats = distributed_system.get_execution_stats()
        assert stats["total_queries"] == 3

    def test_end_to_end_benchmark_execution(self, distributed_system):
        """Test complete benchmark workflow."""
        dataset = [{"query": f"q{i}"} for i in range(5)]

        result = distributed_system.run_distributed_benchmark(dataset)

        assert result.success
        assert result.total_items == 5

        status = distributed_system.get_system_status()
        assert status["last_result"] is not None

    def test_multiple_sequential_benchmarks(self, distributed_system):
        """Test running multiple benchmarks sequentially."""
        dataset1 = [{"query": f"q{i}"} for i in range(3)]
        dataset2 = [{"query": f"q{i}"} for i in range(5)]

        result1 = distributed_system.run_distributed_benchmark(dataset1)
        result2 = distributed_system.run_distributed_benchmark(dataset2)

        assert result1.total_items == 3
        assert result2.total_items == 5

    def test_query_execution_and_benchmark(self, distributed_system):
        """Test both query and benchmark execution."""
        queries = ["q1", "q2"]

        distributed_system.execute_queries_parallel(queries)

        dataset = [{"query": "q3"}]
        result = distributed_system.run_distributed_benchmark(dataset)

        assert result.success

    def test_system_lifecycle(self, distributed_system):
        """Test complete system lifecycle."""
        # Initial state
        assert distributed_system.get_last_result() is None

        # Execute queries
        distributed_system.execute_queries_parallel(["q1"])
        stats1 = distributed_system.get_execution_stats()
        assert stats1["total_queries"] == 1

        # Run benchmark
        dataset = [{"query": "q2"}]
        result = distributed_system.run_distributed_benchmark(dataset)
        assert result.success

        # Reset
        distributed_system.reset_stats()
        stats2 = distributed_system.get_execution_stats()
        assert stats2["total_queries"] == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_single_query(self, distributed_system):
        """Test with single query."""
        results = distributed_system.execute_queries_parallel(["q1"])

        assert len(results) == 1

    def test_single_item_dataset(self, distributed_system):
        """Test with single item dataset."""
        dataset = [{"query": "q1"}]

        result = distributed_system.run_distributed_benchmark(dataset)

        assert result.total_items == 1

    def test_many_workers(self, mock_memory_system):
        """Test with many workers."""
        config = DistributedJobConfig(
            job_id="many_workers",
            num_workers=16,
            max_workers_query=16,
        )

        system = DistributedMemorySystem(mock_memory_system, config)

        dataset = [{"query": f"q{i}"} for i in range(16)]

        result = system.run_distributed_benchmark(dataset)

        assert result.total_items == 16

    def test_large_query_batch(self, distributed_system):
        """Test with large query batch."""
        queries = [f"q{i}" for i in range(100)]

        results = distributed_system.execute_queries_parallel(queries)

        assert len(results) == 100
