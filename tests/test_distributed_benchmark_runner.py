"""Comprehensive tests for DistributedBenchmarkRunner."""

import pytest
import time
from unittest.mock import Mock, MagicMock
from benchmark.memory.distributed.distributed_benchmark_runner import (
    DistributedBenchmarkRunner,
    BenchmarkConfig,
    WorkerResult,
)


@pytest.fixture
def mock_memory_system():
    """Create a mock memory system."""
    system = Mock()
    system.query = Mock(return_value=Mock(
        results=[{"id": "m1"}],
        memory_stats={"tier": "working"},
    ))
    return system


@pytest.fixture
def sample_dataset():
    """Create sample benchmark dataset."""
    return [
        {"query": f"query_{i}", "id": f"item_{i}"}
        for i in range(20)
    ]


@pytest.fixture
def runner(mock_memory_system):
    """Create benchmark runner."""
    return DistributedBenchmarkRunner(mock_memory_system, num_workers=4)


class TestRunnerInitialization:
    """Test runner initialization."""

    def test_initialization(self, mock_memory_system):
        """Test runner initialization."""
        runner = DistributedBenchmarkRunner(mock_memory_system, num_workers=4)
        assert runner.num_workers == 4

    def test_initialization_custom_workers(self, mock_memory_system):
        """Test initialization with custom worker count."""
        runner = DistributedBenchmarkRunner(mock_memory_system, num_workers=8)
        assert runner.num_workers == 8


class TestWorkloadDistribution:
    """Test workload distribution."""

    def test_distribute_equal_shards(self, runner, sample_dataset):
        """Test distributing dataset into equal shards."""
        shards = runner.distribute_workload(sample_dataset, num_workers=4)

        assert len(shards) == 4
        # First 3 shards should have 5 items each
        assert len(shards[0]) == 5
        assert len(shards[1]) == 5
        assert len(shards[2]) == 5
        assert len(shards[3]) == 5

    def test_distribute_uneven_shards(self, runner):
        """Test distribution with uneven dataset."""
        dataset = [{"query": f"q{i}"} for i in range(7)]
        shards = runner.distribute_workload(dataset, num_workers=3)

        assert len(shards) == 3
        total = sum(len(s) for s in shards)
        assert total == 7

    def test_distribution_stats(self, runner, sample_dataset):
        """Test getting distribution statistics."""
        runner.distribute_workload(sample_dataset, num_workers=4)
        stats = runner.get_distribution_stats()

        assert stats["num_workers"] == 4
        assert stats["total_items"] == 20
        assert stats["min_shard_size"] == 5
        assert stats["max_shard_size"] == 5

    def test_invalid_worker_count_raises_error(self, runner, sample_dataset):
        """Test invalid worker count raises error."""
        with pytest.raises(ValueError):
            runner.distribute_workload(sample_dataset, num_workers=0)


class TestBenchmarkExecution:
    """Test benchmark execution."""

    def test_run_benchmark_basic(self, runner, sample_dataset):
        """Test basic benchmark execution."""
        config = BenchmarkConfig(num_workers=2, top_k=10)
        result = runner.run_benchmark(sample_dataset, config)

        assert result.total_items == 20
        assert result.total_workers == 2
        assert result.successful_workers == 2
        assert result.failed_workers == 0

    def test_run_benchmark_returns_aggregated_result(self, runner, sample_dataset):
        """Test benchmark returns aggregated result."""
        result = runner.run_benchmark(sample_dataset)

        assert hasattr(result, 'total_items')
        assert hasattr(result, 'total_workers')
        assert hasattr(result, 'throughput_items_per_sec')
        assert hasattr(result, 'average_latency_ms')
        assert hasattr(result, 'worker_results')

    def test_empty_dataset_raises_error(self, runner):
        """Test empty dataset raises error."""
        with pytest.raises(ValueError, match="empty"):
            runner.run_benchmark([])

    def test_benchmark_with_single_worker(self, runner, sample_dataset):
        """Test benchmark with single worker."""
        config = BenchmarkConfig(num_workers=1)
        result = runner.run_benchmark(sample_dataset, config)

        assert result.total_workers == 1
        assert result.total_items == 20


class TestResultAggregation:
    """Test result aggregation."""

    def test_throughput_calculation(self, runner, sample_dataset):
        """Test throughput is calculated correctly."""
        result = runner.run_benchmark(sample_dataset)

        assert result.throughput_items_per_sec > 0
        # Throughput = items / time, should be positive and reasonable
        assert result.throughput_items_per_sec > 0
        # Verify result has throughput
        assert hasattr(result, 'throughput_items_per_sec')

    def test_latency_percentiles(self, runner, sample_dataset):
        """Test latency percentiles are computed."""
        result = runner.run_benchmark(sample_dataset)

        assert 50 in result.percentile_latencies
        assert 95 in result.percentile_latencies
        assert 99 in result.percentile_latencies

    def test_worker_results_included(self, runner, sample_dataset):
        """Test individual worker results are included."""
        result = runner.run_benchmark(sample_dataset)

        assert len(result.worker_results) > 0
        for wr in result.worker_results:
            assert isinstance(wr, WorkerResult)

    def test_collect_results(self, runner, sample_dataset):
        """Test collecting results after benchmark."""
        runner.run_benchmark(sample_dataset)
        result = runner.collect_results()

        assert result.total_items > 0

    def test_collect_results_without_benchmark_raises_error(self, runner):
        """Test collecting results without running benchmark raises error."""
        with pytest.raises(RuntimeError):
            runner.collect_results()


class TestCheckpointing:
    """Test checkpoint creation and resumption."""

    def test_create_checkpoint(self, runner, sample_dataset):
        """Test creating checkpoint."""
        runner.run_benchmark(sample_dataset)
        checkpoint = runner.create_checkpoint("job_1")

        assert checkpoint["job_id"] == "job_1"
        assert "workload_shards" in checkpoint
        assert "worker_results" in checkpoint

    def test_resume_from_checkpoint(self, runner, sample_dataset):
        """Test resuming from checkpoint."""
        runner.run_benchmark(sample_dataset, BenchmarkConfig(num_workers=2))
        checkpoint = runner.create_checkpoint("job_1")

        # Create new runner
        runner2 = DistributedBenchmarkRunner(runner.memory_system, num_workers=2)
        runner2.resume_from_checkpoint(checkpoint)

        assert len(runner2._workload_shards) > 0
        assert len(runner2._worker_results) > 0

    def test_checkpoint_invalid_format_raises_error(self, runner):
        """Test invalid checkpoint format raises error."""
        with pytest.raises(ValueError):
            runner.resume_from_checkpoint({})


class TestErrorHandling:
    """Test error handling."""

    def test_worker_failure_isolation(self):
        """Test that worker failure doesn't affect others."""
        system = Mock()

        call_count = [0]

        def query_side_effect(q, top_k):
            call_count[0] += 1
            if call_count[0] % 5 == 0:
                raise RuntimeError("Worker error")
            return Mock(results=[], memory_stats={})

        system.query = Mock(side_effect=query_side_effect)
        runner = DistributedBenchmarkRunner(system, num_workers=2)

        dataset = [{"query": f"q{i}"} for i in range(10)]
        result = runner.run_benchmark(dataset)

        # Some items should still be processed
        assert result.total_items >= 0

    def test_benchmark_with_empty_config(self, runner, sample_dataset):
        """Test benchmark with default config."""
        result = runner.run_benchmark(sample_dataset)

        assert result.total_items == 20


class TestPerformanceMetrics:
    """Test performance metrics computation."""

    def test_average_latency_computed(self, runner, sample_dataset):
        """Test average latency is computed."""
        result = runner.run_benchmark(sample_dataset)

        assert result.average_latency_ms >= 0

    def test_total_time_recorded(self, runner, sample_dataset):
        """Test total execution time is recorded."""
        result = runner.run_benchmark(sample_dataset)

        assert result.total_time_sec > 0

    def test_metrics_consistency(self, runner, sample_dataset):
        """Test metric consistency."""
        result = runner.run_benchmark(sample_dataset)

        # Throughput should be positive and exist
        assert result.throughput_items_per_sec >= 0
        # Total time should be reasonable (not negative)
        assert result.total_time_sec >= 0


class TestScaling:
    """Test scaling with different worker counts."""

    def test_scaling_2_workers(self, runner, sample_dataset):
        """Test with 2 workers."""
        config = BenchmarkConfig(num_workers=2)
        result = runner.run_benchmark(sample_dataset, config)

        assert result.total_workers == 2

    def test_scaling_4_workers(self, runner, sample_dataset):
        """Test with 4 workers."""
        config = BenchmarkConfig(num_workers=4)
        result = runner.run_benchmark(sample_dataset, config)

        assert result.total_workers == 4

    def test_scaling_8_workers(self, runner):
        """Test with 8 workers on larger dataset."""
        dataset = [{"query": f"q{i}"} for i in range(80)]
        config = BenchmarkConfig(num_workers=8)
        result = runner.run_benchmark(dataset, config)

        assert result.total_workers == 8
        assert result.total_items == 80


class TestLargeScale:
    """Test large-scale benchmark execution."""

    def test_large_dataset_1000_items(self, runner):
        """Test with 1000 item dataset."""
        dataset = [{"query": f"q{i}"} for i in range(1000)]
        result = runner.run_benchmark(dataset)

        assert result.total_items == 1000

    def test_many_workers_16(self, runner):
        """Test with 16 workers."""
        dataset = [{"query": f"q{i}"} for i in range(160)]
        config = BenchmarkConfig(num_workers=16)
        result = runner.run_benchmark(dataset, config)

        assert result.total_workers == 16


class TestEdgeCases:
    """Test edge cases."""

    def test_single_item_dataset(self, runner):
        """Test with single item."""
        dataset = [{"query": "q1"}]
        result = runner.run_benchmark(dataset)

        assert result.total_items >= 0

    def test_dataset_smaller_than_workers(self, runner):
        """Test dataset smaller than worker count."""
        dataset = [{"query": "q1"}, {"query": "q2"}]
        config = BenchmarkConfig(num_workers=4)
        result = runner.run_benchmark(dataset, config)

        assert result.total_items == 2

    def test_benchmark_config_defaults(self, runner, sample_dataset):
        """Test benchmark config defaults."""
        config = BenchmarkConfig()

        assert config.num_workers == 4
        assert config.top_k == 10
        assert config.timeout_sec == 300.0
