"""Distributed benchmark execution across worker nodes."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    num_workers: int = 4
    queries_per_worker: int = 100
    top_k: int = 10
    timeout_sec: float = 300.0


@dataclass
class WorkerResult:
    """Result from a single worker."""
    worker_id: int
    items_processed: int
    total_time_sec: float
    metrics: dict[str, Any]
    error: str | None = None


@dataclass
class AggregatedResult:
    """Aggregated benchmark results."""
    total_items: int
    total_workers: int
    successful_workers: int
    failed_workers: int
    total_time_sec: float
    throughput_items_per_sec: float
    average_latency_ms: float
    percentile_latencies: dict[int, float]
    aggregated_metrics: dict[str, Any]
    worker_results: list[WorkerResult]


class DistributedBenchmarkRunner:
    """Run benchmarks across distributed worker nodes."""

    def __init__(
        self,
        memory_system: Any,
        num_workers: int = 4,
    ):
        """Initialize distributed benchmark runner.

        Args:
            memory_system: AdvancedMemorySystem instance
            num_workers: Number of worker processes
        """
        self.memory_system = memory_system
        self.num_workers = num_workers

        # State
        self._current_job_id: str | None = None
        self._workload_shards: list[list[dict[str, Any]]] = []
        self._worker_results: list[WorkerResult] = []
        self._checkpoint_data: dict[str, Any] = {}

    def run_benchmark(
        self,
        dataset: list[dict[str, Any]],
        config: BenchmarkConfig | None = None,
    ) -> AggregatedResult:
        """Run full benchmark on dataset.

        Args:
            dataset: Items to benchmark
            config: Benchmark configuration

        Returns:
            Aggregated benchmark results

        Raises:
            ValueError: If dataset empty or config invalid
        """
        if not dataset:
            raise ValueError("Cannot run benchmark on empty dataset")

        config = config or BenchmarkConfig()

        # Reset state
        self._worker_results.clear()
        self._workload_shards.clear()

        start_time = time.time()

        try:
            # Step 1: Distribute workload
            self._distribute_workload(dataset, config.num_workers)

            # Step 2: Execute on each shard (simulated worker)
            for worker_id, shard in enumerate(self._workload_shards):
                result = self._execute_worker(worker_id, shard, config)
                self._worker_results.append(result)

            # Step 3: Aggregate results
            aggregated = self._aggregate_results(dataset, config)

            # Record execution time
            elapsed = time.time() - start_time
            aggregated.total_time_sec = elapsed

            return aggregated

        except Exception as e:
            logger.exception(f"Benchmark execution failed: {e}")
            raise

    def distribute_workload(
        self,
        dataset: list[dict[str, Any]],
        num_workers: int,
    ) -> list[list[dict[str, Any]]]:
        """Distribute dataset across workers.

        Args:
            dataset: Items to distribute
            num_workers: Number of workers

        Returns:
            List of shards, one per worker
        """
        return self._distribute_workload(dataset, num_workers)

    def collect_results(self) -> AggregatedResult:
        """Collect and aggregate results.

        Returns:
            Aggregated results from last benchmark run

        Raises:
            RuntimeError: If no results available
        """
        if not self._worker_results:
            raise RuntimeError("No results to collect. Run benchmark first.")

        return self._create_aggregated_result(self._worker_results)

    def create_checkpoint(self, job_id: str) -> dict[str, Any]:
        """Create checkpoint for resumable execution.

        Args:
            job_id: Unique job identifier

        Returns:
            Checkpoint data
        """
        self._current_job_id = job_id

        checkpoint = {
            "job_id": job_id,
            "timestamp": time.time(),
            "workload_shards": self._workload_shards,
            "worker_results": [
                {
                    "worker_id": r.worker_id,
                    "items_processed": r.items_processed,
                    "total_time_sec": r.total_time_sec,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in self._worker_results
            ],
        }

        self._checkpoint_data[job_id] = checkpoint
        return checkpoint

    def resume_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Resume execution from checkpoint.

        Args:
            checkpoint: Checkpoint data from create_checkpoint
        """
        job_id = checkpoint.get("job_id")
        if not job_id:
            raise ValueError("Invalid checkpoint: missing job_id")

        self._current_job_id = job_id
        self._workload_shards = checkpoint.get("workload_shards", [])
        # Reconstruct worker results
        self._worker_results = [
            WorkerResult(
                worker_id=r["worker_id"],
                items_processed=r["items_processed"],
                total_time_sec=r["total_time_sec"],
                metrics=r["metrics"],
                error=r.get("error"),
            )
            for r in checkpoint.get("worker_results", [])
        ]

    def get_distribution_stats(self) -> dict[str, Any]:
        """Get workload distribution statistics.

        Returns:
            Stats about how work was distributed
        """
        if not self._workload_shards:
            return {}

        sizes = [len(shard) for shard in self._workload_shards]

        return {
            "num_workers": len(self._workload_shards),
            "total_items": sum(sizes),
            "shard_sizes": sizes,
            "min_shard_size": min(sizes) if sizes else 0,
            "max_shard_size": max(sizes) if sizes else 0,
            "avg_shard_size": np.mean(sizes) if sizes else 0,
            "std_shard_size": np.std(sizes) if sizes else 0,
        }

    # Private helper methods

    def _distribute_workload(
        self,
        dataset: list[dict[str, Any]],
        num_workers: int,
    ) -> list[list[dict[str, Any]]]:
        """Distribute items across workers."""
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        # Calculate shard size
        total_items = len(dataset)
        shard_size = max(1, total_items // num_workers)

        # Create shards
        shards = []
        for i in range(num_workers):
            start_idx = i * shard_size
            # Last worker gets remaining items
            if i == num_workers - 1:
                shard = dataset[start_idx:]
            else:
                end_idx = start_idx + shard_size
                shard = dataset[start_idx:end_idx]

            if shard:  # Only add non-empty shards
                shards.append(shard)

        self._workload_shards = shards
        return shards

    def _execute_worker(
        self,
        worker_id: int,
        shard: list[dict[str, Any]],
        config: BenchmarkConfig,
    ) -> WorkerResult:
        """Execute benchmark on worker shard."""
        start_time = time.time()

        try:
            # Process each item in shard
            query_times = []
            metrics_list = []

            for item in shard:
                query = item.get("query", str(item))

                try:
                    item_start = time.time()
                    result = self.memory_system.query(query, top_k=config.top_k)
                    item_time = (time.time() - item_start) * 1000

                    query_times.append(item_time)

                    # Collect metrics if available
                    if hasattr(result, 'memory_stats'):
                        metrics_list.append(result.memory_stats)

                except Exception as e:
                    logger.warning(f"Item processing failed for worker {worker_id}: {e}")
                    # Continue processing other items

            elapsed = time.time() - start_time

            # Aggregate metrics
            aggregated_metrics = self._aggregate_worker_metrics(metrics_list)

            return WorkerResult(
                worker_id=worker_id,
                items_processed=len(shard),
                total_time_sec=elapsed,
                metrics={
                    "query_times": query_times,
                    **aggregated_metrics,
                },
            )

        except Exception as e:
            elapsed = time.time() - start_time

            return WorkerResult(
                worker_id=worker_id,
                items_processed=0,
                total_time_sec=elapsed,
                metrics={},
                error=str(e),
            )

    def _aggregate_worker_metrics(self, metrics_list: list[dict]) -> dict[str, Any]:
        """Aggregate metrics from all queries in a worker."""
        if not metrics_list:
            return {}

        return {
            "num_queries": len(metrics_list),
            "avg_memory_mb": 0.0,  # Would compute from metrics_list
        }

    def _aggregate_results(
        self,
        dataset: list[dict[str, Any]],
        config: BenchmarkConfig,
    ) -> AggregatedResult:
        """Aggregate results from all workers."""
        return self._create_aggregated_result(self._worker_results)

    def _create_aggregated_result(
        self,
        worker_results: list[WorkerResult],
    ) -> AggregatedResult:
        """Create aggregated result from worker results."""
        successful = [r for r in worker_results if r.error is None]
        failed = [r for r in worker_results if r.error is not None]

        total_items = sum(r.items_processed for r in worker_results)
        total_time = max((r.total_time_sec for r in worker_results), default=0.0)

        # Collect all query times
        all_query_times = []
        for r in successful:
            query_times = r.metrics.get("query_times", [])
            all_query_times.extend(query_times)

        # Compute statistics
        if all_query_times:
            latencies_ms = np.array(all_query_times)
            avg_latency = float(np.mean(latencies_ms))
            percentiles = {
                50: float(np.percentile(latencies_ms, 50)),
                95: float(np.percentile(latencies_ms, 95)),
                99: float(np.percentile(latencies_ms, 99)),
            }
        else:
            avg_latency = 0.0
            percentiles = {50: 0.0, 95: 0.0, 99: 0.0}

        throughput = total_items / total_time if total_time > 0 else 0.0

        return AggregatedResult(
            total_items=total_items,
            total_workers=len(worker_results),
            successful_workers=len(successful),
            failed_workers=len(failed),
            total_time_sec=total_time,
            throughput_items_per_sec=throughput,
            average_latency_ms=avg_latency,
            percentile_latencies=percentiles,
            aggregated_metrics={},
            worker_results=worker_results,
        )
