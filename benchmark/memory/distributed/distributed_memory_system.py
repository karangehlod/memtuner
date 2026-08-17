"""Unified distributed memory system orchestrating all Phase 5 components."""

import logging
from typing import Any, Optional
from dataclasses import dataclass

from benchmark.memory.distributed.parallel_query_executor import ParallelQueryExecutor
from benchmark.memory.distributed.distributed_benchmark_runner import (
    DistributedBenchmarkRunner,
    BenchmarkConfig,
)
from benchmark.memory.distributed.result_aggregator import ResultAggregator
from benchmark.memory.distributed.progress_monitor import ProgressMonitor
from benchmark.memory.distributed.fault_tolerance_manager import FaultToleranceManager

logger = logging.getLogger(__name__)


@dataclass
class DistributedJobConfig:
    """Configuration for distributed job execution."""
    job_id: str
    num_workers: int = 4
    max_workers_query: int = 4
    top_k: int = 10
    enable_fault_tolerance: bool = True
    enable_progress_monitoring: bool = True


@dataclass
class DistributedJobResult:
    """Result from distributed job execution."""
    job_id: str
    total_items: int
    processed_items: int
    failed_items: int
    total_time_sec: float
    throughput: float
    success: bool
    error_message: Optional[str] = None


class DistributedMemorySystem:
    """Unified distributed memory system orchestrator."""

    def __init__(
        self,
        memory_system: Any,
        config: Optional[DistributedJobConfig] = None,
    ):
        """Initialize distributed memory system.

        Args:
            memory_system: AdvancedMemorySystem instance
            config: Job configuration
        """
        self.memory_system = memory_system
        self.config = config or DistributedJobConfig(job_id="job_default")

        # Initialize components
        self._parallel_executor = ParallelQueryExecutor(
            memory_system,
            max_workers=self.config.max_workers_query,
        )

        self._benchmark_runner = DistributedBenchmarkRunner(
            memory_system,
            num_workers=self.config.num_workers,
        )

        self._aggregator = ResultAggregator()
        self._progress_monitor = ProgressMonitor()

        if self.config.enable_fault_tolerance:
            self._fault_tolerance = FaultToleranceManager()
        else:
            self._fault_tolerance = None

        # State
        self._current_job_id: Optional[str] = None
        self._last_result: Optional[DistributedJobResult] = None

    def execute_queries_parallel(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Execute queries in parallel.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of query results
        """
        top_k = top_k or self.config.top_k

        logger.info(f"Executing {len(queries)} queries in parallel")

        try:
            results = self._parallel_executor.execute_queries(queries, top_k=top_k)

            # Aggregate results
            if results:
                aggregated = self._aggregator.aggregate_query_results(results)
                logger.info(
                    f"Query execution complete: "
                    f"{aggregated.successful_results}/{aggregated.total_results} succeeded"
                )

            return results

        except Exception as exc:
            logger.exception(f"Query execution failed: {exc}")
            raise

    def run_distributed_benchmark(
        self,
        dataset: list[dict[str, Any]],
        config: Optional[BenchmarkConfig] = None,
    ) -> DistributedJobResult:
        """Run benchmark on distributed system.

        Args:
            dataset: Items to benchmark
            config: Benchmark configuration

        Returns:
            Distributed job result
        """
        self._current_job_id = self.config.job_id

        logger.info(
            f"Starting distributed benchmark {self._current_job_id} "
            f"with {len(dataset)} items"
        )

        try:
            # Setup progress monitoring
            if self.config.enable_progress_monitoring:
                self._progress_monitor.start_job(
                    self._current_job_id,
                    total_items=len(dataset),
                )

            # Run benchmark
            benchmark_config = config or BenchmarkConfig(
                num_workers=self.config.num_workers
            )

            try:
                aggregated = self._benchmark_runner.run_benchmark(dataset, benchmark_config)

                # Record progress
                if self.config.enable_progress_monitoring:
                    self._progress_monitor.record_progress(
                        self._current_job_id,
                        completed=aggregated.total_items,
                    )

                # Aggregate results
                result = self._create_distributed_result(
                    aggregated=aggregated,
                    dataset_size=len(dataset),
                )

                logger.info(f"Benchmark complete: {result}")

                self._last_result = result
                return result

            except Exception as exc:
                logger.exception(f"Benchmark execution failed: {exc}")

                result = DistributedJobResult(
                    job_id=self._current_job_id,
                    total_items=len(dataset),
                    processed_items=0,
                    failed_items=len(dataset),
                    total_time_sec=0.0,
                    throughput=0.0,
                    success=False,
                    error_message=str(exc),
                )

                self._last_result = result
                return result

        finally:
            # Cleanup
            if self.config.enable_progress_monitoring:
                self._progress_monitor.clear_job(self._current_job_id)

    def get_current_progress(self) -> Optional[dict[str, Any]]:
        """Get current job progress.

        Returns:
            Progress dict or None if no job running
        """
        if not self._current_job_id:
            return None

        progress = self._progress_monitor.get_progress(self._current_job_id)
        if progress:
            return self._progress_monitor.get_job_status_summary(self._current_job_id)

        return None

    def get_worker_health(self) -> Optional[dict[str, Any]]:
        """Get worker health status.

        Returns:
            Worker health dict
        """
        if not self._current_job_id:
            return None

        worker_health = self._progress_monitor.get_worker_health(self._current_job_id)
        if worker_health:
            return {
                wid: {
                    "status": health.status,
                    "items_processed": health.items_processed,
                    "errors": health.errors,
                    "latency_ms": health.avg_latency_ms,
                }
                for wid, health in worker_health.items()
            }

        return None

    def get_fault_tolerance_stats(self) -> Optional[dict[str, Any]]:
        """Get fault tolerance statistics.

        Returns:
            Fault tolerance stats
        """
        if not self._fault_tolerance:
            return None

        return self._fault_tolerance.get_stats()

    def get_last_result(self) -> Optional[DistributedJobResult]:
        """Get result from last job.

        Returns:
            Last distributed job result
        """
        return self._last_result

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics from parallel executor.

        Returns:
            Execution statistics
        """
        stats = self._parallel_executor.get_execution_stats()

        return {
            "total_queries": stats.total_queries,
            "successful_queries": stats.successful_queries,
            "failed_queries": stats.failed_queries,
            "total_time_ms": stats.total_time_ms,
            "avg_query_time_ms": stats.avg_query_time_ms,
            "throughput_qps": stats.throughput_qps,
            "percentiles": {
                "min_ms": stats.min_query_time_ms,
                "max_ms": stats.max_query_time_ms,
            },
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._parallel_executor.reset_stats()
        self._aggregator.clear_buffer()

        if self._fault_tolerance:
            self._fault_tolerance.clear_all()

        self._current_job_id = None
        self._last_result = None

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status.

        Returns:
            System status dict
        """
        return {
            "current_job": self._current_job_id,
            "last_result": (
                {
                    "success": self._last_result.success,
                    "processed_items": self._last_result.processed_items,
                    "total_time_sec": self._last_result.total_time_sec,
                    "throughput": self._last_result.throughput,
                }
                if self._last_result
                else None
            ),
            "num_workers": self.config.num_workers,
            "max_query_workers": self.config.max_workers_query,
            "fault_tolerance_enabled": self.config.enable_fault_tolerance,
            "progress_monitoring_enabled": self.config.enable_progress_monitoring,
        }

    # Private helper methods

    def _create_distributed_result(
        self,
        aggregated: Any,
        dataset_size: int,
    ) -> DistributedJobResult:
        """Create distributed job result from benchmark result."""
        return DistributedJobResult(
            job_id=self._current_job_id,
            total_items=dataset_size,
            processed_items=aggregated.total_items,
            failed_items=aggregated.failed_workers,
            total_time_sec=aggregated.total_time_sec,
            throughput=aggregated.throughput_items_per_sec,
            success=aggregated.failed_workers == 0,
        )

    def get_aggregator(self) -> ResultAggregator:
        """Get result aggregator.

        Returns:
            ResultAggregator instance
        """
        return self._aggregator

    def get_progress_monitor(self) -> ProgressMonitor:
        """Get progress monitor.

        Returns:
            ProgressMonitor instance
        """
        return self._progress_monitor

    def get_fault_tolerance_manager(self) -> Optional[FaultToleranceManager]:
        """Get fault tolerance manager.

        Returns:
            FaultToleranceManager instance or None
        """
        return self._fault_tolerance
