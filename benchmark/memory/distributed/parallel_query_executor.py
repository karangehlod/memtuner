"""Parallel query execution across multiple workers."""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics from parallel execution."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    min_query_time_ms: float = 0.0
    max_query_time_ms: float = 0.0
    throughput_qps: float = 0.0
    worker_utilization: dict[str, float] = field(default_factory=dict)


class ParallelQueryExecutor:
    """Execute queries in parallel across worker pool."""

    def __init__(
        self,
        memory_system: Any,
        max_workers: int = 4,
        timeout_sec: float = 30.0,
    ):
        """Initialize parallel query executor.

        Args:
            memory_system: AdvancedMemorySystem instance
            max_workers: Maximum worker threads
            timeout_sec: Timeout per query
        """
        self.memory_system = memory_system
        self.max_workers = max_workers
        self.timeout_sec = timeout_sec

        # Execution tracking
        self._query_times: dict[str, float] = {}
        self._query_errors: dict[str, Exception] = {}
        self._query_results: dict[str, dict[str, Any]] = {}
        self._worker_query_count: dict[int, int] = defaultdict(int)

    def execute_queries(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute queries in parallel.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of query results maintaining input order

        Raises:
            ValueError: If queries empty
        """
        if not queries:
            raise ValueError("Cannot execute empty query list")

        start_time = time.time()
        results = [None] * len(queries)
        query_to_idx = {query: idx for idx, query in enumerate(queries)}

        # Execute queries in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            futures = {}
            for idx, query in enumerate(queries):
                future = executor.submit(
                    self._execute_single_query,
                    query,
                    top_k,
                    idx,
                )
                futures[future] = (query, idx)

            # Collect results as they complete
            for future in as_completed(futures, timeout=self.timeout_sec * len(queries)):
                query, idx = futures[future]

                try:
                    result = future.result(timeout=self.timeout_sec)
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Query execution failed: {query}: {e}")
                    self._query_errors[query] = e
                    results[idx] = {
                        "query": query,
                        "error": str(e),
                        "results": [],
                    }

        elapsed_ms = (time.time() - start_time) * 1000
        self._record_execution_stats(queries, elapsed_ms)

        return results

    def execute_batch_parallel(
        self,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute batch of pre-formed queries in parallel.

        Args:
            batch: List of query dicts (must have 'query' key)

        Returns:
            Results with execution metadata

        Raises:
            ValueError: If batch invalid
        """
        if not batch:
            raise ValueError("Cannot execute empty batch")

        # Extract query strings
        queries = [q.get("query", "") for q in batch]

        # Execute in parallel
        results = self.execute_queries(queries)

        # Merge with original batch metadata
        for i, (query_dict, result) in enumerate(zip(batch, results)):
            if result is not None:
                result["original_query"] = query_dict

        return results

    def get_execution_stats(self) -> ExecutionStats:
        """Get execution statistics.

        Returns:
            ExecutionStats object
        """
        total = len(self._query_times)
        successful = total - len(self._query_errors)

        if total == 0:
            return ExecutionStats()

        query_times = list(self._query_times.values())
        total_time = sum(query_times)

        return ExecutionStats(
            total_queries=total,
            successful_queries=successful,
            failed_queries=len(self._query_errors),
            total_time_ms=total_time,
            avg_query_time_ms=total_time / total if total > 0 else 0,
            min_query_time_ms=min(query_times) if query_times else 0,
            max_query_time_ms=max(query_times) if query_times else 0,
            throughput_qps=total / (total_time / 1000) if total_time > 0 else 0,
            worker_utilization=dict(self._worker_query_count),
        )

    def reset_stats(self) -> None:
        """Reset all execution statistics."""
        self._query_times.clear()
        self._query_errors.clear()
        self._query_results.clear()
        self._worker_query_count.clear()

    # Private helper methods

    def _execute_single_query(
        self,
        query: str,
        top_k: int,
        query_idx: int,
    ) -> dict[str, Any]:
        """Execute single query and track timing.

        Args:
            query: Query string
            top_k: Results to return
            query_idx: Query index for worker tracking

        Returns:
            Query result dict
        """
        import threading

        worker_id = threading.current_thread().ident
        self._worker_query_count[worker_id] += 1

        start_time = time.time()

        try:
            result = self.memory_system.query(query, top_k=top_k)

            elapsed_ms = (time.time() - start_time) * 1000
            self._query_times[query] = elapsed_ms

            return {
                "query": query,
                "results": result.results if hasattr(result, 'results') else result,
                "time_ms": elapsed_ms,
                "success": True,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._query_times[query] = elapsed_ms
            self._query_errors[query] = e

            return {
                "query": query,
                "error": str(e),
                "time_ms": elapsed_ms,
                "success": False,
                "results": [],
            }

    def _record_execution_stats(
        self,
        queries: list[str],
        total_ms: float,
    ) -> None:
        """Record execution statistics.

        Args:
            queries: Executed queries
            total_ms: Total execution time
        """
        # Stats are computed on demand via get_execution_stats()
        pass
