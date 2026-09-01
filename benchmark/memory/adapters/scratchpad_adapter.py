"""Adapter for Scratchpad - ultra-fast working memory."""

import json
import time
from typing import Any

from benchmark.memory.adapters._sys_metrics import cpu_percent_snapshot as _cpu
from benchmark.memory.adapters._sys_metrics import peak_rss_mb as _rss
from benchmark.memory.adapters._sys_metrics import percentile as _pct
from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry


class ScratchpadAdapter(MemoryAdapter):
    """Benchmarks scratchpad for ultra-fast working memory.

    Scratchpad is volatile working memory for temporary computations.
    Optimized for speed (microseconds), not persistence.
    """

    name = "scratchpad"

    def __init__(self):
        self.scratch: dict[str, Any] = {}
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0
        self.capacity = 20  # Very small for speed

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize scratchpad with configuration."""
        self.config = config
        self.capacity = config.get("capacity", 20)
        self.scratch = {}
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write to scratchpad (ultra-fast, volatile).

        Simple dict storage optimized for speed over persistence.
        """
        try:
            start = time.perf_counter()

            memory_id = memory.get("id", "")
            content = memory.get("content", "")

            # Just store it directly (no fancy indexing)
            self.scratch[memory_id] = {
                "content": content,
                "importance": memory.get("importance", 0.5),
            }

            # If over capacity, remove oldest (by id order)
            if len(self.scratch) > self.capacity:
                # Remove first item
                first_key = next(iter(self.scratch))
                del self.scratch[first_key]

            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write to scratchpad: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query scratchpad (ultra-fast lookup).

        Simple linear scan optimized for speed with tiny dataset.
        """
        try:
            start = time.perf_counter()

            if not self.scratch:
                return []

            # Simple importance-based ranking (O(n) with n=20)
            scores: list[tuple[str, float, str]] = []

            for memory_id, item in self.scratch.items():
                # Score by importance (only factor for scratchpad)
                importance = item.get("importance", 0.5)
                scores.append((memory_id, importance, item["content"]))

            # Sort by importance
            scores.sort(key=lambda x: x[1], reverse=True)

            # Return top-k
            results = [
                {
                    "memory_id": memory_id,
                    "score": score,
                    "content": content,
                }
                for memory_id, score, content in scores[:top_k]
            ]

            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(results)

            return results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query scratchpad: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for scratchpad."""
        try:
            elapsed_seconds = time.time() - self.start_time

            # Per-query MRR, Recall@K, Precision@K — computed correctly: one
            # reciprocal rank per query at the rank of the first relevant result
            # within that query's ranked list, then macro-averaged across queries.
            # Score-based relevance (threshold=0.5) is used since gold labels are
            # not available at this layer; Pipeline 1 (ScenarioRunner) provides
            # gold-grounded metrics for the leaderboard.
            from benchmark.retrieval.metrics_utils import compute_metric_summary as _cms
            _all_q = [
                [{"doc_id": r.get("memory_id", ""), "score": r.get("score", 0.0)} for r in qr]
                for qr in self._per_query_results
            ]
            _ms = _cms(_all_q, use_score_estimation=True)
            recall_at_1   = _ms.get("recall_at_1", 0.0)
            recall_at_5   = _ms.get("recall_at_5", 0.0)
            recall_at_10  = _ms["recall_at_10"]
            recall_at_100 = _ms["recall_at_100"]
            mrr           = _ms["mrr"]
            ndcg          = _ms["ndcg"]

            # Efficiency: scratchpad is FAST!
            avg_write_latency = sum(self.write_times) / len(self.write_times) if self.write_times else 0.0
            avg_query_latency = sum(self.query_times) / len(self.query_times) if self.query_times else 0.0

            # Storage: minimal (tiny capacity)
            storage_bytes = sum(
                len(json.dumps(item).encode())
                for item in self.scratch.values()
            )

            # Reliability
            success_rate = 1.0 - (self.num_failures / max(1, self.num_writes + self.num_queries))

            return MemoryMetrics(
                recall_at_1=min(1.0, recall_at_1),
                recall_at_5=min(1.0, recall_at_5),
                recall_at_10=min(1.0, recall_at_10),
                recall_at_100=min(1.0, recall_at_100),
                mrr=min(1.0, mrr),
                ndcg=min(1.0, ndcg),
                write_latency_ms=avg_write_latency,
                query_latency_ms=avg_query_latency,
                storage_bytes=float(storage_bytes),
                success_rate=success_rate,
                error_count=self.num_failures,
                dataset_name=self.config.get("dataset_name", "unknown"),
                num_memories=len(self.scratch),
                num_queries=self.num_queries,
                elapsed_seconds=elapsed_seconds,
                query_latency_p50_ms=_pct(self.query_times, 50) * 1000,
                query_latency_p95_ms=_pct(self.query_times, 95) * 1000,
                index_build_ms=sum(self.write_times) * 1000,
                peak_rss_mb=_rss(),
                cpu_percent=_cpu(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute scratchpad metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.scratch.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()


# Auto-register on import
MemoryRegistry.register("scratchpad", ScratchpadAdapter)
