"""Adapter for Context Buffer - active context management."""

import json
import time
from typing import Any

from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry
from benchmark.memory.adapters._sys_metrics import percentile as _pct, peak_rss_mb as _rss, cpu_percent_snapshot as _cpu


class ContextBufferAdapter(MemoryAdapter):
    """Benchmarks context buffer for active context management.

    Context buffer maintains relevant context for current task.
    Dynamic sizing based on relevance and importance signals.
    """

    name = "context_buffer"

    def __init__(self):
        self.context: dict[str, dict[str, Any]] = {}
        self.task_context: dict[str, list[str]] = {}
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0
        self.max_context_size = 50
        self.relevance_threshold = 0.5

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize context buffer with configuration."""
        self.config = config
        self.max_context_size = config.get("max_context_size", 50)
        self.relevance_threshold = config.get("relevance_threshold", 0.5)
        self.context = {}
        self.task_context = {}
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write to context buffer (relevance-based).

        Only keeps items above relevance threshold. Evicts low-relevance items.
        """
        try:
            start = time.time()

            memory_id = memory.get("id", "")
            task_id = memory.get("task_id", "default")
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)

            # Compute relevance (importance for now, in production would be semantic)
            relevance = importance

            # Only add if above threshold
            if relevance < self.relevance_threshold:
                elapsed = time.time() - start
                self.write_times.append(elapsed)
                self.num_writes += 1
                return

            # Store in context
            self.context[memory_id] = {
                "id": memory_id,
                "task_id": task_id,
                "content": content,
                "importance": importance,
                "relevance": relevance,
                "timestamp": time.time(),
            }

            # Track per-task context
            if task_id not in self.task_context:
                self.task_context[task_id] = []

            if memory_id not in self.task_context[task_id]:
                self.task_context[task_id].append(memory_id)

            # Evict low-relevance items if context too large
            if len(self.context) > self.max_context_size:
                self._evict_low_relevance()

            elapsed = time.time() - start
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write to context buffer: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query context buffer for active context.

        Returns top relevant items currently in active context.
        """
        try:
            start = time.time()

            if not self.context:
                return []

            # Score items in context by relevance and recency
            scores: dict[str, tuple[float, str]] = {}
            current_time = time.time()

            for memory_id, item in self.context.items():
                relevance = item.get("relevance", 0.5)
                importance = item.get("importance", 0.5)
                timestamp = item.get("timestamp", time.time())

                # Recency decay (more recent = higher score)
                age = current_time - timestamp
                recency_score = 1.0 / (1.0 + age / 60.0)  # 60s half-life

                # Combined score: 50% relevance, 30% importance, 20% recency
                score = (
                    0.5 * relevance +
                    0.3 * importance +
                    0.2 * recency_score
                )

                scores[memory_id] = (score, item.get("content", ""))

            # Sort by score and return top-k
            sorted_results = sorted(
                scores.items(),
                key=lambda x: x[1][0],
                reverse=True
            )[:top_k]

            results = [
                {
                    "memory_id": memory_id,
                    "score": score,
                    "content": content,
                    "active_context": True,
                }
                for memory_id, (score, content) in sorted_results
            ]

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(results)

            return results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query context buffer: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for context buffer."""
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

            # Efficiency: fast query of filtered context
            avg_write_latency = sum(self.write_times) / len(self.write_times) if self.write_times else 0.0
            avg_query_latency = sum(self.query_times) / len(self.query_times) if self.query_times else 0.0

            # Storage: only active context items
            storage_bytes = sum(
                len(json.dumps(item).encode())
                for item in self.context.values()
            )

            # Reliability
            success_rate = 1.0 - (self.num_failures / max(1, self.num_writes + self.num_queries))
            context_size = len(self.context)

            return MemoryMetrics(
                recall_at_1=min(1.0, recall_at_1),
                recall_at_5=min(1.0, recall_at_5),
                recall_at_10=min(1.0, recall_at_10),
                recall_at_100=min(1.0, recall_at_100),
                mrr=min(1.0, mrr),
                ndcg=min(1.0, ndcg),
                write_latency_ms=avg_write_latency * 1000,
                query_latency_ms=avg_query_latency * 1000,
                storage_bytes=float(storage_bytes),
                success_rate=success_rate,
                error_count=self.num_failures,
                dataset_name=self.config.get("dataset_name", "unknown"),
                num_memories=context_size,
                num_queries=self.num_queries,
                elapsed_seconds=elapsed_seconds,
                query_latency_p50_ms=_pct(self.query_times, 50) * 1000,
                query_latency_p95_ms=_pct(self.query_times, 95) * 1000,
                index_build_ms=sum(self.write_times) * 1000,
                peak_rss_mb=_rss(),
                cpu_percent=_cpu(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute context buffer metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.context.clear()
        self.task_context.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()

    def _evict_low_relevance(self) -> None:
        """Evict low-relevance items from context to maintain size limit."""
        if len(self.context) <= self.max_context_size:
            return

        # Sort by relevance and remove lowest
        sorted_items = sorted(
            self.context.items(),
            key=lambda x: x[1].get("relevance", 0),
        )

        # Remove 10% of lowest relevance items
        num_to_remove = max(1, len(sorted_items) // 10)
        for memory_id, _ in sorted_items[:num_to_remove]:
            del self.context[memory_id]


# Auto-register on import
MemoryRegistry.register("context_buffer", ContextBufferAdapter)
