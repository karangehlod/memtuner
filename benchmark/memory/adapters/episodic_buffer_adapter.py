"""Adapter for Episodic Buffer - limited capacity working memory."""

import json
import time
from collections import deque
from typing import Any

from benchmark.memory.adapters._sys_metrics import cpu_percent_snapshot as _cpu
from benchmark.memory.adapters._sys_metrics import peak_rss_mb as _rss
from benchmark.memory.adapters._sys_metrics import percentile as _pct
from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry


class EpisodicBufferAdapter(MemoryAdapter):
    """Benchmarks episodic buffer for working memory management.

    Episodic buffer maintains limited recent events (FIFO/LRU eviction).
    Fast write/query at cost of limited capacity and retention.
    """

    name = "episodic_buffer"

    def __init__(self):
        self.buffer: deque = deque()
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.evictions = 0
        self.cache_hits = 0
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0
        self.capacity = 100

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize episodic buffer with capacity config."""
        self.config = config
        self.capacity = config.get("capacity", 100)
        self.buffer = deque(maxlen=self.capacity)
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.evictions = 0
        self.cache_hits = 0
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write memory to episodic buffer (FIFO eviction).

        When buffer is full, oldest item is evicted.
        """
        try:
            start = time.time()

            memory_id = memory.get("id", "")
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)
            timestamp = memory.get("timestamp", time.time())

            # Check if buffer is full before adding
            if len(self.buffer) >= self.capacity:
                self.evictions += 1

            # Add to buffer (deque with maxlen auto-evicts oldest)
            self.buffer.append({
                "id": memory_id,
                "content": content,
                "importance": importance,
                "timestamp": timestamp,
                "position": len(self.buffer),
            })

            elapsed = time.time() - start
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write to episodic buffer: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query episodic buffer for recent memories.

        Searches buffer by recency and importance, faster than full stores.
        """
        try:
            start = time.time()

            results = []
            buffer_list = list(self.buffer)

            if not buffer_list:
                return []

            # Score by recency (position in buffer) and importance
            for idx, memory in enumerate(buffer_list):
                # Recency: items at end are more recent
                recency_score = (idx + 1) / len(buffer_list)

                # Importance score
                importance = memory.get("importance", 0.5)

                # Combined: 60% recency, 40% importance (buffer is for recent access)
                score = 0.6 * recency_score + 0.4 * importance

                results.append({
                    "memory_id": memory["id"],
                    "score": score,
                    "content": memory["content"],
                    "recency": recency_score,
                })

            # Sort by score and take top-k
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

            # Check if we hit mostly buffer (cache hit rate)
            if len(sorted_results) > 0 and sorted_results[0]["score"] > 0.7:
                self.cache_hits += 1

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(sorted_results)

            return sorted_results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query episodic buffer: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for episodic buffer."""
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

            # Efficiency (buffer is fast!)
            avg_write_latency = sum(self.write_times) / len(self.write_times) if self.write_times else 0.0
            avg_query_latency = sum(self.query_times) / len(self.query_times) if self.query_times else 0.0

            # Storage: only capacity items
            storage_bytes = sum(
                len(json.dumps(m).encode())
                for m in self.buffer
            )

            # Reliability: track evictions
            self.evictions / max(1, self.num_writes)
            success_rate = 1.0 - (self.num_failures / max(1, self.num_writes + self.num_queries))

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
                num_memories=len(self.buffer),
                num_queries=self.num_queries,
                elapsed_seconds=elapsed_seconds,
                query_latency_p50_ms=_pct(self.query_times, 50) * 1000,
                query_latency_p95_ms=_pct(self.query_times, 95) * 1000,
                index_build_ms=sum(self.write_times) * 1000,
                peak_rss_mb=_rss(),
                cpu_percent=_cpu(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute episodic buffer metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.buffer.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()


# Auto-register on import
MemoryRegistry.register("episodic_buffer", EpisodicBufferAdapter)
