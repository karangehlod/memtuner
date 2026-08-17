"""Adapter for Episodic Store - temporal event memory."""

import hashlib
import json
import time
from typing import Any

from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry
from benchmark.memory.adapters._sys_metrics import percentile as _pct, peak_rss_mb as _rss, cpu_percent_snapshot as _cpu
from benchmark.gold.schema import GoldDataset, GoldQuery


class EpisodicStoreAdapter(MemoryAdapter):
    """Benchmarks episodic memory store for temporal event retrieval.

    Episodic memory stores timestamped events organized by occurrence time.
    Queries retrieve events by temporal range or temporal relevance.
    """

    name = "episodic_store"

    def __init__(self):
        self.memories: dict[str, dict[str, Any]] = {}
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize episodic store with configuration."""
        self.config = config
        self.memories = {}
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write memory event to episodic store.

        Stores event with timestamp for temporal ordering.
        """
        try:
            start = time.time()

            memory_id = memory.get("id", "")
            timestamp = memory.get("timestamp", time.time())
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)

            # Store in temporal structure
            self.memories[memory_id] = {
                "id": memory_id,
                "timestamp": timestamp,
                "content": content,
                "importance": importance,
                "day": memory.get("day", 0),
            }

            elapsed = time.time() - start
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write episodic memory: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query episodic store for relevant memories.

        Retrieves memories ranked by temporal recency and importance.
        """
        try:
            start = time.time()

            # Score memories by temporal recency and importance
            scores: dict[str, float] = {}

            current_time = time.time()
            max_age = 365 * 24 * 3600  # 1 year in seconds

            for memory_id, memory in self.memories.items():
                timestamp = memory["timestamp"]
                age = current_time - timestamp

                if age < 0:
                    age = 0
                if age > max_age:
                    age = max_age

                # Recency score (0-1)
                recency = 1.0 - (age / max_age)

                # Importance score (0-1)
                importance = memory["importance"]

                # Combined score: 70% recency, 30% importance
                scores[memory_id] = 0.7 * recency + 0.3 * importance

            # Sort by score and return top-k
            sorted_results = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            results = [
                {
                    "memory_id": memory_id,
                    "score": score,
                    "content": self.memories[memory_id]["content"],
                }
                for memory_id, score in sorted_results
            ]

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(results)

            return results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query episodic memories: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for episodic store."""
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

            # Efficiency
            avg_write_latency = sum(self.write_times) / len(self.write_times) if self.write_times else 0.0
            avg_query_latency = sum(self.query_times) / len(self.query_times) if self.query_times else 0.0
            storage_bytes = sum(
                len(json.dumps(m).encode())
                for m in self.memories.values()
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
                write_latency_ms=avg_write_latency * 1000,
                query_latency_ms=avg_query_latency * 1000,
                storage_bytes=float(storage_bytes),
                success_rate=success_rate,
                error_count=self.num_failures,
                dataset_name=self.config.get("dataset_name", "unknown"),
                num_memories=self.num_writes,
                num_queries=self.num_queries,
                elapsed_seconds=elapsed_seconds,
                query_latency_p50_ms=_pct(self.query_times, 50) * 1000,
                query_latency_p95_ms=_pct(self.query_times, 95) * 1000,
                index_build_ms=sum(self.write_times) * 1000,
                peak_rss_mb=_rss(),
                cpu_percent=_cpu(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute episodic store metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.memories.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()


# Auto-register on import
MemoryRegistry.register("episodic_store", EpisodicStoreAdapter)
