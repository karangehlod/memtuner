"""Adapter for Semantic Store - concept and topic memory."""

import json
import time
from typing import Any

from benchmark.memory.adapters._sys_metrics import cpu_percent_snapshot as _cpu
from benchmark.memory.adapters._sys_metrics import peak_rss_mb as _rss
from benchmark.memory.adapters._sys_metrics import percentile as _pct
from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry


class SemanticStoreAdapter(MemoryAdapter):
    """Benchmarks semantic memory store for concept and topic retrieval.

    Semantic memory stores facts and knowledge organized by topic/concept.
    Queries retrieve facts by semantic similarity and topic relevance.
    """

    name = "semantic_store"

    def __init__(self):
        self.topics: dict[str, list[dict[str, Any]]] = {}
        self.concept_index: dict[str, list[str]] = {}
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize semantic store with configuration."""
        self.config = config
        self.topics = {}
        self.concept_index = {}
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write memory to semantic store.

        Stores fact organized by topic and concepts.
        """
        try:
            start = time.time()

            memory_id = memory.get("id", "")
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)
            topic = memory.get("task_id", "general")

            # Initialize topic if needed
            if topic not in self.topics:
                self.topics[topic] = []

            # Store memory in topic
            self.topics[topic].append({
                "id": memory_id,
                "content": content,
                "importance": importance,
                "timestamp": time.time(),
            })

            # Index key concepts from content
            concepts = self._extract_concepts(content)
            for concept in concepts:
                if concept not in self.concept_index:
                    self.concept_index[concept] = []
                self.concept_index[concept].append(memory_id)

            elapsed = time.time() - start
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write semantic memory: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query semantic store for relevant facts.

        Retrieves facts ranked by semantic relevance and importance.
        """
        try:
            start = time.time()

            # Extract query concepts
            query_concepts = self._extract_concepts(query)

            # Score all memories by concept overlap and importance
            scores: dict[str, tuple[float, str, float]] = {}

            for _topic, memories in self.topics.items():
                for memory in memories:
                    memory_id = memory["id"]
                    memory_content = memory["content"]
                    importance = memory["importance"]

                    # Get memory concepts
                    memory_concepts = self._extract_concepts(memory_content)

                    # Concept overlap score
                    overlap = len(set(query_concepts) & set(memory_concepts))
                    concept_score = overlap / max(len(query_concepts), 1)

                    # Combined score: 70% concept, 30% importance
                    score = 0.7 * concept_score + 0.3 * importance

                    scores[memory_id] = (score, memory_content, concept_score)

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
                    "semantic_relevance": sem_rel,
                }
                for memory_id, (score, content, sem_rel) in sorted_results
            ]

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(results)

            return results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query semantic memories: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for semantic store."""
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

            # Storage: topics + index
            storage_bytes = sum(
                len(json.dumps(m).encode())
                for memories in self.topics.values()
                for m in memories
            ) + sum(
                len(concept.encode()) * len(ids)
                for concept, ids in self.concept_index.items()
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
            raise RuntimeError(f"Failed to compute semantic store metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.topics.clear()
        self.concept_index.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()

    @staticmethod
    def _extract_concepts(text: str) -> list[str]:
        """Extract key concepts from text (simplified).

        In production, use NLP for proper concept extraction.
        """
        # Simple word-based concept extraction
        if not text:
            return []

        # Split and filter
        words = text.lower().split()
        # Keep words > 4 chars as concepts
        concepts = [w.strip(".,!?;:") for w in words if len(w) > 4]
        return list(set(concepts))[:10]  # Top 10 unique concepts


# Auto-register on import
MemoryRegistry.register("semantic_store", SemanticStoreAdapter)
