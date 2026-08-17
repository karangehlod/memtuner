"""Adapter for Entity Store - entity and relationship memory."""

import hashlib
import json
import time
from typing import Any

from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry
from benchmark.memory.adapters._sys_metrics import percentile as _pct, peak_rss_mb as _rss, cpu_percent_snapshot as _cpu


class EntityStoreAdapter(MemoryAdapter):
    """Benchmarks entity store for entity attributes and relationships.

    Entity memory stores information about entities (people, places, things)
    and relationships between them. Queries retrieve entities by properties
    and relationship traversal.
    """

    name = "entity_store"

    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: dict[str, list[tuple[str, str]]] = {}
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize entity store with configuration."""
        self.config = config
        self.entities = {}
        self.relationships = {}
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write entity to store.

        Stores entity with attributes and relationships.
        """
        try:
            start = time.time()

            memory_id = memory.get("id", "")
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)

            # Extract entity name from content (first noun-like word)
            entity_name = memory.get("user_id", f"entity_{memory_id}")

            # Store entity
            self.entities[memory_id] = {
                "id": memory_id,
                "name": entity_name,
                "content": content,
                "importance": importance,
                "attributes": self._extract_attributes(content),
                "timestamp": time.time(),
            }

            # Initialize relationship list for this entity
            if memory_id not in self.relationships:
                self.relationships[memory_id] = []

            # Extract relationships (mentioned entities)
            related_entities = self._extract_relationships(content)
            for related_id in related_entities:
                if related_id != memory_id:
                    self.relationships[memory_id].append((related_id, "mentioned"))

            elapsed = time.time() - start
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write entity memory: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query entity store for relevant entities.

        Retrieves entities ranked by attribute match and relationship relevance.
        """
        try:
            start = time.time()

            # Extract query attributes
            query_attrs = self._extract_attributes(query)

            # Score all entities by attribute match and importance
            scores: dict[str, tuple[float, str]] = {}

            for entity_id, entity in self.entities.items():
                entity_attrs = entity.get("attributes", [])
                importance = entity.get("importance", 0.5)

                # Attribute match score
                attr_overlap = len(set(query_attrs) & set(entity_attrs))
                attr_score = attr_overlap / max(len(query_attrs), 1)

                # Relationship score (how many relationships?)
                relations = self.relationships.get(entity_id, [])
                relation_score = min(len(relations) / 10.0, 1.0)

                # Combined score: 50% attributes, 30% importance, 20% relationships
                score = 0.5 * attr_score + 0.3 * importance + 0.2 * relation_score

                scores[entity_id] = (score, entity.get("name", entity_id))

            # Sort by score and return top-k
            sorted_results = sorted(
                scores.items(),
                key=lambda x: x[1][0],
                reverse=True
            )[:top_k]

            results = [
                {
                    "entity_id": entity_id,
                    "entity_name": name,
                    "score": score,
                    "entity_info": self.entities[entity_id],
                    "relations": len(self.relationships.get(entity_id, [])),
                }
                for entity_id, (score, name) in sorted_results
            ]

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self._per_query_results.append(results)

            return results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query entity memories: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for entity store."""
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
                [{"doc_id": r.get("entity_id", ""), "score": r.get("score", 0.0)} for r in qr]
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

            # Storage: entities + relationship graph
            storage_bytes = sum(
                len(json.dumps(e).encode())
                for e in self.entities.values()
            ) + sum(
                len(str(r).encode())
                for rels in self.relationships.values()
                for r in rels
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
            raise RuntimeError(f"Failed to compute entity store metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.entities.clear()
        self.relationships.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()

    @staticmethod
    def _extract_attributes(text: str) -> list[str]:
        """Extract entity attributes from text."""
        if not text:
            return []

        # Simple attribute extraction: adjectives and descriptive words
        words = text.lower().split()
        # Keep short descriptive words
        attributes = [
            w.strip(".,!?;:")
            for w in words
            if 3 <= len(w.strip(".,!?;:")) <= 15
        ]
        return list(set(attributes))[:5]  # Top 5 unique attributes

    @staticmethod
    def _extract_relationships(text: str) -> list[str]:
        """Extract related entity references from text."""
        if not text:
            return []

        # Very simple: find potential entity names (capitalized words)
        words = text.split()
        entities = [w for w in words if w and w[0].isupper()]
        return entities[:3]  # Top 3 relationships


# Auto-register on import
MemoryRegistry.register("entity_store", EntityStoreAdapter)
