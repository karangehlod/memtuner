"""End-to-end advanced memory system orchestrating all components."""

import logging
from dataclasses import dataclass
from typing import Any

from benchmark.memory.strategies.adaptive_strategy_selector import AdaptiveStrategySelector
from benchmark.memory.strategies.attention_weighter import AttentionWeighter
from benchmark.memory.strategies.memory_consolidation_engine import (
    MemoryConsolidationEngine,
)
from benchmark.memory.strategies.personalized_memory_ranker import (
    PersonalizedMemoryRanker,
    UserProfile,
)
from benchmark.memory.strategies.semantic_cluster_manager import SemanticClusterManager

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a query through the system."""
    query: str
    strategy_used: str
    query_type: str
    results: list[dict[str, Any]]
    total_time_ms: float
    memory_stats: dict[str, Any]


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    total_queries: int = 0
    total_memories: int = 0
    avg_query_latency_ms: float = 0.0
    consolidation_count: int = 0
    memory_overhead_mb: float = 0.0
    tier_distribution: dict[str, int] = None


class AdvancedMemorySystem:
    """Orchestrates all advanced memory components into integrated system."""

    def __init__(
        self,
        user_profile: UserProfile,
        retrieval_strategies: list[str] | None = None,
    ):
        """Initialize the advanced memory system.

        Args:
            user_profile: User preference profile
            retrieval_strategies: Available retrieval strategy names
        """
        self.user_profile = user_profile

        # Initialize components
        self.cluster_manager = SemanticClusterManager(num_clusters=5)
        self.attention_weighter = AttentionWeighter()
        self.consolidation_engine = MemoryConsolidationEngine()
        self.strategy_selector = AdaptiveStrategySelector(
            strategies=retrieval_strategies
        )
        self.ranker = PersonalizedMemoryRanker(user_profile)

        # System state
        self._all_memories: dict[str, dict[str, Any]] = {}
        self._is_initialized = False
        self.metrics = SystemMetrics()

        # Query history for analytics
        self._query_history: list[QueryResult] = []

    def initialize(self, memories: list[dict[str, Any]]) -> None:
        """Initialize system with memory collection.

        Args:
            memories: List of memory dicts to load

        Raises:
            ValueError: If memories invalid
        """
        if not memories:
            raise ValueError("Cannot initialize with empty memory collection")

        # Store all memories
        for memory in memories:
            mem_id = memory.get("id", f"mem_{len(self._all_memories)}")
            self._all_memories[mem_id] = memory

        # Add to consolidation engine
        for mem_id, mem_content in self._all_memories.items():
            value_score = mem_content.get("score", 0.5)
            self.consolidation_engine.add_memory(
                mem_id,
                mem_content,
                tier="semantic",
                value_score=value_score,
            )

        # Fit clustering if we have valid memories
        try:
            memory_list = [
                {
                    "id": mem_id,
                    "content": mem_content.get("content", ""),
                    "embedding": mem_content.get("embedding"),
                }
                for mem_id, mem_content in self._all_memories.items()
            ]
            self.cluster_manager.fit(memory_list)
        except RuntimeError as e:
            # If clustering fails (e.g., no embeddings), still initialize successfully
            logger.warning(f"Clustering failed, continuing without clustering: {e}")

        self._is_initialized = True
        self.metrics.total_memories = len(self._all_memories)

        logger.info(f"Advanced memory system initialized with {len(memories)} memories")

    def query(
        self,
        query: str,
        top_k: int = 10,
    ) -> QueryResult:
        """Execute query through full pipeline.

        Args:
            query: User query string
            top_k: Number of results to return

        Returns:
            QueryResult with ranked results

        Raises:
            ValueError: If system not initialized
        """
        import time

        if not self._is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        start_time = time.time()

        # Step 1: Classify query
        query_type = self.strategy_selector.classify_query(query)

        # Step 2: Select retrieval strategy
        strategy = self.strategy_selector.select_strategy(query)

        # Step 3: Retrieve candidates (simulated - would call actual retrieval)
        candidates = self._retrieve_candidates(query, strategy)

        # Step 4: Apply attention weighting
        self.attention_weighter.compute_scores(candidates, query)

        # Step 5: Apply personalized ranking
        ranked_results = self.ranker.rank(candidates, query)

        # Step 6: Add cluster context
        for result in ranked_results:
            mem_id = result.get("id")
            try:
                cluster_id = self.cluster_manager.predict(result)
                result["cluster_id"] = cluster_id
            except (RuntimeError, ValueError):
                # If clustering not available, use default
                result["cluster_id"] = 0

        # Step 7: Return top-k
        final_results = ranked_results[:top_k]

        # Record access
        for result in final_results:
            mem_id = result.get("id")
            if mem_id in self._all_memories:
                self.consolidation_engine.record_access(mem_id)

        elapsed_ms = (time.time() - start_time) * 1000

        # Build result object
        query_result = QueryResult(
            query=query,
            strategy_used=strategy,
            query_type=query_type,
            results=final_results,
            total_time_ms=elapsed_ms,
            memory_stats=self.consolidation_engine.get_all_tier_statistics(),
        )

        # Track metrics
        self._query_history.append(query_result)
        self.metrics.total_queries += 1

        if len(self._query_history) > 1:
            avg_latency = sum(q.total_time_ms for q in self._query_history) / len(
                self._query_history
            )
            self.metrics.avg_query_latency_ms = avg_latency

        return query_result

    def add_memory(self, memory: dict[str, Any]) -> bool:
        """Add new memory to system.

        Args:
            memory: Memory dict to add

        Returns:
            True if added successfully
        """
        mem_id = memory.get("id", f"mem_{len(self._all_memories)}")

        # Add to store
        self._all_memories[mem_id] = memory

        # Add to consolidation engine
        value_score = memory.get("score", 0.5)
        added = self.consolidation_engine.add_memory(
            mem_id,
            memory,
            tier="working",
            value_score=value_score,
        )

        if added:
            self.metrics.total_memories += 1

        return added

    def provide_feedback(self, memory_id: str, helpful: bool) -> None:
        """Provide feedback on memory usefulness.

        Args:
            memory_id: Memory to rate
            helpful: Whether user found it helpful
        """
        if memory_id not in self._all_memories:
            logger.warning(f"Feedback for unknown memory: {memory_id}")
            return

        # Record with ranker
        latest_query = self._query_history[-1].query if self._query_history else "unknown"
        self.ranker.add_feedback(latest_query, memory_id, helpful)

        # Update consolidation engine value
        if helpful:
            self.consolidation_engine.update_value_score(memory_id, 0.9)
        else:
            self.consolidation_engine.update_value_score(memory_id, 0.2)

        # Learn in strategy selector
        if self._query_history:
            strategy = self._query_history[-1].strategy_used
            self.strategy_selector.learn_from_feedback(latest_query, strategy, helpful)

    def consolidate(self) -> None:
        """Execute memory consolidation pass.

        Promotes high-value, demotes low-value, applies decay.
        """
        self.consolidation_engine.consolidate()
        self.metrics.consolidation_count += 1

    def get_system_metrics(self) -> SystemMetrics:
        """Get system performance metrics.

        Returns:
            SystemMetrics object
        """
        tier_stats = self.consolidation_engine.get_all_tier_statistics()

        tier_distribution = {
            tier: stats["size"]
            for tier, stats in tier_stats.items()
        }

        self.metrics.tier_distribution = tier_distribution

        return self.metrics

    def get_memory_statistics(self) -> dict[str, Any]:
        """Get detailed memory statistics.

        Returns:
            Dict with per-tier statistics
        """
        return {
            "tiers": self.consolidation_engine.get_all_tier_statistics(),
            "clusters": [
                {
                    "id": info.cluster_id,
                    "size": info.size,
                    "summary": info.summary,
                }
                for info in self.cluster_manager.get_all_clusters()
            ],
            "total_memories": self.metrics.total_memories,
            "total_queries": self.metrics.total_queries,
        }

    def get_query_history(self, limit: int = 10) -> list[QueryResult]:
        """Get recent query history.

        Args:
            limit: Number of recent queries to return

        Returns:
            List of QueryResult objects
        """
        return self._query_history[-limit:]

    # Private helper methods

    def _retrieve_candidates(
        self,
        query: str,
        strategy: str,
    ) -> list[dict[str, Any]]:
        """Retrieve candidate memories for query.

        In production, this would call actual retrieval strategy.
        For now, returns all memories ranked by query similarity.

        Args:
            query: Query string
            strategy: Strategy name

        Returns:
            List of candidate memories
        """
        # Simple scoring: word overlap with query
        query_words = set(query.lower().split())

        scored_memories = []
        for mem_id, mem_content in self._all_memories.items():
            content = str(mem_content.get("content", "")).lower()
            content_words = set(content.split())

            # Jaccard similarity
            if len(query_words | content_words) > 0:
                overlap = len(query_words & content_words)
                score = overlap / len(query_words | content_words)
            else:
                score = 0.0

            scored_memories.append({
                "id": mem_id,
                "content": mem_content.get("content", ""),
                "score": score,
                "type": mem_content.get("type", "unknown"),
                **mem_content,
            })

        # Sort by score descending
        scored_memories.sort(key=lambda m: m["score"], reverse=True)

        return scored_memories
