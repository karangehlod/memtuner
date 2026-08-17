"""Attention-based weighting system for intelligent memory scoring."""

import numpy as np
from datetime import datetime, timedelta
from typing import Any, Optional, Literal
from dataclasses import dataclass
from collections import defaultdict


QueryType = Literal["factual", "semantic", "exact", "complex"]


@dataclass
class AttentionWeights:
    """Configuration for weighting factors."""
    relevance_weight: float = 0.4
    recency_weight: float = 0.3
    frequency_weight: float = 0.2
    coherence_weight: float = 0.1

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = (
            self.relevance_weight
            + self.recency_weight
            + self.frequency_weight
            + self.coherence_weight
        )
        if not (0.99 <= total <= 1.01):  # Allow small float errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class AttentionWeighter:
    """Score memories using multiple relevance factors."""

    def __init__(
        self,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.2,
        coherence_weight: float = 0.1,
    ):
        """Initialize attention weighter with factor weights.

        Args:
            relevance_weight: Weight for semantic relevance (0-1)
            recency_weight: Weight for temporal recency (0-1)
            frequency_weight: Weight for access frequency (0-1)
            coherence_weight: Weight for user preference alignment (0-1)
                All weights must sum to 1.0
        """
        self.weights = AttentionWeights(
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            frequency_weight=frequency_weight,
            coherence_weight=coherence_weight,
        )

        # Query-adaptive weights (overrides defaults when set)
        self._adaptive_weights: dict[QueryType, AttentionWeights] = {
            "factual": AttentionWeights(0.5, 0.2, 0.2, 0.1),
            "semantic": AttentionWeights(0.4, 0.3, 0.2, 0.1),
            "exact": AttentionWeights(0.3, 0.2, 0.3, 0.2),
            "complex": AttentionWeights(0.4, 0.25, 0.2, 0.15),
        }

        # Access tracking for frequency
        self._access_count: dict[str, int] = defaultdict(int)

        # Preference tracking for coherence
        self._preference_scores: dict[str, float] = defaultdict(lambda: 0.5)

        # Reference time for recency computation
        self._reference_time = datetime.now()

    def set_query_adaptive_weights(self, query_type: QueryType) -> None:
        """Switch to query-type-specific weights.

        Args:
            query_type: Type of query ('factual', 'semantic', 'exact', 'complex')

        Raises:
            ValueError: If query_type is unknown
        """
        if query_type not in self._adaptive_weights:
            raise ValueError(f"Unknown query type: {query_type}")

        self.weights = self._adaptive_weights[query_type]

    def compute_scores(
        self,
        memories: list[dict[str, Any]],
        query: str,
    ) -> dict[str, float]:
        """Compute attention scores for memories.

        Args:
            memories: List of memory dicts
            query: Query string (for relevance scoring)

        Returns:
            Dict mapping memory_id → score (0-1)
        """
        scores = {}

        for memory in memories:
            memory_id = memory.get("id", "unknown")

            # Compute individual factors
            relevance = self._compute_relevance(memory, query)
            recency = self._compute_recency(memory)
            frequency = self._compute_frequency(memory_id)
            coherence = self._compute_coherence(memory_id)

            # Weighted sum
            total_score = (
                self.weights.relevance_weight * relevance
                + self.weights.recency_weight * recency
                + self.weights.frequency_weight * frequency
                + self.weights.coherence_weight * coherence
            )

            # Clamp to [0, 1]
            scores[memory_id] = min(1.0, max(0.0, total_score))

            # Track access
            self._access_count[memory_id] += 1

        return scores

    def get_top_k(
        self,
        memories: list[dict[str, Any]],
        query: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top-k memories by attention score.

        Args:
            memories: List of memory dicts
            query: Query string
            k: Number of top memories to return

        Returns:
            Sorted list of top k memories (highest score first)

        Raises:
            ValueError: If k is invalid
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        # Compute scores
        scores = self.compute_scores(memories, query)

        # Sort by score (descending)
        sorted_memories = sorted(
            memories,
            key=lambda m: scores.get(m.get("id", ""), 0.0),
            reverse=True,
        )

        return sorted_memories[:k]

    def record_user_preference(self, memory_id: str, score: float) -> None:
        """Record user preference for a memory.

        Args:
            memory_id: ID of memory
            score: User preference score (0-1)

        Raises:
            ValueError: If score not in [0, 1]
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score must be in [0,1], got {score}")

        self._preference_scores[memory_id] = score

    def update_reference_time(self, reference_time: datetime | None = None) -> None:
        """Update reference time for recency decay.

        Args:
            reference_time: New reference time (default: now)
        """
        self._reference_time = reference_time or datetime.now()

    def get_weight_summary(self) -> dict[str, float]:
        """Get current weight configuration.

        Returns:
            Dict with all weight values
        """
        return {
            "relevance": self.weights.relevance_weight,
            "recency": self.weights.recency_weight,
            "frequency": self.weights.frequency_weight,
            "coherence": self.weights.coherence_weight,
        }

    def reset_tracking(self) -> None:
        """Reset access counts and preferences."""
        self._access_count.clear()
        self._preference_scores.clear()

    # Private helper methods

    def _compute_relevance(self, memory: dict[str, Any], query: str) -> float:
        """Compute semantic relevance score (0-1)."""
        # Extract content for similarity
        content = str(memory.get("content", "")).lower()
        query_terms = query.lower().split()

        if not content or not query_terms:
            return 0.5  # Neutral score

        # Simple term overlap scoring
        matching_terms = sum(1 for term in query_terms if term in content)
        max_matches = len(query_terms)

        if max_matches == 0:
            return 0.5

        # Normalize to [0, 1]
        return min(1.0, matching_terms / max(1, max_matches * 0.5))

    def _compute_recency(self, memory: dict[str, Any]) -> float:
        """Compute recency score with exponential decay (0-1)."""
        # Get memory timestamp
        if "timestamp" not in memory:
            return 0.5  # Neutral for undated memories

        try:
            if isinstance(memory["timestamp"], str):
                mem_time = datetime.fromisoformat(memory["timestamp"])
            else:
                mem_time = memory["timestamp"]
        except (ValueError, TypeError):
            return 0.5

        # Compute age in days
        age_days = (self._reference_time - mem_time).total_seconds() / 86400

        # Exponential decay: decay_rate determines half-life
        decay_rate = 0.1  # Half-life ~ 7 days
        recency_score = np.exp(-decay_rate * age_days)

        return min(1.0, max(0.0, recency_score))

    def _compute_frequency(self, memory_id: str) -> float:
        """Compute access frequency score (0-1)."""
        # Normalize to [0, 1] with saturation at 10 accesses
        count = self._access_count.get(memory_id, 0)
        frequency_score = min(1.0, count / 10.0)

        return frequency_score

    def _compute_coherence(self, memory_id: str) -> float:
        """Compute coherence with user preferences (0-1)."""
        # Return stored preference score
        return self._preference_scores.get(memory_id, 0.5)
