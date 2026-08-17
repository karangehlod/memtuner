"""Semantic store — in-memory long-term semantic memory.

Stores factual/semantic knowledge memories with text-similarity retrieval and decay.
This is the in-memory reference implementation (no vector DB dependency).

Only accepts memories with type SEMANTIC. Other memory types (EPISODIC,
PREFERENCE, ENTITY) are silently skipped — they belong in their
respective stores.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from benchmark.memory.long_term.base_store import BaseLongTermStore
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery

# Memory types accepted by this store
_ACCEPTED_TYPES = frozenset({MemoryType.SEMANTIC})


class SemanticStore(BaseLongTermStore):
    """In-memory long-term store for semantic/factual knowledge.

    Optimized for storing facts, definitions, and general knowledge.
    Uses a slower default decay rate than episodic stores.

    Only stores memories with type SEMANTIC.
    Scoring: text_similarity × decay_factor
    (importance is NOT multiplied in — semantic facts are equally important)
    """

    def __init__(
        self,
        decay_type: str = "exponential",
        decay_lambda: float = 0.03,
        pruning_strategy: str = "score_threshold",
        pruning_threshold: float = 0.25,
        module_name: str | None = None,
        retrieval_strategy: object | None = None,
        allow_strategy_fallback: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialize the semantic store.

        Args:
            decay_type: Type of decay function.
            decay_lambda: Decay rate parameter (slower than episodic by default).
            pruning_strategy: Pruning strategy name (forwarded to base).
            pruning_threshold: Score threshold for pruning.
            module_name: Logical module name (defaults to 'semantic_store').
            retrieval_strategy: Optional retrieval strategy instance.
            allow_strategy_fallback: Whether to allow fallback to default scoring.
            **kwargs: Forwarded to BaseLongTermStore (archival_floor, etc.).
        """
        super().__init__(
            module_name=module_name or "semantic_store",
            decay_type=decay_type,
            decay_lambda=decay_lambda,
            pruning_threshold=pruning_threshold,
            retrieval_strategy=retrieval_strategy,
            allow_strategy_fallback=allow_strategy_fallback,
            **kwargs,
        )

    def write(self, event: MemoryEvent) -> None:
        """Write only if memory type is accepted by this store."""
        if event.type in _ACCEPTED_TYPES:
            super().write(event)

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Write only if memory type is accepted by this store."""
        if event.type in _ACCEPTED_TYPES:
            super().write_on_day(event, day)

    def _compute_relevance_score(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        decay_factor: float,
    ) -> float:
        """Compute semantic relevance: text_similarity × decay.

        Semantic facts are weighted by similarity and freshness, not importance.

        Args:
            query: The read query.
            event: The memory event to score.
            decay_factor: Pre-computed decay factor.

        Returns:
            Combined relevance score.
        """
        similarity = SequenceMatcher(None, query.query.lower(), event.content.lower()).ratio()
        return similarity * decay_factor
