"""Entity store — in-memory long-term entity memory.

Stores entity-type memories (people, organizations, things) with
entity-aware retrieval that boosts results matching query entities.
Only accepts memories with type ENTITY.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from benchmark.memory.long_term.base_store import BaseLongTermStore
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery

# Memory types accepted by this store
_ACCEPTED_TYPES = frozenset({MemoryType.ENTITY})


class EntityStore(BaseLongTermStore):
    """In-memory long-term store for entity-type memories.

    Optimized for storing and retrieving information about named entities.
    Applies entity-matching boost to retrieval scores.
    Only stores memories with type ENTITY.

    Scoring: (text_similarity + entity_boost) × decay_factor
    """

    ENTITY_BOOST_FACTOR: float = 0.15

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
        """Initialize the entity store.

        Args:
            decay_type: Type of decay function.
            decay_lambda: Decay rate parameter.
            pruning_strategy: Pruning strategy name (forwarded to base).
            pruning_threshold: Score threshold for pruning.
            module_name: Logical module name (defaults to 'entity_store').
            retrieval_strategy: Optional retrieval strategy instance.
            allow_strategy_fallback: Whether to allow fallback to default scoring.
            **kwargs: Forwarded to BaseLongTermStore (archival_floor, etc.).
        """
        super().__init__(
            module_name=module_name or "entity_store",
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
        """Compute entity-boosted relevance score.

        Boosts score when event entities appear in the query text.

        Args:
            query: The read query.
            event: The memory event to score.
            decay_factor: Pre-computed decay factor.

        Returns:
            Combined relevance score.
        """
        query_lower = query.query.lower()
        similarity = SequenceMatcher(None, query_lower, event.content.lower()).ratio()

        entity_boost = sum(
            self.ENTITY_BOOST_FACTOR for entity in event.entities if entity.lower() in query_lower
        )

        return (similarity + entity_boost) * decay_factor

    def _apply_module_weight(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        strategy_score: float,
        decay_factor: float,
    ) -> float:
        """Entity weighting: boosts score when entities match the query.

        Decay is excluded — it is applied post-ranking by the base class.
        Entity matching gives a 0.15 boost per matched entity, improving
        retrieval for entity-centric queries even with generic strategies.
        """
        query_lower = query.query.lower()
        entity_boost = sum(
            self.ENTITY_BOOST_FACTOR for entity in event.entities if entity.lower() in query_lower
        )
        return (strategy_score + entity_boost) * event.importance
