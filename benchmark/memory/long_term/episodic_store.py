"""Episodic store — in-memory long-term episodic memory.

Stores episodic memories with decay support.
This is the in-memory reference implementation (no pgvector dependency).

Only accepts memories with type EPISODIC or ENTITY. Other memory types
(SEMANTIC, PREFERENCE) are silently skipped — they belong in their
respective stores.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from benchmark.memory.long_term.base_store import BaseLongTermStore
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery


class EpisodicStore(BaseLongTermStore):
    """In-memory long-term episodic memory store.

    Supports text-similarity retrieval and decay-based scoring.
    Memories decay over simulated days based on the configured lambda.

    Only stores memories with type EPISODIC or ENTITY.
    Scoring: text_similarity × importance × decay_factor
    """

    def __init__(
        self,
        decay_type: str = "exponential",
        decay_lambda: float = 0.05,
        pruning_strategy: str = "score_threshold",
        pruning_threshold: float = 0.35,
        module_name: str | None = None,
        retrieval_strategy: object | None = None,
        allow_strategy_fallback: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialize the episodic store.

        Args:
            decay_type: Type of decay function.
            decay_lambda: Decay rate parameter.
            pruning_strategy: Pruning strategy name (accepted but forwarded to base).
            pruning_threshold: Score threshold for pruning.
            module_name: Logical module name (defaults to 'episodic_store').
            retrieval_strategy: Optional retrieval strategy instance.
            allow_strategy_fallback: Whether to allow fallback to default scoring.
            **kwargs: Forwarded to BaseLongTermStore (e.g. archival_floor,
                      archival_day_threshold, tiered_working_days, decay_ranking_alpha).
        """
        super().__init__(
            module_name=module_name or "episodic_store",
            decay_type=decay_type,
            decay_lambda=decay_lambda,
            pruning_threshold=pruning_threshold,
            retrieval_strategy=retrieval_strategy,
            allow_strategy_fallback=allow_strategy_fallback,
            **kwargs,
        )

    def write(self, event: MemoryEvent) -> None:
        """Write all memory types — episodic is the primary/universal store."""
        super().write(event)

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Write all memory types — episodic is the primary/universal store."""
        super().write_on_day(event, day)

    def _compute_relevance_score(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        decay_factor: float,
    ) -> float:
        """Compute episodic relevance: text_similarity × importance × decay.

        Args:
            query: The read query.
            event: The memory event to score.
            decay_factor: Pre-computed decay factor.

        Returns:
            Combined relevance score.
        """
        text_sim = SequenceMatcher(None, query.query.lower(), event.content.lower()).ratio()
        return text_sim * event.importance * decay_factor

    def _apply_module_weight(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        strategy_score: float,
        decay_factor: float,
    ) -> float:
        """Episodic weighting: strategy_score × importance.

        Decay is excluded — it is applied post-ranking by the base class.
        Episodic memories are weighted by importance so more significant
        episodes are ranked higher in retrieval results.
        """
        return strategy_score * event.importance
