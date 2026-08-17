"""Preference store — long-term memory for user preferences.

Stores and retrieves preference-type memories with relevance scoring.
Only accepts memories with type PREFERENCE.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from benchmark.memory.long_term.base_store import BaseLongTermStore
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery

# Memory types accepted by this store
_ACCEPTED_TYPES = frozenset({MemoryType.PREFERENCE})


class PreferenceStore(BaseLongTermStore):
    """Long-term store for preference-type memories.

    Optimized for storing and retrieving user preference data.
    Applies a task-matching boost when the event's task_id matches the query.
    Only stores memories with type PREFERENCE.

    Scoring: text_similarity × importance × decay_factor × task_boost
    """

    TASK_BOOST_FACTOR: float = 1.2

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
        """Initialize the preference store.

        Args:
            decay_type: Type of decay function.
            decay_lambda: Decay rate parameter.
            pruning_strategy: Pruning strategy name (forwarded to base).
            pruning_threshold: Score threshold for pruning.
            module_name: Logical module name (defaults to 'preference_store').
            retrieval_strategy: Optional retrieval strategy instance.
            allow_strategy_fallback: Whether to allow fallback to default scoring.
            **kwargs: Forwarded to BaseLongTermStore (archival_floor, etc.).
        """
        super().__init__(
            module_name=module_name or "preference_store",
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
        """Compute preference relevance with task-matching boost.

        Args:
            query: The read query.
            event: The memory event to score.
            decay_factor: Pre-computed decay factor.

        Returns:
            Combined relevance score, clamped to [0, 1].
        """
        text_sim = SequenceMatcher(None, query.query.lower(), event.content.lower()).ratio()
        task_boost = self.TASK_BOOST_FACTOR if event.task_id == query.context.task_id else 1.0
        return min(1.0, text_sim * event.importance * decay_factor * task_boost)

    def _apply_module_weight(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        strategy_score: float,
        decay_factor: float,
    ) -> float:
        """Preference weighting: strategy_score × importance × task_boost.

        Decay is excluded — it is applied post-ranking by the base class.
        Task affinity gives a 1.2× boost when the query's task matches
        the memory's task_id, improving preference retrieval for task-scoped
        queries even when using generic retrieval strategies.
        """
        task_boost = self.TASK_BOOST_FACTOR if event.task_id == query.context.task_id else 1.0
        return strategy_score * event.importance * task_boost
