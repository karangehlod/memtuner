"""Episodic buffer — short-term memory with limited capacity.

Stores recent memory events in a bounded FIFO buffer.
Implements both MemoryWriter and MemoryReader interfaces.
"""

from __future__ import annotations

import time as time_module
from collections import deque
from difflib import SequenceMatcher

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory


class EpisodicBuffer(MemoryWriter, MemoryReader):
    """Short-term episodic buffer with bounded FIFO capacity.

    Stores the most recent N memory events. When capacity is exceeded,
    oldest events are evicted. Retrieval uses simple text similarity.

    This is an in-memory implementation with no external dependencies.
    """

    def __init__(
        self,
        capacity: int = 50,
        module_name: str | None = None,
        **_kwargs: object,
    ) -> None:
        """Initialize the episodic buffer.

        Args:
            capacity: Maximum number of events to store.
            module_name: Logical module name (defaults to 'episodic_buffer').
            **_kwargs: Ignored (allows uniform construction from factory).
        """
        self._capacity = capacity
        self._module_name = module_name or "episodic_buffer"
        self._buffer: deque[MemoryEvent] = deque(maxlen=capacity)
        # BUG-006 FIX: Track the injection day for each event so temporal
        # evaluators see the correct day instead of always 0.
        self._creation_days: dict[str, int] = {}

    @property
    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event to the buffer.

        If the buffer is at capacity, the oldest event is evicted.

        Args:
            event: The memory event to store.
        """
        self._buffer.append(event)
        self._creation_days[event.id] = 0

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Write a memory event tagged with its injection day.

        Used by the scenario runner to preserve temporal context.

        Args:
            event: The memory event to store.
            day: The simulated day of injection.
        """
        self._buffer.append(event)
        self._creation_days[event.id] = day

    def get_creation_day(self, memory_id: str) -> int | None:
        """Return the day a memory was injected, or None if not found."""
        return self._creation_days.get(memory_id)

    def read(self, query: ReadQuery) -> ReadResponse:
        """Retrieve memories matching the query by text similarity.

        Filters by user_id from query context. Applies ReadQueryFilters
        for memory_types and min_importance if provided.

        Args:
            query: The read query.

        Returns:
            ReadResponse with top-K results ordered by relevance score.
        """
        start_time = time_module.monotonic()
        user_id = query.context.user_id
        candidates = self._get_filtered_candidates(query, user_id)
        scored_memories = self._score_candidate_list(query.query, candidates)
        scored_memories.sort(key=lambda pair: pair[1], reverse=True)
        top_k = scored_memories[: query.top_k]

        retrieved = [
            RetrievedMemory(
                memory_id=event.id,
                source_module=self._module_name,
                score=score,
                confidence=self._compute_confidence(score),
                timestamp=event.timestamp,
                tier=MemoryTier.HOT,
                decay_factor=1.0,
            )
            for event, score in top_k
        ]

        elapsed_ms = (time_module.monotonic() - start_time) * 1000.0

        return ReadResponse(
            retrieved_memories=retrieved,
            latency_ms=elapsed_ms,
            total_candidates=len(self._buffer),
        )

    def count(self) -> int:
        """Return the number of events currently in the buffer.

        Returns:
            Current buffer size.
        """
        return len(self._buffer)

    def clear(self) -> None:
        """Clear all events from the buffer."""
        self._buffer.clear()
        self._creation_days.clear()

    def _get_filtered_candidates(self, query: ReadQuery, user_id: str) -> list[MemoryEvent]:
        """Get candidates filtered by user, memory_types, and min_importance.

        Args:
            query: The read query with optional filters.
            user_id: The user whose memories to retrieve.

        Returns:
            Filtered list of candidate events.
        """
        candidates = [e for e in self._buffer if e.user_id == user_id]

        if query.filters.memory_types:
            allowed = set(query.filters.memory_types)
            candidates = [e for e in candidates if e.type in allowed]

        if query.filters.min_importance > 0.0:
            threshold = query.filters.min_importance
            candidates = [e for e in candidates if e.importance >= threshold]

        return candidates

    def _score_candidate_list(
        self, query_text: str, candidates: list[MemoryEvent]
    ) -> list[tuple[MemoryEvent, float]]:
        """Score a list of candidate events against query text.

        Args:
            query_text: The query string.
            candidates: Pre-filtered list of events.

        Returns:
            List of (event, score) tuples.
        """
        results: list[tuple[MemoryEvent, float]] = []
        query_lower = query_text.lower()

        for event in candidates:
            content_lower = event.content.lower()
            similarity = SequenceMatcher(None, query_lower, content_lower).ratio()
            keyword_boost = self._keyword_overlap_score(query_lower, content_lower)
            combined_score = min(1.0, similarity * 0.6 + keyword_boost * 0.4)
            results.append((event, combined_score))

        return results

    def _keyword_overlap_score(self, query: str, content: str) -> float:
        """Compute keyword overlap between query and content.

        Args:
            query: Lowered query string.
            content: Lowered content string.

        Returns:
            Overlap score between 0.0 and 1.0.
        """
        query_words = set(query.split())
        content_words = set(content.split())
        if not query_words:
            return 0.0
        overlap = query_words & content_words
        return len(overlap) / len(query_words)

    def _compute_confidence(self, score: float) -> float:
        """Compute retrieval confidence for short-term buffer.

        Short-term memories are always HOT (decay_factor=1.0),
        so confidence is based purely on relevance score.

        Args:
            score: The relevance score of the result.

        Returns:
            A confidence value between 0.0 and 1.0.
        """
        raw = (score * 0.6) + (1.0 * 0.4)
        return max(0.0, min(1.0, raw))
