"""Scratchpad — temporary working memory with manual clear.

A simple key-value scratchpad for temporary computations.
"""

from __future__ import annotations

import time as time_module
from difflib import SequenceMatcher

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory


class Scratchpad(MemoryWriter, MemoryReader):
    """Temporary working memory scratchpad.

    Stores all events without capacity limits. Intended for
    short-lived scratch computations within a scenario.
    """

    def __init__(
        self,
        module_name: str | None = None,
        **_kwargs: object,
    ) -> None:
        """Initialize the scratchpad.

        Args:
            module_name: Logical module name (defaults to 'scratchpad').
            **_kwargs: Ignored (allows uniform construction from factory).
        """
        self._module_name = module_name or "scratchpad"
        self._memories: dict[str, MemoryEvent] = {}

    @property
    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event to the scratchpad.

        Overwrites if the same ID already exists.

        Args:
            event: The memory event to store.
        """
        self._memories[event.id] = event

    def read(self, query: ReadQuery) -> ReadResponse:
        """Retrieve memories from the scratchpad by similarity.

        Filters by user_id from query context. Applies ReadQueryFilters.

        Args:
            query: The read query.

        Returns:
            ReadResponse with scored results.
        """
        start_time = time_module.monotonic()
        user_id = query.context.user_id
        candidates = [event for event in self._memories.values() if event.user_id == user_id]

        # Apply ReadQueryFilters
        if query.filters.memory_types:
            allowed = set(query.filters.memory_types)
            candidates = [e for e in candidates if e.type in allowed]

        if query.filters.min_importance > 0.0:
            threshold = query.filters.min_importance
            candidates = [e for e in candidates if e.importance >= threshold]

        scored = [
            (event, SequenceMatcher(None, query.query.lower(), event.content.lower()).ratio())
            for event in candidates
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top_k = scored[: query.top_k]

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
            total_candidates=len(candidates),
        )

    def clear(self) -> None:
        """Clear the scratchpad."""
        self._memories.clear()

    def count(self) -> int:
        """Return number of memories stored.

        Returns:
            Current count.
        """
        return len(self._memories)

    def _compute_confidence(self, score: float) -> float:
        """Compute retrieval confidence for scratchpad.

        Scratchpad memories are always HOT (decay_factor=1.0),
        so confidence is based purely on relevance score.

        Args:
            score: The relevance score of the result.

        Returns:
            A confidence value between 0.0 and 1.0.
        """
        raw = (score * 0.6) + (1.0 * 0.4)
        return max(0.0, min(1.0, raw))
