"""Context buffer — short-term memory for current task context.

Stores memories scoped to a specific task, clearing on task switch.
"""

from __future__ import annotations

import time as time_module
from difflib import SequenceMatcher

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory


class ContextBuffer(MemoryWriter, MemoryReader):
    """Short-term context buffer scoped to tasks.

    Stores memories grouped by task_id. Retrieval only returns
    memories matching the query's task_id context.
    """

    def __init__(
        self,
        module_name: str | None = None,
        **_kwargs: object,
    ) -> None:
        """Initialize the context buffer.

        Args:
            module_name: Logical module name (defaults to 'context_buffer').
            **_kwargs: Ignored (allows uniform construction from factory).
        """
        self._module_name = module_name or "context_buffer"
        self._memories_by_task: dict[str, list[MemoryEvent]] = {}

    @property
    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event scoped to its task.

        Args:
            event: The memory event to store.
        """
        if event.task_id not in self._memories_by_task:
            self._memories_by_task[event.task_id] = []
        self._memories_by_task[event.task_id].append(event)

    def read(self, query: ReadQuery) -> ReadResponse:
        """Retrieve memories scoped to the query's task and user.

        Filters by task_id, user_id, and ReadQueryFilters.

        Args:
            query: The read query (must include context.task_id).

        Returns:
            ReadResponse with task-scoped results.
        """
        start_time = time_module.monotonic()
        task_memories = self._memories_by_task.get(query.context.task_id, [])
        user_id = query.context.user_id
        candidates = [event for event in task_memories if event.user_id == user_id]

        # Apply ReadQueryFilters
        if query.filters.memory_types:
            allowed = set(query.filters.memory_types)
            candidates = [e for e in candidates if e.type in allowed]

        if query.filters.min_importance > 0.0:
            threshold = query.filters.min_importance
            candidates = [e for e in candidates if e.importance >= threshold]

        scored = self._score_memories(query.query, candidates)
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
            total_candidates=len(task_memories),
        )

    def clear_task(self, task_id: str) -> None:
        """Clear all memories for a specific task.

        Args:
            task_id: The task to clear.
        """
        self._memories_by_task.pop(task_id, None)

    def clear(self) -> None:
        """Clear all memories across all tasks."""
        self._memories_by_task.clear()

    def _score_memories(
        self,
        query_text: str,
        memories: list[MemoryEvent],
    ) -> list[tuple[MemoryEvent, float]]:
        """Score memories against query text.

        Args:
            query_text: The query string.
            memories: List of memories to score.

        Returns:
            List of (event, score) tuples.
        """
        query_lower = query_text.lower()
        return [
            (event, SequenceMatcher(None, query_lower, event.content.lower()).ratio())
            for event in memories
        ]

    def _compute_confidence(self, score: float) -> float:
        """Compute retrieval confidence for context buffer.

        Context buffer memories are always HOT (decay_factor=1.0),
        so confidence is based purely on relevance score.

        Args:
            score: The relevance score of the result.

        Returns:
            A confidence value between 0.0 and 1.0.
        """
        raw = (score * 0.6) + (1.0 * 0.4)
        return max(0.0, min(1.0, raw))
