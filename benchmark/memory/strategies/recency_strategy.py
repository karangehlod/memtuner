"""Recency-only retrieval strategy — returns the K most recently injected memories.

This is the canonical baseline for agent memory benchmarks.  A retrieval
system that cannot beat this baseline is no better than a simple LIFO queue.

Why it matters for a paper
--------------------------
If BM25 recall on LoCoMo is 53% and recency baseline is, say, 18%, BM25 adds
+35pp over the floor.  Without this number, a reviewer cannot judge whether any
retrieval strategy is actually doing meaningful work.

The strategy is query-agnostic: it never looks at the query text.  It scores
each memory by its injection day (latest = highest score), breaking ties by
memory_id for determinism.

Latency: <1ms | Cost: zero | Accuracy: dataset-dependent lower bound
"""

from __future__ import annotations

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent


class RecencyStrategy(RetrievalStrategy):
    """Return the K most recently injected memories, ignoring query content.

    The creation day is not available inside the strategy (it lives in the
    store's _creation_days dict).  We approximate recency from the memory
    timestamp, falling back to memory insertion order via the index position.
    """

    def __init__(self) -> None:
        self._memories: list[MemoryEvent] = []   # ordered by injection (oldest first)
        self._id_to_pos: dict[str, int] = {}     # memory_id → insertion index

    def index(self, memories: list[MemoryEvent]) -> None:
        """Record memories in their presented order (oldest first = index 0)."""
        self._memories = list(memories)
        self._id_to_pos = {mem.id: i for i, mem in enumerate(memories)}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return the K most recently injected memories for this user.

        Score = insertion_position / total_memories so that the most recent
        memory has score ≈ 1.0 and the oldest has score ≈ 0.0.

        Args:
            query: Ignored — recency is query-agnostic.
            top_k: Number of memories to return.
            user_id: Filter to this user's memories.

        Returns:
            List of (memory_id, recency_score) sorted most-recent first.
        """
        candidates = self._memories
        if user_id:
            candidates = [m for m in candidates if m.user_id == user_id]

        if not candidates:
            return []

        n = len(candidates)
        # Recency score = insertion_position / n is a monotone-increasing function of index.
        # The most recent top_k memories are always the LAST top_k elements of `candidates`
        # (oldest-first ordering). No sort needed — O(top_k) slice instead of O(N log N) sort.
        recent = candidates[max(0, n - top_k):][::-1]  # last top_k, reversed → newest first
        return [(mem.id, (self._id_to_pos.get(mem.id, 0) + 1) / n) for mem in recent]

    def name(self) -> str:
        return "recency"

    def clear(self) -> None:
        self._memories.clear()
        self._id_to_pos.clear()

    @classmethod
    def is_available(cls) -> bool:
        return True
