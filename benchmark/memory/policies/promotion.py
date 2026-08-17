"""Promotion policy — STM to LTM promotion decisions.

Pure, stateless, deterministic promotion decisions.
"""

from __future__ import annotations

from benchmark.memory.interfaces.lifecycle import LifecyclePolicy


class ImportanceBasedPromotionPolicy(LifecyclePolicy):
    """Promote memories from STM to LTM based on importance threshold.

    Memories with importance above the threshold are flagged for promotion.
    """

    def __init__(self, importance_threshold: float = 0.6) -> None:
        """Initialize with importance threshold.

        Args:
            importance_threshold: Minimum importance to qualify for promotion.
        """
        self._threshold = importance_threshold

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Return memory IDs with scores above promotion threshold.

        Args:
            day: The current simulated day.
            memory_scores: Mapping of memory_id → importance score.

        Returns:
            List of memory IDs to promote.
        """
        return [memory_id for memory_id, score in memory_scores.items() if score >= self._threshold]
