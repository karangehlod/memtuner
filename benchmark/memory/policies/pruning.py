"""Pruning policy strategies.

Pure, stateless, deterministic pruning decisions.
Policies return IDs to prune but do NOT mutate storage.
"""

from __future__ import annotations

from benchmark.memory.interfaces.lifecycle import LifecyclePolicy


class ScoreThresholdPruningPolicy(LifecyclePolicy):
    """Prune memories whose score falls below a threshold.

    This is the simplest pruning strategy — any memory with a
    current score below the threshold is flagged for removal.
    """

    def __init__(self, threshold: float = 0.35) -> None:
        """Initialize with score threshold.

        Args:
            threshold: Minimum score to survive pruning.
        """
        self._threshold = threshold

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Return memory IDs with scores below threshold.

        Args:
            day: The current simulated day (not used for score-based pruning).
            memory_scores: Mapping of memory_id → current score.

        Returns:
            List of memory IDs to prune.
        """
        return [memory_id for memory_id, score in memory_scores.items() if score < self._threshold]


class CapacityBasedPruningPolicy(LifecyclePolicy):
    """Prune lowest-scoring memories when capacity is exceeded.

    Keeps only the top-N memories by score, flagging the rest.
    """

    def __init__(self, max_capacity: int = 100) -> None:
        """Initialize with maximum capacity.

        Args:
            max_capacity: Maximum number of memories to keep.
        """
        self._max_capacity = max_capacity

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Return memory IDs that exceed capacity (lowest scores first).

        Args:
            day: The current simulated day (not used directly).
            memory_scores: Mapping of memory_id → current score.

        Returns:
            List of memory IDs to prune (those with lowest scores beyond capacity).
        """
        if len(memory_scores) <= self._max_capacity:
            return []

        sorted_by_score = sorted(memory_scores.items(), key=lambda pair: pair[1], reverse=True)
        to_keep = {memory_id for memory_id, _ in sorted_by_score[: self._max_capacity]}
        return [memory_id for memory_id in memory_scores if memory_id not in to_keep]


class AgeBasedPruningPolicy(LifecyclePolicy):
    """Prune memories whose decay-adjusted score has fallen below threshold.

    Identical to ScoreThresholdPruningPolicy in logic, but semantically
    represents "age-based" pruning configured for periodic decay policies.
    The distinction is in config naming, not algorithm.
    """

    def __init__(self, threshold: float = 0.30) -> None:
        """Initialize with score threshold.

        Args:
            threshold: Minimum score to survive pruning.
        """
        self._threshold = threshold

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Return memory IDs with scores below threshold.

        Args:
            day: The current simulated day.
            memory_scores: Mapping of memory_id → current score.

        Returns:
            List of memory IDs to prune.
        """
        return [memory_id for memory_id, score in memory_scores.items() if score < self._threshold]
