"""LifecyclePolicy interface.

Defines the contract for memory lifecycle operations (decay, pruning, promotion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LifecyclePolicy(ABC):
    """Abstract interface for memory lifecycle management.

    Policies are stateless, pure, and deterministic.
    They decide WHAT should happen but do NOT mutate storage directly.

    The return value is a list of memory IDs that the policy
    recommends for action (e.g., pruning, promotion).
    """

    @abstractmethod
    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Apply the lifecycle policy for a given simulated day.

        Args:
            day: The current simulated day.
            memory_scores: Mapping of memory_id → current score.

        Returns:
            List of memory IDs that this policy recommends for action.
            The caller decides what action to take (prune, promote, etc.).
        """
