"""Decay policy strategies.

Pure, stateless, deterministic decay computations.
Policies decide WHAT should happen but do NOT mutate storage.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from benchmark.memory.interfaces.lifecycle import LifecyclePolicy


class DecayPolicy(ABC):
    """Abstract decay policy for computing decay factor.

    Used for testing and calculating raw decay values.
    """

    @abstractmethod
    def compute_decay(self, initial_score: float, days_elapsed: int) -> float:
        """Compute the decayed score.

        Args:
            initial_score: The original score.
            days_elapsed: Days that have elapsed.

        Returns:
            The decayed score value.
        """


class ExponentialDecayPolicy(LifecyclePolicy, DecayPolicy):
    """Exponential decay policy.

    Score decays as: score * e^(-λ * days_elapsed)
    Returns memory IDs whose decayed scores fall below threshold.
    """

    def __init__(
        self,
        decay_lambda: float = 0.05,
        threshold: float = 0.35,
    ) -> None:
        """Initialize exponential decay policy.

        Args:
            decay_lambda: The decay rate parameter (λ).
            threshold: Score threshold below which memories are flagged.
        """
        self._decay_lambda = decay_lambda
        self._threshold = threshold

    def compute_decay(self, initial_score: float, days_elapsed: int) -> float:
        """Compute exponential decay.

        Args:
            initial_score: The original score.
            days_elapsed: Days that have elapsed.

        Returns:
            The decayed score value.
        """
        return initial_score * math.exp(-self._decay_lambda * days_elapsed)

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Apply exponential decay and return IDs below threshold.

        The scores passed in should be the original (non-decayed) scores.
        This method applies decay internally and returns IDs that fall below threshold.

        Args:
            day: The current simulated day (used as elapsed time).
            memory_scores: Mapping of memory_id → current raw score.

        Returns:
            List of memory IDs whose decayed score is below the threshold.
        """
        # exp(-λ*day) is constant across all memories on a given day — compute once.
        factor = math.exp(-self._decay_lambda * day)
        threshold = self._threshold
        return [mid for mid, score in memory_scores.items() if score * factor < threshold]


class LinearDecayPolicy(LifecyclePolicy, DecayPolicy):
    """Linear decay policy.

    Score decays as: score * max(0, 1 - rate * days_elapsed)
    Returns memory IDs whose decayed scores fall below threshold.
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        threshold: float = 0.35,
    ) -> None:
        """Initialize linear decay policy.

        Args:
            decay_rate: The linear decay rate per day.
            threshold: Score threshold below which memories are flagged.
        """
        self._decay_rate = decay_rate
        self._threshold = threshold

    def compute_decay(self, initial_score: float, days_elapsed: int) -> float:
        """Compute linear decay.

        Args:
            initial_score: The original score.
            days_elapsed: Days that have elapsed.

        Returns:
            The decayed score value.
        """
        decay_factor = max(0.0, 1.0 - self._decay_rate * days_elapsed)
        return initial_score * decay_factor

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Apply linear decay and return IDs below threshold.

        Args:
            day: The current simulated day.
            memory_scores: Mapping of memory_id → current raw score.

        Returns:
            List of memory IDs whose decayed score is below the threshold.
        """
        # (1 - rate*day) is constant across all memories on a given day — compute once.
        factor = max(0.0, 1.0 - self._decay_rate * day)
        threshold = self._threshold
        return [mid for mid, score in memory_scores.items() if score * factor < threshold]


class StepDecayPolicy(LifecyclePolicy, DecayPolicy):
    """Step decay policy.

    Score drops to zero after a fixed number of days.
    Returns memory IDs that have exceeded their lifespan.
    """

    def __init__(
        self,
        lifespan_days: int = 10,
        threshold: float = 0.35,
    ) -> None:
        """Initialize step decay policy.

        Args:
            lifespan_days: Days after which memories are flagged (decayed score = 0.0).
            threshold: Applied to raw scores for memories before lifespan_days.
                       Memories past lifespan always flag (decay=0.0 < any positive threshold).
        """
        self._lifespan_days = lifespan_days
        self._threshold = threshold

    def compute_decay(self, initial_score: float, days_elapsed: int) -> float:
        """Compute step decay.

        Args:
            initial_score: The original score.
            days_elapsed: Days that have elapsed.

        Returns:
            The decayed score value (0 if past lifespan, initial_score otherwise).
        """
        if days_elapsed >= self._lifespan_days:
            return 0.0
        return initial_score

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        """Apply step decay and return IDs whose decayed score falls below threshold.

        The step function outcome for `day` is the same for every memory — hoist it
        once instead of calling compute_decay() N times (N = number of memories).

        Args:
            day: The current simulated day.
            memory_scores: Mapping of memory_id → current raw score.

        Returns:
            List of memory IDs whose decayed score is below the threshold.
        """
        if day >= self._lifespan_days:
            # All memories past lifespan decay to 0.0 — always below any positive threshold.
            return list(memory_scores.keys())
        # Before lifespan: score is unchanged; check raw score against threshold.
        return [mid for mid, score in memory_scores.items() if score < self._threshold]
