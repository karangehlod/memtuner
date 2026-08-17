"""Reliability curve evaluator.

Computes Memory Survival Rate per simulated day:
    SurvivalRate(day) = |MemoriesAlive(day)| / |MemoriesInjected(day=0)|
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator


@dataclass(frozen=True)
class ReliabilityCurveResult:
    """Result containing per-day survival rates.

    Attributes:
        survival_rates: Mapping of day → survival rate.
        total_injected: Total number of memories injected at start.
    """

    survival_rates: dict[int, float]
    total_injected: int


class ReliabilityCurveEvaluator(MetricEvaluator):
    """Computes memory survival rate over simulated days.

    Tracks how many memories remain alive (not pruned/decayed) over time.

    Formula:
        SurvivalRate(day) = |MemoriesAlive(day)| / |MemoriesInjected|

    This evaluator accumulates data across days and computes the curve.
    """

    def __init__(self) -> None:
        """Initialize the evaluator with empty tracking state."""
        self._total_injected: int = 0
        self._alive_per_day: dict[int, int] = {}

    def record_day(self, day: int, alive_count: int, injected_count: int = 0) -> None:
        """Record the memory state for a simulated day.

        Args:
            day: The simulated day number.
            alive_count: Number of memories still alive.
            injected_count: Number of new memories injected on this day.
        """
        self._total_injected += injected_count
        self._alive_per_day[day] = alive_count

    def compute_curve(self) -> ReliabilityCurveResult:
        """Compute the survival rate curve across all recorded days.

        Returns:
            ReliabilityCurveResult with per-day survival rates.
        """
        if self._total_injected == 0:
            return ReliabilityCurveResult(survival_rates={}, total_injected=0)

        survival_rates = {
            day: alive / self._total_injected for day, alive in sorted(self._alive_per_day.items())
        }

        return ReliabilityCurveResult(
            survival_rates=survival_rates,
            total_injected=self._total_injected,
        )

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Evaluate survival rate (returns latest day's rate).

        Args:
            retrieved_ids: Not used for survival rate.
            expected_ids: Not used for survival rate.

        Returns:
            EvaluationResult with the latest survival rate.
        """
        curve = self.compute_curve()
        latest_rate = 0.0
        if curve.survival_rates:
            latest_day = max(curve.survival_rates.keys())
            latest_rate = curve.survival_rates[latest_day]

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=latest_rate,
            query_count=1,
            details={str(day): rate for day, rate in curve.survival_rates.items()},
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.memory_survival_rate"

    def reset(self) -> None:
        """Reset internal tracking state between scenario runs."""
        self._total_injected = 0
        self._alive_per_day.clear()
