"""Temporal accuracy evaluator.

TemporalAccuracy = (1/N) * Σ I(day_retrieved ∈ [expected_window ± tolerance])

Interpretation
--------------
Measures whether retrieved memories fall within the expected time window.
A score of 1.0 means every retrieved memory was created in the right period.
A score of 0.5 means half of retrieved memories were temporally on-target.

Design notes
------------
- When no memories are retrieved (N=0), the score is 0.0 — a system that
  returns nothing has not demonstrated temporal accuracy.
- When no temporal constraint exists (temporal_window=None), the evaluator
  returns query_count=0 so it is excluded from the weighted-mean aggregation.
  This avoids inflating or deflating the mean for datasets without timestamps.
- Tolerance (default 1 day) accounts for off-by-one errors in day assignment.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
from benchmark.evaluation.context import EvaluationContext


class TemporalAccuracyEvaluator(MetricEvaluator):
    """Computes Temporal Accuracy metric.

    Measures whether retrieved memories fall within the expected temporal window.

    Formula:
        TemporalAccuracy = (1/N) * Σ I(day_retrieved ∈ [day_expected ± tolerance])

    Where N is the number of retrieved memories that have temporal expectations.
    If N = 0, temporal accuracy is defined as 1.0 (vacuously true).
    """

    def __init__(
        self,
        tolerance_days: int = 1,
        temporal_tolerance_days: int | None = None,
    ) -> None:
        """Initialize with temporal tolerance.

        Args:
            tolerance_days: Number of days of tolerance for temporal matching.
            temporal_tolerance_days: Alias for tolerance_days (preferred name).
        """
        self._tolerance_days = (
            temporal_tolerance_days if temporal_tolerance_days is not None else tolerance_days
        )

    def evaluate_temporal(
        self,
        retrieved_days: list[int],
        expected_day_range: tuple[int, int],
    ) -> EvaluationResult:
        """Compute temporal accuracy for retrieved memory timestamps.

        Args:
            retrieved_days: Days when retrieved memories were created.
            expected_day_range: Tuple of (not_before_day, not_after_day).

        Returns:
            EvaluationResult with temporal accuracy between 0.0 and 1.0.
        """
        if not retrieved_days:
            # Nothing retrieved → 0.0.
            # "Vacuously true" (1.0) would reward systems that retrieve nothing,
            # masking retrieval failures in the composite score.
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=0.0,
                query_count=1,
                details={"note": "No memories retrieved — temporal accuracy 0.0"},
            )

        # Expand the window by ±tolerance to absorb off-by-one day-assignment errors.
        not_before = expected_day_range[0] - self._tolerance_days
        not_after  = expected_day_range[1] + self._tolerance_days

        # Count how many retrieved memories fall inside the window.
        matches = sum(1 for day in retrieved_days if not_before <= day <= not_after)

        # Denominator = N (total retrieved with known days), not |gold|.
        # We measure the temporal quality of what WAS retrieved, not what was missed.
        accuracy = matches / len(retrieved_days)

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=accuracy,
            query_count=1,
            details={
                "matches": matches,
                "total": len(retrieved_days),
                "tolerance_days": self._tolerance_days,
            },
        )

    def evaluate_with_context(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate temporal accuracy using the rich EvaluationContext.

        Extracts creation days from context and uses the temporal window
        from the gold dataset. Falls back to vacuously-true if no
        temporal window is specified.

        Args:
            context: Full evaluation context with creation days and temporal window.

        Returns:
            EvaluationResult with temporal accuracy.
        """
        if context.temporal_window is None:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=1.0,
                query_count=0,
                details={"note": "No temporal constraint — not evaluated"},
            )

        # Build creation-day list only for IDs that have a known day.
        # IDs without a day (module doesn't implement CreationDayTracker) are
        # excluded from the denominator — they don't count as in-window or out-of-window.
        # If NO retrieved IDs have known days, evaluate_temporal returns 0.0.
        retrieved_days = [
            context.retrieved_creation_days[rid]
            for rid in context.retrieved_ids
            if rid in context.retrieved_creation_days
        ]

        return self.evaluate_temporal(retrieved_days, context.temporal_window)

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Evaluate temporal accuracy (ID-only fallback).

        For the standard interface, this returns a neutral result.
        Use evaluate_with_context() or evaluate_temporal() for full evaluation.

        Args:
            retrieved_ids: Memory IDs returned by memory system.
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            EvaluationResult with value 1.0 (no temporal data available).
        """
        return EvaluationResult(
            metric_name=self.metric_name(),
            value=1.0,
            query_count=1,
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.temporal_accuracy"
