"""MetricEvaluator interface and evaluation base types.

Defines the contract for all metric evaluators.
Evaluators are pure functions — they depend ONLY on memory IDs + gold oracle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from benchmark.evaluation.context import EvaluationContext


@dataclass(frozen=True)
class EvaluationResult:
    """Result of a single metric evaluation.

    Attributes:
        metric_name: The name of the metric (matches OTel metric names).
        value: The computed metric value.
        query_count: Number of queries evaluated.
        details: Optional per-query breakdown.
    """

    metric_name: str
    value: float
    query_count: int
    details: dict[str, float] | None = None


class MetricEvaluator(ABC):
    """Abstract interface for metric evaluators.

    Each evaluator computes exactly ONE metric.
    Evaluators are stateless and deterministic.

    They receive an EvaluationContext containing:
    - retrieved memory IDs (from the memory system under test)
    - expected memory IDs (from the gold oracle)
    - source module mapping
    - creation-day mapping
    - temporal window constraints
    - conversation metadata

    Each evaluator picks only the fields it needs.
    They NEVER access memory storage directly.
    """

    @abstractmethod
    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Evaluate a single query's results against gold truth.

        This is the legacy interface for simple ID-based evaluators.

        Args:
            retrieved_ids: Memory IDs returned by the memory system (ordered by score).
            expected_ids: Memory IDs expected from the gold dataset.

        Returns:
            An EvaluationResult with the computed metric value.
        """

    def evaluate_with_context(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate using the rich EvaluationContext.

        Default implementation delegates to evaluate(retrieved_ids, expected_ids).
        Evaluators that need richer context override this method.

        Args:
            context: Full evaluation context with IDs, modules, days, etc.

        Returns:
            An EvaluationResult with the computed metric value.
        """
        return self.evaluate(context.retrieved_ids, context.expected_ids)

    @abstractmethod
    def metric_name(self) -> str:
        """Return the fixed metric name for this evaluator.

        Returns:
            The metric name string matching OTel schema.
        """
