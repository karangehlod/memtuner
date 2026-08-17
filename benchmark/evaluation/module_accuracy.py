"""Module accuracy evaluator.

Computes what fraction of retrieved memories came from acceptable source modules.
Uses the gold query's `acceptable_modules` list to validate source attribution.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
from benchmark.evaluation.context import EvaluationContext


class ModuleAccuracyEvaluator(MetricEvaluator):
    """Computes Module Accuracy metric.

    Measures whether retrieved memories came from the correct memory modules
    as specified by the gold query's acceptable_modules list.

    Formula:
        ModuleAccuracy = (count of retrieved from acceptable modules) / (total retrieved)

    If acceptable_modules is empty, all modules are acceptable (score = 1.0).
    If no memories were retrieved, score = 1.0 (vacuously true).
    """

    def evaluate_with_context(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate module accuracy using EvaluationContext.

        Args:
            context: Full evaluation context with source module mappings
                and acceptable_modules constraint.

        Returns:
            EvaluationResult with module accuracy between 0.0 and 1.0.
            Returns value=1.0 when acceptable_modules is empty (all acceptable).
            Returns value=0.0 with query_count=0 when nothing was retrieved
            (excluded from averages, not a vacuous 1.0).
        """
        # No constraints means all source modules are acceptable.
        if not context.acceptable_modules:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=1.0,
                query_count=1,
                details={"context_called": True, "reason": "no_module_constraints"},
            )

        # Nothing retrieved — do not count toward averages.
        if not context.retrieved_ids:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=0.0,
                query_count=0,
                details={"context_called": True, "reason": "nothing_retrieved"},
            )

        acceptable = set(context.acceptable_modules)
        from_acceptable = sum(
            1
            for rid in context.retrieved_ids
            if context.retrieved_source_modules.get(rid, "") in acceptable
        )

        accuracy = from_acceptable / len(context.retrieved_ids)

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=accuracy,
            query_count=1,
            details={
                "context_called": True,
                "from_acceptable": from_acceptable,
                "total_retrieved": len(context.retrieved_ids),
                "acceptable_modules": len(context.acceptable_modules),
            },
        )

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """ID-only evaluation not supported for module accuracy.

        Args:
            retrieved_ids: Memory IDs returned by memory system.
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            Never returns; always raises.

        Raises:
            ValueError: Module accuracy requires EvaluationContext with source module data.
        """
        raise ValueError(
            "ModuleAccuracyEvaluator requires EvaluationContext with source module mappings. "
            "Use evaluate_with_context() instead of evaluate()."
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.module_accuracy"
