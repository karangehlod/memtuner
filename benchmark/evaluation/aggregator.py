"""Metric aggregator — combines results from all evaluators.

Produces aggregate metrics for a complete benchmark run.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult


class MetricAggregator:
    """Aggregates evaluation results across multiple queries and scenarios.

    Computes weighted averages of all metrics for the final benchmark result.
    This class is stateless per invocation — all data passed via arguments.
    """

    def aggregate_results(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """Aggregate a list of evaluation results by metric name.

        Computes the weighted average (by query_count) for each metric.

        Args:
            results: List of EvaluationResult from individual evaluations.

        Returns:
            Dictionary mapping metric_name → aggregated value.
        """
        if not results:
            return {}

        grouped = self._group_by_metric(results)
        return {
            metric_name: self._weighted_average(metric_results)
            for metric_name, metric_results in grouped.items()
        }

    def _group_by_metric(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, list[EvaluationResult]]:
        """Group evaluation results by metric name.

        Args:
            results: List of evaluation results.

        Returns:
            Dictionary mapping metric name to list of results.
        """
        grouped: dict[str, list[EvaluationResult]] = {}
        for result in results:
            if result.metric_name not in grouped:
                grouped[result.metric_name] = []
            grouped[result.metric_name].append(result)
        return grouped

    def _weighted_average(self, results: list[EvaluationResult]) -> float:
        """Compute weighted average of evaluation results.

        Weight is the query_count of each result.

        Args:
            results: List of results for the same metric.

        Returns:
            Weighted average value.
        """
        total_weight = sum(result.query_count for result in results)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(result.value * result.query_count for result in results)
        return weighted_sum / total_weight
