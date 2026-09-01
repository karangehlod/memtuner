"""Aggregate and analyze distributed execution results."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics from multiple executions."""
    metric_name: str
    values: list[float]
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    count: int
    percentiles: dict[int, float]


@dataclass
class AggregatedResult:
    """Result from aggregating multiple query results."""
    total_results: int
    successful_results: int
    failed_results: int
    aggregated_metrics: dict[str, AggregatedMetrics]
    percentile_latencies: dict[int, float]
    consistency_score: float


class ResultAggregator:
    """Aggregate results from parallel executions."""

    def __init__(self):
        """Initialize result aggregator."""
        self.results_buffer: list[dict[str, Any]] = []

    def aggregate_query_results(
        self,
        results: list[dict[str, Any]],
    ) -> AggregatedResult:
        """Aggregate query results from multiple executions.

        Args:
            results: List of query result dicts

        Returns:
            AggregatedResult with combined metrics
        """
        if not results:
            raise ValueError("Cannot aggregate empty result list")

        # Count successes and failures
        successful = [r for r in results if r.get("success", True)]
        failed = [r for r in results if not r.get("success", True)]

        # Collect all latencies for percentile computation
        all_latencies = []
        for result in successful:
            if "time_ms" in result:
                all_latencies.append(result["time_ms"])

        # Compute percentiles
        percentiles = self._compute_percentiles(all_latencies)

        # Aggregate metrics
        metrics_dict = self._aggregate_metrics(results)

        # Compute consistency score
        consistency = self._compute_consistency_score(results)

        return AggregatedResult(
            total_results=len(results),
            successful_results=len(successful),
            failed_results=len(failed),
            aggregated_metrics=metrics_dict,
            percentile_latencies=percentiles,
            consistency_score=consistency,
        )

    def aggregate_metrics(
        self,
        metrics_list: list[dict[str, Any]],
    ) -> dict[str, AggregatedMetrics]:
        """Aggregate metrics from multiple sources.

        Args:
            metrics_list: List of metric dicts

        Returns:
            Dict mapping metric name to aggregated stats
        """
        if not metrics_list:
            return {}

        # Collect values by metric name
        metric_values: dict[str, list[float]] = {}

        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(float(value))

        # Aggregate each metric
        aggregated = {}
        for metric_name, values in metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(
                metric_name,
                values,
            )

        return aggregated

    def compute_aggregate_statistics(
        self,
        results: list[Any],
    ) -> dict[str, Any]:
        """Compute aggregate statistics from results.

        Args:
            results: List of result objects or dicts

        Returns:
            Dict with computed statistics
        """
        if not results:
            return self._empty_statistics()

        # Extract numeric values from results
        all_values: dict[str, list[float]] = {}

        for result in results:
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in all_values:
                            all_values[key] = []
                        all_values[key].append(float(value))

        # Compute statistics per value
        statistics_dict = {}

        for metric_name, values in all_values.items():
            values_clean = [v for v in values if not np.isnan(v) and not np.isinf(v)]

            if values_clean:
                statistics_dict[metric_name] = {
                    "count": len(values_clean),
                    "mean": float(np.mean(values_clean)),
                    "median": float(np.median(values_clean)),
                    "std_dev": float(np.std(values_clean)) if len(values_clean) > 1 else 0.0,
                    "min": float(np.min(values_clean)),
                    "max": float(np.max(values_clean)),
                    "sum": float(np.sum(values_clean)),
                }

        return statistics_dict

    # Private helper methods

    def _aggregate_single_metric(
        self,
        metric_name: str,
        values: list[float],
    ) -> AggregatedMetrics:
        """Aggregate a single metric."""
        if not values:
            return AggregatedMetrics(
                metric_name=metric_name,
                values=[],
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_val=0.0,
                max_val=0.0,
                count=0,
                percentiles={},
            )

        # Clean NaN and inf values
        clean_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

        if not clean_values:
            return AggregatedMetrics(
                metric_name=metric_name,
                values=[],
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_val=0.0,
                max_val=0.0,
                count=0,
                percentiles={},
            )

        mean_val = float(np.mean(clean_values))
        median_val = float(np.median(clean_values))
        std_dev = float(np.std(clean_values)) if len(clean_values) > 1 else 0.0
        min_val = float(np.min(clean_values))
        max_val = float(np.max(clean_values))

        # Compute percentiles
        percentiles = {
            50: float(np.percentile(clean_values, 50)),
            95: float(np.percentile(clean_values, 95)),
            99: float(np.percentile(clean_values, 99)),
        }

        return AggregatedMetrics(
            metric_name=metric_name,
            values=clean_values,
            mean=mean_val,
            median=median_val,
            std_dev=std_dev,
            min_val=min_val,
            max_val=max_val,
            count=len(clean_values),
            percentiles=percentiles,
        )

    def _aggregate_metrics(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, AggregatedMetrics]:
        """Extract and aggregate metrics from results."""
        metrics_dict: dict[str, list[float]] = {}

        for result in results:
            # Extract time_ms if present
            if "time_ms" in result and isinstance(result["time_ms"], (int, float)):
                if "latency_ms" not in metrics_dict:
                    metrics_dict["latency_ms"] = []
                metrics_dict["latency_ms"].append(float(result["time_ms"]))

            # Extract other numeric metrics
            if "metrics" in result and isinstance(result["metrics"], dict):
                for key, value in result["metrics"].items():
                    if isinstance(value, (int, float)):
                        if key not in metrics_dict:
                            metrics_dict[key] = []
                        metrics_dict[key].append(float(value))

        # Aggregate each metric
        aggregated = {}
        for metric_name, values in metrics_dict.items():
            aggregated[metric_name] = self._aggregate_single_metric(
                metric_name,
                values,
            )

        return aggregated

    def _compute_percentiles(
        self,
        values: list[float],
    ) -> dict[int, float]:
        """Compute percentiles for given values."""
        if not values:
            return {50: 0.0, 95: 0.0, 99: 0.0}

        clean_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

        if not clean_values:
            return {50: 0.0, 95: 0.0, 99: 0.0}

        return {
            50: float(np.percentile(clean_values, 50)),
            95: float(np.percentile(clean_values, 95)),
            99: float(np.percentile(clean_values, 99)),
        }

    def _compute_consistency_score(
        self,
        results: list[dict[str, Any]],
    ) -> float:
        """Compute consistency score (0.0 to 1.0).

        High score means results are consistent across runs.
        Based on success rate and metric variance.
        """
        if not results:
            return 0.0

        # Success rate component
        successful = sum(1 for r in results if r.get("success", True))
        success_rate = successful / len(results)

        # Variance component (lower variance = higher consistency)
        latencies = [r.get("time_ms", 0) for r in results if isinstance(r.get("time_ms"), (int, float))]
        if latencies and len(latencies) > 1:
            variance = float(np.std(latencies))
            mean_latency = float(np.mean(latencies))
            # Normalize variance to 0-1 (cv = coefficient of variation)
            cv = variance / mean_latency if mean_latency > 0 else 0.0
            # Convert to consistency (inverse of CV)
            variance_component = 1.0 / (1.0 + cv)
        else:
            variance_component = 1.0

        # Combine components
        consistency = (success_rate + variance_component) / 2.0

        return min(1.0, max(0.0, consistency))

    def _empty_statistics(self) -> dict[str, Any]:
        """Return empty statistics template."""
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    def add_result(self, result: dict[str, Any]) -> None:
        """Add a single result to buffer.

        Args:
            result: Result dict to add
        """
        if result:
            self.results_buffer.append(result)

    def clear_buffer(self) -> None:
        """Clear results buffer."""
        self.results_buffer.clear()

    def get_buffered_results(self) -> list[dict[str, Any]]:
        """Get all buffered results."""
        return self.results_buffer.copy()
