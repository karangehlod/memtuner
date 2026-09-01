"""Parameter sweep service for configurable parallel optimization.

Extracts decay parameter sweep logic from analyze_command.py into a
reusable service that can handle any parameter sweep with configurable
parallelism.

This service reduces 127 lines of sweep logic by providing clean,
focused API for parallel parameter optimization.
"""

import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SweepResult:
    """Result of testing a single parameter combination."""

    config: tuple[float, float]
    """The parameter combination tested (lambda, threshold)."""

    status: str
    """Status: 'success' or 'failed'."""

    metrics: dict[str, Any] | None = None
    """Performance metrics if successful."""

    elapsed: float | None = None
    """Elapsed time in seconds."""

    error: str | None = None
    """Error message if failed."""


class ParameterSweepService:
    """Sweep over parameter space using parallel workers.

    This service orchestrates parallel testing of parameter configurations,
    with configurable worker count, progress reporting, and error handling.

    Usage:
        sweep_service = ParameterSweepService(max_workers=4)
        results = sweep_service.sweep_decay_configs(
            gold_dataset,
            seed=42,
            lambdas=[0.0, 0.01, 0.05, 0.10, 0.20],
            thresholds=[0.01, 0.15, 0.35],
        )
    """

    def __init__(
        self,
        max_workers: int | None = None,
        logger_instance: Any = None,
    ) -> None:
        """Initialize sweep service.

        Args:
            max_workers: Maximum parallel workers (default: CPU count).
            logger_instance: Logger for output (uses default if None).
        """
        self.max_workers = max_workers or ((os.cpu_count()//2) or 1)
        self.logger = logger_instance or logger

    def sweep_decay_configs(
        self,
        gold_dataset: Any,
        seed: int,
        lambdas: list[float] | None = None,
        thresholds: list[float] | None = None,
        test_func: Callable | None = None,
    ) -> list[SweepResult]:
        """Sweep over decay configurations in parallel.

        Tests all combinations of lambda and threshold values using
        parallel workers for CPU-bound computation.

        Args:
            gold_dataset: The dataset to test with.
            seed: Random seed for reproducibility.
            lambdas: Decay lambda values to test (default: [0.0, 0.01, 0.05, 0.10, 0.20]).
            thresholds: Pruning thresholds to test (default: [0.01, 0.15, 0.35]).
            test_func: Function to test each config (required parameter).

        Returns:
            List of SweepResult objects.

        Raises:
            ValueError: If test_func is not provided.
        """
        if test_func is None:
            raise ValueError("test_func parameter is required for sweep_decay_configs()")

        lambdas = lambdas or [0.0, 0.01, 0.05, 0.10, 0.20]
        thresholds = thresholds or [0.01, 0.15, 0.35]

        # Create all parameter combinations
        configs = [
            (lam, threshold) for lam in lambdas for threshold in thresholds
        ]

        self.logger.info(
            f"Starting parameter sweep: {len(configs)} configs "
            f"({len(lambdas)} lambdas × {len(thresholds)} thresholds) "
            f"with {self.max_workers} workers"
        )

        results = []
        start_sweep = time.monotonic()

        # Execute configs in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {}
            for config in configs:
                future = executor.submit(test_func, config, gold_dataset, seed)
                futures[future] = config

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                config = futures[future]
                try:
                    result_dict = future.result()
                    results.append(
                        SweepResult(
                            config=config,
                            status="success",
                            metrics=result_dict,
                            elapsed=result_dict.get("elapsed"),
                        )
                    )
                    completed += 1

                    # Progress reporting
                    if completed % 3 == 0:
                        self.logger.info(
                            f"   ✓ {completed}/{len(configs)} configs tested"
                        )

                except Exception as exc:
                    self.logger.error(f"Config {config} failed: {exc}")
                    results.append(
                        SweepResult(
                            config=config,
                            status="failed",
                            error=str(exc),
                        )
                    )

        elapsed_sweep = time.monotonic() - start_sweep
        self.logger.info(
            f"Parameter sweep complete: {len(results)} configs tested in {elapsed_sweep:.1f}s"
        )

        return results

    def get_best_config(
        self,
        results: list[SweepResult],
        metric_name: str = "recall",
    ) -> SweepResult | None:
        """Find the parameter combination with best metric.

        Args:
            results: List of sweep results.
            metric_name: Metric to optimize (default: 'recall').

        Returns:
            SweepResult with best metric, or None if no successful results.
        """
        successful = [r for r in results if r.status == "success"]
        if not successful:
            return None

        best = max(
            successful,
            key=lambda r: r.metrics.get(metric_name, 0)
            if r.metrics
            else 0,
        )
        return best

    def get_summary(self, results: list[SweepResult]) -> dict[str, Any]:
        """Get summary statistics from sweep results.

        Args:
            results: List of sweep results.

        Returns:
            Dictionary with summary statistics.
        """
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status == "failed"]

        if not successful:
            return {
                "total": len(results),
                "successful": 0,
                "failed": len(failed),
                "success_rate": 0.0,
            }

        # Aggregate metrics
        metrics_summary = {}
        if successful[0].metrics:
            for metric_name in successful[0].metrics.keys():
                values = [
                    r.metrics[metric_name]
                    for r in successful
                    if r.metrics and metric_name in r.metrics
                ]
                if values:
                    metrics_summary[f"{metric_name}_mean"] = sum(values) / len(
                        values
                    )
                    metrics_summary[f"{metric_name}_max"] = max(values)
                    metrics_summary[f"{metric_name}_min"] = min(values)

        return {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0.0,
            **metrics_summary,
        }

    def log_summary(self, results: list[SweepResult]) -> None:
        """Log sweep results summary.

        Args:
            results: List of sweep results.
        """
        summary = self.get_summary(results)
        success_rate = summary.get("success_rate", 0.0)

        if success_rate == 1.0:
            self.logger.info(
                f"All {summary['successful']} parameter combinations passed"
            )
        else:
            self.logger.warning(
                f"Parameter sweep: {summary['successful']}/{summary['total']} "
                f"successful ({success_rate:.0%})"
            )

        best = self.get_best_config(results)
        if best:
            recall = best.metrics.get("recall", 0) if best.metrics else 0
            self.logger.info(f"Best config: λ={best.config[0]}, τ={best.config[1]}, recall={recall:.1%}")
