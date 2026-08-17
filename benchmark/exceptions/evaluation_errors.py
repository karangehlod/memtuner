"""Evaluation-related exceptions.

Raised during metric computation, gold dataset loading, or evaluation operations.
"""

from __future__ import annotations

from benchmark.exceptions.config_errors import BenchmarkError


class EvaluationError(BenchmarkError):
    """Raised when an evaluation operation fails.

    Examples:
        - Metric computation returns invalid result
        - Evaluation preconditions not met
    """


class GoldDatasetError(BenchmarkError):
    """Raised when a gold dataset cannot be loaded or is invalid.

    Examples:
        - Dataset file not found
        - Schema validation failure
        - Corrupt or incomplete dataset
    """


class MetricComputationError(BenchmarkError):
    """Raised when a specific metric computation fails.

    Examples:
        - Division by zero in metric formula
        - Invalid input data for metric
    """
