"""Custom exception hierarchy for the benchmark tool.

All exceptions inherit from BenchmarkError to enable
catch-all handling at the CLI boundary.
"""

from benchmark.exceptions.config_errors import (
    ConfigLoadError,
    ConfigValidationError,
)
from benchmark.exceptions.evaluation_errors import (
    EvaluationError,
    GoldDatasetError,
    MetricComputationError,
)
from benchmark.exceptions.memory_errors import (
    MemoryReadError,
    MemoryWriteError,
    RegistryResolutionError,
)

__all__ = [
    "ConfigLoadError",
    "ConfigValidationError",
    "EvaluationError",
    "GoldDatasetError",
    "MemoryReadError",
    "MemoryWriteError",
    "MetricComputationError",
    "RegistryResolutionError",
]
