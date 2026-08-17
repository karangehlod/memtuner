"""Default configuration values.

Provides factory methods for creating default configs.
"""

from __future__ import annotations

from benchmark.config.schema import BenchmarkConfig


def create_default_config() -> BenchmarkConfig:
    """Create a default benchmark configuration with sensible defaults.

    Returns:
        A BenchmarkConfig with all default values applied.
    """
    return BenchmarkConfig()
