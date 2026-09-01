"""Dataset adapter framework for plug-and-play dataset support.

This module provides a registry-based system for loading, validating, and
analyzing different dataset formats through a unified interface.

Architecture:
  - adapter.py: Abstract base class and interfaces
  - [name]_adapter.py: Concrete implementations (LoCoMo, LongMemEval, Synthetic)
  - __init__.py: Registry and exports (this file)

Usage:
    >>> from benchmark.gold.adapters import AdapterRegistry
    >>> adapter = AdapterRegistry.get("locomo")
    >>> dataset = adapter.load("data/locomo10.json")
    >>> validation = adapter.validate(dataset)
    >>> stats = adapter.statistics(dataset)
"""

from benchmark.gold.adapters.adapter import (
    AdapterError,
    AdapterRegistry,
    DatasetAdapter,
    FingerprintError,
    MetadataError,
    StatisticsError,
    ValidationError,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from benchmark.gold.statistics import DatasetStatistics

__all__ = [
    # Exceptions
    "AdapterError",
    "AdapterRegistry",
    # Core interfaces
    "DatasetAdapter",
    "DatasetStatistics",
    "FingerprintError",
    "MetadataError",
    "StatisticsError",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    # Validation
    "ValidationSeverity",
]
