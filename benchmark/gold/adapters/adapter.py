"""Dataset adapter abstraction for plug-and-play dataset support.

This module defines the abstract interface that all dataset adapters must implement,
enabling a standardized way to load, validate, and analyze different dataset formats.

Follows:
  - SOLID principles (Interface Segregation, Dependency Inversion)
  - Protocol-based design for extensibility
  - Type safety with 100% type hints
  - Comprehensive error handling with specific exception types
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from benchmark.gold.schema import GoldDataset


# ============================================================================
# Exception Types
# ============================================================================


class AdapterError(Exception):
    """Base exception for adapter-related errors."""

    pass


class ValidationError(AdapterError):
    """Raised when dataset validation fails."""

    pass


class FingerprintError(AdapterError):
    """Raised when fingerprint computation fails."""

    pass


class StatisticsError(AdapterError):
    """Raised when statistics computation fails."""

    pass


class MetadataError(AdapterError):
    """Raised when metadata retrieval fails."""

    pass


# ============================================================================
# Validation & Reporting
# ============================================================================


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Must be fixed, dataset unusable
    WARNING = "warning"  # Should be fixed, dataset may be incomplete
    INFO = "info"  # Informational, no action required


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation issue found during dataset validation.

    Attributes:
        severity: How critical this issue is (ERROR, WARNING, INFO)
        validator: Name of the validator that found this issue (e.g., "schema", "temporal")
        message: Human-readable description of the issue
        location: Where in the dataset the issue occurred (e.g., "query[5].user_id")
    """

    severity: ValidationSeverity
    validator: str
    message: str
    location: str = ""

    def __str__(self) -> str:
        """Format as readable error message."""
        loc = f" ({self.location})" if self.location else ""
        return f"[{self.severity.value.upper()}] {self.validator}: {self.message}{loc}"


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report for a dataset.

    Attributes:
        passed: Whether validation succeeded (no ERROR-level issues)
        issues: All issues found (ERRORs, WARNINGs, INFOs)
    """

    passed: bool
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only ERROR-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only WARNING-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def info_messages(self) -> list[ValidationIssue]:
        """Get only INFO-level messages."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def __str__(self) -> str:
        """Format as readable report."""
        if not self.issues:
            return "✓ Validation passed (no issues)"

        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  {e}")
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  {w}")
        if self.info_messages:
            lines.append(f"\nInfo ({len(self.info_messages)}):")
            for info in self.info_messages:
                lines.append(f"  {info}")

        return "\n".join(lines)


# ============================================================================
# Abstract Adapter Interface
# ============================================================================


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    All dataset adapters must implement this interface to enable standardized
    loading, validation, and analysis of different dataset formats.

    The adapter pattern allows each dataset format to define its own loading logic
    and transformation rules while maintaining a uniform interface for consumers.

    Design principles:
      - Single Responsibility: Each adapter handles one dataset format
      - Open/Closed: New formats can be added by creating new adapters
      - Interface Segregation: Methods are focused and single-purpose
      - Dependency Inversion: Consumers depend on this interface, not concrete adapters
    """

    @abstractmethod
    def load(self, source: Path | str) -> GoldDataset:
        """Load dataset from source path or identifier.

        Transforms the dataset from its native format to the standardized
        GoldDataset schema used by the benchmark framework.

        Args:
            source: File path or identifier for the dataset source.
                   Can be absolute path, relative path, or dataset identifier.

        Returns:
            Loaded and normalized GoldDataset instance.

        Raises:
            AdapterError: If the dataset cannot be loaded or parsed.
            ValidationError: If the loaded data fails basic validation.

        Examples:
            >>> adapter = LoCoMoAdapter()
            >>> dataset = adapter.load("data/locomo10.json")
            >>> print(f"Loaded {len(dataset.queries)} queries")
        """
        pass

    @abstractmethod
    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate dataset structure and contents.

        Performs comprehensive validation including schema checks, temporal
        ordering, referential integrity, and format-specific constraints.

        Args:
            dataset: The dataset to validate.

        Returns:
            ValidationReport with all issues found (if any).

        Raises:
            AdapterError: If validation cannot be performed (e.g., configuration missing).

        Examples:
            >>> report = adapter.validate(dataset)
            >>> if not report.passed:
            ...     for error in report.errors:
            ...         print(f"ERROR: {error}")
        """
        pass

    @abstractmethod
    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint for dataset.

        The fingerprint must be:
          - Deterministic: same input → same output, always
          - Version-aware: schema changes → different fingerprint
          - Stable: doesn't change between runs
          - Content-based: reflects dataset contents

        The fingerprint enables:
          - Dataset deduplication
          - Change detection
          - Reproducibility verification
          - Artifact versioning

        Args:
            dataset: The dataset to fingerprint.

        Returns:
            32-character hex string (SHA256 based) uniquely identifying the dataset.

        Raises:
            FingerprintError: If fingerprint computation fails.

        Examples:
            >>> fp1 = adapter.fingerprint(dataset)
            >>> fp2 = adapter.fingerprint(dataset)
            >>> assert fp1 == fp2  # Deterministic
        """
        pass

    @abstractmethod
    def statistics(self, dataset: GoldDataset) -> "DatasetStatistics":
        """Compute dataset statistics.

        Computes metrics including:
          - Counts: queries, memories, users
          - Temporal: coverage, distribution
          - Quality: diversity, density
          - Distribution: per-user, per-day stats

        Statistics enable:
          - Dataset characterization
          - Benchmark difficulty assessment
          - Cross-dataset comparison
          - Quality validation

        Args:
            dataset: The dataset to analyze.

        Returns:
            DatasetStatistics object with all computed metrics.

        Raises:
            StatisticsError: If statistics computation fails.

        Examples:
            >>> stats = adapter.statistics(dataset)
            >>> print(f"Coverage: {stats.day_range} days")
            >>> print(f"Density: {stats.memory_density:.2f}")
        """
        pass

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return dataset metadata.

        Metadata includes immutable, versioned information about the dataset:
          - Name and version
          - Description and source
          - Schema version
          - Creation date (for adapters with this info)

        Metadata is distinct from statistics:
          - Metadata: immutable, versioned, stored
          - Statistics: computed on-demand, includes distributions

        Returns:
            Dictionary with dataset metadata (name, version, description, source).

        Raises:
            MetadataError: If metadata cannot be retrieved.

        Examples:
            >>> metadata = adapter.metadata()
            >>> print(f"Dataset: {metadata['name']} v{metadata['version']}")
        """
        pass


# Note: DatasetStatistics is imported from benchmark.gold.statistics
# to avoid circular imports. See adapters/__init__.py for the import.


# ============================================================================
# Adapter Registry (basic version - full in __init__.py)
# ============================================================================


class AdapterRegistry:
    """Registry for discovering and instantiating dataset adapters.

    Provides a plugin-like system for adding new dataset formats:
      - register: Add a new adapter class
      - get: Retrieve adapter instance by name
      - list_all: See all available adapters

    This enables:
      - Dynamic adapter discovery
      - Extensibility without modifying core code
      - Decoupling consumers from concrete adapters
    """

    _adapters: dict[str, type[DatasetAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[DatasetAdapter]) -> None:
        """Register a new adapter.

        Args:
            name: Unique name for the adapter (e.g., "locomo", "longmemeval")
            adapter_cls: The adapter class to register

        Raises:
            ValueError: If an adapter with this name is already registered
        """
        if name in cls._adapters:
            raise ValueError(
                f"Adapter '{name}' is already registered. "
                f"Use a different name or unregister the existing adapter."
            )
        cls._adapters[name] = adapter_cls

    @classmethod
    def get(cls, name: str) -> DatasetAdapter:
        """Get adapter instance by name.

        Args:
            name: The adapter name to retrieve

        Returns:
            New instance of the requested adapter

        Raises:
            AdapterError: If the adapter name is not found

        Examples:
            >>> adapter = AdapterRegistry.get("locomo")
            >>> dataset = adapter.load("data/locomo10.json")
        """
        if name not in cls._adapters:
            available = ", ".join(sorted(cls._adapters.keys()))
            raise AdapterError(
                f"Unknown adapter '{name}'. Available adapters: {available}"
            )
        return cls._adapters[name]()

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered adapter names.

        Returns:
            Sorted list of available adapter names
        """
        return sorted(cls._adapters.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an adapter is registered.

        Args:
            name: The adapter name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._adapters
