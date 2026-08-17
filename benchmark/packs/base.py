"""Base interface for benchmark packs.

A BenchmarkPack normalizes an external dataset into the fixed benchmark schema:
- MemoryEvent (write operations)
- ReadQuery (read operations)
- GoldExpectedResult (expected outputs for evaluation)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.gold.schema import GoldDataset


@dataclass(frozen=True)
class PackMetadata:
    """Immutable metadata about a benchmark pack."""

    name: str
    version: str
    description: str
    source_url: str
    license: str
    citation: str
    total_queries: int = 0
    total_sessions: int = 0
    memory_abilities: list[str] = field(default_factory=list)


class BenchmarkPack(ABC):
    """Abstract base class for all benchmark packs.

    A pack provides:
    1. Metadata (name, version, source, license)
    2. Data loading (from local files or download)
    3. Schema adaptation (source format → benchmark schema)
    4. Gold dataset generation (GoldDataset compatible output)
    """

    @abstractmethod
    def metadata(self) -> PackMetadata:
        """Return pack metadata."""

    @abstractmethod
    def load(self, data_dir: Path) -> None:
        """Load raw dataset from data directory.

        Args:
            data_dir: Directory containing the raw dataset files.

        Raises:
            FileNotFoundError: If required data files are missing.
            ValueError: If data format is invalid.
        """

    @abstractmethod
    def to_gold_dataset(
        self,
        *,
        max_queries: int | None = None,
        seed: int = 42,
        evaluation_horizon: int | None = None,
    ) -> GoldDataset:
        """Convert loaded data into benchmark GoldDataset format.

        Args:
            max_queries: Maximum number of queries to include (None = all).
            seed: Random seed for reproducibility.
            evaluation_horizon: Number of dataset days to spread events across.

        Returns:
            GoldDataset compatible with the benchmark evaluation engine.
        """

    @abstractmethod
    def download_instructions(self) -> str:
        """Return human-readable download instructions."""

    def validate_data(self, data_dir: Path) -> bool:
        """Check if required data files exist in directory.

        Args:
            data_dir: Directory to check.

        Returns:
            True if all required files are present.
        """
        for required_file in self.required_files():
            if not (data_dir / required_file).exists():
                return False
        return True

    @abstractmethod
    def required_files(self) -> list[str]:
        """List of required file names in the data directory."""
