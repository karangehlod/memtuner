"""Abstract base class for memory module benchmarking adapters.

Each memory adapter standardizes the interface for benchmarking a specific
memory module family (episodic store, semantic store, buffer, etc.).

Design principles:
  - Single Responsibility: Each adapter benchmarks one memory type
  - Open/Closed: New memory types added without modifying existing code
  - Interface Segregation: Minimal, focused interface
  - Dependency Inversion: Consumers depend on abstract MemoryAdapter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# Memory Metrics
# ============================================================================


@dataclass(frozen=True)
class MemoryMetrics:
    """Frozen snapshot of performance metrics for a single memory adapter run.

    Fields
    ------
    Accuracy (dimensionless, range 0–1):
      recall_at_1          Fraction of queries where the relevant item ranks #1
      recall_at_5          Fraction of queries where the relevant item is in top 5
      recall_at_10         Fraction of queries where the relevant item is in top 10
      recall_at_100        Fraction of queries where the relevant item is in top 100
      mrr                  Mean Reciprocal Rank — mean(1 / rank) across queries
      ndcg                 Normalized Discounted Cumulative Gain

    Efficiency:
      write_latency_ms     Mean wall-clock time per write_memory() call (ms)
      query_latency_ms     Mean wall-clock time per query_memories() call (ms)
      query_latency_p50_ms Median query latency = percentile(latencies, 50) (ms)
      query_latency_p95_ms 95th-pct query latency = percentile(latencies, 95) (ms)
      storage_bytes        Total bytes used by the memory store (bytes)
      index_build_ms       Wall-clock time to write all memories (ms)

    System resources (sampled by _sys_metrics helpers at get_metrics() time):
      peak_rss_mb          Lifetime peak RSS — source: _sys_metrics.peak_rss_mb() (MiB)
      cpu_percent          Mean CPU% during query phase — mean(_sys_metrics.cpu_percent_snapshot()) (%)

    Reliability:
      success_rate         Fraction of operations that succeeded (0–1)
      error_count          Absolute count of failed operations (count)

    Metadata:
      dataset_name         Name of the benchmark dataset (e.g. "ms_marco_small")
      num_memories         Number of memories written during the run (count)
      num_queries          Number of queries issued during the run (count)
      elapsed_seconds      Total wall-clock duration of the benchmark run (s)
    """

    # Accuracy metrics (0-1 scale)
    recall_at_1: float = 0.0       # dimensionless [0,1]; hits_at_k(k=1) / num_queries
    recall_at_5: float = 0.0       # dimensionless [0,1]; hits_at_k(k=5) / num_queries
    recall_at_10: float = 0.0      # dimensionless [0,1]; hits_at_k(k=10) / num_queries
    recall_at_100: float = 0.0     # dimensionless [0,1]; hits_at_k(k=100) / num_queries
    mrr: float = 0.0               # dimensionless [0,1]; Mean Reciprocal Rank = mean(1/rank)
    ndcg: float = 0.0              # dimensionless [0,1]; Normalized Discounted Cumulative Gain

    # Efficiency metrics (milliseconds / bytes)
    write_latency_ms: float = 0.0       # ms; mean wall-clock time per write_memory() call
    query_latency_ms: float = 0.0       # ms; mean wall-clock time per query_memories() call
    query_latency_p50_ms: float = 0.0   # ms; median query latency = percentile(latencies, 50)
    query_latency_p95_ms: float = 0.0   # ms; 95th-pct query latency = percentile(latencies, 95)
    storage_bytes: float = 0.0          # bytes; total size of the in-memory or on-disk store
    index_build_ms: float = 0.0         # ms; total wall-clock time to write all memories

    # System resource usage (sampled during the run)
    peak_rss_mb: float = 0.0       # MiB; lifetime peak RSS — source: _sys_metrics.peak_rss_mb()
    cpu_percent: float = 0.0       # %; mean CPU% during query phase — mean(_sys_metrics.cpu_percent_snapshot())

    # Reliability
    success_rate: float = 1.0      # dimensionless [0,1]; successful_ops / total_ops
    error_count: int = 0           # count; absolute number of failed operations

    # Metadata
    dataset_name: str = ""         # str; name of the benchmark dataset
    num_memories: int = 0          # count; number of memories written during the run
    num_queries: int = 0           # count; number of queries issued during the run
    elapsed_seconds: float = 0.0   # s; total wall-clock duration of the benchmark run


# ============================================================================
# Abstract Memory Adapter
# ============================================================================


class MemoryAdapter(ABC):
    """Abstract base class for memory module benchmarking.

    All memory adapters must implement this interface to enable standardized
    benchmarking across different memory architecture families.

    Example:
        >>> adapter = EpisodicStoreAdapter()
        >>> adapter.initialize(config)
        >>> for memory in memories:
        ...     adapter.write_memory(memory)
        >>> results = adapter.query_memories(query)
        >>> metrics = adapter.get_metrics()
        >>> adapter.teardown()
    """

    name: str = ""
    """Unique adapter identifier (e.g., 'episodic_store')."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize memory module with configuration.

        Args:
            config: Configuration dictionary with module-specific parameters.
                   Should include parameters like capacity, eviction policy, etc.

        Raises:
            RuntimeError: If initialization fails.
        """
        pass

    @abstractmethod
    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write a memory event to the module.

        Args:
            memory: Memory object with id, content, importance, etc.

        Raises:
            RuntimeError: If write fails.
        """
        pass

    @abstractmethod
    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query memories and return ranked results.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of ranked memory results with scores.

        Raises:
            RuntimeError: If query fails.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> MemoryMetrics:
        """Get current performance metrics.

        Returns:
            MemoryMetrics with all computed metrics.

        Raises:
            RuntimeError: If metrics cannot be computed.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources.

        Should be called when benchmarking is complete to free memory,
        close connections, etc.
        """
        pass


# ============================================================================
# Memory Adapter Registry
# ============================================================================


class MemoryRegistry:
    """Registry for discovering and instantiating memory adapters.

    Provides a plugin-like system for benchmarking different memory types.
    """

    _adapters: dict[str, type[MemoryAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[MemoryAdapter]) -> None:
        """Register a memory adapter.

        Args:
            name: Unique name for the adapter.
            adapter_cls: The adapter class to register.

        Raises:
            ValueError: If adapter name already registered.
        """
        if name in cls._adapters:
            raise ValueError(f"Adapter '{name}' already registered")
        cls._adapters[name] = adapter_cls

    @classmethod
    def get(cls, name: str) -> MemoryAdapter:
        """Get memory adapter instance by name.

        Args:
            name: The adapter name to retrieve.

        Returns:
            New instance of the requested adapter.

        Raises:
            RuntimeError: If adapter name not found.
        """
        if name not in cls._adapters:
            available = ", ".join(sorted(cls._adapters.keys()))
            raise RuntimeError(
                f"Unknown adapter '{name}'. Available: {available}"
            )
        return cls._adapters[name]()

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered adapter names.

        Returns:
            Sorted list of available adapter names.
        """
        return sorted(cls._adapters.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if adapter is registered.

        Args:
            name: The adapter name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in cls._adapters
