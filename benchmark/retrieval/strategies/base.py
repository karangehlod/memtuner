"""Abstract base class for retrieval strategy benchmarking.

Each retrieval strategy standardizes the interface for benchmarking a specific
retrieval approach (BM25, dense vectors, hybrid, etc.).

Design principles:
  - Single Responsibility: Each adapter benchmarks one retrieval family
  - Open/Closed: New retrieval types added without modifying existing code
  - Interface Segregation: Minimal, focused interface
  - Dependency Inversion: Consumers depend on abstract RetrievalStrategy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ============================================================================
# Retrieval Metrics
# ============================================================================


@dataclass(frozen=True)
class RetrievalMetrics:
    """Performance metrics for a retrieval strategy."""

    # Recall metrics (0-1 scale)
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    precision_at_10: float = 0.0

    # Efficiency metrics (milliseconds / bytes)
    query_latency_ms: float = 0.0
    index_build_time_sec: float = 0.0
    index_size_bytes: float = 0.0

    # Reliability
    success_rate: float = 1.0
    error_count: int = 0

    # Metadata
    strategy_name: str = ""
    dataset_name: str = ""
    num_queries: int = 0
    num_documents: int = 0
    elapsed_seconds: float = 0.0


# ============================================================================
# Abstract Retrieval Strategy
# ============================================================================


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategy benchmarking.

    All retrieval strategies must implement this interface to enable standardized
    benchmarking across different retrieval families (sparse, dense, hybrid).

    Example:
        >>> strategy = BM25Strategy()
        >>> strategy.initialize(documents)
        >>> results = strategy.search("query", top_k=10)
        >>> metrics = strategy.get_metrics()
        >>> strategy.teardown()
    """

    name: str = ""
    """Unique strategy identifier (e.g., 'bm25', 'dense_vector')."""

    @abstractmethod
    def initialize(self, documents: list[dict[str, Any]]) -> None:
        """Initialize retrieval strategy with documents.

        Args:
            documents: List of documents with 'id' and 'content' fields.

        Raises:
            RuntimeError: If initialization fails.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents using the strategy.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of ranked document results with scores.

        Raises:
            RuntimeError: If search fails.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> RetrievalMetrics:
        """Get current performance metrics.

        Returns:
            RetrievalMetrics with all computed metrics.

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
# Retrieval Strategy Registry
# ============================================================================


class RetrievalStrategyRegistry:
    """Registry for discovering and instantiating retrieval strategies.

    Provides a plugin-like system for benchmarking different retrieval types.
    """

    _strategies: dict[str, type[RetrievalStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: type[RetrievalStrategy]) -> None:
        """Register a retrieval strategy.

        Args:
            name: Unique name for the strategy.
            strategy_cls: The strategy class to register.

        Raises:
            ValueError: If strategy name already registered.
        """
        if name in cls._strategies:
            raise ValueError(f"Strategy '{name}' already registered")
        cls._strategies[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> RetrievalStrategy:
        """Get retrieval strategy instance by name.

        Args:
            name: The strategy name to retrieve.

        Returns:
            New instance of the requested strategy.

        Raises:
            RuntimeError: If strategy name not found.
        """
        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise RuntimeError(
                f"Unknown strategy '{name}'. Available: {available}"
            )
        return cls._strategies[name]()

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered strategy names.

        Returns:
            Sorted list of available strategy names.
        """
        return sorted(cls._strategies.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if strategy is registered.

        Args:
            name: The strategy name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name in cls._strategies
