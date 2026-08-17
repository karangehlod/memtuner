"""Retrieval strategy interface for pluggable memory retrieval.

Allows swapping between different retrieval algorithms:
- BM25 (keyword matching)
- Embeddings (semantic similarity)
- LLM-based (using GPT-4, Claude)
- Database (PostgreSQL with pgvector)
- Hybrid (combines multiple strategies)
"""

from abc import ABC, abstractmethod

from benchmark.models.memory_event import MemoryEvent


class RetrievalStrategy(ABC):
    """Abstract base for memory retrieval strategies."""

    @abstractmethod
    def index(self, memories: list[MemoryEvent]) -> None:
        """Build/update index from memories.

        Args:
            memories: List of memory events to index.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve relevant memories for a query.

        Args:
            query: The query text.
            top_k: Number of results to return.
            user_id: Optional user filter.

        Returns:
            List of (memory_id, score) tuples, ordered by relevance.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return strategy name for logging.

        Returns:
            Strategy identifier (e.g., "bm25", "embeddings", "llm").
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed data."""
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if strategy dependencies are installed.

        Returns:
            True if all required dependencies are available.
        """
        return True
