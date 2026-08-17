"""Protocol definitions for optional capabilities in memory modules.

These protocols enable type-safe structural typing for capabilities that are
not required by all implementations but can be checked at runtime.

Usage:
    if isinstance(module, LifecycleAwareWriter):
        module.write_on_day(event, day)  # Type-safe, IDE-aware
"""

from typing import Protocol, runtime_checkable

from benchmark.models.memory_event import MemoryEvent


@runtime_checkable
class LifecycleAwareWriter(Protocol):
    """Memory writer supporting day-aware batch processing.

    Implementations that support this can write memories with explicit
    day information, enabling batch processing and temporal logic.
    """

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Write a memory event with explicit day information.

        Args:
            event: The memory event to write.
            day: The simulated day this event occurred on.
        """
        ...


@runtime_checkable
class CreationDayTracker(Protocol):
    """Memory provider tracking creation day per memory ID.

    Implementations that support this can return the creation day
    for each memory, enabling temporal filtering and analysis.
    """

    def get_creation_day(self, memory_id: str) -> int:
        """Get the creation day for a specific memory.

        Args:
            memory_id: The ID of the memory.

        Returns:
            The simulated day this memory was created on.
        """
        ...


@runtime_checkable
class MemoryScoreComputer(Protocol):
    """Strategy that can compute relevance scores for memories.

    Implementations that support this can compute custom scores
    for memories, enabling advanced ranking and filtering.
    """

    def compute_scores(self, memory_ids: list[str]) -> dict[str, float]:
        """Compute relevance scores for memories.

        Args:
            memory_ids: List of memory IDs to score.

        Returns:
            Mapping of memory ID to score (0-1 range).
        """
        ...


@runtime_checkable
class StrategyContaining(Protocol):
    """Object that contains a retrieval strategy.

    Implementations that support this provide direct access to
    their underlying retrieval strategy for advanced operations.
    """

    @property
    def strategy(self) -> "RetrievalStrategy":  # noqa: F821
        """Get the underlying retrieval strategy.

        Returns:
            The RetrievalStrategy used by this module.
        """
        ...


@runtime_checkable
class BulkOperationSupport(Protocol):
    """Memory module supporting bulk operations.

    Implementations that support this can handle multiple
    operations in a single batch for improved performance.
    """

    def write_batch(self, events: list[MemoryEvent]) -> None:
        """Write multiple memory events in a single batch.

        Args:
            events: List of memory events to write.
        """
        ...

    def delete_batch(self, memory_ids: list[str]) -> None:
        """Delete multiple memories in a single batch.

        Args:
            memory_ids: List of memory IDs to delete.
        """
        ...
