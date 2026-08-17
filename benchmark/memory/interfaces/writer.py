"""MemoryWriter interface.

Defines the contract for writing memory events into a memory module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.models.memory_event import MemoryEvent


class MemoryWriter(ABC):
    """Abstract interface for writing memory events.

    Any memory module that accepts write operations must implement this interface.
    Implementations must be deterministic given the same input sequence.

    Raises:
        MemoryWriteError: If the write operation fails.
    """

    @abstractmethod
    def write(self, event: MemoryEvent) -> None:
        """Write a memory event into the store.

        Args:
            event: The memory event to store. Must not be modified by the implementation.

        Raises:
            MemoryWriteError: If the write fails.
        """
