"""MemoryReader interface.

Defines the contract for reading/querying memories from a memory module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.models.query import ReadQuery
from benchmark.models.response import ReadResponse


class MemoryReader(ABC):
    """Abstract interface for reading memories.

    Any memory module that supports retrieval must implement this interface.
    Results must be ordered by score in monotonic descending order.

    Raises:
        MemoryReadError: If the read operation fails.
    """

    @abstractmethod
    def read(self, query: ReadQuery) -> ReadResponse:
        """Retrieve memories matching the query.

        Args:
            query: The read query containing search parameters and filters.

        Returns:
            ReadResponse with retrieved memories ordered by score (descending).

        Raises:
            MemoryReadError: If the read fails.
        """
