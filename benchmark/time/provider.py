"""TimeProvider interface.

Defines the contract for time abstractions used throughout the benchmark.
No concrete time.time() calls should exist outside this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class TimeProvider(ABC):
    """Abstract interface for time operations in the benchmark.

    All time-dependent logic must use this interface instead of
    direct calls to time.time(), datetime.now(), etc.
    This ensures deterministic benchmark execution.
    """

    @abstractmethod
    def current_day(self) -> int:
        """Return the current simulated day.

        Returns:
            The current simulated day number (0-indexed).
        """

    @abstractmethod
    def current_timestamp(self) -> datetime:
        """Return the current simulated timestamp.

        Returns:
            A datetime representing the current simulated point in time.
        """

    @abstractmethod
    def advance_day(self) -> None:
        """Advance the simulated clock by one day.

        After calling this method, current_day() returns the next day
        and current_timestamp() reflects the new day.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the clock back to day 0.

        Used between scenario runs to ensure clean state.
        """
