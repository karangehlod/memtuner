"""Simulated clock implementation.

Provides deterministic time progression for benchmark scenarios.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from benchmark.time.provider import TimeProvider


class SimulatedClock(TimeProvider):
    """Deterministic simulated clock for benchmark execution.

    Starts at a fixed epoch and advances day-by-day.
    No dependency on system time — fully deterministic.

    Attributes:
        _epoch: The fixed starting timestamp.
        _current_day: The current simulated day number.
    """

    def __init__(self, epoch: datetime | None = None) -> None:
        """Initialize the simulated clock.

        Args:
            epoch: The starting timestamp. Defaults to 2026-01-01T00:00:00Z.
        """
        self._epoch = epoch or datetime(2026, 1, 1, tzinfo=UTC)
        self._current_day = 0

    def current_day(self) -> int:
        """Return the current simulated day.

        Returns:
            The current simulated day number (0-indexed).
        """
        return self._current_day

    def current_timestamp(self) -> datetime:
        """Return the current simulated timestamp.

        Returns:
            The epoch plus the current number of simulated days.
        """
        return self._epoch + timedelta(days=self._current_day)

    def advance_day(self) -> None:
        """Advance the simulated clock by one day."""
        self._current_day += 1

    def reset(self) -> None:
        """Reset the clock back to day 0."""
        self._current_day = 0
