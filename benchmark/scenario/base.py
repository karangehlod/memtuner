"""Scenario base interface.

Defines the contract for benchmark scenarios.
Scenarios are data + sequencing — they do NOT contain evaluation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.gold.schema import GoldDayEvents, GoldQuery


class BenchmarkScenario(ABC):
    """Abstract interface for benchmark scenarios.

    A scenario defines:
    - What memory events to inject and when
    - What queries to execute and when
    - Gold references for evaluation (by ID only)

    Scenarios are interchangeable (LSP) and define no evaluation logic.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this scenario.

        Returns:
            The scenario name string.
        """

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description.

        Returns:
            Description of what this scenario tests.
        """

    @abstractmethod
    def get_events_for_day(self, day: int) -> GoldDayEvents | None:
        """Get memory events to inject on a given simulated day.

        Args:
            day: The simulated day number.

        Returns:
            GoldDayEvents for the day, or None if no events on this day.
        """

    @abstractmethod
    def get_queries_for_day(self, day: int) -> list[GoldQuery]:
        """Get queries to execute on a given simulated day.

        Args:
            day: The simulated day number.

        Returns:
            List of GoldQuery objects for the day (may be empty).
        """

    @abstractmethod
    def total_days(self) -> int:
        """Return the total number of simulated days for this scenario.

        Returns:
            Number of simulated days.
        """

    @abstractmethod
    def recall_k(self) -> int:
        """Return the K value for Recall@K evaluation.

        Returns:
            The K value.
        """
