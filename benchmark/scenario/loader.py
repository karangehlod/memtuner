"""Gold-dataset-backed scenario implementation.

Loads a scenario from a GoldDataset and exposes it through the BenchmarkScenario interface.
"""

from __future__ import annotations

from benchmark.gold.schema import GoldDataset, GoldDayEvents, GoldQuery
from benchmark.scenario.base import BenchmarkScenario


class GoldDatasetScenario(BenchmarkScenario):
    """A benchmark scenario backed by a gold dataset.

    Wraps a GoldDataset and provides day-by-day access to events and queries.

    test_frac=0.2 reserves the last 20% of days for evaluation only.
    Events on those held-out days are never injected into the memory system;
    only queries on those days are used for scoring.  Days before the split
    are used for indexing (write phase) but their queries are excluded.
    """

    def __init__(
        self,
        dataset: GoldDataset,
        evaluation_horizon: int | None = None,
        test_frac: float = 0.0,
    ) -> None:
        """Initialize from a gold dataset.

        Args:
            dataset: The gold dataset defining this scenario.
            evaluation_horizon: Override for total days. If None, uses max day from dataset.
            test_frac: Fraction of days (from the end) to hold out as a test split.
                Events on held-out days are not injected; only their queries are evaluated.
                Must be in [0.0, 1.0).  Default 0.0 disables the split (original behaviour).
        """
        self._dataset = dataset
        # Merge multiple GoldDayEvents that share the same day number.
        # This prevents data loss when normalization collapses day ranges.
        self._events_by_day: dict[int, GoldDayEvents] = {}
        for day_events in dataset.events:
            if day_events.day in self._events_by_day:
                existing = self._events_by_day[day_events.day]
                merged_memories = list(existing.memory_events) + list(day_events.memory_events)
                self._events_by_day[day_events.day] = GoldDayEvents(
                    day=day_events.day, memory_events=merged_memories
                )
            else:
                self._events_by_day[day_events.day] = day_events

        self._queries_by_day: dict[int, list[GoldQuery]] = {}
        for query in dataset.queries:
            if query.day not in self._queries_by_day:
                self._queries_by_day[query.day] = []
            self._queries_by_day[query.day].append(query)

        if evaluation_horizon is not None:
            self._total_days = evaluation_horizon
        else:
            all_days = set(self._events_by_day.keys()) | set(self._queries_by_day.keys())
            self._total_days = max(all_days) + 1 if all_days else 1

        if not (0.0 <= test_frac < 1.0):
            raise ValueError(
                f"test_frac must be in [0.0, 1.0), got {test_frac}. "
                "test_frac=1.0 would hold out all events leaving nothing to index."
            )
        self._split_day: int | None = (
            int(self._total_days * (1 - test_frac)) if test_frac > 0.0 else None
        )

    def name(self) -> str:
        """Return the scenario name from the gold dataset.

        Returns:
            The scenario name string.
        """
        return self._dataset.scenario

    def description(self) -> str:
        """Return the scenario description from the gold dataset.

        Returns:
            Human-readable description.
        """
        return self._dataset.description

    def get_events_for_day(self, day: int) -> GoldDayEvents | None:
        """Get memory events for a specific day.

        When a test split is active, days >= split_day return an empty
        GoldDayEvents so those events are never injected into the memory system.

        Args:
            day: The simulated day number.

        Returns:
            GoldDayEvents or None.
        """
        if self._split_day is not None and day >= self._split_day:
            return GoldDayEvents(day=day, memory_events=[])
        return self._events_by_day.get(day)

    def get_queries_for_day(self, day: int) -> list[GoldQuery]:
        """Get queries for a specific day.

        When a test split is active, days < split_day return no queries so only
        the held-out tail of the timeline contributes to evaluation scores.

        Args:
            day: The simulated day number.

        Returns:
            List of queries (may be empty).
        """
        if self._split_day is not None and day < self._split_day:
            return []
        return self._queries_by_day.get(day, [])

    @property
    def test_frac_applied(self) -> float:
        """Return the effective held-out fraction.

        Returns:
            0.0 when no split is active, otherwise
            (total_days - split_day) / total_days.
        """
        if self._split_day is None:
            return 0.0
        return (self._total_days - self._split_day) / self._total_days

    def total_days(self) -> int:
        """Return total dataset days in the evaluation horizon.

        Returns:
            Number of dataset days to replay.
        """
        return self._total_days

    def recall_k(self) -> int:
        """Return K for Recall@K from evaluation criteria.

        Returns:
            The K value.
        """
        return self._dataset.evaluation_criteria.recall_k
