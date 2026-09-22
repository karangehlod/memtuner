"""Gold-dataset-backed scenario implementation.

Loads a scenario from a GoldDataset and exposes it through the BenchmarkScenario interface.
"""

from __future__ import annotations

import logging

from benchmark.gold.schema import GoldDataset, GoldDayEvents, GoldQuery
from benchmark.scenario.base import BenchmarkScenario

logger = logging.getLogger(__name__)


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
        query_start_fraction: float = 0.0,
        query_end_fraction: float = 1.0,
    ) -> None:
        """Initialize from a gold dataset.

        Args:
            dataset: The gold dataset defining this scenario.
            evaluation_horizon: Override for total days. If None, uses max day from dataset.
            test_frac: Fraction of days (from the end) to hold out as a test split.
                Events on held-out days are not injected; only their queries are evaluated.
                Must be in [0.0, 1.0).  Default 0.0 disables the split (original behaviour).
            query_start_fraction: Inclusive fraction of the replay horizon at which queries
                are scored. Defaults to the start of the holdout interval.
            query_end_fraction: Exclusive fraction of the replay horizon at which queries
                stop being scored. Defaults to the end of the replay horizon.
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

        all_days = set(self._events_by_day.keys()) | set(self._queries_by_day.keys())
        natural_span = max(all_days) + 1 if all_days else 1
        if evaluation_horizon is not None:
            self._total_days = evaluation_horizon
        else:
            self._total_days = natural_span

        if not (0.0 <= test_frac < 1.0):
            raise ValueError(
                f"test_frac must be in [0.0, 1.0), got {test_frac}. "
                "test_frac=1.0 would hold out all events leaving nothing to index."
            )
        # The split and query window are fractions of the days that actually
        # contain data, never of a padded horizon. Otherwise a horizon larger
        # than the dataset (e.g. --evaluation-horizon 50 on a 30-day dataset)
        # pushes the holdout window past every query and the run silently
        # scores nothing.
        self._split_basis = min(self._total_days, natural_span)
        self._split_day: int | None = (
            int(self._split_basis * (1 - test_frac)) if test_frac > 0.0 else None
        )
        if not (0.0 <= query_start_fraction < query_end_fraction <= 1.0):
            raise ValueError(
                "query window must satisfy 0.0 <= start < end <= 1.0, got "
                f"start={query_start_fraction}, end={query_end_fraction}"
            )
        default_query_start = 1.0 - test_frac if test_frac > 0.0 else 0.0
        self._query_start_day = int(self._split_basis * query_start_fraction)
        self._query_end_day = int(self._split_basis * query_end_fraction)
        if query_start_fraction == 0.0 and test_frac > 0.0:
            self._query_start_day = int(self._split_basis * default_query_start)
        # Query days are compared with an exclusive end bound; when the window
        # ends at the split basis, let it cover the final data day too.
        if query_end_fraction == 1.0:
            self._query_end_day = self._total_days

        self._warn_if_holdout_unscorable()

    def _warn_if_holdout_unscorable(self) -> None:
        """Warn when the holdout/query window cannot produce a meaningful score.

        Two failure modes, both of which previously surfaced only as a silent
        recall=0.0:
        - the query window contains no queries at all;
        - every scored query's gold memories live on held-out days, so the
          correct answers are never injected (typical for converted QA datasets
          that place each query on the same day as its source passage).
        """
        scored = [
            q
            for day, queries in self._queries_by_day.items()
            if self._query_start_day <= day < self._query_end_day
            for q in queries
        ]
        total = sum(len(v) for v in self._queries_by_day.values())
        if total and not scored:
            logger.warning(
                "Holdout query window [day %d, %d) contains none of the dataset's "
                "%d queries — every metric will be 0.0. Check the evaluation "
                "horizon against the dataset's day span.",
                self._query_start_day,
                self._query_end_day,
                total,
            )
            return
        if self._split_day is None or not scored:
            return
        mem_day = {
            mem.id: day_events.day
            for day_events in self._events_by_day.values()
            for mem in day_events.memory_events
        }
        unreachable = sum(
            1
            for q in scored
            if q.expected.memory_ids
            and all(mem_day.get(mid, 0) >= self._split_day for mid in q.expected.memory_ids)
        )
        if unreachable == len(scored):
            logger.warning(
                "All %d scored holdout queries reference only memories on held-out "
                "days (>= day %d) that are never injected — recall is structurally "
                "0.0. The day-based holdout cannot evaluate this dataset; use "
                "test_frac=0 with a query-level holdout instead.",
                len(scored),
                self._split_day,
            )

    def active_days(self) -> list[int]:
        """Return sorted list of days that have events or queries.

        Used by the scenario runner to skip empty days and avoid iterating
        the full day range when only a small fraction of days have data
        (e.g. LoCoMo: 9 active days out of 722 total → 99% of iterations are empty).
        """
        return sorted(set(self._events_by_day.keys()) | set(self._queries_by_day.keys()))

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
            # Return None rather than GoldDayEvents(memory_events=[]) — the Pydantic
            # model enforces min_length=1 on memory_events and would raise ValidationError.
            # None signals "no injection on this held-out day" to the scenario runner.
            return None
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
        if day < self._query_start_day or day >= self._query_end_day:
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
        return (self._split_basis - self._split_day) / self._split_basis

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
