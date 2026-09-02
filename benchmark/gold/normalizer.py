"""Dataset timestamp normalizer.

Shifts memory event days forward so that memories are genuinely recent when
queries run, enabling decay functions to produce meaningful differentiation
between old and new memories.

Background
----------
LoCoMo and similar long-span datasets are recorded over 600–720 simulated days.
Memory events cluster near day 0–100 while queries cluster near day 291–721.
The resulting gap (190+ days) means all memories are "old" at query time.

The base_store archival floor (0.65 for t ≥ 90 days with λ=0.01) kicks in at
~43 simulated days for exponential decay, so every memory in a 190+ day old
dataset is clamped to the same floor weight of 0.65 regardless of policy.
This causes `decay_policy: none`, `exponential`, and `logarithmic` to produce
identical composite scores — the decay variant that should be observable cannot
fire below the floor.

Fix
---
After loading the dataset, shift ALL event days forward by
    shift = min_query_day − max_event_day − TARGET_GAP_DAYS

so that the newest memory is TARGET_GAP_DAYS days old when the first query runs.
Queries stay at their original days (the temporal structure of the evaluation is
preserved). Only event days change — they all shift forward together, preserving
every relative gap between events.

The resulting age distribution at first-query time spans
    [TARGET_GAP_DAYS, TARGET_GAP_DAYS + (max_event_day − min_event_day)]
days.  The youngest memories (≤ 43 days at λ=0.01) see genuine decay variation;
older ones still hit the archival floor but no longer all hit it uniformly.

Diagnostic summary
------------------
After normalization a one-line INFO log is emitted:

    [dataset] Timestamps normalized: delta=+186d,
      range [day 186 → day 286] (was [day 0 → day 100]),
      memory age at first query: min=5d  median=55d  max=105d

The `timestamp_normalization` dict returned from `normalize_timestamps()` is
suitable for embedding directly into the run's summary JSON.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from benchmark.gold.schema import GoldDataset, GoldDayEvents

logger = logging.getLogger(__name__)

# The SimulatedClock's default epoch, used to convert day ↔ datetime for
# the diagnostic log and summary JSON.
_SIMULATED_EPOCH: datetime = datetime(2026, 1, 1, tzinfo=UTC)

# Number of simulated days between the latest memory event and the first
# query after normalization.  Chosen so that the newest events (within
# TARGET_GAP_DAYS of first query) sit inside the effective decay range
# for λ=0.01 (effective up to ~43 days before the archival floor).
TARGET_GAP_DAYS: int = 5


def normalize_timestamps(
    dataset: GoldDataset,
    reference_date: datetime | None = None,
) -> tuple[GoldDataset, dict[str, Any]]:
    """Shift event days so memories are fresh when queries execute.

    The latest memory event is placed TARGET_GAP_DAYS before the first query.
    All other event days are shifted by the same constant, preserving every
    relative gap between events.  Query days are NOT changed.

    Args:
        dataset: The loaded GoldDataset to normalize.
        reference_date: Reference datetime for the diagnostic log.
            Defaults to ``datetime.now(UTC)``.  Does not affect the
            computed shift — the shift is derived from the dataset's own
            day structure.

    Returns:
        A ``(normalized_dataset, metadata)`` tuple where ``metadata`` is a
        dict suitable for embedding in the per-run summary JSON::

            {
                "applied": True,
                "delta_days": 186,
                "original_max_event_day": 100,
                "normalized_max_event_day": 286,
                "original_max_ts": "2026-04-11T00:00:00+00:00",
                "normalized_max_ts": "2026-10-24T00:00:00+00:00",
                "memory_age_at_query_min_days": 5,
                "memory_age_at_query_median_days": 55,
                "memory_age_at_query_max_days": 105,
            }

        Normalization is ALWAYS applied if the dataset has events and queries.
        The newest event is placed exactly TARGET_GAP_DAYS before the first query,
        ensuring memories are fresh at query time and decay policies can differentiate.

        If the dataset is empty (no events/queries), returns unchanged with applied=False.
    """
    if reference_date is None:
        reference_date = datetime.now(UTC)

    if not dataset.events or not dataset.queries:
        return dataset, _meta_not_applied("empty dataset")

    # Materialize once — three separate generator scans over the same list is 3× work.
    event_days = [de.day for de in dataset.events]
    max_event_day: int = max(event_days)
    min_event_day: int = min(event_days)
    min_query_day: int = min(q.day for q in dataset.queries)

    current_gap: int = min_query_day - max_event_day

    # DESIGN FIX: Only normalize when events are genuinely stale (far from queries).
    # If the gap between the newest event and first query is already within
    # TARGET_GAP_DAYS or events overlap with queries, normalization would produce
    # a negative or zero shift that collapses event days via max(0, ...) clamping.
    # This destroys relative event spacing and causes data loss in GoldDatasetScenario
    # (which keys events by day — collisions overwrite earlier entries).
    if current_gap <= TARGET_GAP_DAYS:
        # Events are already fresh enough. No normalization needed.
        return dataset, _meta_not_applied(
            f"gap ({current_gap}d) already <= target ({TARGET_GAP_DAYS}d)"
        )

    target_event_day: int = min_query_day - TARGET_GAP_DAYS
    shift: int = target_event_day - max_event_day

    # Rebuild GoldDayEvents with shifted day numbers.  GoldDayEvents is
    # frozen so we must construct new objects — memory_events are shared
    # (they are immutable and contain no day reference themselves).
    new_events: list[GoldDayEvents] = [
        GoldDayEvents(day=max(0, de.day + shift), memory_events=de.memory_events)
        for de in dataset.events
    ]

    normalized: GoldDataset = GoldDataset(
        schema_version=dataset.schema_version,
        scenario=dataset.scenario,
        description=dataset.description,
        user_ids=dataset.user_ids,
        total_conversation_turns=dataset.total_conversation_turns,
        events=new_events,
        queries=dataset.queries,  # queries stay at original days
        evaluation_criteria=dataset.evaluation_criteria,
    )

    # Compute age distribution at first-query time for the diagnostic log.
    # CRITICAL: Use new_events (post-normalization) to ensure ages reflect the shifted timeline.
    new_max_event_day: int = max_event_day + shift
    new_min_event_day: int = min_event_day + shift
    ages_at_first_query: list[int] = [
        min_query_day - new_de.day for new_de in new_events if new_de.day <= min_query_day
    ]
    age_min = min(ages_at_first_query) if ages_at_first_query else 0
    age_median = int(median(ages_at_first_query)) if ages_at_first_query else 0
    age_max = max(ages_at_first_query) if ages_at_first_query else 0

    # Convert day numbers to datetimes for the diagnostic log / JSON.
    orig_max_ts: datetime = _SIMULATED_EPOCH + timedelta(days=max_event_day)
    norm_max_ts: datetime = _SIMULATED_EPOCH + timedelta(days=new_max_event_day)

    logger.info(
        "[dataset] Timestamps normalized: delta=+%dd, "
        "range [day %d → day %d] (was [day %d → day %d]), "
        "memory age at first query: min=%dd  median=%dd  max=%dd",
        shift,
        new_min_event_day,
        new_max_event_day,
        min_event_day,
        max_event_day,
        age_min,
        age_median,
        age_max,
    )

    metadata: dict[str, Any] = {
        "applied": True,
        "delta_days": shift,
        "original_max_event_day": max_event_day,
        "normalized_max_event_day": new_max_event_day,
        "original_max_ts": orig_max_ts.isoformat(),
        "normalized_max_ts": norm_max_ts.isoformat(),
        "memory_age_at_query_min_days": age_min,
        "memory_age_at_query_median_days": age_median,
        "memory_age_at_query_max_days": age_max,
    }
    return normalized, metadata


def _meta_not_applied(reason: str) -> dict[str, Any]:
    """Return a metadata dict indicating normalization was skipped."""
    return {"applied": False, "reason": reason}
