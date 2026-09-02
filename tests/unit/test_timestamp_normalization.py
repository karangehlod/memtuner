"""Unit tests for benchmark.gold.normalizer.normalize_timestamps."""

from __future__ import annotations

import logging

from benchmark.gold.normalizer import TARGET_GAP_DAYS, normalize_timestamps
from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldEvaluationCriteria,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_event(event_id: str, day: int) -> GoldDayEvents:
    """Build a minimal GoldDayEvents for a single day."""
    mem = GoldMemoryEvent(
        id=event_id,
        user_id="u1",
        type="episodic",
        content=f"event {event_id}",
        importance=0.8,
        task_id="task-1",
    )
    return GoldDayEvents(day=day, memory_events=[mem])


def _make_query(query_id: str, day: int) -> GoldQuery:
    """Build a minimal GoldQuery for a single day."""
    expected = GoldExpectedResult(memory_ids=[f"mem-{query_id}"])
    return GoldQuery(
        day=day,
        query=f"q {query_id}",
        task_id=query_id,
        user_id="u1",
        expected=expected,
        is_followup=False,
        references_turn=None,
    )


def _make_dataset(
    event_days: list[int],
    query_days: list[int],
    scenario: str = "test",
) -> GoldDataset:
    """Assemble a minimal GoldDataset from day lists."""
    events = [_make_event(f"e{i}", d) for i, d in enumerate(event_days)]
    queries = [_make_query(f"q{i}", d) for i, d in enumerate(query_days)]
    return GoldDataset(
        schema_version="1.0",
        scenario=scenario,
        description="unit test dataset",
        user_ids=["u1"],
        total_conversation_turns=len(events),
        events=events,
        queries=queries,
        evaluation_criteria=GoldEvaluationCriteria(recall_k=5),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNormalizeTimestamps:

    def test_shift_places_newest_event_at_target_gap(self):
        """After normalization the newest event is exactly TARGET_GAP_DAYS
        before the first query."""
        # Events on days 0–100, queries on days 291–350 (gap = 191 days)
        ds = _make_dataset(event_days=[0, 50, 100], query_days=[291, 350])
        norm_ds, meta = normalize_timestamps(ds)

        assert meta["applied"] is True
        new_max_event_day = max(e.day for e in norm_ds.events)
        min_query_day = min(q.day for q in norm_ds.queries)
        assert min_query_day - new_max_event_day == TARGET_GAP_DAYS

    def test_relative_gaps_between_events_preserved(self):
        """Shifting forward must not change any relative gap between events."""
        event_days = [0, 30, 70, 100]
        ds = _make_dataset(event_days=event_days, query_days=[400])
        norm_ds, meta = normalize_timestamps(ds)

        assert meta["applied"] is True
        orig_gaps = [
            event_days[i + 1] - event_days[i] for i in range(len(event_days) - 1)
        ]
        norm_days = sorted(e.day for e in norm_ds.events)
        norm_gaps = [norm_days[i + 1] - norm_days[i] for i in range(len(norm_days) - 1)]
        assert orig_gaps == norm_gaps

    def test_query_days_unchanged(self):
        """Query days must NOT be altered by normalization."""
        query_days = [291, 350, 500, 720]
        ds = _make_dataset(event_days=[0, 50, 100], query_days=query_days)
        norm_ds, _ = normalize_timestamps(ds)

        assert [q.day for q in norm_ds.queries] == query_days

    def test_normalization_idempotent(self):
        """Calling normalize_timestamps twice produces the same result (idempotent).
        - First call: shifts from 100 to make gap = TARGET_GAP_DAYS (applied=True)
        - Second call: gap is now at target, no shift needed (applied=False)

        The key property: calling normalize_timestamps(normalize_timestamps(ds))
        produces the same events."""
        ds = _make_dataset(event_days=[0, 50, 100], query_days=[291])
        first_ds, first_meta = normalize_timestamps(ds)
        second_ds, second_meta = normalize_timestamps(first_ds)

        # First call applies, second does not (gap is now within target)
        assert first_meta["applied"] is True
        assert second_meta["applied"] is False
        # Days should be identical after second call (idempotent)
        assert [e.day for e in first_ds.events] == [e.day for e in second_ds.events]

    def test_no_shift_when_gap_already_within_target(self):
        """Dataset whose gap is already ≤ TARGET_GAP_DAYS should NOT be normalized.
        Normalization would produce a negative shift that destroys event spacing."""
        # Latest event on day 100, first query on day 104 → gap = 4 < TARGET_GAP = 5
        ds = _make_dataset(event_days=[0, 100], query_days=[104])
        norm_ds, meta = normalize_timestamps(ds)

        # Not applied — gap is already acceptable
        assert meta["applied"] is False
        # Events unchanged
        assert [e.day for e in norm_ds.events] == [0, 100]

    def test_no_shift_exact_target_gap(self):
        """Gap exactly equal to TARGET_GAP_DAYS: no shift needed."""
        ds = _make_dataset(event_days=[0, 100], query_days=[100 + TARGET_GAP_DAYS])
        norm_ds, meta = normalize_timestamps(ds)

        # Not applied — gap equals target exactly
        assert meta["applied"] is False
        # Events unchanged
        assert [e.day for e in norm_ds.events] == [0, 100]

    def test_small_gap_not_shifted(self):
        """When gap is smaller than TARGET_GAP_DAYS, normalization is skipped
        to prevent negative shifts that collapse event days."""
        # Gap = 3 < TARGET_GAP_DAYS=5
        ds = _make_dataset(event_days=[0, 100], query_days=[103])
        returned_ds, meta = normalize_timestamps(ds)

        # Not applied — gap is within target
        assert meta["applied"] is False
        # Events unchanged (no destructive shift)
        assert [e.day for e in returned_ds.events] == [0, 100]

    def test_info_log_emitted_when_shift_applied(self, caplog):
        """An INFO-level log line must be emitted when normalization shifts days."""
        ds = _make_dataset(event_days=[0, 100], query_days=[291])
        with caplog.at_level(logging.INFO, logger="benchmark.gold.normalizer"):
            _, meta = normalize_timestamps(ds)

        assert meta["applied"] is True
        assert any("[dataset] Timestamps normalized" in rec.message for rec in caplog.records)

    def test_metadata_fields_present_when_applied(self):
        """All expected metadata fields must be present when normalization runs."""
        ds = _make_dataset(event_days=[0, 50, 100], query_days=[291])
        _, meta = normalize_timestamps(ds)

        assert meta["applied"] is True
        required = {
            "delta_days",
            "original_max_event_day",
            "normalized_max_event_day",
            "original_max_ts",
            "normalized_max_ts",
            "memory_age_at_query_min_days",
            "memory_age_at_query_median_days",
            "memory_age_at_query_max_days",
        }
        assert required.issubset(meta.keys())

    def test_delta_days_equals_gap_minus_target(self):
        """delta_days == min_query_day - max_event_day - TARGET_GAP_DAYS."""
        event_days = [0, 100]
        query_days = [291]
        ds = _make_dataset(event_days=event_days, query_days=query_days)
        _, meta = normalize_timestamps(ds)

        expected_delta = min(query_days) - max(event_days) - TARGET_GAP_DAYS
        assert meta["delta_days"] == expected_delta

    def test_all_event_days_non_negative_after_shift(self):
        """No event day may become negative after normalization."""
        ds = _make_dataset(event_days=[0, 1, 2, 100], query_days=[291])
        norm_ds, meta = normalize_timestamps(ds)

        assert meta["applied"] is True
        assert all(e.day >= 0 for e in norm_ds.events)
