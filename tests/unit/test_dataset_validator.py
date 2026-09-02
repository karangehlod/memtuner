"""Unit tests for DatasetValidator.

Verifies each validation rule triggers on crafted invalid datasets
and that valid datasets pass without error.
"""

from __future__ import annotations

import pytest

from benchmark.application.errors import DatasetValidationError
from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldEvaluationCriteria,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
)
from benchmark.gold.validator import DatasetValidator


def _event(event_id: str, user_id: str = "user-alice") -> GoldMemoryEvent:
    """Create a minimal test memory event."""
    return GoldMemoryEvent(
        id=event_id,
        user_id=user_id,
        type="episodic",
        content=f"Content for {event_id}",
        importance=0.8,
        task_id="test-task",
    )


def _query(
    query_text: str,
    expected_ids: list[str],
    user_id: str = "user-alice",
    day: int = 5,
) -> GoldQuery:
    """Create a minimal test query."""
    return GoldQuery(
        day=day,
        query=query_text,
        task_id="test-task",
        user_id=user_id,
        expected=GoldExpectedResult(memory_ids=expected_ids),
        is_followup=False,
        references_turn=None,
    )


def _dataset(
    events: list[GoldDayEvents] | None = None,
    queries: list[GoldQuery] | None = None,
) -> GoldDataset:
    """Create a minimal test dataset."""
    return GoldDataset(
        schema_version="1.0",
        scenario="test",
        description="test dataset",
        user_ids=["user-alice"],
        total_conversation_turns=3,
        events=events or [GoldDayEvents(day=0, memory_events=[_event("M-001")])],
        queries=queries or [_query("test query", ["M-001"])],
        evaluation_criteria=GoldEvaluationCriteria(recall_k=5),
    )


@pytest.mark.unit
class TestDatasetValidatorValid:
    """Tests that valid datasets pass validation."""

    def test_valid_dataset_passes(self) -> None:
        """A structurally valid dataset passes without error."""
        dataset = _dataset()
        DatasetValidator().validate(dataset)

    def test_multi_user_valid_dataset_passes(self) -> None:
        """Multiple users with their own events and queries pass."""
        events = [
            GoldDayEvents(day=0, memory_events=[_event("M-001", "user-alice")]),
            GoldDayEvents(day=1, memory_events=[_event("M-002", "user-bob")]),
        ]
        queries = [
            _query("alice query", ["M-001"], user_id="user-alice"),
            _query("bob query", ["M-002"], user_id="user-bob"),
        ]
        dataset = _dataset(events=events, queries=queries)
        DatasetValidator().validate(dataset)


@pytest.mark.unit
class TestDatasetValidatorDuplicateIds:
    """Tests for duplicate memory ID detection."""

    def test_duplicate_memory_ids_raises(self) -> None:
        """Duplicate memory IDs across events are detected."""
        events = [
            GoldDayEvents(day=0, memory_events=[_event("M-001")]),
            GoldDayEvents(day=1, memory_events=[_event("M-001")]),
        ]
        dataset = _dataset(events=events)

        with pytest.raises(DatasetValidationError, match="Duplicate memory ID.*M-001"):  # noqa: RUF043
            DatasetValidator().validate(dataset)

    def test_duplicate_within_same_day_raises(self) -> None:
        """Duplicate IDs within the same day are detected."""
        events = [
            GoldDayEvents(day=0, memory_events=[_event("M-001"), _event("M-001")]),
        ]
        dataset = _dataset(events=events)

        with pytest.raises(DatasetValidationError, match="Duplicate memory ID"):
            DatasetValidator().validate(dataset)


@pytest.mark.unit
class TestDatasetValidatorOrphanedIds:
    """Tests for orphaned expected ID detection."""

    def test_orphaned_expected_id_raises(self) -> None:
        """Expected IDs not in events are detected."""
        events = [GoldDayEvents(day=0, memory_events=[_event("M-001")])]
        queries = [_query("test", ["M-001", "M-999"])]
        dataset = _dataset(events=events, queries=queries)

        with pytest.raises(DatasetValidationError, match="M-999.*does not exist"):  # noqa: RUF043
            DatasetValidator().validate(dataset)


@pytest.mark.unit
class TestDatasetValidatorNegativeDays:
    """Tests for negative day detection.

    Note: GoldDayEvents enforces day >= 0 at model level, so this check
    is defense-in-depth for datasets constructed outside Pydantic (raw dicts).
    We test the validator logic directly by patching the dataset.
    """

    def test_negative_event_day_detected_by_validator(self) -> None:
        """Validator detects negative days when checking raw data."""
        validator = DatasetValidator()
        # Test the internal method directly with a crafted object
        from unittest.mock import MagicMock

        mock_day_events = MagicMock()
        mock_day_events.day = -1
        mock_day_events.memory_events = [_event("M-001")]

        mock_dataset = MagicMock()
        mock_dataset.events = [mock_day_events]
        mock_dataset.queries = []

        errors = validator._check_negative_days(mock_dataset)
        assert len(errors) == 1
        assert "-1" in errors[0]


@pytest.mark.unit
class TestDatasetValidatorInvalidUsers:
    """Tests for invalid user reference detection."""

    def test_query_user_not_in_events_raises(self) -> None:
        """Query referencing a user with no events is detected."""
        events = [GoldDayEvents(day=0, memory_events=[_event("M-001", "user-alice")])]
        queries = [_query("test", ["M-001"], user_id="user-ghost")]
        dataset = _dataset(events=events, queries=queries)

        with pytest.raises(DatasetValidationError, match="user-ghost.*has no events"):  # noqa: RUF043
            DatasetValidator().validate(dataset)


@pytest.mark.unit
class TestDatasetValidatorEmptyExpected:
    """Tests for empty expected set detection.

    Note: GoldExpectedResult enforces min_length=1 at model level.
    This test verifies the validator's internal check using a mock.
    """

    def test_empty_expected_set_detected_by_validator(self) -> None:
        """Validator detects empty expected sets in raw data."""
        validator = DatasetValidator()
        from unittest.mock import MagicMock

        mock_query = MagicMock()
        mock_query.query = "What happened?"
        mock_query.expected.memory_ids = []

        mock_dataset = MagicMock()
        mock_dataset.events = []
        mock_dataset.queries = [mock_query]

        errors = validator._check_empty_expected_sets(mock_dataset)
        assert len(errors) == 1
        assert "empty expected set" in errors[0]


@pytest.mark.unit
class TestDatasetValidatorMultipleErrors:
    """Tests that multiple errors are collected in one pass."""

    def test_collects_multiple_errors(self) -> None:
        """Multiple validation failures are reported together."""
        # Duplicate IDs + orphaned expected + invalid user
        events = [
            GoldDayEvents(day=0, memory_events=[_event("M-001"), _event("M-001")]),
        ]
        queries = [_query("test", ["M-999"], user_id="user-ghost")]
        dataset = _dataset(events=events, queries=queries)

        with pytest.raises(DatasetValidationError) as exc_info:
            DatasetValidator().validate(dataset)

        error_msg = str(exc_info.value)
        assert "Duplicate" in error_msg
        assert "does not exist" in error_msg
        assert "user-ghost" in error_msg
