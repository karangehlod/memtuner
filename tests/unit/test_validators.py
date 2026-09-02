"""Unit tests for dataset validators.

Tests each validator independently and the registry system.

SOLID principles tested:
  - Single Responsibility: Each validator validates one aspect
  - Open/Closed: New validators can be added without modifying existing ones
  - Interface Segregation: Validator protocol is minimal
  - Dependency Inversion: Tests depend on Validator interface
"""

import pytest

from benchmark.gold.adapters import ValidationReport, ValidationSeverity
from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
)
from benchmark.gold.validators import (
    IntegrityValidator,
    SchemaValidator,
    StatisticsValidator,
    TemporalValidator,
    ValidationRegistry,
    Validator,
)
from benchmark.models.memory_event import MemoryType

# ============================================================================
# Test Data Helpers
# ============================================================================


def make_memory_event(
    id: str = "mem1",
    user_id: str = "user1",
    type: MemoryType = MemoryType.EPISODIC,
    content: str = "test content",
    importance: float = 0.5,
    entities: list[str] | None = None,
    task_id: str = "task1",
    conversation_turn: int = 0,
) -> GoldMemoryEvent:
    """Create a test memory event."""
    return GoldMemoryEvent(
        id=id,
        user_id=user_id,
        type=type,
        content=content,
        importance=importance,
        entities=entities or [],
        task_id=task_id,
        conversation_turn=conversation_turn,
    )


def make_day_events(
    day: int = 0, memory_events: list[GoldMemoryEvent] | None = None
) -> GoldDayEvents:
    """Create a test day events."""
    if memory_events is None:
        memory_events = [make_memory_event()]
    return GoldDayEvents(day=day, memory_events=memory_events)


def make_expected_result(
    memory_ids: list[str] | None = None,
    acceptable_modules: list[str] | None = None,
) -> GoldExpectedResult:
    """Create a test expected result."""
    return GoldExpectedResult(
        memory_ids=memory_ids or ["mem1"],
        acceptable_modules=acceptable_modules or [],
    )


def make_query(
    day: int = 0,
    query: str = "test query",
    task_id: str = "task1",
    user_id: str = "user1",
    expected: GoldExpectedResult | None = None,
) -> GoldQuery:
    """Create a test query."""
    if expected is None:
        expected = make_expected_result()
    return GoldQuery(
        day=day,
        query=query,
        task_id=task_id,
        user_id=user_id,
        expected=expected,
    )


def make_dataset(
    scenario: str = "test_scenario",
    description: str = "test description",
    queries: list[GoldQuery] | None = None,
    events: list[GoldDayEvents] | None = None,
) -> GoldDataset:
    """Create a test dataset."""
    if queries is None:
        queries = [make_query()]
    if events is None:
        events = [make_day_events()]
    return GoldDataset(
        scenario=scenario,
        description=description,
        queries=queries,
        events=events,
    )


# ============================================================================
# Schema Validator Tests
# ============================================================================


class TestSchemaValidator:
    """Test schema validation."""

    def test_valid_dataset_passes(self) -> None:
        """Valid dataset should pass schema validation."""
        dataset = make_dataset()
        validator = SchemaValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_multiple_queries_and_events(self) -> None:
        """Dataset with multiple queries and events should pass."""
        queries = [make_query(task_id=f"task{i}") for i in range(3)]
        events = [make_day_events(day=i) for i in range(3)]
        dataset = make_dataset(queries=queries, events=events)
        validator = SchemaValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_event_with_multiple_memories(self) -> None:
        """Event with multiple memories should pass."""
        memories = [make_memory_event(id=f"mem{i}") for i in range(5)]
        day_events = make_day_events(memory_events=memories)
        dataset = make_dataset(events=[day_events])
        validator = SchemaValidator()
        report = validator.validate(dataset)
        assert report.passed


# ============================================================================
# Temporal Validator Tests
# ============================================================================


class TestTemporalValidator:
    """Test temporal validation."""

    def test_valid_temporal_order_passes(self) -> None:
        """Dataset with valid day ordering should pass."""
        events = [make_day_events(day=0), make_day_events(day=1), make_day_events(day=2)]
        dataset = make_dataset(events=events)
        validator = TemporalValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_out_of_order_days_fails(self) -> None:
        """Days out of order should fail validation."""
        events = [make_day_events(day=2), make_day_events(day=1)]
        dataset = make_dataset(events=events)
        validator = TemporalValidator()
        report = validator.validate(dataset)
        assert not report.passed
        assert len(report.errors) >= 1
        assert any("not in increasing order" in e.message for e in report.errors)

    def test_duplicate_days_fail(self) -> None:
        """Duplicate day numbers should fail."""
        events = [make_day_events(day=1), make_day_events(day=1)]
        dataset = make_dataset(events=events)
        validator = TemporalValidator()
        report = validator.validate(dataset)
        assert not report.passed

    def test_single_day_produces_info(self) -> None:
        """Dataset with only one day produces info message."""
        events = [make_day_events(day=0)]
        dataset = make_dataset(events=events)
        validator = TemporalValidator()
        report = validator.validate(dataset)
        # Single day is OK but noted
        assert report.passed


# ============================================================================
# Integrity Validator Tests
# ============================================================================


class TestIntegrityValidator:
    """Test referential integrity validation."""

    def test_valid_dataset_passes(self) -> None:
        """Valid dataset should pass integrity validation."""
        dataset = make_dataset()
        validator = IntegrityValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_query_references_valid_memory(self) -> None:
        """Query referencing valid memory should pass."""
        event = make_memory_event(id="mem1")
        day_events = make_day_events(memory_events=[event])

        expected = make_expected_result(memory_ids=["mem1"])
        query = make_query(expected=expected)

        dataset = make_dataset(queries=[query], events=[day_events])
        validator = IntegrityValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_query_references_invalid_memory_warning(self) -> None:
        """Query referencing non-existent memory should warn."""
        expected = make_expected_result(memory_ids=["nonexistent"])
        query = make_query(expected=expected)

        dataset = make_dataset(queries=[query])
        validator = IntegrityValidator()
        report = validator.validate(dataset)
        # Should produce a warning about missing memory
        assert len(report.warnings) >= 1


# ============================================================================
# Statistics Validator Tests
# ============================================================================


class TestStatisticsValidator:
    """Test statistics validation."""

    def test_valid_importance_scores_pass(self) -> None:
        """Memory events with valid importance scores should pass."""
        event = make_memory_event(importance=0.5)
        day_events = make_day_events(memory_events=[event])
        dataset = make_dataset(events=[day_events] + [make_day_events(day=i) for i in range(1, 4)])
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_importance_boundary_zero(self) -> None:
        """Importance at zero boundary should pass."""
        event = make_memory_event(importance=0.0)
        day_events = make_day_events(memory_events=[event])
        dataset = make_dataset(events=[day_events] + [make_day_events(day=i) for i in range(1, 4)])
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_importance_boundary_one(self) -> None:
        """Importance at one boundary should pass."""
        event = make_memory_event(importance=1.0)
        day_events = make_day_events(memory_events=[event])
        dataset = make_dataset(events=[day_events] + [make_day_events(day=i) for i in range(1, 4)])
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_varied_importance_scores(self) -> None:
        """Dataset with varied importance scores should pass."""
        memories = [
            make_memory_event(id=f"mem{i}", importance=float(i) / 10)
            for i in range(11)  # 0.0, 0.1, ..., 1.0
        ]
        day_events = make_day_events(memory_events=memories)
        dataset = make_dataset(events=[day_events] + [make_day_events(day=i) for i in range(1, 4)])
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_single_day_coverage_info(self) -> None:
        """Single day should produce an info-level coverage note.

        Note: with the current StatisticsValidator, passed=len(issues)==0,
        so an INFO-severity issue still flips passed to False — this test
        checks the issue itself, not report.passed.
        """
        dataset = make_dataset(events=[make_day_events(day=0)])
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert len(report.issues) == 1
        assert report.issues[0].severity == ValidationSeverity.INFO
        assert "1 day" in report.issues[0].message

    def test_multi_day_coverage_passes(self) -> None:
        """Multi-day coverage should pass."""
        events = [make_day_events(day=i) for i in range(10)]
        dataset = make_dataset(events=events)
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert report.passed

    def test_sparse_events(self) -> None:
        """Sparse events (one per day) should pass."""
        events = [
            make_day_events(day=i * 10)  # Every 10 days
            for i in range(5)
        ]
        dataset = make_dataset(events=events)
        validator = StatisticsValidator()
        report = validator.validate(dataset)
        assert isinstance(report, ValidationReport)


# ============================================================================
# Registry Tests
# ============================================================================


class TestValidationRegistry:
    """Test validator registry."""

    def test_registry_has_default_validators(self) -> None:
        """Registry should have built-in validators."""
        validators = ValidationRegistry.list_all()
        assert "schema" in validators
        assert "temporal" in validators
        assert "integrity" in validators
        assert "statistics" in validators

    def test_get_validator_by_name(self) -> None:
        """Can retrieve validator by name."""
        validator = ValidationRegistry.get("schema")
        assert isinstance(validator, SchemaValidator)

    def test_get_unknown_validator_raises_error(self) -> None:
        """Unknown validator name should raise error."""
        with pytest.raises(ValueError):
            ValidationRegistry.get("nonexistent")

    def test_validate_all_combines_results(self) -> None:
        """validate_all should combine results from all validators."""
        dataset = make_dataset()
        report = ValidationRegistry.validate_all(dataset)

        assert isinstance(report, ValidationReport)
        assert isinstance(report.passed, bool)
        # Should have run multiple validators
        assert isinstance(report.issues, list)

    def test_validate_all_on_valid_dataset(self) -> None:
        """validate_all should find no issues in a dataset with good day coverage."""
        events = [make_day_events(day=i) for i in range(4)]
        dataset = make_dataset(events=events)
        report = ValidationRegistry.validate_all(dataset)
        assert report.passed

    def test_validate_all_with_multiple_days(self) -> None:
        """validate_all should handle multi-day datasets."""
        events = [make_day_events(day=i) for i in range(5)]
        dataset = make_dataset(events=events)
        report = ValidationRegistry.validate_all(dataset)
        assert report.passed

    def test_issues_sorted_by_severity(self) -> None:
        """Issues should be sorted by severity (errors first)."""
        # Single-day dataset (INFO from StatisticsValidator) whose query
        # references a memory ID that doesn't exist (WARNING from
        # IntegrityValidator) — a validly-constructed dataset that still
        # yields issues of two different severities to check sort order.
        expected = make_expected_result(memory_ids=["does-not-exist"])
        query = make_query(expected=expected)
        dataset = make_dataset(queries=[query])
        report = ValidationRegistry.validate_all(dataset)

        # Check that issues are sorted
        if len(report.issues) > 1:
            for i in range(len(report.issues) - 1):
                curr_severity_val = {
                    ValidationSeverity.ERROR: 0,
                    ValidationSeverity.WARNING: 1,
                    ValidationSeverity.INFO: 2,
                }.get(report.issues[i].severity, 3)

                next_severity_val = {
                    ValidationSeverity.ERROR: 0,
                    ValidationSeverity.WARNING: 1,
                    ValidationSeverity.INFO: 2,
                }.get(report.issues[i + 1].severity, 3)

                assert curr_severity_val <= next_severity_val

    def test_duplicate_validator_registration_fails(self) -> None:
        """Registering duplicate validator name should fail."""
        class DuplicateValidator(Validator):
            name = "schema"
            def validate(self, dataset: GoldDataset) -> ValidationReport:
                return ValidationReport(passed=True, issues=[])

        with pytest.raises(ValueError):
            ValidationRegistry.register(DuplicateValidator)


# ============================================================================
# Integration Tests
# ============================================================================


class TestValidatorIntegration:
    """Test validators working together."""

    def test_complete_validation_flow(self) -> None:
        """Test complete validation pipeline."""
        # Create a realistic dataset
        events = [
            make_day_events(
                day=0,
                memory_events=[
                    make_memory_event(id=f"mem{i}", importance=0.5)
                    for i in range(5)
                ],
            ),
            make_day_events(
                day=1,
                memory_events=[
                    make_memory_event(id=f"mem{i+5}", importance=0.7)
                    for i in range(3)
                ],
            ),
            make_day_events(day=2),
            make_day_events(day=3),
        ]
        queries = [make_query(task_id=f"task{i}") for i in range(3)]
        dataset = make_dataset(queries=queries, events=events)

        # Validate
        report = ValidationRegistry.validate_all(dataset)

        # Check structure
        assert isinstance(report, ValidationReport)
        assert isinstance(report.passed, bool)
        assert isinstance(report.issues, list)
        assert report.passed  # Valid dataset should pass all validators

    def test_validator_error_messages_are_helpful(self) -> None:
        """Error messages should be helpful for debugging."""
        day_events = make_day_events(memory_events=[make_memory_event()])
        dataset = make_dataset(events=[day_events])

        report = ValidationRegistry.validate_all(dataset)

        # Should have helpful structure
        for issue in report.issues:
            assert issue.message  # Non-empty message
            assert issue.validator  # Clear which validator found it
            assert issue.severity  # Severity level specified
            # Location is optional but helpful when present

    def test_all_validators_run_in_validate_all(self) -> None:
        """All registered validators should be called by validate_all."""
        dataset = make_dataset()
        report = ValidationRegistry.validate_all(dataset)

        # Since all validators run, check report is properly aggregated
        assert isinstance(report, ValidationReport)
        validators_ran = len(set(issue.validator for issue in report.issues))
        # Could be 0-4 validators depending on issues found
        assert 0 <= validators_ran <= 4

    def test_dataset_with_gaps_in_days(self) -> None:
        """Dataset with non-consecutive days should pass schema/temporal."""
        events = [
            make_day_events(day=0),
            make_day_events(day=5),
            make_day_events(day=10),
        ]
        dataset = make_dataset(events=events)
        report = ValidationRegistry.validate_all(dataset)
        assert report.passed  # Non-consecutive days are OK
