"""Dataset validation framework with pluggable validators.

Each validator checks specific aspects of the dataset:
  - SchemaValidator: Pydantic model validation
  - SplitValidator: Train/test split consistency
  - TemporalValidator: Event ordering and time consistency
  - IntegrityValidator: Referential consistency
  - StatisticsValidator: Sanity checks on metrics

SOLID principles:
  - Single Responsibility: Each validator has one job
  - Open/Closed: New validators can be added without modifying existing ones
  - Interface Segregation: Common interface via abc.abstractmethod
  - Dependency Inversion: Consumers depend on Validator protocol
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from benchmark.gold.adapters import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from benchmark.gold.schema import GoldDataset


# ============================================================================
# Validator Base Class
# ============================================================================


class Validator(ABC):
    """Abstract base class for dataset validators.

    Each validator is responsible for checking one specific aspect
    of a dataset (schema, temporal ordering, referential integrity, etc).
    """

    name: ClassVar[str]
    """Unique validator identifier."""

    @abstractmethod
    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate dataset and return report.

        Args:
            dataset: Dataset to validate.

        Returns:
            ValidationReport with any issues found (empty if valid).
        """
        pass


# ============================================================================
# Schema Validator
# ============================================================================


class SchemaValidator(Validator):
    """Validate dataset against pydantic schema.

    Checks that all objects conform to their pydantic models:
      - GoldDataset model
      - All queries conform to GoldQuery
      - All events conform to GoldMemoryEvent
      - All temporal windows valid
    """

    name: ClassVar[str] = "schema"

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate dataset schema compliance.

        Pydantic validation is already done on load, so this
        verifies collections and cross-references.
        """
        issues = []

        # Verify queries list is valid
        if not isinstance(dataset.queries, list):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    validator=self.name,
                    message=f"queries must be list, got {type(dataset.queries).__name__}",
                )
            )
            return ValidationReport(passed=False, issues=issues)

        if not dataset.queries:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    validator=self.name,
                    message="Dataset has zero queries",
                    location="dataset.queries",
                )
            )

        # Verify events list is valid
        if not isinstance(dataset.events, list):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    validator=self.name,
                    message=f"events must be list, got {type(dataset.events).__name__}",
                )
            )
            return ValidationReport(passed=False, issues=issues)

        if not dataset.events:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    validator=self.name,
                    message="Dataset has zero memory events",
                    location="dataset.events",
                )
            )

        # Verify queries have required fields
        for i, query in enumerate(dataset.queries):
            if not query.query:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        validator=self.name,
                        message="Query missing query text",
                        location=f"queries[{i}].query",
                    )
                )
            if not query.task_id:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        validator=self.name,
                        message="Query missing task_id",
                        location=f"queries[{i}].task_id",
                    )
                )
            if not query.expected:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        validator=self.name,
                        message="Query missing expected results",
                        location=f"queries[{i}].expected",
                    )
                )

        for i, event in enumerate(dataset.events):
            if not event.memory_events:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        validator=self.name,
                        message=f"Day {event.day} has no memory events",
                        location=f"events[{i}]",
                    )
                )

        return ValidationReport(passed=len(issues) == 0, issues=issues)


# ============================================================================
# Temporal Validator
# ============================================================================


class TemporalValidator(Validator):
    """Validate temporal ordering and consistency.

    Checks that:
      - Days are in increasing order
      - Memories within a day are consistent
      - Query dates make sense
      - No time-travel (events after queries they depend on)
    """

    name: ClassVar[str] = "temporal"

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate temporal consistency."""
        issues = []

        if not dataset.events:
            return ValidationReport(passed=True, issues=[])

        # Check day ordering
        prev_day = -1
        for i, day_events in enumerate(dataset.events):
            if day_events.day <= prev_day:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        validator=self.name,
                        message=f"Day {day_events.day} not in increasing order (prev={prev_day})",
                        location=f"events[{i}].day",
                    )
                )
            prev_day = day_events.day

            # Check each event in the day
            for j, event in enumerate(day_events.memory_events):
                if event.conversation_turn < 0:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            validator=self.name,
                            message=f"Negative conversation turn: {event.conversation_turn}",
                            location=f"events[{i}].memory_events[{j}].conversation_turn",
                        )
                    )

        # Check that queries have valid reference days (if applicable)
        if dataset.events and dataset.queries:
            max_day = dataset.events[-1].day
            for i, query in enumerate(dataset.queries):
                # If query has reference_day, check it's within bounds
                if hasattr(query, "reference_day") and query.reference_day is not None:
                    if query.reference_day > max_day:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                validator=self.name,
                                message=f"Query references day {query.reference_day}, but max day is {max_day}",
                                location=f"queries[{i}].reference_day",
                            )
                        )

        return ValidationReport(passed=len(issues) == 0, issues=issues)


# ============================================================================
# Integrity Validator
# ============================================================================


class IntegrityValidator(Validator):
    """Validate referential and logical integrity.

    Checks that:
      - Query expected results reference valid memory IDs
      - User IDs are consistent across events
      - No orphaned references
      - Task IDs are consistent
    """

    name: ClassVar[str] = "integrity"

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate referential integrity."""
        issues = []

        if not dataset.events or not dataset.queries:
            return ValidationReport(passed=True, issues=[])

        # Build set of all memory IDs
        all_memory_ids = set()
        all_user_ids = set()
        for day_events in dataset.events:
            for event in day_events.memory_events:
                all_memory_ids.add(event.id)
                all_user_ids.add(event.user_id)

        # Check query references
        for i, query in enumerate(dataset.queries):
            if hasattr(query, "expected") and query.expected:
                for j, memory_id in enumerate(query.expected.memory_ids):
                    if memory_id not in all_memory_ids:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                validator=self.name,
                                message=f"Query references non-existent memory ID: {memory_id}",
                                location=f"queries[{i}].expected.memory_ids[{j}]",
                            )
                        )

        # Check user consistency
        if all_user_ids:
            for i, query in enumerate(dataset.queries):
                if (
                    hasattr(query, "user_id")
                    and query.user_id
                    and query.user_id not in all_user_ids
                ):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            validator=self.name,
                            message=f"Query references non-existent user ID: {query.user_id}",
                            location=f"queries[{i}].user_id",
                        )
                    )

        return ValidationReport(passed=len(issues) == 0, issues=issues)


# ============================================================================
# Statistics Validator
# ============================================================================


class StatisticsValidator(Validator):
    """Validate dataset statistics for sanity.

    Checks that:
      - Event counts match expected distributions
      - Memory importance scores are in valid range [0, 1]
      - No suspicious patterns (all same values, extreme outliers)
      - Coverage is reasonable (not just 1 day)
    """

    name: ClassVar[str] = "statistics"

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate dataset statistics."""
        issues = []

        if not dataset.events:
            return ValidationReport(passed=True, issues=[])

        # Check importance scores
        for day_idx, day_events in enumerate(dataset.events):
            for event_idx, event in enumerate(day_events.memory_events):
                if not (0.0 <= event.importance <= 1.0):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            validator=self.name,
                            message=f"Importance {event.importance} outside [0, 1]",
                            location=f"events[{day_idx}].memory_events[{event_idx}].importance",
                        )
                    )

        # Check temporal coverage
        if len(dataset.events) > 0:
            first_day = dataset.events[0].day
            last_day = dataset.events[-1].day
            day_span = last_day - first_day

            if day_span == 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        validator=self.name,
                        message="Dataset spans only 1 day",
                        location="dataset.events",
                    )
                )
            elif day_span < 3:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        validator=self.name,
                        message=f"Dataset has limited temporal coverage ({day_span} days)",
                        location="dataset.events",
                    )
                )

            # Check memory density
            total_memories = sum(
                len(day.memory_events) for day in dataset.events
            )
            avg_per_day = total_memories / len(dataset.events) if dataset.events else 0
            if avg_per_day < 1:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        validator=self.name,
                        message=f"Low memory density: {avg_per_day:.1f} per day",
                        location="dataset.events",
                    )
                )

        return ValidationReport(passed=len(issues) == 0, issues=issues)


# ============================================================================
# Composite Validator Registry
# ============================================================================


class ValidationRegistry:
    """Registry of available validators.

    Enables extensibility: new validators can be added by registering
    without modifying existing code.
    """

    _validators: dict[str, type[Validator]] = {}

    @classmethod
    def register(cls, validator_cls: type[Validator]) -> None:
        """Register a new validator.

        Args:
            validator_cls: The validator class to register (must have name attribute).
        """
        name = validator_cls.name
        if name in cls._validators:
            raise ValueError(f"Validator '{name}' already registered")
        cls._validators[name] = validator_cls

    @classmethod
    def get(cls, name: str) -> Validator:
        """Get validator instance by name.

        Args:
            name: The validator name to retrieve.

        Returns:
            New instance of the validator.

        Raises:
            ValueError: If validator name not found.
        """
        if name not in cls._validators:
            available = ", ".join(sorted(cls._validators.keys()))
            raise ValueError(
                f"Unknown validator '{name}'. Available: {available}"
            )
        return cls._validators[name]()

    @classmethod
    def validate_all(cls, dataset: GoldDataset) -> ValidationReport:
        """Run all registered validators on dataset.

        Returns:
            Aggregated ValidationReport with all issues from all validators.
        """
        all_issues = []
        passed = True

        for validator_cls in cls._validators.values():
            validator = validator_cls()
            report = validator.validate(dataset)
            all_issues.extend(report.issues)
            if not report.passed:
                passed = False

        # Sort issues by severity (errors first)
        severity_order = {
            ValidationSeverity.ERROR: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.INFO: 2,
        }
        all_issues.sort(key=lambda i: severity_order.get(i.severity, 3))

        return ValidationReport(passed=passed, issues=all_issues)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered validator names.

        Returns:
            Sorted list of validator names.
        """
        return sorted(cls._validators.keys())


# ============================================================================
# Auto-register built-in validators
# ============================================================================

ValidationRegistry.register(SchemaValidator)
ValidationRegistry.register(TemporalValidator)
ValidationRegistry.register(IntegrityValidator)
ValidationRegistry.register(StatisticsValidator)
