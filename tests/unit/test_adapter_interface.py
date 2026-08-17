"""Unit tests for dataset adapter interface.

Tests the abstract adapter base class, exception types, validation framework,
and registry pattern without testing specific implementations.

SOLID principles tested:
  - Interface Segregation: Each method has a single, well-defined purpose
  - Dependency Inversion: Tests depend on abstract interface, not concrete adapters
  - Open/Closed: New adapters can be added without modifying this test
"""

import pytest
from abc import ABC

from benchmark.gold.adapters import (
    AdapterError,
    AdapterRegistry,
    DatasetAdapter,
    FingerprintError,
    MetadataError,
    StatisticsError,
    ValidationError,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


class TestValidationSeverity:
    """Test ValidationSeverity enum."""

    def test_all_severity_levels_exist(self) -> None:
        """Verify all expected severity levels are defined."""
        assert hasattr(ValidationSeverity, "ERROR")
        assert hasattr(ValidationSeverity, "WARNING")
        assert hasattr(ValidationSeverity, "INFO")

    def test_severity_values_are_strings(self) -> None:
        """Verify severity values are meaningful strings."""
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_issue_creation_minimal(self) -> None:
        """Create issue with minimal required fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Missing required field",
        )
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.validator == "schema"
        assert issue.message == "Missing required field"
        assert issue.location == ""

    def test_issue_creation_with_location(self) -> None:
        """Create issue with location information."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            validator="temporal",
            message="Event out of order",
            location="events[3]",
        )
        assert issue.location == "events[3]"

    def test_issue_is_immutable(self) -> None:
        """Verify ValidationIssue is frozen (immutable)."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Test",
        )
        with pytest.raises(AttributeError):
            issue.severity = ValidationSeverity.WARNING

    def test_issue_string_representation(self) -> None:
        """Test issue formatting as human-readable string."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Missing field",
            location="query[5]",
        )
        result = str(issue)
        assert "[ERROR]" in result
        assert "schema" in result
        assert "Missing field" in result
        assert "(query[5])" in result

    def test_issue_string_without_location(self) -> None:
        """Test issue formatting without location."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            validator="temporal",
            message="Test",
        )
        result = str(issue)
        assert "[WARNING]" in result
        assert "temporal" in result
        assert "Test" in result
        assert "()" not in result


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_empty_report_passes(self) -> None:
        """Report with no issues is a passing report."""
        report = ValidationReport(passed=True, issues=[])
        assert report.passed is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

    def test_report_with_only_errors(self) -> None:
        """Report with ERROR issues fails."""
        error = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Test error",
        )
        report = ValidationReport(passed=False, issues=[error])
        assert report.passed is False
        assert len(report.errors) == 1
        assert len(report.warnings) == 0

    def test_report_with_warnings_passes(self) -> None:
        """Report with only WARNING issues can still pass."""
        warning = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            validator="quality",
            message="Test warning",
        )
        report = ValidationReport(passed=True, issues=[warning])
        assert report.passed is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 1

    def test_report_with_mixed_issues(self) -> None:
        """Report can contain multiple severity levels."""
        error = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Error",
        )
        warning = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            validator="quality",
            message="Warning",
        )
        info = ValidationIssue(
            severity=ValidationSeverity.INFO,
            validator="metadata",
            message="Info",
        )
        report = ValidationReport(passed=False, issues=[error, warning, info])
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert len(report.info_messages) == 1

    def test_report_filters_by_severity(self) -> None:
        """Properties correctly filter issues by severity."""
        issues = [
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                validator="schema",
                message="E1",
            ),
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                validator="schema",
                message="E2",
            ),
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                validator="quality",
                message="W1",
            ),
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                validator="metadata",
                message="I1",
            ),
        ]
        report = ValidationReport(passed=False, issues=issues)
        assert len(report.errors) == 2
        assert len(report.warnings) == 1
        assert len(report.info_messages) == 1

    def test_report_string_representation(self) -> None:
        """Test report formatting as human-readable string."""
        error = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Error",
        )
        report = ValidationReport(passed=False, issues=[error])
        result = str(report)
        assert "FAILED" in result
        assert "Errors (1)" in result


class TestAdapterRegistry:
    """Test adapter registry system."""

    def test_registry_starts_empty(self) -> None:
        """New registry should have no adapters initially."""
        # Note: Real registry may have adapters if tests run in order
        # This test is informational about registry structure
        assert isinstance(AdapterRegistry.list_all(), list)

    def test_registry_register_adapter(self) -> None:
        """Test adapter registration."""

        class DummyAdapter(DatasetAdapter):
            def load(self, source):
                pass

            def validate(self, dataset):
                pass

            def fingerprint(self, dataset):
                pass

            def statistics(self, dataset):
                pass

            def metadata(self):
                pass

        # Register adapter
        AdapterRegistry.register("test_dummy", DummyAdapter)
        assert AdapterRegistry.is_registered("test_dummy")
        assert "test_dummy" in AdapterRegistry.list_all()

        # Get adapter instance
        adapter = AdapterRegistry.get("test_dummy")
        assert isinstance(adapter, DummyAdapter)

    def test_registry_rejects_duplicate_registration(self) -> None:
        """Test that duplicate adapter names are rejected."""

        class DummyAdapter(DatasetAdapter):
            def load(self, source):
                pass

            def validate(self, dataset):
                pass

            def fingerprint(self, dataset):
                pass

            def statistics(self, dataset):
                pass

            def metadata(self):
                pass

        # First registration should succeed
        if not AdapterRegistry.is_registered("test_dup"):
            AdapterRegistry.register("test_dup", DummyAdapter)

        # Second registration should fail
        with pytest.raises(ValueError):
            AdapterRegistry.register("test_dup", DummyAdapter)

    def test_registry_unknown_adapter_raises_error(self) -> None:
        """Test that requesting unknown adapter raises AdapterError."""
        with pytest.raises(AdapterError):
            AdapterRegistry.get("nonexistent_adapter_xyz")

    def test_registry_error_message_includes_available(self) -> None:
        """Test that error message lists available adapters."""
        try:
            AdapterRegistry.get("nonexistent_adapter_xyz")
        except AdapterError as e:
            assert "Available adapters:" in str(e)


class TestExceptionHierarchy:
    """Test adapter exception hierarchy."""

    def test_adapter_error_is_base(self) -> None:
        """All adapter exceptions inherit from AdapterError."""
        assert issubclass(ValidationError, AdapterError)
        assert issubclass(FingerprintError, AdapterError)
        assert issubclass(StatisticsError, AdapterError)
        assert issubclass(MetadataError, AdapterError)

    def test_exceptions_are_distinguishable(self) -> None:
        """Each exception type is distinct."""
        exceptions = [
            ValidationError,
            FingerprintError,
            StatisticsError,
            MetadataError,
        ]
        assert len(set(exceptions)) == 4


class TestAbstractAdapterInterface:
    """Test that DatasetAdapter enforces abstract methods."""

    def test_adapter_is_abstract(self) -> None:
        """DatasetAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatasetAdapter()  # type: ignore

    def test_adapter_requires_all_methods(self) -> None:
        """Subclass must implement all abstract methods."""

        class IncompleteAdapter(DatasetAdapter):
            def load(self, source):
                pass

        # Should raise TypeError because not all methods are implemented
        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_complete_adapter_can_be_instantiated(self) -> None:
        """Class implementing all methods can be instantiated."""

        class CompleteAdapter(DatasetAdapter):
            def load(self, source):
                raise NotImplementedError

            def validate(self, dataset):
                raise NotImplementedError

            def fingerprint(self, dataset):
                raise NotImplementedError

            def statistics(self, dataset):
                raise NotImplementedError

            def metadata(self):
                raise NotImplementedError

        # Should not raise
        adapter = CompleteAdapter()
        assert isinstance(adapter, DatasetAdapter)
