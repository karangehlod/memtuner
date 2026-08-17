"""Unit tests for dataset report generation.

Tests all report formats and the report generation facade.

SOLID principles tested:
  - Single Responsibility: Each reporter handles one format
  - Open/Closed: New formats can be added without modifying existing code
  - Interface Segregation: Reporters expose minimal interfaces
  - Dependency Inversion: Tests depend on Reporter interface
"""

import json
import tempfile
from pathlib import Path

import pytest

from benchmark.gold.adapters import ValidationIssue, ValidationReport, ValidationSeverity
from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldMemoryEvent,
    GoldQuery,
    GoldExpectedResult,
)
from benchmark.gold.statistics import StatisticsComputer
from benchmark.gold.reports import (
    DatasetReport,
    Reporter,
    JSONReporter,
    HTMLReporter,
    TextReporter,
    ReportGenerator,
)
from benchmark.models.memory_event import MemoryType


# ============================================================================
# Test Helpers
# ============================================================================


def make_dataset() -> GoldDataset:
    """Create a simple test dataset."""
    memory = GoldMemoryEvent(
        id="mem1",
        user_id="user1",
        type=MemoryType.EPISODIC,
        content="test memory",
        importance=0.5,
        entities=["alice"],
        task_id="task1",
    )
    day_events = GoldDayEvents(day=0, memory_events=[memory])

    expected = GoldExpectedResult(memory_ids=["mem1"])
    query = GoldQuery(
        day=0,
        query="test query",
        task_id="task1",
        user_id="user1",
        expected=expected,
    )

    return GoldDataset(
        scenario="test",
        description="test dataset",
        events=[day_events],
        queries=[query],
    )


def make_report() -> DatasetReport:
    """Create a test report."""
    dataset = make_dataset()
    validation = ValidationReport(passed=True, issues=[])
    statistics = StatisticsComputer.compute(dataset)

    return DatasetReport(
        dataset_name="test_dataset",
        validation_report=validation,
        statistics=statistics,
        fingerprint="abc123def456",
    )


# ============================================================================
# DatasetReport Tests
# ============================================================================


class TestDatasetReport:
    """Test DatasetReport container."""

    def test_create_report(self) -> None:
        """Create a simple report."""
        report = make_report()
        assert report.dataset_name == "test_dataset"
        assert report.fingerprint == "abc123def456"
        assert report.validation_report.passed
        assert report.generated_at is not None

    def test_report_with_metadata(self) -> None:
        """Report can include custom metadata."""
        report = make_report()
        report.metadata["custom_field"] = "custom_value"
        assert report.metadata["custom_field"] == "custom_value"

    def test_report_with_issues(self) -> None:
        """Report can include validation issues."""
        dataset = make_dataset()
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            validator="test",
            message="Test warning",
        )
        validation = ValidationReport(passed=True, issues=[issue])
        statistics = StatisticsComputer.compute(dataset)

        report = DatasetReport(
            dataset_name="test",
            validation_report=validation,
            statistics=statistics,
        )

        assert len(report.validation_report.issues) == 1


# ============================================================================
# JSON Reporter Tests
# ============================================================================


class TestJSONReporter:
    """Test JSON report generation."""

    def test_generate_json(self) -> None:
        """Generate JSON report."""
        report = make_report()
        reporter = JSONReporter()
        json_report = reporter.generate(report)

        assert isinstance(json_report, dict)
        assert "metadata" in json_report
        assert "validation" in json_report
        assert "statistics" in json_report

    def test_json_metadata_section(self) -> None:
        """JSON report includes metadata."""
        report = make_report()
        reporter = JSONReporter()
        json_report = reporter.generate(report)

        assert json_report["metadata"]["dataset_name"] == "test_dataset"
        assert json_report["metadata"]["fingerprint"] == "abc123def456"
        assert "generated_at" in json_report["metadata"]

    def test_json_validation_section(self) -> None:
        """JSON report includes validation results."""
        report = make_report()
        reporter = JSONReporter()
        json_report = reporter.generate(report)

        validation = json_report["validation"]
        assert validation["passed"] is True
        assert validation["error_count"] == 0
        assert validation["warning_count"] == 0
        assert isinstance(validation["issues"], list)

    def test_json_with_validation_issues(self) -> None:
        """JSON report includes validation issues."""
        dataset = make_dataset()
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Test error",
            location="query[0]",
        )
        validation = ValidationReport(passed=False, issues=[issue])
        statistics = StatisticsComputer.compute(dataset)

        report = DatasetReport(
            dataset_name="test",
            validation_report=validation,
            statistics=statistics,
        )

        reporter = JSONReporter()
        json_report = reporter.generate(report)

        assert json_report["validation"]["passed"] is False
        assert json_report["validation"]["error_count"] == 1
        assert len(json_report["validation"]["issues"]) == 1
        assert json_report["validation"]["issues"][0]["severity"] == "error"

    def test_json_statistics_section(self) -> None:
        """JSON report includes statistics."""
        report = make_report()
        reporter = JSONReporter()
        json_report = reporter.generate(report)

        stats = json_report["statistics"]
        assert "counts" in stats
        assert "distributions" in stats
        assert "quality" in stats
        assert stats["counts"]["queries"] > 0


# ============================================================================
# HTML Reporter Tests
# ============================================================================


class TestHTMLReporter:
    """Test HTML report generation."""

    def test_generate_html(self) -> None:
        """Generate HTML report."""
        report = make_report()
        reporter = HTMLReporter()
        html = reporter.generate(report)

        assert isinstance(html, str)
        assert "<html>" in html.lower()
        assert "</html>" in html.lower()
        assert "dataset report" in html.lower()

    def test_html_includes_title(self) -> None:
        """HTML includes dataset name in title."""
        report = make_report()
        reporter = HTMLReporter()
        html = reporter.generate(report)

        assert "test_dataset" in html

    def test_html_includes_validation_status(self) -> None:
        """HTML shows validation status."""
        report = make_report()
        reporter = HTMLReporter()
        html = reporter.generate(report)

        assert "PASSED" in html or "FAILED" in html

    def test_html_passes_status(self) -> None:
        """HTML shows PASSED status for valid dataset."""
        report = make_report()
        assert report.validation_report.passed

        reporter = HTMLReporter()
        html = reporter.generate(report)

        assert "PASSED" in html

    def test_html_fail_status(self) -> None:
        """HTML shows FAILED status for invalid dataset."""
        dataset = make_dataset()
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            validator="schema",
            message="Test error",
        )
        validation = ValidationReport(passed=False, issues=[issue])
        statistics = StatisticsComputer.compute(dataset)

        report = DatasetReport(
            dataset_name="test",
            validation_report=validation,
            statistics=statistics,
        )

        reporter = HTMLReporter()
        html = reporter.generate(report)

        assert "FAILED" in html

    def test_html_well_formed(self) -> None:
        """HTML is well-formed."""
        report = make_report()
        reporter = HTMLReporter()
        html = reporter.generate(report)

        # Check basic structure
        assert html.count("<table>") == html.count("</table>")
        assert html.count("<div") <= html.count("</div>") + 5  # Allow self-closing
        assert "<!" not in html or "DOCTYPE" in html


# ============================================================================
# Text Reporter Tests
# ============================================================================


class TestTextReporter:
    """Test text report generation."""

    def test_generate_text(self) -> None:
        """Generate text report."""
        report = make_report()
        reporter = TextReporter()
        text = reporter.generate(report)

        assert isinstance(text, str)
        assert "Dataset Report" in text
        assert "test_dataset" in text

    def test_text_includes_status(self) -> None:
        """Text includes validation status."""
        report = make_report()
        reporter = TextReporter()
        text = reporter.generate(report)

        assert "PASSED" in text or "FAILED" in text

    def test_text_includes_statistics(self) -> None:
        """Text includes statistics section."""
        report = make_report()
        reporter = TextReporter()
        text = reporter.generate(report)

        assert "STATISTICS" in text
        assert "Queries:" in text or "queries" in text.lower()

    def test_text_passed_status(self) -> None:
        """Text shows checkmark for passed validation."""
        report = make_report()
        reporter = TextReporter()
        text = reporter.generate(report)

        assert "✓" in text or "PASSED" in text

    def test_text_with_issues(self) -> None:
        """Text includes validation issues."""
        dataset = make_dataset()
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            validator="temporal",
            message="Event out of order",
        )
        validation = ValidationReport(passed=True, issues=[issue])
        statistics = StatisticsComputer.compute(dataset)

        report = DatasetReport(
            dataset_name="test",
            validation_report=validation,
            statistics=statistics,
        )

        reporter = TextReporter()
        text = reporter.generate(report)

        assert "Issues:" in text
        assert "WARNING" in text


# ============================================================================
# Report Generator Facade Tests
# ============================================================================


class TestReportGenerator:
    """Test the report generation facade."""

    def test_generate_json(self) -> None:
        """Facade generates JSON."""
        report = make_report()
        result = ReportGenerator.generate(report, format="json")
        assert isinstance(result, dict)

    def test_generate_html(self) -> None:
        """Facade generates HTML."""
        report = make_report()
        result = ReportGenerator.generate(report, format="html")
        assert isinstance(result, str)
        assert "<html>" in result.lower()

    def test_generate_text(self) -> None:
        """Facade generates text."""
        report = make_report()
        result = ReportGenerator.generate(report, format="text")
        assert isinstance(result, str)
        assert "Dataset Report" in result

    def test_unknown_format_raises_error(self) -> None:
        """Unknown format raises ValueError."""
        report = make_report()
        with pytest.raises(ValueError):
            ReportGenerator.generate(report, format="unknown")

    def test_save_json(self) -> None:
        """Facade saves JSON to file."""
        report = make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            ReportGenerator.save_json(report, path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "metadata" in data

    def test_save_html(self) -> None:
        """Facade saves HTML to file."""
        report = make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            ReportGenerator.save_html(report, path)

            assert path.exists()
            content = path.read_text()
            assert "<html>" in content.lower()

    def test_save_text(self) -> None:
        """Facade saves text to file."""
        report = make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.txt"
            ReportGenerator.save_text(report, path)

            assert path.exists()
            content = path.read_text()
            assert "Dataset Report" in content


# ============================================================================
# Integration Tests
# ============================================================================


class TestReportIntegration:
    """Test complete report generation flow."""

    def test_generate_all_formats(self) -> None:
        """Can generate all supported formats."""
        report = make_report()

        json_result = ReportGenerator.generate(report, format="json")
        html_result = ReportGenerator.generate(report, format="html")
        text_result = ReportGenerator.generate(report, format="text")

        assert json_result is not None
        assert html_result is not None
        assert text_result is not None

    def test_reports_consistent(self) -> None:
        """All formats report the same data."""
        report = make_report()

        json_result = ReportGenerator.generate(report, format="json")
        html_result = ReportGenerator.generate(report, format="html")
        text_result = ReportGenerator.generate(report, format="text")

        # Same dataset name appears in all formats
        assert "test_dataset" in str(json_result)
        assert "test_dataset" in html_result
        assert "test_dataset" in text_result

        # Same validation status in all formats
        if report.validation_report.passed:
            assert "PASSED" in html_result or "PASSED" in text_result

    def test_save_all_formats(self) -> None:
        """Can save all formats to disk."""
        report = make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            ReportGenerator.save_json(report, tmppath / "report.json")
            ReportGenerator.save_html(report, tmppath / "report.html")
            ReportGenerator.save_text(report, tmppath / "report.txt")

            assert (tmppath / "report.json").exists()
            assert (tmppath / "report.html").exists()
            assert (tmppath / "report.txt").exists()
