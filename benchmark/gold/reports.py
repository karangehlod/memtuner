"""Dataset validation and statistics report generation.

Generates comprehensive reports about datasets in multiple formats:
  - JSON reports for programmatic access
  - HTML reports for human review
  - Summary reports for console output

SOLID principles:
  - Single Responsibility: Each reporter handles one format
  - Open/Closed: New formats can be added without modifying existing code
  - Interface Segregation: Reporters expose minimal, focused interfaces
  - Dependency Inversion: Consumers depend on abstract Reporter interface
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.gold.schema import GoldDataset
from benchmark.gold.adapters import ValidationReport
from benchmark.gold.statistics import DatasetStatistics, StatisticsFormatter


# ============================================================================
# Report Data Container
# ============================================================================


class DatasetReport:
    """Complete report for a dataset with validation and statistics."""

    def __init__(
        self,
        dataset_name: str,
        validation_report: ValidationReport,
        statistics: DatasetStatistics,
        fingerprint: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize report.

        Args:
            dataset_name: Name of the dataset.
            validation_report: Validation results.
            statistics: Dataset statistics.
            fingerprint: Optional dataset fingerprint for reproducibility.
            metadata: Optional additional metadata.
        """
        self.dataset_name = dataset_name
        self.validation_report = validation_report
        self.statistics = statistics
        self.fingerprint = fingerprint
        self.metadata = metadata or {}
        self.generated_at = datetime.now().isoformat()


# ============================================================================
# Abstract Reporter
# ============================================================================


class Reporter(ABC):
    """Base class for dataset report generators."""

    @abstractmethod
    def generate(self, report: DatasetReport) -> Any:
        """Generate report in format-specific representation.

        Args:
            report: Report data to format.

        Returns:
            Format-specific representation (str, dict, etc).
        """
        pass


# ============================================================================
# JSON Reporter
# ============================================================================


class JSONReporter(Reporter):
    """Generate JSON reports."""

    def generate(self, report: DatasetReport) -> dict[str, Any]:
        """Generate JSON-serializable report.

        Args:
            report: Report data to format.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "metadata": {
                "dataset_name": report.dataset_name,
                "generated_at": report.generated_at,
                "fingerprint": report.fingerprint,
                **report.metadata,
            },
            "validation": {
                "passed": report.validation_report.passed,
                "error_count": len(report.validation_report.errors),
                "warning_count": len(report.validation_report.warnings),
                "info_count": len(report.validation_report.info_messages),
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "validator": issue.validator,
                        "message": issue.message,
                        "location": issue.location,
                    }
                    for issue in report.validation_report.issues
                ],
            },
            "statistics": StatisticsFormatter.format_json(report.statistics),
        }


# ============================================================================
# HTML Reporter
# ============================================================================


class HTMLReporter(Reporter):
    """Generate HTML reports."""

    def generate(self, report: DatasetReport) -> str:
        """Generate HTML report.

        Args:
            report: Report data to format.

        Returns:
            HTML string.
        """
        validation_status = (
            "✓ PASSED" if report.validation_report.passed else "✗ FAILED"
        )
        validation_status_color = "green" if report.validation_report.passed else "red"

        issues_html = self._format_issues(report.validation_report.issues)
        stats_html = self._format_statistics(report.statistics)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dataset Report: {report.dataset_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #222;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #0066cc;
            margin-top: 30px;
        }}
        .status {{
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }}
        .status-pass {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-fail {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .metadata {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #0066cc;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .issue-error {{
            background-color: #ffe6e6;
        }}
        .issue-warning {{
            background-color: #fff3cd;
        }}
        .issue-info {{
            background-color: #d1ecf1;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .stat-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #0066cc;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }}
    </style>
</head>
<body>
    <h1>Dataset Report: {report.dataset_name}</h1>

    <div class="metadata">
        <strong>Generated:</strong> {report.generated_at}<br>
        <strong>Fingerprint:</strong> <code>{report.fingerprint}</code>
    </div>

    <h2>Validation Status</h2>
    <div class="status status-{'pass' if report.validation_report.passed else 'fail'}">
        {validation_status}
    </div>
    <p>
        Errors: {len(report.validation_report.errors)} |
        Warnings: {len(report.validation_report.warnings)} |
        Info: {len(report.validation_report.info_messages)}
    </p>

    {issues_html}

    <h2>Statistics</h2>
    {stats_html}

</body>
</html>
"""
        return html

    @staticmethod
    def _format_issues(issues: list) -> str:
        """Format validation issues as HTML table."""
        if not issues:
            return "<p>No validation issues found.</p>"

        rows = []
        for issue in issues:
            severity_class = f"issue-{issue.severity.value}"
            rows.append(
                f"""
            <tr class="{severity_class}">
                <td><strong>{issue.severity.value.upper()}</strong></td>
                <td>{issue.validator}</td>
                <td>{issue.message}</td>
                <td><code>{issue.location}</code></td>
            </tr>
            """
            )

        return f"""
    <h3>Validation Issues</h3>
    <table>
        <thead>
            <tr>
                <th>Severity</th>
                <th>Validator</th>
                <th>Message</th>
                <th>Location</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

    @staticmethod
    def _format_statistics(stats: DatasetStatistics) -> str:
        """Format statistics as HTML."""
        counts = stats.counts
        quality = stats.quality

        return f"""
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-label">Queries</div>
            <div class="stat-value">{counts.query_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Memories</div>
            <div class="stat-value">{counts.memory_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Users</div>
            <div class="stat-value">{counts.user_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Days</div>
            <div class="stat-value">{counts.day_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Avg Memories/Day</div>
            <div class="stat-value">{counts.avg_memories_per_day:.1f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Entity Coverage</div>
            <div class="stat-value">{quality.entity_coverage:.0%}</div>
        </div>
    </div>

    <h3>Detailed Statistics</h3>
    <pre>{StatisticsFormatter.format_summary(stats)}</pre>
    """


# ============================================================================
# Text Reporter (Console Output)
# ============================================================================


class TextReporter(Reporter):
    """Generate plain text reports for console output."""

    def generate(self, report: DatasetReport) -> str:
        """Generate text report.

        Args:
            report: Report data to format.

        Returns:
            Plain text report.
        """
        lines = [
            f"Dataset Report: {report.dataset_name}",
            "=" * 70,
            "",
            f"Generated: {report.generated_at}",
            f"Fingerprint: {report.fingerprint}",
            "",
            "VALIDATION",
            "-" * 70,
        ]

        if report.validation_report.passed:
            lines.append("✓ Validation PASSED")
        else:
            lines.append("✗ Validation FAILED")

        lines.extend([
            f"  Errors: {len(report.validation_report.errors)}",
            f"  Warnings: {len(report.validation_report.warnings)}",
            f"  Info: {len(report.validation_report.info_messages)}",
            "",
        ])

        if report.validation_report.issues:
            lines.append("Issues:")
            for issue in report.validation_report.issues:
                lines.append(f"  {issue}")

        lines.extend([
            "",
            "STATISTICS",
            "-" * 70,
            StatisticsFormatter.format_summary(report.statistics),
        ])

        return "\n".join(lines)


# ============================================================================
# Report Generator (Facade)
# ============================================================================


class ReportGenerator:
    """Generate and save reports in multiple formats."""

    _reporters = {
        "json": JSONReporter(),
        "html": HTMLReporter(),
        "text": TextReporter(),
    }

    @classmethod
    def generate(
        self,
        report: DatasetReport,
        format: str = "json",
    ) -> Any:
        """Generate report in specified format.

        Args:
            report: Report to generate.
            format: Output format ("json", "html", "text").

        Returns:
            Format-specific representation.

        Raises:
            ValueError: If format is not supported.
        """
        if format not in self._reporters:
            available = ", ".join(self._reporters.keys())
            raise ValueError(
                f"Unknown format '{format}'. Available: {available}"
            )

        reporter = self._reporters[format]
        return reporter.generate(report)

    @classmethod
    def save_json(
        self,
        report: DatasetReport,
        path: Path | str,
    ) -> None:
        """Generate and save JSON report.

        Args:
            report: Report to save.
            path: Output file path.
        """
        data = self.generate(report, format="json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def save_html(
        self,
        report: DatasetReport,
        path: Path | str,
    ) -> None:
        """Generate and save HTML report.

        Args:
            report: Report to save.
            path: Output file path.
        """
        html = self.generate(report, format="html")
        with open(path, "w") as f:
            f.write(html)

    @classmethod
    def save_text(
        self,
        report: DatasetReport,
        path: Path | str,
    ) -> None:
        """Generate and save text report.

        Args:
            report: Report to save.
            path: Output file path.
        """
        text = self.generate(report, format="text")
        with open(path, "w") as f:
            f.write(text)
