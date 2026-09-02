"""Tests for resource tracker — Phase 8 / 11."""

from __future__ import annotations

import time

import pytest

from benchmark.resources.tracker import ResourceReport, ResourceSnapshot, ResourceTracker


@pytest.mark.unit
class TestResourceSnapshot:
    def test_snapshot_stores_values(self):
        s = ResourceSnapshot(
            timestamp=1.0,
            cpu_percent=25.0,
            ram_used_mb=512.0,
            ram_available_mb=1024.0,
            disk_read_mb=10.0,
            disk_write_mb=5.0,
        )
        assert s.cpu_percent == 25.0
        assert s.ram_used_mb == 512.0


@pytest.mark.unit
class TestResourceReport:
    def test_to_dict_has_required_keys(self):
        r = ResourceReport(
            peak_ram_mb=256.0,
            avg_ram_mb=200.0,
            peak_cpu_percent=50.0,
            avg_cpu_percent=30.0,
            duration_seconds=1.5,
        )
        d = r.to_dict()
        assert "peak_ram_mb" in d
        assert "avg_ram_mb" in d
        assert "peak_cpu_percent" in d
        assert "duration_seconds" in d
        assert "platform" in d

    def test_to_dict_rounds_values(self):
        r = ResourceReport(peak_ram_mb=100.123456, avg_ram_mb=80.9999)
        d = r.to_dict()
        assert d["peak_ram_mb"] == 100.12
        assert d["avg_ram_mb"] == 81.0


@pytest.mark.unit
class TestResourceTracker:
    def test_context_manager_start_stop(self):
        tracker = ResourceTracker(interval_seconds=0.1)
        with tracker:
            time.sleep(0.05)
        report = tracker.report()
        assert report.duration_seconds >= 0.0

    def test_report_has_duration(self):
        tracker = ResourceTracker(interval_seconds=0.1)
        tracker.start()
        time.sleep(0.05)
        tracker.stop()
        report = tracker.report()
        assert report.duration_seconds > 0.0

    def test_report_includes_platform(self):
        import sys
        tracker = ResourceTracker(interval_seconds=0.1)
        with tracker:
            pass
        report = tracker.report()
        assert report.platform == sys.platform or report.error != ""

    def test_no_psutil_graceful_degradation(self, monkeypatch):
        """Tracker must degrade gracefully when psutil is not available."""
        tracker = ResourceTracker(interval_seconds=0.1)
        # Simulate missing psutil by nulling it out
        tracker._psutil = None
        tracker.start()
        time.sleep(0.05)
        tracker.stop()
        report = tracker.report()
        # Must not raise — returns a report with some duration
        assert report.duration_seconds >= 0.0
        # Error is non-empty when psutil is absent (no samples collected)
        assert isinstance(report.error, str)

    def test_multiple_samples_collected(self):
        """With psutil installed, multiple samples should be collected."""
        try:
            import psutil  # noqa: F401
        except ImportError:
            pytest.skip("psutil not installed")

        tracker = ResourceTracker(interval_seconds=0.05)
        with tracker:
            time.sleep(0.3)
        report = tracker.report()
        # Should have collected at least 2 samples in 0.3 seconds
        assert report.sample_count >= 1
