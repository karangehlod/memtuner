"""Comprehensive tests for ProgressMonitor."""

import pytest
import time
from benchmark.memory.distributed.progress_monitor import (
    ProgressMonitor,
    ProgressInfo,
    WorkerHealth,
)


@pytest.fixture
def monitor():
    """Create progress monitor instance."""
    return ProgressMonitor()


class TestMonitorInitialization:
    """Test monitor initialization."""

    def test_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor is not None
        assert len(monitor.get_active_jobs()) == 0


class TestJobTracking:
    """Test job tracking."""

    def test_start_job(self, monitor):
        """Test starting a job."""
        monitor.start_job("job_1", total_items=100)

        assert len(monitor.get_active_jobs()) == 1
        progress = monitor.get_progress("job_1")
        assert progress is not None
        assert progress.total_items == 100

    def test_start_multiple_jobs(self, monitor):
        """Test starting multiple jobs."""
        monitor.start_job("job_1", total_items=100)
        monitor.start_job("job_2", total_items=200)

        assert len(monitor.get_active_jobs()) == 2

    def test_clear_job(self, monitor):
        """Test clearing a job."""
        monitor.start_job("job_1", total_items=100)
        monitor.clear_job("job_1")

        assert len(monitor.get_active_jobs()) == 0


class TestProgressRecording:
    """Test recording progress."""

    def test_record_progress_basic(self, monitor):
        """Test recording basic progress."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=50)

        progress = monitor.get_progress("job_1")
        assert progress.completed_items == 50
        assert progress.progress_percent == 50.0

    def test_record_progress_with_failures(self, monitor):
        """Test recording progress with failures."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=80, failed=10)

        progress = monitor.get_progress("job_1")
        assert progress.completed_items == 80
        assert progress.failed_items == 10

    def test_progress_percentage_calculation(self, monitor):
        """Test progress percentage calculation."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=25)

        progress = monitor.get_progress("job_1")
        assert progress.progress_percent == 25.0

    def test_throughput_calculation(self, monitor):
        """Test throughput calculation."""
        monitor.start_job("job_1", total_items=100)
        time.sleep(0.1)  # Small delay to ensure time passes
        monitor.record_progress("job_1", completed=50)

        progress = monitor.get_progress("job_1")
        assert progress.throughput > 0


class TestETAEstimation:
    """Test ETA estimation."""

    def test_estimate_completion_time(self, monitor):
        """Test ETA estimation."""
        monitor.start_job("job_1", total_items=100)
        time.sleep(0.1)
        monitor.record_progress("job_1", completed=50)

        eta = monitor.estimate_completion_time("job_1")
        # ETA should be roughly equal to elapsed time (50 more items at same rate)
        assert eta is not None
        assert eta > 0

    def test_eta_zero_when_complete(self, monitor):
        """Test ETA is zero when complete."""
        monitor.start_job("job_1", total_items=100)
        time.sleep(0.05)
        monitor.record_progress("job_1", completed=100)

        eta = monitor.estimate_completion_time("job_1")
        assert eta == 0.0

    def test_eta_none_before_progress(self, monitor):
        """Test ETA is None before any progress."""
        monitor.start_job("job_1", total_items=100)

        eta = monitor.estimate_completion_time("job_1")
        # Before any progress, throughput is 0
        assert eta is None

    def test_eta_string_format(self, monitor):
        """Test ETA string formatting."""
        monitor.start_job("job_1", total_items=100)
        time.sleep(0.1)
        monitor.record_progress("job_1", completed=50)

        eta_str = monitor.get_estimated_completion_time_str("job_1")
        assert isinstance(eta_str, str)
        assert eta_str != "Unknown"


class TestWorkerTracking:
    """Test worker tracking."""

    def test_record_worker_progress_healthy(self, monitor):
        """Test recording healthy worker progress."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=50)

        health = monitor.get_worker_health("job_1")
        assert 1 in health
        assert health[1].status == "healthy"

    def test_record_worker_progress_degraded(self, monitor):
        """Test recording degraded worker progress."""
        monitor.start_job("job_1", total_items=100)
        # 15% error rate = degraded
        monitor.record_worker_progress(
            "job_1",
            worker_id=1,
            items_processed=85,
            errors=15,
        )

        health = monitor.get_worker_health("job_1")
        assert health[1].status == "degraded"

    def test_record_worker_progress_failed(self, monitor):
        """Test recording failed worker progress."""
        monitor.start_job("job_1", total_items=100)
        # 60% error rate = failed
        monitor.record_worker_progress(
            "job_1",
            worker_id=1,
            items_processed=40,
            errors=60,
        )

        health = monitor.get_worker_health("job_1")
        assert health[1].status == "failed"

    def test_multiple_workers(self, monitor):
        """Test tracking multiple workers."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=30)
        monitor.record_worker_progress("job_1", worker_id=2, items_processed=40)
        monitor.record_worker_progress("job_1", worker_id=3, items_processed=30)

        health = monitor.get_worker_health("job_1")
        assert len(health) == 3


class TestJobStatusSummary:
    """Test job status summary."""

    def test_status_summary_basic(self, monitor):
        """Test basic status summary."""
        monitor.start_job("job_1", total_items=100)
        time.sleep(0.05)
        monitor.record_progress("job_1", completed=50)

        summary = monitor.get_job_status_summary("job_1")

        assert summary["job_id"] == "job_1"
        assert summary["total_items"] == 100
        assert summary["completed_items"] == 50
        assert summary["progress_percent"] == 50.0

    def test_status_summary_with_workers(self, monitor):
        """Test status summary includes worker info."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=50)
        monitor.record_worker_progress("job_1", worker_id=2, items_processed=40)

        summary = monitor.get_job_status_summary("job_1")

        assert summary["workers_total"] == 2
        assert summary["workers_healthy"] == 2


class TestJobCompletion:
    """Test job completion detection."""

    def test_is_job_complete_false(self, monitor):
        """Test incomplete job detection."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=50)

        assert not monitor.is_job_complete("job_1")

    def test_is_job_complete_true(self, monitor):
        """Test complete job detection."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=100)

        assert monitor.is_job_complete("job_1")

    def test_is_job_complete_over(self, monitor):
        """Test job with more items completed than total."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=105)

        assert monitor.is_job_complete("job_1")


class TestJobHealth:
    """Test job health monitoring."""

    def test_is_job_healthy_all_healthy(self, monitor):
        """Test healthy job with all workers healthy."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=50)
        monitor.record_worker_progress("job_1", worker_id=2, items_processed=50)

        assert monitor.is_job_healthy("job_1")

    def test_is_job_healthy_with_failed_worker(self, monitor):
        """Test unhealthy job with failed worker."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=50)
        # Failed worker
        monitor.record_worker_progress(
            "job_1",
            worker_id=2,
            items_processed=10,
            errors=90,
        )

        assert not monitor.is_job_healthy("job_1")

    def test_is_job_healthy_degraded_ok(self, monitor):
        """Test job health with degraded workers is still considered healthy."""
        monitor.start_job("job_1", total_items=100)
        # Degraded worker (not failed)
        monitor.record_worker_progress(
            "job_1",
            worker_id=1,
            items_processed=85,
            errors=15,
        )

        assert monitor.is_job_healthy("job_1")


class TestBottleneckDetection:
    """Test bottleneck detection."""

    def test_get_bottleneck_worker(self, monitor):
        """Test identifying bottleneck worker."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress(
            "job_1",
            worker_id=1,
            items_processed=50,
            latency_ms=10.0,
        )
        monitor.record_worker_progress(
            "job_1",
            worker_id=2,
            items_processed=50,
            latency_ms=50.0,  # Slowest
        )

        bottleneck = monitor.get_bottleneck_worker("job_1")
        assert bottleneck == 2

    def test_bottleneck_none_when_no_workers(self, monitor):
        """Test bottleneck is None when no workers."""
        monitor.start_job("job_1", total_items=100)

        bottleneck = monitor.get_bottleneck_worker("job_1")
        assert bottleneck is None


class TestThroughput:
    """Test throughput monitoring."""

    def test_get_total_throughput(self, monitor):
        """Test getting total throughput."""
        monitor.start_job("job_1", total_items=100)
        time.sleep(0.1)
        monitor.record_progress("job_1", completed=50)

        throughput = monitor.get_total_throughput("job_1")
        assert throughput > 0

    def test_throughput_zero_before_progress(self, monitor):
        """Test throughput is zero before progress."""
        monitor.start_job("job_1", total_items=100)

        throughput = monitor.get_total_throughput("job_1")
        assert throughput == 0


class TestWorkerUtilization:
    """Test worker utilization computation."""

    def test_compute_utilization_equal(self, monitor):
        """Test utilization with equal work distribution."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=50)
        monitor.record_worker_progress("job_1", worker_id=2, items_processed=50)

        util = monitor.compute_worker_utilization("job_1")

        assert util[1] == 50.0
        assert util[2] == 50.0

    def test_compute_utilization_unequal(self, monitor):
        """Test utilization with unequal work distribution."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=30)
        monitor.record_worker_progress("job_1", worker_id=2, items_processed=70)

        util = monitor.compute_worker_utilization("job_1")

        assert util[1] == 30.0
        assert util[2] == 70.0

    def test_compute_utilization_empty(self, monitor):
        """Test utilization with no workers."""
        monitor.start_job("job_1", total_items=100)

        util = monitor.compute_worker_utilization("job_1")
        assert util == {}


class TestProgressAccuracy:
    """Test progress calculation accuracy."""

    def test_progress_starts_at_zero(self, monitor):
        """Test progress starts at 0%."""
        monitor.start_job("job_1", total_items=100)
        progress = monitor.get_progress("job_1")

        assert progress.progress_percent == 0.0

    def test_progress_increases(self, monitor):
        """Test progress increases with updates."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=25)
        p1 = monitor.get_progress("job_1").progress_percent

        monitor.record_progress("job_1", completed=75)
        p2 = monitor.get_progress("job_1").progress_percent

        assert p1 < p2
        assert p2 == 75.0


class TestIntegration:
    """Integration tests."""

    def test_full_job_lifecycle(self, monitor):
        """Test full job lifecycle."""
        # Start job
        monitor.start_job("job_1", total_items=100)
        assert len(monitor.get_active_jobs()) == 1

        # Record worker progress
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=40)
        monitor.record_worker_progress("job_1", worker_id=2, items_processed=40)

        # Record overall progress
        time.sleep(0.05)
        monitor.record_progress("job_1", completed=80)

        # Check status
        progress = monitor.get_progress("job_1")
        assert progress.progress_percent == 80.0

        # Check summary
        summary = monitor.get_job_status_summary("job_1")
        assert summary["workers_total"] == 2

        # Complete job
        monitor.record_progress("job_1", completed=100)
        assert monitor.is_job_complete("job_1")

        # Clear job
        monitor.clear_job("job_1")
        assert len(monitor.get_active_jobs()) == 0

    def test_multiple_concurrent_jobs(self, monitor):
        """Test tracking multiple concurrent jobs."""
        monitor.start_job("job_1", total_items=100)
        monitor.start_job("job_2", total_items=200)

        monitor.record_progress("job_1", completed=50)
        monitor.record_progress("job_2", completed=100)

        p1 = monitor.get_progress("job_1")
        p2 = monitor.get_progress("job_2")

        assert p1.progress_percent == 50.0
        assert p2.progress_percent == 50.0  # Same percentage but different absolute values

    def test_job_with_worker_failures(self, monitor):
        """Test job with worker failures."""
        monitor.start_job("job_1", total_items=100)

        # Worker 1: healthy
        monitor.record_worker_progress("job_1", worker_id=1, items_processed=50)

        # Worker 2: degraded
        monitor.record_worker_progress(
            "job_1",
            worker_id=2,
            items_processed=30,
            errors=10,
        )

        # Worker 3: failed
        monitor.record_worker_progress(
            "job_1",
            worker_id=3,
            items_processed=5,
            errors=45,
        )

        summary = monitor.get_job_status_summary("job_1")
        assert summary["workers_healthy"] == 1
        assert summary["workers_degraded"] == 1
        assert summary["workers_failed"] == 1
        assert not monitor.is_job_healthy("job_1")


class TestEdgeCases:
    """Test edge cases."""

    def test_unknown_job_id(self, monitor):
        """Test operations on unknown job ID."""
        progress = monitor.get_progress("unknown_job")
        assert progress is None

    def test_zero_total_items(self, monitor):
        """Test job with zero total items."""
        monitor.start_job("job_1", total_items=0)
        # Progress should still work even with 0 total
        progress = monitor.get_progress("job_1")
        assert progress.total_items == 0

    def test_progress_exceeds_total(self, monitor):
        """Test when progress exceeds total items."""
        monitor.start_job("job_1", total_items=100)
        monitor.record_progress("job_1", completed=150)

        progress = monitor.get_progress("job_1")
        # Progress percentage should be capped or handled gracefully
        assert progress.progress_percent >= 100.0
