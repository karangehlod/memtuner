"""Monitor progress of distributed job execution."""

import logging
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Progress information for a job."""
    job_id: str
    total_items: int
    completed_items: int
    failed_items: int
    start_time: float
    last_update_time: float
    throughput: float = 0.0
    eta_seconds: Optional[float] = None
    progress_percent: float = 0.0
    worker_health: dict[int, str] = field(default_factory=dict)


@dataclass
class WorkerHealth:
    """Health status of a worker."""
    worker_id: int
    status: str  # "healthy", "degraded", "failed"
    last_heartbeat: float
    items_processed: int
    errors: int
    avg_latency_ms: float


class ProgressMonitor:
    """Monitor progress of distributed job execution."""

    def __init__(self):
        """Initialize progress monitor."""
        self._jobs: dict[str, ProgressInfo] = {}
        self._worker_health: dict[str, dict[int, WorkerHealth]] = {}
        self._start_times: dict[str, float] = {}

    def start_job(self, job_id: str, total_items: int) -> None:
        """Start tracking a new job.

        Args:
            job_id: Unique job identifier
            total_items: Total items to process
        """
        current_time = time.time()

        self._jobs[job_id] = ProgressInfo(
            job_id=job_id,
            total_items=total_items,
            completed_items=0,
            failed_items=0,
            start_time=current_time,
            last_update_time=current_time,
            worker_health={},
        )

        self._start_times[job_id] = current_time
        self._worker_health[job_id] = {}

        logger.info(f"Started job {job_id} with {total_items} items")

    def record_progress(
        self,
        job_id: str,
        completed: int,
        failed: int = 0,
    ) -> None:
        """Record progress update for job.

        Args:
            job_id: Job identifier
            completed: Number of completed items
            failed: Number of failed items
        """
        if job_id not in self._jobs:
            logger.warning(f"Job {job_id} not found in progress monitor")
            return

        current_time = time.time()
        progress = self._jobs[job_id]

        progress.completed_items = completed
        progress.failed_items = failed
        progress.last_update_time = current_time

        # Update throughput
        elapsed = current_time - progress.start_time
        if elapsed > 0:
            progress.throughput = completed / elapsed

        # Update progress percentage
        if progress.total_items > 0:
            progress.progress_percent = (completed / progress.total_items) * 100.0

        # Compute ETA
        if progress.throughput > 0:
            remaining = progress.total_items - completed
            progress.eta_seconds = remaining / progress.throughput
        else:
            progress.eta_seconds = None

    def record_worker_progress(
        self,
        job_id: str,
        worker_id: int,
        items_processed: int,
        errors: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Record progress from individual worker.

        Args:
            job_id: Job identifier
            worker_id: Worker identifier
            items_processed: Items processed by worker
            errors: Number of errors
            latency_ms: Average latency in milliseconds
        """
        if job_id not in self._worker_health:
            self._worker_health[job_id] = {}

        current_time = time.time()

        # Determine health status
        status = "healthy"
        if errors > 0:
            error_rate = errors / max(1, items_processed + errors)
            if error_rate > 0.1:
                status = "degraded"
            if error_rate > 0.5:
                status = "failed"

        self._worker_health[job_id][worker_id] = WorkerHealth(
            worker_id=worker_id,
            status=status,
            last_heartbeat=current_time,
            items_processed=items_processed,
            errors=errors,
            avg_latency_ms=latency_ms,
        )

        # Update job progress
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.worker_health[worker_id] = status

    def get_progress(self, job_id: str) -> Optional[ProgressInfo]:
        """Get current progress for job.

        Args:
            job_id: Job identifier

        Returns:
            ProgressInfo or None if job not found
        """
        return self._jobs.get(job_id)

    def estimate_completion_time(self, job_id: str) -> Optional[float]:
        """Estimate when job will complete.

        Args:
            job_id: Job identifier

        Returns:
            Estimated seconds to completion, or None if unavailable
        """
        progress = self._jobs.get(job_id)
        if not progress or progress.throughput <= 0:
            return None

        remaining = progress.total_items - progress.completed_items
        if remaining <= 0:
            return 0.0

        return remaining / progress.throughput

    def get_worker_health(self, job_id: str) -> dict[int, WorkerHealth]:
        """Get health status of all workers for job.

        Args:
            job_id: Job identifier

        Returns:
            Dict mapping worker_id to WorkerHealth
        """
        return self._worker_health.get(job_id, {})

    def get_job_status_summary(self, job_id: str) -> dict[str, Any]:
        """Get summary status of job.

        Args:
            job_id: Job identifier

        Returns:
            Status summary dict
        """
        progress = self._jobs.get(job_id)
        if not progress:
            return {}

        worker_health = self._worker_health.get(job_id, {})
        health_counts = {"healthy": 0, "degraded": 0, "failed": 0}

        for health in worker_health.values():
            health_counts[health.status] = health_counts.get(health.status, 0) + 1

        elapsed = time.time() - progress.start_time
        eta = self.estimate_completion_time(job_id)

        return {
            "job_id": job_id,
            "total_items": progress.total_items,
            "completed_items": progress.completed_items,
            "failed_items": progress.failed_items,
            "progress_percent": progress.progress_percent,
            "throughput_items_per_sec": progress.throughput,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "workers_total": len(worker_health),
            "workers_healthy": health_counts["healthy"],
            "workers_degraded": health_counts["degraded"],
            "workers_failed": health_counts["failed"],
        }

    def is_job_complete(self, job_id: str) -> bool:
        """Check if job is complete.

        Args:
            job_id: Job identifier

        Returns:
            True if all items processed
        """
        progress = self._jobs.get(job_id)
        if not progress:
            return False

        return progress.completed_items >= progress.total_items

    def is_job_healthy(self, job_id: str) -> bool:
        """Check if job is healthy.

        Args:
            job_id: Job identifier

        Returns:
            True if no failed workers
        """
        worker_health = self._worker_health.get(job_id, {})

        for health in worker_health.values():
            if health.status == "failed":
                return False

        return True

    def get_bottleneck_worker(self, job_id: str) -> Optional[int]:
        """Identify bottleneck worker (slowest).

        Args:
            job_id: Job identifier

        Returns:
            Worker ID with highest latency, or None
        """
        worker_health = self._worker_health.get(job_id, {})

        if not worker_health:
            return None

        slowest_worker = max(
            worker_health.items(),
            key=lambda x: x[1].avg_latency_ms,
        )[0]

        return slowest_worker

    def get_total_throughput(self, job_id: str) -> float:
        """Get total throughput across all workers.

        Args:
            job_id: Job identifier

        Returns:
            Total items per second
        """
        progress = self._jobs.get(job_id)
        if not progress:
            return 0.0

        return progress.throughput

    def clear_job(self, job_id: str) -> None:
        """Clear job tracking data.

        Args:
            job_id: Job identifier
        """
        self._jobs.pop(job_id, None)
        self._worker_health.pop(job_id, None)
        self._start_times.pop(job_id, None)

    def get_active_jobs(self) -> list[str]:
        """Get list of active job IDs.

        Returns:
            List of job IDs currently being tracked
        """
        return list(self._jobs.keys())

    def compute_worker_utilization(self, job_id: str) -> dict[int, float]:
        """Compute utilization percentage per worker.

        Args:
            job_id: Job identifier

        Returns:
            Dict mapping worker_id to utilization percentage (0-100)
        """
        worker_health = self._worker_health.get(job_id, {})
        progress = self._jobs.get(job_id)

        if not progress or not worker_health:
            return {}

        utilization = {}

        total_items = sum(w.items_processed for w in worker_health.values())

        for worker_id, health in worker_health.items():
            if total_items > 0:
                utilization[worker_id] = (health.items_processed / total_items) * 100.0
            else:
                utilization[worker_id] = 0.0

        return utilization

    def get_estimated_completion_time_str(self, job_id: str) -> str:
        """Get human-readable ETA.

        Args:
            job_id: Job identifier

        Returns:
            Formatted ETA string
        """
        eta_secs = self.estimate_completion_time(job_id)

        if eta_secs is None:
            return "Unknown"

        if eta_secs <= 0:
            return "Complete"

        # Convert to timedelta for nice formatting
        eta_time = timedelta(seconds=int(eta_secs))
        return str(eta_time)
