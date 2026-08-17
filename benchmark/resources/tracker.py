"""Cross-platform resource tracker for benchmark runs.

Tracks CPU, RAM, and disk usage during benchmark execution.
Works on macOS, Linux, and Windows using psutil.

Usage:
    with ResourceTracker() as tracker:
        run_benchmark(...)
    report = tracker.report()
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


def _try_import_psutil():
    try:
        import psutil

        return psutil
    except ImportError:
        return None


@dataclass
class ResourceSnapshot:
    """A single point-in-time resource measurement."""

    timestamp: float
    cpu_percent: float  # 0–100 per core, summed
    ram_used_mb: float  # RSS in MB
    ram_available_mb: float
    disk_read_mb: float  # cumulative since start
    disk_write_mb: float


@dataclass
class ResourceReport:
    """Aggregated resource usage for a complete benchmark run."""

    peak_ram_mb: float = 0.0
    avg_ram_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    duration_seconds: float = 0.0
    sample_count: int = 0
    platform: str = ""
    error: str = ""  # Non-empty if tracking failed (psutil unavailable)

    def to_dict(self) -> dict:
        return {
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "avg_ram_mb": round(self.avg_ram_mb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "total_disk_read_mb": round(self.total_disk_read_mb, 2),
            "total_disk_write_mb": round(self.total_disk_write_mb, 2),
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_count": self.sample_count,
            "platform": self.platform,
            "error": self.error,
        }


class ResourceTracker:
    """Polls system resources at regular intervals during a benchmark run.

    Context manager interface:
        with ResourceTracker(interval=0.5) as tracker:
            run_benchmark()
        report = tracker.report()

    Also works for manual start/stop:
        tracker = ResourceTracker()
        tracker.start()
        run_benchmark()
        tracker.stop()
        report = tracker.report()
    """

    def __init__(self, interval_seconds: float = 0.5):
        """Initialize resource tracker.

        Args:
            interval_seconds: Polling interval in seconds.
        """
        self._interval = interval_seconds
        self._psutil = _try_import_psutil()
        self._snapshots: list[ResourceSnapshot] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._start_disk_read = 0.0
        self._start_disk_write = 0.0
        self._use_process_io = False
        self._process = None

    def start(self) -> None:
        """Start resource tracking in a background thread."""
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._snapshots.clear()

        if self._psutil:
            try:
                self._process = self._psutil.Process(os.getpid())
                # Prime cpu_percent — the first call always returns 0.0 because it
                # measures the interval since the *previous* call.  Priming here
                # establishes a baseline so the first poll loop reading is non-zero.
                self._psutil.cpu_percent(interval=None)
                self._process.cpu_percent(interval=None)
                # Try process-level I/O counters (Linux only).
                # On macOS/Windows, fall back to disabled disk tracking to avoid
                # reporting misleading system-wide I/O from parallel workers.
                try:
                    io = self._process.io_counters()
                    self._start_disk_read = io.read_bytes / 1024 / 1024
                    self._start_disk_write = io.write_bytes / 1024 / 1024
                    self._use_process_io = True
                except (AttributeError, self._psutil.AccessDenied, NotImplementedError):
                    self._start_disk_read = 0.0
                    self._start_disk_write = 0.0
                    self._use_process_io = False
            except Exception:
                self._process = None

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop resource tracking."""
        self._end_time = time.monotonic()
        # Take a final sample to ensure short-lived tasks have at least one reading
        self._take_snapshot()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> ResourceTracker:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def report(self) -> ResourceReport:
        """Build aggregated report from collected samples."""
        import sys

        platform_name = sys.platform  # darwin | linux | win32

        if not self._snapshots:
            return ResourceReport(
                duration_seconds=self._end_time - self._start_time,
                platform=platform_name,
                error="no_samples" if not self._psutil else "tracking_started_but_no_samples",
            )

        if not self._psutil:
            return ResourceReport(
                duration_seconds=self._end_time - self._start_time,
                platform=platform_name,
                error="psutil_not_installed",
            )

        rams = [s.ram_used_mb for s in self._snapshots]
        cpus = [s.cpu_percent for s in self._snapshots]

        last = self._snapshots[-1]
        disk_read = max(0.0, last.disk_read_mb - self._start_disk_read)
        disk_write = max(0.0, last.disk_write_mb - self._start_disk_write)

        return ResourceReport(
            peak_ram_mb=max(rams),
            avg_ram_mb=sum(rams) / len(rams),
            peak_cpu_percent=max(cpus),
            avg_cpu_percent=sum(cpus) / len(cpus),
            total_disk_read_mb=disk_read,
            total_disk_write_mb=disk_write,
            duration_seconds=self._end_time - self._start_time,
            sample_count=len(self._snapshots),
            platform=platform_name,
        )

    def _take_snapshot(self) -> None:
        """Take a single resource snapshot. Safe to call from any thread."""
        if not self._psutil or not self._process:
            return
        try:
            mem_info = self._process.memory_info()
            ram_mb = mem_info.rss / 1024 / 1024
            cpu = self._psutil.cpu_percent(interval=None)

            # Disk I/O: use process-level counters if available, else 0
            disk_r = 0.0
            disk_w = 0.0
            if self._use_process_io:
                try:
                    io = self._process.io_counters()
                    disk_r = io.read_bytes / 1024 / 1024
                    disk_w = io.write_bytes / 1024 / 1024
                except Exception:
                    pass

            vm = self._psutil.virtual_memory()
            self._snapshots.append(
                ResourceSnapshot(
                    timestamp=time.monotonic() - self._start_time,
                    cpu_percent=cpu,
                    ram_used_mb=ram_mb,
                    ram_available_mb=vm.available / 1024 / 1024,
                    disk_read_mb=disk_r,
                    disk_write_mb=disk_w,
                )
            )
        except Exception:
            pass

    def _poll_loop(self) -> None:
        """Background polling loop."""
        if not self._psutil or not self._process:
            return

        while not self._stop_event.wait(timeout=self._interval):
            self._take_snapshot()
