"""OS-level system metrics helpers used by memory adapters.

These functions sample OS-level resource counters at the moment they are
called; they are not background monitors.  Adapters invoke them inside
``get_metrics()`` to snapshot the process state after a benchmark phase.

Functions
---------
percentile(values, p)    — p-th percentile via linear interpolation
peak_rss_mb()            — lifetime peak RSS of the process in MiB
cpu_percent_snapshot()   — instantaneous CPU% for this PID
"""

from __future__ import annotations

from collections.abc import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Return the p-th percentile (0–100) of *values*. Returns 0.0 if empty.

    Algorithm: linear interpolation at fractional index
    ---------------------------------------------------
    idx    = (p / 100) × (n − 1)       # fractional position in sorted list
    lo     = floor(idx)
    hi     = min(lo + 1, n − 1)
    frac   = idx − lo
    result = values[lo] × (1 − frac) + values[hi] × frac

    This is identical to numpy.percentile with method='linear' (the default).

    Example
    -------
    values = [1, 2, 3, 4, 5],  p = 95
    idx  = 0.95 × 4 = 3.8
    lo = 3,  hi = 4,  frac = 0.8
    result = 4 × 0.2 + 5 × 0.8 = 4.8
    """
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100.0) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def peak_rss_mb() -> float:
    """Return the lifetime peak RSS of this process in MiB (0.0 if unavailable).

    Source: ``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss``

    Platform units
    --------------
    macOS (Darwin): ru_maxrss is in **bytes**      → divide by 1 024 × 1 024
    Linux          : ru_maxrss is in **kilobytes** → divide by 1 024

    Why getrusage instead of psutil.rss
    ------------------------------------
    psutil.Process.memory_info().rss is a live snapshot — it reflects the
    current resident-set size and drops back down after memory is freed.
    resource.getrusage tracks the **high-water mark** across the entire
    process lifetime, so it correctly captures peak memory even if the
    allocation occurred earlier in the benchmark run.
    """
    try:
        import platform
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return ru.ru_maxrss / (1024 * 1024)   # bytes → MiB
        else:
            return ru.ru_maxrss / 1024              # kilobytes → MiB
    except Exception:
        return 0.0


def cpu_percent_snapshot() -> float:
    """Return instantaneous CPU% for this PID (0.0 if unavailable).

    Implementation: ``psutil.Process(os.getpid()).cpu_percent(interval=None)``

    Caveats
    -------
    The first call to ``cpu_percent(interval=None)`` always returns **0.0**
    because psutil needs a reference measurement to compute the delta.
    Subsequent calls return the CPU% accumulated since the previous call.
    """
    try:
        import os

        import psutil
        return psutil.Process(os.getpid()).cpu_percent(interval=None)
    except Exception:
        return 0.0
