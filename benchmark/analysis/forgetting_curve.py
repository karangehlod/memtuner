"""Forgetting curve validation — compare system decay against Ebbinghaus.

The Ebbinghaus forgetting curve (1885) models human memory retention:
    R(t) = e^(-t/S)  where S is memory stability

This module compares the benchmark's decay curves against the theoretical
human forgetting curve to validate that:
1. The system's decay is calibrated realistically
2. Important memories (high importance) resist decay longer
3. The half-life parameters produce meaningful differentiation
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForgettingCurvePoint:
    """A single point on the forgetting curve."""

    day: int
    retention_benchmark: float  # System's decay factor
    retention_ebbinghaus: float  # Theoretical human retention
    retention_with_rehearsal: float  # With spaced repetition


def compute_ebbinghaus_curve(
    days: list[int],
    stability: float = 14.0,
) -> list[float]:
    """Compute the Ebbinghaus forgetting curve.

    Args:
        days: List of day values to compute retention for.
        stability: Memory stability parameter (higher = slower forgetting).
            Default 14 corresponds to approximately 2-week half-life.

    Returns:
        Retention values (0.0-1.0) for each day.
    """
    return [math.exp(-d / stability) for d in days]


def compute_benchmark_curve(
    days: list[int],
    decay_lambda: float,
    archival_floor: float = 0.65,
) -> list[float]:
    """Compute the benchmark's exponential decay curve.

    Matches the actual implementation in BaseLongTermStore._compute_decay_factor().

    Args:
        days: List of day values.
        decay_lambda: The λ parameter.
        archival_floor: Minimum decay factor for memories > 90 days old.

    Returns:
        Decay factors for each day.
    """
    results = []
    for d in days:
        if decay_lambda == 0:
            results.append(1.0)
        else:
            raw = math.exp(-decay_lambda * d)
            if d >= 90:
                results.append(max(archival_floor, raw))
            else:
                results.append(raw)
    return results


def compute_rehearsal_curve(
    days: list[int],
    stability: float = 14.0,
    rehearsal_boost: float = 2.0,
    rehearsal_interval: int = 7,
) -> list[float]:
    """Compute forgetting curve with spaced repetition (rehearsal).

    Models the effect of periodic review: each rehearsal increases
    the effective stability by rehearsal_boost.

    Args:
        days: List of day values.
        stability: Base stability.
        rehearsal_boost: Multiplier applied per rehearsal.
        rehearsal_interval: Days between rehearsals.

    Returns:
        Retention values with rehearsal effect.
    """
    results = []
    for d in days:
        rehearsals = d // rehearsal_interval
        # Cap exponent to prevent overflow: 2^43 ≈ 8.8T produces degenerate 1.0 retention.
        effective_stability = stability * (rehearsal_boost ** min(rehearsals, 40))
        retention = math.exp(-d / effective_stability)
        results.append(min(1.0, retention))
    return results


def generate_forgetting_comparison(
    decay_lambda: float = 0.05,
    max_days: int = 100,
) -> list[ForgettingCurvePoint]:
    """Generate comparison data between benchmark decay and Ebbinghaus.

    Args:
        decay_lambda: Benchmark's decay parameter.
        max_days: Number of days to simulate.

    Returns:
        List of comparison points for plotting.
    """
    days = list(range(0, max_days + 1, 1))
    ebbinghaus_stability = math.log(2) / decay_lambda if decay_lambda > 0 else 1000

    benchmark = compute_benchmark_curve(days, decay_lambda)
    ebbinghaus = compute_ebbinghaus_curve(days, ebbinghaus_stability)
    rehearsal = compute_rehearsal_curve(days, ebbinghaus_stability)

    return [
        ForgettingCurvePoint(
            day=d,
            retention_benchmark=b,
            retention_ebbinghaus=e,
            retention_with_rehearsal=r,
        )
        for d, b, e, r in zip(days, benchmark, ebbinghaus, rehearsal)
    ]
