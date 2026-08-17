"""Pareto frontier analysis — find the optimal recall-precision tradeoff.

Identifies configurations on the Pareto front: no other config has
BOTH higher recall AND higher precision simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    """A single point in the recall-precision space."""

    label: str
    recall: float
    precision: float
    latency_ms: float
    cost: float
    config: dict


def compute_pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Compute the Pareto-optimal set from a list of configurations.

    A point is Pareto-optimal if no other point dominates it
    (has both higher recall AND higher precision).

    Args:
        points: All benchmark results as ParetoPoints.

    Returns:
        Pareto-optimal subset, sorted by recall ascending.
    """
    if not points:
        return []

    # Sort by recall descending
    sorted_points = sorted(points, key=lambda p: p.recall, reverse=True)

    pareto: list[ParetoPoint] = []
    max_precision_seen = -1.0

    for point in sorted_points:
        if point.precision > max_precision_seen:
            pareto.append(point)
            max_precision_seen = point.precision

    return sorted(pareto, key=lambda p: p.recall)


def format_pareto_report(pareto_points: list[ParetoPoint], all_points: list[ParetoPoint]) -> str:
    """Format a text report of the Pareto frontier.

    Args:
        pareto_points: Points on the Pareto frontier.
        all_points: All tested configurations.

    Returns:
        Formatted text report.
    """
    lines = [
        "PARETO FRONTIER — Recall vs Precision Optimal Set",
        "=" * 65,
        f"Total configurations tested: {len(all_points)}",
        f"Pareto-optimal points: {len(pareto_points)}",
        "",
        f"{'Config':<35} {'Recall':>8} {'Prec@K':>8} {'Latency':>9}",
        "-" * 65,
    ]

    for p in pareto_points:
        lines.append(
            f"  ★ {p.label:<33} {p.recall:>7.1%} {p.precision:>7.1%} {p.latency_ms:>7.1f}ms"
        )

    lines.append("-" * 65)
    lines.append("")
    lines.append("Points NOT on frontier (dominated):")

    dominated = [p for p in all_points if p not in pareto_points]
    for p in sorted(dominated, key=lambda x: x.recall, reverse=True)[:10]:
        lines.append(
            f"    {p.label:<33} {p.recall:>7.1%} {p.precision:>7.1%} {p.latency_ms:>7.1f}ms"
        )

    return "\n".join(lines)
