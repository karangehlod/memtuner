"""Cost-aware optimization — find cheapest config meeting a recall threshold.

Answers: "What's the minimum cost configuration that achieves at least X% recall?"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostPoint:
    """A configuration with its cost and quality metrics."""

    label: str
    recall: float
    precision: float
    latency_ms: float
    estimated_cost_per_query: float
    config: dict


def find_cheapest_above_threshold(
    points: list[CostPoint],
    recall_threshold: float = 0.5,
) -> CostPoint | None:
    """Find the cheapest configuration that meets the recall threshold.

    Args:
        points: All tested configurations with costs.
        recall_threshold: Minimum acceptable recall (0.0-1.0).

    Returns:
        Cheapest qualifying point, or None if no config meets threshold.
    """
    qualifying = [p for p in points if p.recall >= recall_threshold]
    if not qualifying:
        return None
    return min(qualifying, key=lambda p: p.estimated_cost_per_query)


def build_cost_efficiency_curve(
    points: list[CostPoint],
) -> list[tuple[float, CostPoint]]:
    """Build a cost-efficiency curve showing cost at each recall level.

    Returns:
        List of (recall_threshold, cheapest_config) pairs,
        sorted by threshold ascending.
    """
    thresholds = [i / 20 for i in range(1, 20)]  # 0.05 to 0.95
    curve = []

    for threshold in thresholds:
        best = find_cheapest_above_threshold(points, threshold)
        if best:
            curve.append((threshold, best))

    return curve


def estimate_query_cost(strategy: str, latency_ms: float) -> float:
    """Estimate cost per query based on strategy and latency.

    Cost model (simplified):
    - BM25: ~$0.0001/query (CPU only)
    - Embeddings: ~$0.001/query (GPU inference)
    - Hybrid: ~$0.0012/query (both)
    - Storage read: $0.000001/query

    Args:
        strategy: Strategy name.
        latency_ms: Query latency in milliseconds.

    Returns:
        Estimated cost in USD per query.
    """
    base_costs = {
        "bm25": 0.0001,
        "embeddings": 0.001,
        "hybrid": 0.0012,
        "llm_rerank": 0.0003,
    }
    storage_cost = 0.000001
    return base_costs.get(strategy, 0.0005) + storage_cost
