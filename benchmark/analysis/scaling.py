"""Incremental scaling analysis — how metrics degrade with memory count.

Shows how recall, precision, and latency change as the memory store grows
from 100 → 1K → 10K → 100K memories. Helps predict production behavior.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext


@dataclass(frozen=True)
class ScalingPoint:
    """Metrics at a specific memory count."""

    memory_count: int
    recall: float
    precision: float
    latency_p50_ms: float
    latency_p99_ms: float
    index_time_ms: float


def run_scaling_test(
    store_factory,
    gold_memories: list[tuple[str, str, float]],
    queries: list[tuple[str, list[str]]],
    scale_points: list[int] | None = None,
    user_id: str = "user-scale-test",
) -> list[ScalingPoint]:
    """Run incremental scaling test.

    Injects increasing numbers of noise memories and measures how
    metrics degrade. Gold memories are always present; noise grows.

    Args:
        store_factory: Callable that creates a fresh store instance.
        gold_memories: List of (id, content, importance) tuples for gold events.
        queries: List of (query_text, expected_ids) tuples.
        scale_points: Memory counts to test at. Default: [100, 500, 1000, 5000, 10000].
        user_id: User ID for all memories and queries.

    Returns:
        List of ScalingPoints showing metrics at each scale.
    """
    if scale_points is None:
        scale_points = [100, 500, 1000, 5000, 10000]

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    results: list[ScalingPoint] = []

    for target_count in scale_points:
        store = store_factory()

        # Inject gold memories first
        for mem_id, content, importance in gold_memories:
            event = MemoryEvent(
                id=mem_id,
                user_id=user_id,
                type=MemoryType.EPISODIC,
                content=content,
                timestamp=ts,
                importance=importance,
                entities=[],
                task_id="scale-test",
            )
            store.write_on_day(event, 0)

        # Fill with noise to reach target_count
        noise_count = max(0, target_count - len(gold_memories))
        index_start = time.monotonic()

        for i in range(noise_count):
            noise_event = MemoryEvent(
                id=f"NOISE-{i:06d}",
                user_id=user_id,
                type=MemoryType.EPISODIC,
                content=f"Noise memory number {i} about topic {i % 50} discussion {i % 100}",
                timestamp=ts,
                importance=0.3 + (i % 5) * 0.1,
                entities=[f"topic-{i % 50}"],
                task_id=f"task-{i % 20}",
            )
            store.write_on_day(noise_event, i % 30)

        index_time = (time.monotonic() - index_start) * 1000

        # Run queries and measure
        recalls = []
        precisions = []
        latencies = []

        for query_text, expected_ids in queries:
            query = ReadQuery(
                query=query_text,
                top_k=10,
                context=ReadQueryContext(dataset_day=15, task_id="scale-test", user_id=user_id),
            )

            t0 = time.monotonic()
            response = store.read(query)
            t1 = time.monotonic()
            latencies.append((t1 - t0) * 1000)

            retrieved_ids = [m.memory_id for m in response.retrieved_memories]
            found = set(retrieved_ids[:10]) & set(expected_ids)

            recall = len(found) / len(expected_ids) if expected_ids else 0
            precision = len(found) / 10
            recalls.append(recall)
            precisions.append(precision)

        latencies.sort()
        # Nearest-rank percentile: ceil(p/100 * N) - 1 (0-indexed).
        # Matches the formula used in _build_scenario_metrics; floor-division
        # underestimates tail latency (e.g. N=100 → p99 index 98 vs correct 99).
        if latencies:
            n_lat = len(latencies)
            import math as _math
            p50 = latencies[min(n_lat - 1, _math.ceil(0.50 * n_lat) - 1)]
            p99 = latencies[min(n_lat - 1, _math.ceil(0.99 * n_lat) - 1)]
        else:
            p50 = p99 = 0

        results.append(
            ScalingPoint(
                memory_count=target_count,
                recall=sum(recalls) / len(recalls) if recalls else 0,
                precision=sum(precisions) / len(precisions) if precisions else 0,
                latency_p50_ms=p50,
                latency_p99_ms=p99,
                index_time_ms=index_time,
            )
        )

    return results
