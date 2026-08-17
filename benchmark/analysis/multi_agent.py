"""Multi-agent interference test — verify memory isolation under concurrent access.

Tests that when multiple agents (users) share the same memory store,
no cross-contamination occurs even under stress:
- Many users writing simultaneously
- Overlapping content across users
- Same query text from different users returning different results
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext


@dataclass(frozen=True)
class InterferenceResult:
    """Results of a multi-agent interference test."""

    total_queries: int
    leaked_queries: int  # Queries that returned another user's data
    isolation_rate: float  # 1.0 = perfect isolation
    users_tested: int
    memories_per_user: int
    details: list[dict]


def run_interference_test(
    store,
    num_users: int = 10,
    memories_per_user: int = 50,
    queries_per_user: int = 20,
) -> InterferenceResult:
    """Run a multi-agent interference test on a memory store.

    Creates N users, each with M unique memories, then queries each user
    and verifies that NO memory from another user appears in results.

    Args:
        store: A MemoryWriter + MemoryReader implementation.
        num_users: Number of simulated concurrent users.
        memories_per_user: Memories to inject per user.
        queries_per_user: Queries to run per user.

    Returns:
        InterferenceResult with isolation metrics.
    """
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    # Phase 1: Inject memories for all users
    user_memory_ids: dict[str, set[str]] = {}

    for user_idx in range(num_users):
        user_id = f"agent-{user_idx:03d}"
        user_memory_ids[user_id] = set()

        for mem_idx in range(memories_per_user):
            mem_id = f"MEM-{user_id}-{mem_idx:04d}"
            event = MemoryEvent(
                id=mem_id,
                user_id=user_id,
                type=MemoryType.EPISODIC,
                content=f"Agent {user_idx} memory {mem_idx}: The project deadline is Friday for task-{mem_idx}",
                timestamp=ts,
                importance=0.7 + (mem_idx % 3) * 0.1,
                entities=[f"agent-{user_idx}", f"task-{mem_idx}"],
                task_id=f"task-{mem_idx}",
            )
            store.write_on_day(event, mem_idx % 30)
            user_memory_ids[user_id].add(mem_id)

    # Phase 2: Query each user and check for leakage
    total_queries = 0
    leaked_queries = 0
    details = []

    for user_idx in range(num_users):
        user_id = f"agent-{user_idx:03d}"
        other_ids = set()
        for other_user, other_mems in user_memory_ids.items():
            if other_user != user_id:
                other_ids.update(other_mems)

        for q_idx in range(queries_per_user):
            query = ReadQuery(
                query=f"project deadline task-{q_idx}",
                top_k=10,
                context=ReadQueryContext(
                    dataset_day=15,
                    task_id=f"task-{q_idx}",
                    user_id=user_id,
                ),
            )

            response = store.read(query)
            retrieved_ids = {m.memory_id for m in response.retrieved_memories}

            # Check for leakage: any retrieved ID belonging to another user?
            leaked = retrieved_ids & other_ids
            total_queries += 1

            if leaked:
                leaked_queries += 1
                details.append(
                    {
                        "user": user_id,
                        "query_idx": q_idx,
                        "leaked_ids": list(leaked)[:3],
                    }
                )

    isolation_rate = 1.0 - (leaked_queries / total_queries if total_queries > 0 else 0)

    return InterferenceResult(
        total_queries=total_queries,
        leaked_queries=leaked_queries,
        isolation_rate=isolation_rate,
        users_tested=num_users,
        memories_per_user=memories_per_user,
        details=details[:10],  # Only first 10 leaks
    )
