"""Adversarial query analysis — contradiction and knowledge update handling.

Tests whether the memory system correctly retrieves the LATEST information
when contradictory facts exist (e.g., "I live in Boston" then "I moved to Seattle").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext


@dataclass(frozen=True)
class ContradictionScenario:
    """A contradiction test case — old fact vs new fact."""

    label: str
    old_memory: MemoryEvent
    old_day: int
    new_memory: MemoryEvent
    new_day: int
    query: str
    query_day: int
    expected_id: str  # Should retrieve the NEW memory


def build_contradiction_scenarios() -> list[ContradictionScenario]:
    """Build standard contradiction test cases.

    Returns:
        List of scenarios testing knowledge updates.
    """
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    return [
        ContradictionScenario(
            label="Location change",
            old_memory=MemoryEvent(
                id="CONTRA-001-old",
                user_id="user-test",
                type=MemoryType.EPISODIC,
                content="I live in Boston, have been here for years.",
                timestamp=ts,
                importance=0.8,
                entities=["Boston"],
                task_id="location",
            ),
            old_day=10,
            new_memory=MemoryEvent(
                id="CONTRA-001-new",
                user_id="user-test",
                type=MemoryType.EPISODIC,
                content="I just moved to Seattle last week!",
                timestamp=ts,
                importance=0.9,
                entities=["Seattle"],
                task_id="location",
            ),
            new_day=100,
            query="Where does the user currently live?",
            query_day=110,
            expected_id="CONTRA-001-new",
        ),
        ContradictionScenario(
            label="Preference change",
            old_memory=MemoryEvent(
                id="CONTRA-002-old",
                user_id="user-test",
                type=MemoryType.PREFERENCE,
                content="I prefer Python for all my projects.",
                timestamp=ts,
                importance=0.8,
                entities=["Python"],
                task_id="language",
            ),
            old_day=5,
            new_memory=MemoryEvent(
                id="CONTRA-002-new",
                user_id="user-test",
                type=MemoryType.PREFERENCE,
                content="I've switched to Rust, it's much better for performance.",
                timestamp=ts,
                importance=0.85,
                entities=["Rust"],
                task_id="language",
            ),
            new_day=90,
            query="What programming language does the user prefer?",
            query_day=95,
            expected_id="CONTRA-002-new",
        ),
        ContradictionScenario(
            label="Job change",
            old_memory=MemoryEvent(
                id="CONTRA-003-old",
                user_id="user-test",
                type=MemoryType.EPISODIC,
                content="I work at Google as a senior engineer.",
                timestamp=ts,
                importance=0.8,
                entities=["Google"],
                task_id="job",
            ),
            old_day=20,
            new_memory=MemoryEvent(
                id="CONTRA-003-new",
                user_id="user-test",
                type=MemoryType.EPISODIC,
                content="I just started at OpenAI, really excited about the new role!",
                timestamp=ts,
                importance=0.9,
                entities=["OpenAI"],
                task_id="job",
            ),
            new_day=200,
            query="Where does the user work?",
            query_day=210,
            expected_id="CONTRA-003-new",
        ),
        ContradictionScenario(
            label="Dietary change",
            old_memory=MemoryEvent(
                id="CONTRA-004-old",
                user_id="user-test",
                type=MemoryType.PREFERENCE,
                content="I'm vegetarian, have been for 5 years.",
                timestamp=ts,
                importance=0.7,
                entities=[],
                task_id="food",
            ),
            old_day=0,
            new_memory=MemoryEvent(
                id="CONTRA-004-new",
                user_id="user-test",
                type=MemoryType.PREFERENCE,
                content="I started eating fish again, so I'm pescatarian now.",
                timestamp=ts,
                importance=0.75,
                entities=[],
                task_id="food",
            ),
            new_day=180,
            query="What is the user's diet?",
            query_day=185,
            expected_id="CONTRA-004-new",
        ),
    ]


def run_adversarial_test(store, scenarios: list[ContradictionScenario]) -> dict:
    """Run contradiction scenarios against a memory store.

    Args:
        store: A MemoryWriter + MemoryReader implementation.
        scenarios: List of contradiction test cases.

    Returns:
        Dict with results: total, correct, accuracy, details.
    """
    correct = 0
    details = []

    for scenario in scenarios:
        # Inject both memories
        store.write_on_day(scenario.old_memory, scenario.old_day)
        store.write_on_day(scenario.new_memory, scenario.new_day)

        # Query
        query = ReadQuery(
            query=scenario.query,
            top_k=5,
            context=ReadQueryContext(
                dataset_day=scenario.query_day,
                task_id=scenario.old_memory.task_id,
                user_id="user-test",
            ),
        )

        response = store.read(query)
        retrieved_ids = [m.memory_id for m in response.retrieved_memories]

        # Build a rank dict once — O(K) — instead of two O(K) list.index() scans.
        rank_of = {mid: rank for rank, mid in enumerate(retrieved_ids)}
        new_rank = rank_of.get(scenario.expected_id, -1)
        old_id = scenario.old_memory.id
        old_rank = rank_of.get(old_id, -1)

        is_correct = new_rank != -1 and (old_rank == -1 or new_rank < old_rank)
        if is_correct:
            correct += 1

        details.append(
            {
                "label": scenario.label,
                "correct": is_correct,
                "new_rank": new_rank + 1 if new_rank >= 0 else "not found",
                "old_rank": old_rank + 1 if old_rank >= 0 else "not found",
            }
        )

    return {
        "total": len(scenarios),
        "correct": correct,
        "accuracy": correct / len(scenarios) if scenarios else 0,
        "details": details,
    }
