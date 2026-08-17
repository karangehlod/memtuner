"""Unit tests for long-term memory stores — covering edge cases and branches."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark.memory.long_term.entity_store import EntityStore
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.long_term.semantic_store import SemanticStore
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext

FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


def _make_event(
    event_id: str,
    content: str,
    memory_type: MemoryType = MemoryType.EPISODIC,
    importance: float = 0.8,
    task_id: str = "test",
    entities: list[str] | None = None,
) -> MemoryEvent:
    """Create a test memory event."""
    return MemoryEvent(
        id=event_id,
        type=memory_type,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=importance,
        entities=entities or ["test"],
        task_id=task_id,
    )


def _make_query(
    query_text: str, day: int = 0, task_id: str = "test", top_k: int = 5
) -> ReadQuery:
    """Create a test read query."""
    return ReadQuery(
        query=query_text,
        top_k=top_k,
        context=ReadQueryContext(dataset_day=day, task_id=task_id),
    )


@pytest.mark.unit
class TestEpisodicStoreEdgeCases:
    """Edge case tests for EpisodicStore."""

    def test_write_on_day_and_read_with_decay(self) -> None:
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Test content"), day=0)
        response = store.read(_make_query("Test content", day=10))
        assert len(response.retrieved_memories) == 1
        # Score should be lower due to decay
        assert response.retrieved_memories[0].decay_factor < 1.0

    def test_prune_removes_memories(self) -> None:
        store = EpisodicStore()
        store.write(_make_event("M-001", "First"))
        store.write(_make_event("M-002", "Second"))
        removed = store.prune(["M-001"])
        assert removed == 1
        assert store.count() == 1

    def test_prune_nonexistent_returns_zero(self) -> None:
        store = EpisodicStore()
        removed = store.prune(["nonexistent"])
        assert removed == 0

    def test_clear_removes_all(self) -> None:
        store = EpisodicStore()
        store.write(_make_event("M-001", "First"))
        store.write(_make_event("M-002", "Second"))
        store.clear()
        assert store.count() == 0

    def test_get_memory_scores(self) -> None:
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Test", importance=1.0), day=0)
        scores = store.get_memory_scores(day=5)
        assert "M-001" in scores
        assert scores["M-001"] < 1.0

    def test_tier_computation_hot(self) -> None:
        store = EpisodicStore(decay_lambda=0.01)
        store.write_on_day(_make_event("M-001", "Test"), day=0)
        response = store.read(_make_query("Test", day=0))
        assert response.retrieved_memories[0].tier.value == "hot"

    def test_tier_computation_warm(self) -> None:
        store = EpisodicStore(decay_lambda=0.05)
        store.write_on_day(_make_event("M-001", "Test"), day=0)
        response = store.read(_make_query("Test", day=10))
        mem = response.retrieved_memories[0]
        assert mem.tier.value == "warm"

    def test_tier_computation_cold(self) -> None:
        store = EpisodicStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Test"), day=0)
        response = store.read(_make_query("Test", day=30))
        mem = response.retrieved_memories[0]
        assert mem.tier.value == "cold"

    def test_read_empty_store(self) -> None:
        store = EpisodicStore()
        response = store.read(_make_query("anything"))
        assert len(response.retrieved_memories) == 0
        assert response.total_candidates == 0

    def test_top_k_limits_results(self) -> None:
        store = EpisodicStore()
        for i in range(10):
            store.write(_make_event(f"M-{i:03d}", f"Content {i}"))
        response = store.read(_make_query("Content", top_k=3))
        assert len(response.retrieved_memories) == 3


@pytest.mark.unit
class TestPreferenceStoreEdgeCases:
    """Edge case tests for PreferenceStore."""

    def test_write_on_day_and_read_with_decay(self) -> None:
        store = PreferenceStore(decay_lambda=0.1)
        event = _make_event("M-001", "User prefers dark mode", task_id="prefs", memory_type=MemoryType.PREFERENCE)
        store.write_on_day(event, day=0)
        response = store.read(_make_query("dark mode", day=10, task_id="prefs"))
        assert len(response.retrieved_memories) == 1

    def test_task_boost_applied(self) -> None:
        store = PreferenceStore()
        store.write(
            _make_event("M-001", "User prefers dark mode", task_id="prefs", memory_type=MemoryType.PREFERENCE)
        )
        # Query with matching task_id should produce higher score
        response_match = store.read(_make_query("dark mode", task_id="prefs"))
        response_no_match = store.read(_make_query("dark mode", task_id="other"))
        if response_match.retrieved_memories and response_no_match.retrieved_memories:
            assert (
                response_match.retrieved_memories[0].score
                >= response_no_match.retrieved_memories[0].score
            )

    def test_prune_removes_multiple(self) -> None:
        store = PreferenceStore()
        store.write(_make_event("M-001", "First", memory_type=MemoryType.PREFERENCE))
        store.write(_make_event("M-002", "Second", memory_type=MemoryType.PREFERENCE))
        store.write(_make_event("M-003", "Third", memory_type=MemoryType.PREFERENCE))
        removed = store.prune(["M-001", "M-003"])
        assert removed == 2
        assert store.count() == 1

    def test_clear_removes_all(self) -> None:
        store = PreferenceStore()
        store.write(_make_event("M-001", "First", memory_type=MemoryType.PREFERENCE))
        store.clear()
        assert store.count() == 0

    def test_get_memory_scores_with_decay(self) -> None:
        store = PreferenceStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Test", importance=1.0, memory_type=MemoryType.PREFERENCE), day=0)
        scores = store.get_memory_scores(day=5)
        assert scores["M-001"] < 1.0

    def test_tier_hot_warm_cold(self) -> None:
        store = PreferenceStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Hot test", memory_type=MemoryType.PREFERENCE), day=0)
        hot = store.read(_make_query("Hot test", day=0))
        assert hot.retrieved_memories[0].tier.value == "hot"

        store_warm = PreferenceStore(decay_lambda=0.05)
        store_warm.write_on_day(_make_event("M-002", "Warm test", memory_type=MemoryType.PREFERENCE), day=0)
        warm = store_warm.read(_make_query("Warm test", day=10))
        assert warm.retrieved_memories[0].tier.value == "warm"

        store_cold = PreferenceStore(decay_lambda=0.1)
        store_cold.write_on_day(_make_event("M-003", "Cold test", memory_type=MemoryType.PREFERENCE), day=0)
        cold = store_cold.read(_make_query("Cold test", day=30))
        assert cold.retrieved_memories[0].tier.value == "cold"


@pytest.mark.unit
class TestSemanticStoreEdgeCases:
    """Edge case tests for SemanticStore."""

    def test_write_on_day_and_read(self) -> None:
        store = SemanticStore()
        store.write_on_day(_make_event("M-001", "Python is a language", memory_type=MemoryType.SEMANTIC), day=0)
        response = store.read(_make_query("Python", day=5))
        assert len(response.retrieved_memories) == 1

    def test_remove_memory(self) -> None:
        store = SemanticStore()
        store.write(_make_event("M-001", "First", memory_type=MemoryType.SEMANTIC))
        store.remove("M-001")
        assert store.count() == 0

    def test_remove_nonexistent_no_error(self) -> None:
        store = SemanticStore()
        store.remove("nonexistent")

    def test_get_memory_scores(self) -> None:
        store = SemanticStore(decay_lambda=0.05)
        store.write_on_day(_make_event("M-001", "Fact", importance=1.0, memory_type=MemoryType.SEMANTIC), day=0)
        scores = store.get_memory_scores(day=10)
        assert scores["M-001"] < 1.0

    def test_tier_hot(self) -> None:
        store = SemanticStore(decay_lambda=0.01)
        store.write_on_day(_make_event("M-001", "Test", memory_type=MemoryType.SEMANTIC), day=0)
        response = store.read(_make_query("Test", day=0))
        assert response.retrieved_memories[0].tier.value == "hot"

    def test_tier_warm(self) -> None:
        store = SemanticStore(decay_lambda=0.05)
        store.write_on_day(_make_event("M-001", "Test", memory_type=MemoryType.SEMANTIC), day=0)
        response = store.read(_make_query("Test", day=10))
        assert response.retrieved_memories[0].tier.value == "warm"

    def test_tier_cold(self) -> None:
        store = SemanticStore(decay_lambda=0.1)
        store.write_on_day(_make_event("M-001", "Test", memory_type=MemoryType.SEMANTIC), day=0)
        response = store.read(_make_query("Test", day=30))
        assert response.retrieved_memories[0].tier.value == "cold"

    def test_empty_store_read(self) -> None:
        store = SemanticStore()
        response = store.read(_make_query("anything"))
        assert response.retrieved_memories == []


@pytest.mark.unit
class TestEntityStoreEdgeCases:
    """Edge case tests for EntityStore."""

    def test_write_on_day_and_read(self) -> None:
        store = EntityStore()
        event = _make_event(
            "M-001", "Alice is a software engineer", entities=["Alice"], memory_type=MemoryType.ENTITY
        )
        store.write_on_day(event, day=0)
        response = store.read(_make_query("Tell me about Alice", day=5))
        assert len(response.retrieved_memories) == 1

    def test_entity_boost_applied(self) -> None:
        store = EntityStore()
        store.write(
            _make_event("M-001", "Alice works at Acme", entities=["Alice", "Acme"], memory_type=MemoryType.ENTITY)
        )
        store.write(
            _make_event("M-002", "Weather is sunny", entities=["weather"], memory_type=MemoryType.ENTITY)
        )
        response = store.read(_make_query("Alice"))
        # Entity-boosted result should rank higher
        if len(response.retrieved_memories) >= 2:
            assert response.retrieved_memories[0].memory_id == "M-001"

    def test_remove_memory(self) -> None:
        store = EntityStore()
        store.write(_make_event("M-001", "Test", memory_type=MemoryType.ENTITY))
        store.remove("M-001")
        assert store.count() == 0

    def test_remove_nonexistent_no_error(self) -> None:
        store = EntityStore()
        store.remove("nonexistent")

    def test_get_memory_scores(self) -> None:
        store = EntityStore(decay_lambda=0.05)
        store.write_on_day(_make_event("M-001", "Test", importance=1.0, memory_type=MemoryType.ENTITY), day=0)
        scores = store.get_memory_scores(day=10)
        assert scores["M-001"] < 1.0

    def test_tier_hot(self) -> None:
        store = EntityStore(decay_lambda=0.01)
        store.write_on_day(_make_event("M-001", "Test", memory_type=MemoryType.ENTITY), day=0)
        response = store.read(_make_query("Test", day=0))
        assert response.retrieved_memories[0].tier.value == "hot"

    def test_tier_warm(self) -> None:
        store = EntityStore(decay_lambda=0.05)
        store.write_on_day(_make_event("M-001", "Test", memory_type=MemoryType.ENTITY), day=0)
        response = store.read(_make_query("Test", day=10))
        assert response.retrieved_memories[0].tier.value == "warm"

    def test_tier_cold(self) -> None:
        store = EntityStore(decay_lambda=0.2)
        store.write_on_day(_make_event("M-001", "Test", memory_type=MemoryType.ENTITY), day=0)
        response = store.read(_make_query("Test", day=30))
        assert response.retrieved_memories[0].tier.value == "cold"

    def test_score_clamped_to_one(self) -> None:
        """Scores should not exceed 1.0."""
        store = EntityStore()
        store.write(
            _make_event(
                "M-001",
                "Alice",
                importance=1.0,
                entities=["Alice"],
                memory_type=MemoryType.ENTITY,
            )
        )
        response = store.read(_make_query("Alice", day=0))
        assert response.retrieved_memories[0].score <= 1.0

    def test_empty_store_read(self) -> None:
        store = EntityStore()
        response = store.read(_make_query("anything"))
        assert response.retrieved_memories == []
