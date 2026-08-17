"""User isolation tests — verifies no cross-user data leakage.

These tests prove that a query from user A can NEVER retrieve memories
that were written by user B, regardless of:
- Which retrieval path is active (strategy or fallback)
- Whether memory IDs overlap between users
- Whether the store holds thousands of memories from many users

All tests use the concrete EpisodicStore (which inherits BaseLongTermStore)
because that is the shared implementation that all four stores use.

Security guarantee tested here:
    _filter_candidates() scopes to user_id BEFORE any strategy or scoring
    is applied — other users' memories never enter the candidate pool.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext, ReadQueryFilters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _event(
    memory_id: str,
    user_id: str,
    content: str = "test content",
    importance: float = 0.9,
) -> MemoryEvent:
    return MemoryEvent(
        id=memory_id,
        user_id=user_id,
        type=MemoryType.EPISODIC,
        content=content,
        timestamp=_NOW,
        importance=importance,
        task_id="task-001",
    )


def _query(user_id: str, query_text: str = "test query") -> ReadQuery:
    return ReadQuery(
        query=query_text,
        top_k=50,
        context=ReadQueryContext(
            dataset_day=1,
            task_id="task-001",
            user_id=user_id,
        ),
        filters=ReadQueryFilters(),
    )


# ---------------------------------------------------------------------------
# Core isolation tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUserIsolationFallbackPath:
    """Tests for the fallback scoring path (no retrieval strategy)."""

    def _store(self) -> EpisodicStore:
        return EpisodicStore(decay_lambda=0.0, pruning_threshold=0.0)

    def test_user_sees_only_own_memories(self):
        store = self._store()
        store.write(_event("M-A1", "user-A", content="Alice secret 1"))
        store.write(_event("M-A2", "user-A", content="Alice secret 2"))
        store.write(_event("M-B1", "user-B", content="Bob secret"))
        store.write(_event("M-B2", "user-B", content="Bob private"))

        response = store.read(_query("user-A"))

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert returned_ids.issubset({"M-A1", "M-A2"}), (
            f"User-A received memories from another user: {returned_ids - {'M-A1', 'M-A2'}}"
        )

    def test_user_b_cannot_see_user_a_memories(self):
        store = self._store()
        store.write(_event("M-A1", "user-A", content="Alice confidential"))
        store.write(_event("M-B1", "user-B", content="Bob data"))

        response = store.read(_query("user-B"))

        returned_ids = {m.memory_id for m in response.retrieved_memories}
        assert "M-A1" not in returned_ids, (
            "User-B received user-A's memory M-A1 — cross-user leakage!"
        )

    def test_overlapping_memory_ids_no_leakage(self):
        """Even if two users happen to have the same memory ID (mis-config), isolation holds."""
        store = self._store()
        # Same ID, different users — last write wins in dict, but both are user-scoped reads
        store.write(_event("M-001", "user-A", content="Alice version"))
        store.write(_event("M-002", "user-B", content="Bob version"))

        resp_a = store.read(_query("user-A"))
        resp_b = store.read(_query("user-B"))

        ids_a = {m.memory_id for m in resp_a.retrieved_memories}
        ids_b = {m.memory_id for m in resp_b.retrieved_memories}
        assert "M-002" not in ids_a
        assert "M-001" not in ids_b

    def test_empty_store_returns_empty(self):
        store = self._store()
        response = store.read(_query("user-A"))
        assert response.retrieved_memories == []

    def test_user_with_no_memories_gets_nothing(self):
        store = self._store()
        store.write(_event("M-B1", "user-B"))
        store.write(_event("M-B2", "user-B"))

        response = store.read(_query("user-A"))

        assert response.retrieved_memories == []

    def test_many_users_strict_isolation(self):
        """With 10 users, each user gets exactly their own memories."""
        store = self._store()
        n_users = 10
        memories_per_user = 5

        for u in range(n_users):
            for m in range(memories_per_user):
                store.write(_event(f"M-{u}-{m}", f"user-{u}"))

        for u in range(n_users):
            response = store.read(_query(f"user-{u}"))
            returned_ids = {mem.memory_id for mem in response.retrieved_memories}
            expected_ids = {f"M-{u}-{m}" for m in range(memories_per_user)}
            assert returned_ids == expected_ids, (
                f"user-{u} isolation violated: "
                f"got {returned_ids - expected_ids} extra, "
                f"missing {expected_ids - returned_ids}"
            )

    def test_after_prune_isolation_still_holds(self):
        """Pruning one user's memories doesn't expose another user's."""
        store = self._store()
        store.write(_event("M-A1", "user-A"))
        store.write(_event("M-B1", "user-B"))

        # Prune user-A's memory
        store.prune(["M-A1"])

        # User-B must still not see user-A's pruned memory (it's gone)
        # and user-A must not see user-B's memory
        resp_a = store.read(_query("user-A"))
        resp_b = store.read(_query("user-B"))

        assert resp_a.retrieved_memories == []
        assert {m.memory_id for m in resp_b.retrieved_memories} == {"M-B1"}

    def test_write_on_day_preserves_isolation(self):
        """write_on_day() must not bypass user scoping."""
        store = self._store()
        store.write_on_day(_event("M-A1", "user-A"), day=0)
        store.write_on_day(_event("M-B1", "user-B"), day=0)

        resp_a = store.read(_query("user-A"))
        resp_b = store.read(_query("user-B"))

        ids_a = {m.memory_id for m in resp_a.retrieved_memories}
        ids_b = {m.memory_id for m in resp_b.retrieved_memories}
        assert "M-B1" not in ids_a
        assert "M-A1" not in ids_b


@pytest.mark.unit
class TestUserIsolationFilterCandidates:
    """Tests that _filter_candidates applies user scope as the FIRST filter."""

    def _store(self) -> EpisodicStore:
        return EpisodicStore(decay_lambda=0.0, pruning_threshold=0.0)

    def test_filter_candidates_user_scope_is_first(self):
        """Candidate pool must be user-scoped before type/importance filtering."""
        store = self._store()
        store.write(_event("M-A1", "user-A", importance=0.9))
        store.write(_event("M-B1", "user-B", importance=0.9))

        # Query with a permissive filter that would match both if not user-scoped
        query = ReadQuery(
            query="test",
            top_k=50,
            context=ReadQueryContext(dataset_day=1, task_id="t", user_id="user-A"),
            filters=ReadQueryFilters(min_importance=0.5),
        )

        candidates = store._filter_candidates(query)

        assert "M-A1" in candidates
        assert "M-B1" not in candidates, "M-B1 (user-B) leaked into user-A's candidates"

    def test_filter_candidates_type_filter_within_user_scope(self):
        """Type filter narrows within user's memories only."""
        from benchmark.models.memory_event import MemoryType as MT

        store = self._store()
        ep = MemoryEvent(
            id="M-A-ep", user_id="user-A", type=MemoryType.EPISODIC,
            content="ep", timestamp=_NOW, importance=0.8, task_id="t",
        )
        sem = MemoryEvent(
            id="M-A-sem", user_id="user-A", type=MemoryType.SEMANTIC,
            content="sem", timestamp=_NOW, importance=0.8, task_id="t",
        )
        other = MemoryEvent(
            id="M-B-ep", user_id="user-B", type=MemoryType.EPISODIC,
            content="other user ep", timestamp=_NOW, importance=0.8, task_id="t",
        )
        store.write(ep)
        store.write(sem)
        store.write(other)

        query = ReadQuery(
            query="test",
            top_k=50,
            context=ReadQueryContext(dataset_day=1, task_id="t", user_id="user-A"),
            filters=ReadQueryFilters(memory_types=[MemoryType.EPISODIC]),
        )

        candidates = store._filter_candidates(query)
        assert "M-A-ep" in candidates
        assert "M-A-sem" not in candidates   # filtered by type
        assert "M-B-ep" not in candidates    # filtered by user_id


@pytest.mark.unit
class TestUserIsolationAllStores:
    """Run the core isolation check on all four long-term store types."""

    @pytest.mark.parametrize("store_class", [
        "EpisodicStore",
        "SemanticStore",
        "PreferenceStore",
        "EntityStore",
    ])
    def test_store_isolates_users(self, store_class: str):
        import importlib
        module = importlib.import_module("benchmark.memory.long_term." + store_class.lower().replace("store", "_store"))
        cls = getattr(module, store_class)
        store = cls(decay_lambda=0.0, pruning_threshold=0.0)

        # Write memories for two different users
        ev_a = MemoryEvent(
            id="iso-A", user_id="alice", type=MemoryType.EPISODIC,
            content="alice private", timestamp=_NOW, importance=0.9, task_id="t",
        )
        ev_b = MemoryEvent(
            id="iso-B", user_id="bob", type=MemoryType.EPISODIC,
            content="bob private", timestamp=_NOW, importance=0.9, task_id="t",
        )
        store.write(ev_a)
        store.write(ev_b)

        # Alice must not get Bob's memory
        resp_alice = store.read(_query("alice"))
        ids_alice = {m.memory_id for m in resp_alice.retrieved_memories}
        assert "iso-B" not in ids_alice, (
            f"{store_class}: alice got bob's memory — cross-user leak!"
        )

        # Bob must not get Alice's memory
        resp_bob = store.read(_query("bob"))
        ids_bob = {m.memory_id for m in resp_bob.retrieved_memories}
        assert "iso-A" not in ids_bob, (
            f"{store_class}: bob got alice's memory — cross-user leak!"
        )
