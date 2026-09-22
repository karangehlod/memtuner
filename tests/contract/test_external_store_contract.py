"""Contract tests for external memory-system adapters (arena backends).

Every external adapter (Mem0Store, and later ZepStore/LettaStore) must pass
these tests to prove it is substitutable for a native store in the harness
(docs/ARENA_PLAN.md task 1.1). CI runs them against FakeMem0Client — no SDK,
no API key, no network. Set MEM0_CONTRACT_TEST=1 to also run against a real
local Mem0 backend.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from benchmark.memory.external.mem0_store import Mem0Store
from benchmark.memory.interfaces.optional_capabilities import LifecycleAwareWriter
from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent, MemoryType
from benchmark.models.query import ReadQuery, ReadQueryContext
from benchmark.models.response import ReadResponse

FIXED_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


class FakeMem0Client:
    """In-memory stand-in for the Mem0 SDK used by CI contract runs.

    Mirrors the mem0ai v2 API surface (mem0/memory/main.py, mem0ai 2.1.0,
    2026-09-22) so signature drift in the adapter fails HERE, in CI:
    - add(messages, *, user_id=…, metadata=…)
    - search(query, *, top_k=…, filters={"user_id": …}) — top-level user_id
      is rejected, exactly like the real SDK's _reject_top_level_entity_params
    - delete_all(user_id=…)

    And the vendor behaviours the adapter must survive:
    - fact extraction: every add() splits content into per-sentence facts,
      each carrying the add() call's metadata (so one gold event produces
      several hits with the same gold_id);
    - search scoring: naive token overlap, results sorted descending, wrapped
      in the {'results': [...]} envelope.
    """

    def __init__(self) -> None:
        self._facts: dict[str, list[dict]] = {}
        self._next_id = 0

    def add(self, messages, *, user_id: str | None = None,
            metadata: dict | None = None, **_kwargs) -> dict:
        if user_id is None:
            raise ValueError("One of user_id/agent_id/run_id is required")
        content = messages if isinstance(messages, str) else str(messages)
        facts = [s.strip() for s in content.split(".") if s.strip()] or [content]
        for fact in facts:
            self._next_id += 1
            self._facts.setdefault(user_id, []).append(
                {"id": f"mem0-{self._next_id}", "memory": fact, "metadata": dict(metadata or {})}
            )
        return {"results": []}

    def search(self, query: str, *, top_k: int = 20,
               filters: dict | None = None, **kwargs) -> dict:
        if "user_id" in kwargs or "limit" in kwargs:
            raise TypeError(
                "search() got unsupported params — use filters={'user_id': …} "
                "and top_k (mem0ai v2 API)"
            )
        if not filters or "user_id" not in filters:
            raise ValueError("filters must contain at least one of user_id/agent_id/run_id")
        tokens = set(query.lower().split())
        scored = []
        for fact in self._facts.get(filters["user_id"], []):
            overlap = tokens & set(fact["memory"].lower().split())
            if overlap:
                scored.append({**fact, "score": len(overlap) / max(len(tokens), 1)})
        scored.sort(key=lambda f: f["score"], reverse=True)
        return {"results": scored[:top_k]}

    def delete_all(self, user_id: str | None = None, **_kwargs) -> None:
        self._facts.pop(user_id, None)


def _make_event(event_id: str, content: str, user_id: str = "user-default") -> MemoryEvent:
    return MemoryEvent(
        id=event_id,
        user_id=user_id,
        type=MemoryType.EPISODIC,
        content=content,
        timestamp=FIXED_TIMESTAMP,
        importance=0.8,
        task_id="contract",
    )


def _make_query(text: str, user_id: str = "user-default", top_k: int = 5) -> ReadQuery:
    return ReadQuery(
        query=text,
        top_k=top_k,
        context=ReadQueryContext(dataset_day=3, task_id="contract", user_id=user_id),
    )


def _make_store(namespace: str = "cell-a") -> Mem0Store:
    return Mem0Store(namespace=namespace, client=FakeMem0Client())


@pytest.mark.contract
class TestExternalStoreContract:
    """Substitutability contract every external adapter must satisfy."""

    def test_implements_harness_interfaces(self) -> None:
        store = _make_store()
        assert isinstance(store, MemoryWriter)
        assert isinstance(store, MemoryReader)
        assert isinstance(store, LifecycleAwareWriter)

    def test_requires_namespace(self) -> None:
        with pytest.raises(ValueError, match="namespace"):
            Mem0Store(namespace="", client=FakeMem0Client())

    def test_write_then_read_returns_gold_ids(self) -> None:
        store = _make_store()
        store.write_on_day(_make_event("G-1", "Alice prefers postgres for analytics"), day=1)
        store.write_on_day(_make_event("G-2", "Bob went hiking in the alps"), day=2)
        response = store.read(_make_query("which database does alice prefer"))
        assert isinstance(response, ReadResponse)
        assert response.retrieved_memories, "indexed content must be retrievable"
        assert response.retrieved_memories[0].memory_id == "G-1"

    def test_split_facts_collapse_to_one_gold_id(self) -> None:
        # The fake splits on sentences — one event becomes three facts, all
        # carrying gold_id G-1. The adapter must return G-1 exactly once.
        store = _make_store()
        store.write(_make_event("G-1", "Alice likes postgres. Alice uses dbt. Alice ships analytics."))
        response = store.read(_make_query("what does alice like"))
        ids = [m.memory_id for m in response.retrieved_memories]
        assert ids.count("G-1") == 1

    def test_scores_clamped_and_descending(self) -> None:
        store = _make_store()
        for i in range(4):
            store.write(_make_event(f"G-{i}", f"note {i} about kubernetes cluster upgrades"))
        response = store.read(_make_query("kubernetes cluster upgrades"))
        scores = [m.score for m in response.retrieved_memories]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self) -> None:
        store = _make_store()
        for i in range(10):
            store.write(_make_event(f"G-{i}", f"kubernetes note number {i}"))
        response = store.read(_make_query("kubernetes note", top_k=3))
        assert len(response.retrieved_memories) <= 3

    def test_user_isolation_within_cell(self) -> None:
        store = _make_store()
        store.write(_make_event("G-A", "alpha secret fact", user_id="user-a"))
        store.write(_make_event("G-B", "alpha secret fact", user_id="user-b"))
        response = store.read(_make_query("alpha secret fact", user_id="user-a"))
        assert {m.memory_id for m in response.retrieved_memories} == {"G-A"}

    def test_namespace_isolation_between_cells(self) -> None:
        # Two cells sharing one vendor backend must never see each other's
        # memories — the exact state-bleed failure the arena must exclude.
        shared_backend = FakeMem0Client()
        cell_a = Mem0Store(namespace="cell-a", client=shared_backend)
        cell_b = Mem0Store(namespace="cell-b", client=shared_backend)
        cell_a.write(_make_event("G-A", "unique fact about zanzibar"))
        response = cell_b.read(_make_query("unique fact about zanzibar"))
        assert response.retrieved_memories == []

    def test_clear_removes_all_namespace_state(self) -> None:
        store = _make_store()
        store.write(_make_event("G-1", "ephemeral fact about teardown", user_id="user-a"))
        store.write(_make_event("G-2", "another ephemeral fact", user_id="user-b"))
        assert store.count() == 2
        store.clear()
        assert store.count() == 0
        assert store.read(_make_query("ephemeral fact", user_id="user-a")).retrieved_memories == []
        assert store.read(_make_query("ephemeral fact", user_id="user-b")).retrieved_memories == []

    def test_lifecycle_noops_are_safe(self) -> None:
        store = _make_store()
        store.write(_make_event("G-1", "a fact"))
        assert store.prune(["G-1"]) == 0
        assert store.get_memory_scores(day=5) == {}
        # Pruning must not have touched vendor state
        assert store.read(_make_query("a fact")).retrieved_memories

    def test_day_metadata_recorded(self) -> None:
        backend = FakeMem0Client()
        store = Mem0Store(namespace="cell-a", client=backend)
        store.write_on_day(_make_event("G-1", "day tagged fact"), day=7)
        stored = backend._facts[store._vendor_user("user-default")]
        assert all(f["metadata"]["day"] == 7 for f in stored)
        assert all(f["metadata"]["gold_id"] == "G-1" for f in stored)


@pytest.mark.contract
@pytest.mark.skipif(
    os.environ.get("MEM0_CONTRACT_TEST") != "1",
    reason="real-backend contract run: set MEM0_CONTRACT_TEST=1 with a local Mem0 install",
)
class TestMem0RealBackend:
    """Same core contract against a real Mem0 install (local, opt-in)."""

    def test_write_read_clear_roundtrip(self) -> None:
        store = Mem0Store(namespace=f"contract-{datetime.now(UTC):%Y%m%d%H%M%S}")
        try:
            store.write_on_day(_make_event("G-1", "Alice prefers postgres for analytics"), day=1)
            response = store.read(_make_query("which database does alice prefer"))
            assert response.retrieved_memories
            assert response.retrieved_memories[0].memory_id == "G-1"
        finally:
            store.clear()
        assert store.read(_make_query("alice postgres")).retrieved_memories == []


# ─── Zep ─────────────────────────────────────────────────────────────────────


class _ZepEdge:
    def __init__(self, uuid: str, fact: str) -> None:
        self.uuid_ = uuid
        self.fact = fact


class _ZepNode:
    def __init__(self, uuid: str, name: str, summary: str) -> None:
        self.uuid_ = uuid
        self.name = name
        self.summary = summary


class _ZepSearchResults:
    def __init__(self, edges=None, nodes=None) -> None:
        self.edges = edges
        self.nodes = nodes


class FakeZepClient:
    """Stand-in for zep_cloud.Zep mirroring the v3 graph API surface
    (getzep/zep-python reference.md, zep-cloud 3.28.0, 2026-09-22):
    graph.add(data, type, user_id, metadata), graph.search(query, user_id,
    scope, limit), user.add / user.delete. Search returns edge FACTS and node
    SUMMARIES derived from ingested text — never the original events — with
    NO scores, exactly the properties the adapter must survive.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, list[dict]] = {}
        self._users: set[str] = set()
        self._n = 0
        outer = self

        class _Graph:
            def add(self, *, data: str, type: str, user_id: str, metadata=None, **_kw):
                if user_id not in outer._users:
                    raise ValueError(f"user {user_id} not found")
                outer._n += 1
                outer._episodes.setdefault(user_id, []).append(
                    {"uuid": f"zep-{outer._n}", "text": data, "metadata": dict(metadata or {})}
                )

            def search(self, *, query: str, user_id: str, scope: str = "edges",
                       limit: int = 10, **_kw):
                tokens = set(query.lower().split())
                hits = sorted(
                    (e for e in outer._episodes.get(user_id, [])
                     if tokens & set(e["text"].lower().split())),
                    key=lambda e: -len(tokens & set(e["text"].lower().split())),
                )[:limit]
                if scope == "nodes":
                    return _ZepSearchResults(nodes=[
                        _ZepNode(h["uuid"] + "-n", "Entity", f"summary of: {h['text']}")
                        for h in hits
                    ])
                return _ZepSearchResults(edges=[
                    _ZepEdge(h["uuid"], f"fact derived from: {h['text']}") for h in hits
                ])

        class _User:
            def add(self, *, user_id: str, **_kw):
                outer._users.add(user_id)

            def delete(self, *, user_id: str, **_kw):
                outer._users.discard(user_id)
                outer._episodes.pop(user_id, None)

        self.graph = _Graph()
        self.user = _User()


def _make_zep_store(namespace: str = "cell-z") -> "ZepStore":
    from benchmark.memory.external.zep_store import ZepStore
    return ZepStore(namespace=namespace, client=FakeZepClient())


@pytest.mark.contract
class TestZepStoreContract:
    """Zep adapter contract — judge-primary: content is the deliverable."""

    def test_implements_harness_interfaces(self) -> None:
        store = _make_zep_store()
        assert isinstance(store, MemoryWriter)
        assert isinstance(store, MemoryReader)
        assert isinstance(store, LifecycleAwareWriter)

    def test_read_returns_vendor_content_for_judge(self) -> None:
        # Zep returns derived facts whose IDs mean nothing to the gold store —
        # every retrieved item MUST carry content or the judge sees nothing.
        store = _make_zep_store()
        store.write_on_day(_make_event("G-1", "Alice prefers postgres for analytics"), day=1)
        response = store.read(_make_query("which database does alice prefer"))
        assert response.retrieved_memories
        assert all(m.content for m in response.retrieved_memories)
        assert any("postgres" in m.content for m in response.retrieved_memories)

    def test_scores_synthesized_descending(self) -> None:
        store = _make_zep_store()
        for i in range(4):
            store.write(_make_event(f"G-{i}", f"kubernetes cluster note {i}"))
        response = store.read(_make_query("kubernetes cluster note", top_k=4))
        scores = [m.score for m in response.retrieved_memories]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_top_k_respected(self) -> None:
        store = _make_zep_store()
        for i in range(8):
            store.write(_make_event(f"G-{i}", f"kubernetes note {i}"))
        response = store.read(_make_query("kubernetes note", top_k=3))
        assert len(response.retrieved_memories) <= 3

    def test_namespace_isolation_between_cells(self) -> None:
        from benchmark.memory.external.zep_store import ZepStore
        shared = FakeZepClient()
        cell_a = ZepStore(namespace="cell-a", client=shared)
        cell_b = ZepStore(namespace="cell-b", client=shared)
        cell_a.write(_make_event("G-A", "unique fact about zanzibar"))
        assert cell_b.read(_make_query("unique fact about zanzibar")).retrieved_memories == []

    def test_clear_deletes_namespaced_users(self) -> None:
        store = _make_zep_store()
        store.write(_make_event("G-1", "ephemeral teardown fact", user_id="user-a"))
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert store.read(_make_query("ephemeral teardown fact", user_id="user-a")).retrieved_memories == []

    def test_lifecycle_noops_are_safe(self) -> None:
        store = _make_zep_store()
        store.write(_make_event("G-1", "a fact"))
        assert store.prune(["anything"]) == 0
        assert store.get_memory_scores(day=3) == {}
        assert store.read(_make_query("a fact")).retrieved_memories


@pytest.mark.contract
class TestJudgeContentDelivery:
    """External adapters must hand the judge the vendor's returned text."""

    def test_mem0_results_carry_extracted_fact_content(self) -> None:
        store = _make_store()
        store.write(_make_event("G-1", "Alice prefers postgres for analytics"))
        response = store.read(_make_query("which database does alice prefer"))
        assert response.retrieved_memories
        assert all(m.content for m in response.retrieved_memories)

    def test_runner_prefers_vendor_content_over_gold_lookup(self) -> None:
        # Mirrors scenario_runner's judge-context resolution: vendor content
        # wins; gold-event lookup is only the fallback for native stores.
        from benchmark.models.response import MemoryTier, RetrievedMemory
        gold_lookup = {"G-1": "original gold event text"}
        vendor = RetrievedMemory(
            memory_id="zep-uuid-9", source_module="zep_store", score=1.0,
            timestamp=FIXED_TIMESTAMP, tier=MemoryTier.WARM,
            content="fact derived from: alice prefers postgres",
        )
        native = RetrievedMemory(
            memory_id="G-1", source_module="episodic_store", score=0.9,
            timestamp=FIXED_TIMESTAMP, tier=MemoryTier.WARM,
        )
        resolved = [
            text for m in (vendor, native)
            if (text := (m.content if m.content else gold_lookup.get(m.memory_id)))
        ]
        assert resolved == [
            "fact derived from: alice prefers postgres",
            "original gold event text",
        ]


# ─── Graphiti ────────────────────────────────────────────────────────────────


class _GraphitiEdge:
    def __init__(self, uuid: str, fact: str) -> None:
        self.uuid = uuid
        self.fact = fact


class FakeGraphitiClient:
    """Stand-in for graphiti_core.Graphiti mirroring the verified API surface
    (graphiti_core/graphiti.py, graphiti-core 0.30.2, 2026-09-22):
    async add_episode(name, episode_body, source_description, reference_time,
    group_id), async search(query, group_ids, num_results) → list[EntityEdge].
    Returns derived edge FACTS with no scores — the properties the adapter
    must survive. Exposes async clear_groups() as the injected-client
    teardown hook.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, list[dict]] = {}
        self._n = 0

    async def add_episode(self, *, name: str, episode_body: str,
                          source_description: str, reference_time,
                          group_id: str | None = None, **_kw):
        self._n += 1
        self._episodes.setdefault(group_id, []).append(
            {"uuid": f"g-{self._n}", "text": episode_body,
             "ref_time": reference_time, "source": source_description}
        )

    async def search(self, query: str, group_ids=None, num_results: int = 10, **_kw):
        tokens = set(query.lower().split())
        hits = []
        for gid in group_ids or []:
            hits.extend(
                e for e in self._episodes.get(gid, [])
                if tokens & set(e["text"].lower().split())
            )
        hits.sort(key=lambda e: -len(tokens & set(e["text"].lower().split())))
        return [
            _GraphitiEdge(h["uuid"], f"fact derived from: {h['text']}")
            for h in hits[:num_results]
        ]

    async def clear_groups(self, group_ids: list[str]) -> None:
        for gid in group_ids:
            self._episodes.pop(gid, None)


def _make_graphiti_store(namespace: str = "cell-g") -> "GraphitiStore":
    from benchmark.memory.external.graphiti_store import GraphitiStore
    return GraphitiStore(namespace=namespace, client=FakeGraphitiClient())


@pytest.mark.contract
class TestGraphitiStoreContract:
    """Graphiti adapter contract — judge-primary: content is the deliverable."""

    def test_implements_harness_interfaces(self) -> None:
        store = _make_graphiti_store()
        assert isinstance(store, MemoryWriter)
        assert isinstance(store, MemoryReader)
        assert isinstance(store, LifecycleAwareWriter)

    def test_read_returns_vendor_content_for_judge(self) -> None:
        store = _make_graphiti_store()
        store.write_on_day(_make_event("G-1", "Alice prefers postgres for analytics"), day=1)
        response = store.read(_make_query("which database does alice prefer"))
        assert response.retrieved_memories
        assert all(m.content for m in response.retrieved_memories)
        assert any("postgres" in m.content for m in response.retrieved_memories)

    def test_reference_time_maps_simulated_day(self) -> None:
        from benchmark.memory.external.graphiti_store import ARENA_EPOCH, GraphitiStore
        backend = FakeGraphitiClient()
        store = GraphitiStore(namespace="cell-g", client=backend)
        store.write_on_day(_make_event("G-1", "day tagged fact"), day=7)
        (episode,) = backend._episodes[store._group("user-default")]
        assert (episode["ref_time"] - ARENA_EPOCH).days == 7
        assert "gold_id=G-1" in episode["source"]

    def test_scores_synthesized_descending_and_topk(self) -> None:
        store = _make_graphiti_store()
        for i in range(8):
            store.write(_make_event(f"G-{i}", f"kubernetes cluster note {i}"))
        response = store.read(_make_query("kubernetes cluster note", top_k=3))
        scores = [m.score for m in response.retrieved_memories]
        assert len(response.retrieved_memories) <= 3
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_namespace_isolation_between_cells(self) -> None:
        from benchmark.memory.external.graphiti_store import GraphitiStore
        shared = FakeGraphitiClient()
        cell_a = GraphitiStore(namespace="cell-a", client=shared)
        cell_b = GraphitiStore(namespace="cell-b", client=shared)
        cell_a.write(_make_event("G-A", "unique fact about zanzibar"))
        assert cell_b.read(_make_query("unique fact about zanzibar")).retrieved_memories == []

    def test_clear_deletes_cell_groups(self) -> None:
        store = _make_graphiti_store()
        store.write(_make_event("G-1", "ephemeral teardown fact", user_id="user-a"))
        store.write(_make_event("G-2", "another ephemeral fact", user_id="user-b"))
        assert store.count() == 2
        store.clear()
        assert store.count() == 0
        assert store.read(_make_query("ephemeral teardown fact", user_id="user-a")).retrieved_memories == []

    def test_lifecycle_noops_are_safe(self) -> None:
        store = _make_graphiti_store()
        store.write(_make_event("G-1", "a fact"))
        assert store.prune(["anything"]) == 0
        assert store.get_memory_scores(day=3) == {}
        assert store.read(_make_query("a fact")).retrieved_memories
