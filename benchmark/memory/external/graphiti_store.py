"""Graphiti adapter — the open-source graph-memory framework that powers Zep.

Fills the arena's knowledge-graph-memory slot under ground rule 9
(zero-cost/local-only): Zep Cloud is a paid managed service, so the arena
runs Graphiti (getzep/graphiti, graphiti-core 0.30.2) — self-hosted, free,
and labeled **Graphiti**, never "Zep".

API verified against graphiti_core/graphiti.py and the project README
(2026-09-22):

    write / write_on_day  →  await graphiti.add_episode(name=…, episode_body=…,
                              source_description=…, reference_time=…, group_id=…)
    read                  →  await graphiti.search(query, group_ids=[…],
                              num_results=…) → list[EntityEdge] (hybrid
                              semantic + BM25 + graph search)
    clear                 →  clear_data(driver, group_ids=[…])
                              (graphiti_core.utils.maintenance.graph_data_operations)

Graphiti's API is async; the harness contract is sync, so this adapter owns a
private event loop and drives each call to completion.

Scoring: identical situation to Zep — search returns derived edge FACTS whose
UUIDs never match gold memory IDs, and no relevance scores are exposed. So
``RetrievedMemory.content`` carries the fact text for the LLM judge,
``score`` is a synthesized descending rank score, and cells run
``scoring_mode=judge_primary`` (``backend="graphiti"``).

Isolation: Graphiti has first-class graph partitions — ``group_id``. Each
(namespace, user) pair maps to its own group; ``clear()`` deletes exactly
those groups.

Day semantics: ``reference_time`` is a real datetime, so simulated day N maps
to ``ARENA_EPOCH + N days`` — Graphiti's temporal reasoning sees a consistent
timeline instead of ingest wall-time.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)

# Day 0 of every simulated timeline. Arbitrary but fixed: reproducible
# reference_times across runs and machines.
ARENA_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


class GraphitiStore(MemoryWriter, MemoryReader):
    """Graphiti-backed memory module conforming to the harness store contract."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        client: Any | None = None,
        module_name: str = "graphiti_store",
        **_kwargs: object,
    ) -> None:
        """Initialize the adapter.

        Args:
            namespace: Unique per-cell prefix for Graphiti group_ids.
                Omit to auto-generate (fresh instance per cell → isolated).
            client: Pre-built Graphiti-compatible object (tests inject a
                fake). Must expose async add_episode()/search() and a
                ``driver`` attribute for teardown.
            module_name: Logical module name reported in results.
            **_kwargs: Ignored (uniform factory construction).
        """
        if namespace is None:
            import uuid
            namespace = f"graphiti-{uuid.uuid4().hex[:12]}"
        if not namespace:
            raise ValueError("GraphitiStore requires a non-empty per-cell namespace")
        self._namespace = namespace
        self._module_name = module_name
        self._loop: Any = None  # created lazily; leaked loops print GC noise
        self._client = client if client is not None else self._build_client()
        self._written_event_ids: set[str] = set()
        self._group_ids: set[str] = set()

    def _run(self, coro: Any) -> Any:
        if self._loop is None:
            import asyncio
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def __del__(self) -> None:
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed():
            loop.close()

    def _build_client(self) -> Any:
        """Build a fully-local Graphiti from configs/arena/graphiti.yaml.

        Local stack per the vendor's own documented recipe (README section
        "Using Graphiti with OpenAI-compatible providers and local LLMs"):
        OpenAIGenericClient + OpenAIEmbedder pointed at Ollama, and an
        embedded FalkorDB-Lite graph driver (no server).
        """
        try:
            from graphiti_core import Graphiti
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "GraphitiStore requires graphiti-core: "
                'pip install "graphiti-core[falkordblite]"  (Python 3.12+, embedded '
                "graph DB) or graphiti-core[falkordb] with a FalkorDB docker. "
                "For tests, inject a fake via GraphitiStore(client=...)."
            ) from exc

        cfg = self._load_arena_config()
        llm_cfg = LLMConfig(
            api_key="ollama",
            model=cfg.get("llm_model", "deepseek-r1:7b"),
            small_model=cfg.get("llm_model", "deepseek-r1:7b"),
            base_url=cfg.get("ollama_base_url", "http://localhost:11434/v1"),
        )
        llm_client = OpenAIGenericClient(config=llm_cfg)
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="ollama",
                embedding_model=cfg.get("embedding_model", "nomic-embed-text"),
                embedding_dim=int(cfg.get("embedding_dim", 768)),
                base_url=cfg.get("ollama_base_url", "http://localhost:11434/v1"),
            )
        )

        driver = self._build_driver(cfg)
        client = Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)
        self._run(client.build_indices_and_constraints())
        return client

    @staticmethod
    def _build_driver(cfg: dict) -> Any:
        backend = cfg.get("graph_backend", "falkordblite")
        if backend == "falkordblite":
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from redislite.async_falkordb_client import AsyncFalkorDB
            db_path = cfg.get("falkordblite_path", "/tmp/memtuner_arena_graphiti.db")
            return FalkorDriver(falkor_db=AsyncFalkorDB(dbfilename=db_path))
        if backend == "falkordb":
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            return FalkorDriver(
                host=cfg.get("falkordb_host", "localhost"),
                port=int(cfg.get("falkordb_port", 6379)),
            )
        if backend == "neo4j":
            from graphiti_core.driver.neo4j_driver import Neo4jDriver
            return Neo4jDriver(
                uri=cfg.get("neo4j_uri", "bolt://localhost:7687"),
                user=cfg.get("neo4j_user", "neo4j"),
                password=cfg.get("neo4j_password", "password"),
            )
        raise ValueError(f"unknown graphiti graph_backend: {backend}")

    @staticmethod
    def _load_arena_config() -> dict:
        from pathlib import Path
        path = Path(__file__).resolve().parents[3] / "configs" / "arena" / "graphiti.yaml"
        if not path.exists():
            return {}
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("graphiti_config") or {}
        except Exception:
            logger.warning("Could not parse %s — using built-in local defaults", path)
            return {}

    # ------------------------------------------------------------------
    # Identity / bookkeeping
    # ------------------------------------------------------------------

    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    def count(self) -> int:
        """Events written through this adapter (graph node count differs)."""
        return len(self._written_event_ids)

    def _group(self, user_id: str | None) -> str:
        return f"{self._namespace}:{user_id or 'user-default'}"

    # ------------------------------------------------------------------
    # MemoryWriter contract
    # ------------------------------------------------------------------

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event to Graphiti (day defaults to 0)."""
        self.write_on_day(event, 0)

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Ingest the event as an episode on the simulated timeline."""
        group_id = self._group(event.user_id)
        self._run(
            self._client.add_episode(
                name=event.id,
                episode_body=event.content,
                source_description=f"gold_id={event.id};day={day}",
                reference_time=ARENA_EPOCH + timedelta(days=day),
                group_id=group_id,
            )
        )
        self._written_event_ids.add(event.id)
        self._group_ids.add(group_id)

    # ------------------------------------------------------------------
    # MemoryReader contract
    # ------------------------------------------------------------------

    def read(self, query: ReadQuery) -> ReadResponse:
        """Graphiti hybrid search — returns edge facts as judge context."""
        start = time.monotonic()
        group_id = self._group(query.context.user_id)
        edges = self._run(
            self._client.search(
                query.query,
                group_ids=[group_id],
                num_results=query.top_k,
            )
        )

        retrieved: list[RetrievedMemory] = []
        seen: set[str] = set()
        rank = 0
        for edge in edges or []:
            fact = str(getattr(edge, "fact", "") or "")
            vendor_id = str(getattr(edge, "uuid", "") or f"edge-{rank}")
            if not fact or vendor_id in seen:
                continue
            seen.add(vendor_id)
            retrieved.append(
                RetrievedMemory(
                    memory_id=vendor_id,
                    source_module=self._module_name,
                    # Graphiti exposes no relevance scores on edges — the
                    # synthesized rank score keeps the descending invariant.
                    score=1.0 / (rank + 1),
                    confidence=1.0,
                    timestamp=datetime.now(UTC),
                    tier=MemoryTier.WARM,
                    decay_factor=1.0,
                    content=fact,
                )
            )
            rank += 1
            if len(retrieved) >= query.top_k:
                break

        return ReadResponse(
            retrieved_memories=retrieved,
            latency_ms=(time.monotonic() - start) * 1000.0,
            total_candidates=len(edges or []),
        )

    # ------------------------------------------------------------------
    # Lifecycle no-ops + teardown
    # ------------------------------------------------------------------

    def prune(self, memory_ids: list[str]) -> int:
        """External systems manage their own lifecycle — never prune."""
        return 0

    def get_memory_scores(self, day: int) -> dict[str, float]:
        """No harness-side decay scores exist for an external system."""
        return {}

    def clear(self) -> None:
        """Delete every graph partition this cell created."""
        if not self._group_ids:
            self._written_event_ids.clear()
            return
        # Injected clients (fakes, custom wrappers) may provide their own
        # group teardown; the real path uses graphiti's maintenance util.
        clear_groups = getattr(self._client, "clear_groups", None)
        if clear_groups is not None:
            self._run(clear_groups(sorted(self._group_ids)))
        else:
            from graphiti_core.utils.maintenance.graph_data_operations import clear_data
            self._run(clear_data(self._client.driver, group_ids=sorted(self._group_ids)))
        self._group_ids.clear()
        self._written_event_ids.clear()
