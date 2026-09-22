"""Zep adapter — runs Zep Cloud as a memory module inside the benchmark harness.

Maps the harness contract onto the zep-cloud SDK (v3 graph API, verified
against getzep/zep-python reference.md, zep-cloud 3.28.0, 2026-09-22):

    write / write_on_day  →  client.graph.add(data=…, type="text",
                              user_id=…, metadata={"gold_id": …, "day": …})
    read                  →  client.graph.search(query=…, user_id=…, limit=…)
                              (+ a second search with scope="nodes" — the
                              retrieval recipe Zep's own LongMemEval benchmark
                              uses: facts/edges plus entity summaries)
    clear                 →  client.user.delete(user_id=…)

Zep Community Edition is deprecated (getzep/zep README: "legacy/ — Deprecated
Zep Community Edition (unsupported)"), so this adapter targets **Zep Cloud** —
a managed service. That asymmetry vs. self-hosted backends is disclosed in
configs/arena/zep.yaml and must be disclosed next to any results.

Scoring
-------
Zep returns knowledge-graph artifacts (edge facts, node summaries) derived
from ingested text — their UUIDs never correspond to gold memory IDs, and the
SDK exposes no relevance scores (Zep's own benchmark ignores scores too). So:

- ``RetrievedMemory.content`` carries the fact/summary text — the LLM judge
  scores what Zep actually returned (without this the judge would see an
  empty context and silently score Zep 0).
- ``score`` is a synthesized, strictly-descending rank score (1/(rank+1)),
  satisfying the harness's monotonicity invariant without inventing
  relevance the vendor doesn't expose.
- Gold-ID retrieval metrics are undefined for Zep — cells must run
  ``scoring_mode=judge_primary`` (the arena runner sets this via
  ``backend="zep"``).

Isolation: all vendor-side state lives under per-cell namespaced user IDs;
``clear()`` deletes those users (which deletes their graphs).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)

# graph.search limit is capped at 50 by the API (reference.md).
_ZEP_SEARCH_LIMIT_MAX = 50


class ZepStore(MemoryWriter, MemoryReader):
    """Zep-Cloud-backed memory module conforming to the harness store contract."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        module_name: str = "zep_store",
        include_nodes: bool = True,
        **_kwargs: object,
    ) -> None:
        """Initialize the adapter.

        Args:
            namespace: Unique per-cell prefix for vendor-side user IDs.
                Omit to auto-generate (fresh instance per cell → isolated).
            client: Pre-built Zep-compatible client (tests inject a fake).
                Must expose graph.add(), graph.search(), user.add(), user.delete().
            api_key: Zep Cloud project API key; defaults to $ZEP_API_KEY.
            module_name: Logical module name reported in results.
            include_nodes: Also search entity summaries (scope="nodes") and
                append them after edge facts — Zep's own benchmark recipe.
            **_kwargs: Ignored (uniform factory construction).
        """
        if namespace is None:
            import uuid
            namespace = f"zep-{uuid.uuid4().hex[:12]}"
        if not namespace:
            raise ValueError("ZepStore requires a non-empty per-cell namespace")
        self._namespace = namespace
        self._module_name = module_name
        self._include_nodes = include_nodes
        self._client = client if client is not None else self._build_client(api_key)
        self._written_event_ids: set[str] = set()
        self._vendor_user_ids: set[str] = set()

    @staticmethod
    def _build_client(api_key: str | None) -> Any:
        try:
            from zep_cloud.client import Zep
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "ZepStore requires the zep-cloud SDK: pip install zep-cloud. "
                "For tests, inject a fake via ZepStore(client=...)."
            ) from exc
        import os
        key = api_key or os.environ.get("ZEP_API_KEY", "")
        if not key:
            raise RuntimeError(
                "ZepStore requires a Zep Cloud project API key — set ZEP_API_KEY. "
                "(Zep Community Edition is deprecated; this adapter targets Zep Cloud.)"
            )
        return Zep(api_key=key)

    # ------------------------------------------------------------------
    # Identity / bookkeeping
    # ------------------------------------------------------------------

    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    def count(self) -> int:
        """Events written through this adapter (vendor graph size differs)."""
        return len(self._written_event_ids)

    def _vendor_user(self, user_id: str | None) -> str:
        return f"{self._namespace}:{user_id or 'user-default'}"

    def _ensure_user(self, vendor_user: str) -> None:
        if vendor_user in self._vendor_user_ids:
            return
        try:
            self._client.user.add(user_id=vendor_user)
        except Exception:
            # Already exists (or the fake doesn't care) — graph.add will fail
            # loudly if the user genuinely can't be created.
            pass
        self._vendor_user_ids.add(vendor_user)

    # ------------------------------------------------------------------
    # MemoryWriter contract
    # ------------------------------------------------------------------

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event to Zep (day defaults to 0)."""
        self.write_on_day(event, 0)

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Add the event text to the user's graph, day-tagged in metadata."""
        vendor_user = self._vendor_user(event.user_id)
        self._ensure_user(vendor_user)
        self._client.graph.add(
            data=event.content,
            type="text",
            user_id=vendor_user,
            # Zep metadata: max 10 keys, scalar values (reference.md).
            metadata={
                "gold_id": event.id,
                "day": day,
                "memory_type": str(getattr(event.type, "value", event.type)),
                "task_id": event.task_id,
            },
        )
        self._written_event_ids.add(event.id)

    # ------------------------------------------------------------------
    # MemoryReader contract
    # ------------------------------------------------------------------

    def read(self, query: ReadQuery) -> ReadResponse:
        """Zep's own benchmark recipe: edge facts, then entity summaries."""
        start = time.monotonic()
        vendor_user = self._vendor_user(query.context.user_id)
        limit = min(max(query.top_k, 1), _ZEP_SEARCH_LIMIT_MAX)

        edge_results = self._client.graph.search(
            query=query.query, user_id=vendor_user, limit=limit,
        )
        texts: list[tuple[str, str]] = [  # (vendor_id, text)
            (str(getattr(e, "uuid_", None) or getattr(e, "uuid", "") or f"edge-{i}"),
             str(getattr(e, "fact", "") or ""))
            for i, e in enumerate(getattr(edge_results, "edges", None) or [])
        ]
        if self._include_nodes:
            node_results = self._client.graph.search(
                query=query.query, user_id=vendor_user, scope="nodes", limit=limit,
            )
            texts.extend(
                (str(getattr(n, "uuid_", None) or getattr(n, "uuid", "") or f"node-{i}"),
                 f"{getattr(n, 'name', '')}: {getattr(n, 'summary', '')}".strip(": "))
                for i, n in enumerate(getattr(node_results, "nodes", None) or [])
            )

        retrieved: list[RetrievedMemory] = []
        seen: set[str] = set()
        for rank, (vendor_id, text) in enumerate(t for t in texts if t[1]):
            if vendor_id in seen:
                continue
            seen.add(vendor_id)
            retrieved.append(
                RetrievedMemory(
                    memory_id=vendor_id,
                    source_module=self._module_name,
                    # Zep exposes no scores — synthesized rank score keeps the
                    # harness's descending-score invariant honest.
                    score=1.0 / (rank + 1),
                    confidence=1.0,
                    timestamp=datetime.now(UTC),
                    tier=MemoryTier.WARM,
                    decay_factor=1.0,
                    content=text,
                )
            )
            if len(retrieved) >= query.top_k:
                break

        return ReadResponse(
            retrieved_memories=retrieved,
            latency_ms=(time.monotonic() - start) * 1000.0,
            total_candidates=len(texts),
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
        """Delete every namespaced vendor user (and thereby their graphs)."""
        for vendor_user in sorted(self._vendor_user_ids):
            try:
                self._client.user.delete(user_id=vendor_user)
            except Exception:
                logger.warning(
                    "ZepStore.clear: user.delete failed for %s — vendor state "
                    "may leak into later cells",
                    vendor_user,
                )
                raise
        self._vendor_user_ids.clear()
        self._written_event_ids.clear()
