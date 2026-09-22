"""Mem0 adapter — runs Mem0 as a memory module inside the benchmark harness.

Maps the harness contract onto the Mem0 SDK (mem0ai v2 API, verified against
mem0/memory/main.py of mem0ai 2.1.0 on 2026-09-22):

    write / write_on_day  →  client.add(content, user_id=…, metadata=…)
    read                  →  client.search(query, top_k=…, filters={"user_id": …})
    clear                 →  client.delete_all(user_id=…)

Gold-ID provenance
------------------
Mem0 rewrites memories at ingest (LLM fact extraction), so its own memory IDs
never match the benchmark's gold IDs. This adapter threads the original event
ID through ``metadata={"gold_id": …}``, which Mem0 preserves on every fact it
extracts from an add() call, and maps search hits back to gold IDs on read.
Caveats, disclosed wherever these numbers are reported:

- One event may split into several facts → duplicate gold IDs are collapsed
  to their best-ranked occurrence.
- Mem0 may merge or update facts across add() calls; a fact updated by a
  later event carries the *later* event's gold_id. Recall against gold IDs is
  therefore a lower bound for Mem0; judge-based answer scoring is the primary
  cross-system metric (ARENA_PLAN ground rule 6).

Isolation
---------
Every benchmark cell must construct its own Mem0Store with a unique
``namespace``. All vendor-side state lives under user IDs prefixed with that
namespace, and ``clear()`` deletes exactly those. The contract test suite
(tests/contract/test_external_store_contract.py) enforces this.

The Mem0 client is injectable (``client=``) so contract tests run against a
fake without the SDK or an API key; real-backend runs construct the client
lazily from vendor-recommended defaults in configs/arena/mem0.yaml.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


class Mem0Store(MemoryWriter, MemoryReader):
    """Mem0-backed memory module conforming to the harness store contract."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        client: Any | None = None,
        mem0_config: dict | None = None,
        module_name: str = "mem0_store",
        **_kwargs: object,
    ) -> None:
        """Initialize the adapter.

        Args:
            namespace: Unique per-cell prefix for all vendor-side user IDs.
                Omit (None) to auto-generate a fresh one — the registry
                constructs one adapter instance per cell, so an auto-generated
                namespace gives per-cell isolation by default. An explicitly
                passed empty string is an error, never silently shared state.
            client: Pre-built Mem0-compatible client (tests inject a fake).
                Must expose add(), search(), delete_all().
            mem0_config: Optional Mem0 ``Memory.from_config`` dict. Ignored
                when ``client`` is given.
            module_name: Logical module name reported in results.
            **_kwargs: Ignored (uniform factory construction).
        """
        if namespace is None:
            import uuid
            namespace = f"mem0-{uuid.uuid4().hex[:12]}"
        if not namespace:
            raise ValueError("Mem0Store requires a non-empty per-cell namespace")
        self._namespace = namespace
        self._module_name = module_name
        self._client = client if client is not None else self._build_client(mem0_config)
        # Local bookkeeping: events written (harness-side view; Mem0's own
        # fact count may differ) and vendor user IDs touched, for teardown.
        self._written_event_ids: set[str] = set()
        self._vendor_user_ids: set[str] = set()

    @staticmethod
    def _build_client(mem0_config: dict | None) -> Any:
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Mem0Store requires the mem0 SDK: pip install mem0ai. "
                "For tests, inject a fake via Mem0Store(client=...)."
            ) from exc
        if mem0_config is None:
            # The committed arena config is authoritative for real runs —
            # vendor-recommended defaults, deviations disclosed in that file.
            mem0_config = Mem0Store._load_arena_config()
        if mem0_config:
            return Memory.from_config(mem0_config)
        return Memory()

    @staticmethod
    def _load_arena_config() -> dict | None:
        """Load the mem0_config block from configs/arena/mem0.yaml, if any."""
        path = Path(__file__).resolve().parents[3] / "configs" / "arena" / "mem0.yaml"
        if not path.exists():
            return None
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            block = data.get("mem0_config") or {}
            return block or None
        except Exception:
            logger.warning("Could not parse %s — falling back to SDK defaults", path)
            return None

    # ------------------------------------------------------------------
    # Identity / bookkeeping
    # ------------------------------------------------------------------

    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    def count(self) -> int:
        """Return the number of events written through this adapter.

        Harness-side view: Mem0's internal fact count may be higher (one
        event can extract to several facts) or lower (merges/updates).
        """
        return len(self._written_event_ids)

    def _vendor_user(self, user_id: str | None) -> str:
        return f"{self._namespace}:{user_id or 'user-default'}"

    # ------------------------------------------------------------------
    # MemoryWriter contract
    # ------------------------------------------------------------------

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event to Mem0 (day defaults to 0)."""
        self.write_on_day(event, 0)

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Write a memory event tagged with its simulated injection day."""
        vendor_user = self._vendor_user(event.user_id)
        self._client.add(
            event.content,
            user_id=vendor_user,
            metadata={
                "gold_id": event.id,
                "day": day,
                "memory_type": str(getattr(event.type, "value", event.type)),
                "task_id": event.task_id,
            },
        )
        self._written_event_ids.add(event.id)
        self._vendor_user_ids.add(vendor_user)

    # ------------------------------------------------------------------
    # MemoryReader contract
    # ------------------------------------------------------------------

    def read(self, query: ReadQuery) -> ReadResponse:
        """Search Mem0 and map hits back to gold memory IDs.

        Over-fetches from the vendor because fact extraction can split one
        gold event into several hits; duplicates collapse to the best rank.
        """
        start = time.monotonic()
        vendor_user = self._vendor_user(query.context.user_id)
        # mem0ai v2+ API: entity scoping goes in `filters` (top-level user_id
        # is rejected by _reject_top_level_entity_params) and the result cap
        # is `top_k`, not `limit`. Verified against mem0/memory/main.py
        # (mem0ai 2.1.0, retrieved 2026-09-22).
        raw = self._client.search(
            query.query,
            top_k=max(query.top_k * 3, query.top_k),
            filters={"user_id": vendor_user},
        )
        hits = self._normalize_search_results(raw)

        seen: set[str] = set()
        retrieved: list[RetrievedMemory] = []
        for hit in hits:
            metadata = hit.get("metadata") or {}
            memory_id = str(metadata.get("gold_id") or hit.get("id") or "")
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            retrieved.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    source_module=self._module_name,
                    score=self._clamp_score(hit.get("score")),
                    confidence=1.0,
                    timestamp=datetime.now(UTC),
                    tier=MemoryTier.WARM,
                    decay_factor=1.0,
                    # Mem0's extracted fact — the judge must score what Mem0
                    # actually returned, not the original gold event text.
                    content=str(hit.get("memory") or "") or None,
                )
            )
            if len(retrieved) >= query.top_k:
                break

        # Harness invariant: scores are monotonic descending. Vendor scores
        # normally arrive sorted; enforce rather than assume.
        retrieved.sort(key=lambda m: m.score, reverse=True)

        return ReadResponse(
            retrieved_memories=retrieved,
            latency_ms=(time.monotonic() - start) * 1000.0,
            total_candidates=len(hits),
        )

    @staticmethod
    def _normalize_search_results(raw: Any) -> list[dict]:
        """Accept both Mem0 result shapes: {'results': [...]} and bare list."""
        if isinstance(raw, dict):
            results = raw.get("results", [])
        else:
            results = raw or []
        return [r for r in results if isinstance(r, dict)]

    @staticmethod
    def _clamp_score(score: Any) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    # ------------------------------------------------------------------
    # Lifecycle no-ops — Mem0 manages its own memory lifecycle. Exposed so
    # the scenario runner's optional-capability checks are safe.
    # ------------------------------------------------------------------

    def prune(self, memory_ids: list[str]) -> int:
        """External systems manage their own lifecycle — never prune."""
        return 0

    def get_memory_scores(self, day: int) -> dict[str, float]:
        """No harness-side decay scores exist for an external system."""
        return {}

    def clear(self) -> None:
        """Delete all vendor-side state written under this cell's namespace."""
        for vendor_user in sorted(self._vendor_user_ids):
            try:
                self._client.delete_all(user_id=vendor_user)
            except Exception:
                logger.warning(
                    "Mem0Store.clear: delete_all failed for %s — vendor state "
                    "may leak into later cells",
                    vendor_user,
                )
                raise
        self._vendor_user_ids.clear()
        self._written_event_ids.clear()
