"""Private/Custom Data Benchmark Pack Adapter.

Allows users to benchmark their OWN memory events and queries against the
same evaluation engine used for public benchmark packs.

Users provide:
1. events.jsonl — Memory events in the benchmark schema
2. queries.jsonl — Queries with expected results

This enables internal architecture decisions without modifying public packs.

Usage:
  benchmark run --pack private --data-dir ./my_data/

Custom data format (events.jsonl):
  {"memory_id": "evt-001", "user_id": "user-1", "type": "episodic", "content": "...", "day": 0, "importance": 0.7}

Custom data format (queries.jsonl):
  {"query_id": "q-001", "user_id": "user-1", "query_text": "...", "day": 10, "expected_memory_ids": ["evt-001"]}
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
    TemporalWindow,
)
from benchmark.models.memory_event import MemoryType
from benchmark.packs.base import BenchmarkPack, PackMetadata
from benchmark.packs.registry import register_pack

if TYPE_CHECKING:
    from pathlib import Path


@register_pack("private")
class PrivateDataPack(BenchmarkPack):
    """Private/custom data benchmark pack.

    Users map their own production memory events and query traces into
    the benchmark schema for internal evaluation.
    """

    def __init__(self):
        self._events: list[dict[str, Any]] = []
        self._queries: list[dict[str, Any]] = []
        self._loaded = False

    def metadata(self) -> PackMetadata:
        return PackMetadata(
            name="private",
            version="user-defined",
            description="Custom/private workload benchmark. User provides their own events and queries.",
            source_url="local",
            license="user-owned",
            citation="N/A",
            total_queries=0,
            total_sessions=0,
            memory_abilities=["user-defined"],
        )

    def required_files(self) -> list[str]:
        return ["events.jsonl", "queries.jsonl"]

    def load(self, data_dir: Path) -> None:
        """Load custom events and queries.

        Args:
            data_dir: Directory containing events.jsonl and queries.jsonl
        """
        events_path = data_dir / "events.jsonl"
        queries_path = data_dir / "queries.jsonl"

        if not events_path.exists():
            raise FileNotFoundError(
                f"Events file not found at {events_path}. "
                f"Create it following the schema below:\n{self.download_instructions()}"
            )
        if not queries_path.exists():
            raise FileNotFoundError(
                f"Queries file not found at {queries_path}. "
                f"Create it following the schema below:\n{self.download_instructions()}"
            )

        self._events = _load_jsonl(events_path)
        self._queries = _load_jsonl(queries_path)
        self._loaded = True

    def to_gold_dataset(
        self,
        *,
        max_queries: int | None = None,
        seed: int = 42,
        evaluation_horizon: int | None = None,
    ) -> GoldDataset:
        """Convert custom data to GoldDataset.

        Events must have: memory_id, user_id, type, content, day
        Queries must have: query_id, user_id, query_text, day, expected_memory_ids
        """
        if not self._loaded:
            raise RuntimeError("Pack not loaded. Call load() first.")

        # Determine evaluation horizon
        if evaluation_horizon is None:
            max_event_day = max((e.get("day", 0) for e in self._events), default=0)
            max_query_day = max((q.get("day", 0) for q in self._queries), default=0)
            evaluation_horizon = max(max_event_day, max_query_day) + 1

        # Group events by day
        day_to_events: dict[int, list[GoldMemoryEvent]] = {}
        user_ids_set: set[str] = set()

        for event in self._events:
            day = event.get("day", 0)
            if day >= evaluation_horizon:
                continue

            user_id = event.get("user_id", "default-user")
            user_ids_set.add(user_id)

            memory_event = GoldMemoryEvent(
                id=event["memory_id"],
                user_id=user_id,
                type=MemoryType(event.get("type", "episodic")),
                content=event.get("content", ""),
                importance=event.get("importance", 0.5),
                entities=event.get("entities", []),
                task_id=event.get("task_id", ""),
                conversation_turn=event.get("turn", 0),
            )
            day_to_events.setdefault(day, []).append(memory_event)

        # Build GoldDayEvents only for days with events
        all_day_events = [
            GoldDayEvents(day=day, memory_events=events)
            for day, events in sorted(day_to_events.items())
        ]

        # Build queries
        queries_to_use = self._queries
        if max_queries is not None:
            queries_to_use = queries_to_use[:max_queries]

        all_queries: list[GoldQuery] = []
        for q in queries_to_use:
            user_id = q.get("user_id", "default-user")
            user_ids_set.add(user_id)

            expected_ids = q.get("expected_memory_ids", [])
            acceptable_modules = q.get("acceptable_modules", ["episodic_store", "semantic_store"])

            query = GoldQuery(
                day=q.get("day", evaluation_horizon - 1),
                query=q.get("query_text", ""),
                task_id=q.get("task_id", ""),
                user_id=user_id,
                expected=GoldExpectedResult(
                    memory_ids=expected_ids,
                    acceptable_modules=acceptable_modules,
                    temporal_window=TemporalWindow(
                        not_before_day=q.get("earliest_day", 0),
                        not_after_day=q.get("latest_day", evaluation_horizon - 1),
                    ),
                ),
                gold_answer=str(q.get("gold_answer", "")) or None,
            )
            all_queries.append(query)

        return GoldDataset(
            schema_version="1.0",
            scenario="private-custom",
            description="Custom/private workload provided by user",
            user_ids=sorted(user_ids_set),
            events=all_day_events,
            queries=all_queries,
        )

    def download_instructions(self) -> str:
        return """
CUSTOM DATA FORMAT
==================

Create two files in your data directory:

1. events.jsonl — One memory event per line:
   {"memory_id": "evt-001", "user_id": "user-1", "type": "episodic", "content": "User discussed project deadline", "day": 0, "importance": 0.7, "entities": ["project"], "task_id": "task-1"}
   {"memory_id": "evt-002", "user_id": "user-1", "type": "preference", "content": "User prefers dark mode", "day": 1, "importance": 0.9, "entities": [], "task_id": "task-2"}
   {"memory_id": "evt-003", "user_id": "user-1", "type": "semantic", "content": "The API uses REST with JSON", "day": 2, "importance": 0.6, "entities": ["API"], "task_id": "task-1"}

2. queries.jsonl — One query per line:
   {"query_id": "q-001", "user_id": "user-1", "query_text": "What was the project deadline?", "day": 10, "expected_memory_ids": ["evt-001"], "acceptable_modules": ["episodic_store"], "task_id": "task-1"}
   {"query_id": "q-002", "user_id": "user-1", "query_text": "Does user prefer light or dark mode?", "day": 10, "expected_memory_ids": ["evt-002"], "acceptable_modules": ["preference_store"]}

FIELD DESCRIPTIONS:
  memory_id:            Unique ID for this memory event (string)
  user_id:              User who owns this memory (string)
  type:                 One of: episodic, semantic, preference, entity
  content:              The actual memory text content
  day:                  Simulated day number (0-indexed)
  importance:           0.0-1.0 relevance score
  entities:             List of entity names mentioned
  task_id:              Optional task/session grouping
  query_text:           The question to ask
  expected_memory_ids:  List of memory_ids that should be retrieved
  acceptable_modules:   Which memory modules should retrieve this
  earliest_day:         Optional: earliest expected retrieval day
  latest_day:           Optional: latest expected retrieval day

USAGE:
  benchmark run --pack private --data-dir ./my_data/
"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
