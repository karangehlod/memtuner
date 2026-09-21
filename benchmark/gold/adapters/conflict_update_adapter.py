"""Deterministic controlled tasks for conflict resolution and memory updates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmark.gold.adapters.adapter import (
    DatasetAdapter,
    FingerprintError,
    StatisticsError,
    ValidationError,
    ValidationReport,
)
from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
    TemporalWindow,
)
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class ConflictUpdateAdapter(DatasetAdapter):
    """Build a small, reproducible benchmark where stale evidence is incorrect."""

    def load(self, source: Any = None) -> GoldDataset:
        """Build the controlled conflict/update dataset; source is intentionally unused."""
        events = [
            GoldDayEvents(
                day=0,
                memory_events=[
                    _event("address-v1", "Shipping address is 10 Pine Street.", 0, "address", "initial"),
                    _event("timezone-v1", "User prefers meetings in Pacific time.", 0, "timezone", "initial"),
                ],
            ),
            GoldDayEvents(
                day=2,
                memory_events=[
                    _event("address-v2", "Shipping address changed to 42 Cedar Avenue.", 2, "address", "correction", "address-v1"),
                    _event("timezone-v2", "Calendar owner confirms the user now prefers Eastern time.", 2, "timezone", "authoritative", "timezone-v1"),
                ],
            ),
            GoldDayEvents(
                day=4,
                memory_events=[
                    _event("address-deleted", "The previous address record was deleted; use 42 Cedar Avenue.", 4, "address", "deletion", "address-v1"),
                ],
            ),
        ]
        queries = [
            _query("address-current", "What is the current shipping address?", 3, "address-v2", "42 Cedar Avenue"),
            _query("timezone-current", "Which time zone should meetings use now?", 3, "timezone-v2", "Eastern time"),
            _query("address-deletion", "Which record explains why 10 Pine Street is no longer valid?", 5, "address-deleted", "The record was deleted."),
        ]
        return GoldDataset(
            schema_version="1.0",
            scenario="conflict-update-controlled",
            description="Controlled current-state retrieval tasks with stale competing memories.",
            user_ids=["conflict-user"],
            total_conversation_turns=5,
            events=events,
            queries=queries,
            metadata={
                "dataset_tier": "controlled_verification",
                "conflict_policy": "latest_authoritative_update_or_tombstone",
                "evaluation_mode": "current_state_retrieval",
            },
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as exc:
            raise ValidationError(f"Conflict/update validation error: {exc}") from exc

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            content = json.dumps(dataset.model_dump(mode="json"), sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception as exc:
            raise FingerprintError(f"Failed to fingerprint conflict/update dataset: {exc}") from exc

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as exc:
            raise StatisticsError(f"Failed to compute conflict/update statistics: {exc}") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "ConflictUpdate",
            "version": "1.0",
            "description": "Controlled correction, authority, and deletion retrieval tasks",
            "reproducible": True,
        }


def _event(
    memory_id: str,
    content: str,
    day: int,
    subject: str,
    conflict_label: str,
    update_of: str | None = None,
) -> GoldMemoryEvent:
    provenance = {"source_record_id": memory_id, "conflict_label": conflict_label}
    if update_of:
        provenance["update_of"] = update_of
    return GoldMemoryEvent(
        id=memory_id,
        user_id="conflict-user",
        type=MemoryType.SEMANTIC,
        content=content,
        importance=0.9,
        entities=[subject],
        task_id=subject,
        conversation_turn=day,
        provenance=provenance,
    )


def _query(query_id: str, text: str, day: int, memory_id: str, answer: str) -> GoldQuery:
    return GoldQuery(
        day=day,
        query=text,
        task_id=query_id,
        user_id="conflict-user",
        expected=GoldExpectedResult(
            memory_ids=[memory_id],
            acceptable_modules=[],
            temporal_window=TemporalWindow(not_before_day=0, not_after_day=day),
        ),
        gold_answer=answer,
        provenance={"relevance_judgment": "current_authoritative_evidence", "split": "final_test"},
    )
