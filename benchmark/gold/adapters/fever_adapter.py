"""Adapter for FEVER - fact verification."""

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.gold.adapters.adapter import (
    AdapterError,
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
)
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class FEVERAdapter(DatasetAdapter):
    """Adapter for FEVER fact verification dataset."""

    name = "fever"

    def load(self, source: Path | str) -> GoldDataset:
        """Load FEVER dataset."""
        try:
            with open(source) as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read FEVER file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in FEVER file: {e}")

        if not data:
            raise ValidationError("FEVER dataset is empty")

        all_memories, all_queries = {}, []

        for c_idx, claim in enumerate(data[:10000]):
            try:
                day = c_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                claim_text = claim.get("claim", "")
                if not claim_text:
                    continue

                # Evidence facts as memories. Each raw evidence entry is
                # [annotation_id, evidence_id, wiki_page, sentence_id]; only
                # ev[2] (the wiki page) is available without the wiki dump.
                evidence = claim.get("evidence", [])
                memory_ids: list[str] = []
                for e_idx, evidence_set in enumerate(evidence[:5]):
                    if not isinstance(evidence_set, list):
                        continue
                    for s_idx, ev in enumerate(evidence_set[:2]):
                        if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                            ev_text = ev[2]
                        else:
                            ev_text = str(ev)

                        if ev_text:
                            mem_id = f"fact_{c_idx}_{e_idx}_{s_idx}"
                            memory = GoldMemoryEvent(
                                id=mem_id,
                                user_id="user-default",
                                type=MemoryType.EPISODIC,
                                content=str(ev_text)[:300],
                                importance=0.9,
                                entities=[],
                                task_id=f"claim_{c_idx}",
                                conversation_turn=e_idx,
                            )
                            all_memories[day].append(memory)
                            memory_ids.append(mem_id)

                # NOT ENOUGH INFO claims have no evidence — nothing to retrieve.
                if not memory_ids:
                    continue

                expected = GoldExpectedResult(memory_ids=memory_ids)

                query = GoldQuery(
                    day=day,
                    query=claim_text,
                    task_id=f"claim_{c_idx}",
                    user_id="user-default",
                    expected=expected,
                )
                all_queries.append(query)

            except Exception:
                continue

        if not all_memories:
            raise ValidationError("No facts found")
        if not all_queries:
            raise ValidationError("No claims found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="FEVER",
            description="FEVER Fact Verification",
            user_ids=["user-default"],
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"FEVER validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "FEVER",
                "claim_count": len(dataset.queries),
                "fact_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute FEVER fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute FEVER statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "FEVER",
            "version": "1.0",
            "description": "FEVER - Fact Extraction and Verification with Wikipedia evidence",
            "source": "Facebook & University of Washington",
            "format": "JSON Lines with claims and evidence",
            "typical_size": "185k claims",
            "focus": "Fact memory, evidence retrieval, verification",
        }
