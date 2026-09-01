"""Adapter for HotpotQA - multi-hop reasoning."""

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


class HotpotQAAdapter(DatasetAdapter):
    """Adapter for HotpotQA multi-hop reasoning dataset."""

    name = "hotpotqa"

    def load(self, source: Path | str) -> GoldDataset:
        """Load HotpotQA dataset."""
        try:
            with open(source) as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read HotpotQA file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in HotpotQA file: {e}")

        if not data:
            raise ValidationError("HotpotQA dataset is empty")

        all_memories, all_queries = {}, []

        for q_idx, item in enumerate(data[:5000]):
            try:
                day = q_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                question = item.get("question", "")
                if not question:
                    continue

                # Context paragraphs as memories
                context = item.get("context", [])
                for c_idx, (title, content) in enumerate(context[:4]):
                    memory = GoldMemoryEvent(
                        id=f"doc_{q_idx}_{c_idx}",
                        user_id="user-default",
                        type=MemoryType.EPISODIC,
                        content=f"{title}: {content[:400]}",
                        importance=0.85,
                        entities=[],
                        task_id=f"q_{q_idx}",
                        conversation_turn=c_idx,
                    )
                    all_memories[day].append(memory)

                memory_ids = [f"doc_{q_idx}_{i}" for i in range(min(len(context), 4))] or ["doc_0"]
                expected = GoldExpectedResult(memory_ids=memory_ids)

                query = GoldQuery(
                    day=day,
                    query=question,
                    task_id=f"q_{q_idx}",
                    user_id="user-default",
                    expected=expected,
                )
                all_queries.append(query)

            except Exception:
                continue

        if not all_memories:
            raise ValidationError("No contexts found")
        if not all_queries:
            raise ValidationError("No questions found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="HotpotQA",
            description="HotpotQA Multi-Hop Question Answering",
            user_ids=["user-default"],
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"HotpotQA validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "HotpotQA",
                "query_count": len(dataset.queries),
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute HotpotQA fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute HotpotQA statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "HotpotQA",
            "version": "1.0",
            "description": "HotpotQA - Multi-hop reasoning requiring multiple facts",
            "source": "Carnegie Mellon University",
            "format": "JSON Lines with questions and supporting facts",
            "typical_size": "113k questions",
            "focus": "Multi-fact reasoning, complex memory access patterns",
        }
