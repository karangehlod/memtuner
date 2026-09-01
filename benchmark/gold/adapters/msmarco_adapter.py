"""Adapter for MS MARCO - large-scale web QA."""

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


class MSMarcoAdapter(DatasetAdapter):
    """Adapter for MS MARCO large-scale web QA."""

    name = "msmarco"

    def load(self, source: Path | str) -> GoldDataset:
        """Load MS MARCO dataset."""
        try:
            with open(source) as f:
                data = [json.loads(line) for line in f if line.strip()]
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read MS MARCO file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in MS MARCO file: {e}")

        if not data:
            raise ValidationError("MS MARCO dataset is empty")

        all_memories, all_queries = {}, []

        for q_idx, item in enumerate(data[:10000]):  # Limit for memory
            try:
                day = q_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                query = item.get("query", item.get("query_text", ""))
                if not query:
                    continue

                # Document passages as memories
                passages = item.get("passages", [])
                for p_idx, passage in enumerate(passages[:5]):
                    text = passage.get("passage_text", passage.get("passage", "")) if isinstance(passage, dict) else str(passage)

                    memory = GoldMemoryEvent(
                        id=f"doc_{q_idx}_{p_idx}",
                        user_id="user-default",
                        type=MemoryType.EPISODIC,
                        content=text[:300],
                        importance=0.75,
                        entities=[],
                        task_id=f"q_{q_idx}",
                        conversation_turn=p_idx,
                    )
                    all_memories[day].append(memory)

                memory_ids = [f"doc_{q_idx}_{i}" for i in range(min(len(passages), 5))] or ["doc_0"]
                expected = GoldExpectedResult(memory_ids=memory_ids)

                q = GoldQuery(
                    day=day,
                    query=query,
                    task_id=f"q_{q_idx}",
                    user_id="user-default",
                    expected=expected,
                )
                all_queries.append(q)

            except Exception:
                continue

        if not all_memories:
            raise ValidationError("No documents found")
        if not all_queries:
            raise ValidationError("No queries found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="MS MARCO",
            description="MS MARCO Large-Scale Web QA",
            user_ids=["user-default"],
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"MS MARCO validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "MS MARCO",
                "query_count": len(dataset.queries),
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute MS MARCO fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute MS MARCO statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "MS MARCO",
            "version": "1.0",
            "description": "MS MARCO - Large-scale web search QA at IR scale",
            "source": "Microsoft Research",
            "format": "JSON Lines with queries and passages",
            "typical_size": "100k+ questions",
            "focus": "Large-scale IR, web document retrieval",
        }
