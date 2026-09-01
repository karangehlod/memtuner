"""Adapter for NarrativeQA - long narrative comprehension."""

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


class NarrativeQAAdapter(DatasetAdapter):
    """Adapter for NarrativeQA long narrative comprehension."""

    name = "narrativeqa"

    def load(self, source: Path | str) -> GoldDataset:
        """Load NarrativeQA dataset."""
        try:
            with open(source) as f:
                data = json.load(f)
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read NarrativeQA file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in NarrativeQA file: {e}")

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list):
            raise ValidationError("NarrativeQA data must be a list")

        if not data:
            raise ValidationError("NarrativeQA dataset is empty")

        all_memories, all_queries = {}, []

        for n_idx, narrative in enumerate(data):
            try:
                day = n_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                # Narrative text as long-form memory
                narrative_text = narrative.get("story", narrative.get("text", ""))
                if not narrative_text:
                    continue

                # Split long narrative into chunks
                chunk_size = 2000
                for chunk_idx in range(0, len(narrative_text), chunk_size):
                    chunk = narrative_text[chunk_idx:chunk_idx+chunk_size]

                    memory = GoldMemoryEvent(
                        id=f"narrative_{n_idx}_{chunk_idx//chunk_size}",
                        user_id="user-default",
                        type=MemoryType.EPISODIC,
                        content=chunk,
                        importance=0.8,
                        entities=[],
                        task_id=f"narrative_{n_idx}",
                        conversation_turn=chunk_idx//chunk_size,
                    )
                    all_memories[day].append(memory)

                # Questions as queries
                questions = narrative.get("questions", [])
                for _q_idx, q_data in enumerate(questions):
                    question = q_data if isinstance(q_data, str) else q_data.get("question", "")
                    if not question:
                        continue

                    memory_ids = [f"narrative_{n_idx}_{i}" for i in range(len(narrative_text)//chunk_size + 1)]
                    expected = GoldExpectedResult(memory_ids=memory_ids)

                    query = GoldQuery(
                        day=day,
                        query=question,
                        task_id=f"narrative_{n_idx}",
                        user_id="user-default",
                        expected=expected,
                    )
                    all_queries.append(query)

            except Exception:
                continue

        if not all_memories:
            raise ValidationError("No narratives found")
        if not all_queries:
            raise ValidationError("No questions found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="NarrativeQA",
            description="NarrativeQA Long Narrative Comprehension",
            user_ids=["user-default"],
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"NarrativeQA validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "NarrativeQA",
                "query_count": len(dataset.queries),
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute NarrativeQA fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute NarrativeQA statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "NarrativeQA",
            "version": "1.0",
            "description": "NarrativeQA - Comprehension of long narratives (books/movies)",
            "source": "DeepMind",
            "format": "JSON with narrative text and questions",
            "typical_size": "1.5k narratives, 31k questions",
            "focus": "Long-form memory, narrative comprehension, synthesis",
        }
