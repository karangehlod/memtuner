"""Adapter for PersonaChat dataset - persona-grounded dialogue."""

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


class PersonaChatAdapter(DatasetAdapter):
    """Adapter for PersonaChat persona-grounded dialogue dataset."""

    name = "personachat"

    def load(self, source: Path | str) -> GoldDataset:
        """Load PersonaChat dataset."""
        try:
            source_path = Path(source)
            with open(source_path) as f:
                data = json.load(f)
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read PersonaChat file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in PersonaChat file: {e}")

        if isinstance(data, dict) and "train" in data:
            data = data["train"]

        if not isinstance(data, list):
            raise ValidationError("PersonaChat data must be a list of dialogues")

        if not data:
            raise ValidationError("PersonaChat dataset is empty")

        all_memories: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids = set()

        for dialogue_idx, dialogue in enumerate(data):
            try:
                day = dialogue_idx % 30

                if day not in all_memories:
                    all_memories[day] = []

                # Extract personas as memories
                personas = dialogue.get("personas", [])
                for persona_idx, persona_text in enumerate(personas):
                    memory = GoldMemoryEvent(
                        id=f"persona_{dialogue_idx}_{persona_idx}",
                        user_id=f"user_{dialogue_idx}",
                        type=MemoryType.EPISODIC,
                        content=persona_text,
                        importance=0.9,
                        entities=[],
                        task_id=f"dialogue_{dialogue_idx}",
                        conversation_turn=persona_idx,
                    )
                    all_memories[day].append(memory)
                    user_ids.add(f"user_{dialogue_idx}")

                # Extract dialogue turns as queries
                history = dialogue.get("history", [])
                for _turn_idx, turn in enumerate(history):
                    if isinstance(turn, list) and len(turn) >= 2:
                        query_text = turn[0]
                        relevant_memories = [
                            f"persona_{dialogue_idx}_{i}" for i in range(len(personas))
                        ]

                        expected = GoldExpectedResult(memory_ids=relevant_memories)

                        query = GoldQuery(
                            day=day,
                            query=query_text,
                            task_id=f"dialogue_{dialogue_idx}",
                            user_id=f"user_{dialogue_idx}",
                            expected=expected,
                        )
                        all_queries.append(query)

            except Exception as e:
                raise ValidationError(
                    f"Error parsing PersonaChat dialogue {dialogue_idx}: {e}"
                )

        if not all_memories:
            raise ValidationError("No personas found in PersonaChat dataset")
        if not all_queries:
            raise ValidationError("No dialogue turns found in PersonaChat dataset")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="PersonaChat",
            description="Persona-grounded Dialogue Dataset",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate PersonaChat dataset."""
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"PersonaChat validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint."""
        try:
            query_count = len(dataset.queries)
            memory_count = sum(len(d.memory_events) for d in dataset.events)

            fp_data = {
                "scenario": "PersonaChat",
                "query_count": query_count,
                "memory_count": memory_count,
            }

            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute PersonaChat fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute dataset statistics."""
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute PersonaChat statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return PersonaChat metadata."""
        return {
            "name": "PersonaChat",
            "version": "1.0",
            "description": "Persona-grounded Dialogue - User profile memory in conversation",
            "source": "Facebook Research",
            "format": "JSON with personas and dialogue history",
            "typical_size": "164k utterances",
            "focus": "User modeling, persona consistency",
        }
