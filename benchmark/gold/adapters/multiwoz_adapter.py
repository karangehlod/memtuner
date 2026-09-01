"""Adapter for MultiWOZ - task-oriented dialogue state tracking."""

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


class MultiWOZAdapter(DatasetAdapter):
    """Adapter for MultiWOZ task-oriented dialogue."""

    name = "multiwoz"

    def load(self, source: Path | str) -> GoldDataset:
        """Load MultiWOZ dataset."""
        try:
            with open(source) as f:
                data = json.load(f)
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read MultiWOZ file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in MultiWOZ file: {e}")

        if not isinstance(data, dict):
            raise ValidationError("MultiWOZ data must be a dict of dialogues")

        if not data:
            raise ValidationError("MultiWOZ dataset is empty")

        all_memories, all_queries = {}, []

        for d_idx, (_dialogue_id, dialogue) in enumerate(data.items()):
            try:
                day = d_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                # Dialogue state as memories
                turns = dialogue.get("turns", dialogue.get("messages", []))
                for turn_idx, turn in enumerate(turns):
                    # Extract user utterance
                    if isinstance(turn, dict):
                        speaker = turn.get("speaker", "")
                        if speaker.lower() != "user":
                            continue
                        utterance = turn.get("utterance", "")
                    else:
                        utterance = str(turn)

                    if not utterance:
                        continue

                    # Create memory from turn context
                    memory = GoldMemoryEvent(
                        id=f"turn_{d_idx}_{turn_idx}",
                        user_id=f"user_{d_idx % 5}",
                        type=MemoryType.EPISODIC,
                        content=utterance,
                        importance=0.75,
                        entities=[],
                        task_id=f"dialogue_{d_idx}",
                        conversation_turn=turn_idx,
                    )
                    all_memories[day].append(memory)

                    # Next turn is query
                    if turn_idx + 1 < len(turns):
                        next_turn = turns[turn_idx + 1]
                        if isinstance(next_turn, dict):
                            next_utterance = next_turn.get("utterance", "")
                        else:
                            next_utterance = str(next_turn)

                        if next_utterance:
                            memory_ids = [f"turn_{d_idx}_{i}" for i in range(turn_idx + 1)]
                            expected = GoldExpectedResult(memory_ids=memory_ids)

                            query = GoldQuery(
                                day=day,
                                query=next_utterance,
                                task_id=f"dialogue_{d_idx}",
                                user_id=f"user_{d_idx % 5}",
                                expected=expected,
                            )
                            all_queries.append(query)

            except Exception:
                continue

        if not all_memories:
            raise ValidationError("No dialogue turns found")
        if not all_queries:
            raise ValidationError("No queries found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="MultiWOZ",
            description="MultiWOZ Task-Oriented Dialogue",
            user_ids=[f"user_{i}" for i in range(5)],
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"MultiWOZ validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "MultiWOZ",
                "turn_count": len(dataset.queries),
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute MultiWOZ fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute MultiWOZ statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "MultiWOZ",
            "version": "2.1",
            "description": "MultiWOZ - Task-oriented dialogue state tracking",
            "source": "Cambridge University",
            "format": "JSON with dialogue turns and state annotations",
            "typical_size": "10k dialogues",
            "focus": "Task state memory, dialogue history management",
        }
