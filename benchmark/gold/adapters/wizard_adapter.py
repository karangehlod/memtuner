"""Adapter for Wizard of Wikipedia - knowledge dialogue."""

import hashlib, json
from pathlib import Path
from typing import Any

from benchmark.gold.adapters.adapter import AdapterError, DatasetAdapter, FingerprintError, StatisticsError, ValidationError, ValidationReport
from benchmark.gold.schema import GoldDataset, GoldDayEvents, GoldMemoryEvent, GoldQuery, GoldExpectedResult
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class WizardAdapter(DatasetAdapter):
    """Adapter for Wizard of Wikipedia knowledge-grounded dialogue."""

    name = "wizard"

    def load(self, source: Path | str) -> GoldDataset:
        """Load Wizard of Wikipedia dataset."""
        try:
            with open(source) as f:
                data = [json.loads(line) for line in f]
        except (FileNotFoundError, IOError) as e:
            raise AdapterError(f"Cannot read Wizard file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in Wizard file: {e}")

        if not data:
            raise ValidationError("Wizard dataset is empty")

        all_memories, all_queries = {}, []

        for d_idx, dialogue in enumerate(data):
            try:
                day = d_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                # Knowledge passages as memories
                retrieved_passages = dialogue.get("retrieved_passages", []) or dialogue.get("knowledge", [])
                for k_idx, passage in enumerate(retrieved_passages[:5]):
                    text = passage.get("passage", "") if isinstance(passage, dict) else str(passage)
                    if not text:
                        continue

                    memory = GoldMemoryEvent(
                        id=f"know_{d_idx}_{k_idx}",
                        user_id=f"user_{d_idx % 10}",
                        type=MemoryType.EPISODIC,
                        content=text[:500],
                        importance=0.85,
                        entities=[],
                        task_id=f"dialogue_{d_idx}",
                        conversation_turn=k_idx,
                    )
                    all_memories[day].append(memory)

                # Dialogue turns as queries
                dialogue_history = dialogue.get("history", dialogue.get("dialogue", []))
                for turn_idx, turn in enumerate(dialogue_history):
                    if isinstance(turn, str):
                        query_text = turn
                    elif isinstance(turn, dict):
                        query_text = turn.get("text", "")
                    else:
                        continue

                    if not query_text:
                        continue

                    memory_ids = [f"know_{d_idx}_{i}" for i in range(min(len(retrieved_passages), 5))]
                    if not memory_ids:
                        memory_ids = ["know_0_0"]

                    expected = GoldExpectedResult(memory_ids=memory_ids)

                    query = GoldQuery(
                        day=day,
                        query=query_text,
                        task_id=f"dialogue_{d_idx}",
                        user_id=f"user_{d_idx % 10}",
                        expected=expected,
                    )
                    all_queries.append(query)

            except Exception as e:
                raise ValidationError(f"Error parsing Wizard dialogue {d_idx}: {e}")

        if not all_memories:
            raise ValidationError("No knowledge passages found")
        if not all_queries:
            raise ValidationError("No dialogue turns found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="Wizard of Wikipedia",
            description="Knowledge-Grounded Dialogue Dataset",
            user_ids=[f"user_{i}" for i in range(10)],
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"Wizard validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "Wizard of Wikipedia",
                "query_count": len(dataset.queries),
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute Wizard fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute Wizard statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "Wizard of Wikipedia",
            "version": "1.0",
            "description": "Wizard of Wikipedia - Knowledge-grounded dialogue with retrieved passages",
            "source": "Facebook Research",
            "format": "JSON Lines with dialogues and knowledge",
            "typical_size": "18k dialogues",
            "focus": "Knowledge-grounded dialogue, grounding memory in facts",
        }
