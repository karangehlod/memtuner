"""Adapter for CoQA (Conversational Question Answering) dataset.

CoQA is a large-scale conversational QA dataset where the context is a passage
of text and the task is to answer a series of questions that refer to each other
conversationally. This adapter transforms CoQA into the standardized GoldDataset format.

Dataset characteristics:
  - 12k questions across 8 domains
  - Conversational context: questions refer to previous Q&A
  - Coreference resolution required
  - Multi-turn dialogue structure
"""

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
    GoldMemoryEvent,
    GoldQuery,
    GoldExpectedResult,
)
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class CoQAAdapter(DatasetAdapter):
    """Adapter for CoQA conversational QA dataset."""

    name = "coqa"

    def load(self, source: Path | str) -> GoldDataset:
        """Load CoQA dataset.

        Args:
            source: Path to CoQA JSON file.

        Returns:
            Normalized GoldDataset.

        Raises:
            AdapterError: If file cannot be loaded or parsed.
        """
        try:
            source_path = Path(source)
            with open(source_path) as f:
                data = json.load(f)
        except (FileNotFoundError, IOError) as e:
            raise AdapterError(f"Cannot read CoQA file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in CoQA file: {e}")

        # CoQA format: {"data": [{"story": ..., "questions": [...], "answers": [...]}]}
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list):
            raise ValidationError("CoQA data must be a list of stories")

        if not data:
            raise ValidationError("CoQA dataset is empty")

        all_memories: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids = set()

        for story_idx, story_item in enumerate(data):
            try:
                # Story text becomes base memory
                story_text = story_item.get("story", "")
                if not story_text:
                    continue

                day = story_idx % 30

                # Create memory from story
                story_memory = GoldMemoryEvent(
                    id=f"story_{story_idx}",
                    user_id="user-default",
                    type=MemoryType.EPISODIC,
                    content=story_text,
                    importance=0.8,
                    entities=[],
                    task_id=f"story_{story_idx}",
                    conversation_turn=0,
                )

                if day not in all_memories:
                    all_memories[day] = []
                all_memories[day].append(story_memory)

                # Process Q&A turns
                questions = story_item.get("questions", [])
                answers = story_item.get("answers", [])

                for turn_idx, question_raw in enumerate(questions):
                    # CoQA questions are dicts {"input_text": "...", "turn_id": N}
                    # in v1.0 format, or plain strings in older formats.
                    if isinstance(question_raw, dict):
                        question_text = question_raw.get("input_text", "")
                    else:
                        question_text = str(question_raw) if question_raw else ""
                    if not question_text:
                        continue

                    # Previous Q&A creates context/memory
                    context_memories = [story_memory.id]

                    expected = GoldExpectedResult(memory_ids=context_memories)

                    query = GoldQuery(
                        day=day,
                        query=question_text,
                        task_id=f"story_{story_idx}",
                        user_id="user-default",
                        expected=expected,
                    )
                    all_queries.append(query)
                    user_ids.add("user-default")

            except Exception as e:
                raise ValidationError(f"Error parsing CoQA story {story_idx}: {e}")

        if not all_memories:
            raise ValidationError("No stories found in CoQA dataset")
        if not all_queries:
            raise ValidationError("No questions found in CoQA dataset")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="CoQA",
            description="Conversational Question Answering Dataset",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate CoQA dataset."""
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"CoQA validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint."""
        try:
            query_ids = sorted(q.query for q in dataset.queries)
            memory_count = sum(len(d.memory_events) for d in dataset.events)

            fp_data = {
                "scenario": "CoQA",
                "query_count": len(dataset.queries),
                "memory_count": memory_count,
                "sample_queries": query_ids[:5],
            }

            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute CoQA fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute dataset statistics."""
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute CoQA statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return CoQA metadata."""
        return {
            "name": "CoQA",
            "version": "1.0",
            "description": "Conversational Question Answering - Multi-turn dialogue with coreference",
            "source": "Stanford NLP",
            "format": "JSON with stories and questions",
            "typical_size": "12k questions",
            "domains": ["news", "stories", "medical", "sci-fi"],
        }
