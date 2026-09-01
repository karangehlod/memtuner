"""Adapter for SQuAD 2.0 dataset - reading comprehension."""

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


class SQuADAdapter(DatasetAdapter):
    """Adapter for SQuAD 2.0 reading comprehension dataset."""

    name = "squad"

    def load(self, source: Path | str) -> GoldDataset:
        """Load SQuAD 2.0 dataset."""
        try:
            source_path = Path(source)
            with open(source_path) as f:
                data = json.load(f)
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read SQuAD file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in SQuAD file: {e}")

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list):
            raise ValidationError("SQuAD data must be a list of articles")

        if not data:
            raise ValidationError("SQuAD dataset is empty")

        all_memories: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids = set()

        for article_idx, article in enumerate(data):
            try:
                day = article_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                # Paragraphs are memories
                paragraphs = article.get("paragraphs", [])
                for para_idx, paragraph in enumerate(paragraphs):
                    para_text = paragraph.get("context", "")

                    memory = GoldMemoryEvent(
                        id=f"para_{article_idx}_{para_idx}",
                        user_id="user-default",
                        type=MemoryType.EPISODIC,
                        content=para_text,
                        importance=0.8,
                        entities=[],
                        task_id=f"article_{article_idx}",
                        conversation_turn=para_idx,
                    )
                    all_memories[day].append(memory)

                    # Questions are queries
                    qas = paragraph.get("qas", [])
                    for qa in qas:
                        question = qa.get("question", "")
                        if not question:
                            continue

                        expected = GoldExpectedResult(
                            memory_ids=[f"para_{article_idx}_{para_idx}"]
                        )

                        query = GoldQuery(
                            day=day,
                            query=question,
                            task_id=f"article_{article_idx}",
                            user_id="user-default",
                            expected=expected,
                        )
                        all_queries.append(query)
                        user_ids.add("user-default")

            except Exception as e:
                raise ValidationError(f"Error parsing SQuAD article {article_idx}: {e}")

        if not all_memories:
            raise ValidationError("No paragraphs found in SQuAD dataset")
        if not all_queries:
            raise ValidationError("No questions found in SQuAD dataset")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="SQuAD 2.0",
            description="Reading Comprehension Dataset",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate SQuAD dataset."""
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"SQuAD validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint."""
        try:
            query_count = len(dataset.queries)
            memory_count = sum(len(d.memory_events) for d in dataset.events)
            fp_data = {"scenario": "SQuAD 2.0", "query_count": query_count, "memory_count": memory_count}
            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute SQuAD fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute dataset statistics."""
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute SQuAD statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return SQuAD metadata."""
        return {
            "name": "SQuAD 2.0",
            "version": "2.0",
            "description": "Reading Comprehension Dataset - Adversarial unanswerable questions",
            "source": "Stanford NLP",
            "format": "JSON with articles, paragraphs, and QA",
            "typical_size": "100k questions",
            "focus": "Adversarial evaluation, document memory",
        }
