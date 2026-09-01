"""Adapter for Natural Questions dataset - real user queries."""

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


class NaturalQuestionsAdapter(DatasetAdapter):
    """Adapter for Natural Questions open-domain QA dataset."""

    name = "naturalquestions"

    def load(self, source: Path | str) -> GoldDataset:
        """Load Natural Questions dataset."""
        try:
            source_path = Path(source)
            with open(source_path) as f:
                data = json.load(f) if str(source_path).endswith('.json') else [json.loads(line) for line in f]
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read Natural Questions file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in Natural Questions file: {e}")

        if not isinstance(data, list):
            raise ValidationError("Natural Questions data must be a list of questions")

        if not data:
            raise ValidationError("Natural Questions dataset is empty")

        all_memories: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids = set()

        for q_idx, item in enumerate(data):
            try:
                day = q_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                question = item.get("question", "") if isinstance(item.get("question"), str) else item.get("question_text", "")
                if not question:
                    continue

                # Context passages as memories
                context = item.get("long_answer_candidates", []) or item.get("document_text", "")
                if isinstance(context, str):
                    memory = GoldMemoryEvent(
                        id=f"doc_{q_idx}",
                        user_id="user-default",
                        type=MemoryType.EPISODIC,
                        content=context[:500],
                        importance=0.7,
                        entities=[],
                        task_id=f"question_{q_idx}",
                        conversation_turn=0,
                    )
                    all_memories[day].append(memory)
                elif isinstance(context, list) and context:
                    for ctx_idx, ctx in enumerate(context[:3]):
                        ctx_text = ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)
                        memory = GoldMemoryEvent(
                            id=f"doc_{q_idx}_{ctx_idx}",
                            user_id="user-default",
                            type=MemoryType.EPISODIC,
                            content=ctx_text[:500],
                            importance=0.7,
                            entities=[],
                            task_id=f"question_{q_idx}",
                            conversation_turn=ctx_idx,
                        )
                        all_memories[day].append(memory)

                memory_ids = [m.id for m in all_memories[day]] if all_memories[day] else ["doc_0"]
                expected = GoldExpectedResult(memory_ids=memory_ids)

                query = GoldQuery(
                    day=day,
                    query=question,
                    task_id=f"question_{q_idx}",
                    user_id="user-default",
                    expected=expected,
                )
                all_queries.append(query)
                user_ids.add("user-default")

            except Exception as e:
                raise ValidationError(f"Error parsing Natural Questions item {q_idx}: {e}")

        if not all_memories:
            raise ValidationError("No documents found in Natural Questions dataset")
        if not all_queries:
            raise ValidationError("No questions found in Natural Questions dataset")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="Natural Questions",
            description="Natural Questions Open-Domain QA",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate dataset."""
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"Natural Questions validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint."""
        try:
            query_count = len(dataset.queries)
            memory_count = sum(len(d.memory_events) for d in dataset.events)
            fp_data = {"scenario": "NaturalQuestions", "query_count": query_count, "memory_count": memory_count}
            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute Natural Questions fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute statistics."""
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute Natural Questions statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return metadata."""
        return {
            "name": "Natural Questions",
            "version": "1.0",
            "description": "Natural Questions - Real user search queries with document context",
            "source": "Google Research",
            "format": "JSON with questions and candidate passages",
            "typical_size": "320k questions",
            "focus": "Real-world query distribution, open-domain retrieval",
        }
