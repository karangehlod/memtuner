"""Adapter for WebQuestions dataset - KB QA."""

import hashlib, json
from pathlib import Path
from typing import Any

from benchmark.gold.adapters.adapter import AdapterError, DatasetAdapter, FingerprintError, StatisticsError, ValidationError, ValidationReport
from benchmark.gold.schema import GoldDataset, GoldDayEvents, GoldMemoryEvent, GoldQuery, GoldExpectedResult
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class WebQuestionsAdapter(DatasetAdapter):
    """Adapter for WebQuestions KB-based QA dataset."""

    name = "webquestions"

    def load(self, source: Path | str) -> GoldDataset:
        """Load WebQuestions dataset."""
        try:
            with open(source) as f:
                data = [json.loads(line) for line in f]
        except (FileNotFoundError, IOError) as e:
            raise AdapterError(f"Cannot read WebQuestions file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in WebQuestions file: {e}")

        if not data:
            raise ValidationError("WebQuestions dataset is empty")

        all_memories, all_queries = {}, []
        user_ids = set()

        for q_idx, item in enumerate(data):
            try:
                day = q_idx % 30
                if day not in all_memories:
                    all_memories[day] = []

                question = item.get("utterance", item.get("question", ""))
                if not question:
                    continue

                # KB facts as memories
                answers = item.get("answers", [])
                for ans_idx, answer in enumerate(answers[:3]):
                    ans_text = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
                    memory = GoldMemoryEvent(
                        id=f"kb_{q_idx}_{ans_idx}",
                        user_id="user-default",
                        type=MemoryType.EPISODIC,
                        content=ans_text,
                        importance=0.8,
                        entities=[],
                        task_id=f"q_{q_idx}",
                        conversation_turn=ans_idx,
                    )
                    all_memories[day].append(memory)

                memory_ids = [f"kb_{q_idx}_{i}" for i in range(min(len(answers), 3))] or ["kb_0"]
                expected = GoldExpectedResult(memory_ids=memory_ids)

                query = GoldQuery(
                    day=day,
                    query=question,
                    task_id=f"q_{q_idx}",
                    user_id="user-default",
                    expected=expected,
                )
                all_queries.append(query)
                user_ids.add("user-default")

            except Exception as e:
                raise ValidationError(f"Error parsing WebQuestions item {q_idx}: {e}")

        if not all_memories:
            raise ValidationError("No KB facts found")
        if not all_queries:
            raise ValidationError("No questions found")

        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        return GoldDataset(
            scenario="WebQuestions",
            description="WebQuestions KB QA",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"WebQuestions validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        try:
            fp_data = {
                "scenario": "WebQuestions",
                "query_count": len(dataset.queries),
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
            }
            return hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            raise FingerprintError(f"Failed to compute WebQuestions fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute WebQuestions statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "WebQuestions",
            "version": "1.0",
            "description": "WebQuestions - KB facts for semantic parsing",
            "source": "Facebook Research",
            "format": "JSON Lines with questions and answers",
            "typical_size": "5k questions",
            "focus": "KB-based QA, knowledge memory",
        }
