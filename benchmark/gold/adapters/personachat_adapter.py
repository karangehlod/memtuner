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
    """Adapter for PersonaChat persona-grounded dialogue dataset.

    Task design: each dialogue turn is a query whose relevant memories are the
    persona sentences of its source dialogue (PersonaChat does not annotate
    which single persona grounds a turn, so all-personas-of-the-dialogue is the
    standard proxy label). Dialogues are pooled round-robin into POOL_USERS
    shared users, so each query must rank its dialogue's ~4-5 personas against
    the ~110 personas of ~25 other dialogues sharing the user store. Without
    this pooling (one user per dialogue) the candidate pool equals the gold set
    and every strategy scores Recall@10 = 1.0 — a saturated, meaningless
    benchmark.

    Remaining caveat: persona sentences repeated verbatim across dialogues act
    as unlabeled distractors, so a perfect Recall@10 is not attainable; scores
    are comparable between configs, not absolute.
    """

    name = "personachat"

    # Dialogues per-user pooling factor. With the 500-dialogue benchmark subset
    # this yields ~25 dialogues (~110 personas) per user — far above top_k, so
    # retrieval quality actually differentiates configurations.
    POOL_USERS = 20

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
                pool_user = f"user_{dialogue_idx % self.POOL_USERS}"

                if day not in all_memories:
                    all_memories[day] = []

                # Support both format variants:
                #   bavard/personachat_truecased: {"personality": [...], "utterances": [...]}
                #   older format:                 {"personas": [...], "history": [...]}
                personas = dialogue.get("personality", dialogue.get("personas", []))

                for persona_idx, persona_text in enumerate(personas):
                    memory = GoldMemoryEvent(
                        id=f"persona_{dialogue_idx}_{persona_idx}",
                        user_id=pool_user,
                        type=MemoryType.EPISODIC,
                        content=persona_text,
                        importance=0.9,
                        entities=[],
                        task_id=f"dialogue_{dialogue_idx}",
                        conversation_turn=persona_idx,
                    )
                    all_memories[day].append(memory)
                    user_ids.add(pool_user)

                relevant_memories = [
                    f"persona_{dialogue_idx}_{i}" for i in range(len(personas))
                ]

                # bavard format: utterances is a list of {history, candidates} dicts
                utterances = dialogue.get("utterances", [])
                if utterances:
                    for utt in utterances:
                        # history[-1] is the most recent user turn (query to retrieve persona)
                        utt_history = utt.get("history", [])
                        if utt_history:
                            query_text = utt_history[-1]
                            expected = GoldExpectedResult(memory_ids=relevant_memories) if relevant_memories else None
                            all_queries.append(GoldQuery(
                                day=day,
                                query=query_text,
                                task_id=f"dialogue_{dialogue_idx}",
                                user_id=pool_user,
                                expected=expected,
                            ))
                else:
                    # Older format: history is a flat list of [query, response] pairs
                    history = dialogue.get("history", [])
                    for turn in history:
                        if isinstance(turn, list) and len(turn) >= 2:
                            query_text = turn[0]
                            expected = GoldExpectedResult(memory_ids=relevant_memories) if relevant_memories else None
                            if expected is None:
                                continue
                            all_queries.append(GoldQuery(
                                day=day,
                                query=query_text,
                                task_id=f"dialogue_{dialogue_idx}",
                                user_id=pool_user,
                                expected=expected,
                            ))

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
