"""Adapter for LoCoMo (Long Context Multi-turn Observation) datasets.

LoCoMo datasets contain multi-user, multi-turn conversation logs with
memory events and question-answer pairs. This adapter transforms them
into the standardized GoldDataset format.

Dataset structure:
  - Conversations: multi-turn exchanges between agents
  - Event summaries: memory events extracted from conversations
  - Q&A pairs: questions and answers from the conversation
  - Session data: user/session identifiers

Transformation:
  1. Parse conversation JSON
  2. Extract memory events from event_summary
  3. Convert Q&A pairs to queries
  4. Assign temporal structure (conversations → days)
  5. Normalize to GoldDataset schema
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
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
)
from benchmark.gold.statistics import DatasetStatistics, StatisticsComputer
from benchmark.gold.validators import ValidationRegistry
from benchmark.models.memory_event import MemoryType


class LoCoMoAdapter(DatasetAdapter):
    """Adapter for LoCoMo multi-turn conversation datasets.

    LoCoMo datasets contain realistic multi-turn conversations with
    extracted memory events and question-answer pairs for evaluation.

    Usage:
        >>> adapter = LoCoMoAdapter()
        >>> dataset = adapter.load("data/locomo10.json")
        >>> validation = adapter.validate(dataset)
        >>> stats = adapter.statistics(dataset)
        >>> fp = adapter.fingerprint(dataset)
    """

    name = "locomo"

    def load(self, source: Path | str) -> GoldDataset:
        """Load LoCoMo dataset from JSON file.

        Args:
            source: Path to LoCoMo JSON file.

        Returns:
            Normalized GoldDataset.

        Raises:
            AdapterError: If file cannot be loaded or parsed.
            ValidationError: If data structure is invalid.
        """
        try:
            source_path = Path(source)
            with open(source_path) as f:
                data = json.load(f)
        except (OSError, FileNotFoundError) as e:
            raise AdapterError(f"Cannot read LoCoMo file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in LoCoMo file: {e}")

        if not isinstance(data, list):
            raise ValidationError("LoCoMo data must be a list of conversations")

        if not data:
            raise ValidationError("LoCoMo dataset is empty")

        # Parse conversations into memories and queries
        all_memories: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids = set()

        for day, conversation_item in enumerate(data):
            day_memories: list[GoldMemoryEvent] = []

            try:
                # Extract memory events from event_summary
                if "event_summary" in conversation_item:
                    events = conversation_item["event_summary"]
                    if isinstance(events, list):
                        for event_idx, event in enumerate(events):
                            memory = self._parse_event(
                                event, day, event_idx, user_ids
                            )
                            if memory:
                                day_memories.append(memory)

                # Extract queries from Q&A pairs
                if "qa" in conversation_item:
                    qa_data = conversation_item["qa"]
                    if isinstance(qa_data, dict):
                        for qa_idx, (question, answer) in enumerate(qa_data.items()):
                            query = self._parse_qa(
                                question, answer, day, qa_idx, day_memories
                            )
                            if query:
                                all_queries.append(query)
                                user_ids.add(query.user_id)

                # Store day memories
                if day_memories:
                    all_memories[day] = day_memories

            except Exception as e:
                raise ValidationError(
                    f"Error parsing LoCoMo conversation {day}: {e}"
                )

        # Validate we have data
        if not all_memories:
            raise ValidationError("No memory events found in LoCoMo dataset")
        if not all_queries:
            raise ValidationError("No queries found in LoCoMo dataset")

        # Build events list
        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        # Create GoldDataset
        return GoldDataset(
            scenario="LoCoMo",
            description="Long Context Multi-turn Observation Dataset",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate LoCoMo-specific constraints.

        Uses the standard validation registry to check schema, temporal
        ordering, integrity, and statistics.

        Args:
            dataset: Dataset to validate.

        Returns:
            Validation report with all issues found.
        """
        try:
            return ValidationRegistry.validate_all(dataset)
        except Exception as e:
            raise ValidationError(f"LoCoMo validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint for LoCoMo dataset.

        Fingerprint is based on:
          - Query count and IDs
          - Memory count and IDs
          - User count
          - Importance distribution

        Args:
            dataset: Dataset to fingerprint.

        Returns:
            32-character hex string (SHA256).

        Raises:
            FingerprintError: If fingerprint computation fails.
        """
        try:
            # Collect deterministic data
            query_ids = sorted(q.query for q in dataset.queries)
            memory_ids = sorted(
                m.id for day in dataset.events for m in day.memory_events
            )
            user_count = len(dataset.user_ids)
            importance_vals = sorted(
                m.importance for day in dataset.events for m in day.memory_events
            )

            # Create fingerprint
            fp_data = {
                "scenario": dataset.scenario,
                "query_count": len(dataset.queries),
                "query_ids": query_ids[:10],  # First 10 to keep reasonable size
                "memory_count": sum(len(d.memory_events) for d in dataset.events),
                "memory_ids": memory_ids[:10],
                "user_count": user_count,
                "importance_summary": {
                    "mean": sum(importance_vals) / len(importance_vals)
                    if importance_vals
                    else 0,
                    "min": min(importance_vals) if importance_vals else 0,
                    "max": max(importance_vals) if importance_vals else 0,
                },
            }

            # Hash
            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()

        except Exception as e:
            raise FingerprintError(f"Failed to compute LoCoMo fingerprint: {e}")

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute LoCoMo-specific statistics.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Comprehensive statistics object.

        Raises:
            StatisticsError: If computation fails.
        """
        try:
            return StatisticsComputer.compute(dataset)
        except Exception as e:
            raise StatisticsError(f"Failed to compute LoCoMo statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return LoCoMo dataset metadata.

        Returns:
            Metadata dictionary.
        """
        return {
            "name": "LoCoMo",
            "version": "1.0",
            "description": "Long Context Multi-turn Observation Dataset - Multi-user, multi-turn conversations with memory events",
            "source": "Internal benchmark dataset",
            "format": "JSON list of conversations",
            "typical_size": "10 scenarios, 2.7 MB",
        }

    # ========================================================================
    # Private Parsing Methods
    # ========================================================================

    def _parse_event(
        self,
        event: Any,
        day: int,
        event_idx: int,
        user_ids: set[str],
    ) -> GoldMemoryEvent | None:
        """Parse an event from event_summary into a GoldMemoryEvent.

        Args:
            event: Event data (dict or string).
            day: Day number for this event.
            event_idx: Index within day.
            user_ids: Set to collect user IDs.

        Returns:
            GoldMemoryEvent or None if parsing fails.
        """
        if isinstance(event, str):
            content = event
        elif isinstance(event, dict):
            content = event.get("summary", event.get("content", str(event)))
        else:
            content = str(event)

        user_id = "user-default"
        entities = []

        # Try to extract entities if dict
        if isinstance(event, dict):
            if "entities" in event:
                entities = event["entities"]
                if isinstance(entities, str):
                    entities = [entities]
            if "user_id" in event:
                user_id = event["user_id"]

        user_ids.add(user_id)

        memory_id = f"mem_{day}_{event_idx}"

        return GoldMemoryEvent(
            id=memory_id,
            user_id=user_id,
            type=MemoryType.EPISODIC,
            content=content,
            importance=0.5,  # Default for LoCoMo
            entities=entities or [],
            task_id=f"task_{day}",
            conversation_turn=event_idx,
        )

    def _parse_qa(
        self,
        question: str,
        answer: Any,
        day: int,
        qa_idx: int,
        day_memories: list[GoldMemoryEvent],
    ) -> GoldQuery | None:
        """Parse a Q&A pair into a GoldQuery.

        Args:
            question: Question text.
            answer: Answer text or dict.
            day: Day number.
            qa_idx: Index within day.
            day_memories: Memories available for this day.

        Returns:
            GoldQuery or None if parsing fails.
        """
        if not question or not isinstance(question, str):
            return None

        # Extract memory IDs that should be relevant for this day's queries.
        relevant_memory_ids = [m.id for m in day_memories]
        if not relevant_memory_ids:
            # Skip queries with no available memories — returning a hardcoded
            # "mem_0_0" would point to a non-existent memory on days where
            # the first conversation had no events, producing a silent zero recall.
            return None

        expected = GoldExpectedResult(memory_ids=relevant_memory_ids)


        return GoldQuery(
            day=day,
            query=question,
            task_id=f"task_{day}",
            user_id="user-default",
            expected=expected,
        )
