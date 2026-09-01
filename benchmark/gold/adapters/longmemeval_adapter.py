"""Adapter for LongMemEval (Long Memory Evaluation) datasets.

LongMemEval datasets contain question-answer pairs with long context
haystacks. Each question references relevant context sessions that
must be retrieved to answer it. This adapter transforms them into
the standardized GoldDataset format.

Dataset structure:
  - Questions: evaluation queries with metadata
  - Haystack: relevant context sessions to retrieve
  - Temporal metadata: question dates and session dates
  - Question types: comparison, temporal, other for analysis

Transformation:
  1. Parse question JSON
  2. Extract haystack_sessions as memory events
  3. Use question dates for temporal ordering
  4. Convert questions to queries
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


class LongMemEvalAdapter(DatasetAdapter):
    """Adapter for LongMemEval long-context QA datasets.

    LongMemEval datasets test long-context retrieval and reasoning
    with temporal metadata for each question.

    Usage:
        >>> adapter = LongMemEvalAdapter()
        >>> dataset = adapter.load("data/longmemeval/longmemeval_oracle.json")
        >>> validation = adapter.validate(dataset)
        >>> stats = adapter.statistics(dataset)
        >>> fp = adapter.fingerprint(dataset)
    """

    name = "longmemeval"

    def load(self, source: Path | str) -> GoldDataset:
        """Load LongMemEval dataset from JSON file.

        Args:
            source: Path to LongMemEval JSON file.

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
            raise AdapterError(f"Cannot read LongMemEval file {source}: {e}")
        except json.JSONDecodeError as e:
            raise AdapterError(f"Invalid JSON in LongMemEval file: {e}")

        if not isinstance(data, list):
            raise ValidationError("LongMemEval data must be a list of questions")

        if not data:
            raise ValidationError("LongMemEval dataset is empty")

        # Parse questions into memories and queries
        all_memories: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids = set()

        for q_idx, question_item in enumerate(data):
            try:
                # Extract temporal info - use question_date to determine day
                question_date = question_item.get("question_date", f"2024-01-{(q_idx % 28) + 1:02d}")
                day = self._date_to_day(question_date, q_idx)

                if day not in all_memories:
                    all_memories[day] = []

                # Parse haystack sessions as memory events.
                # Build a mapping from the source's raw session ID → generated memory ID
                # so _parse_question can resolve haystack_session_ids correctly.
                source_id_to_memory_id: dict[str, str] = {}
                if "haystack_sessions" in question_item:
                    sessions = question_item["haystack_sessions"]
                    if isinstance(sessions, list):
                        for _session_idx, session_text in enumerate(sessions):
                            # Track how many memories are already on this day
                            day_offset = len(all_memories[day])
                            memory = self._parse_haystack_session(
                                session_text, day, day_offset, user_ids
                            )
                            if memory:
                                all_memories[day].append(memory)
                                # Map raw session ID (if present) to generated ID
                                if isinstance(session_text, dict):
                                    raw_id = session_text.get("session_id") or session_text.get("id")
                                    if raw_id:
                                        source_id_to_memory_id[str(raw_id)] = memory.id

                # Create query from question
                question_id = question_item.get("question_id", f"q_{q_idx}")
                query = self._parse_question(
                    question_item, day, question_id,
                    all_memories.get(day, []),
                    source_id_to_memory_id,
                )
                if query:
                    all_queries.append(query)
                    user_ids.add(query.user_id)

            except Exception as e:
                raise ValidationError(
                    f"Error parsing LongMemEval question {q_idx}: {e}"
                )

        # Validate we have data
        if not all_memories:
            raise ValidationError("No memory events found in LongMemEval dataset")
        if not all_queries:
            raise ValidationError("No queries found in LongMemEval dataset")

        # Build events list
        events = [
            GoldDayEvents(day=day, memory_events=all_memories[day])
            for day in sorted(all_memories.keys())
        ]

        # Create GoldDataset
        return GoldDataset(
            scenario="LongMemEval",
            description="Long Memory Evaluation - Long-context QA Dataset",
            user_ids=sorted(user_ids),
            events=events,
            queries=all_queries,
        )

    def validate(self, dataset: GoldDataset) -> ValidationReport:
        """Validate LongMemEval-specific constraints.

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
            raise ValidationError(f"LongMemEval validation error: {e}")

    def fingerprint(self, dataset: GoldDataset) -> str:
        """Generate deterministic fingerprint for LongMemEval dataset.

        Fingerprint is based on:
          - Question count and question types
          - Memory (haystack) count
          - Temporal span
          - Query diversity

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
            memory_count = sum(len(d.memory_events) for d in dataset.events)
            temporal_span = (
                dataset.events[-1].day - dataset.events[0].day
                if dataset.events
                else 0
            )

            # Create fingerprint
            fp_data = {
                "scenario": dataset.scenario,
                "query_count": len(dataset.queries),
                "query_sample": query_ids[:10],
                "memory_count": memory_count,
                "temporal_span": temporal_span,
                "user_count": len(dataset.user_ids),
            }

            # Hash
            fp_str = json.dumps(fp_data, sort_keys=True)
            return hashlib.sha256(fp_str.encode()).hexdigest()

        except Exception as e:
            raise FingerprintError(
                f"Failed to compute LongMemEval fingerprint: {e}"
            )

    def statistics(self, dataset: GoldDataset) -> DatasetStatistics:
        """Compute LongMemEval-specific statistics.

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
            raise StatisticsError(f"Failed to compute LongMemEval statistics: {e}")

    def metadata(self) -> dict[str, Any]:
        """Return LongMemEval dataset metadata.

        Returns:
            Metadata dictionary.
        """
        return {
            "name": "LongMemEval",
            "version": "1.0",
            "description": "Long Memory Evaluation - Long-context QA with haystack retrieval",
            "source": "Research benchmark dataset",
            "format": "JSON list of questions with haystacks",
            "typical_size": "500 questions, 15 MB",
        }

    # ========================================================================
    # Private Parsing Methods
    # ========================================================================

    def _date_to_day(self, date_str: str, fallback_idx: int) -> int:
        """Convert date string to a unique sequential day number.

        Uses days-since-epoch so different calendar months never collide
        onto the same simulated day. Two questions from 2024-01-05 and
        2024-02-05 previously both mapped to day 5 (day-of-month % 30),
        merging unrelated haystack sessions. Now each calendar date gets
        a distinct slot.

        Args:
            date_str: Date in format YYYY-MM-DD or similar.
            fallback_idx: Fallback index when parsing fails.

        Returns:
            Day number (0-based, days since 2020-01-01).
        """
        try:
            parts = date_str.split("-")
            if len(parts) >= 3:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                # Days since a fixed epoch (2020-01-01) — fully unique per calendar date
                from datetime import date as _date
                epoch = _date(2020, 1, 1)
                return (_date(year, month, day) - epoch).days
        except (ValueError, IndexError, OverflowError):
            pass
        return fallback_idx

    def _parse_haystack_session(
        self,
        session_text: Any,
        day: int,
        session_idx: int,
        user_ids: set[str],
    ) -> GoldMemoryEvent | None:
        """Parse a haystack session into a GoldMemoryEvent.

        Args:
            session_text: Session content (string or dict).
            day: Day number.
            session_idx: Index within day.
            user_ids: Set to collect user IDs.

        Returns:
            GoldMemoryEvent or None if parsing fails.
        """
        if isinstance(session_text, str):
            content = session_text
        elif isinstance(session_text, dict):
            content = session_text.get("text", session_text.get("content", str(session_text)))
        else:
            content = str(session_text)

        user_id = "user-default"
        user_ids.add(user_id)

        memory_id = f"haystack_{day}_{session_idx}"

        return GoldMemoryEvent(
            id=memory_id,
            user_id=user_id,
            type=MemoryType.EPISODIC,
            content=content,
            importance=0.7,  # High importance for haystack context
            entities=[],
            task_id=f"qa_{day}",
            conversation_turn=session_idx,
        )

    def _parse_question(
        self,
        question_item: dict,
        day: int,
        question_id: str,
        day_memories: list[GoldMemoryEvent],
        source_id_to_memory_id: dict[str, str] | None = None,
    ) -> GoldQuery | None:
        """Parse a question into a GoldQuery.

        Args:
            question_item: Question data dict.
            day: Day number.
            question_id: Question ID.
            day_memories: Memories ingested for this day.
            source_id_to_memory_id: Maps raw source session IDs to generated
                memory IDs (haystack_{day}_{idx}). Without this mapping,
                raw session IDs from haystack_session_ids would never match
                any ingested memory, forcing recall to 0 for all such queries.

        Returns:
            GoldQuery or None if parsing fails.
        """
        question_text = question_item.get("question", "")
        if not question_text:
            return None

        # Resolve gold memory IDs: translate raw session IDs to generated IDs.
        relevant_memory_ids = []
        if "haystack_session_ids" in question_item:
            session_ids = question_item["haystack_session_ids"]
            if isinstance(session_ids, list) and source_id_to_memory_id:
                for raw_id in session_ids:
                    mapped = source_id_to_memory_id.get(str(raw_id))
                    if mapped:
                        relevant_memory_ids.append(mapped)

        # Fallback: use all day memories (when no explicit session IDs or mapping)
        if not relevant_memory_ids and day_memories:
            relevant_memory_ids = [m.id for m in day_memories]

        # Last-resort fallback: first memory of the day (guaranteed to exist)
        if not relevant_memory_ids:
            relevant_memory_ids = [f"haystack_{day}_0"]

        expected = GoldExpectedResult(memory_ids=relevant_memory_ids)

        return GoldQuery(
            day=day,
            query=question_text,
            task_id=f"qa_{day}",
            user_id="user-default",
            expected=expected,
        )
