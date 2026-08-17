"""LoCoMo dataset loader.

Loads LoCoMo (Long-term Conversational Memory) dataset directly into our
GoldDataset schema. This is the PRIMARY dataset for the benchmark.

LoCoMo contains 10 very long-term conversations with:
  - 300+ turns per conversation
  - 9K tokens avg per conversation
  - Up to 35 sessions per conversation
  - QA annotations with evidence links
  - Temporal reasoning across sessions
  - Multiple speakers (conversational memory)

Dataset fields per sample:
  - sample_id
  - conversation: dict of sessions with timestamps and turns
  - observation: session-level observations
  - session_summary: per-session summaries
  - event_summary: per-speaker events across sessions
  - qa: question-answer annotations with category and evidence

QA categories test different memory abilities:
  - single-session: information from one session
  - multi-session: reasoning across sessions
  - temporal: time-based reasoning
  - knowledge-update: handling changed information
  - preference: stable user preferences
  - adversarial: contradictory/misleading information

Reference: https://github.com/snap-research/locomo
Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldEvaluationCriteria,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
    TemporalWindow,
)
from benchmark.models.memory_event import MemoryType

# Map LoCoMo QA categories to our memory types
CATEGORY_TO_MEMORY_TYPE: dict[str, MemoryType] = {
    "single-session": MemoryType.EPISODIC,
    "multi-session": MemoryType.SEMANTIC,
    "temporal": MemoryType.EPISODIC,
    "knowledge-update": MemoryType.SEMANTIC,
    "preference": MemoryType.PREFERENCE,
    "adversarial": MemoryType.SEMANTIC,
    # Fallbacks
    "event-summarization": MemoryType.EPISODIC,
    "multimodal": MemoryType.EPISODIC,
}

# Map QA categories to acceptable memory modules
CATEGORY_TO_MODULES: dict[str, list[str]] = {
    "single-session": ["episodic_store", "entity_store"],
    "multi-session": ["semantic_store", "episodic_store", "entity_store"],
    "temporal": ["episodic_store"],
    "knowledge-update": ["semantic_store", "episodic_store"],
    "preference": ["preference_store"],
    "adversarial": ["semantic_store", "episodic_store"],
}

# Difficulty classification
CATEGORY_DIFFICULTY: dict[str, str] = {
    "single-session": "easy",
    "multi-session": "hard",
    "temporal": "hard",
    "knowledge-update": "extreme",
    "preference": "medium",
    "adversarial": "extreme",
}


def _generate_memory_id(sample_id: str, session_key: str, turn_idx: int) -> str:
    """Generate a deterministic memory ID from conversation structure."""
    raw = f"{sample_id}:{session_key}:turn-{turn_idx}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"LCM-{sample_id}-{session_key}-T{turn_idx:03d}-{short_hash}"


def _parse_session_datetime(dt_str: str) -> datetime | None:
    """Parse LoCoMo session datetime strings.

    LoCoMo uses the format: "1:56 pm on 8 May, 2023"
    Also supports fallback formats:
    - "2023-01-15 14:30:00"
    - "January 15, 2023"
    - "2023/01/15"
    """
    if not dt_str:
        return None

    dt_str = dt_str.strip()

    # Primary LoCoMo format: "1:56 pm on 8 May, 2023" or "8:56 pm on 20 July, 2023"
    locomo_pattern = re.compile(
        r"(\d{1,2}:\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
        re.IGNORECASE,
    )
    match = locomo_pattern.match(dt_str)
    if match:
        time_str = match.group(1)
        ampm = match.group(2)
        day = match.group(3)
        month = match.group(4)
        year = match.group(5)
        try:
            return datetime.strptime(
                f"{day} {month} {year} {time_str} {ampm}",
                "%d %B %Y %I:%M %p",
            )
        except (ValueError, TypeError):
            pass

    # Fallback formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _datetime_to_day(dt: datetime | None, reference: datetime) -> int:
    """Convert a datetime to a simulated day relative to reference."""
    if dt is None:
        return 0
    delta = dt - reference
    return max(0, delta.days)


def _extract_entities_from_text(text: str) -> list[str]:
    """Extract simple named entities from conversational text."""
    words = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)
    return list(set(words))[:10]


def _compute_importance(text: str, is_evidence: bool) -> float:
    """Compute importance score for a conversation turn."""
    if is_evidence:
        return 0.9
    length_factor = min(1.0, len(text) / 300)
    return 0.3 + 0.3 * length_factor


class LoCoMoLoader:
    """Loads LoCoMo JSON data directly into GoldDataset format.

    This loader handles the full loading pipeline:
    1. Parses LoCoMo's session-based conversation format
    2. Converts dialog turns into memory events organized by day
    3. Maps QA annotations to evaluation queries with evidence-based expected results
    4. Assigns temporal windows based on session timestamps
    5. Preserves QA category metadata for stratified evaluation

    LoCoMo is the primary dataset for this benchmark because it tests:
    - Conversational memory persistence (300+ turns)
    - Temporal reasoning across long dialogues (35 sessions)
    - Preference memory stability
    - Multi-hop reasoning across sessions
    - Knowledge update handling
    """

    def __init__(self, max_sessions_per_sample: int | None = None) -> None:
        """Initialize the loader.

        Args:
            max_sessions_per_sample: Optional limit on sessions per conversation.
        """
        self._max_sessions = max_sessions_per_sample

    def convert_raw(
        self,
        raw_data: Any,
        scenario_name: str = "locomo",
    ) -> GoldDataset:
        """Convert pre-loaded LoCoMo JSON data to GoldDataset.

        This is the primary entry point used by GoldOracle when it
        detects LoCoMo format. No file I/O needed.

        Args:
            raw_data: Already-parsed JSON (list or dict with 'data' key).
            scenario_name: Name for the resulting scenario.

        Returns:
            A fully validated GoldDataset.
        """
        samples = self._normalize_raw(raw_data)
        return self._convert_to_gold_dataset(samples, scenario_name)

    def _normalize_raw(self, raw_data: Any) -> list[dict[str, Any]]:
        """Normalize raw LoCoMo data into a list of samples."""
        if isinstance(raw_data, dict):
            if "data" in raw_data:
                raw_data = raw_data["data"]
            else:
                raw_data = [raw_data]

        if not isinstance(raw_data, list):
            raise ValueError(f"LoCoMo dataset must be a JSON array, got: {type(raw_data).__name__}")

        return raw_data

    def load_and_convert(
        self,
        dataset_path: Path,
        scenario_name: str = "locomo",
        subset_size: int | None = None,
    ) -> GoldDataset:
        """Load a LoCoMo JSON file and convert to GoldDataset.

        Args:
            dataset_path: Path to locomo10.json or similar.
            scenario_name: Name for the resulting scenario.
            subset_size: If set, only convert the first N conversations.

        Returns:
            A fully validated GoldDataset.

        Raises:
            ValueError: If the dataset cannot be loaded or converted.
        """
        raw_data = self._load_raw(dataset_path)

        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        return self._convert_to_gold_dataset(raw_data, scenario_name)

    def _load_raw(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load raw LoCoMo JSON data."""
        if not dataset_path.exists():
            raise ValueError(f"LoCoMo dataset not found: {dataset_path}")

        with dataset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        return self._normalize_raw(data)

    def _convert_to_gold_dataset(
        self, samples: list[dict[str, Any]], scenario_name: str
    ) -> GoldDataset:
        """Convert LoCoMo samples to GoldDataset format."""
        # Find the earliest date across all conversations for day-0 reference
        reference_date = self._find_reference_date(samples)

        all_day_events: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        all_user_ids: set[str] = set()

        for sample in samples:
            sample_id = str(sample.get("sample_id", "unknown"))

            # Extract speakers — but store all memories under a single
            # conversation-scoped user_id so evidence from EITHER speaker is
            # accessible when the QA query runs.
            # Previously each speaker got their own user_id, which caused
            # ~50% of evidence turns (from the non-querying speaker) to be
            # silently filtered out by the user-isolation boundary, halving
            # the recall ceiling.
            conversation = sample.get("conversation", {})
            speaker_a = conversation.get("speaker_a", f"speaker_a_{sample_id}")
            speaker_b = conversation.get("speaker_b", f"speaker_b_{sample_id}")
            conv_user_id = f"conv-{sample_id}"  # shared for all turns + queries
            all_user_ids.add(conv_user_id)

            # Convert sessions to memory events and get evidence mapping
            evidence_map = self._convert_sessions(
                sample,
                sample_id,
                conv_user_id,
                conv_user_id,
                speaker_a,
                speaker_b,
                reference_date,
                all_day_events,
            )

            # Convert QA annotations to queries
            queries = self._convert_qa_annotations(
                sample, sample_id, conv_user_id, reference_date, evidence_map
            )
            all_queries.extend(queries)

        # Build GoldDayEvents list sorted by day
        day_events_list = [
            GoldDayEvents(day=day, memory_events=events)
            for day, events in sorted(all_day_events.items())
            if events
        ]

        if not day_events_list:
            raise ValueError("No memory events could be extracted from the dataset")

        if not all_queries:
            raise ValueError("No queries could be extracted from the dataset")

        total_turns = sum(len(de.memory_events) for de in day_events_list)

        return GoldDataset(
            schema_version="2.0",
            scenario=scenario_name,
            description=(
                f"LoCoMo benchmark dataset with {len(all_queries)} queries "
                f"across {len(day_events_list)} simulated days from "
                f"{len(samples)} long-term conversations. "
                f"Tests: conversational memory, temporal reasoning, "
                f"knowledge updates, preference persistence, multi-hop retrieval."
            ),
            user_ids=sorted(all_user_ids),
            total_conversation_turns=total_turns,
            events=day_events_list,
            queries=all_queries,
            evaluation_criteria=GoldEvaluationCriteria(
                recall_k=int(os.environ.get("BENCHMARK_RECALL_K", "10")),
                temporal_tolerance_days=3,
            ),
        )

    def _find_reference_date(self, samples: list[dict[str, Any]]) -> datetime:
        """Find the earliest date in the dataset to use as day-0."""
        all_dates: list[datetime] = []

        for sample in samples:
            conversation = sample.get("conversation", {})
            for key, value in conversation.items():
                if key.endswith("_date_time") and isinstance(value, str):
                    parsed = _parse_session_datetime(value)
                    if parsed:
                        all_dates.append(parsed)

        if not all_dates:
            return datetime(2023, 1, 1)

        return min(all_dates)

    def _convert_sessions(
        self,
        sample: dict[str, Any],
        sample_id: str,
        user_id_a: str,
        user_id_b: str,
        speaker_a: str,
        speaker_b: str,
        reference_date: datetime,
        all_day_events: dict[int, list[GoldMemoryEvent]],
    ) -> dict[str, str]:
        """Convert conversation sessions into memory events.

        Returns a mapping of dia_id -> memory_id for evidence linking.
        """
        conversation = sample.get("conversation", {})
        qa_evidence_ids = self._collect_evidence_ids(sample)

        # Find all session keys (session_1, session_2, ...)
        session_keys = sorted(
            [
                k
                for k in conversation.keys()
                if k.startswith("session_") and not k.endswith("_date_time")
            ],
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
        )

        if self._max_sessions and len(session_keys) > self._max_sessions:
            session_keys = session_keys[: self._max_sessions]

        evidence_map: dict[str, str] = {}

        for session_key in session_keys:
            session_turns = conversation.get(session_key, [])
            if not isinstance(session_turns, list):
                continue

            # Get session timestamp
            datetime_key = f"{session_key}_date_time"
            session_dt_str = conversation.get(datetime_key, "")
            session_dt = _parse_session_datetime(session_dt_str)
            day = _datetime_to_day(session_dt, reference_date)

            if day not in all_day_events:
                all_day_events[day] = []

            for turn_idx, turn in enumerate(session_turns):
                if not isinstance(turn, dict):
                    continue

                text = turn.get("text", "")
                if not text or len(text.strip()) < 5:
                    continue

                speaker = turn.get("speaker", "")
                dia_id = turn.get("dia_id", f"{session_key}_turn_{turn_idx}")

                # Determine user and memory type
                if speaker == speaker_a:
                    user_id = user_id_a
                else:
                    user_id = user_id_b

                # Check if this turn is evidence for any QA
                is_evidence = str(dia_id) in qa_evidence_ids

                # Determine memory type from content
                memory_type = self._infer_memory_type(text)

                memory_id = _generate_memory_id(sample_id, session_key, turn_idx)
                evidence_map[str(dia_id)] = memory_id

                entities = _extract_entities_from_text(text)
                importance = _compute_importance(text, is_evidence)

                memory_event = GoldMemoryEvent(
                    id=memory_id,
                    user_id=user_id,
                    type=memory_type,
                    content=text,
                    importance=importance,
                    entities=entities,
                    task_id=f"locomo_{sample_id}",
                    conversation_turn=turn_idx,
                )

                all_day_events[day].append(memory_event)

        return evidence_map

    def _collect_evidence_ids(self, sample: dict[str, Any]) -> set[str]:
        """Collect all dialog IDs that are evidence for QA annotations."""
        qa_annotations = sample.get("qa", [])
        evidence_ids: set[str] = set()
        for qa in qa_annotations:
            evidence = qa.get("evidence", [])
            if isinstance(evidence, list):
                for eid in evidence:
                    evidence_ids.add(str(eid))
        return evidence_ids

    def _infer_memory_type(self, text: str) -> MemoryType:
        """Infer memory type from conversation content."""
        text_lower = text.lower()

        # Preference indicators
        preference_words = [
            "i like",
            "i love",
            "i prefer",
            "i enjoy",
            "favorite",
            "i hate",
            "i dislike",
            "always choose",
            "usually pick",
        ]
        if any(pw in text_lower for pw in preference_words):
            return MemoryType.PREFERENCE

        # Entity/factual indicators
        entity_words = [
            "my name is",
            "i live in",
            "i work at",
            "i moved to",
            "my job is",
            "i am a",
            "i'm from",
        ]
        if any(ew in text_lower for ew in entity_words):
            return MemoryType.ENTITY

        # Default to episodic (most conversational memory is episodic)
        return MemoryType.EPISODIC

    def _convert_qa_annotations(
        self,
        sample: dict[str, Any],
        sample_id: str,
        default_user_id: str,
        reference_date: datetime,
        evidence_map: dict[str, str],
    ) -> list[GoldQuery]:
        """Convert QA annotations into GoldQuery objects."""
        qa_annotations = sample.get("qa", [])
        queries: list[GoldQuery] = []

        for qa_idx, qa in enumerate(qa_annotations):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            if not question:
                continue

            category = qa.get("category", "single-session")
            evidence_dia_ids = qa.get("evidence", [])

            # Map evidence dialog IDs to memory IDs
            expected_memory_ids: list[str] = []
            for eid in evidence_dia_ids:
                mid = evidence_map.get(str(eid))
                if mid:
                    expected_memory_ids.append(mid)

            if not expected_memory_ids:
                continue

            # Determine query day (after all evidence is injected)
            # Use the last day in the dataset + 1 as query day
            conversation = sample.get("conversation", {})
            max_day = 0
            for key, value in conversation.items():
                if key.endswith("_date_time") and isinstance(value, str):
                    parsed = _parse_session_datetime(value)
                    if parsed:
                        day = _datetime_to_day(parsed, reference_date)
                        max_day = max(max_day, day)

            query_day = max_day + 1

            # Determine temporal window from evidence sessions
            temporal_window = self._get_temporal_window_for_evidence(
                sample, evidence_dia_ids, reference_date
            )

            # Get acceptable modules
            acceptable_modules = CATEGORY_TO_MODULES.get(category, [])

            # Determine if multi-session
            is_followup = category in ("multi-session", "temporal")

            queries.append(
                GoldQuery(
                    day=query_day,
                    query=question,
                    task_id=f"locomo_{sample_id}_q{qa_idx:03d}",
                    user_id=default_user_id,
                    expected=GoldExpectedResult(
                        memory_ids=expected_memory_ids,
                        acceptable_modules=acceptable_modules,
                        temporal_window=temporal_window,
                    ),
                    gold_answer=str(answer) or None,
                    is_followup=is_followup,
                    references_turn=None,
                )
            )

        return queries

    def _get_temporal_window_for_evidence(
        self,
        sample: dict[str, Any],
        evidence_dia_ids: list[Any],
        reference_date: datetime,
    ) -> TemporalWindow | None:
        """Determine temporal window from evidence dialog IDs."""
        conversation = sample.get("conversation", {})
        evidence_days: list[int] = []

        # Map dia_ids to their session's timestamp
        for session_key in conversation:
            if session_key.startswith("session_") and not session_key.endswith("_date_time"):
                session_turns = conversation.get(session_key, [])
                if not isinstance(session_turns, list):
                    continue

                session_has_evidence = False
                for turn in session_turns:
                    if isinstance(turn, dict) and str(turn.get("dia_id", "")) in map(
                        str, evidence_dia_ids
                    ):
                        session_has_evidence = True
                        break

                if session_has_evidence:
                    dt_key = f"{session_key}_date_time"
                    dt_str = conversation.get(dt_key, "")
                    parsed = _parse_session_datetime(dt_str)
                    if parsed:
                        evidence_days.append(_datetime_to_day(parsed, reference_date))

        if not evidence_days:
            return None

        return TemporalWindow(
            not_before_day=min(evidence_days),
            not_after_day=max(evidence_days),
        )

    def get_difficulty_distribution(self, samples: list[dict[str, Any]]) -> dict[str, int]:
        """Get the distribution of difficulty levels in the dataset."""
        distribution: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "extreme": 0}
        for sample in samples:
            qa_annotations = sample.get("qa", [])
            for qa in qa_annotations:
                category = qa.get("category", "single-session")
                difficulty = CATEGORY_DIFFICULTY.get(category, "medium")
                distribution[difficulty] += 1
        return distribution

    def get_category_distribution(self, samples: list[dict[str, Any]]) -> dict[str, int]:
        """Get the distribution of QA categories."""
        distribution: dict[str, int] = {}
        for sample in samples:
            qa_annotations = sample.get("qa", [])
            for qa in qa_annotations:
                category = qa.get("category", "unknown")
                distribution[category] = distribution.get(category, 0) + 1
        return distribution


def convert_locomo_to_gold(
    input_path: Path,
    output_path: Path,
    scenario_name: str = "locomo",
    subset_size: int | None = None,
) -> GoldDataset:
    """Convenience function to convert and save a LoCoMo dataset.

    Args:
        input_path: Path to the LoCoMo JSON file (locomo10.json).
        output_path: Path to write the converted gold dataset.
        scenario_name: Name for the scenario.
        subset_size: Optional limit on number of conversations.

    Returns:
        The converted GoldDataset.
    """
    loader = LoCoMoLoader()
    dataset = loader.load_and_convert(input_path, scenario_name, subset_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset.model_dump(mode="json"), fh, indent=2, default=str)

    return dataset


def create_test_subset(
    input_path: Path,
    output_path: Path,
    max_conversations: int = 3,
    max_sessions: int = 10,
) -> GoldDataset:
    """Create a small test subset for fast validation.

    Args:
        input_path: Path to the full LoCoMo JSON file.
        output_path: Path to write the test subset.
        max_conversations: Max conversations to include.
        max_sessions: Max sessions per conversation.

    Returns:
        The small GoldDataset for testing.
    """
    loader = LoCoMoLoader(max_sessions_per_sample=max_sessions)
    dataset = loader.load_and_convert(
        input_path,
        scenario_name="locomo_test",
        subset_size=max_conversations,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset.model_dump(mode="json"), fh, indent=2, default=str)

    return dataset


# Backward-compatible alias
LoCoMoAdapter = LoCoMoLoader
