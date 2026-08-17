"""LongMemEval Benchmark Pack Adapter.

Converts the LongMemEval dataset (ICLR 2025) into the benchmark's GoldDataset schema.

LongMemEval tests 5 core long-term memory abilities:
- Information Extraction (single-session-user, single-session-assistant, single-session-preference)
- Multi-Session Reasoning (multi-session)
- Knowledge Updates (knowledge-update)
- Temporal Reasoning (temporal-reasoning)
- Abstention (questions with no answer in history)

Source: https://github.com/xiaowu0162/LongMemEval
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
    TemporalWindow,
)
from benchmark.models.memory_event import MemoryType
from benchmark.packs.base import BenchmarkPack, PackMetadata
from benchmark.packs.registry import register_pack

if TYPE_CHECKING:
    from pathlib import Path

# Note: Question types are preserved in metadata for analysis, but DO NOT restrict
# which memory modules can answer questions. Each system competes on all questions.

_QUESTION_TYPE_TO_MEMORY_TYPE: dict[str, str] = {
    "single-session-user": "episodic",
    "single-session-assistant": "episodic",
    "single-session-preference": "preference",
    "multi-session": "semantic",
    "knowledge-update": "episodic",
    "temporal-reasoning": "episodic",
}


@register_pack("longmemeval")
class LongMemEvalPack(BenchmarkPack):
    """LongMemEval benchmark pack.

    500 high-quality questions testing long-term memory abilities.
    Each question comes with timestamped chat history sessions.
    """

    def __init__(self):
        self._data: list[dict[str, Any]] = []
        self._loaded = False

    def metadata(self) -> PackMetadata:
        return PackMetadata(
            name="longmemeval",
            version="1.0-cleaned",
            description=(
                "LongMemEval: Benchmarking Chat Assistants on Long-Term "
                "Interactive Memory (ICLR 2025)"
            ),
            source_url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
            license="MIT",
            citation=(
                "@article{wu2024longmemeval, title={LongMemEval}, "
                "author={Di Wu et al.}, year={2024}}"
            ),
            total_queries=500,
            total_sessions=0,  # varies per question
            memory_abilities=[
                "information_extraction",
                "multi_session_reasoning",
                "knowledge_updates",
                "temporal_reasoning",
                "abstention",
            ],
        )

    def required_files(self) -> list[str]:
        return ["longmemeval_oracle.json"]

    def _resolve_oracle_path(self, data_dir: Path) -> Path:
        direct_path = data_dir / "longmemeval_oracle.json"
        nested_path = data_dir / "longmemeval" / "longmemeval_oracle.json"

        if direct_path.exists():
            return direct_path
        if nested_path.exists():
            return nested_path

        raise FileNotFoundError(
            f"LongMemEval oracle file not found at {direct_path} or {nested_path}. "
            f"Download it: {self.download_instructions()}"
        )

    def load(self, data_dir: Path) -> None:
        """Load LongMemEval oracle dataset.

        Args:
            data_dir: Directory containing longmemeval_oracle.json
        """
        oracle_path = self._resolve_oracle_path(data_dir)

        with open(oracle_path) as f:
            self._data = json.load(f)

        self._loaded = True

    def to_gold_dataset(
        self,
        *,
        max_queries: int | None = None,
        seed: int = 42,
        evaluation_horizon: int | None = None,
    ) -> GoldDataset:
        """Convert LongMemEval to GoldDataset.

        Maps:
        - Chat history sessions → MemoryEvents (one per turn with has_answer or all turns)
        - Questions → GoldQuery with expected memory_ids
        - Answer sessions → GoldExpectedResult

        Args:
            max_queries: Limit number of questions (default: all 500).
            seed: Random seed for reproducibility.
            evaluation_horizon: Number of dataset days to spread sessions across.
        """
        if not self._loaded:
            raise RuntimeError("Pack not loaded. Call load() first.")

        # Filter out abstention questions (no gold answer location)
        questions = [q for q in self._data if not q["question_id"].endswith("_abs")]

        if max_queries is not None:
            questions = questions[:max_queries]

        # Determine evaluation horizon
        if evaluation_horizon is None:
            evaluation_horizon = max(len(q.get("haystack_sessions", [])) for q in questions)
            evaluation_horizon = min(evaluation_horizon, 100)

        # Generate user IDs (one per question for isolation)
        user_ids = [f"lme-user-{i}" for i in range(len(questions))]

        day_to_events: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []

        for q_idx, question in enumerate(questions):
            user_id = user_ids[q_idx]
            sessions = question.get("haystack_sessions", [])
            session_ids = question.get("haystack_session_ids", [])
            answer_session_ids = set(question.get("answer_session_ids", []))
            question_type = question.get("question_type", "single-session-user")

            # Spread sessions across dataset days
            days_per_session = max(1, evaluation_horizon // max(len(sessions), 1))

            expected_memory_ids: list[str] = []

            for sess_idx, session in enumerate(sessions):
                session_id = (
                    session_ids[sess_idx] if sess_idx < len(session_ids) else f"sess_{sess_idx}"
                )
                day = min(sess_idx * days_per_session, evaluation_horizon - 2)
                is_evidence = session_id in answer_session_ids

                # Convert each turn to a memory event
                for turn_idx, turn in enumerate(session):
                    content = turn.get("content", "")
                    role = turn.get("role", "user")
                    has_answer = turn.get("has_answer", False)

                    memory_id = _generate_memory_id(question["question_id"], sess_idx, turn_idx)
                    memory_type = _QUESTION_TYPE_TO_MEMORY_TYPE.get(question_type, "episodic")

                    event = GoldMemoryEvent(
                        id=memory_id,
                        user_id=user_id,
                        type=MemoryType(memory_type),
                        content=f"[{role}] {content}",
                        importance=0.8 if has_answer else 0.5,
                        entities=[],
                        task_id=question["question_id"],
                        conversation_turn=turn_idx,
                    )

                    # Track expected memories (turns with answer evidence)
                    if has_answer or is_evidence:
                        expected_memory_ids.append(memory_id)

                    # Group events by day
                    day_to_events.setdefault(day, []).append(event)

            # UNRESTRICTED BENCHMARK: All questions answerable by any memory module.
            # For true scientific rigor, don't artificially restrict which systems can
            # answer which questions. Each memory type competes fairly.
            acceptable_modules = [
                "episodic_store",
                "episodic_buffer",
                "semantic_store",
                "preference_store",
                "entity_store",
            ]

            # Query arrives after all sessions are injected
            query_day = min((len(sessions)) * days_per_session, evaluation_horizon - 1)

            query = GoldQuery(
                day=query_day,
                query=question["question"],
                task_id=question["question_id"],
                user_id=user_id,
                expected=GoldExpectedResult(
                    memory_ids=expected_memory_ids[:10],  # Cap at 10 for evaluation
                    acceptable_modules=acceptable_modules,
                    temporal_window=TemporalWindow(
                        not_before_day=0,
                        not_after_day=query_day,
                    ),
                ),
                gold_answer=str(question.get("answer", "")) or None,
            )
            all_queries.append(query)

        # Build GoldDayEvents only for days with events
        all_day_events = [
            GoldDayEvents(day=day, memory_events=events)
            for day, events in sorted(day_to_events.items())
        ]

        total_memory_events = sum(len(day.memory_events) for day in all_day_events)

        return GoldDataset(
            schema_version="1.0",
            scenario="longmemeval-oracle",
            description="LongMemEval: 500 questions testing long-term memory abilities (ICLR 2025)",
            user_ids=user_ids,
            total_conversation_turns=total_memory_events,
            events=all_day_events,
            queries=all_queries,
        )

    def download_instructions(self) -> str:
        return """
To download LongMemEval dataset:

  mkdir -p data/input/longmemeval
  curl -sL -o data/input/longmemeval/longmemeval_oracle.json \\
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"

Optional (large file ~100MB, full chat histories):
  curl -sL -o longmemeval_s_cleaned.json \\
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

Source: https://github.com/xiaowu0162/LongMemEval
License: MIT (ICLR 2025)
"""


def _generate_memory_id(question_id: str, session_idx: int, turn_idx: int) -> str:
    """Generate deterministic memory ID from question, session, and turn."""
    raw = f"{question_id}:s{session_idx}:t{turn_idx}"
    return f"lme-{hashlib.md5(raw.encode()).hexdigest()[:12]}"
