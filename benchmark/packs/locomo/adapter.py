"""LoCoMo Benchmark Pack Adapter.

Converts the LoCoMo dataset (ACL 2024) into the benchmark's GoldDataset schema.

LoCoMo tests very long-term conversational memory with:
- 10 multi-session conversations between 2 speakers
- 199+ QA annotations per conversation
- Temporal spans across months/years
- Evidence-linked answers

Source: https://github.com/snap-research/locomo
License: See repository
"""

from __future__ import annotations

import hashlib
import json
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

# LoCoMo QA categories
_CATEGORY_NAMES = {
    1: "single-hop",
    2: "single-hop-temporal",
    3: "multi-hop",
    4: "open-ended",
    5: "adversarial",
}

# Map categories to memory types
_CATEGORY_TO_MEMORY_TYPE = {
    1: "episodic",
    2: "episodic",
    3: "semantic",
    4: "semantic",
    5: "entity",
}

_CATEGORY_TO_MODULES = {
    1: ["episodic_store"],
    2: ["episodic_store"],
    3: ["episodic_store", "semantic_store"],
    4: ["semantic_store", "preference_store"],
    5: ["entity_store", "episodic_store"],
}


@register_pack("locomo")
class LoCoMoPack(BenchmarkPack):
    """LoCoMo benchmark pack.

    10 very long-term conversations with 199+ QA annotations each.
    Tests multi-session conversational memory and event continuity.
    """

    def __init__(self):
        self._data: list[dict[str, Any]] = []
        self._loaded = False

    def metadata(self) -> PackMetadata:
        return PackMetadata(
            name="locomo",
            version="1.0",
            description=(
                "LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents (ACL 2024)"
            ),
            source_url="https://github.com/snap-research/locomo",
            license="See repository",
            citation=(
                "@article{maharana2024evaluating, title={LoCoMo}, "
                "author={Maharana et al.}, year={2024}}"
            ),
            total_queries=0,  # Set after loading
            total_sessions=10,
            memory_abilities=[
                "single_hop_recall",
                "temporal_recall",
                "multi_hop_reasoning",
                "open_ended_recall",
                "adversarial_robustness",
            ],
        )

    def required_files(self) -> list[str]:
        return ["locomo10.json"]

    def load(self, data_dir: Path) -> None:
        """Load LoCoMo dataset.

        Args:
            data_dir: Directory containing locomo10.json
        """
        data_path = data_dir / "locomo10.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"LoCoMo data file not found at {data_path}. "
                f"Download it: {self.download_instructions()}"
            )

        with open(data_path) as f:
            self._data = json.load(f)

        self._loaded = True

    def to_gold_dataset(
        self,
        *,
        max_queries: int | None = None,
        seed: int = 42,
        evaluation_horizon: int | None = None,
    ) -> GoldDataset:
        """Convert LoCoMo to GoldDataset.

        Maps:
        - Conversation sessions → MemoryEvents (one per dialog turn)
        - QA annotations → GoldQuery with evidence-linked memory_ids
        - Evidence dialog IDs → GoldExpectedResult

        Args:
            max_queries: Maximum queries to include across all conversations.
            seed: Random seed for reproducibility.
            evaluation_horizon: Number of dataset days (default: based on sessions).
        """
        if not self._loaded:
            raise RuntimeError("Pack not loaded. Call load() first.")

        day_to_events: dict[int, list[GoldMemoryEvent]] = {}
        all_queries: list[GoldQuery] = []
        user_ids: list[str] = []

        # Determine evaluation horizon from data
        if evaluation_horizon is None:
            max_sessions = max(_count_sessions(conv.get("conversation", {})) for conv in self._data)
            evaluation_horizon = min(max_sessions * 2, 100)

        total_queries_added = 0

        for conv_idx, conversation in enumerate(self._data):
            sample_id = conversation.get("sample_id", f"conv_{conv_idx}")
            conv_data = conversation.get("conversation", {})
            qa_list = conversation.get("qa", [])

            user_id = f"locomo-{sample_id}"
            user_ids.append(user_id)

            # Extract sessions in order
            sessions = _extract_sessions(conv_data)
            days_per_session = max(1, evaluation_horizon // max(len(sessions), 1))

            # Memory ID registry for this conversation
            dialog_id_to_memory_id: dict[str, str] = {}

            for sess_idx, (session_key, session_turns, _session_date) in enumerate(sessions):
                day = min(sess_idx * days_per_session, evaluation_horizon - 2)

                for turn_idx, turn in enumerate(session_turns):
                    speaker = turn.get("speaker", "unknown")
                    text = turn.get("text", "")
                    dia_id = turn.get("dia_id", f"{session_key}:{turn_idx}")

                    memory_id = _generate_memory_id(sample_id, session_key, turn_idx)
                    dialog_id_to_memory_id[dia_id] = memory_id

                    event = GoldMemoryEvent(
                        id=memory_id,
                        user_id=user_id,
                        type=MemoryType.EPISODIC,
                        content=f"[{speaker}] {text}",
                        importance=0.6,
                        entities=[speaker],
                        task_id=sample_id,
                        conversation_turn=turn_idx,
                    )
                    day_to_events.setdefault(day, []).append(event)

            # Convert QA annotations to queries
            for _qa_idx, qa in enumerate(qa_list):
                if max_queries is not None and total_queries_added >= max_queries:
                    break

                question = qa.get("question", "")
                answer = qa.get("answer", "")
                category = qa.get("category", 1)
                evidence = qa.get("evidence", [])

                # Map evidence dialog IDs to memory IDs
                expected_memory_ids = []
                for ev_id in evidence:
                    if ev_id in dialog_id_to_memory_id:
                        expected_memory_ids.append(dialog_id_to_memory_id[ev_id])

                if not expected_memory_ids:
                    continue  # Skip questions without traceable evidence

                query_day = evaluation_horizon - 1
                acceptable_modules = _CATEGORY_TO_MODULES.get(category, ["episodic_store"])

                query = GoldQuery(
                    day=query_day,
                    query=question,
                    task_id=sample_id,
                    user_id=user_id,
                    expected=GoldExpectedResult(
                        memory_ids=expected_memory_ids[:10],
                        acceptable_modules=acceptable_modules,
                        temporal_window=TemporalWindow(
                            not_before_day=0,
                            not_after_day=query_day,
                        ),
                    ),
                    gold_answer=str(answer) or None,
                )
                all_queries.append(query)
                total_queries_added += 1

        # Build GoldDayEvents only for days with events
        all_day_events = [
            GoldDayEvents(day=day, memory_events=events)
            for day, events in sorted(day_to_events.items())
        ]

        total_memory_events = sum(len(day.memory_events) for day in all_day_events)

        return GoldDataset(
            schema_version="1.0",
            scenario="locomo-10",
            description="LoCoMo: Very long-term conversational memory evaluation (ACL 2024)",
            user_ids=user_ids,
            total_conversation_turns=total_memory_events,
            events=all_day_events,
            queries=all_queries,
        )

    def download_instructions(self) -> str:
        return """
To download LoCoMo dataset:

  mkdir -p data/input
  curl -sL -o data/input/locomo10.json \\
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

Source: https://github.com/snap-research/locomo
Paper: ACL 2024 - "Evaluating Very Long-Term Conversational Memory of LLM Agents"
"""


def _count_sessions(conv_data: dict) -> int:
    """Count number of sessions in a LoCoMo conversation."""
    count = 0
    for key in conv_data:
        if key.startswith("session_") and not key.endswith("_date_time"):
            count += 1
    return count


def _extract_sessions(conv_data: dict) -> list[tuple[str, list[dict], str]]:
    """Extract sessions in chronological order.

    Returns list of (session_key, turns, date_time).
    """
    sessions = []
    idx = 1
    while True:
        session_key = f"session_{idx}"
        date_key = f"session_{idx}_date_time"
        if session_key not in conv_data:
            break
        turns = conv_data[session_key]
        date_time = conv_data.get(date_key, "")
        sessions.append((session_key, turns, date_time))
        idx += 1
    return sessions


def _generate_memory_id(sample_id: str, session_key: str, turn_idx: int) -> str:
    """Generate deterministic memory ID."""
    raw = f"{sample_id}:{session_key}:t{turn_idx}"
    return f"loco-{hashlib.md5(raw.encode()).hexdigest()[:12]}"
