"""Unit tests for LongMemEval benchmark pack adapter.

Tests the conversion from LongMemEval oracle format to the GoldDataset schema.
Uses synthetic data that mirrors the real dataset structure without requiring
the full download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.packs.longmemeval.adapter import LongMemEvalPack, _generate_memory_id

# ============================================================
# Synthetic LongMemEval test data
# ============================================================

SAMPLE_LONGMEMEVAL_DATA = [
    {
        "question_id": "q001",
        "question_type": "single-session-user",
        "question": "What food does Alice like?",
        "answer": "sushi",
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I really enjoy eating sushi.", "has_answer": True},
                {"role": "assistant", "content": "That's great! Sushi is healthy."},
            ],
            [
                {"role": "user", "content": "Let's discuss our project timeline."},
                {"role": "assistant", "content": "Sure, what milestones do you have?"},
            ],
            [
                {"role": "user", "content": "I went to a Japanese restaurant yesterday."},
                {"role": "assistant", "content": "How was it?"},
            ],
        ],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "q002",
        "question_type": "knowledge-update",
        "question": "Where does John currently live?",
        "answer": "Seattle",
        "haystack_session_ids": ["s4", "s5", "s6"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I live in Boston.", "has_answer": True},
                {"role": "assistant", "content": "Boston is lovely!"},
            ],
            [
                {"role": "user", "content": "I'm thinking of moving soon."},
                {"role": "assistant", "content": "Where are you considering?"},
            ],
            [
                {"role": "user", "content": "I just moved to Seattle.", "has_answer": True},
                {"role": "assistant", "content": "Welcome to Seattle!"},
            ],
        ],
        "answer_session_ids": ["s4", "s6"],
    },
    {
        "question_id": "q003",
        "question_type": "multi-session",
        "question": "Who handles Atlas database migrations?",
        "answer": "Priya",
        "haystack_session_ids": ["s9", "s10"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Project Atlas uses PostgreSQL.", "has_answer": True},
                {"role": "assistant", "content": "That's a solid choice."},
            ],
            [
                {"role": "user", "content": "Priya handles all migrations.", "has_answer": True},
                {"role": "assistant", "content": "She's very experienced."},
            ],
        ],
        "answer_session_ids": ["s9", "s10"],
    },
    # Abstention question (should be skipped)
    {
        "question_id": "q004_abs",
        "question_type": "single-session-user",
        "question": "What is the color of unicorns?",
        "answer": "N/A",
        "haystack_session_ids": ["s13"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Just chatting about random things."},
                {"role": "assistant", "content": "Sure!"},
            ],
        ],
        "answer_session_ids": [],
    },
]


@pytest.fixture()
def longmemeval_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a longmemeval_oracle.json file."""
    oracle_path = tmp_path / "longmemeval_oracle.json"
    oracle_path.write_text(json.dumps(SAMPLE_LONGMEMEVAL_DATA), encoding="utf-8")
    return tmp_path


@pytest.mark.unit
class TestLongMemEvalPack:
    """Tests for the LongMemEvalPack adapter."""

    def test_load_succeeds(self, longmemeval_data_dir: Path) -> None:
        """Loading from a valid directory succeeds."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading from an empty directory raises FileNotFoundError."""
        pack = LongMemEvalPack()
        with pytest.raises(FileNotFoundError):
            pack.load(tmp_path)

    def test_to_gold_dataset_basic(self, longmemeval_data_dir: Path) -> None:
        """Conversion produces a valid GoldDataset with queries and events."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        assert dataset.scenario == "longmemeval-oracle"
        assert len(dataset.queries) > 0
        assert len(dataset.events) > 0

    def test_abstention_questions_excluded(self, longmemeval_data_dir: Path) -> None:
        """Questions ending in _abs are excluded from queries."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        question_ids = [q.task_id for q in dataset.queries]
        assert not any("_abs" in qid for qid in question_ids)

    def test_expected_memory_ids_nonempty(self, longmemeval_data_dir: Path) -> None:
        """Every query has at least one expected memory ID."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        for query in dataset.queries:
            assert len(query.expected.memory_ids) > 0, (
                f"Query '{query.query}' has no expected memory IDs"
            )

    def test_memory_events_have_content(self, longmemeval_data_dir: Path) -> None:
        """All memory events have non-empty content."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        for day_events in dataset.events:
            for event in day_events.memory_events:
                assert len(event.content) > 0

    def test_days_sorted(self, longmemeval_data_dir: Path) -> None:
        """Event days are in sorted order."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        days = [de.day for de in dataset.events]
        assert days == sorted(days)

    def test_max_queries_limits_output(self, longmemeval_data_dir: Path) -> None:
        """max_queries parameter caps the number of queries."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        full = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        pack2 = LongMemEvalPack()
        pack2.load(longmemeval_data_dir)
        limited = pack2.to_gold_dataset(max_queries=1, seed=42, evaluation_horizon=30)

        assert len(limited.queries) <= len(full.queries)
        assert len(limited.queries) == 1

    def test_gold_answer_populated(self, longmemeval_data_dir: Path) -> None:
        """Queries have gold_answer from the dataset answer field."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        answers_found = [q.gold_answer for q in dataset.queries if q.gold_answer]
        assert len(answers_found) > 0

    def test_user_isolation(self, longmemeval_data_dir: Path) -> None:
        """Each question gets its own user_id for isolation."""
        pack = LongMemEvalPack()
        pack.load(longmemeval_data_dir)
        dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=30)

        user_ids = [q.user_id for q in dataset.queries]
        assert len(set(user_ids)) == len(user_ids)

    def test_metadata(self) -> None:
        """Pack metadata is populated correctly."""
        pack = LongMemEvalPack()
        meta = pack.metadata()
        assert meta.name == "longmemeval"
        assert "MIT" in meta.license

    def test_required_files(self) -> None:
        """Required files list includes the oracle JSON."""
        pack = LongMemEvalPack()
        assert "longmemeval_oracle.json" in pack.required_files()


@pytest.mark.unit
class TestGenerateMemoryId:
    """Tests for the deterministic memory ID generator."""

    def test_deterministic(self) -> None:
        """Same inputs produce the same ID."""
        id1 = _generate_memory_id("q001", 0, 0)
        id2 = _generate_memory_id("q001", 0, 0)
        assert id1 == id2

    def test_different_inputs_differ(self) -> None:
        """Different inputs produce different IDs."""
        id1 = _generate_memory_id("q001", 0, 0)
        id2 = _generate_memory_id("q001", 0, 1)
        id3 = _generate_memory_id("q001", 1, 0)
        assert id1 != id2
        assert id1 != id3

    def test_prefix(self) -> None:
        """Generated IDs have the lme- prefix."""
        mid = _generate_memory_id("q001", 0, 0)
        assert mid.startswith("lme-")
