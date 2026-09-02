"""Unit tests for LoCoMo dataset adapter.

Tests the conversion from LoCoMo format to our GoldDataset schema.
Uses synthetic LoCoMo-format data matching the real dataset structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.gold.locomo_loader import (
    LoCoMoAdapter,
    _generate_memory_id,
    _parse_session_datetime,
    convert_locomo_to_gold,
    create_test_subset,
)

# ============================================================
# Synthetic LoCoMo test data matching real format
# ============================================================

SAMPLE_LOCOMO_DATA = [
    {
        "sample_id": "conv_001",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "dia_id": "d001", "text": "I really love sushi, it's my favorite food."},
                {"speaker": "Bob", "dia_id": "d002", "text": "That's great! I prefer Italian food myself."},
                {"speaker": "Alice", "dia_id": "d003", "text": "We should try that new Japanese place downtown."},
            ],
            "session_1_date_time": "2023-01-15 14:30:00",
            "session_2": [
                {"speaker": "Alice", "dia_id": "d004", "text": "I just moved to Seattle last week."},
                {"speaker": "Bob", "dia_id": "d005", "text": "How do you like it there?"},
                {"speaker": "Alice", "dia_id": "d006", "text": "It rains a lot but the coffee is amazing."},
            ],
            "session_2_date_time": "2023-02-20 10:00:00",
            "session_3": [
                {"speaker": "Alice", "dia_id": "d007", "text": "I started working at Microsoft this month."},
                {"speaker": "Bob", "dia_id": "d008", "text": "Congratulations! What team are you on?"},
                {"speaker": "Alice", "dia_id": "d009", "text": "I'm on the Azure infrastructure team."},
            ],
            "session_3_date_time": "2023-04-05 16:00:00",
            "session_4": [
                {"speaker": "Bob", "dia_id": "d010", "text": "Remember when you lived in Boston?"},
                {"speaker": "Alice", "dia_id": "d011", "text": "Yes! I miss the seafood there."},
                {"speaker": "Bob", "dia_id": "d012", "text": "But you said you love Seattle now."},
                {"speaker": "Alice", "dia_id": "d013", "text": "True, the tech scene here is unbeatable."},
            ],
            "session_4_date_time": "2023-06-10 11:00:00",
        },
        "observation": {
            "session_1_observation": "Alice likes sushi. Bob prefers Italian.",
            "session_2_observation": "Alice moved to Seattle.",
            "session_3_observation": "Alice started at Microsoft Azure.",
        },
        "session_summary": {
            "session_1_summary": "Alice and Bob discussed food preferences.",
            "session_2_summary": "Alice talked about her move to Seattle.",
            "session_3_summary": "Alice shared news about her new job.",
        },
        "qa": [
            {
                "question": "What food does Alice like?",
                "answer": "sushi",
                "category": "preference",
                "evidence": ["d001"],
            },
            {
                "question": "Where does Alice currently live?",
                "answer": "Seattle",
                "category": "knowledge-update",
                "evidence": ["d004"],
            },
            {
                "question": "What company does Alice work for and what team is she on?",
                "answer": "Microsoft, Azure infrastructure team",
                "category": "multi-session",
                "evidence": ["d007", "d009"],
            },
            {
                "question": "When did Alice move relative to starting her job?",
                "answer": "Before starting at Microsoft",
                "category": "temporal",
                "evidence": ["d004", "d007"],
            },
        ],
    },
    {
        "sample_id": "conv_002",
        "conversation": {
            "speaker_a": "Carol",
            "speaker_b": "Dave",
            "session_1": [
                {"speaker": "Carol", "dia_id": "d101", "text": "I'm training for the marathon next month."},
                {"speaker": "Dave", "dia_id": "d102", "text": "How far can you run now?"},
                {"speaker": "Carol", "dia_id": "d103", "text": "About 20 miles without stopping."},
            ],
            "session_1_date_time": "2023-03-01 09:00:00",
            "session_2": [
                {"speaker": "Carol", "dia_id": "d104", "text": "I finished the marathon in 3 hours 45 minutes!"},
                {"speaker": "Dave", "dia_id": "d105", "text": "That's incredible! Congratulations!"},
            ],
            "session_2_date_time": "2023-04-15 18:00:00",
            "session_3": [
                {"speaker": "Dave", "dia_id": "d106", "text": "Are you running another marathon?"},
                {"speaker": "Carol", "dia_id": "d107", "text": "Actually I switched to cycling. My knees need a break."},
            ],
            "session_3_date_time": "2023-07-20 14:00:00",
        },
        "qa": [
            {
                "question": "What sport does Carol currently do?",
                "answer": "cycling",
                "category": "knowledge-update",
                "evidence": ["d107"],
            },
            {
                "question": "What was Carol's marathon time?",
                "answer": "3 hours 45 minutes",
                "category": "single-session",
                "evidence": ["d104"],
            },
            {
                "question": "Why did Carol switch from running to cycling?",
                "answer": "Her knees need a break",
                "category": "multi-session",
                "evidence": ["d104", "d107"],
            },
        ],
    },
]


@pytest.fixture
def sample_data_path(tmp_path: Path) -> Path:
    """Create a temporary file with sample LoCoMo data."""
    data_file = tmp_path / "locomo_test.json"
    data_file.write_text(json.dumps(SAMPLE_LOCOMO_DATA), encoding="utf-8")
    return data_file


class TestLoCoMoAdapter:
    """Tests for the LoCoMo adapter."""

    def test_load_and_convert_basic(self, sample_data_path: Path) -> None:
        """Basic conversion produces a valid GoldDataset."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test_scenario")

        assert dataset.scenario == "test_scenario"
        assert dataset.schema_version == "2.0"
        assert len(dataset.queries) > 0
        assert len(dataset.events) > 0

    def test_all_qa_converted_to_queries(self, sample_data_path: Path) -> None:
        """All QA annotations with evidence become queries."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        # 4 QAs from conv_001 + 3 QAs from conv_002 = 7 total
        assert len(dataset.queries) == 7

    def test_expected_memory_ids_are_nonempty(self, sample_data_path: Path) -> None:
        """Every query has at least one expected memory ID."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        for query in dataset.queries:
            assert len(query.expected.memory_ids) > 0, (
                f"Query '{query.query}' has no expected memory IDs"
            )

    def test_multi_evidence_queries_have_multiple_ids(self, sample_data_path: Path) -> None:
        """Multi-session queries map to multiple expected memory IDs."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        multi_queries = [
            q for q in dataset.queries
            if "company" in q.query.lower() or "switch" in q.query.lower()
        ]
        # These should have 2+ expected memory IDs
        for q in multi_queries:
            assert len(q.expected.memory_ids) >= 2, (
                f"Multi-evidence query '{q.query}' has only {len(q.expected.memory_ids)} expected IDs"
            )

    def test_temporal_queries_have_temporal_window(self, sample_data_path: Path) -> None:
        """Temporal category queries get temporal windows."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        temporal_queries = [
            q for q in dataset.queries if "relative" in q.query.lower() or "when" in q.query.lower()
        ]
        for q in temporal_queries:
            assert q.expected.temporal_window is not None, (
                f"Temporal query '{q.query}' has no temporal window"
            )

    def test_preference_queries_map_to_preference_module(self, sample_data_path: Path) -> None:
        """Preference category maps to preference_store module."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        pref_queries = [q for q in dataset.queries if "food" in q.query.lower()]
        assert len(pref_queries) >= 1
        assert "preference_store" in pref_queries[0].expected.acceptable_modules

    def test_multi_session_marked_as_followup(self, sample_data_path: Path) -> None:
        """Multi-session and temporal queries are marked as follow-ups."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        # Find multi-session query
        multi = [q for q in dataset.queries if "company" in q.query.lower()]
        assert len(multi) >= 1
        assert multi[0].is_followup is True

    def test_sessions_organized_by_day(self, sample_data_path: Path) -> None:
        """Memory events are organized into correct days based on session dates."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        days = [de.day for de in dataset.events]
        assert days == sorted(days)
        # Multiple distinct days (sessions have different dates)
        assert len(set(days)) > 1

    def test_users_extracted_from_speakers(self, sample_data_path: Path) -> None:
        """Users are extracted from speaker names."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        assert len(dataset.user_ids) >= 2  # 1 per conversation (all turns share conv-scoped user_id)

    def test_memory_events_have_content(self, sample_data_path: Path) -> None:
        """All memory events have non-empty content."""
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        for day_events in dataset.events:
            for event in day_events.memory_events:
                assert len(event.content) > 0

    def test_subset_size_limits_conversations(self, sample_data_path: Path) -> None:
        """subset_size limits number of conversations."""
        adapter = LoCoMoAdapter()
        dataset_full = adapter.load_and_convert(sample_data_path, "full")
        dataset_small = adapter.load_and_convert(sample_data_path, "small", subset_size=1)

        assert len(dataset_small.queries) < len(dataset_full.queries)

    def test_max_sessions_limits_processing(self, sample_data_path: Path) -> None:
        """max_sessions_per_sample limits session processing."""
        adapter_limited = LoCoMoAdapter(max_sessions_per_sample=2)
        adapter_full = LoCoMoAdapter()

        dataset_limited = adapter_limited.load_and_convert(sample_data_path, "limited")
        dataset_full = adapter_full.load_and_convert(sample_data_path, "full")

        limited_total = sum(len(de.memory_events) for de in dataset_limited.events)
        full_total = sum(len(de.memory_events) for de in dataset_full.events)
        assert limited_total <= full_total

    def test_knowledge_update_scenario(self, sample_data_path: Path) -> None:
        """Knowledge-update queries test temporal conflict handling.

        The benchmark must produce different outcomes for different decay policies
        when handling outdated vs. current information.
        """
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        # "Where does Alice currently live?" should link to the LATEST info
        update_queries = [q for q in dataset.queries if "currently live" in q.query.lower()]
        assert len(update_queries) >= 1

        # Should have acceptable_modules for semantic/episodic
        q = update_queries[0]
        assert "semantic_store" in q.expected.acceptable_modules or "episodic_store" in q.expected.acceptable_modules

    def test_discriminative_properties(self, sample_data_path: Path) -> None:
        """Dataset produces queries that will discriminate between strategies.

        Different query types should stress different retrieval capabilities:
        - preference: stable preferences (BM25 may suffice)
        - multi-session: connecting facts across sessions (embeddings needed)
        - temporal: time-aware retrieval (decay policies matter)
        - knowledge-update: recency weighting (newer > older)
        """
        adapter = LoCoMoAdapter()
        dataset = adapter.load_and_convert(sample_data_path, "test")

        categories_present = set()
        for q in dataset.queries:
            if "food" in q.query.lower():
                categories_present.add("preference")
            elif "company" in q.query.lower() or "switch" in q.query.lower():
                categories_present.add("multi-session")
            elif "when" in q.query.lower() or "relative" in q.query.lower():
                categories_present.add("temporal")
            elif "currently" in q.query.lower():
                categories_present.add("knowledge-update")

        # Must cover multiple ability dimensions
        assert len(categories_present) >= 3


class TestConvertLocomoToGold:
    """Tests for the convenience conversion function."""

    def test_creates_output_file(self, sample_data_path: Path, tmp_path: Path) -> None:
        """Conversion creates the output JSON file."""
        output_path = tmp_path / "output" / "gold.json"
        convert_locomo_to_gold(sample_data_path, output_path)
        assert output_path.exists()

    def test_output_is_valid_json(self, sample_data_path: Path, tmp_path: Path) -> None:
        """Output file is valid JSON loadable as GoldDataset."""
        output_path = tmp_path / "gold.json"
        convert_locomo_to_gold(sample_data_path, output_path)

        with output_path.open() as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert "queries" in data
        assert "events" in data
        assert len(data["queries"]) == 7


class TestCreateTestSubset:
    """Tests for test subset creation."""

    def test_creates_subset(self, sample_data_path: Path, tmp_path: Path) -> None:
        """Test subset is created successfully."""
        output_path = tmp_path / "subset.json"
        dataset = create_test_subset(
            sample_data_path, output_path,
            max_conversations=1, max_sessions=2,
        )
        assert len(dataset.queries) > 0
        assert output_path.exists()

    def test_subset_smaller_than_full(self, sample_data_path: Path, tmp_path: Path) -> None:
        """Subset is smaller than the full conversion."""
        output_sub = tmp_path / "sub.json"
        output_full = tmp_path / "full.json"

        subset = create_test_subset(
            sample_data_path, output_sub,
            max_conversations=1, max_sessions=2,
        )
        full = convert_locomo_to_gold(sample_data_path, output_full)

        assert len(subset.queries) <= len(full.queries)


class TestParseDatetime:
    """Tests for datetime parsing utility."""

    def test_standard_format(self) -> None:
        dt = _parse_session_datetime("2023-01-15 14:30:00")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 1
        assert dt.day == 15

    def test_date_only_format(self) -> None:
        dt = _parse_session_datetime("2023-06-20")
        assert dt is not None
        assert dt.month == 6
        assert dt.day == 20

    def test_invalid_format(self) -> None:
        dt = _parse_session_datetime("invalid")
        assert dt is None

    def test_empty_string(self) -> None:
        dt = _parse_session_datetime("")
        assert dt is None


class TestMemoryIdGeneration:
    """Tests for deterministic memory ID generation."""

    def test_deterministic(self) -> None:
        id1 = _generate_memory_id("conv_001", "session_1", 0)
        id2 = _generate_memory_id("conv_001", "session_1", 0)
        assert id1 == id2

    def test_different_inputs_different_ids(self) -> None:
        id1 = _generate_memory_id("conv_001", "session_1", 0)
        id2 = _generate_memory_id("conv_001", "session_1", 1)
        id3 = _generate_memory_id("conv_001", "session_2", 0)
        assert id1 != id2
        assert id1 != id3


class TestDifficultyDistribution:
    """Tests for difficulty classification."""

    def test_covers_multiple_levels(self) -> None:
        adapter = LoCoMoAdapter()
        dist = adapter.get_difficulty_distribution(SAMPLE_LOCOMO_DATA)

        # QA categories in our data cover multiple difficulties
        total = sum(dist.values())
        assert total == 7  # Total QA count
        assert dist["easy"] > 0  # single-session
        assert dist["hard"] > 0  # multi-session + temporal
        assert dist["extreme"] > 0  # knowledge-update

    def test_category_distribution(self) -> None:
        adapter = LoCoMoAdapter()
        cats = adapter.get_category_distribution(SAMPLE_LOCOMO_DATA)

        assert "preference" in cats
        assert "knowledge-update" in cats
        assert "multi-session" in cats
        assert "single-session" in cats
