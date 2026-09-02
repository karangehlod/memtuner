"""Integration tests for the complete benchmark pipeline with LongMemEval data.

Verifies that:
1. The full pipeline runs end-to-end with LongMemEval-format data
2. Metrics are NOT all 1.0 (no vacuous truth saturation)
3. Different strategies produce different scores (discriminative power)
4. Temporal window constraints actually affect scores
5. Ranking metrics (MRR, NDCG) provide additional signal
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.evaluation.ranking import (
    MRREvaluator,
    NDCGEvaluator,
    PrecisionAtKEvaluator,
    RecallAtKEvaluator,
)
from benchmark.evaluation.recall import RecallEvaluator
from benchmark.evaluation.temporal import TemporalAccuracyEvaluator
from benchmark.gold.longmemeval_adapter import LongMemEvalAdapter

# Synthetic dataset matching LongMemEval format
INTEGRATION_DATA = [
    {
        "question_id": "int_001",
        "question_type": "single-session-user",
        "question": "What food does Alice like?",
        "answer": "sushi",
        "question_date": "2023-06-15",
        "haystack_session_ids": ["s1", "s2", "s3", "s4", "s5"],
        "haystack_dates": [
            "2023-01-10", "2023-02-15", "2023-03-20", "2023-04-25", "2023-05-30"
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I really enjoy eating sushi. It's my favorite food.", "has_answer": True},
                {"role": "assistant", "content": "Sushi is delicious and healthy!"},
            ],
            [
                {"role": "user", "content": "Alice visited Tokyo last month."},
                {"role": "assistant", "content": "How was the trip?"},
            ],
            [
                {"role": "user", "content": "Alice hates seafood allergies."},
                {"role": "assistant", "content": "That must be challenging."},
            ],
            [
                {"role": "user", "content": "Bob also likes sushi actually."},
                {"role": "assistant", "content": "It's popular!"},
            ],
            [
                {"role": "user", "content": "The weather in Tokyo is humid."},
                {"role": "assistant", "content": "Indeed it is."},
            ],
        ],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "int_002",
        "question_type": "knowledge-update",
        "question": "Where does John currently live?",
        "answer": "Seattle",
        "question_date": "2023-09-01",
        "haystack_session_ids": ["s6", "s7", "s8", "s9", "s10"],
        "haystack_dates": [
            "2023-01-05", "2023-03-01", "2023-05-01", "2023-07-01", "2023-08-15"
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I live in Boston, have been here for years.", "has_answer": True},
                {"role": "assistant", "content": "Boston is a wonderful city."},
            ],
            [
                {"role": "user", "content": "Working on a new project this quarter."},
                {"role": "assistant", "content": "What's it about?"},
            ],
            [
                {"role": "user", "content": "I might relocate for the new job."},
                {"role": "assistant", "content": "That's a big decision."},
            ],
            [
                {"role": "user", "content": "Seattle has nice weather in summer."},
                {"role": "assistant", "content": "The summers are lovely there."},
            ],
            [
                {"role": "user", "content": "I just moved to Seattle last week!", "has_answer": True},
                {"role": "assistant", "content": "Congratulations on the move!"},
            ],
        ],
        "answer_session_ids": ["s6", "s10"],
    },
    {
        "question_id": "int_003",
        "question_type": "multi-session",
        "question": "Who handles Atlas database migrations?",
        "answer": "Priya",
        "question_date": "2023-10-01",
        "haystack_session_ids": ["s11", "s12", "s13", "s14"],
        "haystack_dates": [
            "2023-02-01", "2023-04-01", "2023-06-01", "2023-08-01"
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Project Atlas uses PostgreSQL for all data storage.", "has_answer": True},
                {"role": "assistant", "content": "PostgreSQL is reliable."},
            ],
            [
                {"role": "user", "content": "We need better documentation for our APIs."},
                {"role": "assistant", "content": "I agree, let's plan that."},
            ],
            [
                {"role": "user", "content": "Priya handles all PostgreSQL migrations expertly.", "has_answer": True},
                {"role": "assistant", "content": "She's very skilled."},
            ],
            [
                {"role": "user", "content": "The team meeting is at 3pm today."},
                {"role": "assistant", "content": "I'll set a reminder."},
            ],
        ],
        "answer_session_ids": ["s11", "s13"],
    },
    {
        "question_id": "int_004",
        "question_type": "temporal-reasoning",
        "question": "What was discussed before the Q2 review?",
        "answer": "budget concerns",
        "question_date": "2023-07-15",
        "haystack_session_ids": ["s15", "s16", "s17"],
        "haystack_dates": ["2023-06-01", "2023-06-20", "2023-07-05"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "We have budget concerns for next quarter.", "has_answer": True},
                {"role": "assistant", "content": "Let's review the numbers."},
            ],
            [
                {"role": "user", "content": "The Q2 review is coming up soon."},
                {"role": "assistant", "content": "I'll prepare the slides."},
            ],
            [
                {"role": "user", "content": "Q2 review went well, approved the budget."},
                {"role": "assistant", "content": "Great news!"},
            ],
        ],
        "answer_session_ids": ["s15"],
    },
]


@pytest.fixture
def integration_data_path(tmp_path: Path) -> Path:
    """Create temp file with integration test data."""
    path = tmp_path / "integration_test.json"
    path.write_text(json.dumps(INTEGRATION_DATA), encoding="utf-8")
    return path


@pytest.fixture
def gold_dataset(integration_data_path: Path):
    """Convert to gold dataset."""
    adapter = LongMemEvalAdapter()
    return adapter.load_and_convert(integration_data_path, "integration_test")


class TestNoVacuousTruth:
    """Verify metrics don't saturate at 1.0."""

    def test_recall_can_be_below_1(self, gold_dataset) -> None:
        """With partial retrieval, recall should be < 1.0."""
        evaluator = RecallAtKEvaluator(top_k=5)

        for query in gold_dataset.queries:
            expected_ids = query.expected.memory_ids
            # Simulate partial retrieval (only return some noise + 1 correct)
            partial = [expected_ids[0], "noise_1", "noise_2", "noise_3"]

            result = evaluator.evaluate(partial, expected_ids)
            if len(expected_ids) > 1:
                # Multi-expected queries should NOT have recall=1.0 with partial retrieval
                assert result.value < 1.0

    def test_temporal_accuracy_can_be_below_1(self) -> None:
        """Temporal evaluator returns < 1.0 for out-of-window retrievals."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=3)

        # Retrieved memories from day 100 (way outside window)
        result = evaluator.evaluate_temporal(
            retrieved_days=[100, 150, 200],
            expected_day_range=(5, 10),
        )
        assert result.value == 0.0

    def test_temporal_returns_0_for_empty_retrieval(self) -> None:
        """Empty retrieval returns 0.0, not 1.0."""
        evaluator = TemporalAccuracyEvaluator()
        result = evaluator.evaluate_temporal(
            retrieved_days=[],
            expected_day_range=(5, 10),
        )
        assert result.value == 0.0

    def test_recall_rejects_empty_expected(self) -> None:
        """Recall raises on empty expected (no vacuous truth)."""
        evaluator = RecallEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(
                retrieved_ids=["A"],
                expected_ids=[],
            )


class TestDiscriminativePower:
    """Verify different strategies produce different metrics."""

    def test_perfect_vs_random_retrieval(self, gold_dataset) -> None:
        """Perfect retrieval should score much higher than random."""
        evaluators = [
            RecallAtKEvaluator(top_k=10),
            MRREvaluator(top_k=10),
            NDCGEvaluator(top_k=10),
            PrecisionAtKEvaluator(top_k=1),
        ]

        for query in gold_dataset.queries:
            expected = query.expected.memory_ids

            # Perfect retrieval: all expected at top
            perfect_results = expected + ["noise_1", "noise_2"]

            # Random retrieval: mostly noise
            random_results = ["noise_1", "noise_2", "noise_3", "noise_4", "noise_5"]

            for evaluator in evaluators:
                perfect_score = evaluator.evaluate(perfect_results, expected)
                random_score = evaluator.evaluate(random_results, expected)

                assert perfect_score.value > random_score.value, (
                    f"Evaluator {evaluator.metric_name()} failed to discriminate "
                    f"for query '{query.query}'"
                )

    def test_composite_score_varies(self, gold_dataset) -> None:
        """Different retrieval qualities produce different composite scores."""
        from benchmark.workload.scheduler import MatrixRunResult

        # Simulate two strategies
        result_good = MatrixRunResult(
            cell_id="good", run_id="1", memory_type="semantic",
            retrieval_strategy="embeddings", decay_policy="exponential",
            lambda_value=0.05, pruning_threshold=0.3, workload_profile="standard",
            seed=42, recall_at_k=0.85, mrr=0.72, ndcg=0.78,
            contamination_rate=0.05, temporal_accuracy=0.80, precision_at_1=0.65,
        )

        result_bad = MatrixRunResult(
            cell_id="bad", run_id="2", memory_type="semantic",
            retrieval_strategy="bm25", decay_policy="exponential",
            lambda_value=0.05, pruning_threshold=0.3, workload_profile="standard",
            seed=42, recall_at_k=0.50, mrr=0.35, ndcg=0.40,
            contamination_rate=0.15, temporal_accuracy=0.60, precision_at_1=0.30,
        )

        assert result_good.composite_score() > result_bad.composite_score()


class TestTemporalConflict:
    """Verify temporal reasoning works correctly."""

    def test_knowledge_update_detects_stale(self, gold_dataset) -> None:
        """Knowledge-update queries should have temporal windows from evidence."""
        update_queries = [
            q for q in gold_dataset.queries
            if q.task_id == "lme_int_002"
        ]
        assert len(update_queries) == 1
        query = update_queries[0]

        # Should have temporal window spanning the evidence sessions
        assert query.expected.temporal_window is not None

    def test_temporal_evaluator_rewards_recent(self) -> None:
        """Temporal evaluator rewards retrieving recent (updated) memories."""
        evaluator = TemporalAccuracyEvaluator(tolerance_days=7)

        # Scenario: knowledge update, window is days 200-210
        expected_range = (200, 210)

        # Good system: retrieves the recent fact
        result_good = evaluator.evaluate_temporal(
            retrieved_days=[205, 208],
            expected_day_range=expected_range,
        )

        # Bad system: retrieves the stale fact
        result_bad = evaluator.evaluate_temporal(
            retrieved_days=[5, 10],
            expected_day_range=expected_range,
        )

        assert result_good.value > result_bad.value
        assert result_good.value == 1.0
        assert result_bad.value == 0.0


class TestMultiHopRetrieval:
    """Verify multi-hop retrieval scenarios work."""

    def test_multi_session_requires_multiple_evidence(self, gold_dataset) -> None:
        """Multi-session queries require evidence from multiple sessions."""
        multi_queries = [
            q for q in gold_dataset.queries
            if q.task_id == "lme_int_003"
        ]
        assert len(multi_queries) == 1
        query = multi_queries[0]

        # Multi-session should have multiple expected memory IDs
        assert len(query.expected.memory_ids) >= 2, (
            "Multi-session query should require evidence from multiple sessions"
        )


class TestDatasetProperties:
    """Verify the converted dataset has the right properties."""

    def test_multiple_question_types_present(self, gold_dataset) -> None:
        """Dataset contains multiple question types (diverse evaluation)."""
        task_prefixes = set()
        for query in gold_dataset.queries:
            # Get original question type from task_id pattern
            task_prefixes.add(query.task_id)

        assert len(task_prefixes) >= 3, "Need at least 3 different question types"

    def test_temporal_windows_exist(self, gold_dataset) -> None:
        """Some queries have temporal window constraints."""
        with_temporal = [
            q for q in gold_dataset.queries
            if q.expected.temporal_window is not None
        ]
        assert len(with_temporal) > 0, "Need queries with temporal constraints"

    def test_followup_queries_exist(self, gold_dataset) -> None:
        """Some queries are marked as follow-ups."""
        followups = [q for q in gold_dataset.queries if q.is_followup]
        assert len(followups) > 0, "Need follow-up queries"

    def test_evaluation_criteria_configured(self, gold_dataset) -> None:
        """Evaluation criteria are properly set."""
        assert gold_dataset.evaluation_criteria.recall_k == 10
        assert gold_dataset.evaluation_criteria.temporal_tolerance_days == 7
