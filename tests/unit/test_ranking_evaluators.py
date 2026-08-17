"""Unit tests for ranking evaluators: MRR, NDCG, Precision@K.

Tests verify:
1. Discriminative power — different rankings produce different scores
2. Edge cases — empty results, all-irrelevant results
3. Monotonicity — better rankings always get higher scores
4. Correctness — exact formula verification
"""

from __future__ import annotations

import math

import pytest

from benchmark.evaluation.ranking import (
    MRREvaluator,
    NDCGEvaluator,
    PrecisionAtKEvaluator,
    RecallAtKEvaluator,
)


class TestMRREvaluator:
    """Tests for Mean Reciprocal Rank."""

    def setup_method(self) -> None:
        self.evaluator = MRREvaluator(top_k=10)

    def test_perfect_mrr_rank_1(self) -> None:
        """Relevant at rank 1 → MRR = 1.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["A", "B", "C"],
            expected_ids=["A"],
        )
        assert result.value == 1.0

    def test_mrr_rank_2(self) -> None:
        """Relevant at rank 2 → MRR = 0.5."""
        result = self.evaluator.evaluate(
            retrieved_ids=["X", "A", "C"],
            expected_ids=["A"],
        )
        assert result.value == 0.5

    def test_mrr_rank_5(self) -> None:
        """Relevant at rank 5 → MRR = 0.2."""
        result = self.evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z", "W", "A"],
            expected_ids=["A"],
        )
        assert result.value == pytest.approx(0.2)

    def test_mrr_not_found(self) -> None:
        """No relevant result found → MRR = 0.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z"],
            expected_ids=["A"],
        )
        assert result.value == 0.0

    def test_mrr_empty_retrieved(self) -> None:
        """Empty retrieval → MRR = 0.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=[],
            expected_ids=["A"],
        )
        assert result.value == 0.0

    def test_mrr_empty_expected_raises(self) -> None:
        """Empty expected → raises ValueError."""
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            self.evaluator.evaluate(
                retrieved_ids=["A", "B"],
                expected_ids=[],
            )

    def test_mrr_multiple_expected_first_wins(self) -> None:
        """Multiple expected IDs → rank of FIRST found counts."""
        result = self.evaluator.evaluate(
            retrieved_ids=["X", "B", "A"],
            expected_ids=["A", "B"],
        )
        # B is at rank 2, A at rank 3 → first relevant is B at rank 2
        assert result.value == 0.5

    def test_mrr_discriminates_strategies(self) -> None:
        """Different strategies produce different MRR scores."""
        expected = ["target_1", "target_2"]

        # Strategy A: targets at ranks 1 and 2
        result_a = self.evaluator.evaluate(
            retrieved_ids=["target_1", "target_2", "noise"],
            expected_ids=expected,
        )

        # Strategy B: targets at ranks 5 and 8
        result_b = self.evaluator.evaluate(
            retrieved_ids=["n1", "n2", "n3", "n4", "target_1", "n5", "n6", "target_2"],
            expected_ids=expected,
        )

        assert result_a.value > result_b.value
        assert result_a.value == 1.0
        assert result_b.value == 0.2

    def test_mrr_respects_top_k(self) -> None:
        """Results beyond top_k are ignored."""
        evaluator = MRREvaluator(top_k=3)
        result = evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z", "A"],  # A is at rank 4, beyond top_k=3
            expected_ids=["A"],
        )
        assert result.value == 0.0


class TestNDCGEvaluator:
    """Tests for Normalized Discounted Cumulative Gain."""

    def setup_method(self) -> None:
        self.evaluator = NDCGEvaluator(top_k=10)

    def test_perfect_ndcg(self) -> None:
        """All relevant at top positions → NDCG = 1.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["A", "B", "C"],
            expected_ids=["A", "B", "C"],
        )
        assert result.value == pytest.approx(1.0)

    def test_ndcg_partial_at_top(self) -> None:
        """Some relevant at top, some missing → NDCG < 1.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["A", "X", "B"],
            expected_ids=["A", "B", "C"],
        )
        assert 0.0 < result.value < 1.0

    def test_ndcg_no_relevant(self) -> None:
        """No relevant results → NDCG = 0.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z"],
            expected_ids=["A", "B"],
        )
        assert result.value == 0.0

    def test_ndcg_empty_retrieved(self) -> None:
        """Empty retrieval → NDCG = 0.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=[],
            expected_ids=["A"],
        )
        assert result.value == 0.0

    def test_ndcg_empty_expected_raises(self) -> None:
        """Empty expected → raises ValueError."""
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            self.evaluator.evaluate(
                retrieved_ids=["A"],
                expected_ids=[],
            )

    def test_ndcg_penalizes_low_ranking(self) -> None:
        """Relevant at lower ranks gets lower NDCG than at higher ranks."""
        # Same result at rank 1 vs rank 5
        result_high = self.evaluator.evaluate(
            retrieved_ids=["A", "X", "Y", "Z", "W"],
            expected_ids=["A"],
        )
        result_low = self.evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z", "W", "A"],
            expected_ids=["A"],
        )
        assert result_high.value > result_low.value

    def test_ndcg_formula_correctness(self) -> None:
        """Verify NDCG formula for a specific case."""
        # 2 relevant items, retrieved at positions 1 and 3
        result = self.evaluator.evaluate(
            retrieved_ids=["A", "X", "B"],
            expected_ids=["A", "B"],
        )
        # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309... = 1.6309...
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_ndcg = dcg / idcg
        assert result.value == pytest.approx(expected_ndcg, abs=1e-4)

    def test_ndcg_discriminates_strategies(self) -> None:
        """NDCG produces different scores for different retrieval rankings."""
        expected = ["t1", "t2", "t3"]

        # Perfect ranking
        result_perfect = self.evaluator.evaluate(
            retrieved_ids=["t1", "t2", "t3", "n1", "n2"],
            expected_ids=expected,
        )

        # Mixed ranking
        result_mixed = self.evaluator.evaluate(
            retrieved_ids=["t1", "n1", "t2", "n2", "t3"],
            expected_ids=expected,
        )

        # Poor ranking
        result_poor = self.evaluator.evaluate(
            retrieved_ids=["n1", "n2", "n3", "n4", "t1", "n5", "t2", "n6", "t3"],
            expected_ids=expected,
        )

        assert result_perfect.value > result_mixed.value > result_poor.value


class TestPrecisionAtKEvaluator:
    """Tests for Precision@K."""

    def test_precision_at_1_correct(self) -> None:
        """Top result is relevant → P@1 = 1.0."""
        evaluator = PrecisionAtKEvaluator(top_k=1)
        result = evaluator.evaluate(
            retrieved_ids=["A", "X", "Y"],
            expected_ids=["A", "B"],
        )
        assert result.value == 1.0

    def test_precision_at_1_wrong(self) -> None:
        """Top result is irrelevant → P@1 = 0.0."""
        evaluator = PrecisionAtKEvaluator(top_k=1)
        result = evaluator.evaluate(
            retrieved_ids=["X", "A", "Y"],
            expected_ids=["A", "B"],
        )
        assert result.value == 0.0

    def test_precision_at_5(self) -> None:
        """3 out of 5 are relevant → P@5 = 0.6."""
        evaluator = PrecisionAtKEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=["A", "X", "B", "Y", "C"],
            expected_ids=["A", "B", "C", "D"],
        )
        assert result.value == pytest.approx(0.6)

    def test_precision_empty_retrieved(self) -> None:
        """No retrieval → P@K = 0.0."""
        evaluator = PrecisionAtKEvaluator(top_k=5)
        result = evaluator.evaluate(
            retrieved_ids=[],
            expected_ids=["A"],
        )
        assert result.value == 0.0

    def test_precision_empty_expected_raises(self) -> None:
        """Empty expected → raises ValueError."""
        evaluator = PrecisionAtKEvaluator(top_k=1)
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            evaluator.evaluate(
                retrieved_ids=["A"],
                expected_ids=[],
            )

    def test_precision_metric_name(self) -> None:
        """Metric name reflects K value."""
        evaluator_1 = PrecisionAtKEvaluator(top_k=1)
        evaluator_5 = PrecisionAtKEvaluator(top_k=5)
        assert evaluator_1.metric_name() == "benchmark.precision_at_1"
        assert evaluator_5.metric_name() == "benchmark.precision_at_5"


class TestRecallAtKEvaluator:
    """Tests for the strict RecallAtKEvaluator (no vacuous truth)."""

    def setup_method(self) -> None:
        self.evaluator = RecallAtKEvaluator(top_k=10)

    def test_perfect_recall(self) -> None:
        """All expected IDs retrieved → Recall = 1.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["A", "B", "C"],
            expected_ids=["A", "B"],
        )
        assert result.value == 1.0

    def test_partial_recall(self) -> None:
        """Some expected IDs retrieved → Recall between 0 and 1."""
        result = self.evaluator.evaluate(
            retrieved_ids=["A", "X", "Y"],
            expected_ids=["A", "B", "C"],
        )
        assert result.value == pytest.approx(1 / 3)

    def test_zero_recall(self) -> None:
        """No expected IDs retrieved → Recall = 0.0."""
        result = self.evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z"],
            expected_ids=["A", "B"],
        )
        assert result.value == 0.0

    def test_empty_expected_raises(self) -> None:
        """Empty gold set → raises ValueError (no vacuous truth!)."""
        with pytest.raises(ValueError, match="expected_ids cannot be empty"):
            self.evaluator.evaluate(
                retrieved_ids=["A"],
                expected_ids=[],
            )

    def test_recall_respects_top_k(self) -> None:
        """Only top_k results are considered."""
        evaluator = RecallAtKEvaluator(top_k=3)
        result = evaluator.evaluate(
            retrieved_ids=["X", "Y", "Z", "A"],  # A beyond top_k
            expected_ids=["A"],
        )
        assert result.value == 0.0


class TestDiscriminativePower:
    """Tests verifying that different strategies produce different scores.

    This is the key requirement: a good benchmark must separate strategies.
    """

    def test_bm25_vs_embedding_scenario(self) -> None:
        """Simulate BM25 vs embedding retrieval on same query.

        BM25 finds lexical matches (partial).
        Embedding finds semantic matches (full).
        """
        expected = ["sem_1", "sem_2", "lex_1"]

        # BM25: finds lexical match at rank 3, misses semantic
        bm25_retrieved = ["noise_1", "noise_2", "lex_1", "noise_3", "noise_4"]

        # Embedding: finds both semantic and lexical
        embed_retrieved = ["sem_1", "sem_2", "lex_1", "noise_1", "noise_2"]

        mrr = MRREvaluator(top_k=10)
        ndcg = NDCGEvaluator(top_k=10)
        recall = RecallAtKEvaluator(top_k=10)
        precision = PrecisionAtKEvaluator(top_k=1)

        # BM25 results
        bm25_mrr = mrr.evaluate(bm25_retrieved, expected)
        bm25_ndcg = ndcg.evaluate(bm25_retrieved, expected)
        bm25_recall = recall.evaluate(bm25_retrieved, expected)
        bm25_p1 = precision.evaluate(bm25_retrieved, expected)

        # Embedding results
        embed_mrr = mrr.evaluate(embed_retrieved, expected)
        embed_ndcg = ndcg.evaluate(embed_retrieved, expected)
        embed_recall = recall.evaluate(embed_retrieved, expected)
        embed_p1 = precision.evaluate(embed_retrieved, expected)

        # Embedding should dominate on all metrics
        assert embed_mrr.value > bm25_mrr.value
        assert embed_ndcg.value > bm25_ndcg.value
        assert embed_recall.value > bm25_recall.value
        assert embed_p1.value > bm25_p1.value

        # Verify actual difference is meaningful (not just epsilon)
        assert embed_recall.value - bm25_recall.value >= 0.3

    def test_ranking_sensitivity(self) -> None:
        """Same recall, different ranking → different scores.

        Two strategies retrieve the same relevant items but rank them differently.
        """
        expected = ["A", "B"]

        # Strategy 1: relevant items at top
        good_ranking = ["A", "B", "X", "Y", "Z"]

        # Strategy 2: relevant items at bottom
        poor_ranking = ["X", "Y", "Z", "A", "B"]

        mrr = MRREvaluator(top_k=10)
        ndcg = NDCGEvaluator(top_k=10)
        recall = RecallAtKEvaluator(top_k=10)

        # Both have same recall (both find A and B)
        assert recall.evaluate(good_ranking, expected).value == recall.evaluate(poor_ranking, expected).value

        # But MRR and NDCG differ significantly
        assert mrr.evaluate(good_ranking, expected).value > mrr.evaluate(poor_ranking, expected).value
        assert ndcg.evaluate(good_ranking, expected).value > ndcg.evaluate(poor_ranking, expected).value

    def test_no_saturation_at_100_percent(self) -> None:
        """Verify that realistic scenarios do NOT saturate at 100%."""
        expected = ["t1", "t2", "t3", "t4", "t5"]  # 5 expected memories

        # Realistic BM25: finds 3/5 with some at lower ranks
        bm25 = ["t1", "noise1", "t3", "noise2", "noise3", "noise4", "noise5", "t5"]

        # Realistic embedding: finds 4/5
        embed = ["t1", "t2", "t4", "noise1", "t3", "noise2"]

        recall = RecallAtKEvaluator(top_k=10)
        bm25_recall = recall.evaluate(bm25, expected)
        embed_recall = recall.evaluate(embed, expected)

        # Neither saturates at 1.0
        assert bm25_recall.value < 1.0
        assert embed_recall.value < 1.0

        # They are distinguishable
        assert bm25_recall.value != embed_recall.value
