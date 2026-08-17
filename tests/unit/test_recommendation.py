"""Tests for recommendation layer (ranker + explainer) — Phase 9 / 11."""

from __future__ import annotations

import pytest

from benchmark.recommendation.ranker import MatrixExplainer, MatrixRanker, QualityThresholds
from benchmark.workload.scheduler import MatrixRunResult


def _make_result(
    cell_id: str = "c1",
    memory_type: str = "episodic",
    retrieval_strategy: str = "bm25",
    decay_policy: str = "none",
    lambda_value: float = 0.0,
    recall_at_k: float = 0.70,
    contamination_rate: float = 0.15,
    temporal_accuracy: float = 0.80,
    module_accuracy: float = 0.90,
    peak_ram_mb: float = 256.0,
    success: bool = True,
):
    return MatrixRunResult(
        cell_id=cell_id,
        run_id="test-run-id",
        memory_type=memory_type,
        retrieval_strategy=retrieval_strategy,
        decay_policy=decay_policy,
        lambda_value=lambda_value,
        pruning_threshold=0.0,
        workload_profile="medium_qpd",
        seed=42,
        recall_at_k=recall_at_k,
        contamination_rate=contamination_rate,
        temporal_accuracy=temporal_accuracy,
        module_accuracy=module_accuracy,
        total_queries=100,
        correct_recalls=int(recall_at_k * 100),
        peak_ram_mb=peak_ram_mb,
        avg_ram_mb=peak_ram_mb * 0.8,
        peak_cpu_percent=30.0,
        duration_seconds=1.0,
        total_cost=0.001,
        success=success,
        platform="darwin",
    )


@pytest.mark.unit
class TestQualityThresholds:
    def test_defaults_are_reasonable(self):
        t = QualityThresholds()
        assert 0 < t.min_recall < 1
        assert 0 < t.max_noise_ratio < 1
        assert 0 <= t.min_temporal < 1
        assert t.max_peak_ram_mb > 0

    def test_immutable(self):
        t = QualityThresholds()
        with pytest.raises((AttributeError, TypeError)):
            t.min_recall = 0.99  # type: ignore


@pytest.mark.unit
class TestMatrixRanker:
    def test_rank_empty_list_returns_empty(self):
        ranker = MatrixRanker()
        assert ranker.rank([]) == []

    def test_rank_all_failed_returns_empty(self):
        ranker = MatrixRanker()
        results = [_make_result(success=False), _make_result(success=False)]
        assert ranker.rank(results) == []

    def test_rank_best_first(self):
        ranker = MatrixRanker()
        results = [
            _make_result(cell_id="bad", recall_at_k=0.50, contamination_rate=0.30),
            _make_result(cell_id="good", recall_at_k=0.90, contamination_rate=0.05),
        ]
        ranked = ranker.rank(results)
        assert ranked[0].memory_type == "episodic"  # both same type
        assert ranked[0].composite_score > ranked[1].composite_score

    def test_rank_rank_numbers_sequential(self):
        ranker = MatrixRanker()
        results = [_make_result(cell_id=f"c{i}", recall_at_k=0.5 + i * 0.05) for i in range(5)]
        ranked = ranker.rank(results)
        for i, rec in enumerate(ranked, 1):
            assert rec.rank == i

    def test_meets_thresholds_for_good_result(self):
        ranker = MatrixRanker(QualityThresholds(
            min_recall=0.50, max_noise_ratio=0.30, min_temporal=0.60
        ))
        result = _make_result(recall_at_k=0.80, contamination_rate=0.10, temporal_accuracy=0.75)
        ranked = ranker.rank([result])
        assert ranked[0].meets_thresholds is True

    def test_fails_thresholds_for_low_recall(self):
        ranker = MatrixRanker(QualityThresholds(min_recall=0.70))
        result = _make_result(recall_at_k=0.40)
        ranked = ranker.rank([result])
        assert ranked[0].meets_thresholds is False

    def test_fails_thresholds_for_high_fpr(self):
        ranker = MatrixRanker(QualityThresholds(max_noise_ratio=0.20))
        result = _make_result(recall_at_k=0.80, contamination_rate=0.35)
        ranked = ranker.rank([result])
        assert ranked[0].meets_thresholds is False

    def test_fails_thresholds_for_high_ram(self):
        ranker = MatrixRanker(QualityThresholds(max_peak_ram_mb=512.0))
        result = _make_result(recall_at_k=0.80, contamination_rate=0.10, peak_ram_mb=2048.0)
        ranked = ranker.rank([result])
        assert ranked[0].meets_thresholds is False

    def test_best_production_config_returns_none_if_none_pass(self):
        ranker = MatrixRanker(QualityThresholds(min_recall=0.99))
        results = [_make_result(recall_at_k=0.50)]
        assert ranker.best_production_config(results) is None

    def test_best_production_config_returns_top_passing(self):
        thresholds = QualityThresholds(min_recall=0.60, max_noise_ratio=0.25, min_temporal=0.60)
        ranker = MatrixRanker(thresholds)
        results = [
            _make_result(cell_id="a", recall_at_k=0.75, contamination_rate=0.15, temporal_accuracy=0.70),
            _make_result(cell_id="b", recall_at_k=0.90, contamination_rate=0.08, temporal_accuracy=0.80),
            _make_result(cell_id="c", recall_at_k=0.40, contamination_rate=0.20),  # fails recall threshold
        ]
        best = ranker.best_production_config(results)
        assert best is not None
        assert best.meets_thresholds is True

    def test_explanation_contains_verdict(self):
        ranker = MatrixRanker(QualityThresholds(min_recall=0.50))
        result = _make_result(recall_at_k=0.80, contamination_rate=0.10, temporal_accuracy=0.75)
        ranked = ranker.rank([result])
        assert "PASS" in ranked[0].explanation or "FAIL" in ranked[0].explanation

    def test_explanation_mentions_retrieval_strategy(self):
        ranker = MatrixRanker()
        result = _make_result(retrieval_strategy="bm25", recall_at_k=0.70, contamination_rate=0.10, temporal_accuracy=0.70)
        ranked = ranker.rank([result])
        # 'bm25' should appear in the explanation note
        assert "bm25" in ranked[0].explanation.lower() or "sparse" in ranked[0].explanation.lower()


@pytest.mark.unit
class TestMatrixExplainer:
    def test_compare_returns_string(self):
        a = _make_result(cell_id="a", recall_at_k=0.80, contamination_rate=0.10)
        b = _make_result(cell_id="b", recall_at_k=0.60, contamination_rate=0.20)
        explainer = MatrixExplainer()
        text = explainer.compare(a, b)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_compare_identifies_winner(self):
        a = _make_result(cell_id="a", recall_at_k=0.90, contamination_rate=0.05, temporal_accuracy=0.85)
        b = _make_result(cell_id="b", recall_at_k=0.50, contamination_rate=0.30, temporal_accuracy=0.50)
        explainer = MatrixExplainer()
        text = explainer.compare(a, b)
        assert "A" in text

    def test_compare_shows_all_metrics(self):
        a = _make_result()
        b = _make_result(recall_at_k=0.80)
        explainer = MatrixExplainer()
        text = explainer.compare(a, b)
        assert "Recall" in text
        assert "Noise" in text or "contamination" in text.lower()
        assert "Composite" in text
