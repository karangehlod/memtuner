"""Tests for matrix aggregator and reporter — Phase 7 / 11."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from benchmark.workload.aggregator import MatrixAggregator, MatrixReporter
from benchmark.workload.scheduler import MatrixRunResult


def _make_result(
    cell_id: str = "cell-001",
    memory_type: str = "episodic",
    retrieval_strategy: str = "bm25",
    decay_policy: str = "none",
    lambda_value: float = 0.0,
    pruning_threshold: float = 0.0,
    recall_at_k: float = 0.70,
    contamination_rate: float = 0.15,
    temporal_accuracy: float = 0.80,
    module_accuracy: float = 0.90,
    mrr: float = 0.0,
    ndcg: float = 0.0,
    precision_at_1: float = 0.0,
    precision_at_k: float | None = None,
    total_queries: int = 100,
    correct_recalls: int = 70,
    peak_ram_mb: float = 256.0,
    avg_ram_mb: float = 200.0,
    peak_cpu_percent: float = 40.0,
    duration_seconds: float = 2.0,
    total_cost: float = 0.001,
    success: bool = True,
    error_message: str = "",
    workload_profile: str = "medium_qpd",
    platform: str = "darwin",
):
    r = MatrixRunResult(
        cell_id=cell_id,
        run_id="test-run-id",
        memory_type=memory_type,
        retrieval_strategy=retrieval_strategy,
        decay_policy=decay_policy,
        lambda_value=lambda_value,
        pruning_threshold=pruning_threshold,
        workload_profile=workload_profile,
        seed=42,
        recall_at_k=recall_at_k,
        contamination_rate=contamination_rate,
        precision_at_k=(1.0 - contamination_rate) if precision_at_k is None else precision_at_k,
        temporal_accuracy=temporal_accuracy,
        module_accuracy=module_accuracy,
        mrr=mrr,
        ndcg=ndcg,
        precision_at_1=precision_at_1,
        total_queries=total_queries,
        correct_recalls=correct_recalls,
        peak_ram_mb=peak_ram_mb,
        avg_ram_mb=avg_ram_mb,
        peak_cpu_percent=peak_cpu_percent,
        duration_seconds=duration_seconds,
        total_cost=total_cost,
        success=success,
        error_message=error_message,
        platform=platform,
    )
    return r


@pytest.mark.unit
class TestMatrixRunResultCompositeScore:
    def test_composite_score_formula(self):
        r = _make_result(
            recall_at_k=0.80,
            contamination_rate=0.10,
            temporal_accuracy=0.70,
            module_accuracy=0.90,
            mrr=0.0,
            ndcg=0.0,
            precision_at_1=0.0,
        )
        # New formula: recall(40%) + precision@k(25%) + mrr(20%) + temporal(15%).
        expected = (
            0.40 * 0.80
            + 0.25 * (1.0 - 0.10)
            + 0.20 * 0.0
            + 0.15 * 0.70
        )
        assert abs(r.composite_score() - expected) < 1e-9

    def test_composite_score_is_higher_for_better_recall(self):
        r_good = _make_result(recall_at_k=0.90, contamination_rate=0.10)
        r_bad = _make_result(recall_at_k=0.50, contamination_rate=0.10)
        assert r_good.composite_score() > r_bad.composite_score()

    def test_composite_score_is_lower_for_higher_noise(self):
        r_good = _make_result(recall_at_k=0.70, contamination_rate=0.05)
        r_bad = _make_result(recall_at_k=0.70, contamination_rate=0.40)
        assert r_good.composite_score() > r_bad.composite_score()

    def test_to_dict_is_json_serializable(self):
        r = _make_result()
        d = r.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_has_metrics_section(self):
        r = _make_result()
        d = r.to_dict()
        assert "metrics" in d
        assert "recall_at_k" in d["metrics"]
        assert "composite_score" in d["metrics"]

    def test_to_dict_has_resources_section(self):
        r = _make_result()
        d = r.to_dict()
        assert "resources" in d
        assert "peak_ram_mb" in d["resources"]


@pytest.mark.unit
class TestMatrixAggregator:
    def _three_results(self):
        a = _make_result(cell_id="a", memory_type="episodic", retrieval_strategy="bm25",
                         recall_at_k=0.70, contamination_rate=0.15, decay_policy="none")
        b = _make_result(cell_id="b", memory_type="semantic", retrieval_strategy="embeddings",
                         recall_at_k=0.80, contamination_rate=0.10, decay_policy="exponential")
        c = _make_result(cell_id="c", memory_type="preference", retrieval_strategy="hybrid",
                         recall_at_k=0.60, contamination_rate=0.25, decay_policy="none")
        return [a, b, c]

    def test_total_count(self):
        agg = MatrixAggregator(self._three_results())
        assert agg.total == 3

    def test_success_count(self):
        results = self._three_results()
        results.append(_make_result(cell_id="fail", success=False, error_message="crashed"))
        agg = MatrixAggregator(results)
        assert agg.success_count == 3
        assert agg.failure_count == 1

    def test_best_overall_picks_highest_composite(self):
        agg = MatrixAggregator(self._three_results())
        best = agg.best_overall()
        # b has recall=0.80, contamination=0.10 → highest composite
        assert best.cell_id == "b"

    def test_best_overall_none_if_all_failed(self):
        results = [_make_result(cell_id="x", success=False)]
        agg = MatrixAggregator(results)
        assert agg.best_overall() is None

    def test_rank_by_memory_type_returns_all_types(self):
        agg = MatrixAggregator(self._three_results())
        ranking = agg.rank_by_memory_type()
        types = {row["memory_type"] for row in ranking}
        assert types == {"episodic", "semantic", "preference"}

    def test_rank_by_memory_type_sorted_descending(self):
        agg = MatrixAggregator(self._three_results())
        ranking = agg.rank_by_memory_type()
        scores = [row["avg_composite"] for row in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_retrieval_strategy_sorted(self):
        agg = MatrixAggregator(self._three_results())
        ranking = agg.rank_by_retrieval_strategy()
        scores = [row["avg_composite"] for row in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_decay_policy_sorted(self):
        agg = MatrixAggregator(self._three_results())
        ranking = agg.rank_by_decay_policy()
        scores = [row["avg_composite"] for row in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_lambda_sweep_returns_correct_cells(self):
        results = []
        for lam in [0.01, 0.05, 0.10]:
            results.append(_make_result(
                cell_id=f"cell-{lam}",
                memory_type="episodic",
                retrieval_strategy="bm25",
                decay_policy="exponential",
                lambda_value=lam,
                recall_at_k=0.60 + lam,  # recall increases with lambda (just for test)
            ))
        # Add a distractor (different type)
        results.append(_make_result(cell_id="d", memory_type="semantic", decay_policy="none"))
        agg = MatrixAggregator(results)

        sweep = agg.lambda_sweep_for("episodic", "bm25", "exponential")
        assert len(sweep) == 3
        lambdas = [row["lambda"] for row in sweep]
        assert lambdas == sorted(lambdas)

    def test_top_n_returns_n_items(self):
        results = [_make_result(cell_id=f"c{i}", recall_at_k=i * 0.05) for i in range(1, 15)]
        agg = MatrixAggregator(results)
        top5 = agg.top_n(5)
        assert len(top5) == 5

    def test_top_n_sorted_by_composite(self):
        results = [_make_result(cell_id=f"c{i}", recall_at_k=i * 0.05) for i in range(1, 10)]
        agg = MatrixAggregator(results)
        top = agg.top_n(5)
        scores = [d["metrics"]["composite_score"] for d in top]
        assert scores == sorted(scores, reverse=True)

    def test_build_grid_table_has_one_row_per_result(self):
        agg = MatrixAggregator(self._three_results())
        table = agg.build_grid_table()
        assert len(table) == 3

    def test_build_grid_table_has_required_columns(self):
        agg = MatrixAggregator(self._three_results())
        table = agg.build_grid_table()
        required = {"cell_id", "memory_type", "retrieval_strategy", "decay_policy",
                    "recall_at_k", "contamination_rate", "composite_score"}
        for row in table:
            assert required.issubset(row.keys())

    def test_summary_has_all_sections(self):
        agg = MatrixAggregator(self._three_results())
        s = agg.summary()
        assert "total_cells" in s
        assert "best_config" in s
        assert "memory_type_ranking" in s
        assert "retrieval_strategy_ranking" in s
        assert "decay_policy_ranking" in s
        assert "top_10" in s


@pytest.mark.unit
class TestMatrixReporter:
    def _temp_dir(self):
        return Path(tempfile.mkdtemp())

    def _results(self):
        return [
            _make_result(cell_id="a", memory_type="episodic"),
            _make_result(cell_id="b", memory_type="semantic"),
        ]

    def test_write_all_creates_three_files(self):
        output_dir = self._temp_dir()
        agg = MatrixAggregator(self._results())
        reporter = MatrixReporter(output_dir)
        paths = reporter.write_all(agg, run_id="test_run")
        assert len(paths) == 3
        for path_str in paths.values():
            assert Path(path_str).exists()

    def test_summary_json_is_valid(self):
        output_dir = self._temp_dir()
        agg = MatrixAggregator(self._results())
        reporter = MatrixReporter(output_dir)
        paths = reporter.write_all(agg, run_id="test_json")
        json_path = Path(paths["summary_json"])
        with open(json_path) as f:
            data = json.load(f)
        assert "total_cells" in data

    def test_grid_csv_has_header_and_rows(self):
        import csv
        output_dir = self._temp_dir()
        agg = MatrixAggregator(self._results())
        reporter = MatrixReporter(output_dir)
        paths = reporter.write_all(agg, run_id="test_csv")
        csv_path = Path(paths["grid_csv"])
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "memory_type" in rows[0]

    def test_text_report_contains_best_config(self):
        output_dir = self._temp_dir()
        agg = MatrixAggregator(self._results())
        reporter = MatrixReporter(output_dir)
        paths = reporter.write_all(agg, run_id="test_txt")
        txt_path = Path(paths["text_report"])
        text = txt_path.read_text()
        assert "BEST CONFIGURATION" in text

    def test_text_report_contains_rankings(self):
        output_dir = self._temp_dir()
        agg = MatrixAggregator(self._results())
        reporter = MatrixReporter(output_dir)
        paths = reporter.write_all(agg, run_id="test_rank")
        txt_path = Path(paths["text_report"])
        text = txt_path.read_text()
        assert "MEMORY TYPE RANKING" in text
        assert "RETRIEVAL STRATEGY RANKING" in text
        assert "DECAY POLICY RANKING" in text
