"""Regression tests for reliability safeguards in generated benchmark artifacts."""

from __future__ import annotations

from scripts import generate_reports


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_name": "Synthetic",
        "retrieval_strategy": "bm25",
        "study_phase": "phase1_baselines",
        "memory_type": "episodic",
        "embedding_model": "none",
        "embedding_backend": "none",
        "decay_policy": "none",
        "lambda": "0.0",
        "bm25_weight": "1.0",
        "recall_at_k": "0.4",
        "precision_at_k": "0.3",
        "precision_at_1": "0.3",
        "mrr": "0.4",
        "temporal_accuracy": "0.0",
        "composite_score": "0.38",
        "total_queries": "10",
    }
    row.update(overrides)
    return row


def test_report_marks_best_tuning_cell_as_non_deployable_without_final_test() -> None:
    data = generate_reports.build_reports_data([_row()])
    dataset = data["datasets"][0]

    assert dataset["recommendationStatus"] == "tuning_only"
    assert dataset["bestConfig"] is None
    assert dataset["tuningCandidate"]["strategy"] == "bm25"


def test_report_uses_final_test_row_for_deployable_recommendation() -> None:
    tuning = _row(composite_score="0.9")
    final = _row(study_phase="final_test", recall_at_k="0.5", composite_score="0.45")
    data = generate_reports.build_reports_data([tuning, final])
    dataset = data["datasets"][0]

    assert dataset["recommendationStatus"] == "final_tested"
    assert dataset["bestConfig"]["recall10"] == 0.5
