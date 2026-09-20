"""Tests for study visualization metadata that do not require matplotlib."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.reporting.study_visualizer import StudyVisualizer


@pytest.mark.unit
def test_visualizer_discloses_partial_study_and_evaluation_context(tmp_path) -> None:
    result = SimpleNamespace(
        success=True,
        study_phase="phase1_baselines",
        decay_policy="none",
        lambda_value=0.0,
        retrieval_strategy="bm25",
        test_holdout_fraction=0.2,
    )
    visualizer = StudyVisualizer(
        [result],
        tmp_path,
        run_metadata={"leakage": {"status": "clean"}},
    )

    assert visualizer._evaluation_caption() == "Holdout: 20% | Leakage: clean"
    assert visualizer._missing_full_report_sections() == [
        "embedding",
        "hybrid",
        "reranker",
        "decay",
    ]
