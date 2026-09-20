"""Tests for plot-specific safeguards when matplotlib is available."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from scripts import plot_benchmark


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "retrieval_strategy": "hybrid",
        "study_phase": "phase4_decay_broad",
        "memory_type": "episodic",
        "decay_policy": "none",
        "total_queries": "10",
    }
    row.update(overrides)
    return row


def test_ranking_comparison_never_falls_back_to_tuning_sweeps() -> None:
    tuning_row = _row(study_phase="phase3_hybrid_weight", memory_type="all")

    assert plot_benchmark._ranking_comparable([tuning_row]) == []


def test_decay_comparison_requires_equal_repeated_measurements() -> None:
    uneven = [
        _row(decay_policy="none"),
        _row(decay_policy="none"),
        _row(decay_policy="exponential"),
    ]
    balanced = uneven + [_row(decay_policy="exponential")]

    assert plot_benchmark._equal_replication_decay(uneven) == {}
    assert set(plot_benchmark._equal_replication_decay(balanced)) == {"none", "exponential"}
