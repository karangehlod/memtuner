"""Unit tests for memory lifecycle policies: decay, pruning, and promotion."""

from __future__ import annotations

import pytest

from benchmark.memory.policies.decay import ExponentialDecayPolicy, LinearDecayPolicy
from benchmark.memory.policies.promotion import ImportanceBasedPromotionPolicy
from benchmark.memory.policies.pruning import (
    CapacityBasedPruningPolicy,
    ScoreThresholdPruningPolicy,
)


@pytest.mark.unit
class TestExponentialDecayPolicy:
    """Tests for exponential decay."""

    def test_no_decay_at_day_zero(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=0.05, threshold=0.35)
        scores = {"M-001": 0.9, "M-002": 0.5}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert flagged == []

    def test_high_decay_flags_low_scores(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=0.1, threshold=0.35)
        scores = {"M-001": 0.4, "M-002": 0.9}
        # At day 10: 0.4 * e^(-0.1*10) = 0.4 * 0.368 = 0.147 < 0.35
        # At day 10: 0.9 * e^(-0.1*10) = 0.9 * 0.368 = 0.331 < 0.35
        flagged = policy.apply(day=10, memory_scores=scores)
        assert "M-001" in flagged

    def test_deterministic_output(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=0.05, threshold=0.35)
        scores = {"M-001": 0.7, "M-002": 0.4, "M-003": 0.2}
        result_a = sorted(policy.apply(day=15, memory_scores=scores))
        result_b = sorted(policy.apply(day=15, memory_scores=scores))
        assert result_a == result_b

    def test_correct_formula(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=0.05, threshold=0.35)
        scores = {"M-001": 0.8}
        # At day 20: 0.8 * e^(-0.05*20) = 0.8 * e^(-1) ≈ 0.8 * 0.368 = 0.294
        flagged = policy.apply(day=20, memory_scores=scores)
        assert "M-001" in flagged  # 0.294 < 0.35


@pytest.mark.unit
class TestLinearDecayPolicy:
    """Tests for linear decay."""

    def test_no_decay_at_day_zero(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.05, threshold=0.35)
        scores = {"M-001": 0.9}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert flagged == []

    def test_full_decay_at_rate_boundary(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.1, threshold=0.01)
        scores = {"M-001": 0.5}
        # At day 10: 0.5 * max(0, 1 - 0.1*10) = 0.5 * 0 = 0
        flagged = policy.apply(day=10, memory_scores=scores)
        assert "M-001" in flagged

    def test_partial_decay(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.05, threshold=0.35)
        scores = {"M-001": 0.8}
        # At day 5: 0.8 * max(0, 1 - 0.05*5) = 0.8 * 0.75 = 0.6
        flagged = policy.apply(day=5, memory_scores=scores)
        assert "M-001" not in flagged  # 0.6 > 0.35


@pytest.mark.unit
class TestScoreThresholdPruning:
    """Tests for score threshold pruning."""

    def test_prunes_below_threshold(self) -> None:
        policy = ScoreThresholdPruningPolicy(threshold=0.5)
        scores = {"M-001": 0.8, "M-002": 0.3, "M-003": 0.1}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert "M-002" in flagged
        assert "M-003" in flagged
        assert "M-001" not in flagged

    def test_nothing_pruned_above_threshold(self) -> None:
        policy = ScoreThresholdPruningPolicy(threshold=0.1)
        scores = {"M-001": 0.8, "M-002": 0.5}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert flagged == []

    def test_day_parameter_ignored(self) -> None:
        policy = ScoreThresholdPruningPolicy(threshold=0.5)
        scores = {"M-001": 0.3}
        result_day_0 = policy.apply(day=0, memory_scores=scores)
        result_day_100 = policy.apply(day=100, memory_scores=scores)
        assert result_day_0 == result_day_100


@pytest.mark.unit
class TestCapacityBasedPruning:
    """Tests for capacity-based pruning."""

    def test_no_pruning_under_capacity(self) -> None:
        policy = CapacityBasedPruningPolicy(max_capacity=5)
        scores = {"M-001": 0.9, "M-002": 0.8, "M-003": 0.7}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert flagged == []

    def test_prunes_lowest_scores(self) -> None:
        policy = CapacityBasedPruningPolicy(max_capacity=2)
        scores = {"M-001": 0.9, "M-002": 0.5, "M-003": 0.1}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert "M-003" in flagged
        assert len(flagged) == 1  # Only 1 over capacity

    def test_exact_capacity_no_pruning(self) -> None:
        policy = CapacityBasedPruningPolicy(max_capacity=3)
        scores = {"M-001": 0.9, "M-002": 0.5, "M-003": 0.1}
        flagged = policy.apply(day=0, memory_scores=scores)
        assert flagged == []


@pytest.mark.unit
class TestImportanceBasedPromotion:
    """Tests for importance-based promotion."""

    def test_promotes_above_threshold(self) -> None:
        policy = ImportanceBasedPromotionPolicy(importance_threshold=0.6)
        scores = {"M-001": 0.9, "M-002": 0.3, "M-003": 0.7}
        promoted = policy.apply(day=0, memory_scores=scores)
        assert "M-001" in promoted
        assert "M-003" in promoted
        assert "M-002" not in promoted

    def test_nothing_promoted_below_threshold(self) -> None:
        policy = ImportanceBasedPromotionPolicy(importance_threshold=0.95)
        scores = {"M-001": 0.9, "M-002": 0.8}
        promoted = policy.apply(day=0, memory_scores=scores)
        assert promoted == []

    def test_boundary_value_included(self) -> None:
        policy = ImportanceBasedPromotionPolicy(importance_threshold=0.6)
        scores = {"M-001": 0.6}
        promoted = policy.apply(day=0, memory_scores=scores)
        assert "M-001" in promoted
