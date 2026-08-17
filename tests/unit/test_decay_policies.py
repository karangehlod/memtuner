"""Unit tests for decay policies — covering all types and branches."""

from __future__ import annotations

import pytest

from benchmark.memory.policies.decay import (
    ExponentialDecayPolicy,
    LinearDecayPolicy,
    StepDecayPolicy,
)


@pytest.mark.unit
class TestLinearDecayPolicy:
    """Tests for LinearDecayPolicy."""

    def test_no_decay_at_day_zero(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.1, threshold=0.35)
        scores = {"M-001": 0.8, "M-002": 0.5}
        flagged = policy.apply(day=0, memory_scores=scores)
        # At day 0, decay_factor=1.0, so M-002 (0.5) is still above 0.35
        assert flagged == []

    def test_flags_low_scoring_after_decay(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.1, threshold=0.35)
        scores = {"M-001": 0.4}
        # day=10: decay_factor = max(0, 1 - 0.1*10) = 0
        flagged = policy.apply(day=10, memory_scores=scores)
        assert "M-001" in flagged

    def test_decay_factor_clamped_at_zero(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.2, threshold=0.1)
        scores = {"M-001": 1.0}
        # day=10: decay_factor = max(0, 1 - 2.0) = 0
        flagged = policy.apply(day=10, memory_scores=scores)
        assert "M-001" in flagged

    def test_partial_flagging(self) -> None:
        policy = LinearDecayPolicy(decay_rate=0.05, threshold=0.35)
        scores = {"M-001": 0.9, "M-002": 0.3}
        # day=5: decay_factor=0.75, M-001: 0.675 > 0.35, M-002: 0.225 < 0.35
        flagged = policy.apply(day=5, memory_scores=scores)
        assert "M-002" in flagged
        assert "M-001" not in flagged


@pytest.mark.unit
class TestStepDecayPolicy:
    """Tests for StepDecayPolicy."""

    def test_no_flags_before_lifespan(self) -> None:
        policy = StepDecayPolicy(lifespan_days=10)
        scores = {"M-001": 0.8, "M-002": 0.5}
        flagged = policy.apply(day=5, memory_scores=scores)
        assert flagged == []

    def test_all_flagged_at_lifespan(self) -> None:
        policy = StepDecayPolicy(lifespan_days=10)
        scores = {"M-001": 0.8, "M-002": 0.5}
        flagged = policy.apply(day=10, memory_scores=scores)
        assert set(flagged) == {"M-001", "M-002"}

    def test_all_flagged_after_lifespan(self) -> None:
        policy = StepDecayPolicy(lifespan_days=5)
        scores = {"M-001": 1.0}
        flagged = policy.apply(day=20, memory_scores=scores)
        assert "M-001" in flagged

    def test_empty_scores(self) -> None:
        policy = StepDecayPolicy(lifespan_days=10)
        flagged = policy.apply(day=15, memory_scores={})
        assert flagged == []


@pytest.mark.unit
class TestExponentialDecayPolicyEdgeCases:
    """Additional edge case tests for ExponentialDecayPolicy."""

    def test_empty_scores(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=0.05, threshold=0.35)
        flagged = policy.apply(day=10, memory_scores={})
        assert flagged == []

    def test_high_lambda_flags_quickly(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=1.0, threshold=0.35)
        scores = {"M-001": 0.8}
        flagged = policy.apply(day=2, memory_scores=scores)
        assert "M-001" in flagged

    def test_zero_lambda_no_decay(self) -> None:
        policy = ExponentialDecayPolicy(decay_lambda=0.0, threshold=0.35)
        scores = {"M-001": 0.5}
        flagged = policy.apply(day=100, memory_scores=scores)
        assert "M-001" not in flagged
