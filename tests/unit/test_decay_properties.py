"""Property-based tests for decay policies using Hypothesis.

Tests invariants about decay behavior:
- Decay factor is always in [0, 1]
- Decay is monotonically non-increasing
- Decay approaches 0 as time increases
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmark.memory.policies.decay import ExponentialDecayPolicy


@pytest.mark.unit
class TestDecayPolicyProperties:
    """Property-based tests for decay policy."""

    @given(
        initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        days_elapsed=st.integers(min_value=0, max_value=365),
    )
    def test_decay_always_in_valid_range(
        self, initial_score: float, days_elapsed: int
    ) -> None:
        """Decay factor must always be in [0, 1]."""
        policy = ExponentialDecayPolicy()
        result = policy.compute_decay(initial_score, days_elapsed)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0, f"Decay {result} not in [0, 1]"
        assert not any(
            [float(result) != float(result)]
        ), "Result should not be NaN"

    @given(
        initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        days_elapsed=st.integers(min_value=0, max_value=365),
    )
    def test_decay_never_exceeds_initial_score(
        self, initial_score: float, days_elapsed: int
    ) -> None:
        """Decay factor must never exceed the initial score."""
        policy = ExponentialDecayPolicy()
        result = policy.compute_decay(initial_score, days_elapsed)

        assert result <= initial_score + 1e-9, (
            f"Decay {result} exceeds initial {initial_score}"
        )

    @given(
        initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        days1=st.integers(min_value=0, max_value=100),
        days2=st.integers(min_value=100, max_value=365),
    )
    def test_decay_monotonically_decreasing(
        self, initial_score: float, days1: int, days2: int
    ) -> None:
        """Decay should be monotonically non-increasing as time increases."""
        policy = ExponentialDecayPolicy()
        decay1 = policy.compute_decay(initial_score, days1)
        decay2 = policy.compute_decay(initial_score, days2)

        assert decay1 >= decay2 - 1e-9, (
            f"Decay not monotonic: day {days1}={decay1} > day {days2}={decay2}"
        )

    @given(initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_decay_at_zero_days(self, initial_score: float) -> None:
        """Decay at day 0 should be the initial score."""
        policy = ExponentialDecayPolicy()
        result = policy.compute_decay(initial_score, 0)

        assert abs(result - initial_score) < 1e-9

    @given(initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_decay_approaches_zero(self, initial_score: float) -> None:
        """Decay should approach 0 as time increases significantly."""
        policy = ExponentialDecayPolicy()
        result = policy.compute_decay(initial_score, 365)

        assert result < 0.1, f"Decay at 365 days should be < 0.1, got {result}"

    @given(days=st.integers(min_value=0, max_value=365))
    def test_decay_zero_initial_score(self, days: int) -> None:
        """Decay of 0 initial score should always be 0."""
        policy = ExponentialDecayPolicy()
        result = policy.compute_decay(0.0, days)

        assert result == 0.0

    @given(days=st.integers(min_value=0, max_value=365))
    def test_decay_full_initial_score(self, days: int) -> None:
        """Decay computation should work for full initial score."""
        policy = ExponentialDecayPolicy()
        result = policy.compute_decay(1.0, days)

        assert 0.0 <= result <= 1.0
        if days > 0:
            assert result < 1.0
