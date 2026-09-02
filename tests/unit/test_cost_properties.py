"""Property-based tests for cost calculations using Hypothesis.

Tests invariants about cost:
- Cost is always non-negative
- Cost is monotonically non-decreasing with token count
- Cost never NaN or infinite
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmark.cost.token_cost import TokenCost
from benchmark.tokenizer.bpe import SimpleBPETokenizer


@pytest.mark.unit
class TestTokenCostProperties:
    """Property-based tests for token cost calculation."""

    @given(tokens=st.integers(min_value=0, max_value=1_000_000))
    def test_cost_always_nonnegative(self, tokens: int) -> None:
        """Cost must always be non-negative."""
        cost_calculator = TokenCost()
        cost = cost_calculator.compute(tokens)

        assert cost >= 0.0
        assert not any(
            [float(cost) != float(cost)]
        ), "Cost should not be NaN"

    @given(tokens=st.integers(min_value=0, max_value=10_000))
    def test_cost_not_infinite(self, tokens: int) -> None:
        """Cost must never be infinite."""
        cost_calculator = TokenCost()
        cost = cost_calculator.compute(tokens)

        assert not any([float(cost) == float("inf")])
        assert not any([float(cost) == float("-inf")])

    @given(
        tokens1=st.integers(min_value=0, max_value=5000),
        tokens2=st.integers(min_value=5000, max_value=10_000),
    )
    def test_cost_monotonically_nondecreasing(
        self, tokens1: int, tokens2: int
    ) -> None:
        """Cost should be non-decreasing as token count increases."""
        cost_calculator = TokenCost()
        cost1 = cost_calculator.compute(tokens1)
        cost2 = cost_calculator.compute(tokens2)

        assert cost2 >= cost1 - 1e-9

    @given(text=st.text(max_size=1000))
    def test_tokenizer_cost_consistent(self, text: str) -> None:
        """Token cost should be consistent for same text."""
        tokenizer = SimpleBPETokenizer()
        token_count = tokenizer.count_tokens(text)

        cost_calculator = TokenCost()
        cost1 = cost_calculator.compute(token_count)
        cost2 = cost_calculator.compute(token_count)

        assert cost1 == cost2

    def test_zero_tokens_zero_cost(self) -> None:
        """Cost of zero tokens should be zero."""
        cost_calculator = TokenCost()
        cost = cost_calculator.compute(0)
        assert cost == 0.0

    @given(tokens=st.integers(min_value=1, max_value=100_000))
    def test_cost_proportional_to_tokens(self, tokens: int) -> None:
        """Cost should scale proportionally (roughly) with token count."""
        cost_calculator = TokenCost()

        cost_1 = cost_calculator.compute(1)
        cost_tokens = cost_calculator.compute(tokens)

        # Cost should be roughly: cost_1 * tokens
        # Allow 20% tolerance for rounding
        if cost_1 > 0:
            ratio = cost_tokens / cost_1
            assert abs(ratio - tokens) / tokens < 0.2
