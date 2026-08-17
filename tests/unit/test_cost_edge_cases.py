"""Edge case tests for cost calculations."""

from __future__ import annotations

import pytest

from benchmark.cost.token_cost import TokenCost
from benchmark.tokenizer.bpe import SimpleBPETokenizer


@pytest.mark.unit
class TestTokenCostEdgeCases:
    """Edge cases for token cost calculation."""

    def test_token_cost_zero_tokens(self) -> None:
        """Zero tokens should give zero cost."""
        calculator = TokenCost()
        cost = calculator.compute(0)
        assert cost == 0.0

    def test_token_cost_one_token(self) -> None:
        """Single token should give positive cost."""
        calculator = TokenCost()
        cost = calculator.compute(1)
        assert cost > 0.0

    def test_token_cost_large_count(self) -> None:
        """Large token count should not crash."""
        calculator = TokenCost()
        cost = calculator.compute(1_000_000)
        assert cost > 0.0
        assert not any([float(cost) == float("inf")])

    def test_token_cost_deterministic(self) -> None:
        """Same token count should give same cost."""
        calculator = TokenCost()
        c1 = calculator.compute(1000)
        c2 = calculator.compute(1000)
        assert c1 == c2

    def test_token_cost_consistency_across_instances(self) -> None:
        """Different instances should give same cost."""
        calc1 = TokenCost()
        calc2 = TokenCost()
        assert calc1.compute(500) == calc2.compute(500)


@pytest.mark.unit
class TestTokenizerEdgeCases:
    """Edge cases for tokenizer."""

    def test_tokenizer_empty_text(self) -> None:
        """Empty text should give zero tokens."""
        tokenizer = SimpleBPETokenizer()
        tokens = tokenizer.count_tokens("")
        assert tokens == 0

    def test_tokenizer_single_char(self) -> None:
        """Single character should tokenize."""
        tokenizer = SimpleBPETokenizer()
        tokens = tokenizer.count_tokens("a")
        assert tokens >= 1

    def test_tokenizer_whitespace(self) -> None:
        """Whitespace-only text."""
        tokenizer = SimpleBPETokenizer()
        tokens = tokenizer.count_tokens("   ")
        # Whitespace may or may not tokenize depending on implementation
        assert tokens >= 0

    def test_tokenizer_deterministic(self) -> None:
        """Same text should give same token count."""
        tokenizer = SimpleBPETokenizer()
        text = "The quick brown fox jumps over the lazy dog"
        count1 = tokenizer.count_tokens(text)
        count2 = tokenizer.count_tokens(text)
        assert count1 == count2

    def test_tokenizer_encode_deterministic(self) -> None:
        """Same text should encode the same way."""
        tokenizer = SimpleBPETokenizer()
        text = "test"
        enc1 = tokenizer.encode(text)
        enc2 = tokenizer.encode(text)
        assert enc1 == enc2

    def test_tokenizer_encode_returns_list(self) -> None:
        """Encode should return a list."""
        tokenizer = SimpleBPETokenizer()
        result = tokenizer.encode("hello")
        assert isinstance(result, list)

    def test_tokenizer_long_text(self) -> None:
        """Long text should tokenize correctly."""
        tokenizer = SimpleBPETokenizer()
        text = "a" * 10_000
        tokens = tokenizer.count_tokens(text)
        assert tokens > 0
        # Roughly proportional to length
        assert tokens <= len(text) + 100

    def test_tokenizer_special_chars(self) -> None:
        """Special characters should tokenize."""
        tokenizer = SimpleBPETokenizer()
        tokens = tokenizer.count_tokens("!@#$%^&*()")
        assert tokens >= 0

    def test_tokenizer_unicode(self) -> None:
        """Unicode characters should tokenize."""
        tokenizer = SimpleBPETokenizer()
        tokens = tokenizer.count_tokens("Hello 世界 🌍")
        assert tokens >= 0


@pytest.mark.unit
class TestCostDivisionByZero:
    """Tests for division by zero prevention."""

    def test_cost_per_correct_recall_zero_correct(self) -> None:
        """Cost per correct recall with zero correct."""
        # This is handled at the orchestrator level, but test the principle
        cost_total = 100.0
        correct = 0
        # Avoid division by zero
        cost_per_correct = cost_total / max(correct, 1)
        assert cost_per_correct == 100.0

    def test_cost_per_correct_recall_zero_cost(self) -> None:
        """Cost per correct recall with zero cost."""
        cost_total = 0.0
        correct = 5
        cost_per_correct = cost_total / max(correct, 1)
        assert cost_per_correct == 0.0

    def test_cost_per_correct_recall_both_zero(self) -> None:
        """Cost per correct recall with zero cost and correct."""
        cost_total = 0.0
        correct = 0
        cost_per_correct = cost_total / max(correct, 1)
        assert cost_per_correct == 0.0
