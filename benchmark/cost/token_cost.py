"""Token cost calculator for LLM API calls.

Computes costs based on token usage and model pricing.
"""

from __future__ import annotations

from benchmark.cost.models import CostEntry
from benchmark.models.answer import TokenUsage
from benchmark.tokenizer.interface import Tokenizer

# Fixed pricing table — non-configurable benchmark rule.
# Prices are per 1000 tokens in USD.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}

DEFAULT_PRICING: dict[str, float] = {"prompt": 0.01, "completion": 0.03}


class TokenCost:
    """Simple token cost calculator for testing.

    Computes cost based on token count and default pricing.
    """

    def __init__(self, rate_per_1k: float = 0.01) -> None:
        """Initialize token cost calculator.

        Args:
            rate_per_1k: Cost per 1000 tokens in USD.
        """
        self._rate_per_1k = rate_per_1k

    def compute(self, tokens: int) -> float:
        """Compute cost for a number of tokens.

        Args:
            tokens: Number of tokens.

        Returns:
            Cost in USD.
        """
        return (tokens / 1000.0) * self._rate_per_1k


class TokenCostCalculator:
    """Calculates cost of LLM API calls based on token usage.

    Uses a fixed pricing table for known models, with a fallback
    for unknown models.

    Optionally accepts a Tokenizer to compute token counts deterministically
    when TokenUsage is not explicitly provided.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer

    def compute_cost(
        self,
        token_usage: TokenUsage | None,
        model: str,
        prompt_text: str | None = None,
        completion_text: str | None = None,
    ) -> CostEntry:
        """Compute the cost for a given token usage and model.

        Args:
            token_usage: Optional TokenUsage. If not provided and a tokenizer is
                available, prompt_text and completion_text will be tokenized to
                produce deterministic counts.
            model: The model name (e.g., "gpt-4o").
            prompt_text: Optional prompt text to tokenize when token_usage is None.
            completion_text: Optional completion text to tokenize when token_usage is None.

        Returns:
            A CostEntry with the computed cost.
        """
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

        if token_usage is None:
            if self._tokenizer is None:
                raise ValueError(
                    "Either token_usage must be provided or a Tokenizer must be configured"
                )
            prompt_tokens = self._tokenizer.count_tokens(prompt_text or "")
            completion_tokens = self._tokenizer.count_tokens(completion_text or "")
        else:
            prompt_tokens = token_usage.prompt
            completion_tokens = token_usage.completion

        prompt_cost = (prompt_tokens / 1000.0) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000.0) * pricing["completion"]
        total = prompt_cost + completion_cost

        return CostEntry(
            source="llm_tokens",
            amount_usd=total,
            details={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_cost_usd": prompt_cost,
                "completion_cost_usd": completion_cost,
            },
        )
