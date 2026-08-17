"""LLM-as-Judge — evaluates answer quality using an LLM.

Three scoring methods (all using OpenAI-compatible endpoint):
1. Token F1: word-level overlap (fast, no LLM needed)
2. Semantic similarity: embedding cosine (uses sentence-transformers)
3. LLM judge: another model rates the answer 1-5

The LLM judge is the gold standard for open-ended evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.judge.llm_client import LLMClient, get_judge_config

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge. Your job is to score how well a generated answer matches the expected answer.

Score on a scale of 1-5:
  5 = Perfect match — contains the correct information, possibly with additional context
  4 = Mostly correct — captures the key fact but may miss minor details
  3 = Partially correct — some relevant information but incomplete or slightly wrong
  2 = Mostly wrong — mentions related topics but misses the actual answer
  1 = Completely wrong — irrelevant or contradicts the expected answer

Respond with ONLY a single number (1-5). No explanation."""

JUDGE_USER_TEMPLATE = """Question: {question}

Expected answer: {expected}

Generated answer: {generated}

Score (1-5):"""


@dataclass(frozen=True)
class JudgeResult:
    """Result from the LLM judge evaluation."""

    score: float  # 1-5 scale (normalized to 0-1 for metrics)
    raw_score: int  # Original 1-5 integer
    question: str
    expected_answer: str
    generated_answer: str
    method: str  # "llm_judge", "token_f1", "semantic"


class AnswerJudge:
    """Evaluates generated answers against expected answers.

    Supports three methods:
    - token_f1: Fast word overlap (no LLM call)
    - semantic: Embedding cosine similarity
    - llm_judge: Full LLM evaluation (most accurate)
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialize with a judge LLM client.

        Args:
            client: LLM client for judge calls. If None, creates from env.
        """
        self._client = client or LLMClient(get_judge_config())

    def judge_with_llm(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
    ) -> JudgeResult:
        """Score using LLM-as-judge (most accurate, requires LLM endpoint).

        Args:
            question: The original query.
            expected_answer: The gold answer from the dataset.
            generated_answer: The answer produced from retrieved memories.

        Returns:
            JudgeResult with 1-5 score.
        """
        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            expected=expected_answer,
            generated=generated_answer,
        )

        response = self._client.generate(prompt=prompt, system=JUDGE_SYSTEM_PROMPT)

        # Parse score from response
        raw_score = self._parse_score(response)

        return JudgeResult(
            score=raw_score / 5.0,  # Normalize to 0-1
            raw_score=raw_score,
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            method="llm_judge",
        )

    @staticmethod
    def judge_with_token_f1(
        question: str,
        expected_answer: str,
        generated_answer: str,
    ) -> JudgeResult:
        """Score using token-level F1 overlap (fast, no LLM needed).

        F1 = 2 * precision * recall / (precision + recall)
        where precision = matched_tokens / generated_tokens
        and recall = matched_tokens / expected_tokens

        Args:
            question: The original query.
            expected_answer: The gold answer.
            generated_answer: The generated answer.

        Returns:
            JudgeResult with F1 score (0-1).
        """
        expected_tokens = set(expected_answer.lower().split())
        generated_tokens = set(generated_answer.lower().split())

        if not expected_tokens or not generated_tokens:
            return JudgeResult(
                score=0.0,
                raw_score=1,
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                method="token_f1",
            )

        matched = expected_tokens & generated_tokens
        precision = len(matched) / len(generated_tokens)
        recall = len(matched) / len(expected_tokens)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Map F1 to 1-5 scale
        raw_score = max(1, min(5, int(f1 * 5) + 1))

        return JudgeResult(
            score=f1,
            raw_score=raw_score,
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            method="token_f1",
        )

    @staticmethod
    def _parse_score(response: str) -> int:
        """Parse a 1-5 score from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Integer score 1-5.
        """
        # Try to find a digit 1-5 in the response
        for char in response.strip():
            if char.isdigit() and 1 <= int(char) <= 5:
                return int(char)
        # Default to 1 if unparseable
        return 1
