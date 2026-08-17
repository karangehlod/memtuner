"""FollowUp accuracy evaluator.

Measures how well the memory system handles follow-up queries in conversations.
A follow-up query is marked as is_followup=True and references a prior conversation turn.

The evaluator scores whether follow-up queries successfully retrieve context
from the referenced prior turn.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
from benchmark.evaluation.context import EvaluationContext


class FollowUpAccuracyEvaluator(MetricEvaluator):
    """Evaluates follow-up query accuracy in conversations.

    For follow-up queries (is_followup=True), this evaluator measures whether
    the memory system successfully retrieved relevant context from the referenced
    prior conversation turn.

    A follow-up is considered successful if:
    1. It's marked as a follow-up
    2. It has expected results
    3. The retrieved IDs contain at least one expected ID

    For non-follow-up queries, this returns 1.0 (no evaluation needed).
    """

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Evaluate a single query's follow-up accuracy.

        This is the legacy interface and doesn't have access to is_followup flag.
        Use evaluate_with_context() for proper follow-up evaluation.

        Args:
            retrieved_ids: Memory IDs returned by memory system.
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            EvaluationResult with value 1.0 (no follow-up info in this interface).
        """
        return EvaluationResult(
            metric_name=self.metric_name(),
            value=1.0,
            query_count=0,
            details={"note": "Use evaluate_with_context for follow-up evaluation"},
        )

    def evaluate_with_context(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate follow-up query using rich context.

        Args:
            context: EvaluationContext with follow-up metadata.

        Returns:
            EvaluationResult with follow-up accuracy (0.0 or 1.0 for individual query).
        """
        # Non-follow-ups are not evaluated by this metric
        if not context.is_followup:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=1.0,
                query_count=0,
            )

        # Follow-up with no expected results is invalid
        if not context.expected_ids:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=0.0,
                query_count=1,
                details={
                    "error": "Follow-up with no expected results",
                    "references_turn": context.references_turn,
                },
            )

        # Check if any expected ID was retrieved
        retrieved_set = set(context.retrieved_ids)
        expected_set = set(context.expected_ids)
        has_relevant = bool(retrieved_set & expected_set)

        # Score: 1.0 if at least one expected memory retrieved, 0.0 otherwise
        score = 1.0 if has_relevant else 0.0

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=score,
            query_count=1,
            details={
                "references_turn": context.references_turn,
                "retrieved_count": len(context.retrieved_ids),
                "expected_count": len(context.expected_ids),
                "matched_count": len(retrieved_set & expected_set),
                "score": score,
            },
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.followup_accuracy"
