"""Contradiction resolution evaluator.

Measures how well the memory system handles contradictory memories.
Contradictions occur when multiple memories about the same topic
have conflicting content (e.g., different preferences).

This evaluator measures whether the system correctly:
1. Retrieves contradictory memory pairs
2. Prioritizes newer memories over older ones
3. Provides visibility into conflicting information
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
from benchmark.evaluation.context import EvaluationContext


class ContradictionResolutionEvaluator(MetricEvaluator):
    """Evaluates how well the system handles contradictory memories.

    Contradictions are scenarios where multiple expected results exist.
    This evaluator measures:
    1. Coverage: Were all expected memories retrieved? (no incomplete context)
    2. Ranking: Are they ranked in reverse chronological order? (newer first)
    3. Resolution: Was there a clear winner or ambiguity detected?

    A query with only one expected result has no contradictions (score=1.0).
    A query with multiple expected results is a contradiction scenario:
    - If all retrieved: score = 1.0 (user sees full context to decide)
    - If none retrieved: score = 0.0 (missing context)
    - If partial: score = 0.5 (incomplete context may cause confusion)
    """

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Evaluate contradiction handling for a single query.

        This legacy interface doesn't have access to temporal ordering.
        Use evaluate_with_context() for proper evaluation.

        Args:
            retrieved_ids: Memory IDs returned by memory system.
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            EvaluationResult with contradiction resolution score.
        """
        # No contradiction if only one or zero expected results
        if len(expected_ids) <= 1:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=1.0,
                query_count=0,
            )

        # For contradictions, measure coverage
        retrieved_set = set(retrieved_ids)
        expected_set = set(expected_ids)
        matched = retrieved_set & expected_set

        if len(matched) == 0:
            score = 0.0  # No context retrieved
        elif len(matched) == len(expected_set):
            score = 1.0  # Full context retrieved
        else:
            score = 0.5  # Partial context (may be confusing)

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=score,
            query_count=1,
            details={
                "expected_count": len(expected_ids),
                "retrieved_count": len(retrieved_ids),
                "matched_count": len(matched),
                "coverage": len(matched) / len(expected_set) if expected_set else 0.0,
            },
        )

    def evaluate_with_context(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate contradiction handling using rich context.

        Args:
            context: EvaluationContext with source module and creation day info.

        Returns:
            EvaluationResult with contradiction resolution score.
        """
        # No contradiction scenario if expected results is singular or empty
        if len(context.expected_ids) <= 1:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=1.0,
                query_count=0,
                details={"type": "no_contradiction"},
            )

        retrieved_set = set(context.retrieved_ids)
        expected_set = set(context.expected_ids)
        matched = retrieved_set & expected_set

        # Base coverage score
        if len(matched) == 0:
            coverage_score = 0.0
            coverage_type = "no_retrieval"
        elif len(matched) == len(expected_set):
            coverage_score = 1.0
            coverage_type = "full_coverage"
        else:
            coverage_score = 0.5
            coverage_type = "partial_coverage"

        # Check ordering: newer memories should be retrieved first.
        # When nothing was retrieved, ordering is undefined — score 0.0.
        # Use the ORDER of retrieved_ids (not arbitrary set order) to check ordering.
        ordering_score = 0.0 if len(matched) == 0 else 1.0
        ordering_details: dict[str, object] = {}

        if len(matched) > 1:
            # Preserve retrieval order: iterate retrieved_ids in sequence
            matched_in_order = [mid for mid in context.retrieved_ids if mid in matched]
            creation_days = [
                context.retrieved_creation_days.get(mid, 0) for mid in matched_in_order
            ]

            # Check if ordered by descending creation day (newer first)
            is_properly_ordered = all(
                creation_days[i] >= creation_days[i + 1] for i in range(len(creation_days) - 1)
            )

            if not is_properly_ordered:
                ordering_score = 0.5
                ordering_details["warning"] = "Memories not in reverse chronological order"

        # Final score: 70% coverage + 30% ordering.
        # When coverage is 0 this naturally gives 0.0 (ordering is also 0 then).
        final_score = 0.7 * coverage_score + 0.3 * ordering_score

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=final_score,
            query_count=1,
            details={
                "type": "contradiction",
                "contradiction_count": len(expected_set),
                "coverage_score": coverage_score,
                "coverage_type": coverage_type,
                "ordering_score": ordering_score,
                "final_score": final_score,
                **ordering_details,
            },
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.contradiction_resolution"
