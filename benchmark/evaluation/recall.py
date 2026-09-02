"""Recall@K evaluator for the MemTuner benchmark pipeline.

Recall@K = |retrieved[:K] ∩ gold| / |gold|

WHERE WIRED INTO THE PIPELINE
------------------------------
Instantiated by benchmark/application/composer.py _build_evaluators() ~line 367.
Called per-query inside benchmark/orchestrator/scenario_runner.py
_execute_queries_sequential() ~line 453.
Results aggregated (macro-averaged) in _build_scenario_metrics() ~line 790.

PAIRED UTILITY (same formula, adapter self-evaluation path)
-----------------------------------------------------------
benchmark/retrieval/metrics_utils.py :: compute_recall_at_k()
Difference: compute_recall_at_k() accepts list[dict] with 'doc_id' keys
and an explicit k parameter; RecallEvaluator accepts list[str] and uses
self._top_k set at construction time.

FORMULA
-------
Recall@K = |retrieved[:K] ∩ gold| / |gold|

Denominator is always |gold| (total known-relevant items), never K.
This measures evidence *coverage*: what fraction of all relevant memories
were surfaced within the top-K window?

RANGE: [0.0, 1.0].  Higher is better.

EDGE CASES
----------
- retrieved_ids empty  -> 0.0 (numerator is zero, nothing retrieved)
- expected_ids empty   -> raises ValueError (data-quality guard; vacuous 1.0
                           would silently inflate aggregate recall)
- duplicates in retrieved_ids -> set() deduplication prevents recall > 1.0
- |gold| > K           -> perfect recall impossible unless all gold items fall
                           in top-K; maximum achievable = K / |gold|

REFERENCE: Manning, Raghavan & Schutze, "Introduction to Information Retrieval",
           Cambridge University Press, 2008.  Recall: §8.3.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator


class RecallEvaluator(MetricEvaluator):
    """Computes Recall@K for a single (retrieved_ids, expected_ids) pair.

    Implements Recall@K as defined in Manning, Raghavan & Schutze (2008) §8.3.

    FORMULA
    -------
    Recall@K = |retrieved[:K] ∩ gold| / |gold|

    RANGE: [0.0, 1.0].  Higher is better.
    1.0 = all gold memories found in top-K.

    EDGE CASES
    ----------
    - retrieved_ids is empty          -> recall = 0.0
    - expected_ids is empty           -> raises ValueError (data quality guard)
    - duplicate IDs in retrieved_ids  -> set conversion deduplicates naturally;
                                         recall cannot exceed 1.0
    - |gold| > top_k                  -> perfect recall unreachable;
                                         maximum = top_k / |gold|

    METRIC KEY
    ----------
    metric_name() -> "benchmark.recall_at_k"
    ScenarioRunner looks up results under this key in _build_scenario_metrics().

    PAIRED UTILITY (same formula, adapter self-evaluation path)
    -----------------------------------------------------------
    benchmark/retrieval/metrics_utils.py :: compute_recall_at_k()
    Difference: compute_recall_at_k() accepts list[dict] with 'doc_id' keys
    and an explicit k parameter; this evaluator accepts list[str] directly.
    """

    def __init__(self, top_k: int = 5) -> None:
        """Initialize with the K value for top-K evaluation.

        Args:
            top_k: Maximum number of results to consider.
        """
        self._top_k = top_k

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Compute Recall@K for a single query.

        Args:
            retrieved_ids: Memory IDs returned by memory system (ordered by score).
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            EvaluationResult with recall value between 0.0 and 1.0.
        """
        if not expected_ids:
            # Empty gold set = data quality problem, not a valid query.
            # Returning 1.0 would silently inflate recall scores.
            raise ValueError(
                "RecallEvaluator: expected_ids cannot be empty. "
                "Every query must have at least one expected memory in gold dataset."
            )

        # Truncate to top-K and convert to set for O(1) intersection.
        # Duplicates in retrieved_ids are naturally deduplicated here —
        # the same memory ID counted twice is still one retrieved item.
        top_k_retrieved = set(retrieved_ids[: self._top_k])
        gold_set = set(expected_ids)
        relevant_retrieved = top_k_retrieved & gold_set

        # Denominator is |gold| not K: recall measures coverage of evidence,
        # not density of the result list.
        recall_value = len(relevant_retrieved) / len(gold_set)

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=recall_value,
            query_count=1,
            details={
                "retrieved_relevant": len(relevant_retrieved),
                "gold_set_size": len(gold_set),
                "top_k": self._top_k,
            },
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.recall_at_k"
