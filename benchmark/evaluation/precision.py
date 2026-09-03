"""Standard IR Precision@K evaluator for the MemTuner pipeline.

Precision@K = |retrieved[:K] ∩ gold| / K

WHERE WIRED INTO THE PIPELINE
------------------------------
Instantiated by benchmark/application/composer.py _build_evaluators() ~line 367.
Called per-query inside benchmark/orchestrator/scenario_runner.py
_execute_queries_sequential() ~line 453.
Results aggregated (macro-averaged) in _build_scenario_metrics() ~line 790.

PAIRED UTILITY (same formula, adapter self-evaluation path)
-----------------------------------------------------------
benchmark/retrieval/metrics_utils.py :: compute_precision_at_k()
Difference: compute_precision_at_k() accepts list[dict] with 'doc_id' keys
and an explicit k parameter; StandardPrecisionEvaluator accepts list[str]
and uses self._top_k set at construction time.

See also: PrecisionAtKEvaluator in benchmark/evaluation/ranking.py, which
implements the same formula in the ranking-aware evaluation path.

FORMULA
-------
Precision@K = |retrieved[:K] ∩ gold| / K

Denominator is always K (the configured cutoff), never the actual returned
count.  This is the standard IR definition (Manning et al. §8.3) and makes
scores comparable across systems with different fallback behaviours.

RANGE: [0.0, 1.0].  Higher is better.
Maximum achievable = min(|gold|, K) / K.

EDGE CASES
----------
- retrieved_ids empty   -> 0.0 (nothing returned, nothing relevant)
- expected_ids empty    -> 0.0 (no gold set; all retrieved items are not-relevant)
- len(retrieved_ids) < K -> missing slots count as not-relevant; denominator is K
- duplicates in retrieved_ids -> set() deduplication; a memory returned twice
                                  occupies one rank slot, credited at most once
- top_k < 1             -> raises ValueError at construction time

REFERENCE: Manning, Raghavan & Schutze, "Introduction to Information Retrieval",
           Cambridge University Press, 2008.  Precision: §8.3.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator


class StandardPrecisionEvaluator(MetricEvaluator):
    """Computes standard IR Precision@K for a single (retrieved_ids, expected_ids) pair.

    Implements Precision@K as defined in Manning, Raghavan & Schutze (2008) §8.3.

    FORMULA
    -------
    Precision@K = |retrieved[:K] ∩ gold| / K

    RANGE: [0.0, 1.0].  Higher is better.
    Maximum achievable = min(|gold|, K) / K.

    EDGE CASES
    ----------
    - retrieved_ids is empty          -> returns 0.0 immediately (early exit)
    - expected_ids is empty           -> gold_set is empty; precision = 0.0
    - len(retrieved_ids) < K          -> missing slots are not-relevant; denominator is K
    - duplicate IDs in retrieved_ids  -> set intersection deduplicates; a memory
                                         returned twice is credited at most once
    - top_k < 1                       -> raises ValueError at __init__ time

    METRIC KEY
    ----------
    metric_name() -> "benchmark.precision_at_k"
    ScenarioRunner looks up results under this key in _build_scenario_metrics().

    PAIRED UTILITY (same formula, adapter self-evaluation path)
    -----------------------------------------------------------
    benchmark/retrieval/metrics_utils.py :: compute_precision_at_k()

    See also: PrecisionAtKEvaluator in benchmark/evaluation/ranking.py, which
    implements the same formula in the ranking-aware evaluation path.
    """

    def __init__(self, top_k: int = 5) -> None:
        """Initialize with the K value for evaluation.

        Args:
            top_k: Number of top results to consider. Must be >= 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self._top_k = top_k

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Compute Precision@K for a single query.

        Args:
            retrieved_ids: Memory IDs returned by memory system (ordered by score).
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            EvaluationResult with precision value between 0.0 and 1.0.
        """
        if not retrieved_ids:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=0.0,
                query_count=1,
                details={
                    "relevant_in_top_k": 0,
                    "k": self._top_k,
                    "gold_set_size": len(expected_ids),
                    "max_achievable": 0.0,
                },
            )

        # Deduplicate: a memory returned twice still occupies one rank slot.
        top_k_retrieved = set(retrieved_ids[: self._top_k])
        gold_set = set(expected_ids)
        relevant_in_top_k = len(top_k_retrieved & gold_set)

        # Denominator is always K — not the actual returned count.
        # This penalises systems that return fewer than K results.
        precision = relevant_in_top_k / self._top_k

        # Ceiling: even a perfect system can only reach min(|gold|,K)/K
        # when the gold set is smaller than K.
        max_achievable = min(len(gold_set), self._top_k) / self._top_k

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=precision,
            query_count=1,
            details={
                "relevant_in_top_k": relevant_in_top_k,
                "k": self._top_k,
                "gold_set_size": len(gold_set),
                "max_achievable": max_achievable,
            },
        )

    def metric_name(self) -> str:
        """Return the metric identifier.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.precision_at_k"
