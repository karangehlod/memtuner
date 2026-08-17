"""False positive rate evaluator.

Computes two metrics:
1. Contamination Rate = |Retrieved \\ GoldSet| / |Retrieved|
   - What fraction of retrieved results are irrelevant (hallucination fuel)
   - This is actually 1 - Precision@K
   - With sparse gold (avg |gold|=1.42), even perfect recall gives ~0.86

2. Statistical FPR = |Retrieved \\ GoldSet| / (|Corpus| - |GoldSet|)
   - What fraction of the irrelevant corpus was incorrectly retrieved
   - Proper IR false positive rate
   - Comparable across different k values

Standard Precision@K is computed separately by StandardPrecisionEvaluator.
For agentic systems: contamination rate measures hallucination risk per query.
Statistical FPR measures how selective the retrieval is overall.
"""

from __future__ import annotations

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator


class FalsePositiveEvaluator(MetricEvaluator):
    """Computes contamination rate for retrieved results.

    Uses contamination rate formula for backward compatibility with existing
    benchmark results, but stores proper statistical FPR in details.

    Contamination Rate = |Retrieved \\ GoldSet| / |Retrieved|

    This tells you: "of the k memories you'll feed to the LLM, what fraction
    is noise?" — directly measures hallucination risk.
    """

    def __init__(self, corpus_size: int = 5879) -> None:
        """Initialize with corpus size for statistical FPR.

        Args:
            corpus_size: Total number of memories in the corpus.
                Used to compute proper statistical FPR.
                Default 5879 for LoCoMo dataset.
        """
        self._corpus_size = corpus_size

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Compute false positive rate for a single query.

        Args:
            retrieved_ids: Memory IDs returned by memory system.
            expected_ids: Memory IDs expected from gold dataset.

        Returns:
            EvaluationResult with contamination rate value between 0.0 and 1.0.
            Details include statistical FPR and precision@k.
        """
        if not retrieved_ids:
            return EvaluationResult(
                metric_name=self.metric_name(),
                value=0.0,
                query_count=1,
                details={
                    "false_positives": 0,
                    "total_retrieved": 0,
                    "precision_at_k": 0.0,
                    "statistical_fpr": 0.0,
                    "corpus_size": self._corpus_size,
                    "gold_set_size": len(expected_ids) if expected_ids else 0,
                },
            )

        retrieved_set = set(retrieved_ids)
        gold_set = set(expected_ids)
        false_positives = retrieved_set - gold_set
        fp_count = len(false_positives)
        k = len(retrieved_set)

        # Contamination rate (backward-compatible "FPR")
        contamination_rate = fp_count / k

        # Keep a diagnostic precision value for backward-compatible details.
        # Scenario aggregation uses StandardPrecisionEvaluator independently.
        precision_at_k = len(retrieved_set & gold_set) / k

        # Statistical FPR: FP / (total negatives in corpus)
        total_negatives = max(self._corpus_size - len(gold_set), 1)
        statistical_fpr = fp_count / total_negatives

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=contamination_rate,
            query_count=1,
            details={
                "false_positives": fp_count,
                "total_retrieved": k,
                "precision_at_k": precision_at_k,
                "statistical_fpr": statistical_fpr,
                "corpus_size": self._corpus_size,
                "gold_set_size": len(gold_set),
            },
        )

    def metric_name(self) -> str:
        """Return the fixed metric name.

        Returns:
            The OTel-compatible metric name.
        """
        return "benchmark.contamination_rate"
