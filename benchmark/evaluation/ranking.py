"""Ranking-aware metric evaluators used by ScenarioRunner.

These evaluators receive per-query (retrieved_ids, expected_ids) pairs from
ScenarioRunner._execute_queries_sequential() and emit EvaluationResult objects
that are collected and macro-averaged in _build_scenario_metrics().

FORMULA QUICK-REFERENCE
-----------------------
MRR@K    = 1 / rank_of_first_relevant   (0 if no hit in top-K)
NDCG@K   = DCG@K / IDCG@K
           DCG@K  = sum_{i=1}^{K} rel_i / log2(i+1)   rel_i in {0,1}
           IDCG@K = sum_{i=1}^{min(|gold|,K)} 1 / log2(i+1)
P@K      = |retrieved[:K] ∩ gold| / K   denominator always K
Recall@K = |retrieved[:K] ∩ gold| / |gold|

DEDUPLICATION RULE (applies to all evaluators):
  A memory ID appearing more than once in retrieved_ids occupies a rank slot
  but is credited at most once toward any metric. Without dedup, DCG can exceed
  IDCG giving NDCG > 1.0, and recall can exceed 1.0.

WHERE THESE ARE WIRED:
  benchmark/application/composer.py  _build_evaluators()  line ~367
  benchmark/orchestrator/scenario_runner.py  _execute_queries_sequential()  line ~453
  benchmark/orchestrator/scenario_runner.py  _build_scenario_metrics()  line ~790

PAIRED UTILITIES (same formulas, used by adapter self-evaluation):
  benchmark/retrieval/metrics_utils.py  compute_mrr()
  benchmark/retrieval/metrics_utils.py  compute_ndcg()
  benchmark/retrieval/metrics_utils.py  compute_recall_at_k()
  benchmark/retrieval/metrics_utils.py  compute_precision_at_k()

REFERENCE: Manning, Raghavan & Schutze, "Introduction to Information Retrieval",
           Cambridge University Press, 2008.
           MRR: §8.4  NDCG: §8.4  P@K: §8.3  Recall: §8.3
"""

from __future__ import annotations

import math

from benchmark.evaluation.base import EvaluationResult, MetricEvaluator


class MRREvaluator(MetricEvaluator):
    """Mean Reciprocal Rank evaluator (MRR@K).

    Implements MRR as defined in Manning, Raghavan & Schutze (2008) §8.4.

    FORMULA
    -------
    MRR@K = 1 / rank_of_first_relevant   (0 if no hit in top-K)

    where rank_of_first_relevant is the 1-based position of the first
    retrieved ID that appears in the gold set, within the top-K window.

    RANGE: [0.0, 1.0].  1.0 = first result is relevant; 0.0 = no hit in top-K.
    Higher is better.

    EDGE CASES
    ----------
    - retrieved_ids is empty            -> returns 0.0 (no hit by definition)
    - no relevant ID in top-K window    -> returns 0.0
    - expected_ids is empty             -> raises ValueError (data quality guard)
    - duplicate IDs in retrieved_ids    -> only the first occurrence is checked;
                                          duplicates at higher ranks are simply
                                          not credited (first-hit semantics)

    METRIC KEY
    ----------
    metric_name() -> "benchmark.mrr"
    ScenarioRunner looks up results under this key in _build_scenario_metrics().

    PAIRED UTILITY (same formula, adapter self-evaluation path)
    -----------------------------------------------------------
    benchmark/retrieval/metrics_utils.py :: compute_mrr()
    Difference: compute_mrr() accepts list[dict] with 'doc_id' keys;
    this evaluator accepts list[str] directly.

    PIPELINE WIRING
    ---------------
    Instantiated by benchmark/application/composer.py _build_evaluators() ~line 367.
    Called per-query inside benchmark/orchestrator/scenario_runner.py
    _execute_queries_sequential() ~line 453.
    Results aggregated (macro-averaged) in _build_scenario_metrics() ~line 790.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize MRR evaluator.

        Args:
            top_k: Maximum rank to consider.
        """
        self._top_k = top_k

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Compute MRR for a single query.

        Args:
            retrieved_ids: Ordered list of retrieved memory IDs (by score, descending).
            expected_ids: Expected memory IDs from gold dataset.

        Returns:
            EvaluationResult with MRR value between 0.0 and 1.0.
        """
        if not expected_ids:
            raise ValueError(
                "MRREvaluator: expected_ids cannot be empty. "
                "Every query must have at least one expected memory."
            )

        gold_set = set(expected_ids)
        # Slice before iterating so rank numbers are 1-indexed within the window.
        top_k_retrieved = retrieved_ids[: self._top_k]

        reciprocal_rank = 0.0
        first_relevant_rank = None  # None = "not found in top-K", not a numeric rank
        for rank, memory_id in enumerate(top_k_retrieved, start=1):
            if memory_id in gold_set:
                # RR = 1/rank: rank 1 → 1.0, rank 2 → 0.5, rank 10 → 0.1
                reciprocal_rank = 1.0 / rank
                first_relevant_rank = rank
                break
        # MRR = 0.0 when no relevant result appears in the top-K window.
        # The mean over queries is computed by the aggregator, not here.

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=reciprocal_rank,
            query_count=1,
            details={
                "first_relevant_rank": first_relevant_rank,  # None if not found
                "top_k": self._top_k,
                "gold_set_size": len(gold_set),
            },
        )

    def metric_name(self) -> str:
        return "benchmark.mrr"


class NDCGEvaluator(MetricEvaluator):
    """Normalized Discounted Cumulative Gain evaluator (NDCG@K).

    Implements NDCG as defined in Manning, Raghavan & Schutze (2008) §8.4.

    FORMULA
    -------
    DCG@K  = sum_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = sum_{i=1}^{min(|gold|,K)} 1 / log2(i+1)
    NDCG@K = DCG@K / IDCG@K

    where rel_i in {0, 1}: 1 if retrieved[i-1] is in the gold set, else 0.
    Discount at rank 1: 1/log2(2) = 1.000
    Discount at rank 2: 1/log2(3) ~= 0.631
    Discount at rank 3: 1/log2(4) = 0.500

    RANGE: [0.0, 1.0].  1.0 = perfect ranking; 0.0 = no relevant result in top-K.
    Higher is better.

    EDGE CASES
    ----------
    - retrieved_ids is empty          -> DCG = 0.0, NDCG = 0.0
    - expected_ids is empty           -> raises ValueError (data quality guard)
    - top_k = 0                       -> IDCG = 0.0, NDCG = 0.0 by convention
    - duplicate IDs in retrieved_ids  -> a relevant ID appearing more than once is
                                         credited only at its first occurrence;
                                         without dedup DCG can exceed IDCG and
                                         NDCG > 1.0 would result
    - |gold| > K                      -> IDCG uses min(|gold|, K) ideal positions;
                                         perfect recall within K still gives NDCG = 1.0

    METRIC KEY
    ----------
    metric_name() -> "benchmark.ndcg"
    ScenarioRunner looks up results under this key in _build_scenario_metrics().

    PAIRED UTILITY (same formula, adapter self-evaluation path)
    -----------------------------------------------------------
    benchmark/retrieval/metrics_utils.py :: compute_ndcg()
    Difference: compute_ndcg() accepts list[dict] with 'doc_id' keys and an
    explicit k parameter; this evaluator accepts list[str] and uses self._top_k.

    PIPELINE WIRING
    ---------------
    Instantiated by benchmark/application/composer.py _build_evaluators() ~line 367.
    Called per-query inside benchmark/orchestrator/scenario_runner.py
    _execute_queries_sequential() ~line 453.
    Results aggregated (macro-averaged) in _build_scenario_metrics() ~line 790.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize NDCG evaluator.

        Args:
            top_k: Maximum rank to consider.
        """
        self._top_k = top_k
        # Precompute IDCG for every n in [0, top_k] — avoids top_k log2 calls per query.
        # idcg_table[n] = sum(1/log2(i+2) for i in range(n))
        self._idcg_table = [0.0] * (top_k + 1)
        for n in range(1, top_k + 1):
            self._idcg_table[n] = self._idcg_table[n - 1] + 1.0 / math.log2(n + 1)

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Compute NDCG@K for a single query.

        Args:
            retrieved_ids: Ordered list of retrieved memory IDs.
            expected_ids: Expected memory IDs from gold dataset.

        Returns:
            EvaluationResult with NDCG value between 0.0 and 1.0.
        """
        if not expected_ids:
            raise ValueError(
                "NDCGEvaluator: expected_ids cannot be empty. "
                "Every query must have at least one expected memory."
            )

        gold_set = set(expected_ids)
        top_k_retrieved = retrieved_ids[: self._top_k]

        # DCG: sum gain/discount for each relevant item in the ranked list.
        # Gain is binary (1 = relevant, 0 = not relevant).
        # Discount at 1-indexed rank r = 1 / log2(r + 1):
        #   rank 1 → 1/log2(2) = 1.000
        #   rank 2 → 1/log2(3) ≈ 0.631
        #   rank 3 → 1/log2(4) = 0.500
        # Deduplicate: a relevant doc appearing twice should only be credited once.
        # Without dedup, DCG can exceed IDCG → NDCG > 1.0.
        dcg = 0.0
        _seen_ndcg: set[str] = set()
        for rank, memory_id in enumerate(top_k_retrieved, start=1):
            if memory_id in gold_set and memory_id not in _seen_ndcg:
                dcg += 1.0 / math.log2(rank + 1)
                _seen_ndcg.add(memory_id)

        # IDCG: ideal DCG — O(1) lookup in precomputed table built in __init__.
        num_relevant = min(len(gold_set), self._top_k)
        idcg = self._idcg_table[num_relevant]

        # NDCG = DCG / IDCG. When idcg=0 (top_k=0), return 0 by convention.
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=ndcg,
            query_count=1,
            details={
                "dcg": round(dcg, 4),
                "idcg": round(idcg, 4),
                "top_k": self._top_k,
                "relevant_in_top_k": sum(1 for mid in top_k_retrieved if mid in gold_set),
            },
        )

    def metric_name(self) -> str:
        return "benchmark.ndcg"


class PrecisionAtKEvaluator(MetricEvaluator):
    """Precision@K evaluator (ranking-context variant).

    Implements P@K as defined in Manning, Raghavan & Schutze (2008) §8.3.

    FORMULA
    -------
    P@K = |retrieved[:K] ∩ gold| / K

    Denominator is always K regardless of how many items were actually
    returned.  This matches StandardPrecisionEvaluator in precision.py and
    compute_precision_at_k() in metrics_utils.py.

    At K=1 this reduces to Precision@1:
        P@1 = 1.0 if retrieved[0] in gold else 0.0

    RANGE: [0.0, 1.0].  Higher is better.
    Maximum achievable = min(|gold|, K) / K.

    EDGE CASES
    ----------
    - retrieved_ids is empty          -> precision = 0.0
    - expected_ids is empty           -> raises ValueError (data quality guard)
    - len(retrieved_ids) < K          -> missing slots treated as not-relevant;
                                         denominator is still K (system is penalised)
    - duplicate IDs in retrieved_ids  -> set intersection deduplicates; a memory
                                         returned twice occupies one rank slot and
                                         is credited at most once

    METRIC KEY
    ----------
    metric_name() -> f"benchmark.precision_at_{self._top_k}"
    e.g. "benchmark.precision_at_1" for top_k=1.
    ScenarioRunner looks up results under this key in _build_scenario_metrics().

    PAIRED UTILITY (same formula, adapter self-evaluation path)
    -----------------------------------------------------------
    benchmark/retrieval/metrics_utils.py :: compute_precision_at_k()
    Difference: compute_precision_at_k() accepts list[dict] with 'doc_id'
    keys; this evaluator accepts list[str] directly.

    PIPELINE WIRING
    ---------------
    Instantiated by benchmark/application/composer.py _build_evaluators() ~line 367.
    Called per-query inside benchmark/orchestrator/scenario_runner.py
    _execute_queries_sequential() ~line 453.
    Results aggregated (macro-averaged) in _build_scenario_metrics() ~line 790.
    """

    def __init__(self, top_k: int = 1) -> None:
        """Initialize Precision@K evaluator.

        Args:
            top_k: The K value. Defaults to 1 for Precision@1.
        """
        self._top_k = top_k

    def evaluate(
        self,
        retrieved_ids: list[str],
        expected_ids: list[str],
    ) -> EvaluationResult:
        """Compute Precision@K for a single query.

        Args:
            retrieved_ids: Ordered list of retrieved memory IDs.
            expected_ids: Expected memory IDs from gold dataset.

        Returns:
            EvaluationResult with precision value between 0.0 and 1.0.
        """
        if not expected_ids:
            raise ValueError(
                "PrecisionAtKEvaluator: expected_ids cannot be empty. "
                "Every query must have at least one expected memory."
            )

        gold_set = set(expected_ids)
        top_k_retrieved = retrieved_ids[: self._top_k]

        if not top_k_retrieved:
            precision = 0.0
        else:
            # Use set intersection — a duplicate ID occupies one rank slot and
            # should not be credited twice. Denominator is always K.
            relevant_count = len(set(top_k_retrieved) & gold_set)
            precision = relevant_count / self._top_k

        return EvaluationResult(
            metric_name=self.metric_name(),
            value=precision,
            query_count=1,
            details={
                "relevant_in_top_k": sum(1 for mid in top_k_retrieved if mid in gold_set),
                "returned_count": len(top_k_retrieved),
                "top_k": self._top_k,
            },
        )

    def metric_name(self) -> str:
        return f"benchmark.precision_at_{self._top_k}"


class RecallAtKEvaluator(MetricEvaluator):
    """Recall@K evaluator — ranking-context variant that rejects empty gold sets.

    Implements Recall@K as defined in Manning, Raghavan & Schutze (2008) §8.3.

    FORMULA
    -------
    Recall@K = |retrieved[:K] ∩ gold| / |gold|

    Denominator is always |gold| (total relevant items), never K.
    This means Recall@K measures coverage: what fraction of all known
    relevant memories were found within the top-K results.

    RANGE: [0.0, 1.0].  Higher is better.
    1.0 = every gold memory was found in top-K.

    EDGE CASES
    ----------
    - retrieved_ids is empty          -> recall = 0.0
    - expected_ids is empty           -> raises ValueError (data quality guard;
                                         same behaviour as RecallEvaluator in recall.py)
    - duplicate IDs in retrieved_ids  -> set conversion deduplicates; a memory
                                         returned twice is still counted once
                                         (prevents recall > 1.0)
    - |gold| > K                      -> perfect recall within K impossible unless
                                         |gold| <= K; maximum = K / |gold|

    METRIC KEY
    ----------
    metric_name() -> f"benchmark.recall_at_{self._top_k}"
    e.g. "benchmark.recall_at_10" for top_k=10.
    ScenarioRunner looks up results under this key in _build_scenario_metrics().

    PAIRED UTILITY (same formula, adapter self-evaluation path)
    -----------------------------------------------------------
    benchmark/retrieval/metrics_utils.py :: compute_recall_at_k()
    Difference: compute_recall_at_k() accepts list[dict] with 'doc_id'
    keys; this evaluator accepts list[str] directly.

    PIPELINE WIRING
    ---------------
    Instantiated by benchmark/application/composer.py _build_evaluators() ~line 367.
    Called per-query inside benchmark/orchestrator/scenario_runner.py
    _execute_queries_sequential() ~line 453.
    Results aggregated (macro-averaged) in _build_scenario_metrics() ~line 790.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize Recall@K evaluator.

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

        Raises:
            ValueError: If expected_ids is empty.
        """
        if not expected_ids:
            raise ValueError(
                "RecallAtKEvaluator: expected_ids cannot be empty. "
                "Every query must have at least one expected memory in gold dataset."
            )

        top_k_retrieved = set(retrieved_ids[: self._top_k])
        gold_set = set(expected_ids)
        relevant_retrieved = top_k_retrieved & gold_set
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
        return f"benchmark.recall_at_{self._top_k}"
