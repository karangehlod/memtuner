"""Metric utility functions for gold-grounded and score-estimated evaluation.

TWO USAGE PATHS
---------------
Path 1 — Gold-grounded (primary, used by study_runner.py):
  compute_mrr(), compute_recall_at_k(), compute_precision_at_k(), compute_ndcg()
  are called directly in benchmark_runner.py and study_runner.py after calling
  adapter.query_memories() for each GoldQuery. The gold set comes from
  GoldQuery.expected.memory_ids. Results are then averaged via compute_metric_summary().

Path 2 — Score-estimated (adapter self-evaluation only):
  When no gold labels are available (adapter-layer get_metrics()), estimate_relevance_from_scores()
  builds a pseudo-gold set using score >= 0.5 threshold. Results are self-referential
  (the retriever controls both scores and the gold set) so they indicate operational
  health, NOT benchmark quality. Do not report score-estimated metrics as benchmark results.

FORMULA SUMMARY
---------------
MRR         compute_mrr(results, relevant)
            = 1/rank_first_relevant   (0 if none in top-K, 0 if relevant=None)
            Ref: MRS2008 §8.4

Recall@K    compute_recall_at_k(results, relevant, k)
            = |{doc_id for doc in results[:k]} ∩ relevant| / |relevant|
            Denominator: always |relevant| (NOT k)
            Dedup: set-based — duplicate doc_id in results[:k] counted once
            Ref: MRS2008 §8.3

Precision@K compute_precision_at_k(results, relevant, k)
            = |{doc_id for doc in results[:k]} ∩ relevant| / k
            Denominator: always k — systems returning < k results are penalised
            Dedup: set-based — duplicate doc_id counted once
            Ref: MRS2008 §8.3

NDCG@K      compute_ndcg(results, relevant, k)
            DCG@K  = sum_{i=1..K} 1/log2(i+1)  if doc_i in relevant (binary gain)
            IDCG@K = sum_{i=1..min(|relevant|,K)} 1/log2(i+1)
            NDCG@K = DCG@K / IDCG@K   (0 if relevant=None or IDCG=0)
            Dedup: relevant doc_id credited at most once in DCG
            Ref: MRS2008 §8.4

MACRO-AVERAGING
---------------
compute_metric_summary(all_results, relevant_sets) averages all metrics
macro-style: each query contributes equally regardless of |gold|.
This is the standard in IR evaluation (Voorhees & Harman 2005).

WHERE CALLED
------------
Gold-grounded path:
  benchmark_runner.py  run_adapter_on_dataset()  after each query_memories() call
  study_runner.py      _write_leaderboards_json()  for CI computation input
Score-estimated path:
  benchmark/memory/adapters/*_adapter.py  get_metrics()  via _cms alias
"""

import math
from typing import Any


def compute_mrr(
    results: list[dict[str, Any]],
    relevant_doc_ids: set[str] | None = None,
    k: int = 10,
) -> float:
    """Compute MRR@K (Mean Reciprocal Rank at K) for a single query.

    Formula:
        MRR@K = 1 / rank_first_relevant   (0.0 if no hit in top-K)

    where rank_first_relevant is the 1-indexed position of the first result
    whose doc_id is in relevant_doc_ids, scanning only the top-K results.

    The K cutoff is required for MRR@K semantics: a first hit at rank 15 must
    return 0.0 when K=10, not 1/15. The previous implementation had no K limit
    and was effectively computing MRR over the full result list regardless of K,
    inflating reported MRR compared to the bounded MRREvaluator in ranking.py.

    Edge cases:
        - results is empty            → returns 0.0
        - relevant_doc_ids is None    → returns 0.0
        - no relevant doc in top-K    → returns 0.0

    Cross-reference:
        MRREvaluator (benchmark/evaluation/ranking.py) implements the same
        formula operating on list[str] memory IDs; both are now MRR@K.

    Args:
        results: List of result dicts each containing a 'doc_id' key.
        relevant_doc_ids: Set of gold-label document IDs. If None, returns 0.0.
        k: Maximum rank to consider. Default 10 matches benchmark standard.

    Returns:
        MRR@K score in [0, 1].
    """
    if not results or relevant_doc_ids is None:
        return 0.0

    for rank, result in enumerate(results[:k], 1):
        if result["doc_id"] in relevant_doc_ids:
            return 1.0 / rank

    return 0.0  # No relevant document in top-K


def compute_ndcg(
    results: list[dict[str, Any]],
    relevant_doc_ids: set[str] | None = None,
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@k) for a single query.

    Formula (binary gain):
        DCG@K  = sum_{i=1}^{K}  1/log2(i+1)   if results[i-1].doc_id in relevant
        IDCG@K = sum_{i=1}^{min(|relevant|,K)} 1/log2(i+1)
        NDCG@K = DCG@K / IDCG@K

    Deduplication:
        A relevant doc_id appearing more than once in results[:k] is credited
        at most once in DCG. Without dedup, DCG can exceed IDCG, producing
        NDCG > 1.0. The _seen set enforces this constraint.

    Edge cases:
        - results is empty            → returns 0.0
        - relevant_doc_ids is None    → returns 0.0 (no gold set available)
        - IDCG == 0 (k == 0)          → returns 0.0 (by convention)
        - no relevant doc in results  → DCG == 0.0, NDCG == 0.0

    Cross-reference:
        NDCGEvaluator (benchmark/evaluation/ranking.py) implements the same
        DCG/IDCG formula operating on list[str] memory IDs rather than list[dict].

    Where called:
        compute_metric_summary() — aggregates per-query NDCG across a dataset.
        Called directly in benchmark_runner.py run_adapter_on_dataset() and
        study_runner.py _write_leaderboards_json() for gold-grounded evaluation.

    Args:
        results: List of result dicts each containing a 'doc_id' key.
        relevant_doc_ids: Set of gold-label document IDs.
        k: Rank cutoff for NDCG@k (default 10).

    Returns:
        NDCG@k score in [0, 1].
    """
    if not results:
        return 0.0

    if relevant_doc_ids is None:
        # Cannot compute NDCG without a gold set — fabricating one from the
        # retrieved list makes every result "relevant" and returns 1.0 for
        # any input, which is meaningless. Return 0 consistently with compute_mrr.
        return 0.0

    # Compute DCG (Discounted Cumulative Gain).
    # Deduplicate: a relevant doc appearing twice in top-k is credited once only;
    # without dedup DCG can exceed IDCG giving NDCG > 1.0.
    dcg = 0.0
    _seen: set[str] = set()
    for i, result in enumerate(results[:k], 1):
        doc_id = result["doc_id"]
        if doc_id in relevant_doc_ids and doc_id not in _seen:
            dcg += 1.0 / math.log2(i + 1)
            _seen.add(doc_id)

    # Compute IDCG (Ideal DCG - perfect ranking)
    num_relevant = len(relevant_doc_ids)
    idcg = 0.0
    for i in range(1, min(num_relevant, k) + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_recall_at_k(
    results: list[dict[str, Any]],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Recall@k for a single query.

    Formula:
        Recall@K = |{doc_id for doc in results[:k]} ∩ relevant_doc_ids|
                   / |relevant_doc_ids|

    The denominator is always |relevant_doc_ids| (the total gold-set size),
    NOT k. This measures coverage of the gold set, not precision.

    Deduplication:
        The numerator is a set intersection. A doc_id appearing more than once
        in results[:k] is counted once only; without dedup the numerator could
        exceed |relevant_doc_ids|, producing Recall > 1.0.

    Edge cases:
        - relevant_doc_ids is empty   → returns 0.0 (avoids ZeroDivisionError)
        - results is empty            → retrieved_set is empty, returns 0.0
        - no relevant doc in results  → intersection is empty, returns 0.0

    Cross-reference:
        RecallAtKEvaluator (benchmark/evaluation/ranking.py) implements the
        same formula operating on list[str] memory IDs rather than list[dict].

    Where called:
        compute_metric_summary() — called four times per query at k=1,5,10,100.
        Called directly in benchmark_runner.py run_adapter_on_dataset() and
        study_runner.py _write_leaderboards_json() for gold-grounded evaluation.

    Args:
        results: List of result dicts each containing a 'doc_id' key.
        relevant_doc_ids: Set of all gold-label document IDs for this query.
        k: Rank cutoff (default 10).

    Returns:
        Recall@k score in [0, 1].
    """
    if not relevant_doc_ids:
        return 0.0

    # Use set intersection — a duplicate doc_id in results[:k] must not be
    # counted twice (numerator would exceed denominator, giving recall > 1.0).
    retrieved_set = {r["doc_id"] for r in results[:k]}
    return len(retrieved_set & relevant_doc_ids) / len(relevant_doc_ids)


def compute_precision_at_k(
    results: list[dict[str, Any]],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Precision@k for a single query.

    Formula:
        Precision@K = |{doc_id for doc in results[:k]} ∩ relevant_doc_ids| / k

    The denominator is always k (NOT len(results[:k])). A system that returns
    fewer than k results is penalised — the missing slots count as not-relevant.

    Deduplication:
        The numerator is a set intersection. A doc_id appearing more than once
        in results[:k] is counted once only; without dedup the numerator could
        exceed k, producing Precision > 1.0.

    Edge cases:
        - results is empty or k == 0  → returns 0.0 (avoids ZeroDivisionError)
        - no relevant doc in results  → intersection is empty, returns 0.0

    Cross-reference:
        PrecisionAtKEvaluator (benchmark/evaluation/ranking.py) implements the
        same formula operating on list[str] memory IDs rather than list[dict].

    Where called:
        compute_metric_summary() — called once per query at k=10.
        Called directly in benchmark_runner.py run_adapter_on_dataset() and
        study_runner.py _write_leaderboards_json() for gold-grounded evaluation.

    Args:
        results: List of result dicts each containing a 'doc_id' key.
        relevant_doc_ids: Set of gold-label document IDs for this query.
        k: Rank cutoff (default 10).

    Returns:
        Precision@k score in [0, 1].
    """
    if not results or k == 0:
        return 0.0

    # Set intersection — duplicates in results[:k] must not inflate the count.
    # Denominator is always k: a system returning fewer than k results is penalised.
    retrieved_set = {r["doc_id"] for r in results[:k]}
    return len(retrieved_set & relevant_doc_ids) / k


def estimate_relevance_from_scores(
    results: list[dict[str, Any]],
    score_threshold: float = 0.5,
) -> set[str]:
    """Estimate a pseudo-gold relevant set from retriever scores (score-estimated path).

    Used exclusively in Path 2 (adapter self-evaluation): when no GoldQuery
    gold labels are available, this function synthesises a relevance set by
    treating any result whose 'score' field meets the threshold as relevant.

    Formula:
        pseudo_gold = {r['doc_id'] for r in results if r.get('score', 0.0) >= threshold}

    WARNING — self-referential bias:
        The retriever controls both the scores and the resulting gold set.
        Metrics computed against this pseudo-gold set reflect the retriever's
        internal confidence, NOT true recall or precision. Do NOT report
        score-estimated metrics as benchmark results; use them only as an
        operational health indicator.

    Edge cases:
        - results is empty                    → returns empty set
        - result dict has no 'score' key      → treated as score 0.0 (excluded
                                                by default threshold of 0.5)
        - all scores < score_threshold        → returns empty set

    Where called:
        compute_metric_summary() — invoked when relevant_sets is None and
        use_score_estimation=True (default). Also called via _cms alias in
        benchmark/memory/adapters/*_adapter.py get_metrics() methods.

    Args:
        results: List of result dicts each expected to contain 'doc_id' and
            optionally 'score' (float). Missing 'score' defaults to 0.0.
        score_threshold: Minimum score for a result to be considered relevant.
            Default 0.5 matches the convention used by all adapter get_metrics().

    Returns:
        Set of doc_id strings estimated to be relevant.
    """
    return {
        r["doc_id"] for r in results
        if r.get("score", 0.0) >= score_threshold
    }


def compute_metric_summary(
    all_results: list[list[dict[str, Any]]],
    relevant_sets: list[set[str]] | None = None,
    use_score_estimation: bool = True,
) -> dict[str, float]:
    """Compute macro-averaged IR metrics across multiple queries.

    For each query i, the relevant set is resolved in this priority order:
      1. relevant_sets[i]                     if relevant_sets is provided (gold path)
      2. estimate_relevance_from_scores(...)   if use_score_estimation=True (score path)
      3. top-10 doc_ids from results[i]        fallback (treats retrieved as relevant)

    Metrics computed per query:
        Recall@1, Recall@5, Recall@10, Recall@100  — via compute_recall_at_k()
        Precision@10                               — via compute_precision_at_k()
        MRR                                        — via compute_mrr()
        NDCG@10                                    — via compute_ndcg()

    Aggregation (macro-averaging):
        Each query contributes equally to each aggregate metric regardless of
        gold-set size (Voorhees & Harman 2005). Final values are clamped to
        [0, 1] to guard against floating-point edge cases.

    Edge cases:
        - all_results is empty  → returns all-zeros dict with the seven metric keys
        - result elements that are not dicts are silently dropped before scoring

    Where called:
        Gold-grounded path:
            benchmark_runner.py  run_adapter_on_dataset()  final summary
            study_runner.py      _write_leaderboards_json()
        Score-estimated path:
            benchmark/retrieval/strategies/*_adapter.py  get_metrics()
            benchmark/memory/adapters/*_adapter.py       get_metrics()  via _cms alias

    Args:
        all_results: List of per-query result lists; each element is a list of
            dicts containing at least 'doc_id' and optionally 'score'.
        relevant_sets: Optional list of gold-label doc_id sets, one per query.
            Pass None to fall back to score estimation or top-10 heuristic.
        use_score_estimation: When relevant_sets is None, use
            estimate_relevance_from_scores() to build pseudo-gold sets.
            Set to False only when you want the top-10 fallback explicitly.

    Returns:
        Dict with keys: recall_at_1, recall_at_5, recall_at_10, recall_at_100,
        precision_at_10, mrr, ndcg — each a float in [0, 1].
    """
    if not all_results:
        return {
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_100": 0.0,
            "precision_at_10": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }

    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    recalls_100 = []
    precisions_10 = []
    mrrs = []
    ndcgs = []

    for i, results in enumerate(all_results):
        # Normalise: drop any elements that are not dicts (e.g. accidental nesting)
        results = [r for r in results if isinstance(r, dict)]

        # Determine relevant set
        if relevant_sets and i < len(relevant_sets):
            relevant = relevant_sets[i]
        elif use_score_estimation:
            relevant = estimate_relevance_from_scores(results, score_threshold=0.5)
        else:
            # No relevance information, estimate all results as relevant
            relevant = {r["doc_id"] for r in results[:10]}

        # Compute recall@{1,5,10,100} and precision@10 in one pass over the top-100 list.
        # Single pass eliminates 5 separate set constructions per query (9885 → 1977 per cell).
        ids_100 = [r["doc_id"] for r in results[:100]]
        gold_set = relevant if isinstance(relevant, set) else set(relevant)
        seen: set[str] = set()
        r1 = r5 = r10 = r100 = 0
        denom = len(gold_set) or 1
        for rank_i, doc_id in enumerate(ids_100, 1):
            if doc_id not in seen and doc_id in gold_set:
                seen.add(doc_id)
                r100 += 1
                if rank_i <= 10:
                    r10 += 1
                if rank_i <= 5:
                    r5 += 1
                if rank_i == 1:
                    r1 = 1
        recalls_1.append(r1 / denom)
        recalls_5.append(r5 / denom)
        recalls_10.append(r10 / denom)
        recalls_100.append(r100 / denom)
        precisions_10.append(r10 / 10)  # reuse r10 — no extra set needed
        mrrs.append(compute_mrr(results, relevant, k=10))
        ndcgs.append(compute_ndcg(results, relevant, k=10))

    # Average across all queries (clamp to [0, 1] range)
    return {
        "recall_at_1": min(1.0, max(0.0, sum(recalls_1) / len(recalls_1))),
        "recall_at_5": min(1.0, max(0.0, sum(recalls_5) / len(recalls_5))),
        "recall_at_10": min(1.0, max(0.0, sum(recalls_10) / len(recalls_10))),
        "recall_at_100": min(1.0, max(0.0, sum(recalls_100) / len(recalls_100))),
        "precision_at_10": min(1.0, max(0.0, sum(precisions_10) / len(precisions_10))),
        "mrr": min(1.0, max(0.0, sum(mrrs) / len(mrrs))),
        "ndcg": min(1.0, max(0.0, sum(ndcgs) / len(ndcgs))),
    }
