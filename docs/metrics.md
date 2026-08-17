# Metric Semantics Reference

> Canonical definitions for every metric emitted by the benchmark evaluation module.
> This is the single source of truth for formulas, K propagation, aggregation, and edge cases.

**Metric Semantics Version: 2.0**

This version identifier is embedded in every benchmark result via the `metric_semantics_version`
field in the run plan. When metric formulas change, this version is incremented to prevent
invalid comparisons between results produced under different definitions.

---

## Metrics Summary

| Metric | Formula | Direction | Range |
|--------|---------|-----------|-------|
| `benchmark.recall_at_k` | \|R ∩ G\| / \|G\| | higher = better | [0.0, 1.0] |
| `benchmark.precision_at_k` | \|R ∩ G\| / K | higher = better | [0.0, 1.0] |
| `benchmark.contamination_rate` | \|R \\ G\| / \|R\| | lower = better | [0.0, 1.0] |
| `benchmark.temporal_accuracy` | fraction within window | higher = better | [0.0, 1.0] |
| `benchmark.mrr` | 1 / rank(first relevant) | higher = better | [0.0, 1.0] |
| `benchmark.ndcg_at_k` | DCG / IDCG | higher = better | [0.0, 1.0] |

---

## Detailed Definitions

### benchmark.recall_at_k

**Formula:** `|Retrieved ∩ GoldSet| / |GoldSet|`

| Property | Value |
|----------|-------|
| K source | `dataset.evaluation_criteria.recall_k` |
| K propagation | Same K passed to query `top_k`, evaluator, and report |
| Direction | Higher = better |
| Empty gold set | Invalid — raises `ValueError` (every query must have expected results) |
| Empty retrieval | 0.0 |
| Aggregation | Macro average across all queries |

**Interpretation:** "Of the memories we should have found, what fraction did we actually find in the top K?"

---

### benchmark.precision_at_k

**Formula:** `|Retrieved[:K] ∩ GoldSet| / K`

| Property | Value |
|----------|-------|
| K source | `dataset.evaluation_criteria.recall_k` (same as Recall) |
| Direction | Higher = better |
| Empty retrieval | 0.0 |
| Empty gold set | 0.0 (no relevant items possible) |
| Max achievable | `min(|GoldSet|, K) / K` — reported in `details.max_achievable` |
| Aggregation | Macro average across all queries |

**Interpretation:** "Of the K results we returned, what fraction is actually relevant?"

**Important:** This is the standard IR formula. It is computed **independently** from contamination rate.

---

### benchmark.contamination_rate

**Formula:** `|Retrieved \\ GoldSet| / |Retrieved|`

| Property | Value |
|----------|-------|
| K source | N/A — evaluates all returned results |
| Direction | Lower = better |
| Empty retrieval | 0.0 (no contamination possible) |
| Aggregation | Macro average across all queries |

**Interpretation:** "Of the memories we'll feed to the LLM, what fraction is noise?" This directly measures hallucination risk — a high contamination rate means the retrieval system is surfacing irrelevant context that could confuse downstream generation.

**Distinction from Precision@K:** Contamination rate measures noise proportion in the full returned set. Precision@K measures relevance proportion in the top K positions. They operate on different denominators and are NOT mathematical complements in general (only when |Retrieved| = K).

---

### benchmark.temporal_accuracy

**Formula:** `|retrieved memories within temporal window| / |retrieved memories|`

| Property | Value |
|----------|-------|
| Tolerance | `dataset.evaluation_criteria.temporal_tolerance_days` (default: 1) |
| Direction | Higher = better |
| Empty retrieval | 0.0 |
| No temporal window | 1.0 (unconstrained queries are always temporally correct) |
| Aggregation | Macro average across queries that have temporal constraints |

**Interpretation:** "Did we return memories from the correct time period?" A score of 0.7 means 70% of returned memories were created within the expected temporal window.

---

### benchmark.mrr (Mean Reciprocal Rank)

**Formula:** `1 / rank_of_first_relevant_result`

| Property | Value |
|----------|-------|
| K source | `dataset.evaluation_criteria.recall_k` |
| Direction | Higher = better |
| No relevant in top K | 0.0 |
| Aggregation | Mean across all queries |

**Interpretation:** "How high up in the ranked list does the first correct result appear?" MRR = 1.0 means the correct result is always at position 1.

---

### benchmark.ndcg_at_k (Normalized Discounted Cumulative Gain)

**Formula:** `DCG@K / IDCG@K` where `DCG = Σ rel_i / log2(i + 1)`

| Property | Value |
|----------|-------|
| K source | `dataset.evaluation_criteria.recall_k` |
| Relevance | Binary: 1 if in gold set, 0 otherwise |
| Direction | Higher = better |
| No relevant in top K | 0.0 |
| Aggregation | Mean across all queries |

**Interpretation:** "Are the relevant results concentrated at the top of the ranked list?" NDCG rewards systems that place relevant items earlier.

---

## Aggregation Rules

All metrics use **macro averaging** (per-query mean):

```
aggregate_metric = sum(per_query_values) / number_of_evaluated_queries
```

This gives equal weight to every query regardless of gold set size.

---

## K Propagation

The value of K flows from one authoritative source:

```
dataset.evaluation_criteria.recall_k
        ↓
BenchmarkComposer._build_evaluators(recall_k)
        ↓
RecallEvaluator(top_k=K)
StandardPrecisionEvaluator(top_k=K)
        ↓
ReadQuery(top_k=K) in ScenarioRunner
        ↓
Run report includes: recall_k = K
```

There is no separate hard-coded K in the CLI or matrix worker.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-07-30 | Precision@K computed independently (standard IR formula). Contamination rate separated. Module weighting active in strategy path. |
| 1.0 | 2026-03-31 | Initial metrics: Recall@K, FPR (contamination), temporal accuracy. |
