# MemTuner — Metric Definitions

Single source of truth for every reported metric: formula, implementation pointer,
data provenance, and extension guide.

---

## 1. Overview

The benchmark produces metrics through two distinct evaluation pipelines.
Understanding which pipeline produced a number is essential for interpreting it correctly.

### Pipeline 1 — Gold-grounded (authoritative)

```
study_runner.py
  └─ _run_single_dataset()
       └─ BenchmarkComposer.compose()
            └─ BenchmarkRunner.run()
                 └─ ScenarioRunner._execute_queries_sequential()  (or _async variant)
                      │  retrieved_ids ← memory_module.read()
                      │  expected_ids  ← GoldQuery.expected.memory_ids  (ground truth)
                      └─ evaluators: MRREvaluator, NDCGEvaluator, PrecisionAtKEvaluator,
                                     RecallAtKEvaluator, RecallEvaluator, …
                           └─ ScenarioRunner._build_scenario_metrics()
                                └─ ScenarioMetrics  →  StudyRunResult
                                     └─ StudyAggregator  →  leaderboards.json
```

Gold labels come from `GoldQuery.expected.memory_ids` — a curated set of memory IDs
that a correct retrieval system must return for each query.  Every Recall, MRR, NDCG,
and Precision value in `benchmark_results/leaderboards.json` is gold-grounded.

### Pipeline 2 — Score-estimated (adapter self-evaluation only)

```
memory_adapter.get_metrics()
  └─ compute_metric_summary(all_results, relevant_sets=None)
       └─ estimate_relevance_from_scores(results, threshold=0.5)
            pseudo_gold = {r["doc_id"] for r in results if r["score"] >= 0.5}
```

Because the adapter controls both the retrieval scores and the resulting pseudo-gold
set, score-estimated metrics are **self-referential**.  They indicate operational
health (is the adapter returning results at all?), not benchmark quality.  Never
report score-estimated values as benchmark results.  See Section 8 for details.

---

## 2. Core IR Metrics

All evaluators in this section operate on per-query `(retrieved_ids, expected_ids)` pairs.
`retrieved_ids` is an ordered list of memory IDs returned by the system (highest score
first); `expected_ids` is the gold set from `GoldQuery.expected.memory_ids`.

**Deduplication rule (applies to all evaluators):**
A memory ID that appears more than once in `retrieved_ids` occupies one rank slot and
is credited at most once toward any metric.  Without deduplication:
- DCG can exceed IDCG, producing NDCG > 1.0.
- The precision/recall numerator can exceed the denominator.

**Edge case guard:** every evaluator raises `ValueError` when `expected_ids` is empty.
This is a data-quality guard — every gold query must have at least one expected memory.

**Standard reference:** Manning, Raghavan & Schutze, *Introduction to Information
Retrieval*, Cambridge University Press, 2008.

---

### 2.1 Mean Reciprocal Rank (MRR)

**Formula**

```
MRR@K = 1 / rank_first_relevant    if a relevant ID appears within rank 1..K
      = 0.0                         if no relevant ID in top-K
```

`rank_first_relevant` is the 1-based position of the first entry in
`retrieved_ids[:K]` that is a member of `gold_set`.

**Denominator:** none — reciprocal rank is per-query; macro-average is taken
across queries by the aggregator.

**Range:** [0.0, 1.0].  1.0 = first result is relevant.  Higher is better.

**Edge cases:**
- `retrieved_ids` empty → 0.0
- No relevant ID in top-K → 0.0
- Duplicate IDs → first occurrence terminates scan; duplicates not credited

**Reference:** MRS2008 §8.4

---

### 2.2 Recall@K

**Formula**

```
Recall@K = |{retrieved_ids[:K]} ∩ gold_set| / |gold_set|
```

Denominator is always `|gold_set|` (total relevant items), never K.
Measures **coverage**: what fraction of all known relevant memories were
found within the top-K results.

**Denominator rule:** always `|gold_set|`.  A system that finds 3 of 5 gold
memories scores 0.6 regardless of K.

**Range:** [0.0, 1.0].  1.0 = every gold memory found in top-K.

**Edge cases:**
- `retrieved_ids` empty → 0.0
- `|gold_set| > K` → perfect recall impossible unless `|gold_set| ≤ K`; maximum achievable = `K / |gold_set|`
- Duplicate IDs in retrieved → set conversion deduplicates; recall cannot exceed 1.0

**Reference:** MRS2008 §8.3

---

### 2.3 Precision@K

**Formula**

```
P@K = |{retrieved_ids[:K]} ∩ gold_set| / K
```

Denominator is always K, regardless of how many items were actually returned.
A system that returns fewer than K results is penalised — missing slots count
as not-relevant.

**Denominator rule:** always K.  This differs from Recall@K where the denominator
is `|gold_set|`.

**Range:** [0.0, 1.0].  Maximum achievable = `min(|gold_set|, K) / K`.

**Edge cases:**
- `retrieved_ids` empty → 0.0
- `len(retrieved_ids) < K` → denominator still K (penalty for short lists)
- Duplicate IDs in retrieved → set intersection deduplicates; numerator cannot exceed K

**Reference:** MRS2008 §8.3

---

### 2.4 NDCG@K

**Formula**

```
DCG@K  = sum_{i=1}^{K}  rel_i / log2(i+1)      where rel_i ∈ {0, 1}
IDCG@K = sum_{i=1}^{min(|gold_set|, K)}  1 / log2(i+1)
NDCG@K = DCG@K / IDCG@K
```

`rel_i = 1` if `retrieved_ids[i-1]` is in `gold_set`, else 0.  Binary gain only.

Discount at rank 1: `1/log2(2) = 1.000`
Discount at rank 2: `1/log2(3) ≈ 0.631`
Discount at rank 3: `1/log2(4) = 0.500`

**Denominator rule:** IDCG uses `min(|gold_set|, K)` ideal positions.  Perfect recall
within K still gives NDCG = 1.0 even when `|gold_set| > K`.

**Range:** [0.0, 1.0].  1.0 = perfect ranking.  Higher is better.

**Edge cases:**
- `retrieved_ids` empty → DCG = 0.0, NDCG = 0.0
- `IDCG = 0` (K = 0) → NDCG = 0.0 by convention
- Duplicate IDs in retrieved → a relevant ID appearing more than once is credited
  only at its **first occurrence** (a `_seen` set enforces this); without dedup,
  DCG can exceed IDCG giving NDCG > 1.0

**Reference:** MRS2008 §8.4

---

### 2.5 Precision@1

Precision@K evaluated at K = 1.  Answers: "Did the single top result match the gold set?"

```
P@1 = 1.0  if retrieved_ids[0] ∈ gold_set
    = 0.0  otherwise
    = 0.0  if retrieved_ids is empty
```

`PrecisionAtKEvaluator(top_k=1)` is instantiated separately from the general
`PrecisionAtKEvaluator(top_k=recall_k)`.  It emits the metric key
`"benchmark.precision_at_1"` and is consumed as `ScenarioMetrics.precision_at_1`.

---

## 3. Statistical Methods

### 3.1 Macro-Averaging

All aggregate metrics (avg_recall, avg_mrr, avg_ndcg, avg_precision) are
**macro-averages**: each query contributes equally regardless of gold-set size.

```
metric_aggregate = (1 / N) × sum_{i=1}^{N} metric_i
```

where N is the number of queries.

This is standard in IR evaluation (Voorhees & Harman 2005).  It means a query
with a gold set of size 1 and a query with a gold set of size 50 have equal weight.

Implementation: `ScenarioRunner._build_scenario_metrics()` calls `avg()` (a local
helper) on the list of per-query `EvaluationResult.value` fields collected for
each metric key.  `compute_metric_summary()` in `metrics_utils.py` applies the same
`sum / len` pattern with `[0, 1]` clamping.

---

### 3.2 Bootstrap Confidence Intervals (95%)

**Method:** non-parametric percentile bootstrap.

```
alpha   = 1.0 - ci_level          # e.g. 0.05 for 95% CI
n       = n_bootstrap              # 1000 iterations (standard)

# For each group (e.g. each strategy):
1. Collect per-cell metric values.
2. Resample with replacement n times; compute mean each time.
3. Sort bootstrap means array (length n).
4. lo_idx = floor(alpha/2 * n)                   # e.g. floor(0.025 * 1000) = 25
5. hi_idx = ceil((1 - alpha/2) * n) - 1          # e.g. ceil(0.975 * 1000) - 1 = 974
6. ci_low  = boot_means[lo_idx]
7. ci_high = boot_means[hi_idx]
```

The `-1` in step 5 is required because `ceil()` returns a 1-indexed nearest-rank
position; subtracting 1 converts it to a 0-based array subscript.  Omitting it
makes the CI one position wider than the requested level.

**Output fields** in `leaderboards.json`:
- `recall_ci_low`, `recall_ci_high` — 95% CI for Recall@K
- `mrr_ci_low`, `mrr_ci_high` — 95% CI for MRR

**Reference:** Sakai 2006 "Evaluating Evaluation Metrics", SIGIR.
Voorhees & Harman 2005, *TREC: Experiment and Evaluation in IR*.

---

### 3.3 Significance Test

Two strategies are marked **significantly different** when their 95% bootstrap CIs
do not overlap:

```
sig_vs_next = True  iff  strategy_A.ci_low > strategy_B.ci_high
```

where strategy B is the next-lower-ranked strategy.

This is a **conservative test**: non-overlapping CIs is a stronger condition than
p < 0.05 from a paired t-test.  When CIs overlap, the difference may still be
statistically significant with a paired test.  A future improvement would use
Wilcoxon signed-rank over per-query values (noted as future work in the module
docstring).

`sig_vs_next` appears in the `accuracy_leaderboard.entries` array in
`benchmark_results/leaderboards.json`.

---

## 4. System / Hardware Metrics

### 4.1 Query Latency (P50, P90, P99)

Per-query retrieval latency is measured from just before `module.read(query)` is
called to just after the response returns, in milliseconds.  All module latencies
for a single query are summed:

```
query_latency_ms = sum of latency_ms from each memory module's read() response
```

Percentiles use the **nearest-rank formula** (NIST/Hyndman & Fan 1996):

```
sorted_latencies = sort(query_latencies_ms)
n = len(sorted_latencies)

p50 = sorted_latencies[max(0, min(ceil(n * 0.50) - 1, n-1))]
p90 = sorted_latencies[max(0, min(ceil(n * 0.90) - 1, n-1))]
p99 = sorted_latencies[max(0, min(ceil(n * 0.99) - 1, n-1))]
```

The floor-based formula (commonly `int(n * p)`) over-reports P90/P99 on small
samples; the nearest-rank formula is more conservative.

Fields on `ScenarioMetrics`: `latency_p50_ms`, `latency_p90_ms`, `latency_p99_ms`,
`latency_mean_ms`.

---

### 4.2 Index Build Time

Index build time is not reported as a standalone metric in the current pipeline.
The `duration_seconds` field on `MatrixRunResult` covers the full benchmark cell
wall-clock time (events replay + queries), not index build in isolation.

To isolate embedding index build time: set `BENCHMARK_HW_DEBUG=1` and inspect the
console output, or wrap the `EmbeddingsStrategy._index_all()` call with a manual
timer.

---

### 4.3 Peak RSS

Peak RSS (Resident Set Size) is the **high-water mark** of physical RAM used by
the benchmark process.

```
# macOS (Darwin): ru_maxrss is in bytes
peak_rss_mb = resource.getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024)

# Linux: ru_maxrss is in kilobytes
peak_rss_mb = resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024
```

`resource.getrusage` is used (rather than `psutil.rss`) because `psutil.rss` is
a live snapshot that decreases after memory is freed, while `getrusage` tracks the
lifetime maximum across the entire process.

The field `peak_ram_mb` on `MatrixRunResult` / `StudyRunResult` is set from
`ResourceTracker.report().peak_ram_mb` after the benchmark cell completes.  It is
distinct from `peak_rss_mb` used in the adapter-layer path
(`benchmark/memory/adapters/_sys_metrics.py`), which samples at `get_metrics()` time.

---

### 4.4 Batch Size Calculation

The embedding batch size is calculated once at `EmbeddingsStrategy.__init__()` time,
after the model weights are loaded, so that CUDA free memory already reflects the
actual weight footprint.

```
ACTIVATION_FACTOR  = 12
SEQ_LEN            = 256   (SentenceTransformers default max sequence length)
FLOAT32_BYTES      = 4

bytes_per_sample = SEQ_LEN × model_dim × FLOAT32_BYTES × ACTIVATION_FACTOR
                 = 256 × model_dim × 4 × 12
                 = 12288 × model_dim

# Budget selection
CUDA  : budget = torch.cuda.mem_get_info()[0] × 0.50    (free VRAM after model load)
MPS   : budget = GPU_VRAM_MB × 1024² × 0.50             (GPU_VRAM_MB = 75% of total RAM)
CPU   : budget = 512 × 1024²                             (fixed 512 MB)

raw  = budget // bytes_per_sample
size = floor_pow2(raw)          # round down to nearest power of 2

# Output clamps
GPU   : batch_size ∈ [64,  2048]
CPU   : batch_size ∈ [32,   512]
```

`ACTIVATION_FACTOR = 12` accounts for Q/K/V projections (3×), attention score
matrix (1×), FFN up-projection (~4×), and FFN down-projection (~4×) per layer.

---

## 5. Leaderboard Score Formulas

### 5.1 Accuracy Score (Composite)

The **composite score** is the primary ranking signal used by `rank_by_retrieval_strategy()`
and `rank_by_memory_type()` in `MatrixAggregator`.

```
composite_score = recall_gate × (w_R × Recall@K
                               + w_P × Precision@K
                               + w_M × MRR
                               + w_T × TemporalAccuracy)

where:
  recall_gate = 0.0  if Recall@K < 0.01   (prevents an empty store from ranking)
              = 1.0  otherwise

Default weights (COMPOSITE_WEIGHTS):
  w_R = 0.40   (recall is primary retrieval coverage signal)
  w_P = 0.25   (precision = set cleanliness)
  w_M = 0.20   (MRR = ranking quality)
  w_T = 0.15   (temporal accuracy = time-window correctness)
```

**Range:** [0.0, 1.0].  Higher is better.

**Note:** The accuracy_leaderboard in `leaderboards.json` ranks strategies by
`avg_recall_at_k` (not composite score).  The composite score is used for
per-cell rankings inside the study grid.

Use `composite_score_weighted(weights)` for sensitivity analysis with custom weights.

---

### 5.2 Efficiency Score

The efficiency leaderboard ranks strategies by **ascending** P50 query latency
(lower = faster).  A secondary efficiency metric is:

```
recall_per_ms = avg_recall / max(avg_latency_p50_ms, 0.001)
```

Higher `recall_per_ms` = better recall-to-latency tradeoff.  The `max(..., 0.001)`
guard prevents division by zero when latency is not measured.

The `cost_vs_quality` section of `leaderboards.json` sorts entries by
`recall_per_ms` descending.

---

### 5.3 Balanced Score

There is no dedicated "balanced score" field.  The closest equivalent is
`recall_per_ms` (see §5.2), which balances retrieval quality against latency.

The `avg_composite` field in strategy rankings (see §5.1) balances four quality
metrics but does not incorporate latency.

---

## 6. Implementation Map

| Metric | Formula | Evaluator / Function | File | Function / Method | Called From |
|--------|---------|---------------------|------|-------------------|-------------|
| MRR@K | `1 / rank_first_relevant` (0 if no hit) | `MRREvaluator` | `benchmark/evaluation/ranking.py` | `MRREvaluator.evaluate()` line ~95 | `composer.py _build_evaluators()` line ~380 |
| NDCG@K | `DCG@K / IDCG@K` (binary gain, dedup) | `NDCGEvaluator` | `benchmark/evaluation/ranking.py` | `NDCGEvaluator.evaluate()` line ~203 | `composer.py _build_evaluators()` line ~382 |
| Precision@K | `\|retrieved[:K] ∩ gold\| / K` | `PrecisionAtKEvaluator` | `benchmark/evaluation/ranking.py` | `PrecisionAtKEvaluator.evaluate()` line ~323 | `composer.py _build_evaluators()` line ~369 |
| Precision@1 | `P@K` with K=1 | `PrecisionAtKEvaluator(top_k=1)` | `benchmark/evaluation/ranking.py` | `PrecisionAtKEvaluator.evaluate()` line ~323 | `composer.py _build_evaluators()` line ~384 |
| Recall@K | `\|retrieved[:K] ∩ gold\| / \|gold\|` | `RecallAtKEvaluator` | `benchmark/evaluation/ranking.py` | `RecallAtKEvaluator.evaluate()` line ~424 | `composer.py _build_evaluators()` line ~368 |
| MRR (util) | same as MRR@K above | — | `benchmark/retrieval/metrics_utils.py` | `compute_mrr()` line ~61 | `compute_metric_summary()` line ~415 |
| NDCG@K (util) | same as NDCG@K above | — | `benchmark/retrieval/metrics_utils.py` | `compute_ndcg()` line ~113 | `compute_metric_summary()` line ~416 |
| Recall@K (util) | same as Recall@K above | — | `benchmark/retrieval/metrics_utils.py` | `compute_recall_at_k()` line ~185 | `compute_metric_summary()` lines ~410–413 |
| Precision@K (util) | same as P@K above | — | `benchmark/retrieval/metrics_utils.py` | `compute_precision_at_k()` line ~235 | `compute_metric_summary()` line ~414 |
| Macro avg | `sum / N` per metric | — | `benchmark/retrieval/metrics_utils.py` | `compute_metric_summary()` line ~329 | adapters' `get_metrics()` and `study_runner.py` |
| Composite score | `gate × (0.40R + 0.25P + 0.20M + 0.15T)` | — | `benchmark/workload/scheduler.py` | `MatrixRunResult.composite_score()` line ~131 | `MatrixAggregator.rank_by_retrieval_strategy()` line ~110 |
| recall_per_ms | `avg_recall / max(avg_latency_p50_ms, 0.001)` | — | `study_runner.py` | `_write_leaderboards_json()` line ~711 | leaderboard JSON writer |
| Bootstrap CI | percentile bootstrap, n=1000 | — | `benchmark/workload/study_aggregator.py` | `StudyAggregator.bootstrap_ci()` line ~289 | `_write_leaderboards_json()` line ~674 |
| sig_vs_next | `ci_low > next_ci_high` | — | `benchmark/workload/study_aggregator.py` | `StudyAggregator.significance_table()` line ~373 | `rank_by_retrieval_strategy()` (strategy CI) |
| P50/P90/P99 latency | nearest-rank percentile | — | `benchmark/orchestrator/scenario_runner.py` | `_build_scenario_metrics()` line ~817 | called after query loop |
| Peak RSS | `getrusage(RUSAGE_SELF).ru_maxrss` | — | `benchmark/memory/adapters/_sys_metrics.py` | `peak_rss_mb()` line ~49 | adapter `get_metrics()` |
| Embed batch size | `floor_pow2(budget // bytes_per_sample)` | — | `benchmark/resources/hw_probe.py` | `embed_batch_size()` line ~77 | `EmbeddingsStrategy.__init__()` |
| recall_lift_vs_none | `avg_recall(reranker) - baseline_recall_none` | — | `benchmark/workload/study_aggregator.py` | `StudyAggregator.rank_by_reranker()` line ~226 | leaderboard JSON |
| Score-estimated pseudo-gold | `{doc_id for r in results if score >= 0.5}` | — | `benchmark/retrieval/metrics_utils.py` | `estimate_relevance_from_scores()` line ~283 | `compute_metric_summary()` fallback |
| ScenarioMetrics assembly | `avg()` over per-query EvaluationResult values | — | `benchmark/orchestrator/scenario_runner.py` | `_build_scenario_metrics()` line ~768 | `ScenarioRunner._run_scenario()` |

---

## 7. How to Add a New Metric

Follow these steps in order.  Steps 1–3 are mandatory; steps 4–7 are required only
if you want the metric to appear in leaderboard output.

**Step 1 — Add an evaluator class** to `benchmark/evaluation/ranking.py`:

```python
class MyNewEvaluator(MetricEvaluator):
    def evaluate(self, retrieved_ids: list[str], expected_ids: list[str]) -> EvaluationResult:
        ...
        return EvaluationResult(metric_name=self.metric_name(), value=..., query_count=1, details={})

    def metric_name(self) -> str:
        return "benchmark.my_new_metric"
```

Implement `MetricEvaluator` from `benchmark/evaluation/base.py`.  Add a `ValueError`
guard when `expected_ids` is empty to match the existing guard pattern.

**Step 2 — Register the evaluator** in `benchmark/application/composer.py`
`_build_evaluators()` (line ~356):

```python
from benchmark.evaluation.ranking import MyNewEvaluator
evaluators.append(MyNewEvaluator(top_k=recall_k))
```

Wrap in a `try/except ImportError` if the evaluator has optional dependencies.

**Step 3 — Add a utility function** to `benchmark/retrieval/metrics_utils.py`
so the score-estimated path (adapter `get_metrics()`) can also compute the metric:

```python
def compute_my_new_metric(results, relevant_doc_ids, k=10) -> float:
    ...
```

Then add it to `compute_metric_summary()` and to the returned dict.

**Step 4 — Add a field to `ScenarioMetrics`** in
`benchmark/orchestrator/scenario_runner.py`:

```python
@dataclass
class ScenarioMetrics:
    ...
    my_new_metric: float = 0.0
```

And populate it in `_build_scenario_metrics()`:

```python
my_new_metric=avg(metrics_by_name.get("benchmark.my_new_metric", [])),
```

**Step 5 — Add a field to the study run result** in
`benchmark/workload/study_aggregator.py` `rank_by_retrieval_strategy()` and
`rank_by_embedding_model()`, aggregating with `statistics.mean()`.

**Step 6 — Add to leaderboard output** in `study_runner.py`
`_write_leaderboards_json()` (line ~678 in accuracy_entries loop):

```python
"avg_my_new_metric": row.get("avg_my_new_metric", 0.0),
```

**Step 7 — Optionally add to `MemoryMetrics`** in
`benchmark/memory/adapters/memory_adapter.py` if the adapter-layer get_metrics()
path should also emit the field.

**Step 8 — Update this document** with a new subsection under §2, a new row in
the §6 table, and any edge cases or denominator rules specific to the new metric.

---

## 8. Known Limitations

### Score-estimated metrics are self-referential (Issue 16)

When `adapter.get_metrics()` is called without gold labels, `compute_metric_summary()`
falls back to `estimate_relevance_from_scores()`, which creates a pseudo-gold set
from results whose `score >= 0.5`.  Because the retriever controls both the scores
and the pseudo-gold set:

- A retriever that returns high-confidence wrong results scores 1.0 on all metrics.
- A retriever that correctly returns low-scoring relevant items scores 0.0.

Do not report score-estimated metrics in benchmark comparisons.  Use them only to
confirm the adapter is functioning (returning any results at all).

### Macro-averaging weights all queries equally

Each query contributes equally to the aggregate metric regardless of its gold-set
size.  A query with one gold memory and a query with fifty gold memories are treated
identically.  This matches standard TREC evaluation practice but may obscure
performance differences on high-recall tasks.

### Recall@1 ≤ Recall@5 ≤ Recall@10 are correlated (Note 19)

These three values are not independent: the same set of retrievals underlies all
three, with different truncation points.  The `accuracy_score` in the leaderboard
uses `avg_recall_at_k` (a single K from the dataset's `evaluation_criteria.recall_k`
field), which means the accuracy ranking has an implicit recall-at-K bias.  Changing
K changes rankings.

### Composite score weights are fixed defaults

`COMPOSITE_WEIGHTS = {recall: 0.40, precision: 0.25, mrr: 0.20, temporal: 0.15}`
are a single set of defaults chosen to reflect the project's primary goal (retrieval
coverage).  Strategies ranked within 0.01 of each other in composite score may
swap rank under different weight assumptions.  Use `composite_score_weighted()` for
sensitivity analysis before citing composite rankings in papers.

### Latency percentiles use nearest-rank, not linear interpolation

`_build_scenario_metrics()` uses `ceil(n * p) - 1` (nearest-rank), while
`_sys_metrics.percentile()` uses linear interpolation.  On small samples (< 20
queries), these can differ by one latency sample.  The nearest-rank formula is
used for P50/P90/P99 in `ScenarioMetrics`; linear interpolation is used in
the adapter-layer `percentile()` helper.
