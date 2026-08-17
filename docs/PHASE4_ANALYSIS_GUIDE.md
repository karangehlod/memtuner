# Phase 4: Retrieval Benchmarking - Analysis & Interpretation Guide

This guide explains how to interpret Phase 4 benchmark results and make data-driven decisions about retrieval strategy selection.

---

## Quick Start: Understanding Results

### Sample Leaderboard Output
```
Rank | Strategy         | Recall@10 | Latency | Index Size | Score
-----|------------------|-----------|---------|------------|-------
  1. | bm25            | 1.000     | 0.05ms  | 200MB     | 1.000
  2. | learned_dense   | 0.890     | 4.00ms  | 2GB       | 0.915
  3. | cascading       | 0.860     | 3.50ms  | 200MB     | 0.901
  4. | dense_vector    | 0.845     | 3.00ms  | 1GB       | 0.915
  5. | boolean         | 1.000     | 0.01ms  | 10MB      | 0.970
```

### What Each Column Means

**Rank:** Position based on composite score (balanced across metrics)

**Strategy:** Retrieval algorithm name

**Recall@10:** % of relevant documents in top-10 results
- 1.000 = Perfect (all relevant docs found in top-10)
- 0.500 = Half of relevant docs in top-10
- Use: How good are the results?

**Latency:** Average milliseconds per query
- 0.01ms = Extremely fast (Boolean exact match)
- 1.00ms = Fast (BM25)
- 4.00ms = Moderate (Deep learning)
- Use: How fast is the system?

**Index Size:** Total memory for indices
- 10MB = Tiny (Boolean)
- 200MB = Small (BM25)
- 2GB = Large (Deep learning)
- Use: How much memory needed?

**Score:** Composite score (0-1)
- Combines recall (50%), speed (30%), reliability (20%)
- Higher is better overall

---

## How to Read Benchmark Results

### Scenario 1: "Which strategy should we use?"

**Decision Tree:**
1. **Fast AND accurate?** → Cascading (0.86 recall, 3.5ms)
2. **Maximum accuracy?** → LearnedDense (0.89 recall, but 4ms + 2GB)
3. **Memory-constrained?** → Boolean (10MB) or Quantized (256MB)
4. **Mobile device?** → Quantized (ultra-small) or Boolean
5. **Default choice?** → BM25 (always fast, usually accurate)

### Scenario 2: "Why is Strategy X slow?"

**Analysis:**
- Dense strategies (Dense, LearnedDense, etc.) = Need model inference
- Multi-stage strategies (Cascading, RetrievalRerank) = Run multiple passes
- ChainSearch = Runs 3 separate retrievers (slowest)

**Solution:**
- Use Quantized for faster inference
- Use Boolean as pre-filter
- Use Cascading instead of HybridFusion

### Scenario 3: "Why is recall low?"

**Common Causes:**
- **Boolean:** Requires ALL query terms → use BM25 instead
- **TF-IDF:** Simple scoring → use Dense for better semantics
- **Dense without fine-tuning:** Limited to pre-trained model capability
- **ANN approximation:** Trade-off for speed → use exact search if accuracy critical

**Solutions:**
- Use LearnedDense for best recall
- Use HybridFusion or ChainSearch to combine signals
- Expand queries with synonyms
- Use cascading (multi-stage) for better results

### Scenario 4: "Why is index so large?"

**Common Causes:**
- Dense embedding: 384-768 dimensions × corpus size
- Multiple indices: HybridFusion stores both sparse + dense
- String representation: Some overhead from serialization

**Solutions:**
- Use Quantized (4x smaller, minimal recall loss)
- Use CascadingAdapter (only keeps sparse in memory)
- Use BM25 (smallest dense-free alternative)

---

## Detailed Metric Explanations

### Accuracy Metrics

#### Recall@10
"Of all relevant documents, what fraction appears in top-10?"

```
Example:
Total relevant docs = 100
Docs in top-10 = 95
Recall@10 = 95/100 = 0.95
```

**When to optimize:**
- Critical search applications
- Research systems where missing results is bad
- When false negatives are expensive

**Trade-off:** Usually increases latency

#### NDCG (Normalized Discounted Cumulative Gain)
"How good is the ranking order?"

- Considers rank position
- Position 1 worth more than position 10
- Score 1.0 = Perfect ranking
- Score 0.5 = Mediocre ranking

**When to optimize:**
- When user clicks top results most
- When ranking order matters

**Example:**
```
Perfect ranking:  [Relevant, Relevant, Relevant, Irrelevant] = 1.0
Bad ranking:      [Irrelevant, Relevant, Relevant, Relevant] = 0.6
```

#### Precision@10
"Of the top-10 results, how many are relevant?"

```
Top-10 results = [Relevant, Relevant, Relevant, Irrelevant, ...]
Precision@10 = 3/10 = 0.30
```

**When to optimize:**
- User experience (fewer irrelevant results)
- When showing results to users
- Reducing cognitive load

#### MRR (Mean Reciprocal Rank)
"Where is the first relevant result?"

```
Position 1:  MRR = 1.0
Position 5:  MRR = 1/5 = 0.2
Position 50: MRR = 1/50 = 0.02
```

**When to optimize:**
- Navigation/FAQ retrieval
- When user needs one good answer

### Efficiency Metrics

#### Query Latency
"Milliseconds per query"

**Latency Budget (typical targets):**
- < 1ms: Real-time, interactive
- 1-10ms: Fast web response
- 10-100ms: Acceptable background
- > 100ms: Slow, may need optimization

**How to improve:**
- Smaller indices (Quantized, Boolean)
- Faster algorithms (ANN vs exact)
- Caching frequent queries
- Parallel processing

#### Index Build Time
"Seconds to build/initialize"

**Why it matters:**
- One-time cost
- Matters for live updates
- Affects deployment time

#### Index Size
"Bytes of total memory"

**Rule of thumb:**
- Dense: ~1GB per 100k docs
- Sparse: ~10-20MB per 100k docs
- Budget: How much memory available?

### Reliability Metrics

#### Success Rate
"Fraction of queries that completed without errors"

- 1.0 = No errors
- 0.99 = 1 failure per 100 queries
- 0.95 = 5 failures per 100 queries

**When low:**
- External service unavailable
- Input validation failed
- Resource exhaustion

**How to improve:**
- Use fallback implementations (all adapters have them)
- Add retry logic with exponential backoff
- Monitor and alert on failures

---

## Comparative Analysis Examples

### Example 1: Speed vs Accuracy Trade-off

**Scenario:** Need both good recall AND fast response

**Analysis:**
```
Strategy        Recall   Latency   Score   Recommendation
BM25            0.85     0.05ms    1.00    ✓ Best all-around
LearnedDense    0.89     4.00ms    0.91    ✗ Too slow
Cascading       0.86     3.50ms    0.90    ✓ Good compromise
ANN             0.84     0.50ms    0.88    ✗ Slightly lower recall
```

**Decision:** Use **Cascading** for best recall@10 with acceptable latency

### Example 2: Memory-Constrained Device

**Scenario:** Edge device with only 500MB available

**Analysis:**
```
Strategy        Index   Feasible?   Note
Boolean         10MB    ✓ Yes       But very low recall
Quantized       256MB   ✓ Yes       Good balance
BM25            200MB   ✓ Yes       Excellent choice
ANN             1GB     ✗ No        Too large
LearnedDense    2GB     ✗ No        Far too large
```

**Decision:** Use **BM25** - fast, accurate, fits in memory

### Example 3: High-Precision Requirement

**Scenario:** Filter results before human review, false positives costly

**Analysis:**
```
Strategy           Precision   Latency   Use Reranking?
BM25              1.0         0.05ms    No - already perfect
RetrievalRerank   0.95        6.70ms    Yes - neural verification
HybridFusion      0.92        5.60ms    Maybe - dual confirmation
```

**Decision:** Use **RetrievalRerank** - highest confidence in results

---

## Cross-Dataset Analysis

### Pattern: Performance Varies by Data

From sample benchmark:
```
Dataset         BM25    Dense   Cascading   Winner
qa_dataset      1.000   0.875   0.858      BM25 (keyword match)
news_dataset    1.000   0.930   0.912      BM25 (exact terms)
domain_dataset  1.000   0.827   0.810      BM25 (technical terms)
generic_dataset 1.000   0.905   0.886      BM25 (general vocab)
```

**Insights:**
1. BM25 dominates when queries have exact keywords in documents
2. Dense methods better when query reformulation helps
3. Cascading consistently strong (middle-ground)

### Pattern: Index Size vs Recall

```
Index Size    Best Strategy    Typical Recall
10MB          Boolean          0.40-0.50
200MB         BM25            0.82-1.00
256MB         Quantized       0.83
1GB           Dense           0.84
2GB           LearnedDense    0.89
3GB+          ChainSearch     0.88+
```

**Key:** More memory generally = better accuracy, but with diminishing returns

---

## Making Decisions: Decision Matrix

### Step 1: Define Your Constraints

| Constraint | Value | Why? |
|-----------|-------|------|
| Max Latency | 100ms | Web response time |
| Max Index Size | 500MB | Available memory |
| Min Recall | 0.80 | Acceptable quality |
| Cost Budget | Medium | Infrastructure costs |
| Deployment | Cloud | Can scale |

### Step 2: Filter Candidates

```python
candidates = []
for strategy in all_strategies:
    if strategy.latency_ms <= 100 and \
       strategy.index_size_bytes <= 500_000_000 and \
       strategy.recall >= 0.80:
        candidates.append(strategy)
```

### Step 3: Rank by Score

```python
candidates.sort(key=lambda s: s.score, reverse=True)
best = candidates[0]
print(f"Recommended: {best.name}")
```

### Example Application

**Web Search Application:**
```
Constraints:
- Latency: 10ms (responsive)
- Index Size: 1GB (standard server)
- Recall: 0.85 (good quality)
- Queries/sec: 1000s (load)

Candidates Passing Filter:
1. BM25 (0.05ms, 200MB, 0.95 recall) ✓ WINNER
2. ANN (0.50ms, 1GB, 0.84 recall) ✓
3. TF-IDF (0.80ms, 150MB, 0.71 recall) ✗ Recall too low

Recommendation: BM25
Reasoning: Fastest, fits in memory, excellent recall, handles load
```

---

## Optimization Strategies

### To Improve Speed

1. **Use faster strategy:**
   - Boolean (0.01ms) → Cascading (3.5ms) → LearnedDense (4.0ms)

2. **Pre-filter candidates:**
   - Use Boolean first to eliminate obviously wrong docs
   - Then apply expensive retrievers to candidates

3. **Parallelize:**
   - Run multiple queries in parallel
   - Use GPU for dense embeddings

4. **Cache results:**
   - Store frequent query results
   - Implement query normalization

### To Improve Recall

1. **Use better model:**
   - Boolean → BM25 → DenseVector → LearnedDense

2. **Combine signals:**
   - HybridFusion (sparse + dense)
   - ChainSearch (3 chains)

3. **Multi-stage:**
   - Fast filter → slow ranker
   - Recall-oriented retrieval → precision-oriented reranking

4. **Query expansion:**
   - Add synonyms
   - Use LLM for semantic expansion

### To Reduce Memory

1. **Quantize embeddings:**
   - LearnedDense (2GB) → Quantized (256MB) = 8x reduction

2. **Use sparse only:**
   - HybridFusion (3GB) → BM25 (200MB)
   - Cascading (only keeps sparse in memory during queries)

3. **Remove unused indices:**
   - Keep only primary retriever
   - Load reranker on-demand

---

## Debugging: Common Issues

### Issue: "Recall is unexpectedly low"

**Diagnosis:**
```
1. Check strategy chosen
   - Boolean: Does every query term appear? Try BM25 instead
   - TF-IDF: Simple scoring? Try Dense for semantics
   
2. Check dataset
   - Are relevant docs vocabulary-similar to query?
   - Or do they require semantic understanding?
   
3. Check metrics
   - Is recall@100 acceptable even if @10 is low?
   - Might need multi-stage retrieval
```

### Issue: "System is too slow"

**Diagnosis:**
```
1. Measure latency
   - Which strategy is slow? (check latency column)
   - Dense strategies: Expect 3-4ms for inference
   
2. Find bottleneck
   - Index lookup? → Use ANN instead of exact
   - Inference? → Use Quantized or BM25
   - I/O? → Increase batch size
   
3. Optimize
   - Cascading: Filter-then-rank is faster than all-features
   - Boolean: Pre-filter eliminates 90% of docs
   - Caching: Common queries very fast on repeat
```

### Issue: "Index is too large"

**Diagnosis:**
```
1. Check strategy
   - Is it dense (embeddings)? Large by nature
   - Multiple indices? Cascading keeps only one
   
2. Options
   - Quantized: 4x smaller, 1% recall loss
   - Boolean/BM25: Sparse, very small
   - Cascading: Only sparse kept in memory
   
3. Trade-offs
   - Smaller = Faster + Less memory
   - But might need sacrifice on recall
```

---

## Advanced Topics

### Composite Score Deep Dive

The score formula weights three dimensions:

```
Score = 0.50 × Accuracy + 0.30 × Efficiency + 0.20 × Reliability

Example for BM25:
Accuracy   = 0.3×(0.85) + 0.2×(0.92) + 0.3×(0.87) + 0.2×(0.83) = 0.857
Efficiency = 0.6×(0.9995) + 0.4×(0.998) = 0.9985
Reliability= 0.99
Score     = 0.50×0.857 + 0.30×0.9985 + 0.20×0.99 = 0.928
```

**Customizing weights:**

If you care more about speed:
```
Score = 0.30 × Accuracy + 0.50 × Efficiency + 0.20 × Reliability
```

If you care more about reliability:
```
Score = 0.40 × Accuracy + 0.20 × Efficiency + 0.40 × Reliability
```

### Benchmarking Your Own Data

```python
from benchmark.retrieval.benchmark_orchestrator import RetrievalBenchmarkOrchestrator

orchestrator = RetrievalBenchmarkOrchestrator()

# Your data
my_docs = [{"id": "doc_1", "content": "..."}, ...]
my_queries = ["query_1", "query_2", ...]

# Run benchmark
result = orchestrator.benchmark_all_strategies(
    my_docs,
    my_queries,
    dataset_name="my_dataset",
)

# Get results
leaderboard = orchestrator.get_leaderboard("my_dataset")
summary = orchestrator.get_summary()

# Export
json_results = orchestrator.export_results("my_dataset", format="json")
csv_results = orchestrator.export_results("my_dataset", format="csv")
```

---

## Conclusion

Phase 4 provides all data needed to make informed retrieval strategy choices. Use this guide to:

1. **Understand metrics** - Know what each number means
2. **Analyze trade-offs** - Speed vs accuracy vs memory
3. **Make decisions** - Choose the right strategy for your use case
4. **Optimize performance** - Know where improvements can be made
5. **Debug issues** - Understand what went wrong and why

**Key Takeaway:** There's no one-size-fits-all strategy. Choose based on your specific constraints and requirements using the analysis tools provided.
