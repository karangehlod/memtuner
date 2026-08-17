# DatasetProfiler — Specification

## Purpose

`DatasetProfiler` computes deterministic, measurable characteristics of a dataset **before any retrieval runs**. These characteristics describe the dataset's nature and drive downstream pattern analysis across benchmark results.

The goal is to answer: *"What kind of workload is this dataset, and which retrieval configuration fits it best?"*

---

## Pipeline

```
dataset (JSON)
      ↓
DatasetProfiler          ← computed from query + gold memory only
      ↓
dataset_profile.json     ← versioned, checksummed
      ↓
benchmark grid           ← study_runner.py phases 1–5
      ↓
results.csv
      ↓
PatternExtractor         ← correlates profile characteristics with metric outcomes
      ↓
workload → best config → sensitivity
      ↓
PatternComparator        ← cross-dataset patterns
      ↓
deployment guidance
```

---

## V1 Characteristics

All characteristics are computed from **query text + gold memory text only**. No retrieval results are used. This prevents circularity (see below).

### Lexical Fields

| Field | Type | Definition |
|---|---|---|
| `lexical_density` | float [0,1] | Fraction of queries where ≥1 content word appears verbatim in the gold memory |
| `query_paraphrase_rate` | float [0,1] | Fraction of queries with no exact n-gram overlap (n≥3) with their gold memory — proxy for paraphrase difficulty |
| `avg_query_length_tokens` | float | Mean token count of queries |
| `avg_memory_length_tokens` | float | Mean token count of memory entries |
| `query_length_p90_tokens` | float | 90th percentile query length |

### Temporal Fields

| Field | Type | Definition |
|---|---|---|
| `temporal_spread_days` | int | Days between earliest and latest memory timestamp (0 if no timestamps) |
| `temporal_recency_30d` | float [0,1] | Fraction of gold memories dated within 30 days of the query timestamp |
| `has_timestamps` | bool | Whether the dataset contains usable timestamp fields |

### Memory Type Mix

| Field | Type | Definition |
|---|---|---|
| `memory_type_mix.episodic` | float [0,1] | Fraction of memories typed as episodic |
| `memory_type_mix.semantic` | float [0,1] | Fraction of memories typed as semantic |
| `memory_type_mix.preference` | float [0,1] | Fraction of memories typed as preference |
| `memory_type_mix.entity` | float [0,1] | Fraction of memories typed as entity |

---

## Critical Constraint: No Circularity

Characteristics must be computed **only from query + gold memory**, never from retrieval results.

**Wrong** (circular):
```
lexical_density = % queries where BM25 retrieves exact-match memory
                          ↑ uses retrieval result
```

**Correct**:
```
lexical_density = % queries where ≥1 content word from the query
                  appears verbatim in the gold memory text
                          ↑ uses only query + gold label
```

The causal direction must be:
```
query + gold memory
        ↓
dataset characteristics        (DatasetProfiler)
        ↓
retrieval experiments          (benchmark grid)
        ↓
results + patterns             (PatternExtractor / PatternComparator)
```

---

## Output Format

Profiles are versioned JSON. Every result can be traced back to `dataset → profile → experiment → configuration → metric`.

```json
{
  "dataset": "locomo10",
  "profile_version": "1.0",
  "computed_at": "2026-08-11T00:00:00Z",
  "dataset_checksum": "sha256:abc123...",
  "query_count": 500,
  "characteristics": {
    "lexical_density": 0.42,
    "query_paraphrase_rate": 0.74,
    "avg_query_length_tokens": 12.3,
    "avg_memory_length_tokens": 48.7,
    "query_length_p90_tokens": 22.0,
    "temporal_spread_days": 294,
    "temporal_recency_30d": 0.61,
    "has_timestamps": true
  },
  "memory_type_mix": {
    "episodic": 0.71,
    "semantic": 0.18,
    "preference": 0.11,
    "entity": 0.00
  }
}
```

---

## Implementation Sequence

Build in this order — each step depends on the previous:

1. **`DatasetProfiler`** — computes and writes `dataset_profile.json` per dataset
2. **`PatternExtractor`** — joins profiles with `results.csv`, extracts correlations between characteristics and metric outcomes, produces `workload → best config → sensitivity` summary
3. **`PatternComparator`** — compares patterns across datasets, surfaces configuration rules that generalize (e.g. "high `query_paraphrase_rate` → hybrid strategy outperforms BM25 by >8 recall points in 4/5 datasets")

---

## What NOT to Build in V1

| Item | Reason to defer |
|---|---|
| ML classifier for workload types | Need cross-dataset pattern data first; classify after patterns are confirmed |
| LLM-generated characteristics | Non-deterministic, hard to version, slow |
| Runtime profiling (latency profiling during retrieval) | Mixes profiling with benchmarking — keep separate |
| Answer-quality / RAGAS metrics | Out of scope for dataset profiling; belongs in judge layer |
| Automatic deployment optimizer | Build only if cross-dataset patterns are strong enough to generalize |

Start with **deterministic profiling + pandas analysis**. If the resulting patterns are strong, a learned workload→configuration recommender can be considered later.

---

## File Locations (Proposed)

```
benchmark/
  profiler/
    dataset_profiler.py     ← DatasetProfiler class
    pattern_extractor.py    ← PatternExtractor class
    pattern_comparator.py   ← PatternComparator class
data/
  profiles/
    locomo10_profile.json
    coqa_gold_profile.json
    squad_gold_profile.json
    longmemeval_oracle_gold_profile.json
    synthetic_gold_profile.json
```

---

## Implementation TODOs

- [ ] `DatasetProfiler.compute(dataset_path) -> DatasetProfile`
- [ ] Lexical density: tokenize with same tokenizer used for BM25 (rank-bm25 tokenizer) for consistency
- [ ] Paraphrase rate: use character 3-gram overlap as proxy (no model needed)
- [ ] Temporal fields: attempt to parse `timestamp`, `created_at`, `date` fields; set `has_timestamps=False` if none found
- [ ] Memory type mix: read from `memory_type` field; default to `episodic` if missing
- [ ] Write profile to `data/profiles/<dataset_stem>_profile.json`
- [ ] `PatternExtractor.extract(profiles_dir, results_csv) -> PatternReport`
- [ ] `PatternComparator.compare(pattern_reports) -> CrossDatasetSummary`
- [ ] Integrate: auto-run `DatasetProfiler` in `study_runner.py` before Phase 1 if profile not already present
