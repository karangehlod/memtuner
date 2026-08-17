"""OTel span and metric name constants.

These are FIXED and VERSIONED — do not change without a version bump.
All metric and span names used in the benchmark must be defined here.
"""

from __future__ import annotations

# --- Span Names (Trace Taxonomy) ---

SPAN_BENCHMARK_RUN = "benchmark.run"
SPAN_SCENARIO_RUN = "scenario.run"
SPAN_DATASET_DAY = "dataset_day"
SPAN_QUERY = "query"
SPAN_MEMORY_WRITE = "memory.write"
SPAN_MEMORY_READ = "memory.read"
SPAN_STM_READ = "stm.read"
SPAN_LTM_READ = "ltm.read"
SPAN_MEMORY_PRUNE = "memory.prune"
SPAN_EVALUATION = "evaluation"
SPAN_COST_TRACK = "cost.track"

# --- Metric Names ---

METRIC_RECALL_AT_K = "benchmark.recall_at_k"
METRIC_CONTAMINATION_RATE = "benchmark.contamination_rate"
METRIC_TEMPORAL_ACCURACY = "benchmark.temporal_accuracy"
METRIC_LATENCY_MS = "benchmark.latency_ms"
METRIC_COST_PER_CORRECT_RECALL = "benchmark.cost_per_correct_recall"
METRIC_MEMORY_SURVIVAL_RATE = "benchmark.memory_survival_rate"

# --- Required Span Attributes ---

ATTR_RUN_ID = "run_id"
ATTR_SCENARIO = "scenario"
ATTR_MEMORY_MODULES_ENABLED = "memory_modules_enabled"
ATTR_CONFIG_HASH = "config_hash"
ATTR_DATASET_DAY = "dataset_day"
ATTR_MODULE_NAME = "module_name"
ATTR_QUERY_TEXT = "query_text"
ATTR_TOP_K = "top_k"

# --- Schema Version ---

SCHEMA_VERSION = "1.0.0"

# --- All Metric Names (for validation) ---

ALL_METRIC_NAMES = frozenset(
    {
        METRIC_RECALL_AT_K,
        METRIC_CONTAMINATION_RATE,
        METRIC_TEMPORAL_ACCURACY,
        METRIC_LATENCY_MS,
        METRIC_COST_PER_CORRECT_RECALL,
        METRIC_MEMORY_SURVIVAL_RATE,
    }
)

# --- All Span Names (for validation) ---

ALL_SPAN_NAMES = frozenset(
    {
        SPAN_BENCHMARK_RUN,
        SPAN_SCENARIO_RUN,
        SPAN_DATASET_DAY,
        SPAN_QUERY,
        SPAN_MEMORY_WRITE,
        SPAN_MEMORY_READ,
        SPAN_STM_READ,
        SPAN_LTM_READ,
        SPAN_MEMORY_PRUNE,
        SPAN_EVALUATION,
        SPAN_COST_TRACK,
    }
)
