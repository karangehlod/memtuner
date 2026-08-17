# Artifact Schema

## Purpose

This document describes the current artifact schema emitted by the benchmark CLI, with emphasis on `benchmark analyze`.

It is intended to make the current Phase 1 protocol work concrete and auditable without overstating what the code already guarantees.

## Scope

This document describes the current implemented artifact surface for `benchmark analyze`.

It does not claim that all desired future protocol artifacts already exist. Instead, it distinguishes between:

- currently emitted artifacts
- desirable future protocol artifacts

## Primary Implemented Artifacts

### `effective_config.json`

Purpose:
Captures the effective validated benchmark configuration used for the first successful composed analysis run.

Current behavior:
This file is emitted when the analyze command successfully composes at least one strategy run and is referenced by `benchmark_report.json` through `effective_config_artifact`.

### `environment.json`

Purpose:
Captures execution-environment provenance for the run.

Current behavior:
This file is emitted for completed analyze runs and is referenced by `benchmark_report.json` through `environment_artifact`.

### `run_metadata.json`

Purpose:
Captures protocol-oriented run metadata for audit and reproducibility.

Current behavior:
This file is emitted for completed analyze runs and is referenced by `benchmark_report.json` through `run_metadata_artifact`.

Current key fields include:

- `schema_version`
- `run_hash`
- `command`
- `captured_at`
- `seed`
- `max_queries`
- `llm_judge`
- `output_dir`
- `pack`
- `dataset`
- `run_plan`
- `resources`

Current run-hash rule:

`run_hash` is a deterministic short SHA-256 digest over the protocol identity subset of the run metadata. The current input set includes:

- metadata schema version
- command name
- seed
- `max_queries`
- dataset input path and dataset fingerprint summary fields
- pack name and version
- run-plan config hash and resolved execution fields
- LLM judge enablement and method

Operational note:

`captured_at`, `output_dir`, and resource measurements are intentionally excluded from `run_hash` so the hash remains stable for the same logical run identity.

### `benchmark_report.json`

This is the primary structured output for a completed or failed `benchmark analyze` run.

Current top-level fields include:

- `dataset`
- `run_plan`
- `effective_config_artifact`
- `environment_artifact`
- `run_metadata_artifact`
- `strategy_comparison`
- `embedding_model_comparison`
- `ollama_embedding_model_comparison`
- `hf_inference_embedding_model_comparison`
- `reranker_model_comparison`
- `provider_failures`
- `memory_type_comparison`
- `decay_sweep`
- `decay_response`
- `isolation`
- `artifact_manifest`
- `llm_judge`
- `status`

Failed runs may also include:

- `runtime_error`

#### `dataset`

Current fields:

- `scenario`
- `queries`
- `memories`
- `users`

Purpose:
Summarizes the analyzed dataset at report time.

#### `run_plan`

Purpose:
Captures the effective execution plan used by the analyze command when available.

Current behavior:
The field is present but its exact substructure is defined by command logic rather than a standalone versioned schema document.

#### `effective_config_artifact`

Purpose:
Points to the emitted `effective_config.json` artifact when an effective composed configuration snapshot was captured.

Current behavior:
This field may be `null` when no successful composed strategy run produced a config snapshot.

#### `environment_artifact`

Purpose:
Points to the emitted `environment.json` artifact for the run.

#### `run_metadata_artifact`

Purpose:
Points to the emitted `run_metadata.json` artifact for the run.

#### Comparison Arrays

The following arrays hold result records for different comparison families:

- `strategy_comparison`
- `embedding_model_comparison`
- `ollama_embedding_model_comparison`
- `hf_inference_embedding_model_comparison`
- `reranker_model_comparison`
- `memory_type_comparison`
- `decay_sweep`

Purpose:
Store benchmark result rows produced by the analysis command.

Current note:
These arrays are real outputs, but the repository does not yet expose a single canonical schema specification for every row type. This document serves as the first benchmark-facing schema surface for them.

### Row Schemas For Comparison Arrays

The following row schemas are based on the current implementation in
`benchmark/cli/commands/analyze_command.py`.

#### `strategy_comparison[]`

Each row currently includes:

- `strategy`: retrieval strategy name
- `recall`: scenario `recall_at_k`
- `precision`: scenario `precision_at_k`
- `contamination`: scenario `contamination_rate`
- `temporal`: scenario `temporal_accuracy`
- `mrr`: scenario `mrr`
- `ndcg`: scenario `ndcg`
- `time`: total wall-clock elapsed time in seconds for the strategy run
- `ms_per_query`: elapsed milliseconds per executed query

Conditional fields when `--with-llm-judge` is enabled:

- `llm_judge_score`
- `llm_judge_queries`

#### `embedding_model_comparison[]`

Each row currently includes:

- `model`: concrete model identifier
- `label`: human-readable label used in plots and summaries
- `recall`
- `precision`
- `mrr`
- `ndcg`
- `contamination`
- `ms_per_query`

#### `hf_inference_embedding_model_comparison[]`

Each row currently includes:

- `model`: concrete HF inference model identifier
- `label`: current implementation repeats the model identifier
- `recall`
- `precision`
- `mrr`
- `ndcg`
- `contamination`
- `ms_per_query`

#### `ollama_embedding_model_comparison[]`

Each row currently includes:

- `model`: resolved Ollama model identifier used for execution
- `label`: requested model label shown to the operator
- `recall`
- `precision`
- `mrr`
- `ndcg`
- `contamination`
- `ms_per_query`

#### `reranker_model_comparison[]`

Each row currently includes:

- `model`: configured reranker model name
- `strategy`: configured reranker routing strategy
- `provider_order`: configured provider routing order
- `recall`
- `precision`
- `mrr`
- `ndcg`
- `contamination`
- `ms_per_query`

#### `memory_type_comparison[]`

Each row currently includes:

- `module`: memory module name under comparison
- `recall`: average recall computed across dataset queries
- `memories_stored`: store count after dataset injection

#### `decay_sweep[]`

Each row currently includes:

- `lambda`: exponential decay parameter
- `threshold`: pruning threshold
- `recall`
- `precision`

## Supporting Structured Sections

### `provider_failures`

Current shape:

- top-level object keyed by comparison family or runtime category
- each value is a list of failure rows

Observed top-level keys include:

- `strategy_comparison`
- `hf_inference_embedding_model_comparison`
- `embedding_model_comparison`
- `ollama_embedding_model_comparison`
- `reranker_model_comparison`
- `analyze_runtime`

Failure rows are not fully uniform today, but observed fields include:

- `strategy`
- `model`
- `label`
- `resolved_model`
- `status`
- `error`

Current interpretation:

- `status: skipped` is used for provider or availability gating cases
- omitted `status` generally means the comparison attempt failed during execution

### `decay_response`

Current fields include:

- `formula`: currently `exp(-lambda * age_days)`
- `age_days`: sampled ages used for response plotting
- `curves`: list of per-lambda curve objects

Each `curves[]` row currently includes:

- `lambda`
- `relative_weights`

### `isolation`

Current fields include:

- `rate`: measured isolation rate
- `leaks`: leaked query count

### `artifact_manifest`

The embedded `artifact_manifest` array in `benchmark_report.json` uses the same row shape
as the standalone manifest payload.

Each artifact row currently includes:

- `tag`
- `type`
- `path`
- `description`

Current observed tags include:

- `effective_config`
- `environment`
- `run_metadata`
- `overview_benchmark_analysis`

#### `provider_failures`

Purpose:
Captures provider-specific failure details encountered during analysis.

Current behavior:
This field is a dictionary keyed by comparison family or runtime category.

#### `decay_response`

Purpose:
Stores decay-response metadata and curve outputs used in decay analysis.

Current fields include:

- `formula`
- `age_days`
- `curves`

#### `isolation`

Purpose:
Summarizes user-isolation behavior for interference testing.

Current fields include:

- `rate`
- `leaks`

#### `artifact_manifest`

Purpose:
Lists emitted artifacts associated with the run.

Current behavior:
This field appears both inside `benchmark_report.json` and as the basis for the standalone manifest artifact.

#### `llm_judge`

Purpose:
Records whether answer-quality judging was enabled.

Current fields include:

- `enabled`
- `method`

#### `status`

Purpose:
Records whether the run completed or failed.

Current values observed in the command surface:

- `completed`
- `failed`
- row-level `skipped` values may also occur in intermediate result structures

### `artifact_manifest.json`

This artifact is emitted through tagged JSON writing and summarizes generated artifacts.

Current top-level fields include:

- `artifacts`
- `summary`

The artifacts list contains entries describing generated JSON and image outputs.

Each artifact entry currently includes:

- `tag`
- `type`
- `path`
- `description`

## Primary Figure Artifacts

The analyze command currently emits figure artifacts when plotting dependencies are available.

Observed outputs include:

- `benchmark_analysis.png`
- `strategy_recall_precision.png`
- `embedding_backend_sweep.png`
- `decay_cliff_sweep.png`
- `reranker_provider_comparison.png`

These files are also represented in the artifact manifest when generated.

## Failure Semantics

When a run fails after dataset resolution but before full completion, the command can still emit a partial `benchmark_report.json` with:

- partial comparison results
- partial manifest state
- `status: failed`
- `runtime_error`

This is useful for diagnosis and should be treated as partial evidence, not final benchmark evidence.

## Current Gaps In Artifact Formalization

The current implementation does not yet guarantee separate, first-class artifacts for:

- `run.json`
- `metadata.json`
- `environment.json`
- `config.yaml` as a standardized emitted copy for every run
- a versioned artifact schema identifier embedded in every output

These gaps should be treated as active Phase 1 protocol work.

## Protocol Interpretation Guidance

When writing reports, roadmap status, or paper materials, use the following rule:

- cite currently emitted artifacts as current guarantees
- cite richer provenance and schema separation as Phase 1 targets unless they are implemented and verified

## Relationship To The Roadmap

This document supports:

- Phase 0 by making the benchmark-product documentation surface more concrete
- Phase 1 by defining the current artifact baseline that future protocol work must extend
