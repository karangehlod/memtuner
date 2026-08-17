# Configuration Taxonomy

## Purpose

This document maps the benchmark-product configuration language used in the roadmap to the current implementation surfaces in the codebase.

The roadmap uses benchmark-oriented names such as `ExperimentConfig` and `DatasetConfig` to describe responsibilities. The current implementation already contains most of these responsibilities, but some are grouped under broader models. This document is the canonical mapping between the planning taxonomy and the existing code.

## Current Top-Level Implementation Surface

The current top-level validated configuration model is `BenchmarkConfig` in `benchmark/config/schema.py`.

`BenchmarkConfig` currently contains:

- `memory: MemoryConfig`
- `policies: PoliciesConfig`
- `benchmark: BenchmarkScopeConfig`
- `observability: ObservabilityConfig`
- `answering: AnsweringConfig`

This is the runtime configuration entry point for benchmark YAML.

## Planned Benchmark-Oriented Taxonomy

The roadmap uses the following benchmark-oriented configuration concepts:

- `ExperimentConfig`
- `DatasetConfig`
- `RetrievalConfig`
- `EmbeddingConfig`
- `DecayConfig`
- `MemoryConfig`
- `OutputConfig`

These names represent configuration responsibilities, not necessarily one-to-one current classes.

## Mapping: Roadmap Taxonomy To Current Code

### ExperimentConfig

Benchmark-product meaning:
The complete experiment definition used to run a benchmark and reproduce it later.

Current implementation mapping:
- `benchmark.config.schema.BenchmarkConfig`
- especially:
  - `BenchmarkConfig.benchmark`
  - `BenchmarkConfig.memory`
  - `BenchmarkConfig.policies`
  - `BenchmarkConfig.answering`
  - `BenchmarkConfig.observability`

Notes:
`ExperimentConfig` is not currently a concrete class name. The closest concrete implementation is `BenchmarkConfig`.

### DatasetConfig

Benchmark-product meaning:
Dataset identity, dataset source, adapter selection, split semantics, and dataset-specific validation inputs.

Current implementation mapping:
- No standalone `DatasetConfig` class currently exists in `benchmark/config/schema.py`.
- Dataset selection currently happens primarily through CLI arguments and pack resolution, for example in `benchmark analyze`.

Notes:
This is a real taxonomy gap between the roadmap language and the current implementation. The responsibility exists operationally, but it is not yet formalized as a validated config model.

### RetrievalConfig

Benchmark-product meaning:
The retrieval family selection and its strategy-specific parameters.

Current implementation mapping:
- `benchmark.config.schema.RetrievalConfig`
- `benchmark.config.schema.BenchmarkScopeConfig.retrieval_strategy`
- strategy-specific models:
  - `EmbeddingsRetrievalConfig`
  - `HfInferenceRetrievalConfig`
  - `OllamaRetrievalConfig`
  - `HybridRetrievalConfig`
  - `EmptyRetrievalConfig`

Notes:
This responsibility is already strongly represented in the current implementation.

### EmbeddingConfig

Benchmark-product meaning:
Embedding-model identity, routing mode, provider preference, cache behavior, and related runtime settings used for embedding-based retrieval.

Current implementation mapping:
- Primarily the embedding-oriented submodels inside `RetrievalConfig`:
  - `EmbeddingsRetrievalConfig`
  - `HfInferenceRetrievalConfig`
  - `OllamaRetrievalConfig`

Notes:
There is not yet a standalone top-level `EmbeddingConfig`. Embedding settings are currently grouped under retrieval configuration.

### DecayConfig

Benchmark-product meaning:
Decay function type and decay-rate parameters used by memory policies.

Current implementation mapping:
- `benchmark.config.schema.DecayConfig`
- nested via `ModulePolicyConfig.decay`

Notes:
This is already a concrete validated class and directly matches the roadmap concept.

### MemoryConfig

Benchmark-product meaning:
Memory-module enablement and memory-system selection.

Current implementation mapping:
- `benchmark.config.schema.MemoryConfig`
- `benchmark.config.schema.MemorySelectionConfig`

Notes:
This is already a concrete validated class and directly matches the roadmap concept.

### OutputConfig

Benchmark-product meaning:
Output location, output schemas, report destinations, artifact retention policy, and benchmark-result packaging behavior.

Current implementation mapping:
- No standalone `OutputConfig` class currently exists in `benchmark/config/schema.py`.
- Output location and artifact emission are currently handled largely through CLI arguments and command logic, especially in `benchmark/cli/commands/analyze_command.py`.

Notes:
This is another genuine taxonomy gap between the roadmap language and the current code.

## Supporting Runtime Models Outside The Core Taxonomy

The current schema also contains important supporting models:

- `PoliciesConfig`
- `ModulePolicyConfig`
- `PruningConfig`
- `RerankerConfig`
- `BenchmarkScopeConfig`
- `ObservabilityConfig`
- `AnsweringConfig`

These models are already important for benchmark execution even if the roadmap taxonomy groups them under broader benchmark-product concepts.

## Recommended Interpretation

For roadmap and documentation purposes, interpret the current implementation as follows:

- `BenchmarkConfig` is the effective current `ExperimentConfig`
- Dataset configuration is currently CLI-driven and should eventually become a validated `DatasetConfig`
- Retrieval and embedding concerns are currently split between `BenchmarkScopeConfig.retrieval_strategy` and `RetrievalConfig`
- Output behavior is currently command-driven and should eventually become a validated `OutputConfig`

## Concrete Gaps To Track

The following benchmark-product concepts are not yet first-class validated config models:

- `DatasetConfig`
- `EmbeddingConfig` as a standalone top-level model
- `OutputConfig`
- `ExperimentConfig` as a benchmark-facing alias or dedicated model name

## Relationship To The Roadmap

This document satisfies the Phase 0 requirement to document the configuration taxonomy and map benchmark-product concepts to actual implementation surfaces.

It also prepares later work if the project chooses to rename or refactor schema models for clearer benchmark-facing semantics.
