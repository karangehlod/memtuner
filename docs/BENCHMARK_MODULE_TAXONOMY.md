# Benchmark Module Taxonomy

## Purpose

This document maps the benchmark-facing capabilities described in the roadmap to the
actual package layout that exists today. It is intended to keep planning, protocol,
and implementation references aligned.

The taxonomy in this document is descriptive, not aspirational. If a benchmark-facing
concern is not represented by a concrete package or module today, that gap should be
called out explicitly rather than implied away.

## Alignment Principles

The current package layout already reflects the main architecture rules documented in
the repository guidance:

- CLI translates user intent into benchmark commands and does not own benchmark logic.
- The orchestrator coordinates execution order and delegates computation.
- Config loading and validation are isolated from execution.
- Gold truth remains read-only and independent from memory implementations.
- Evaluation consumes benchmark outputs and expected results rather than memory internals.
- Reporting formats pre-computed results rather than recomputing benchmark metrics.

## Benchmark-Facing Capability Map

### CLI and command surface

Owning package: `benchmark/cli`

- `benchmark/cli/main.py` is the top-level click entry point.
- `benchmark/cli/commands/` contains the concrete command surfaces such as `run`,
  `analyze`, `report`, `compare`, `validate`, and data-preparation helpers.

Benchmark interpretation:

- This is the public execution surface for benchmark operators.
- It should translate operator intent into validated execution without embedding
  benchmark scoring or protocol logic.

### Configuration and validation

Owning package: `benchmark/config`

- `loader.py` owns YAML loading and environment hydration.
- `schema.py` defines the validated benchmark configuration model.
- `validation.py` contains configuration checks beyond raw parsing.
- `defaults.py` produces baseline config templates.

Benchmark interpretation:

- This package defines the effective benchmark input contract.
- Roadmap terms such as experiment configuration, retrieval configuration,
  reranker configuration, decay configuration, and observability configuration
  should be mapped back to this package rather than described as separate systems
  unless separate systems actually exist.

### Orchestration and execution flow

Owning package: `benchmark/orchestrator`

- `benchmark_runner.py` is the thin top-level benchmark coordinator.
- `scenario_runner.py` executes a scenario day-by-day, including event injection,
  lifecycle policy application, querying, evaluation, and cost capture.

Benchmark interpretation:

- This package owns when execution stages occur.
- It should not own metric semantics, dataset truth, storage internals, or report
  formatting.
- For roadmap and protocol documents, this is the benchmark runtime coordinator,
  not the benchmark definition layer.

### Factory and runtime resolution

Owning package: `benchmark/factory`

- Registry and resolver modules translate validated configuration into concrete
  implementations.
- `resolver.py` shows that enabled memory modules and retrieval strategies are
  resolved through registries rather than hard-coded execution branches.

Benchmark interpretation:

- This package is the wiring boundary between benchmark configuration and concrete
  implementations.
- It is the correct place to describe provider or strategy resolution behavior.
- It is not the correct place to define evaluation semantics.

### Gold truth and dataset normalization

Owning package: `benchmark/gold`

- `oracle.py` is the read-only gold dataset repository.
- Gold loading handles both native benchmark datasets and LoCoMo-shaped inputs.
- Timestamp normalization is applied here before evaluation proceeds.

Benchmark interpretation:

- This package is the canonical owner of benchmark truth data and query expectations.
- Dataset loading, conversion, normalization, and expected-answer access belong here.
- Benchmark papers and protocol docs should describe this as the truth layer, not as
  part of memory storage or orchestration.

### Scenario specification and workload shape

Owning packages: `benchmark/scenario`, `benchmark/workload`, `benchmark/packs`

- `benchmark/scenario` defines scenario abstractions and executable scenario inputs.
- `benchmark/workload` and `benchmark/packs` are the nearest current homes for
  benchmark pack composition and workload packaging.

Benchmark interpretation:

- This is the closest current implementation surface to benchmark suite definition.
- If future roadmap language introduces benchmark packs, dataset suites, or scenario
  families, those terms should anchor here unless the implementation moves.

### Memory implementations and lifecycle behavior

Owning packages: `benchmark/memory`, `benchmark/memory/interfaces`

- Memory readers, writers, and lifecycle policies are exposed through segregated
  interfaces.
- Concrete implementations are benchmark participants, not benchmark controllers.

Benchmark interpretation:

- This is the participant-under-test surface.
- Benchmark definitions, scoring, and protocol fairness rules should not be embedded
  here.
- Roadmap language about short-term memory, long-term memory, pruning, and decay
  should distinguish between benchmark policies and memory implementation details.

### Evaluation and metric semantics

Owning package: `benchmark/evaluation`

- Evaluation context, evaluators, aggregation, and reliability calculations live here.
- Scenario execution depends on this package to score results after retrieval/query
  execution occurs.

Benchmark interpretation:

- This package is the owner of benchmark scoring logic.
- Metric definitions in benchmark docs should point here as the execution-time home
  of those semantics.
- It should remain independent from report formatting and CLI concerns.

### Cost accounting

Owning package: `benchmark/cost`

- Storage and token cost calculators are applied during scenario execution.
- Tracker services summarize execution costs for final benchmark output.

Benchmark interpretation:

- This package owns economic measurement, not benchmark scoring.
- Cost-aware benchmark claims should map to this package and its emitted summaries.

### Reporting and output projection

Owning package: `benchmark/reporting`

- `json_report.py` serializes `BenchmarkRunResult`.
- `summary.py` generates a human-readable textual summary.
- Other report modules project already-computed benchmark outputs into comparison or
  export views.

Benchmark interpretation:

- This package owns output projection and serialization.
- It should not recompute benchmark results.
- Artifact-schema documentation should map report fields to the models and writers in
  this package, plus any additional files emitted by CLI analysis commands.

### Result models and benchmark data contracts

Owning package: `benchmark/models`

- Core benchmark result models, query models, response models, and memory event models
  live here.

Benchmark interpretation:

- This package is the closest concrete implementation of benchmark data contracts.
- It should remain pure data and should be referenced whenever docs describe stable
  result fields or event shapes.

### Time simulation

Owning package: `benchmark/time`

- Time providers abstract simulated clock behavior.

Benchmark interpretation:

- Temporal correctness and decay experiments depend on this package for deterministic
  time progression.
- Benchmark documents should describe simulated time as an explicit subsystem rather
  than an implicit assumption.

### Observability

Owning package: `benchmark/observability`

- Spans, structured logs, and trace-linked decisions are emitted through this package.

Benchmark interpretation:

- This package supports benchmark execution transparency and debugging.
- It is cross-cutting and should not be described as owning business logic.

## Practical Boundary Summary

The current benchmark-facing architecture can be summarized as follows:

- CLI receives commands.
- Config validates and hydrates benchmark inputs.
- Factory resolves concrete participants.
- Gold loads truth data.
- Orchestrator coordinates execution.
- Memory modules act as systems under test.
- Evaluation computes benchmark metrics.
- Cost records execution economics.
- Reporting projects results into artifacts.
- Observability records how the run behaved.

## Current Taxonomy Gaps

The roadmap language is slightly ahead of the current package naming in a few places:

- There is not yet a dedicated `dataset` package; dataset truth currently lives under
  `benchmark/gold` with scenario and workload support nearby.
- There is not yet a dedicated protocol artifact package; protocol outputs are
  currently produced primarily by CLI command implementations plus reporting writers.
- There is not yet a single benchmark-specification module in code; the benchmark
  product definition is primarily a documentation concern layered over the execution
  architecture.

These gaps are acceptable as long as documentation remains explicit about the current
owners.

## Documentation Usage Rules

When benchmark docs reference implementation surfaces, prefer the following mapping:

- Protocol execution flow: `benchmark/cli` + `benchmark/orchestrator`
- Config contract: `benchmark/config`
- Gold truth and dataset loading: `benchmark/gold`
- Metric computation: `benchmark/evaluation`
- Result and artifact projection: `benchmark/reporting` + `benchmark/models`
- Participant implementations: `benchmark/memory`
- Deterministic time behavior: `benchmark/time`
- Runtime wiring: `benchmark/factory`

This mapping should be updated when code ownership actually changes.