# Evaluation Protocol

## Purpose

This document defines the protocol requirements for running MemTuner in a reproducible, fair, and auditable way.

The protocol specifies what must be recorded, what must remain controlled, and which artifacts are required for benchmark claims.

## Protocol Goals

The protocol exists to guarantee:

- Reproducibility
- Fair comparison
- Artifact traceability
- Explicit configuration identity
- Clear separation between benchmark configuration and machine-specific runtime setup

## Canonical Execution Pipeline

A benchmark run follows this high-level pipeline:

1. Resolve dataset or dataset pack.
2. Validate dataset schema and metadata.
3. Resolve benchmark configuration.
4. Resolve runtime environment and provider settings.
5. Capture provenance metadata.
6. Execute benchmark workload.
7. Compute metric outputs.
8. Generate reporting artifacts.
9. Persist run artifacts for replay and review.

Each step must be represented by auditable artifacts or logs when possible.

## Required Provenance

A protocol-compliant benchmark run must capture at least the following:

- Dataset identifier
- Dataset fingerprint
- Dataset version or source reference when available
- Benchmark configuration
- Retrieval strategy and retrieval settings
- Model identifiers and revisions when applicable
- Memory-policy settings
- Seed or deterministic execution settings
- Repository revision or release version
- Python version
- Platform and hardware details
- Library and dependency versions relevant to the run
- Output path and artifact manifest

## Configuration Rules

### YAML Configuration

YAML defines the benchmark itself.

Examples of benchmark-defined settings:

- Enabled memory modules
- Retrieval strategy
- Per-strategy retrieval parameters
- Embedding model selection when configured there
- Decay and pruning policies
- Simulated days
- Seeds
- Observability behavior
- Answering or judging behavior

### Environment Configuration

Environment variables and `.env` define machine-specific runtime setup.

Examples of environment-defined settings:

- API keys
- Tokens
- Base URLs
- Endpoint URLs
- Provider credentials
- Default provider model names when intended as runtime defaults

The benchmark must not depend on hidden machine-local settings for any comparison-critical behavior that is absent from the stored run artifacts.

## Determinism Rules

A run should be deterministic when the benchmark mode and underlying components support deterministic replay.

At minimum:

- Seeds must be recorded when randomness is used.
- Dataset fingerprints must be deterministic.
- The effective benchmark configuration must be stored.
- Any non-deterministic dependency must be disclosed in the run metadata.

## Fair Comparison Rules

A fair comparison requires:

1. Shared datasets and evaluation tasks
2. Shared protocol version
3. Explicitly recorded configuration differences
4. Shared or justified runtime assumptions
5. No hidden post-processing differences between compared systems

If systems are compared under different budgets, different candidate counts, or different providers, those differences must be disclosed in the report and encoded in metadata.

## Required Artifact Set

Every benchmark run should emit a standard artifact set.

Recommended minimum artifact set:

- Benchmark report artifact
- Artifact manifest
- Optional tagged JSON artifacts emitted by the command
- Optional figure artifacts

## Current Implementation Surface

The current implementation does not yet emit a fully separated protocol artifact set such as `run.json`, `metadata.json`, `environment.json`, and `config.yaml` for every run.

The current concrete artifact surface is centered on `benchmark analyze` and currently includes:

- `benchmark_report.json`
- `artifact_manifest.json`
- `benchmark_analysis.png`
- additional tagged JSON and plot artifacts referenced by the manifest when generated

This means the protocol currently has two layers:

- current implemented artifacts that can be relied on today
- future protocol artifacts that should be added as Phase 1 work matures

Benchmark-facing documentation and reports should distinguish clearly between these two layers.

## Current Gaps Between Protocol Intent And Implementation

The following protocol goals are only partially realized in the current implementation surface:

- separated run metadata artifact
- separated environment artifact
- explicit protocol-version artifacting
- standardized effective-config artifact for every emitted report set
- fully formalized dataset fingerprint artifact

These are valid Phase 1 implementation targets and should not be described as already guaranteed outputs unless the command behavior is updated.

## Reportability Rules

A run may be used in benchmark reporting only if:

- The artifacts are complete enough to interpret the run
- The configuration is reconstructable
- The dataset identity is known
- The reported metrics match the stored outputs
- The run is not known to violate benchmark fairness assumptions

## Failure Handling Rules

A failed or partial run should still preserve enough information for diagnosis when possible.

At minimum, failure records should include:

- Benchmark phase or stage where execution stopped
- Relevant configuration identity
- Dataset identity
- Error summary
- Environment context
- Any partial outputs already produced

Partial or failed runs must not be silently upgraded into benchmark evidence.

## Protocol Versioning

The protocol should be versioned whenever a change affects:

- Required artifacts
- Required metadata
- Fair-comparison assumptions
- Metric semantics
- Benchmark execution stages

Reports and paper artifacts should identify the protocol version used.

## Benchmark Review Checklist

Before using a run in a report or paper, verify:

- Dataset fingerprint exists
- Effective config exists
- Relevant model and retrieval settings are recorded
- Seed or determinism mode is recorded
- Runtime environment is captured
- Metric outputs and report outputs match
- Any caveats are documented

## Relationship to the Roadmap

This protocol document satisfies a Phase 0 benchmark-documentation requirement and is the primary input to Phase 1 benchmark-protocol work.

It is intentionally concrete about the current CLI artifact surface so that documentation claims remain aligned with the codebase as it exists today.
