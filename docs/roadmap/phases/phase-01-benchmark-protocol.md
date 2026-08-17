# Phase 1: Benchmark Protocol

## Purpose

Define a strict, versioned experimental protocol so that every benchmark run is reproducible and comparable.

## Research Question

What protocol, metadata, and artifact model are necessary to ensure benchmark fairness, run reproducibility, and auditable experiment provenance?

## Target Output

Versioned benchmark protocol plus reproducibility artifacts for every run.

## Scope

- Define the canonical benchmark pipeline
- Define required run metadata
- Define run hashing and fingerprinting rules
- Standardize environment capture
- Standardize benchmark outputs

## Deliverables

- Protocol specification
- Experiment runner contract
- Experiment registry contract
- Run metadata schema
- Run hash rules
- Dataset fingerprint rules
- Environment capture rules
- Current artifact schema documentation
- Row-level comparison schema documentation for current `benchmark_report.json` outputs
- Standard run outputs:
  - benchmark_report.json
  - artifact_manifest.json
  - benchmark_analysis.png
  - effective_config.json
  - environment.json
  - run_metadata.json
  - tagged JSON and figure artifacts referenced by the manifest

## Workstreams

- Protocol specification
- Run metadata and hashing
- Environment and provenance capture
- Output schema standardization

## Dependencies

- Phase 0 architecture and documentation
- Existing run orchestration and reporting modules

## Acceptance Criteria

1. Every benchmark run can be traced to a config, dataset fingerprint, code revision, environment snapshot, and seed.
2. The canonical pipeline stages are documented and enforced by the protocol.
3. Output files are standardized and versioned.
4. Re-running the same config and seed on the same revision produces equivalent protocol artifacts.
5. The protocol document is sufficiently detailed for external reproduction.

## Current Canonical Pipeline

The current `benchmark analyze` protocol is enforced in the following stage order:

1. Dataset resolution and validation.
  The command resolves the requested dataset or pack-backed dataset, validates LLM-judge prerequisites, and records dataset counts.
2. Base configuration construction.
  The command builds benchmark configs from a single base template plus strategy-specific retrieval overrides.
3. Strategy comparison.
  The command executes the baseline retrieval strategy sweep for every available backend and captures the first successful effective config and run plan.
4. Provider-specific comparison sweeps.
  The command executes HF inference embedding, local embedding, Ollama embedding, and reranker comparison stages when their dependencies and provider prerequisites are satisfied.
5. Memory module comparison.
  The command benchmarks the supported memory modules against the loaded gold dataset.
6. Decay and pruning sweep.
  The command executes the lambda and threshold parameter sweep and records response curves.
7. Isolation verification.
  The command runs the multi-agent interference test and records the isolation result.
8. Visualization generation.
  The command emits overview and tagged plot artifacts when plotting dependencies are available.
9. Provenance and report emission.
  The command writes `effective_config.json`, `environment.json`, `run_metadata.json`, `benchmark_report.json`, and `artifact_manifest.json`.

Current pipeline boundary:

- provider prerequisites and optional dependencies explicitly gate comparison stages rather than triggering hidden fallback behavior
- report and provenance artifacts are emitted after benchmark execution stages complete
- a fully protocol-compliant sample run artifact set is still pending execution validation in the current environment

## Verification

1. Produce at least one protocol-compliant run artifact set.
2. Verify the current implemented artifact set is documented accurately and separated from future protocol targets.
3. Verify comparison-array row schemas are documented against the current command output.
4. Verify dataset fingerprint generation is deterministic.
5. Review the protocol for benchmark fairness and reproducibility.

Current evidence:

- `benchmark/cli/commands/analyze_command.py` enforces dataset resolution, comparison sweeps, memory and decay analysis, isolation testing, plot generation, and artifact emission in a fixed order.
- `benchmark analyze` emits `environment.json` and `run_metadata.json` and records them in `benchmark_report.json`.
- `benchmark/cli/provenance.py` computes a deterministic `run_hash` from stable protocol identity inputs inside `run_metadata.json`.
- `benchmark analyze` records `effective_config.json` when a composed strategy run yields an effective validated config snapshot.
- `docs/ARTIFACT_SCHEMA.md` documents the current implemented artifact surface and current row schemas for comparison arrays.
- `tests/unit/test_provenance.py` covers provenance helper serialization, deterministic dataset fingerprint generation, and run-hash invariants.
- `tests/integration/test_cli.py` now includes a focused `CliRunner` integration test that exercises the `benchmark analyze` artifact-emission path with stubbed runtime dependencies and verifies creation of `benchmark_report.json`, `artifact_manifest.json`, `effective_config.json`, `environment.json`, and `run_metadata.json`, along with dataset counts, run-plan dataset fingerprint, and manifest/report artifact consistency.
- `tests/integration/test_cli.py` also verifies that if runtime execution fails after dataset load, `benchmark analyze` writes a partial `benchmark_report.json` with failed status and captured runtime error instead of silently dropping protocol state.
- In the current workspace environment, `.venv/bin/python -m benchmark.cli.main validate --config configs/locomo.yaml --check-environment --environment-output analysis_output/environment-validation-venv.json` succeeds and confirms importability for `benchmark.cli.main`, `benchmark.cli.commands.analyze_command`, `benchmark.config.loader`, and `benchmark.gold.oracle`.
- `analysis_output/phase1-sample-run` contains a protocol-compliant artifact set (`benchmark_report.json`, `artifact_manifest.json`, `effective_config.json`, `environment.json`, `run_metadata.json`) produced under `.venv/bin/python` (Python 3.14.6).

## Current Operator Notes

The protocol intentionally treats provider prerequisites as explicit benchmark inputs.

- Local `embeddings` comparisons require the optional embeddings dependency and a locally loadable sentence-transformers model.
- Oversized embedding models must not silently fall back from local execution to API-backed execution.
- Ollama embedding comparisons require a reachable `BENCHMARK_OLLAMA_BASE_URL` and should be recorded as skipped when the execution environment blocks localhost access.
- Router-backed HF reranker comparisons require `BENCHMARK_HF_RERANKER_URL` and should be recorded as skipped when it is absent.

These behaviors are part of the reproducibility contract: missing dependencies and unavailable providers are surfaced as explicit failures or skips, not hidden fallback paths.

Environment validation path:

- `benchmark validate --check-environment --environment-output <path>` now provides a protocol-friendly preflight for Python runtime and module importability.
- Operators should run this preflight under the same interpreter that will execute `benchmark analyze` whenever the sample protocol run is being verified or repaired.
- A failed preflight should be treated as an execution-environment blocker, not as evidence of a benchmark artifact emission regression.
- After a passing preflight, operators should immediately run a fresh sample `benchmark analyze` under that same interpreter; only failures that persist after a passing preflight should be treated as likely analyze-command defects.

## Current Blocker

The protocol sample run is now verified under `.venv` (Python 3.14.6), but the
Python 3.13 environment still exits during strategy comparison with only
partial artifacts captured.

- `.venv313/bin/python` completes dataset loading and enters `strategy_start:bm25`,
  then exits after emitting `OMP: Warning #179: Function Can't set size of /tmp file failed:`.
- The output directory contains `environment.json` and `run_metadata.json`, but
  not `benchmark_report.json` or `artifact_manifest.json`, even with
  `BENCHMARK_ANALYZE_SKIP_MEMORY_TYPE_COMPARISON=1`.
- This does not block protocol verification, but it does block a Python 3.13
  sample run in the current environment.

## Out of Scope

- Expanding baseline coverage
- Statistical testing
- Dashboard UX

## Definition of Done

Phase 1 is complete when benchmark runs are protocolized, versioned, and reproducible in an auditable way.

## Completion Checklist

- [x] Pipeline defined
- [x] Run metadata schema defined
- [x] Run hash defined
- [x] Dataset fingerprint defined
- [x] Environment capture defined
- [x] Standard outputs defined
- [x] Protocol verified with sample run
- [ ] External reproduction review passed
- [ ] Phase accepted and status updated in master roadmap
