# Phase 1 Stories

## Workstream: Protocol Specification

### AMB-P01-S01: Define canonical benchmark pipeline
Goal: specify the required stages and execution order for every benchmark run.
Dependencies: AMB-P00-S05
Acceptance Criteria:
- Pipeline stages documented
- Required inputs and outputs documented
- Stage responsibilities are unambiguous
Completion Evidence:
- `docs/roadmap/phases/phase-01-benchmark-protocol.md` documents the current enforced `benchmark analyze` stage order
- `benchmark/cli/commands/analyze_command.py` executes the documented stages in fixed sequence from dataset resolution through artifact emission
Status: Completed

### AMB-P01-S02: Define run metadata schema
Goal: capture all metadata required for provenance and reproducibility.
Dependencies: AMB-P01-S01
Acceptance Criteria:
- Required metadata fields listed
- Field semantics documented
- Schema covers models, data, system, and seeds
Completion Evidence:
- `benchmark/cli/provenance.py` defines run metadata payload construction
- `docs/ARTIFACT_SCHEMA.md` documents current emitted provenance artifacts
Status: Completed

## Workstream: Provenance and Reproducibility

### AMB-P01-S03: Define run hash and dataset fingerprint rules
Goal: ensure experiment identity can be tracked deterministically.
Dependencies: AMB-P01-S02
Acceptance Criteria:
- Run hash input set defined
- Dataset fingerprint method defined
- Determinism requirements documented
Completion Evidence:
- `benchmark/application/run_plan.py` defines dataset fingerprint generation
- `benchmark/cli/provenance.py` defines the stable `run_hash` input set used by `run_metadata.json`
- `tests/unit/test_provenance.py` verifies dataset fingerprint determinism across user ordering and run-hash stability/change behavior
Status: Completed

### AMB-P01-S04: Define environment capture contract
Goal: standardize environment snapshots for reproduction.
Dependencies: AMB-P01-S02
Acceptance Criteria:
- Hardware metadata fields defined
- Software version fields defined
- Runtime environment fields defined
Completion Evidence:
- `benchmark/cli/provenance.py` captures Python, platform, process environment, and git metadata
- `benchmark analyze` emits `environment.json`
Status: Completed

## Workstream: Output Standardization

### AMB-P01-S05: Standardize run artifact outputs
Goal: define the canonical output file set for benchmark runs.
Dependencies: AMB-P01-S01
Acceptance Criteria:
- Artifact filenames defined
- Artifact responsibilities documented
- Versioning scheme documented
Completion Evidence:
- `docs/ARTIFACT_SCHEMA.md` documents current artifact filenames and responsibilities
- `benchmark analyze` emits and references `effective_config.json`, `environment.json`, and `run_metadata.json`
- `tests/integration/test_cli.py` verifies both successful artifact emission and failed partial-report fallback for `benchmark analyze` using in-process `CliRunner` coverage, including dataset counts, run-plan fingerprint, and artifact-manifest/report consistency checks on the success path
- `.venv/bin/python -m benchmark.cli.main validate --config configs/locomo.yaml --check-environment --environment-output analysis_output/environment-validation-venv.json` verifies the benchmark CLI runtime imports under the active venv
- Live sample-run verification remains blocked by a Python 3.14 venv runtime failure after benchmark execution starts, including an observed `OMP: Warning #179: Function Can't set size of /tmp file failed:` symptom and an empty output directory even after redirecting temp files into the workspace and constraining math backend thread counts
Status: In Progress
