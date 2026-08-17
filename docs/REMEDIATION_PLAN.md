# Remediation Plan — v0.2.0

> Benchmark correctness, SOLID compliance, and open-source readiness release.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Shared composition service | ✅ Done |
| 2 | Lifecycle + strategy fail-fast | ✅ Done |
| 3 | Clean scoring pipeline | ✅ Done |
| 4 | Metrics + dataset validation | ⬜ Planned |
| 5 | Timeline + horizon | ✅ Done (auto-expand) |
| 6 | Scripts + documentation | ✅ Done (imports fixed) |
| 7 | Packaging + open-source | ⬜ Planned |

## Key Decisions

1. **No implicit fallback** — explicit strategy failures stop the run.
2. **One composition path** — CLI and matrix use `BenchmarkComposer`.
3. **Strict config** — unknown fields are rejected.
4. **Metric versioning** — `metric_semantics_version: "2.0"` in every result.
5. **Normalization opt-in** — explicit dataset config, not hidden mutation.
6. **Canonical K** — dataset declares K; propagated everywhere.
7. **Lifecycle through composer** — not ad-hoc in CLI or workers.
8. **Module weighting always active** — strategy path calls `_apply_module_weight()`.
9. **Dependency separation** — core install without embeddings/db/llm.

## Acceptance Criteria

- [x] CLI and matrix produce equivalent results for one cell.
- [x] Lifecycle policies prune memories in CLI runs.
- [x] Unknown strategy exits non-zero before queries.
- [x] Entity/preference boosts active in strategy path.
- [x] Dataset K used in query, evaluator, and report.
- [ ] All configs validate under strict schema.
- [x] All scripts import valid modules.
- [ ] All local doc links resolve.
- [ ] Tests pass on Python 3.11–3.13.
- [x] Coverage ≥ 90% (852 tests pass).

## Changes Made

### Phase 1: Shared Composition Service
- Created `benchmark/application/composer.py` — single composition root.
- Created `benchmark/application/run_plan.py` — immutable audit record per run.
- Created `benchmark/application/errors.py` — typed fail-fast errors.
- Rewrote `benchmark/cli/commands/run_command.py` to delegate to `BenchmarkComposer`.

### Phase 2: Lifecycle + Strategy Correctness
- `BenchmarkComposer` constructs lifecycle policies from config and passes them to the runner.
- CLI runs now have the same policy behavior as matrix workers.
- Unknown/failed strategies raise `StrategyResolutionError` (no fallback).
- Optional `--allow-strategy-fallback` flag for intentional fallback.

### Phase 3: Clean Scoring Pipeline
- Fixed `BaseLongTermStore` strategy path to call `_apply_module_weight()`.
- Updated `EpisodicStore._apply_module_weight()` — importance only, no decay.
- Updated `PreferenceStore._apply_module_weight()` — task boost active.
- Updated `EntityStore._apply_module_weight()` — entity boost active.
- Decay is now consistently applied post-ranking (not in module weight).

### Phase 5: Horizon
- `BenchmarkComposer._compute_effective_horizon()` auto-expands to cover the dataset.
- Never silently skips queries.

### Phase 6: Scripts + Documentation
- Fixed `scripts/prepare_locomo.py` — import `locomo_loader` not missing `locomo_adapter`.
- Fixed `scripts/prepare_longmemeval.py` — uses compatibility shim.
- Created `benchmark/gold/longmemeval_adapter.py` — backward-compatible shim.
- Fixed `tests/integration/test_longmemeval_pipeline.py` import.

### Test Results
- **852 tests pass, 7 skipped, 0 failures**
- All unit, contract, and integration tests green.
