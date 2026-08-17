# Critical Issues Remediation Plan — Section 2

Scope
-----
This plan covers the 7 critical/medium issues identified in Section 2 of the consultant assessment (Lifecycle policies, Temporal evaluation, ReadQueryFilters, Conversation metadata plumbing, acceptable_modules verification, DRY in long-term stores, hardcoded source_module). For each issue the plan lists: ownerable tasks, files to change, unit/contract tests to add, observability requirements, acceptance criteria, and a rough effort estimate.

Guiding constraints
-------------------
- Follow SOLID/DIP: orchestrator must depend on interfaces/registry only.
- Determinism: seed + config → identical results.
- Observability: every public method emits OTel spans and structured logs with trace IDs.
- Tests: unit + contract + deterministic replay tests required for each fix.

Issue A: Lifecycle Policies Are Disconnected
--------------------------------------------
Goal: Ensure lifecycle policies are applied every day after injection and before queries, with pruning executed on flagged IDs and costs recorded.

Tasks:
- Add lifecycle policy wiring to `ScenarioRunner` constructor and `_run_day()` flow.
- Implement `_apply_lifecycle_policies(day)` helper that:
  - resolves per-module policy via Factory Registry
  - calls `module.get_memory_scores(day)`
  - invokes `policy.apply(day, scores)`
  - calls `module.prune(flagged_ids)` and records storage/operation cost
- Update `factory/` to expose policies registry resolution.
- Add tests: contract tests validating policies call order and deterministic pruning; integration tests for sample policies.

Files (likely):
- `benchmark/orchestrator/scenario_runner.py`
- `benchmark/factory/policy_registry.py`
- `benchmark/metrics/cost_engine.py` (cost recording)

Acceptance criteria:
- Unit tests assert `_apply_lifecycle_policies()` calls policy.apply with expected scores.
- End-to-end deterministic replay test demonstrates that memories are pruned per policy and `memory_survival_rates` changes accordingly.
- All public methods produce OTel spans for the policy application step.

Estimate: 2–4 days

Issue B: Temporal Evaluation Uses ID-only Fallback
-------------------------------------------------
Goal: Provide `EvaluationContext` plumbing to feed creation-day data to `TemporalAccuracyEvaluator` and ensure `ScenarioRunner` calls evaluator.evaluate_with_context(context).

Tasks:
- Confirm `EvaluationContext` model contains retrieved_creation_days and temporal_window fields.
- Update `ScenarioRunner` to build EvaluationContext per query and call `evaluate_with_context()` for all evaluators.
- Update evaluators to implement `evaluate_with_context()` with default delegation to `evaluate()`.
- Add deterministic integration tests comparing temporal evaluator results before/after fix.

Files (likely):
- `benchmark/evaluation/context.py`
- `benchmark/evaluation/temporal.py`
- `benchmark/orchestrator/scenario_runner.py`

Acceptance criteria:
- Temporal evaluator uses creation-day values from retrieved memories and gold temporal_window to compute temporal_accuracy.
- Deterministic test with seeded dataset shows different (correct) temporal score than the prior ID-only heuristic.
- OTel spans include temporal evaluation attributes.

Estimate: 1–3 days

Issue C: ReadQueryFilters Ignored
---------------------------------
Goal: Ensure ReadQuery.filters (memory_types, min_importance) are honored by all memory modules.

Tasks:
- Implement `_filter_candidates(filters, candidates)` in `BaseLongTermStore` and replicate pattern for short-term modules.
- Ensure filters are applied before scoring and before promotion/pruning decisions.
- Add contract tests validating deterministic filter behavior.

Files (likely):
- `benchmark/memory/long_term/base_store.py`
- `benchmark/memory/short_term/*.py`

Acceptance criteria:
- Unit tests asserting that queries with min_importance or restricted memory_types return only eligible memories.
- Contract tests across all modules for filter compliance.

Estimate: 1–2 days

Issue D: Conversation Metadata Decorative
-----------------------------------------
Goal: Pipe conversation metadata (is_followup, references_turn, conversation_turn) into EvaluationContext and make it available to evaluators.

Tasks:
- Extend `EvaluationContext` fields and update ScenarioRunner builder.
- Ensure gold dataset parser exposes those fields to orchestrator.
- Add a placeholder `FollowUpAccuracyEvaluator` contract test that consumes the fields.

Files (likely):
- `benchmark/evaluation/context.py`
- `benchmark/orchestrator/scenario_runner.py`
- `benchmark/gold/parser.py`

Acceptance criteria:
- EvaluationContext contains conversation metadata and unit tests verify content for sample queries.
- Spans/logs include conversation attributes when present.

Estimate: 1–2 days

Issue E: acceptable_modules Not Checked
---------------------------------------
Goal: Implement `ModuleAccuracyEvaluator` and ensure source_module is preserved on RetrievedMemory so evaluator can compute module-level accuracy.

Tasks:
- Implement evaluator that compares retrieved memory.source_module against gold.acceptable_modules and returns fraction correct.
- Ensure ScenarioRunner includes ModuleAccuracyEvaluator in evaluator list and EvaluationContext carries acceptable_modules.
- Add tests for edge cases (empty acceptable_modules, unknown modules, ties).

Files (likely):
- `benchmark/evaluation/module_accuracy.py`
- `benchmark/models/retrieved_memory.py`

Acceptance criteria:
- Tests assert module_accuracy is computed deterministically.
- Reports include module_accuracy field per query.

Estimate: 1–2 days

Issue F: Massive DRY Violation in Long-Term Stores
-------------------------------------------------
Goal: Consolidate duplicated logic into `BaseLongTermStore`, normalize tier/confidence logic, and ensure implementations override only scoring logic.

Tasks:
- Create `BaseLongTermStore` with shared methods: write, write_on_day, get_memory_scores, _compute_confidence, _compute_tier, prune, count, clear, _filter_candidates.
- Refactor `EpisodicStore`, `SemanticStore`, `EntityStore`, `PreferenceStore` to extend base and implement `_compute_relevance_score()` only.
- Update unit and parametrized contract tests to assert behavior parity and normalized thresholds.

Files (likely):
- `benchmark/memory/long_term/base_store.py`
- `benchmark/memory/long_term/*.py`

Acceptance criteria:
- All existing behavior preserved via contract tests; new tests assert normalized tier thresholds.
- Duplication reduced (measured) and code coverage unchanged or improved.

Estimate: 3–6 days

Issue G: Hardcoded source_module Strings
---------------------------------------
Goal: Ensure every memory module exposes `module_name` property used by RetrievedMemory.source_module and registry resolution uses that name.

Tasks:
- Add `module_name` constructor parameter to all memory modules (default to registry name or type(self).__name__.lower()).
- Replace hardcoded strings in read() implementations with `self._module_name`.
- Add tests asserting default and override behaviors.

Files (likely):
- `benchmark/memory/*/*.py`
- `benchmark/factory/registry.py`

Acceptance criteria:
- Tests assert source_module returned in read results matches module name passed to constructor or derived default.

Estimate: 1–2 days

Cross-cutting testing & CI
-------------------------
- For each issue, add unit, contract, and deterministic replay tests (use pytest markers: @pytest.mark.unit, @pytest.mark.contract, @pytest.mark.integration).
- Run full test suite, ensure zero warnings, and validate coverage target (≥95%).

Observability & Logging
-----------------------
- Every new public method added must emit OTel spans and include trace IDs in logs.
- Log recoverable errors and trace exceptions in spans.

Governance & Review
-------------------
- Open a PR per issue or grouped by related issues (e.g., F + G together).
- Require at least one code review and passing CI with enforced linting/tests/coverage.

Total estimated effort
----------------------
Approximately 10–20 developer-days depending on parallel work and test maturity.

If you want, I can open the initial skeleton edits for `ScenarioRunner` and `EvaluationContext` next.
