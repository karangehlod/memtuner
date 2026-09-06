## Summary
<!-- What does this PR do? One or two sentences. -->

## Type of change
- [ ] Bug fix
- [ ] New dataset adapter
- [ ] New retrieval strategy
- [ ] New metric or evaluation method
- [ ] Documentation / README
- [ ] CI / tooling

## Checklist
- [ ] `ruff check benchmark/ scripts/` passes with no errors
- [ ] `python -m pytest tests/unit/test_math_correctness.py -q` — all 50+ tests pass
- [ ] `python -m pytest tests/ -m "unit or contract" -q` — all tests pass
- [ ] If adding a dataset adapter: `query.day >= injection_day` holds for all generated queries (see CONTRIBUTING.md)
- [ ] If changing a metric formula: added a test in `tests/unit/test_math_correctness.py` that verifies the formula against a hand-computed example
- [ ] If adding a retrieval strategy: registered in `benchmark/factory/bootstrap.py` and classified in `benchmark/workload/study_scheduler.py::_STRATEGIES_SAFE_IN_THREADPOOL` (or excluded from it with a comment)
- [ ] README / CHANGELOG updated if behaviour visible to users changed
- [ ] No dataset files committed (datasets are downloaded on demand — see NOTICE)

## Testing
<!-- Describe how you tested this. Paste relevant output. -->
