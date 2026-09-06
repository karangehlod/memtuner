# Contributing to MemTuner

Thank you for your interest in contributing. This guide covers everything you need to get started.

---

## Development environment

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/memtuner.git
cd memtuner

# 2. Create a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with all dev extras
pip install -e ".[dev]"
```

---

## Running the test suite

```bash
# Unit tests (fast, no I/O)
python -m pytest tests/unit/ -v

# Math correctness tests (sub-second, pure arithmetic)
python -m pytest tests/unit/test_math_correctness.py -v

# Integration tests (may require network / embeddings)
python -m pytest tests/integration/ -v

# Full suite with coverage
python -m pytest tests/unit/ tests/contract/ --cov=benchmark --cov-report=term-missing
```

---

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting/formatting and [mypy](https://mypy.readthedocs.io/) for type checking.

```bash
# Lint
ruff check benchmark/ tests/

# Format
ruff format benchmark/ tests/

# Type check
mypy benchmark/
```

CI enforces both. Fix all lint errors before opening a PR.

---

## Adding a new dataset adapter

A dataset adapter converts a raw corpus/QA file into the common `GoldDataset` format used by the study runner.

1. Create `benchmark/gold/adapters/<your_dataset>_adapter.py` subclassing `DatasetAdapter`
   (see `benchmark/gold/adapters/adapter.py` for the interface).
2. Implement `load(source: Path) -> GoldDataset`. Every query must have at least one expected memory ID in `query.expected.memory_ids`, and **`query.day` must be ≥ the injection day of every referenced memory** — otherwise the memory hasn't been injected yet when the query fires.
3. The adapter is auto-discovered if you pass its output path to `memtuner study --gold-dataset`. No registration is required for that flow.
4. To add it to `scripts/prepare_extended_datasets.py` (for auto-download), add a `prepare_<name>()` function and entry in `PREPARERS`.
5. Add a unit test in `tests/unit/test_adapter_interface.py` using the existing parametrize fixture.
6. Document the dataset and its license in `README.md` (Datasets table) and `NOTICE`.

---

## Adding a new retrieval strategy

A retrieval strategy ranks memories given a query.

1. Create `benchmark/memory/strategies/<name>_strategy.py` implementing `RetrievalStrategy`
   (interface: `benchmark/memory/interfaces/retrieval_strategy.py`).
2. Implement `index(memories)`, `retrieve(query, top_k, user_id)`, `clear()`, and `name()`.
3. Register in `benchmark/factory/bootstrap.py` by adding a tuple to the `strategies` list.
4. Declare thread-safety in `benchmark/workload/study_scheduler.py::_STRATEGIES_SAFE_IN_THREADPOOL`:
   - Add the name to the set if the strategy is pure Python/numpy (no GPU, no torch).
   - Leave it out if it uses sentence-transformers or any GPU resource (it will run sequentially).
5. Add unit tests in `tests/unit/test_retrieval_strategies.py` covering: empty corpus, exact match, `top_k > len(corpus)`, and user isolation.

---

## Pull request checklist

Before opening a PR, verify:

- [ ] `ruff check` passes with zero errors
- [ ] `ruff format --check` passes (no reformatting needed)
- [ ] `mypy benchmark/` passes in strict mode
- [ ] All unit tests pass: `pytest tests/unit/ -q`
- [ ] Math correctness tests pass: `pytest tests/unit/test_math_correctness.py -v`
- [ ] No new `open()` call without an explicit `encoding=` argument
- [ ] New public functions/classes have docstrings
- [ ] If you changed scoring math, update `tests/unit/test_math_correctness.py`

---

## Where to ask questions

Use [GitHub Discussions](https://github.com/karangehlod/memtuner/discussions) for:

- Design questions before opening a large PR
- Help understanding the codebase
- Proposing new dataset integrations or strategies

For confirmed bugs, open a [GitHub Issue](https://github.com/karangehlod/memtuner/issues) with a minimal reproduction case.
