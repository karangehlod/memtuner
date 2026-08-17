# Contributing to Agentic Memory Benchmark

Thank you for your interest in contributing. This guide covers everything you need to get started.

---

## Development environment

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/Agenticmemory_benchmark.git
cd Agenticmemory_benchmark

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

A dataset adapter translates a raw corpus/QA file into the common `Scenario` format.

1. Create `benchmark/adapters/<your_dataset>_adapter.py` implementing `BaseAdapter`.
2. Implement `load(path: Path) -> list[Scenario]`; every `Scenario` must have a non-empty `gold_ids` list.
3. Register the adapter in `benchmark/adapters/__init__.py` (add to the `ADAPTER_REGISTRY` dict).
4. Add a unit test in `tests/unit/test_adapter_interface.py` using the existing parametrize fixture.
5. Add a note to `README.md` (or `docs/datasets.md`) describing the corpus and its license.

---

## Adding a new retrieval strategy

A retrieval strategy ranks candidate memory chunks given a query.

1. Create `benchmark/strategies/<name>_strategy.py` implementing `BaseRetrievalStrategy`.
2. Implement `retrieve(query: str, memories: list[Memory], top_k: int) -> list[Memory]`.
3. Register the strategy in `benchmark/strategies/__init__.py` (`STRATEGY_REGISTRY`).
4. Add unit tests in `tests/unit/test_retrieval_strategies.py` covering: empty corpus, exact match, `top_k > len(corpus)`.

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

Use [GitHub Discussions](https://github.com/karangehlod/Agenticmemory_benchmark/discussions) for:

- Design questions before opening a large PR
- Help understanding the codebase
- Proposing new dataset integrations or strategies

For confirmed bugs, open a [GitHub Issue](https://github.com/karangehlod/Agenticmemory_benchmark/issues) with a minimal reproduction case.
