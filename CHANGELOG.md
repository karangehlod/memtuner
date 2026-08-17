# Changelog

All notable changes to the Agentic Memory Benchmark are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — development branch (will become 0.1.0 on first PyPI release)

### Added
- **`--doctor` command** — reads CPU, RAM, GPU and installed packages; prints
  hardware capability matrix and copy-paste ready run command tailored to the
  detected hardware. Runs as `python study_runner.py --doctor` or `benchmark doctor`.
- **Recency baseline strategy** (`RecencyStrategy`) — returns K most recently
  injected memories, ignoring query content. Runs automatically in Phase 1
  alongside BM25. Anchors all comparisons with a query-agnostic lower bound.
- **Bootstrap confidence intervals** (`bootstrap_ci`, `significance_table`) —
  95% non-parametric percentile bootstrap per strategy group. Activated via
  `--seeds 42 123 456`. Non-overlapping CIs flagged with ★ in report.
- **Multi-seed runs** (`--seeds`) — run each cell once per seed and pool results
  for statistical significance reporting.
- **Early stopping for Phase 5** (`--early-stop-patience`) — stops testing λ
  values per decay policy once composite score plateaus for N consecutive steps.
  Default patience=3; saves 30–50% of Phase 5 cells on typical datasets.
- **Per-dataset recommendations** — `rank_by_dataset()` in aggregator; separate
  best-config table per dataset in report (LoCoMo vs LongMemEval differ).
- **Model filtering flags** — `--skip-models`, `--only-models`, `--skip-rerankers`
  for excluding large/slow models without editing source.
- **Apple MPS support** — `hw_probe.py` detects Apple Silicon MPS backend;
  all embedding and reranker models now use `DEVICE` from hw_probe instead of
  hardcoded `"cuda" if ... else "cpu"`.
- **Math correctness test suite** — 41 unit tests covering all formulas:
  decay functions (all policies + archival floor), composite score (weights + gate),
  recency ordering, bootstrap CI properties, and RRF k=60 fusion.
- **Simulation figures** — `benchmark.reporting.simulation_plots` generates
  publication-quality mathematical figures: decay curves, composite sensitivity,
  RRF fusion analysis, λ×time phase-space contour plot.
- **Incremental embedding index** — each memory encoded exactly once per cell
  (not once per day); reduces total encoding work from O(N²) to O(N).
- **Query prewarm** — all day's queries batch-encoded in one GPU call before
  the per-query loop; eliminates N single-query kernel launches per day.
- **Full-corpus reranking** — CrossEncoder scores all memories, not a BM25 top-N
  subset. Gold memories can no longer be dropped before reranking.
- **Direct CrossEncoder on GPU** — reranker models (`ms-marco-MiniLM-L6-v2`,
  `bge-reranker-base`) now use `CrossEncoder.predict()` directly; no Ollama or
  HTTP overhead.
- **LLM judge integration** — `--judge-model` + `--ollama-url` enables
  `EndToEndEvaluator` (Nemotron-Mini-4B-Instruct / Gemma-4B via Ollama).
  Works with any OpenAI-compatible endpoint including OpenAI API and local models.
- **CoQA adapter fix** — questions were passed as dicts instead of strings;
  now correctly extracts `input_text` field.
- **LongMemEval adapter fix** — `_date_to_day()` now uses days-since-epoch
  (no more day-of-month collisions across months); `haystack_session_ids` mapped
  to generated memory IDs correctly.
- **Synthetic adapter fix** — query day now guaranteed ≥ max injection day of
  gold memories (temporal validity enforced).
- **`Dockerfile`** — multi-stage CPU/GPU build; `--gpus all` for NVIDIA.
- **GitHub templates** — issue templates (bug report, feature request),
  PR template with math-test checklist.
- **`CONTRIBUTING.md`** — dev setup, test commands, adapter/strategy guides,
  PR checklist.
- **`CHANGELOG.md`** (this file).

### Changed
- `EMBEDDING_MODELS_OLLAMA` is now `[]` — bge-m3 and qwen3-embedding moved to
  `EMBEDDING_MODELS_LOCAL` (direct PyTorch, no HTTP overhead).
- `EMBEDDING_MODELS_LOCAL` now includes `Qwen/Qwen3-Embedding-4B` (7.6 GB,
  fits alone on 16 GB GPU).
- Startup banner shows GPU backend (`cuda`/`mps`/`cpu`), dependency status,
  and auto-skips Phases 2–4 if `sentence-transformers` is not installed.
- pyproject.toml version 0.0.1 → 0.2.0; added keywords, project URLs,
  `bm25`/`judge`/`stats` optional dependency extras.
- `strategy_ranking` in aggregator now separates fallback cells (OOM, 404)
  into `embeddings_fallback` label so they don't dilute real strategy averages.
- 404 "model not found" errors in `base_store.py` are now hard failures (cell
  marked failed) instead of silent strategy-disable producing fallback recalls.

### Fixed
- `_INDEX_CACHE` in `EmbeddingsStrategy` stored a list alias instead of a copy;
  incremental appends corrupted the cached list, causing `IndexError` in later
  cells hitting the same corpus hash.
- All `open()` calls in `aggregator.py` and `study_aggregator.py` now include
  `encoding="utf-8"` (Windows ANSI locale corruption fix).
- `.env` loader now uses `encoding="utf-8"`.
- `EmbeddingsStrategy.is_available()` no longer requires `BENCHMARK_EMBEDDING_MODEL`
  env var — it now checks only whether `sentence-transformers` is importable.
- OOM during embedding encode now halves `batch_size` and retries instead of
  crashing the cell.
- Phase 5 seed sanitization: when recommended strategy requires Ollama but
  recommended model is local-only, strategy corrected to `"embeddings"`.
- Squad gold file preflight check: 0-byte files detected and skipped before
  scheduling 96 cells that would all fail.

---

## [0.0.1] — 2026-08-09 (initial development release)

### Added
- Initial five-phase adaptive study runner (`study_runner.py`)
- BM25 retrieval strategy with corpus cache
- EmbeddingsStrategy with model singleton LRU cache
- HybridStrategy with Reciprocal Rank Fusion (RRF, k=60)
- LLMRerankStrategy (CrossEncoder n-gram baseline)
- Four decay policies: exponential, linear, logarithmic, tiered
- Four memory stores: Episodic, Semantic, Preference, Entity
- Dataset adapters: LoCoMo, LongMemEval, SQuAD 2.0, CoQA, Synthetic
- Metrics: Recall@K, Precision@K, MRR, NDCG, F1, Temporal Accuracy, Composite
- 6-panel PNG study report (publication-quality, 300 DPI)
- `benchmark` CLI entry point
- GitHub Actions CI: lint + test + build

---

[Unreleased]: https://github.com/karangehlod/agentic-memory-benchmark/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/karangehlod/agentic-memory-benchmark/releases/tag/v0.0.1
