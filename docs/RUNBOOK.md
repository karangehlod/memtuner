# MemTuner — Runbook

Step-by-step instructions to install, configure, run, and interpret results
on **macOS**, **Linux**, and **Windows**. Every command is given for both platforms.

---

## TL;DR — Run the Full Benchmark (LoCoMo dataset)

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Verify
python3 -m pytest tests/unit/ -q --tb=no --ignore=tests/unit/test_longmemeval_adapter.py
# Expected: 690 passed

# 3. Quick sanity check (27 cells, ~7 min)
python3 grid_search.py --dataset data/locomo10.json --mode quick --workers 4

# 4. Full grid (400 cells, ~2-4 hours on 8-core workstation)
python3 grid_search.py --dataset data/locomo10.json --mode all --workers 8

# 5. Results
cat data/output/grid_*/grid_*_report.txt
```

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python3 --version` (Mac/Linux) / `python --version` (Win) |
| pip | latest | `pip install --upgrade pip` |
| Git | any | `git --version` |
| Disk space | ≥ 2 GB | For datasets and outputs |
| RAM | ≥ 8 GB | 16 GB recommended for full grid with 8 workers |
| CPU cores | ≥ 4 | 8+ recommended for `--workers 8` |

> **Windows users:** use **PowerShell** (not cmd.exe). All commands below work
> in PowerShell 7+ and the built-in Windows PowerShell 5.1.

---

## Step 1 — Clone and Install

### macOS / Linux

```bash
git clone https://github.com/karangehlod/memtuner.git
cd memtuner

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the benchmark package + all dependencies
pip install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
git clone https://github.com/karangehlod/memtuner.git
cd memtuner

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the benchmark package + all dependencies
pip install -e ".[dev]"
```

### Verify installation

```bash
# macOS / Linux
python3 -c "from benchmark.workload.matrix import MatrixExpander; print('OK')"

# Windows
python -c "from benchmark.workload.matrix import MatrixExpander; print('OK')"
```

Expected output: `OK`

---

## Step 2 — Run the Test Suite

Confirm everything is wired correctly before generating data.

### macOS / Linux

```bash
python3 -m pytest tests/ -q --tb=short
```

### Windows

```powershell
python -m pytest tests/ -q --tb=short
```

Expected output:

```
690 passed, 1 skipped in ~130s
```

If tests fail, check that Python ≥ 3.11 is active in the virtual environment.

### Validate The Runtime Before `benchmark analyze`

When Phase 1 protocol verification is blocked or the CLI appears to exit without emitting expected artifacts, validate the active Python runtime before retrying `benchmark analyze`.

Run:

```bash
python3 -m benchmark.cli.main validate \
  --config configs/locomo.yaml \
  --check-environment \
  --environment-output analysis_output/environment-validation.json
```

This performs two things:

- validates the benchmark config as usual
- emits a JSON environment report covering the active Python executable, version, platform, and required benchmark module imports

Expected outcome:

- if the runtime is healthy, the command reports successful module imports and writes `analysis_output/environment-validation.json`
- if the runtime is unhealthy, the command fails explicitly and identifies which benchmark module imports failed under the active interpreter

Operator rule:

- do not treat a missing `benchmark analyze` artifact set as an analyze-command regression until `benchmark validate --check-environment` passes under the same interpreter you intend to use for the sample run

Minimal recovery sequence:

1. Run `benchmark validate --check-environment` with the exact interpreter you will use for `benchmark analyze`.
2. If environment validation fails, repair that interpreter or virtual environment first and rerun the same validation command until it passes.
3. Once environment validation passes, run a sample `benchmark analyze` with the same interpreter and a fresh output directory.
4. Treat any remaining artifact-emission failure after a passing preflight as an `analyze` execution defect worth debugging in the command path itself.

Example sample-run sequence:

```bash
python3 -m benchmark.cli.main validate \
  --config configs/locomo.yaml \
  --check-environment \
  --environment-output analysis_output/environment-validation.json

python3 -m benchmark.cli.main analyze \
  --dataset data/locomo10.json \
  --output analysis_output/phase1-sample-run \
  --max-queries 1 \
  --seed 42
```

---

## Provider Prerequisites For Analyze

`benchmark analyze` deliberately fails or skips provider-backed comparisons when the
required dependency or endpoint is not available. No implicit fallback is allowed,
because silent fallback would make benchmark comparisons non-reproducible.

### Local sentence-transformers embeddings

Strategies using `embeddings` require the optional embeddings dependency.

Install it with:

```bash
pip install -e ".[embeddings]"
```

Typical failure if the dependency is missing:

```text
Cannot resolve retrieval strategy 'embeddings': Failed to load embedding model all-MiniLM-L6-v2.
Install: pip install sentence-transformers
```

Notes:

- `all-MiniLM-L6-v2` and `sentence-transformers/all-MiniLM-L6-v2` are local models.
- If the model cannot be downloaded in the current environment, the run will fail rather than silently swapping providers.

### Large local embedding models

The adaptive embeddings path prevents oversized local models from running through
the local `sentence-transformers` strategy when they exceed the configured local
size threshold.

Example:

```text
Embedding model BAAI/bge-base-en-v1.5 exceeds local size threshold (100 MB) and should use API providers: ollama, hf_inference
```

Use one of these fixes:

- switch the comparison to an API-backed strategy such as `ollama_embeddings` or `hf_inference_embeddings`
- raise the configured local size threshold only if local execution is actually intended and reproducible
- remove the model from the local embeddings comparison set

Default `benchmark analyze` behavior:

- the local embedding sweep filters `BENCHMARK_EMBEDDING_MODEL` and `BENCHMARK_EMBEDDING_MODELS` through the configured local size threshold
- larger models such as `BAAI/bge-base-en-v1.5` are excluded from the local sweep by default and should be benchmarked through API-backed comparisons instead

### Ollama embedding comparisons

Ollama-backed embedding comparisons require a reachable OpenAI-compatible embeddings endpoint.

Set:

```bash
export BENCHMARK_OLLAMA_BASE_URL=http://localhost:11434/v1
export BENCHMARK_OLLAMA_TIMEOUT=30
export BENCHMARK_OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

Optional for model sweeps:

```bash
export BENCHMARK_OLLAMA_EMBEDDING_MODELS=embeddinggemma,qwen3-embedding:0.6b,bge-m3:latest
```

Expected skip behavior:

- if Ollama is not reachable, the comparison is skipped with an endpoint-unavailable reason
- if the execution environment blocks localhost access, the comparison is skipped as environment-blocked rather than treated as a model failure

### HF reranker benchmarking

Router-backed HF reranker comparison requires an explicit reranker endpoint.

Set:

```bash
export BENCHMARK_HF_RERANKER_URL=https://your-endpoint.example/v1/rerank
```

Without it, `benchmark analyze` will skip the reranker comparison with:

```text
router-backed HF reranker benchmarking requires BENCHMARK_HF_RERANKER_URL
```

This is expected protocol behavior, not an implementation fallback.

---

## Step 3 — Generate a Gold Dataset

The benchmark needs a gold dataset — a file containing simulated memory events
and ground-truth queries. You can generate a small one (for quick testing) or
a production-scale one.

### Option A — Lite dataset (quick, ~130 MB, 1 000 users × 30 days, 300K queries)

```bash
# macOS / Linux
python3 generate_production_dataset.py \
    --lite \
    --output-dir data/generated/lite

# Windows
python generate_production_dataset.py `
    --lite `
    --output-dir data/generated/lite
```

Output files:
- `data/generated/lite/production_gold_dataset.json` (~130 MB)
- `data/generated/lite/dataset_metadata.json`

### Option B — Production dataset (5 000 users × 90 days, ~350 MB)

```bash
# macOS / Linux
python3 generate_production_dataset.py \
    --users 5000 \
    --days 90 \
    --queries-per-day 100 \
    --events-per-day 72 \
    --output-dir data/generated/production

# Windows
python generate_production_dataset.py `
    --users 5000 `
    --days 90 `
    --queries-per-day 100 `
    --events-per-day 72 `
    --output-dir data/generated/production
```

### Dataset metadata

After generation, inspect the metadata file to confirm scale:

```bash
# macOS / Linux
cat data/generated/lite/dataset_metadata.json

# Windows
Get-Content data/generated/lite/dataset_metadata.json
```

---

## Step 4 — Run the Quick Grid (Recommended Starting Point)

The quick grid runs **27 cells** (3 memory types × 3 strategies × 3 decay
policies × 1 lambda). Takes 5–15 minutes. ALL cells use the SAME dataset
so rankings are directly comparable.

### Using LoCoMo dataset (real conversations, included in repo)

The LoCoMo dataset (`data/locomo10.json`, 2.7 MB) contains 10 real conversations
with 1977 queries and 5879 memories — the best default for realistic benchmarking.

```bash
# macOS / Linux
python3 grid_search.py \
    --dataset data/locomo10.json \
    --mode quick \
    --workers 4 \
    --output-dir data/output

# Windows
python grid_search.py `
    --dataset data/locomo10.json `
    --mode quick `
    --workers 4 `
    --output-dir data/output
```

### Using generated dataset (synthetic, larger scale)

```bash
# macOS / Linux
python3 grid_search.py \
    --dataset data/generated/lite/production_gold_dataset.json \
    --mode quick \
    --output-dir data/output

# Windows
python grid_search.py `
    --dataset data/generated/lite/production_gold_dataset.json `
    --mode quick `
    --output-dir data/output
```

### What you will see

```
MemTuner — Full Grid Search
================================================================
  Run ID:          3fb35528afba
  Dataset:         data/locomo10.json
  Dataset size:    2.7 MB
  Mode:            quick
  Workload:        Medium (50 days)
  Simulated days:  5
  Lambda range:    0.05 → 0.3 (step 0.05)
  Seed:            42
  Output dir:      data/output/grid_20260530_125353_3fb35528afba

  Grid dimensions:
    Memory types:         ['episodic', 'preference', 'semantic']
    Retrieval strategies: ['bm25', 'embeddings', 'hybrid']
    Decay policies:       ['exponential', 'logarithmic', 'none']
    Lambda values:        [0.0, 0.05]
    Total cells:          27

  Running 27 cells with 4 parallel workers (platform: darwin)
  [  1/27] OK  episodic     × bm25       × exponential   λ=0.05  composite=+0.1680
  [  2/27] OK  episodic     × bm25       × logarithmic   λ=0.05  composite=+0.2013
  ...
  [ 27/27] OK  semantic     × hybrid     × none          λ=0.00  composite=+0.2734

  Completed 27/27 in 423.1s (7.0 min)
  Successful: 27 | Failed: 0

BEST PRODUCTION CONFIG:
  Memory type:        episodic
  Retrieval strategy: hybrid
  Decay policy:       none (λ=0.00)
  Recall@K:           0.1562
  Composite Score:    0.2734
  ...
```

### Monitor progress in real-time

```bash
# Follow the JSONL progress log (one JSON per completed cell)
tail -f data/output/grid_*/progress_*.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    print(f\"[{d['completed']}/{d['total']}] {d['memory_type']:12s} × {d['retrieval_strategy']:10s} | recall={d['recall_at_k']:.3f} p50={d.get('latency_p50_ms',0):.1f}ms p99={d.get('latency_p99_ms',0):.1f}ms\")
"
```

---

## Step 5 — Full Grid (All Memory Types × All Strategies × Lambda Sweep)

Runs 4 memory types × 3 strategies × 3 decay policies × 6 lambda steps = **216 cells**.
Use `--mode full` for a comprehensive sweep with commonly-used strategies.

### macOS / Linux

```bash
python3 grid_search.py \
    --dataset data/locomo10.json \
    --mode full \
    --lambda-min 0.05 --lambda-max 0.30 --lambda-step 0.05 \
    --workers 4 \
    --output-dir data/output
```

### Windows

```powershell
python grid_search.py `
    --dataset data/locomo10.json `
    --mode full `
    --lambda-min 0.05 --lambda-max 0.30 --lambda-step 0.05 `
    --workers 4 `
    --output-dir data/output
```

**Lambda steps generated:** [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

| λ | Half-life (days) | Decay strength |
|---|-----------------|----------------|
| 0.05 | 14 days | Gentle (default) |
| 0.10 | 7 days | Moderate |
| 0.15 | 5 days | Medium-fast |
| 0.20 | 3 days | Fast |
| 0.25 | 3 days | Very fast |
| 0.30 | 2 days | Aggressive |

---

## Step 6 — Complete Grid (400 cells, all strategies including pgvector/llm_rerank)

Runs the full Cartesian product: 4 memory types × 5 strategies × 5 decay policies
× 6 lambda steps = **400 cells**. Best run on a workstation with ≥8 cores.

### macOS / Linux

```bash
python3 grid_search.py \
    --dataset data/locomo10.json \
    --mode all \
    --workers 8 \
    --output-dir data/output
```

### Windows

```powershell
python grid_search.py `
    --dataset data/locomo10.json `
    --mode all `
    --workers 8 `
    --output-dir data/output
```

> **Estimated time:** 2–4 hours on an 8-core workstation.
> Each cell processes 1977 queries × 5879 memories.

### Workstation tips

- Use `--workers $(nproc)` on Linux to use all cores
- Monitor with: `tail -f data/output/grid_*/progress_*.jsonl | python3 -m json.tool`
- Each worker loads the sentence-transformers model independently (~400 MB RAM each)
- Total RAM needed: ~400 MB × workers + 1 GB overhead
- For 8 workers: plan for ~4.5 GB RAM

---

## Step 7 — Read the Output

Every grid run produces three files under `data/output/grid_<ts>_<run_id>/`:

```
data/output/grid_20260530_125353_3fb35528afba/
├── matrix_<ts>_<run_id>_summary.json    ← full JSON with rankings
├── matrix_<ts>_<run_id>_grid.csv        ← one row per cell (Excel-friendly)
├── matrix_<ts>_<run_id>_report.txt      ← human-readable text report
└── progress_<ts>_<run_id>.jsonl         ← real-time progress (1 line per cell)
```

### Read the text report

```bash
# macOS / Linux
cat data/output/grid_*/matrix_*_report.txt

# Windows
Get-Content data/output/grid_*\matrix_*_report.txt
```

### Open the CSV in Excel / Numbers

The CSV has one row per benchmark cell. Key columns:

| Column | Meaning |
|--------|---------|
| `memory_type` | episodic / semantic / preference / entity |
| `retrieval_strategy` | bm25 / embeddings / hybrid / pgvector / llm_rerank |
| `decay_policy` | none / exponential / logarithmic / linear / periodic |
| `lambda` | Decay rate (0.0 = no decay, 0.20 = aggressive) |
| `recall_at_k` | Fraction of expected memories retrieved (higher = better) |
| `false_positive_rate` | Fraction of retrieved memories that are wrong (lower = better) |
| `mrr` | Mean Reciprocal Rank (higher = better) |
| `ndcg` | Normalized Discounted Cumulative Gain (higher = better) |
| `precision_at_1` | Correct result in first position (higher = better) |
| `temporal_accuracy` | Fraction of retrievals within the expected time window |
| `composite_score` | Weighted: `0.30×recall + 0.15×(1-fpr) + 0.15×mrr + 0.15×ndcg + 0.15×temporal + 0.10×module` |
| `latency_p50_ms` | Median query latency in milliseconds |
| `latency_p90_ms` | 90th percentile query latency |
| `latency_p99_ms` | 99th percentile query latency (tail latency) |
| `latency_mean_ms` | Average query latency |
| `avg_cpu_percent` | Average CPU utilization during run |
| `peak_ram_mb` | Peak RAM used during this cell's run |
| `disk_write_mb` | Total disk writes during run |
| `duration_seconds` | Wall-clock time for this cell |

---

## Step 8 — Run with Your Own Data (Private Pack)

To benchmark against your own conversation data, prepare two JSONL files:

**`events.jsonl`** — one JSON object per line:
```json
{"id": "M-001", "user_id": "user-42", "type": "episodic", "content": "...", "timestamp": "2024-01-01T10:00:00Z", "importance": 0.8, "task_id": "task-1"}
```

**`queries.jsonl`** — one JSON object per line:
```json
{"query": "What did I say about...", "user_id": "user-42", "task_id": "task-1", "day": 5, "expected_memory_ids": ["M-001"]}
```

Then run:

```bash
# macOS / Linux
benchmark run \
  --config configs/locomo.yaml \
    --pack private \
    --data-dir /path/to/your/data/ \
    --output-dir data/output/

# Windows
benchmark run `
  --config configs/locomo.yaml `
    --pack private `
    --data-dir C:\path\to\your\data\ `
    --output-dir data/output\
```

---

## Decay Policy Reference

| Policy | Config `type` | When to use |
|--------|---------------|-------------|
| No decay | *(omit decay block)* | Testing / ablation. All memories survive forever. |
| Exponential | `exponential` | Default. Memories fade naturally. λ=0.05 is a good start. |
| Linear | `linear` | Uniform fade. Better for preference memories with known lifespan. |
| Step | `step` | Hard expiry at a threshold day. Useful for session-scoped memory. |

### Lambda guide

| λ value | Half-life (days) | Use case |
|---------|-----------------|----------|
| 0.01 | 69 | Very slow fade — long-term factual memory |
| 0.03 | 23 | Slow fade — preference and entity memory |
| 0.05 | 14 | **Default** — episodic memory |
| 0.07 | 10 | Moderate — high-churn sessions |
| 0.10 | 7 | Fast — short sessions (weekly resets) |
| 0.15 | 5 | Very fast — nearly session-scoped |
| 0.20 | 3 | Aggressive — scratch-pad style memory |

Half-life formula: $t_{1/2} = \ln(2) / \lambda$

### Example config

```yaml
# configs/locomo.yaml
policies:
  module_policies:
    episodic_store:
      decay:
        type: exponential
        lambda: 0.05        # memories at half-strength after 14 days
      pruning:
        strategy: score_threshold
        threshold: 0.35     # remove memories scoring below 35%
```

---

## Reproducibility

Every benchmark run is deterministic given the same config + seed:

```bash
# Run twice with the same seed — results must be identical
python3 matrix_runner.py --mode core3x3 --seed 42 ...
python3 matrix_runner.py --mode core3x3 --seed 42 ...
```

The run ID in the output file name will differ (it's a UUID), but all metric
values will be byte-for-byte identical.

To pin a seed in the config:

```yaml
benchmark:
  seed: 42   # any integer
```

---

## Hardware Guide

| Run mode | Min RAM | Recommended | Min CPU cores | Time estimate |
|----------|---------|-------------|---------------|---------------|
| core 3×3 (lite dataset) | 4 GB | 8 GB | 2 | 5–15 min |
| lambda-sweep | 4 GB | 8 GB | 2 | 2–5 min |
| core 3×3 (production dataset) | 8 GB | 16 GB | 4 | 30–60 min |
| full 700-cell (production) | 16 GB | 32 GB | 8 | 2–6 hours |

### Parallel workers

The `--workers` flag controls how many benchmark cells run in parallel.
Default is `cpu_count - 1`.

```bash
# Limit to 4 workers on a shared machine
python3 matrix_runner.py --mode full --workers 4 ...
```

### Disk space

| Dataset | Approximate size |
|---------|-----------------|
| Lite gold dataset | 130 MB |
| Production gold dataset | 350 MB |
| Output per matrix run (core3x3) | 1–5 MB |
| Output per matrix run (full) | 20–50 MB |

---

## User Isolation

When benchmarking with multiple users in the same dataset, the benchmark
**guarantees** that user A's queries cannot retrieve user B's memories.

This is enforced at the memory store layer, not at the config or orchestrator.
See [docs/USER_ISOLATION.md](USER_ISOLATION.md) for the full guarantee and
threat model.

To verify isolation yourself:

```bash
# macOS / Linux
python3 -m pytest tests/unit/test_user_isolation.py -v

# Windows
python -m pytest tests/unit/test_user_isolation.py -v
```

Expected: **14 passed**

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'benchmark'`

The package is not installed. Run:

```bash
pip install -e ".[dev]"
```

### `pydantic.ValidationError` on config load

Check your YAML indentation. The `lambda` key must be nested under `decay`:

```yaml
# CORRECT
policies:
  module_policies:
    episodic_store:
      decay:
        type: exponential
        lambda: 0.05

# WRONG — lambda at wrong level
policies:
  module_policies:
    episodic_store:
      lambda: 0.05
```

### Tests fail with `RecursionError` or import errors on Windows

Ensure you are inside the `if __name__ == '__main__':` guard when calling
`matrix_runner.py` directly. The file already has this guard. Do not import
`matrix_runner` as a module.

### `OSError: [Errno 28] No space left on device`

The production dataset is ~350 MB. Ensure at least 2 GB free on the target disk.

### Matrix runner hangs on Windows with `--workers > 1`

Windows requires the `spawn` multiprocessing context (already configured).
If the runner hangs, check that your antivirus is not blocking subprocess creation.
Use `--workers 1` as a workaround to run cells serially.

### `TimeoutError` on dataset download

The LongMemEval dataset download may time out on slow connections. Use the lite
generator instead:

```bash
python3 generate_production_dataset.py --lite --output-dir data/generated/lite
```
