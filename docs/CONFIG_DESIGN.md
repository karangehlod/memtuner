# Configuration Design — .env vs YAML

## Principle

**`.env` holds secrets and endpoint overrides only.**
**YAML holds everything about what to benchmark.**

No hardcoded model lists, strategy names, skip flags, or provider names in Python code.
If you want to change which models are benchmarked, edit the YAML. If you want to change
where to connect, edit `.env`. You never need to touch code for either.

---

## What Goes Where

### `.env` — secrets and endpoints only

| Variable | Example | Why here |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Secret |
| `HF_TOKEN` | `hf_...` | Secret |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Secret |
| `BENCHMARK_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Endpoint — changes per environment |

That is the entire `.env`. Nothing else.

To use Ollama instead of OpenAI, set:
```bash
BENCHMARK_OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```
No code change, no provider flag, no YAML change.

### YAML — everything about what to benchmark

- Which embedding models to run and on which backend (`sentence-transformers`, `mlx`, `api`)
- Which LLM/judge model to use and its backend
- Which memory types to benchmark
- Which strategies (BM25, semantic, hybrid, llm_rerank)
- Decay and pruning policies
- Skip rules (e.g. skip models larger than X MB)
- Deployment profile (`local-mac`, `cloud-gpu`, `api`)
- Seed, evaluation horizon, max queries

---

## Target `.env` (clean)

```bash
# ── Secrets ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY=your_key_here
HF_TOKEN=hf_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here     # optional

# ── Endpoint overrides ───────────────────────────────────────────────────────
# Default: OpenAI. Override to point at Ollama, vLLM, Azure, Together, etc.
# BENCHMARK_OPENAI_BASE_URL=https://api.openai.com/v1
# BENCHMARK_OPENAI_BASE_URL=http://localhost:11434/v1     # Ollama
# BENCHMARK_OPENAI_BASE_URL=http://my-vllm-server/v1     # self-hosted
```

Everything else moves to YAML.

---

## Target YAML structure

```yaml
# configs/local_mac.yaml  — for Apple Silicon local validation (no API cost)

profile: local-mac          # drives hardware-aware defaults in resolver

embedding:
  backend: sentence-transformers    # or: mlx, api
  models:
    - name: BAAI/bge-m3
    - name: BAAI/bge-base-en-v1.5
    - name: all-MiniLM-L6-v2
    - name: google/embeddinggemma-300m
    - name: Qwen/Qwen3-Embedding-0.6B
  skip:
    - Qwen/Qwen3-Embedding-4B       # too slow for local validation

judge:
  backend: mlx                      # or: api
  model: mlx-community/Qwen3-8B-MLX

strategies:
  - bm25
  - semantic
  - hybrid
  - llm_rerank

memory:
  enabled:
    long_term:
      - episodic_store
      - semantic_store
      - preference_store
      - entity_store

benchmark:
  seed: 42
  max_queries: null       # null = all
  evaluation_horizon: 180

policies:
  episodic_store:
    decay: {type: exponential, lambda: 0.05}
    pruning: {strategy: score_threshold, threshold: 0.35}
  semantic_store:
    decay: {type: exponential, lambda: 0.03}
    pruning: {strategy: score_threshold, threshold: 0.30}
```

```yaml
# configs/api.yaml  — for org/team production API benchmarking

profile: api

embedding:
  backend: api
  models:
    - name: text-embedding-3-small
    - name: text-embedding-3-large

judge:
  backend: api
  model: gpt-4o

# rest inherits from base config
```

```yaml
# configs/cloud_gpu.yaml  — for cloud VM reference runs

profile: cloud-gpu

embedding:
  backend: sentence-transformers
  models:
    - name: BAAI/bge-m3
    - name: Qwen/Qwen3-Embedding-0.6B

judge:
  backend: api                        # vLLM endpoint set via BENCHMARK_OPENAI_BASE_URL
  model: meta-llama/Llama-3-70B-Instruct
```

---

## What to Remove from Current `.env`

These variables are currently in `.env` but belong in YAML or should be deleted:

| Variable | Action |
|---|---|
| `BENCHMARK_OLLAMA_BASE_URL` | **Delete** — Ollama uses `BENCHMARK_OPENAI_BASE_URL` |
| `BENCHMARK_OLLAMA_API_KEY` | **Delete** — deprecated |
| `BENCHMARK_OLLAMA_TIMEOUT` | **Delete** — move timeout to YAML per-backend |
| `BENCHMARK_OLLAMA_EMBEDDING_MODEL` | **Delete** → YAML `embedding.models` |
| `BENCHMARK_OLLAMA_EMBEDDING_MODELS` | **Delete** → YAML `embedding.models` |
| `BENCHMARK_LLM_BASE_URL` | **Delete** — same endpoint as `BENCHMARK_OPENAI_BASE_URL` |
| `BENCHMARK_LLM_MODEL` | **Delete** → YAML `judge.model` |
| `BENCHMARK_JUDGE_BASE_URL` | **Delete** — same endpoint as `BENCHMARK_OPENAI_BASE_URL` |
| `BENCHMARK_JUDGE_MODEL` | **Delete** → YAML `judge.model` |
| `BENCHMARK_EMBEDDING_MODEL` | **Delete** → YAML `embedding.models` |
| `BENCHMARK_EMBEDDING_MODELS` | **Delete** → YAML `embedding.models` |
| `BENCHMARK_EMBEDDING_PROVIDER` | **Delete** → YAML `embedding.backend` |
| `BENCHMARK_LLM_PROVIDER` | **Delete** → YAML `judge.backend` |
| `BENCHMARK_JUDGE_PROVIDER` | **Delete** → YAML `judge.backend` |
| `BENCHMARK_EMBEDDING_ROUTING_MODE` | **Delete** → YAML `embedding.backend` |
| `BENCHMARK_LOCAL_MODEL_SIZE_THRESHOLD_MB` | **Delete** → YAML `embedding.skip` |
| `BENCHMARK_RERANKER_STRATEGY` | **Delete** → YAML `strategies` list |
| `BENCHMARK_RERANKER_MODEL` | **Delete** → YAML `judge.model` |
| `BENCHMARK_RERANKER_API_PROVIDER_ORDER` | **Delete** → YAML `embedding.backend` |
| `BENCHMARK_ANALYZE_SKIP_*` | **Delete** → YAML `embedding.skip` |
| `BENCHMARK_WORKERS` | **Delete** → CLI flag `--workers` |
| `BENCHMARK_SKIP_MODELS` | **Delete** → YAML `embedding.skip` |
| `BENCHMARK_DATASET` | **Delete** → CLI flag `--dataset` or auto-discovery |
| `BENCHMARK_SEED` | **Delete** → YAML `benchmark.seed` |

---

## Implementation TODOs

- [ ] Update `benchmark/config/loader.py` to read model lists from YAML `embedding.models` list (not env vars)
- [ ] Update `benchmark/config/schema.py` to add `embedding.backend`, `embedding.models`, `judge.backend`, `judge.model`, `profile` fields
- [ ] Update `benchmark/factory/resolver.py` to resolve backend from YAML `embedding.backend` instead of env var provider flags
- [ ] Create `configs/local_mac.yaml`, `configs/api.yaml`, `configs/cloud_gpu.yaml` as the three standard profiles
- [ ] Create `configs/base.yaml` with shared defaults (policies, strategies, memory types) that profiles extend
- [ ] Strip `.env` down to the 4 variables above (secrets + base_url)
- [ ] Update `.env.example` to match the clean target
- [ ] Update `study_runner.py` to accept `--config configs/local_mac.yaml` and load model list from it
- [ ] Remove all `BENCHMARK_OLLAMA_*` env var reads from the codebase
