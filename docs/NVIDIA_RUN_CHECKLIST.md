# NVIDIA Machine Run — Follow-Along Checklist

The exact sequence for producing the Phase-0 reported numbers and passing the
Phase-1 gate (see `docs/ARENA_PLAN.md`). Work top to bottom; every stage has a
**verify** step — do not continue past a failed verify. PowerShell syntax,
repo at `D:\code\memtuner`.

**Stop conditions (applies to every stage):** any `✗` cell, any `✓` cell with
recall *and* MRR *and* latency all exactly 0.0, or a missing judge banner in a
stage that requires one → stop, save the log, investigate before continuing.
Everything here is free and local — if something asks for an API key, that is
itself a bug.

---

## Stage 0 — Sync & verify the code

- [ ] `cd D:\code\memtuner`
- [ ] `git pull`
- [ ] `pip install -e ".[dev]"`
- [ ] `python -m pytest tests\unit tests\contract -q`
      **Verify:** ~1122 passed, 0 failed.
- [ ] `python -m ruff check benchmark/ scripts/`
      **Verify:** "All checks passed!" (same as CI).

## Stage 1 — One-time machine setup

- [ ] Ollama models (~13 GB total):
  ```powershell
  ollama pull nemotron-3-nano      # pinned judge
  ollama pull llama3.1:8b          # Mem0 local extraction LLM
  ollama pull nomic-embed-text     # Mem0 + Graphiti embedder
  ollama pull deepseek-r1:7b       # Graphiti extraction LLM
  ```
- [ ] Backend SDKs:
  ```powershell
  pip install mem0ai==2.1.0
  python --version
  # Python >= 3.12:
  pip install "graphiti-core[falkordblite]==0.30.2"
  # Python 3.10/3.11 instead:
  #   pip install "graphiti-core[falkordb]==0.30.2"
  #   docker run -d -p 6379:6379 falkordb/falkordb:latest
  #   then set graph_backend: "falkordb" in configs\arena\graphiti.yaml
  ```
- [ ] Refresh converted datasets (required once — PersonaChat pooling fix):
  ```powershell
  python scripts\prepare_datasets.py --convert --force
  ```
- [ ] `memtuner doctor`
      **Verify:** locomo10 / longmemeval / synthetic pass; squad, coqa,
      personachat, conflict_update show the holdout-scorability **warning**
      (that warning is correct behavior, not a problem).

## Stage 2 — Tuning study, judge OFF (overnight, ~6–10 h)

Judge stays off here on purpose: judging every tuning cell would take weeks
(~3 s/query × 470 queries × 141 cells × 10 seeds on LongMemEval alone).

- [ ] ```powershell
      python scripts\study_runner.py --mode full `
        --gold-dataset data\input\locomo10.json data\input\longmemeval_oracle_gold.json data\input\synthetic_gold.json `
        --seeds 42 123 456 777 1010 2024 3141 4242 5555 8888
      ```
      Flags deliberately absent: `--ollama-url` (judge off),
      `--evaluation-horizon` / `--workload` (natural spans).
- [ ] Watch the first ~10 cells before walking away.
      **Verify:** startup prints the judge-skipped warning (expected);
      per-cell recalls are nonzero and *plausible* (LoCoMo episodic bm25
      ≈ 0.25–0.35, not 0.000 and not 0.95).
- [ ] After completion — **Verify:**
  - `STATISTICAL SIGNIFICANCE` tables show **N ≥ 30** per strategy, no
    `N<10` footnotes on headline strategies.
  - The report prints no `[error]` lines; failed-cell count is 0.

## Stage 3 — Freeze the winner, judge it (~4–6 h)

- [ ] Copy the Stage-2 `TOP-RANKED` configuration into
      `configs\arena\baseline_rag.yaml`:
      ```yaml
      baseline:
        retrieval_strategy: <winner strategy>
        embedding_model: <winner model>
        embedding_backend: sentence-transformers
        bm25_weight: <winner weight>
      ```
      (Or send the report back and have this filled in for you.)
- [ ] ```powershell
      python scripts\arena_runner.py --systems native `
        --gold-dataset data\input\locomo10.json data\input\longmemeval_oracle_gold.json `
        --seeds 42 123 456 `
        --ollama-url http://localhost:11434/v1
      ```
      **Verify:** banner prints `LLM judge: nemotron-3-nano:4b @ …` — if it
      doesn't, STOP (these numbers are the ones we compare externally).
      Synthetic is excluded here: it has no gold answers to judge yet.
- [ ] **Verify:** summary table shows a numeric `judge` column (not `—`),
      fail = 0.

## Stage 4 — Real-backend validation (Phase-1 gate, ~1–2 h)

- [ ] Real-SDK contract tests (CI only ever runs the fakes):
      ```powershell
      $env:MEM0_CONTRACT_TEST = "1"
      python -m pytest tests\contract\test_external_store_contract.py -q
      ```
      **Verify:** the previously-skipped real-backend test passes.
- [ ] First real multi-system arena run (synthetic = smallest dataset, keeps
      per-event LLM extraction tractable):
      ```powershell
      python scripts\arena_runner.py --systems native mem0 graphiti `
        --gold-dataset data\input\synthetic_gold.json `
        --seeds 42 `
        --ollama-url http://localhost:11434/v1
      ```
      Expect Mem0/Graphiti ingest to be slow — one local-LLM extraction call
      per memory event is the product working, not a hang. Judge shows `—`
      (synthetic has no gold answers); this stage validates the pipelines.
      **Verify:** fail = 0 for all three systems; mem0's recall column is
      nonzero (gold-ID provenance working).

## Stage 5 — Bring results home

- [ ] Copy back to the analysis machine:
      `data\output\master_results.csv`, all new `data\output\study_*` dirs,
      `benchmark_results\leaderboards.json`, and the console logs.
- [ ] On the analysis machine:
      `python scripts/generate_reports.py --from-master data/output/master_results.csv`
- [ ] Hand over for acceptance-criteria review (ARENA_PLAN Phase 0 + Phase 1
      exit gates) and the reranker-residual check (bge-reranker −0.053
      question from the 2026-09-22 diagnosis).

## Wall-clock summary

| Stage | What | Time |
|---|---|---|
| 0–1 | sync, models, datasets | ~1 h (mostly downloads) |
| 2 | tuning study, 3 datasets × 10 seeds, judge off | ~6–10 h (overnight) |
| 3 | winner judged, 2 datasets × 3 seeds | ~4–6 h |
| 4 | real contract tests + 3-system arena smoke | ~1–2 h |
| 5 | copy back + reports | ~15 min |
