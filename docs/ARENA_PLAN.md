# MemTuner Arena — Execution Plan

**Goal:** position MemTuner as the neutral, reproducible referee for agent-memory
systems. The deliverable is a public, versioned comparison of Mem0, Zep, Letta,
and a plain-RAG baseline on LoCoMo and LongMemEval — LLM-judged, holdout-honest,
with confidence intervals, cost, and latency — reproducible with one command.

**Why us:** vendor numbers in this space are self-reported and publicly disputed
(Mem0 vs Zep over LoCoMo). Nobody runs all systems through one identical
protocol. MemTuner already has the datasets, the judge pipeline
(`benchmark/judge/`), bootstrap CIs, and — after the 2026-09-22 fixes — a
holdout protocol that fails loudly instead of reporting silent zeros.

**Non-goal:** beating Mem0/Zep/Letta on accuracy. MemTuner's internal stack is
the *baseline*, not a contender. Every claim we publish is about measurement,
not victory.

---

## Ground rules for every number (all phases)

These are the rules that make the numbers proper. A violation blocks
publication, no exceptions.

1. **No silent zeros.** A cell with `total_queries == 0` is a failed cell
   (enforced in `study_scheduler.py` since 2026-09-22). Aggregates must exclude
   failed cells and say how many were excluded.
2. **Holdout always on.** `--test-holdout-fraction 0.20` for tuning,
   `--final-test-holdout-fraction` for the winner. Full-pool numbers (phase-3
   fast sweep) are for weight-curve plots only and never appear in rankings or
   public tables.
3. **Dataset fitness is checked, not assumed.** `memtuner doctor` must pass a
   dataset before it enters a run. Same-day-gold datasets (SQuAD, CoQA,
   PersonaChat, HotpotQA, MS MARCO, etc.) are static-retrieval smoke tests
   only; they never appear in memory-system comparisons.
4. **N ≥ 30 or it's an anecdote.** Public CI tables require ≥ 10 seeds. Numbers
   with lower N are labeled "preliminary" and kept off the arena page.
5. **Leakage is disclosed.** Every dataset row carries its query–corpus overlap
   percentage (already computed by `generate_reports.py`).
6. **Judged accuracy is the only cross-system metric.** External systems
   rewrite memories at ingest, so gold-ID Recall@K is undefined for them.
   Recall@K / MRR / composite remain internal-stack tuning metrics.
7. **Published results are immutable.** Results live in `results/vX.Y/` with
   the exact commit hash, config, and command. Corrections ship as a new
   version with a changelog entry, never as an edit.
8. **Every public number has a repro command** printed next to it.
9. **Zero cost, fully local, open-source only** (maintainer decision,
   2026-09-22). Every arena system, the judge, and all embedders run locally
   on open-source software with no per-call billing. Consequences:
   - Managed services are out of the roster: no Zep Cloud, no Mem0 Platform.
     `ZepStore` stays in the codebase (dormant, needs `ZEP_API_KEY`) for
     anyone who wants to fund that comparison themselves.
   - Zep's slot is taken by **Graphiti** (getzep/graphiti) — the open-source
     temporal-knowledge-graph framework that powers Zep — labeled as such,
     never as "Zep".
   - Where a vendor's OSS defaults bill an API (Mem0 → OpenAI), the LLM and
     embedder switch to the vendor's *documented* Ollama provider path, the
     same local models are used consistently across systems where possible,
     and every such switch is a disclosed deviation in `configs/arena/`.
   - Reproduction cost for readers is GPU time only — which also removes the
     Phase-2 budget sign-off as a blocking step (it becomes a wall-clock
     estimate).
   - This is also the positioning: vendors publish managed-platform numbers;
     the arena answers "what do these systems do when *you* run them, for
     free, on your own hardware."

---

## Phase 0 — Credibility groundwork

**Goal:** our own house is in order: harness bugs fixed, invalid data
quarantined, core datasets re-measured properly, postmortem published.

### Tasks

| # | Task | Notes |
|---|------|-------|
| 0.1 | Commit the 2026-09-22 fixes | loader split-basis, zero-query→failure, doctor scorability check, regression tests (already in working tree) |
| 0.2 | Diagnose the reranker negative lift | Phase-5 showed cross-encoders *hurting* recall (−0.28 squad, −0.06 synthetic). Almost certainly an integration bug (score ordering, truncation, wrong text field). Use `scripts/diagnose_cell.py` on one reranker cell; fix or confirm as a real finding. Blocks any publication that mentions rerankers. |
| 0.3 | Quarantine invalid results | Move pre-fix `data/output` studies aside; rebuild `master_results.csv`, `leaderboards.json`, and dashboards from valid cells only (LoCoMo, LongMemEval, Synthetic; drop the 196 zero-query cells and all same-day-dataset rows from rankings). |
| 0.4 | Pin the judge and document the reported-run protocol | Done 2026-09-22: `judge.model` pinned in `configs/study_defaults.yaml` (and actually wired — it was dead config), endpoint via `--ollama-url` / `BENCHMARK_JUDGE_BASE_URL`; full procedure in `docs/RUNBOOK.md` § "Reported Runs". |
| 0.5 | Re-run core datasets on the NVIDIA machine | LoCoMo + LongMemEval + Synthetic. 10 seeds, judge enabled, **no horizon/workload override** — natural spans (LoCoMo 722d, LongMemEval 180d) activate phase-4b where the timeline supports it. Exact command in `docs/RUNBOOK.md` § "Reported Runs". PersonaChat re-converted (`prepare_datasets.py --convert --force`) but reported as static-retrieval only. |
| 0.6 | Write & publish the postmortem | "Our benchmark reported 0.96 recall. It was measuring nothing." — horizon/holdout interaction, same-day gold, 83% leakage, silent-success cells. Publish on the blog before anything arena-related. |

### Acceptance criteria

- [ ] `pytest tests/unit -q` green on main; the five new scenario regression tests included.
- [ ] `memtuner doctor` warns on squad/coqa/personachat/conflict_update gold files and passes locomo/longmemeval/synthetic.
- [ ] A deliberate repro (squad + holdout, pre-fix commit) fails cells with the zero-query error; same run on main scores 2,288 queries.
- [ ] New `master_results.csv` contains **zero** rows with `total_queries == 0` and `success == True`.
- [ ] LoCoMo & LongMemEval CI tables show N ≥ 30 per strategy; no "N<10" footnotes on any headline number.
- [ ] Judge scores exist for the **winner config and arena cells** on LoCoMo +
      LongMemEval (judging every tuning cell is infeasible: ~141 cells × 470
      queries × ~3 s/judge ≈ 55 h *per seed* on LongMemEval alone — corrected
      2026-09-22; the tuning study runs judge-off, the winner and arena run
      judge-on). Synthetic has no gold answers yet, so it is retrieval-only
      until its generator emits them.
- [ ] Reranker lift is either positive/neutral after a fix, or documented as a verified finding with the diagnosis.
- [ ] Postmortem is live and links to the fixing commits.

**Exit gate:** we can state our own baseline numbers (internal stack, judged,
holdout, N≥30) and defend every one of them. These become the plain-RAG rows
of the arena.

---

## Phase 1 — External-system adapters

**Goal:** Mem0, Zep, and Letta run inside the harness through the same
`BaseLongTermStore` contract as our own stores, scored by the judge.

### Design decisions (made now, documented in code)

- **Contract:** adapters implement `write(event)` / `write_on_day(event, day)` /
  `read(query) → ReadResponse` from `benchmark/memory/long_term/base_store.py`.
  New package: `benchmark/memory/external/` (`mem0_store.py`, `zep_store.py`,
  `letta_store.py`).
- **Scoring mode:** new per-cell field `scoring: gold_ids | judge`. External
  systems always run `judge`. Gold-ID metrics are recorded as `NaN`, never `0`.
- **Fairness protocol:** each system runs vendor-recommended defaults from its
  quickstart docs, config committed verbatim to `configs/arena/<system>.yaml`.
  No per-system tuning in v1 (tuning-them-equally is a later feature, and it's
  MemTuner's actual product).
- **Isolation:** fresh collection/user per cell; teardown verified (`count()`
  returns 0 after `clear()`); no state bleed across seeds.
- **Cost & latency:** per-query wall time and token/API cost recorded via the
  existing `cost_tracker` — these are first-class arena columns.
- **Nondeterminism:** external ingest is LLM-based and nondeterministic; seeds
  vary the *system*, not just our sampler. Report per-seed spread explicitly.

### Tasks

| # | Task | Notes |
|---|------|-------|
| 1.1 | Adapter contract test suite | Done 2026-09-22: `tests/contract/test_external_store_contract.py` — 11 tests against `FakeMem0Client` in CI (gold-ID mapping, split-fact dedupe, namespace isolation, teardown, lifecycle no-ops); real-backend roundtrip gated behind `MEM0_CONTRACT_TEST=1`. |
| 1.2 | `Mem0Store` | Done 2026-09-22 (adapter): `benchmark/memory/external/mem0_store.py` — gold-ID provenance via `metadata={"gold_id": …}` (recall is a lower bound for Mem0; judge stays primary), per-cell namespace, injectable client. Remaining: resolver/CLI wiring (`--backend mem0`) lands with 1.3; `configs/arena/mem0.yaml` is DRAFT until vendor-doc citation + SDK pin are filled in. |
| 1.3 | Judge-primary scoring path | Mostly done 2026-09-22: `llm_judge_score`/`llm_judge_queries` now survive worker→result→grid-CSV round-trips (previously computed and silently dropped — judge-enabled runs discarded their own judge data); `StudyCell.backend` + `scoring_mode` flow cell→config→CSV; external configs load `<backend>_store` with no strategy (no model loading); external cells excluded from native rankings via `_is_sweep_variant`. Remaining: judge column in dashboards/plots. Arena run mode done 2026-09-22: `scripts/arena_runner.py` — flat ingest→query→judge cells (no tuning phases), refuses to run without a judge (exit 2; `--allow-no-judge` prints a do-not-publish warning), natural-span horizon, native baseline runs the FULL system via `memory_type="all"`. Smoke-verified end to end: native bm25l reproduces the known 0.1441 on synthetic; a mem0 cell without the SDK fails loudly with an actionable error and exit 1. Bonus fix: multi-store configs shared ONE retrieval-strategy instance, so the last store to index (often empty) wiped the index for the rest — every multi-store native config silently scored 0; composer/resolver now build one strategy instance per module (regression-tested). |
| 1.4 | End-to-end smoke: Mem0 × synthetic_gold | 200 queries, 3 seeds, judge scores recorded, cost/latency populated. |
| 1.5 | `ZepStore`, `LettaStore` | ZepStore done 2026-09-22 — but the plan's docker-compose assumption is dead: **Zep Community Edition is deprecated** (getzep/zep README), so the adapter targets **Zep Cloud** (zep-cloud 3.28.0, `ZEP_API_KEY`), an asymmetry disclosed in `configs/arena/zep.yaml`. Retrieval follows Zep's own LongMemEval benchmark recipe (edges + nodes searches). Zep exposes no scores → synthesized rank scores; gold-ID metrics undefined → judge-primary. Enabling fix: `RetrievedMemory.content` — external systems return rewritten text whose IDs resolve to nothing, so without adapter-carried content the judge scored them against an EMPTY context (silent zero for every external system). Runner now prefers vendor content, falls back to gold lookup for native stores. `LettaStore` still todo (verify current Letta SDK first — vendor APIs churn; Mem0's needed a filters/top_k rewrite, Zep's needed a cloud pivot). **Superseded by ground rule 9 (zero-cost):** ZepStore is dormant (cloud-only, needs `ZEP_API_KEY`). `GraphitiStore` done 2026-09-22 — graphiti-core 0.30.2 API verified (async add_episode/search/clear_data, sync-wrapped on a private loop); fully-local stack from the vendor's own documented Ollama recipe (deepseek-r1:7b + nomic-embed-text) + embedded FalkorDB Lite (no server); `group_id` partitions for per-cell isolation; simulated day N → real `reference_time` (ARENA_EPOCH + N days) so Graphiti's temporal reasoning sees the timeline; judge-primary with fact content, synthesized rank scores (none exposed). Cited config: `configs/arena/graphiti.yaml` (pin `graphiti-core[falkordblite]==0.30.2`, Python 3.12+). 7 contract tests vs an API-faithful fake. Letta **deferred** (verified 2026-09-22): the self-hostable V1 Python API server is retired to letta-ai/letta's `archive` branch; current "Letta Code" is npm-installed with a TypeScript-first SDK. No stable Python memory API to build against — revisit if/when one exists. Arena v1 roster is therefore **native + mem0 + graphiti** (+ dormant zep-cloud). |
| 1.6 | Baseline config freeze | Best internal config from Phase 0 frozen as `configs/arena/baseline_rag.yaml`. |

### Acceptance criteria

- [ ] All three adapters pass the contract suite against real backends (documented local run) and the fake in CI.
- [ ] Mem0 smoke run: 0 failed cells, judge score present for 100% of scored queries, cost > $0 recorded, p50 latency recorded.
- [ ] Gold-ID metrics for external cells appear as N/A (not 0) in CSV, dashboards, and leaderboards.
- [ ] Two consecutive identical smoke runs differ only within seed noise (no state bleed — verified by fresh-collection assertion in teardown).
- [ ] Each `configs/arena/<system>.yaml` cites the vendor doc page it was taken from, with date.

**Exit gate:** `memtuner study --backend mem0 --gold-dataset synthetic_gold.json`
completes clean, and we trust the pipeline enough to spend real judge budget on
full datasets.

---

## Phase 2 — The arena report

**Goal:** one immutable, versioned results page that survives hostile scrutiny —
assume Mem0's and Zep's maintainers read it looking for mistakes, because they will.

### Protocol (frozen before the runs, in `docs/ARENA_METHODOLOGY.md`)

- **Datasets:** LoCoMo; LongMemEval-S **full haystack** (not oracle — oracle is
  evidence-only and inflates everyone). Licenses verified for redistribution
  *before* anything is uploaded (task 2.1).
- **Judge:** one pinned model + version, temperature 0, prompts published
  verbatim. Judge-robustness check: second judge model on a 100-query sample;
  report agreement. Human spot-check of 50 judged answers by the maintainer.
- **Seeds:** ≥ 5 per system×dataset for external systems (API cost caps this;
  disclose), ≥ 10 for the baseline.
- **Budget:** estimate judge + ingest API cost per system×dataset **before**
  running; get sign-off; record actuals in the report.
- **Table columns:** judged accuracy (95% CI), ingest cost /1k memories, query
  p50/p95 latency, N, run date, commit hash.

### Tasks

| # | Task | Notes |
|---|------|-------|
| 2.1 | License audit | LoCoMo, LongMemEval redistribution rights; decide HF upload vs. download-script-only per dataset. Blocks Phase 3 uploads. |
| 2.2 | `docs/ARENA_METHODOLOGY.md` | Protocol above + **Limitations** section written against ourselves: judge bias, external nondeterminism, no-per-system-tuning, leakage figures, what we cannot measure. |
| 2.3 | LongMemEval-S full-haystack profile run | Verify the profile machinery (commit e7b4caf) produces the full corpus; doctor-check it. |
| 2.4 | Arena runs | 4 systems × 2 datasets per protocol, on the NVIDIA machine. |
| 2.5 | Results artifact | `results/v0.1/` (frozen CSVs + config + commands) and a GitHub Pages arena page generated from it. Charts follow `docs/PLOTS_GUIDE.md` conventions: CIs drawn, no unlabeled axes. |
| 2.6 | Independent reproduction | One person who is not the author reproduces the baseline + Mem0 rows from README instructions alone on a machine we don't control. Their friction log becomes fixes. |

### Acceptance criteria

- [ ] Methodology doc frozen (committed) **before** arena runs start; any later change restarts affected runs.
- [ ] Judge agreement between the two judge models ≥ 85% on the sample; human spot-check finds ≤ 5/50 wrong verdicts. Below threshold → judge prompt iterated and *all* runs redone.
- [ ] Zero failed cells in published runs; excluded-cell counts (if any) disclosed on the page.
- [ ] Every table cell traceable: `results/v0.1/` contains the command, config, commit, and raw per-query judge outputs that produced it.
- [ ] Independent reproduction matches published numbers within the stated CIs.
- [ ] Where our numbers diverge from a vendor's published claim, the divergence is stated neutrally with both numbers and our config linked — no editorializing.

**Exit gate:** the page is something we'd be comfortable having quote-tweeted
by the Mem0 and Zep founders with "this is wrong because…" — because every
answer to that is already on the page.

---

## Phase 3 — Launch & adoption

**Goal:** the right 500 people see it; reproduction is trivial; the project has
defined success/kill criteria.

### Tasks

| # | Task | Notes |
|---|------|-------|
| 3.1 | One-command repro | `uvx memtuner arena --systems mem0 --dataset locomo --judge <model>` works on a clean machine (verified in Docker). README quickstart ≤ 5 lines. |
| 3.2 | README restructure | Arena results table above the fold → repro command → methodology link. Tuner documentation moves below. |
| 3.3 | Dataset/artifact publishing | Per 2.1: HF dataset uploads with cards + licenses, or converter scripts where redistribution isn't allowed. |
| 3.4 | Launch sequence | (a) postmortem already live from Phase 0; (b) arena page + Show HN + r/LocalLLaMA post, framed as "we measured, here's the harness" not "we built a better X"; (c) issues on vendor repos where our numbers diverge, each with repro command attached. |
| 3.5 | Standing engagement | When any vendor publishes a new benchmark claim, reproduce within a week and post results in that thread — confirming **or** refuting. Confirmations build the referee brand as much as refutations. |

### Acceptance criteria

- [ ] Docker-clean reproduction of one arena row by a first-time user in ≤ 30 minutes, ≤ 5 copy-pasted commands.
- [ ] No published artifact violates a dataset license (2.1 audit signed off).
- [ ] Launch posts link only to versioned, immutable results.
- [ ] Vendor-repo issues are filed with configs + repro attached — no accusation framing.

### Success / kill criteria (evaluate 3 months after launch)

**Continue investing if any of:** an external contributor lands a PR or files a
substantive methodology issue; a vendor engages with the numbers publicly;
arena results are cited in a thread/post we didn't start; sustained
star/traffic growth after launch week.

**Freeze if none of the above:** tag a final release, mark the arena "archived
snapshot," keep the postmortem and methodology as portfolio. No slow-drip
maintenance of vendor adapters nobody uses.

---

## Risk register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Judge bias favors some system's answer style | Wrong rankings | Two-judge agreement check + human spot-check (Phase 2 gate); prompts published |
| Vendor API/SDK churn breaks adapters | Repro rot | Pin SDK versions in `configs/arena/`; contract tests in CI catch drift |
| Dataset licenses forbid redistribution | Launch blocker | 2.1 audit before any upload; fall back to converter scripts |
| Judge/API cost blowout | Budget | Pre-run cost estimate + sign-off (Phase 2); cap seeds for external systems, disclose |
| "Unfair config" accusations from vendors | Credibility | Vendor-doc-cited default configs, committed verbatim; invite config PRs from vendors — accepting one is a win |
| Reranker finding is our bug, published as fact | Credibility | 0.2 blocks publication until diagnosed |
| Same-day datasets sneak back into aggregates | Wrong numbers | Doctor warnings + rankings filter (`_is_ranking_comparable`); CI test asserting excluded datasets |

## Sequencing

Phases are strictly ordered by their exit gates; tasks within a phase can
parallelize. Rough effort at side-project pace: Phase 0 ≈ 1–2 weekends (mostly
NVIDIA re-run wall time), Phase 1 ≈ 2–3 weekends, Phase 2 ≈ 2 weekends + judge
budget + reproduction lag, Phase 3 ≈ 1 weekend + ongoing engagement. Nothing in
Phases 1–3 starts before the Phase 0 exit gate: publishing anything while our
own numbers are unfixed forfeits the referee position permanently.
