# MemTuner — Plots Guide

What every generated chart shows, how to read it, and what to watch out for.

MemTuner produces two plot families:

| Family | Location | Generator | When |
|--------|----------|-----------|------|
| **Cross-run report plots** | `data/output/plots/` | `scripts/plot_benchmark.py` | After `generate_reports.py` (auto after every study, or manually with `--from-master`) |
| **Per-study plots** | `data/output/study_<id>/` and `merged_<ts>/` | `benchmark/reporting/study_visualizer.py` | At the end of every study run / merge |

Regenerate manually:

```bash
python scripts/plot_benchmark.py                       # scan local study_* dirs
python scripts/plot_benchmark.py --from-master data/output/master_results.csv
python scripts/plot_benchmark.py --csv path/to/one_grid.csv   # single run
```

---

## Shared conventions (read this first)

**Composite score.** Everywhere a "composite" appears it is the same number:

```
composite = gate(R@10 ≥ 0.01) × (0.4·R@10 + 0.25·P@10 + 0.2·MRR + 0.15·TA) / Σw_active
```

- `gate` zeroes the score when Recall@10 < 0.01 (an empty store must not win on temporal accuracy alone).
- `Σw_active` is the renormalization divisor: **0.85** when the dataset has no temporal signal (TA = 0, the temporal weight drops out), **1.00** otherwise. This keeps composites comparable between temporal and non-temporal datasets. Stacked breakdown charts renormalize each segment by the same divisor, so segments always sum to the total shown.

**Saturated datasets.** A dataset where most configs reach near-perfect recall (≥ 50 % of cells at R@10 ≥ 0.99) cannot discriminate between configurations. Such datasets:
- get **no per-dataset charts at all** — the generator logs `[skip] <dataset>: saturated dataset` and deletes any stale copies, so their meaningless numbers can never be screenshotted out of context,
- remain visible in cross-dataset charts, labeled `(saturated)` and drawn hatched/faded, with a red footnote,
- are **excluded from all global rankings**.

Never tune against a saturated dataset. The flag is data-driven: once the dataset is re-run with fixed ground truth, its charts come back automatically.

**Comparability rules.** Head-to-head rankings (strategy, embedding, memory type) use only *condition-comparable* cells:
- tuning-sweep variants are excluded — phase 3 sweeps only hybrid across BM25 weights, phase 4 sweeps deliberately degraded decay settings of one strategy; averaging either into a ranking would skew that strategy against the rest (phase-4 cells at default `decay=none` do count),
- legacy full-pool fast-sweep cells (labeled `memory_type: all`) are excluded — they retrieved from all memory types at once, an easier condition than per-store cells.
The sweep cells still power the charts they were generated for (the BM25-weight and decay sweeps).

**Sample sizes.** `n=…` captions appear on or under most charts. Treat `n < 3` as anecdote, not evidence; charts requiring a "winner" ignore strategies with `n < 3` when a better-sampled alternative exists.

---

## A. Cross-run report plots (`data/output/plots/`)

Six charts per dataset (prefix = dataset name) plus four cross-dataset charts.

**A missing chart is deliberate.** Charts that don't apply to the current data
are skipped and any stale copy from a previous run is deleted (the generator
logs `− removed stale …`): saturated datasets get no per-dataset charts at
all, `_02` needs ≥ 2 decay policies for the winning strategy, `_04` needs
hybrid weight-sweep cells, `_05` needs ≥ 2 memory types.
Every PNG present is therefore guaranteed to reflect the current master data.

### `{DS}_01_strategy_comparison.png` — Retrieval strategy: avg vs best Recall@10

**Shows.** One row per retrieval strategy, best (avg Recall@10) at the top. Three bars per row: a **pale wide bar** = best Recall@10 any config of that strategy reached (its tuned ceiling), a **solid colored bar** = average Recall@10 across all its comparable runs, an **orange bar** = average Precision@1 (is the top result correct?).

**How to read.** The solid bar answers "how good is this strategy typically"; the pale bar answers "how good can it get if tuned". A big gap between them means the strategy is tuning-sensitive. `best=…` annotates the overall winner.

**Watch out.** Averages come from condition-comparable cells only; a strategy measured only in sweeps may show fewer runs than the dataset's total cells.

### `{DS}_02_decay_sweep.png` — Decay policy impact

**Shows.** For the dataset's winning strategy: average Recall@10 (solid) and MRR (faded) per decay policy, best policy on top. The x-label lists `runs: policy(n=…)` per bucket.

**How to read.** Compare `none` against each decay policy — the gap is what forgetting costs you in recall. Zero-length bars are real: aggressive decay (high λ) can push every old memory below the archival floor, collapsing recall to 0.

**Watch out.** Buckets are usually unbalanced (`none` accumulates more runs); read the `n=` caption before trusting small gaps. Fast-sweep cells are excluded (they never vary decay).

### `{DS}_03_recall_curve.png` — Ranking quality (measured at k=1 and k=10)

**Shows.** For the top-4 strategies: exactly **two measured points** each — Precision@1 at k=1 and Recall@10 at k=10 — joined by a *dashed* line. Exact values are in the legend.

**How to read.** High P@1 with similar R@10 = the strategy ranks the right memory first (good for top-1 UX). Low P@1 but decent R@10 = the right memories are found but buried in the list.

**Watch out.** The dashed connector is a visual guide between two *different metrics* at measured cutoffs — it is **not** a sampled curve; nothing between k=1 and k=10 is measured, and nothing is interpolated.

### `{DS}_04_bm25_sweep.png` — BM25 weight sweep (hybrid strategy)

**Shows.** Average Recall@10 of the hybrid strategy as its BM25 weight goes 0 (pure semantic) → 1 (pure BM25), from the dedicated phase-3 sweep cells. The orange marker/annotation flags the optimal weight.

**How to read.** A flat curve = the BM25/semantic mix barely matters on this dataset (pick anything reasonable). A peaked curve = the mix is a real lever; use the annotated optimum. The x-label states the data source (`phase-3 weight sweep, n=… runs/point`).

**Watch out.** If it instead says *"mixed study phases — points may not be comparable"*, the dataset had no dedicated sweep and the curve mixes measurement conditions — treat shape cautiously.

### `{DS}_05_memory_type.png` — Memory type comparison

**Shows.** Per memory type (episodic / semantic / preference / `all`): average composite (solid) and average Recall@10 (faded), with `n=` in each row label. Composite formula in the footer.

**How to read.** Tells you which memory store carries the dataset — e.g. LongMemEval is answered almost entirely from episodic memory.

**Watch out.** `all` rows are legacy full-pool fast-sweep cells (retrieval across every store at once) — a different, easier condition; the footer says so. Don't read `all` vs `episodic` as a like-for-like comparison.

### `{DS}_06_best_config_card.png` — Best configuration recommendation

**Shows.** Left: the full winning configuration (strategy, embedding, backend, BM25 weight, decay + λ, memory type, and which study phase produced it) with its headline composite / Recall@10 / MRR. Right: the composite as a stacked bar — each segment is one weighted, renormalized metric contribution; segments sum exactly to the total in the panel title; exact per-segment values are in the legend.

**How to read.** This is the config to copy into production for this dataset. The stacked bar shows *why* it wins — recall-driven, precision-driven, etc. The footer prints the formula, the divisor used (`Σw_active`), and the gate state so the arithmetic is reproducible by hand.

**Watch out.** The winner is chosen by composite among per-store cells only (never a full-pool cell). `Study phase: phase4_…` means it was found during a sweep — a tuned config, not a default.

### `cross_01_strategy_bars.png` — Strategy comparison across datasets

**Shows.** Grouped horizontal bars: average Recall@10 per strategy (rows) per dataset (colors). Strategies ordered best→worst by their average over *non-saturated* datasets.

**How to read.** A strategy whose bars are consistently long across colors generalizes; one long bar with short siblings is dataset-specific. The best value per dataset is annotated.

**Watch out.** Hatched/faded bars = saturated dataset — visible for completeness, ignored by the ordering.

### `cross_02_decay_lines.png` — Decay policy comparison across datasets

**Shows.** One line per decay policy across datasets (x-axis), using each dataset's top strategy. Only the best policy per dataset is value-labeled.

**How to read.** The `none` line sitting above the rest = decay costs recall on that dataset. Converging lines = decay is a free choice there.

**Watch out.** The x-axis is categorical — the connecting lines aid tracking, they don't imply a trend between datasets.

### `cross_03_composite_breakdown.png` — Best-config composite breakdown per dataset

**Shows.** One stacked column per dataset: the best config's composite split into its four renormalized components; the total is printed above each column.

**How to read.** Compare *why* each dataset's winner scores what it scores — e.g. a column dominated by the blue (recall) segment wins on coverage, a large gold segment means temporal accuracy contributes.

**Watch out.** Red footnote + `(saturated)` label: that column's height reflects a broken benchmark, not retrieval quality.

### `cross_04_best_configs.png` — Best configuration table

**Shows.** One row per dataset: winning strategy, embedding, BM25 weight, decay, memory type, Recall@10, composite.

**How to read.** The copy-paste summary of the whole benchmark. Cross-check any row against the dataset's `_06` card.

**Watch out.** Rows marked `(saturated)` carry a red footnote — they are not meaningful recommendations.

---

## B. Per-study plots (inside each `study_*/` or `merged_*/` folder)

Generated per run by the study visualizer. Strategy-comparison charts here use the same comparability rules as section A.

### `phase1_bm25_baseline.png` — Phase 1: BM25 baseline
Recall/precision/MRR of the lexical baselines (bm25, bm25l, recency) per memory type. This is the floor every later phase must beat.

### `phase2_embedding_comparison.png` — Phase 2: embedding model comparison
Recall and latency per embedding model (phase-2 cells only). Read jointly: the best-recall model is not always worth its latency. Latency is only comparable within one platform.

### `phase3_hybrid_weight.png` — Phase 3: hybrid BM25-weight sweep
Recall@K, MRR, and composite vs BM25 weight, per memory type, from phase-3 sweep cells only. The red dashed line marks the best weight by composite. Flat lines mean the mix doesn't matter.

### `phase5_reranker_comparison.png` — Phase 5: reranker comparison
Recall/P@K/MRR per reranker versus the `none` baseline, with the score *lift vs none* annotated (red = the reranker **hurts**), plus its latency cost. A negative lift with +30–60 ms latency means the reranker is strictly worse — this chart exists to catch exactly that. Requires CUDA; on machines without it the run logs a loud skip and only the baseline appears.

### `phase4_decay_heatmap.png` — Phase 4: decay policy × λ heatmap
Recall (and MRR) as a grid of decay policy × λ. Read row-wise: where a row turns dark with rising λ is the "decay cliff" — memories fall below the archival floor and recall collapses.

### `phase6_per_dataset.png` — Per-dataset strategy & embedding comparison
Only in merged/multi-dataset runs: the same strategy/embedding compared per dataset, showing that one config does not fit all datasets.

### `phase7_leaderboard.png` — Top-10 configurations
Three panels for the 10 best cells by composite: composite bars, the renormalized component breakdown (segments sum to the composite), and P50/P90/P99 latency. The latency panel is the reality check — a top-composite config with a 10× P99 tail may not be shippable.

### `phase_progression.png` — Was each phase worth running?
Running-best recall and composite after each phase, with per-phase improvement deltas, plus the best config's latency per phase. A flat line after phase N means later phases only confirmed, not improved. All phase tags are tracked, including the fast-sweep and reranker phases.

### `ci_comparison.png` — Statistical comparison (95 % bootstrap CI)
Mean Recall@10 per strategy with bootstrap confidence intervals; ★ marks pairs whose CIs don't overlap (≈ p < 0.05). Below each bar: `correct/total` query counts. **N < 10 flags mean the CI is unreliable — treat unstarred or small-N differences as noise.** CI populations match the ranking populations exactly (same comparability rules).

### `noise_quality.png` — Quality beyond recall
Three panels: Recall per memory type per strategy; contamination (fraction of returned results that are irrelevant) at the two measured cutoffs K=1 and K=10 (dashed connector — nothing in between is measured); and P@1 / NDCG / F1 per strategy. Falling contamination from K=1 to K=10 means the noise concentrates at the top ranks.

### `efficiency.png` — Recall per millisecond & cost-vs-recall Pareto
Recall@10 ÷ P50 latency per strategy, and a cost-vs-recall scatter. Use it to pick the best strategy *within* a latency/cost budget rather than the absolute best.

### `resource_usage.png` — RAM / CPU / duration per strategy
Peak RAM, CPU, and wall time per strategy. Answers "can my hardware afford the best strategy?".

### `recall_k_variation.png` — Ranking quality at measured cutoffs
P@1 and R@10 per strategy (measured points only, dashed connectors) plus each strategy's Precision@10-vs-Recall@10 operating point. No intermediate K values are drawn — they are not measured.

### `parameter_sensitivity.png` — Which knob moves the metric
For each tunable parameter (λ, BM25 weight, pruning threshold…): metric response along its swept range. Flat panel = don't bother tuning that knob.

### `decay_curves.png` — Recall & MRR vs λ per decay policy
Continuous view of the λ sweep with the no-decay reference line. Complements the phase-5 heatmap.

### `composite_breakdown.png` / `study_report.png`
Composite variance analysis, and the single-page dashboard that combines the leaderboard, decay grid, and strategy summaries for sharing.

### `cross_dataset_heatmap.png` + `narrative_report.txt` (merged runs)
Strategy × dataset recall matrix (★ = best per dataset), and the plain-text cross-dataset story: winner per dataset with dataset-profile context. Datasets are named per row (never "(unknown)"); explanatory text is labeled as interpretation, not measurement.

---

## Honest-reading checklist

Before quoting any chart:

1. **Saturation** — hatching or a `(saturated)` label ⇒ the numbers do not measure retrieval quality; saturated datasets have no per-dataset charts at all.
2. **Sample size** — find the `n=` / `runs=` caption; `n < 3` is an anecdote, `N < 10` makes CIs unreliable (the CI chart flags this itself).
3. **Avg vs best** — pale/wide bars and "best" annotations are tuned ceilings; solid bars are typical behavior. Don't quote a ceiling as an average.
4. **Dashed lines** — dashed connectors join measured points of possibly different metrics; they are never sampled curves.
5. **Condition labels** — `memory_type: all`, "mixed study phases", and `Study phase: phase3/4_…` mark cells measured under non-default conditions.
6. **Leakage warnings** — the run log may report query↔corpus verbatim overlap for a dataset; high overlap inflates lexical strategies (bm25/bm25l) specifically.
