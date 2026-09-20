"""Study result aggregation: ranking, averaging, and statistical inference.

DATA MODEL
----------
Input: list[StudyRunResult]  — one per benchmark cell (strategy × model × dataset × seed)
Each StudyRunResult carries: recall_at_k, mrr, ndcg, precision_at_k,
  contamination_rate, latency_mean_ms, retrieval_strategy, embedding_model,
  embedding_backend, bm25_weight, reranker_model, platform, seed

AVERAGING MODE
--------------
All rank_by_*() methods use MACRO-AVERAGING: arithmetic mean of per-cell metric
values. Each cell (= one (strategy, seed, dataset) combination) contributes
equally. This matches standard IR evaluation practice.

STATISTICAL INFERENCE
---------------------
bootstrap_ci(): Non-parametric percentile bootstrap confidence intervals.
  Method:
    1. Collect per-cell metric values for each group (e.g. each strategy).
    2. Resample with replacement n_bootstrap=1000 times; compute mean each time.
    3. Sort bootstrap means; read off (alpha/2) and (1-alpha/2) percentiles.
  Index formula (0-based):
    lo_idx = floor(alpha/2 * n)          e.g. floor(0.025*1000) = 25
    hi_idx = ceil((1-alpha/2) * n) - 1  e.g. ceil(0.975*1000)-1 = 974
  The -1 converts the 1-indexed nearest-rank to a 0-based array subscript.
  Reference: Sakai 2006 "Evaluating Evaluation Metrics", SIGIR.
             Voorhees & Harman 2005 "TREC: Experiment and Evaluation in IR".
significance_table(): sig_vs_next=True when ci_low > next_ci_high (non-overlapping CIs).
  This is a conservative test; when scipy is installed, wilcoxon_pairwise() provides
  a paired Wilcoxon signed-rank test over per-cell values, which is more powerful
  for non-normal distributions and small N.

KEY OUTPUT PATHS
----------------
rank_by_retrieval_strategy() → accuracy_leaderboard in benchmark_results/leaderboards.json
rank_by_embedding_model()    → embedding_model_leaderboard in leaderboards.json
rank_by_reranker()           → reranker_leaderboard in leaderboards.json
bootstrap_ci()               → recall_ci_low / recall_ci_high in each leaderboard entry
study_summary()              → study_*_summary.json written by StudyReporter.write_all()

WIRED FROM
----------
study_runner.py  _run_single_dataset()  line ~1016
study_runner.py  _write_leaderboards_json()  line ~541
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmark.workload.aggregator import MatrixAggregator


class StudyAggregator(MatrixAggregator):
    """Aggregates StudyRunResult objects with study-specific rankings."""

    def __init__(self, results: list, dataset_name: str | None = None):
        super().__init__(results)
        # Keep reference with study-specific fields intact
        self._study_results = [r for r in results if r.success]
        # Fallback dataset name used when results lack a dataset_name attribute
        self._dataset_name = dataset_name
        # Stamp it onto rows that don't carry one, so the grid CSV records the
        # dataset per row and merged narratives can label datasets correctly.
        if dataset_name:
            import contextlib
            for r in results:
                if not getattr(r, "dataset_name", ""):
                    with contextlib.suppress(AttributeError, TypeError):
                        object.__setattr__(r, "dataset_name", dataset_name)

    @classmethod
    def from_csv(cls, path: Path, source_run_id: str = "") -> StudyAggregator:
        """Load a StudyAggregator from a previously written grid CSV file.

        Args:
            path: Path to a study_*_grid.csv file.
            source_run_id: Optional tag prepended to run_id for provenance tracking.
        """
        import csv as _csv

        from benchmark.workload.study_scheduler import StudyRunResult
        results = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    r = StudyRunResult.from_csv_row(row, source_run_id)
                    # Carry per-row dataset attribution through merges so
                    # multi-grid narratives don't label datasets "(unknown)".
                    ds = (row.get("dataset_name") or "").strip()
                    if ds:
                        object.__setattr__(r, "dataset_name", ds)
                    results.append(r)
                except Exception:
                    continue
        return cls(results)

    # ─── Strategy axis (override to exclude Phase 4 decay cells) ────────────

    # Strategies whose scores actually depend on the embedding model — shared
    # by rank_by_embedding_model() and the statistical helpers below.
    _EMBEDDING_STRATEGIES = frozenset({"semantic", "hybrid", "colbert", "adaptive"})

    # Dimensions ranked head-to-head from equal-footing cells. CI/significance
    # populations must match the corresponding rank_by_* population.
    _HEAD_TO_HEAD_GROUPS = frozenset({"retrieval_strategy", "memory_type", "embedding_model"})

    def _population_for(self, group_by: str) -> list:
        """Rows to use when grouping by `group_by` for CIs/significance.

        Must select the same cells as the corresponding rank_by_* method —
        otherwise a leaderboard publishes a mean from one population with a
        confidence interval (and sig_vs_next claim) from another."""
        rows = self._study_results
        if group_by in self._HEAD_TO_HEAD_GROUPS:
            rows = [r for r in rows if not self._is_sweep_variant(r)]
        if group_by == "embedding_model":
            rows = [r for r in rows
                    if getattr(r, "retrieval_strategy", "") in self._EMBEDDING_STRATEGIES
                    and getattr(r, "embedding_model", "") not in ("", "none")]
        return rows

    @staticmethod
    def _is_sweep_variant(r) -> bool:
        """Cells from tuning sweeps that must not enter head-to-head rankings.

        Phase 3 sweeps ONE strategy (hybrid) across the BM25-weight grid and
        Phase 4 sweeps mostly-degraded decay settings of the winning strategy;
        averaging either into a ranking skews that strategy against the rest.
        Phase 4 cells at default decay ('none') stay — they are a normal
        measurement of the tuned config. This is the same rule as
        generate_reports._is_ranking_comparable, so the study-time report and
        the offline report rank from the same population.
        """
        phase = getattr(r, "study_phase", "general") or "general"
        if phase in ("phase3_hybrid_broad", "phase3_hybrid_fine", "phase3_hybrid_weight"):
            return True
        return phase.startswith("phase4") and getattr(r, "decay_policy", "none") != "none"

    def rank_by_retrieval_strategy(self) -> list[dict]:
        """Rank strategies by recall, excluding tuning-sweep cells.

        See _is_sweep_variant — strategy quality is judged from cells where
        strategies compete on equal footing, not from one strategy's sweep
        variants. Same contamination bug class as rank_by_bm25_weight().
        """
        strategy_cells = [
            r for r in self._study_results
            if not self._is_sweep_variant(r)
        ]
        if not strategy_cells:
            # Fallback: if all cells are decay-phase (shouldn't happen) use all
            return super().rank_by_retrieval_strategy()
        # Temporarily swap _successful so the parent method sees only strategy cells.
        # try/finally guarantees restoration even if super() raises (e.g. empty group).
        original = self._successful
        try:
            self._successful = [r for r in strategy_cells if r.success]
            return super().rank_by_retrieval_strategy()
        finally:
            self._successful = original

    # ─── Per-memory-type breakdown ───────────────────────────────────────────

    def rank_by_memory_type(self) -> list[dict]:
        """Rank memory types by recall, excluding tuning-sweep cells.

        Same contamination concern as rank_by_retrieval_strategy — sweep
        cells vary weight/decay of one strategy, not memory type.
        """
        strategy_cells = [
            r for r in self._study_results
            if not self._is_sweep_variant(r)
        ]
        original = self._successful
        try:
            self._successful = [r for r in strategy_cells if r.success] if strategy_cells else original
            return super().rank_by_memory_type()
        finally:
            self._successful = original

    # ─── Embedding model axis ────────────────────────────────────────────────

    def rank_by_embedding_model(self) -> list[dict]:
        """Rank (embedding_model, embedding_backend) pairs by avg recall.

        Groups by the (model, backend) tuple so sentence-transformers bge-m3
        and Ollama bge-m3 appear as separate rows — they may produce different
        recall due to quantization or different pooling.

        Latency is reported per-platform (darwin/win32/linux) separately;
        averaging MPS and CUDA latencies produces a meaningless number.

        Averaging mode: macro-average (each benchmark cell contributes equally).

        Returns:
            List of dicts sorted by avg_recall descending. Each dict contains:
              embedding_model      (str)   — model name
              embedding_backend    (str)   — e.g. "sentence-transformers", "ollama"
              avg_recall           (float) — macro-avg recall@k across cells
              avg_precision        (float) — macro-avg precision@k
              avg_f1               (float) — harmonic mean of avg_recall and avg_precision
              avg_mrr              (float) — macro-avg MRR
              avg_ndcg             (float) — macro-avg NDCG
              avg_noise            (float) — macro-avg contamination_rate
              avg_latency_ms       (float) — mean latency when only one platform present,
                                            else 0.0 (use latency_by_platform instead)
              latency_by_platform  (dict)  — {platform_str: mean_latency_ms}
              recall_per_ms        (float) — avg_recall / avg_latency_ms efficiency ratio
              retrieval_strategies (list)  — sorted list of strategies seen for this model
              runs                 (int)   — number of benchmark cells
              platforms            (list)  — sorted list of platforms seen
        """
        Key = tuple  # (model, backend)
        by_key: dict[Key, list] = defaultdict(list)
        by_lat_platform: dict[Key, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        by_noise: dict[Key, list] = defaultdict(list)
        by_mrr: dict[Key, list] = defaultdict(list)
        by_ndcg: dict[Key, list] = defaultdict(list)
        by_prec: dict[Key, list] = defaultdict(list)
        by_strat: dict[Key, set] = defaultdict(set)

        # recency/bm25/bm25l/llm_rerank cells may carry an embedding tag from
        # the sweep config but never consult it, and sweep cells vary the
        # weight/decay of one already-chosen embedding — either would corrupt
        # the ranking (and, via best_embedding_model(), steer later phases
        # toward the wrong model). _population_for applies both exclusions.
        for r in self._population_for("embedding_model"):
            m = getattr(r, "embedding_model", None)
            b = getattr(r, "embedding_backend", "") or "sentence-transformers"
            if not m or m in ("none", ""):
                continue
            key = (m, b)
            platform = getattr(r, "platform", "unknown") or "unknown"
            by_key[key].append(r.recall_at_k)
            by_lat_platform[key][platform].append(r.latency_mean_ms)
            by_noise[key].append(r.contamination_rate)
            by_mrr[key].append(r.mrr)
            by_ndcg[key].append(r.ndcg)
            by_prec[key].append(r.precision_at_k)
            by_strat[key].add(r.retrieval_strategy)

        rows = []
        for (model, backend), recalls in by_key.items():
            avg_recall = statistics.mean(recalls)
            avg_prec = statistics.mean(by_prec[(model, backend)]) if by_prec[(model, backend)] else 0.0
            f1 = (2 * avg_recall * avg_prec / (avg_recall + avg_prec)) if (avg_recall + avg_prec) > 0 else 0.0

            latency_by_platform = {
                plat: round(statistics.mean(lats), 1)
                for plat, lats in by_lat_platform[(model, backend)].items()
            }
            platforms = list(latency_by_platform.keys())
            avg_lat = latency_by_platform[platforms[0]] if len(platforms) == 1 else 0.0

            rows.append({
                "embedding_model": model,
                "embedding_backend": backend,
                "avg_recall": round(avg_recall, 4),
                "avg_precision": round(avg_prec, 4),
                "avg_f1": round(f1, 4),
                "avg_mrr": round(statistics.mean(by_mrr[(model, backend)]), 4) if by_mrr[(model, backend)] else 0.0,
                "avg_ndcg": round(statistics.mean(by_ndcg[(model, backend)]), 4) if by_ndcg[(model, backend)] else 0.0,
                "avg_noise": round(statistics.mean(by_noise[(model, backend)]), 4) if by_noise[(model, backend)] else 0.0,
                "avg_latency_ms": avg_lat,
                "latency_by_platform": latency_by_platform,
                "recall_per_ms": round(avg_recall / max(avg_lat, 0.001), 6) if avg_lat else 0.0,
                "retrieval_strategies": sorted(by_strat[(model, backend)]),
                "runs": len(recalls),
                "platforms": sorted(platforms),
            })

        return sorted(rows, key=lambda x: x["avg_recall"], reverse=True)

    # ─── BM25 weight axis ────────────────────────────────────────────────────

    def rank_by_bm25_weight(self) -> list[dict]:
        """Show recall/MRR by BM25 weight in hybrid configurations.

        Only considers results where retrieval_strategy == "hybrid".
        Returns an empty list when no hybrid results are present.

        Averaging mode: macro-average (each benchmark cell contributes equally).

        Returns:
            List of dicts sorted by avg_recall descending. Each dict contains:
              bm25_weight     (float) — BM25 interpolation weight (0.0–1.0)
              semantic_weight (float) — 1.0 - bm25_weight
              avg_recall      (float) — macro-avg recall@k across hybrid cells
              avg_mrr         (float) — macro-avg MRR
              avg_noise       (float) — macro-avg contamination_rate
              runs            (int)   — number of hybrid benchmark cells
        """
        # Only Phase 3 cells measure the BM25 weight tradeoff in isolation.
        # Phase 4 (decay sweep) also uses hybrid at bm25_weight=0.35 (the Phase 3 winner),
        # but those cells vary decay policy — averaging them into the weight ranking
        # would corrupt the recommendation by pulling the winner's recall down.
        phase3_tags = {"phase3_hybrid_broad", "phase3_hybrid_fine", "phase3_hybrid_weight"}
        hybrid = [
            r for r in self._study_results
            if r.retrieval_strategy == "hybrid"
            and getattr(r, "study_phase", "general") in phase3_tags
        ]
        if not hybrid:
            # Fallback: include all hybrid cells when no Phase 3 label is present
            hybrid = [r for r in self._study_results if r.retrieval_strategy == "hybrid"]
        if not hybrid:
            return []

        by_weight: dict[float, list] = defaultdict(list)
        by_weight_mrr: dict[float, list] = defaultdict(list)
        by_weight_noise: dict[float, list] = defaultdict(list)

        for r in hybrid:
            w = getattr(r, "bm25_weight", None)
            if w is None:
                continue
            by_weight[w].append(r.recall_at_k)
            by_weight_mrr[w].append(r.mrr)
            by_weight_noise[w].append(r.contamination_rate)

        rows = []
        for w, recalls in by_weight.items():
            rows.append({
                "bm25_weight": w,
                "semantic_weight": round(1.0 - w, 2),
                "avg_recall": round(statistics.mean(recalls), 4),
                "avg_mrr": round(statistics.mean(by_weight_mrr[w]), 4),
                "avg_noise": round(statistics.mean(by_weight_noise[w]), 4),
                "runs": len(recalls),
            })

        return sorted(rows, key=lambda x: x["avg_recall"], reverse=True)

    # ─── Reranker axis ───────────────────────────────────────────────────────

    def rank_by_reranker(self) -> list[dict]:
        """Rank reranker models by avg recall, show improvement over no-reranking.

        Lift formula:
          recall_lift_vs_none = avg_recall(reranker) - baseline_recall_none
          mrr_lift_vs_none    = avg_mrr(reranker)    - baseline_mrr_none

        where baseline_recall_none / baseline_mrr_none are the macro-averaged
        recall@k / MRR across all cells whose reranker_model == "none".
        A positive lift means the reranker improved over the no-reranker baseline.
        Note: no division guard is applied to baseline values — applying one to a
        subtraction operand would silently corrupt lift when the baseline is 0.0.

        Averaging mode: macro-average (each benchmark cell contributes equally).

        Returns:
            List of dicts sorted by avg_recall descending. Each dict contains:
              reranker_model        (str)   — model name, "none" for no reranker
              avg_recall            (float) — macro-avg recall@k across cells
              avg_mrr               (float) — macro-avg MRR
              avg_precision         (float) — macro-avg precision@k
              avg_latency_ms        (float) — macro-avg latency_mean_ms
              recall_lift_vs_none   (float) — avg_recall minus baseline "none" recall
              mrr_lift_vs_none      (float) — avg_mrr minus baseline "none" MRR
              runs                  (int)   — number of benchmark cells
        """
        by_rr: dict[str, list] = defaultdict(list)
        by_rr_mrr: dict[str, list] = defaultdict(list)
        by_rr_prec: dict[str, list] = defaultdict(list)
        by_rr_lat: dict[str, list] = defaultdict(list)

        for r in self._study_results:
            rr = getattr(r, "reranker_model", "none")
            by_rr[rr].append(r.recall_at_k)
            by_rr_mrr[rr].append(r.mrr)
            by_rr_prec[rr].append(r.precision_at_k)
            by_rr_lat[rr].append(r.latency_mean_ms)

        # No division guard needed — baselines are used only in subtraction, not
        # division. Applying `or 0.001` to a subtraction operand silently corrupts
        # lift values whenever the 'none' baseline has recall=0.0.
        baseline_recall = statistics.mean(by_rr.get("none", [0.0]))
        baseline_mrr    = statistics.mean(by_rr_mrr.get("none", [0.0]))

        rows = []
        for rr, recalls in by_rr.items():
            avg_r = statistics.mean(recalls)
            avg_mrr = statistics.mean(by_rr_mrr[rr]) if by_rr_mrr[rr] else 0.0
            rows.append({
                "reranker_model": rr,
                "avg_recall": round(avg_r, 4),
                "avg_mrr": round(avg_mrr, 4),
                "avg_precision": round(statistics.mean(by_rr_prec[rr]), 4) if by_rr_prec[rr] else 0.0,
                "avg_latency_ms": round(statistics.mean(by_rr_lat[rr]), 3) if by_rr_lat[rr] else 0.0,
                "recall_lift_vs_none": round(avg_r - baseline_recall, 4),
                "mrr_lift_vs_none": round(avg_mrr - baseline_mrr, 4),
                "runs": len(recalls),
            })

        return sorted(rows, key=lambda x: x["avg_recall"], reverse=True)

    # ─── Statistical significance (bootstrap CI) ────────────────────────────

    def bootstrap_ci(
        self,
        metric: str = "recall_at_k",
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 0,
        group_by: str = "retrieval_strategy",
    ) -> dict[str, dict]:
        """Compute bootstrap confidence intervals for a metric grouped by a dimension.

        Method: non-parametric percentile bootstrap.
        1. For each group (e.g. each strategy), collect the per-cell metric values.
        2. Resample with replacement n_bootstrap times, each time computing the mean.
        3. Sort the bootstrap means array; read off the (alpha/2) and (1-alpha/2)
           percentile positions using the following 0-based index arithmetic:

             alpha    = 1.0 - ci_level          # e.g. 0.05 for 95% CI
             lo_idx   = floor(alpha/2 * n)       # e.g. floor(0.025 * 1000) = 25
             hi_idx   = ceil((1-alpha/2) * n) - 1  # e.g. ceil(0.975 * 1000) - 1 = 974

           The -1 on hi_idx is required because ceil() returns a 1-indexed nearest-rank
           (the smallest integer >= the fractional position), which must be decremented
           by 1 to become a valid 0-based array subscript. Without the -1, hi_idx would
           be 975, making the CI one position wider than the requested level.

        This is the standard IR evaluation approach (Sakai 2006, Voorhees 2001).
        A 95% CI that does not overlap between two strategies indicates a
        statistically significant performance difference.

        Reference: Sakai 2006 "Evaluating Evaluation Metrics", SIGIR.
                   Voorhees & Harman 2005 "TREC: Experiment and Evaluation in IR".

        Args:
            metric: The metric field name on StudyRunResult (e.g. "recall_at_k").
            n_bootstrap: Number of bootstrap iterations. 1000 is standard.
            ci_level: Confidence level (0.95 = 95% CI).
            seed: Random seed for reproducibility.
            group_by: Field to group by (e.g. "retrieval_strategy", "embedding_model").

        Returns:
            Dict mapping group_value → {mean, ci_low, ci_high, std, n, ci_level}.
        """
        alpha = 1.0 - ci_level

        by_group: dict[str, list[float]] = defaultdict(list)
        for r in self._population_for(group_by):
            group_val = str(getattr(r, group_by, "unknown"))
            val = getattr(r, metric, None)
            if val is not None:
                by_group[group_val].append(float(val))

        results = {}
        rng_np = np.random.default_rng(seed)
        for group_val, values in by_group.items():
            n = len(values)
            if n == 0:
                continue
            observed_mean = sum(values) / n
            observed_std = statistics.stdev(values) if n > 1 else 0.0

            # Vectorized bootstrap resampling — 1 call instead of n_bootstrap × n Python RNG calls
            vals_arr = np.array(values, dtype=np.float64)
            samples = rng_np.choice(vals_arr, size=(n_bootstrap, n), replace=True)
            boot_means_arr = np.sort(samples.mean(axis=1))

            # Both indices are 0-based. ceil() gives a 1-indexed rank — subtract 1
            # to convert to a 0-based array subscript; otherwise ci_high is one
            # position too large, making the CI slightly wider than requested.
            lo_idx = max(0, math.floor(alpha / 2 * n_bootstrap))
            hi_idx = min(n_bootstrap - 1, math.ceil((1 - alpha / 2) * n_bootstrap) - 1)

            results[group_val] = {
                "mean":    round(observed_mean, 4),
                "ci_low":  round(float(boot_means_arr[lo_idx]), 4),
                "ci_high": round(float(boot_means_arr[hi_idx]), 4),
                "std":     round(observed_std, 4),
                "n":       n,
                "ci_level": ci_level,
            }

        return results

    def significance_table(
        self,
        metric: str = "recall_at_k",
        n_bootstrap: int = 1000,
        group_by: str = "retrieval_strategy",
    ) -> list[dict]:
        """Return a table of groups sorted by mean, with CI and overlap flags.

        Two groups are marked 'significantly different' when their 95% CIs
        do not overlap — the standard threshold for IR evaluation papers.

        Returns list of dicts sorted by mean descending, each with:
          group, mean, ci_low, ci_high, std, n,
          sig_vs_prev (True if CI does not overlap with the next-lower group).
        """
        ci = self.bootstrap_ci(metric=metric, n_bootstrap=n_bootstrap, group_by=group_by)
        rows = sorted(ci.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        out = []
        for i, (group, stats) in enumerate(rows):
            sig = False
            n = stats.get("n", 0)
            # Non-overlapping CIs are only meaningful with enough observations.
            # N < 10 gives CIs that span most of [0, 1] — flag them instead of
            # marking significance (which would be a false positive).
            if i + 1 < len(rows) and n >= 10:
                _, next_stats = rows[i + 1]
                if next_stats.get("n", 0) >= 10:
                    sig = stats["ci_low"] > next_stats["ci_high"]
            out.append({
                "group": group,
                **stats,
                "sig_vs_next": sig,
                "ci_reliable": n >= 10,  # callers can display a warning when False
            })
        return out

    def wilcoxon_pairwise(
        self,
        metric: str = "recall_at_k",
        group_by: str = "retrieval_strategy",
        alpha: float = 0.05,
    ) -> list[dict]:
        """Paired Wilcoxon signed-rank test between every pair of groups.

        More powerful than non-overlapping CI test for non-normal distributions
        and small N (< 30 observations per group). Requires scipy.

        Uses paired observations: for each (group_a, group_b) pair, we align
        cells by memory_type and dataset so the test measures within-cell
        differences rather than overall level shifts. Falls back to unpaired
        Mann-Whitney U when cell-level pairing is not possible.

        Returns list of dicts sorted by p-value ascending:
          group_a, group_b, statistic, p_value, significant (p < alpha),
          mean_a, mean_b, mean_diff, effect_size (r = Z / sqrt(N)).

        Requires: pip install 'memtuner[stats]'  (scipy >= 1.11)
        """
        try:
            from scipy import stats as _sp_stats
        except ImportError:
            return [{"error": "scipy not installed — run: pip install 'memtuner[stats]'"}]

        by_group: dict[str, list[float]] = defaultdict(list)
        # Build key → value mapping for pairing: (memory_type, dataset, seed) → value
        pair_keys: dict[str, dict[str, float]] = defaultdict(dict)

        for r in self._population_for(group_by):
            g = str(getattr(r, group_by, "unknown"))
            val = float(getattr(r, metric, 0.0))
            by_group[g].append(val)
            # Pairing key: same memory type + dataset within a run
            _ds = getattr(r, "dataset_name", "") or ""
            _mt = getattr(r, "memory_type", "") or ""
            _seed = str(getattr(r, "seed", "0"))
            pair_key = f"{_ds}:{_mt}:{_seed}"
            pair_keys[g][pair_key] = val

        groups = sorted(by_group.keys())
        results = []
        for i, ga in enumerate(groups):
            for gb in groups[i + 1:]:
                # Attempt paired test using aligned keys
                common_keys = set(pair_keys[ga]) & set(pair_keys[gb])
                if len(common_keys) >= 5:
                    a_vals = [pair_keys[ga][k] for k in sorted(common_keys)]
                    b_vals = [pair_keys[gb][k] for k in sorted(common_keys)]
                    diffs = [a - b for a, b in zip(a_vals, b_vals)]
                    # Wilcoxon signed-rank on differences
                    if len(set(diffs)) > 1:  # skip if all differences are zero
                        stat, p = _sp_stats.wilcoxon(diffs, alternative="two-sided")
                        n = len(diffs)
                        z = _sp_stats.norm.ppf(1 - p / 2)  # approximate Z from p
                        effect_r = abs(z) / math.sqrt(n)
                        test_type = "wilcoxon_paired"
                    else:
                        stat, p, effect_r, test_type = 0.0, 1.0, 0.0, "no_variance"
                else:
                    # Unpaired: Mann-Whitney U
                    a_vals = by_group[ga]
                    b_vals = by_group[gb]
                    if len(a_vals) >= 3 and len(b_vals) >= 3:
                        stat, p = _sp_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                        n = len(a_vals) + len(b_vals)
                        z = _sp_stats.norm.ppf(1 - p / 2)
                        effect_r = abs(z) / math.sqrt(n)
                        test_type = "mannwhitney_unpaired"
                    else:
                        stat, p, effect_r, test_type = 0.0, 1.0, 0.0, "insufficient_n"

                mean_a = sum(by_group[ga]) / len(by_group[ga]) if by_group[ga] else 0.0
                mean_b = sum(by_group[gb]) / len(by_group[gb]) if by_group[gb] else 0.0
                results.append({
                    "group_a": ga,
                    "group_b": gb,
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p), 6),
                    "significant": bool(p < alpha),
                    "alpha": alpha,
                    "mean_a": round(mean_a, 4),
                    "mean_b": round(mean_b, 4),
                    "mean_diff": round(mean_a - mean_b, 4),
                    "effect_size_r": round(effect_r, 4),
                    "n_paired": len(common_keys) if len(common_keys) >= 5 else 0,
                    "test_type": test_type,
                })

        ordered = sorted(results, key=lambda x: x["p_value"])
        running_adjusted = 0.0
        total_comparisons = len(ordered)
        for index, row in enumerate(ordered):
            holm_adjusted = min(1.0, (total_comparisons - index) * row["p_value"])
            running_adjusted = max(running_adjusted, holm_adjusted)
            row["p_value_holm"] = round(running_adjusted, 6)
            row["significant_holm"] = bool(running_adjusted < alpha)
        return ordered

    # ─── Phase summary ───────────────────────────────────────────────────────

    def best_per_phase(self) -> dict[str, dict]:
        """Return the single best result (by composite) per study phase."""
        by_phase: dict[str, list] = defaultdict(list)
        for r in self._study_results:
            phase = getattr(r, "study_phase", "general")
            by_phase[phase].append(r)

        out = {}
        for phase, items in by_phase.items():
            best = max(items, key=lambda r: r.composite_score())
            out[phase] = best.to_dict()
            out[phase]["_composite"] = round(best.composite_score(), 4)
            # Add study fields that to_dict() may not include
            out[phase]["embedding_model"] = getattr(best, "embedding_model", None)
            out[phase]["bm25_weight"] = getattr(best, "bm25_weight", None)
            out[phase]["reranker_model"] = getattr(best, "reranker_model", "none")
        return out

    # ─── Seeds for next phases ───────────────────────────────────────────────

    def best_embedding_model(self) -> str | None:
        """Return the best embedding model name by avg recall."""
        ranked = self.rank_by_embedding_model()
        return ranked[0]["embedding_model"] if ranked else None

    def best_embedding_backend(self) -> str | None:
        """Return the backend for the best embedding model (must match best_embedding_model)."""
        ranked = self.rank_by_embedding_model()
        return ranked[0]["embedding_backend"] if ranked else None

    def best_bm25_weight(self) -> float | None:
        """Return the best BM25 weight for hybrid, or None if Phase 3 was not run."""
        ranked = self.rank_by_bm25_weight()
        return ranked[0]["bm25_weight"] if ranked else None

    def best_reranker(self) -> str | None:
        """Return the best reranker, or None if Phase 5 was not run."""
        ranked = self.rank_by_reranker()
        return ranked[0]["reranker_model"] if ranked else None

    # ─── Per-dataset breakdown ───────────────────────────────────────────────

    def rank_by_dataset(self) -> dict[str, dict]:
        """Return per-dataset best strategy and best embedding model.

        Different datasets have very different characteristics:
        - LoCoMo: long episodic conversations — BM25 is surprisingly strong
          because queries often use exact keywords from conversations.
        - LongMemEval: temporal reasoning tasks — semantic models help where
          keywords alone are insufficient.
        - SQuAD: reading comprehension on dense paragraphs — BM25 over exact
          paragraph text tends to dominate.
        - Synthetic: fully controlled — useful for checking configuration sanity.

        Returns a dict keyed by dataset name, each with its own rankings and
        recommendations independent of the other datasets.
        """
        by_dataset: dict[str, list] = defaultdict(list)
        for r in self._study_results:
            # dataset name comes from the run_id prefix added by from_csv()
            # or from the cell's workload_profile as a proxy
            ds_name = getattr(r, "dataset_name", None)
            if not ds_name:
                # Infer from run_id prefix (set by _run_single_dataset source_run_id)
                if ":" in r.run_id:
                    ds_name = r.run_id.split(":")[0]
                else:
                    ds_name = self._dataset_name or "unknown"
            by_dataset[ds_name].append(r)

        out = {}
        for ds_name, results in by_dataset.items():
            sub_agg = StudyAggregator(results)
            strat_rank = sub_agg.rank_by_retrieval_strategy()
            embed_rank = sub_agg.rank_by_embedding_model()
            out[ds_name] = {
                "n_cells": len(results),
                "best_strategy": strat_rank[0]["retrieval_strategy"] if strat_rank else None,
                "best_strategy_recall": strat_rank[0]["avg_recall"] if strat_rank else 0.0,
                "best_embedding": embed_rank[0]["embedding_model"] if embed_rank else None,
                "best_embedding_recall": embed_rank[0]["avg_recall"] if embed_rank else 0.0,
                "strategy_ranking": strat_rank[:4],
                "embedding_ranking": embed_rank[:4],
            }
        return out

    # ─── Full summary ────────────────────────────────────────────────────────

    def study_summary(self) -> dict:
        base = self.summary()
        base["evaluation_protocol"] = {
            "test_holdout_fractions": sorted({
                getattr(result, "test_holdout_fraction", 0.0)
                for result in self._study_results
            }),
            "warning": (
                "A 0.0 holdout fraction is exploratory only and does not provide "
                "an out-of-sample estimate."
            ),
        }
        base["recommendation_scope"] = (
            "Recommendations apply only to the evaluated datasets, workload profile, "
            "hardware, retrieval metric weights, and evaluation protocol. They are "
            "comparative evidence, not universal production prescriptions."
        )
        base["paired_significance"] = {
            "unit": "benchmark cell",
            "method": "paired Wilcoxon signed-rank where cell pairing is available; Mann-Whitney U otherwise",
            "multiple_comparison_correction": "Holm",
            "comparisons": self.wilcoxon_pairwise(),
        }
        # Compute once, reuse below to avoid a second O(N) pass over _study_results.
        _em_ranked = self.rank_by_embedding_model()
        base["embedding_model_ranking"] = _em_ranked
        base["bm25_weight_ranking"] = self.rank_by_bm25_weight()
        base["reranker_ranking"] = self.rank_by_reranker()
        base["best_per_phase"] = self.best_per_phase()
        base["per_dataset"] = self.rank_by_dataset()
        # Reuse the already-computed ranking from base.summary() — avoids a second
        # O(N) pass over all results (rank_by_retrieval_strategy is called there already).
        _strat_ranks = base.get("retrieval_strategy_ranking", [])

        # top_ranked: each axis is ranked independently from its own phase data.
        # embedding_model and embedding_backend are always from the same row —
        # they are never mixed from separate rankings.
        _top_em_row = _em_ranked[0] if _em_ranked else {}
        base["top_ranked"] = {
            "top_retrieval_strategy": _strat_ranks[0]["retrieval_strategy"] if _strat_ranks else None,
            "top_embedding_model":    _top_em_row.get("embedding_model"),
            "top_embedding_backend":  _top_em_row.get("embedding_backend"),
            "top_bm25_weight":        self.best_bm25_weight(),
            "top_reranker":           self.best_reranker(),
            "note": (
                "Each axis is ranked independently from its own phase data. "
                "embedding_model and embedding_backend are always from the same row. "
                "This is a measurement, not a prescription. "
                "Per-dataset breakdowns in per_dataset may show different patterns."
            ),
        }
        # "recommendations" alias for backward compatibility
        base["recommendations"] = base["top_ranked"]
        # also expose the named keys the study_runner.py seed logic reads
        base["recommendations"]["best_retrieval_strategy"] = base["top_ranked"]["top_retrieval_strategy"]
        base["recommendations"]["best_embedding_model"] = base["top_ranked"]["top_embedding_model"]
        base["recommendations"]["best_embedding_backend"] = base["top_ranked"]["top_embedding_backend"]
        base["recommendations"]["best_bm25_weight"] = base["top_ranked"]["top_bm25_weight"]
        return base


class StudyReporter:
    """Writes study results to CSV, JSON, and text summary."""

    def __init__(self, output_dir: Path):
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        agg: StudyAggregator,
        run_id: str,
        skip_plots: bool = False,
        all_results: list | None = None,
        run_metadata: dict | None = None,
    ) -> dict[str, str]:
        """Write all report files and return a dict of {label: absolute_path}.

        Args:
            agg: Aggregator with results.
            run_id: Unique identifier for this run.
            skip_plots: If True, skip matplotlib PNG generation.
            all_results: Raw result list for the visualizer (needed for phase PNGs).
            run_metadata: Dataset and evaluation-integrity details for this run.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {}
        paths["summary_json"] = self._write_json(agg, run_id, ts, run_metadata)
        paths["grid_csv"] = self._write_csv(agg, run_id, ts)
        paths["text_report"] = self._write_text(agg, run_id, ts, run_metadata)

        if not skip_plots:
            try:
                from benchmark.reporting.study_visualizer import StudyVisualizer
                results_for_viz = all_results if all_results is not None else agg._results
                viz = StudyVisualizer(results_for_viz, self._out, run_metadata=run_metadata)
                viz_paths = viz.generate_all()
                for name, p in (viz_paths or {}).items():
                    if p:
                        paths[name] = str(p)
            except ImportError:
                pass  # matplotlib not installed
            except Exception as e:
                paths["viz_error"] = str(e)

        return paths

    def _write_json(
        self, agg: StudyAggregator, run_id: str, ts: str, run_metadata: dict | None
    ) -> str:
        path = self._out / f"study_{ts}_{run_id}_summary.json"
        summary = agg.study_summary()
        metadata = dict(run_metadata or {})
        metadata["failed_cells"] = [
            {
                "cell_id": result.cell_id,
                "error": (result.error_message or "unknown error").splitlines()[0],
            }
            for result in agg._failed
        ]
        summary["run_metadata"] = metadata
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return str(path)

    def _write_csv(self, agg: StudyAggregator, run_id: str, ts: str) -> str:
        path = self._out / f"study_{ts}_{run_id}_grid.csv"
        # build_grid_table() now includes all study columns directly via getattr —
        # no second pass needed. Writing it directly prevents column duplication.
        rows = agg.build_grid_table()
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return str(path)

    def _write_text(
        self, agg: StudyAggregator, run_id: str, ts: str, run_metadata: dict | None
    ) -> str:
        path = self._out / f"study_{ts}_{run_id}_report.txt"
        lines = self._format_report(agg, run_id, run_metadata)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(path)

    def _format_report(
        self, agg: StudyAggregator, run_id: str, run_metadata: dict | None = None
    ) -> list[str]:
        sep = "=" * 76
        metadata = run_metadata or {}
        leakage = metadata.get("leakage", {})
        final_test = metadata.get("final_test", {})
        failed_cells = [
            {
                "cell_id": result.cell_id,
                "error": (result.error_message or "unknown error").splitlines()[0],
            }
            for result in agg._failed
        ]
        lines = [
            sep,
            "MEMTUNER — COMPREHENSIVE STUDY REPORT",
            f"Run ID:       {run_id}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Cells:        {agg.success_count}/{agg.total} successful",
            f"Holdout:      {', '.join(f'{fraction:.0%}' for fraction in agg.study_summary().get('evaluation_protocol', {}).get('test_holdout_fractions', [])) or 'not recorded'}",
            sep,
        ]
        lines += [
            "",
            "RECOMMENDATION SCOPE",
            "  Rankings are comparative evidence for this dataset, workload, hardware,",
            "  metric weights, and evaluation protocol; they are not universal prescriptions.",
        ]
        if metadata or failed_cells:
            lines += ["", "EVALUATION INTEGRITY"]
            if metadata.get("dataset_path"):
                lines.append(f"  Dataset:     {metadata['dataset_path']}")
            if metadata.get("dataset_fingerprint"):
                lines.append(f"  Fingerprint: {metadata['dataset_fingerprint']}")
            if leakage:
                lines.append(
                    f"  Leakage:     {leakage.get('status', 'unknown')} "
                    f"({leakage.get('leaked_queries', 0)}/{leakage.get('total_queries', 0)} queries; "
                    f"policy={leakage.get('policy', 'unknown')})"
                )
            if final_test:
                lines.append(
                    f"  Final test:  {final_test.get('status', 'unknown')} "
                    f"(holdout={final_test.get('holdout_fraction', 0.0):.0%})"
                )
                if final_test.get("status") == "completed":
                    lines.append(
                        f"    Recall@K={final_test.get('recall_at_k', 0.0):.4f}  "
                        f"MRR={final_test.get('mrr', 0.0):.4f}  "
                        f"P50={final_test.get('latency_p50_ms', 0.0):.1f}ms"
                    )
                elif final_test.get("reason"):
                    lines.append(f"    Reason: {final_test['reason']}")
            if failed_cells:
                lines.append(f"  Failed cells: {len(failed_cells)}")
                for cell in failed_cells[:5]:
                    lines.append(f"    {cell['cell_id']}: {cell['error']}")
                if len(failed_cells) > 5:
                    lines.append(f"    ... {len(failed_cells) - 5} additional failures")

        best = agg.best_overall()
        if best:
            best_prec = best.precision_at_k
            best_f1 = (2 * best.recall_at_k * best_prec / (best.recall_at_k + best_prec)) if (best.recall_at_k + best_prec) > 0 else 0.0
            lines += [
                "",
                "BEST OVERALL CONFIGURATION  (ranked by composite score)",
                f"  Memory:     {best.memory_type}",
                f"  Strategy:   {best.retrieval_strategy}",
                f"  Embedding:  {getattr(best, 'embedding_model', '—')}",
                f"  BM25 wt:    {getattr(best, 'bm25_weight', '—')}",
                f"  Reranker:   {getattr(best, 'reranker_model', '—')}",
                f"  Decay:      {best.decay_policy} (λ={best.lambda_value:.3f})",
                f"  Recall@K:   {best.recall_at_k:.4f}",
                f"  Precision@K:{best_prec:.4f}",
                f"  F1:         {best_f1:.4f}",
                f"  MRR:        {best.mrr:.4f}",
                f"  NDCG:       {best.ndcg:.4f}",
                f"  P@1:        {best.precision_at_1:.4f}",
                f"  Noise:      {best.contamination_rate:.4f}",
                f"  Latency p50:{best.latency_p50_ms:.1f}ms  p90:{best.latency_p90_ms:.1f}ms  p99:{best.latency_p99_ms:.1f}ms",
                f"  Composite:  {best.composite_score():.4f}",
            ]

        best_recall = agg.best_by_recall()
        if best_recall and best and best_recall.cell_id != best.cell_id:
            br_prec = best_recall.precision_at_k
            br_f1 = (2 * best_recall.recall_at_k * br_prec / (best_recall.recall_at_k + br_prec)) if (best_recall.recall_at_k + br_prec) > 0 else 0.0
            lines += [
                "",
                "HIGHEST RAW RECALL CONFIGURATION  (may differ from composite-score winner)",
                "  NOTE: Composite score balances Recall×0.40 + Precision×0.25 + MRR×0.20 + Temporal×0.15.",
                "        This config maximises Recall@K alone; the composite winner above has better overall balance.",
                f"  Memory:     {best_recall.memory_type}",
                f"  Strategy:   {best_recall.retrieval_strategy}",
                f"  Embedding:  {getattr(best_recall, 'embedding_model', '—')}",
                f"  BM25 wt:    {getattr(best_recall, 'bm25_weight', '—')}",
                f"  Reranker:   {getattr(best_recall, 'reranker_model', '—')}",
                f"  Decay:      {best_recall.decay_policy} (λ={best_recall.lambda_value:.3f})",
                f"  Recall@K:   {best_recall.recall_at_k:.4f}",
                f"  Precision@K:{br_prec:.4f}",
                f"  F1:         {br_f1:.4f}",
                f"  MRR:        {best_recall.mrr:.4f}",
                f"  Composite:  {best_recall.composite_score():.4f}",
            ]

        lines += ["", "STRATEGY RANKING (avg composite score)"]
        for i, row in enumerate(agg.rank_by_retrieval_strategy(), 1):
            avg_r = row['avg_recall']
            avg_p = row.get('avg_precision', 0.0)
            f1 = (2 * avg_r * avg_p / (avg_r + avg_p)) if (avg_r + avg_p) > 0 else 0.0
            lines.append(
                f"  {i}. {row['retrieval_strategy']:20s}  "
                f"composite={row['avg_composite']:.4f}  recall={avg_r:.4f}  "
                f"precision={avg_p:.4f}  f1={f1:.4f}  mrr={row.get('avg_mrr', 0.0):.4f}"
            )

        embed_rows = agg.rank_by_embedding_model()
        all_platforms = sorted({p for r in embed_rows for p in r.get("platforms", [])})
        is_multi_platform = len(all_platforms) > 1

        lines += ["", "EMBEDDING MODEL RANKING  (grouped by model × backend)"]
        if is_multi_platform:
            lines.append(
                f"  NOTE: Results from multiple platforms: {all_platforms}. "
                "Recall is platform-independent. Latency shown per platform."
            )
        lat_header = "  ".join(f"Lat({p[:3]})(ms)" for p in all_platforms) if is_multi_platform else "Lat(ms)"
        lines.append(f"  {'Model':35s}  {'Backend':20s}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'MRR':>7}  {lat_header}")
        lines.append("  " + "-" * (100 + (len(all_platforms) - 1) * 14 if is_multi_platform else 0))
        for i, row in enumerate(embed_rows, 1):
            if is_multi_platform:
                lat_str = "  ".join(
                    f"{row['latency_by_platform'].get(p, 0.0):12.1f}"
                    for p in all_platforms
                )
            else:
                lat_str = f"{row['avg_latency_ms']:8.1f}"
            lines.append(
                f"  {i}. {row['embedding_model']:33s}  "
                f"{row.get('embedding_backend', ''):20s}  "
                f"{row['avg_recall']:7.4f}  {row['avg_precision']:7.4f}  "
                f"{row['avg_f1']:7.4f}  {row['avg_mrr']:7.4f}  "
                f"{lat_str}"
            )

        lines += ["", "HYBRID BM25 WEIGHT RANKING"]
        for i, row in enumerate(agg.rank_by_bm25_weight(), 1):
            lines.append(
                f"  {i}. bm25={row['bm25_weight']:.2f} semantic={row['semantic_weight']:.2f}  "
                f"recall={row['avg_recall']:.4f}  mrr={row['avg_mrr']:.4f}  noise={row['avg_noise']:.4f}"
            )

        lines += ["", "RERANKER RANKING (lift vs no-reranker baseline)"]
        for i, row in enumerate(agg.rank_by_reranker(), 1):
            lines.append(
                f"  {i}. {row['reranker_model']:40s}  "
                f"recall={row['avg_recall']:.4f}  prec={row['avg_precision']:.4f}  "
                f"mrr={row['avg_mrr']:.4f}  "
                f"recall_lift={row['recall_lift_vs_none']:+.4f}  mrr_lift={row['mrr_lift_vs_none']:+.4f}  "
                f"lat={row['avg_latency_ms']:.1f}ms"
            )

        lines += ["", "DECAY POLICY RANKING"]
        for i, row in enumerate(agg.rank_by_decay_policy(), 1):
            lines.append(
                f"  {i}. {row['decay_policy']:14s}  "
                f"composite={row['avg_composite']:.4f}  recall={row['avg_recall']:.4f}  "
                f"mrr={row.get('avg_mrr', 0.0):.4f}"
            )

        recs = agg.study_summary().get("recommendations", {})
        lines += [
            "",
            "TOP-RANKED CONFIGURATION  (each axis ranked independently from its own phase)",
            f"  Retrieval strategy:  {recs.get('best_retrieval_strategy', '—')}",
            f"  Embedding model:     {recs.get('best_embedding_model', '—')}",
            f"  Embedding backend:   {recs.get('best_embedding_backend', '—')}",
            f"  BM25 weight:         {recs.get('best_bm25_weight', '—')}",
            f"  Reranker:            {recs.get('best_reranker', '—')}",
            "",
            "  NOTE: Each axis is averaged across its own phase only — embedding_model",
            "  and embedding_backend are always from the same row (never mixed).",
            "  Per-phase and per-dataset breakdowns are in the full CSV and JSON.",
        ]

        lines += ["", sep]
        return lines
