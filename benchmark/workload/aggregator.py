"""Matrix run aggregation and comparison reports.

Aggregates MatrixRunResults into:
- Per-axis rankings (which memory type wins? which strategy? which decay?)
- Full grid table (memory × strategy × decay)
- Best configuration recommendation
- CSV and JSON outputs
"""

from __future__ import annotations

import csv
import heapq
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class MatrixAggregator:
    """Aggregates results from a matrix benchmark run."""

    def __init__(self, results: list):
        """Args:
        results: List of MatrixRunResult objects.
        """
        self._results = results
        self._successful = [r for r in results if r.success]
        self._failed = [r for r in results if not r.success]

    @property
    def total(self) -> int:
        return len(self._results)

    @property
    def success_count(self) -> int:
        return len(self._successful)

    @property
    def failure_count(self) -> int:
        return len(self._failed)

    def best_overall(self) -> Any | None:
        """Return the single best result by composite score."""
        if not self._successful:
            return None
        return max(self._successful, key=lambda r: r.composite_score())

    def rank_by_memory_type(self) -> list[dict]:
        """Average composite score per memory type, ranked best→worst."""
        by_type: dict[str, list[float]] = defaultdict(list)
        by_type_recall: dict[str, list[float]] = defaultdict(list)
        by_type_noise: dict[str, list[float]] = defaultdict(list)

        for r in self._successful:
            by_type[r.memory_type].append(r.composite_score())
            by_type_recall[r.memory_type].append(r.recall_at_k)
            by_type_noise[r.memory_type].append(r.contamination_rate)

        rows = []
        for mem_type, scores in by_type.items():
            rows.append(
                {
                    "memory_type": mem_type,
                    "avg_composite": round(statistics.mean(scores), 4),
                    "avg_recall": round(statistics.mean(by_type_recall[mem_type]), 4),
                    "avg_noise": round(statistics.mean(by_type_noise[mem_type]), 4),
                    "best_composite": round(max(scores), 4),
                    "runs": len(scores),
                }
            )

        return sorted(rows, key=lambda x: x["avg_composite"], reverse=True)

    def rank_by_retrieval_strategy(self) -> list[dict]:
        """Average composite score per retrieval strategy, ranked best→worst."""
        by_strat: dict[str, list[float]] = defaultdict(list)
        by_strat_recall: dict[str, list[float]] = defaultdict(list)
        by_strat_prec: dict[str, list[float]] = defaultdict(list)
        by_strat_mrr: dict[str, list[float]] = defaultdict(list)
        by_strat_ndcg: dict[str, list[float]] = defaultdict(list)
        by_strat_noise: dict[str, list[float]] = defaultdict(list)
        by_strat_lat: dict[str, list[float]] = defaultdict(list)

        # Map configured strategy name → expected substring in resolved class name.
        # When a strategy is permanently disabled (OOM, 404, etc.) the store falls
        # back to SequenceMatcher; resolved_retriever_class then reflects the store
        # class (e.g. EpisodicStore) rather than the embedding strategy class.
        # Cells where the strategy fell back are tagged separately so their
        # SequenceMatcher recall doesn't drag down the real strategy average.
        _STRATEGY_CLASS_HINTS = {
            "embeddings": "Embedding",
            "api_embeddings": "Api",
            "hybrid": "Hybrid",
            "llm_rerank": "Rerank",
            "bm25": "BM25",
        }
        for r in self._successful:
            configured = r.retrieval_strategy
            resolved = getattr(r, "resolved_retriever_class", "")
            hint = _STRATEGY_CLASS_HINTS.get(configured, "")
            # If resolved class doesn't match the expected strategy, the cell ran
            # on a fallback — tag it as "<strategy>_fallback" so it's excluded
            # from the intended strategy's averages but still visible in output.
            if hint and resolved and hint.lower() not in resolved.lower():
                label = f"{configured}_fallback"
            else:
                label = configured
            by_strat[label].append(r.composite_score())
            by_strat_recall[label].append(r.recall_at_k)
            by_strat_prec[label].append(r.precision_at_k)
            by_strat_mrr[label].append(r.mrr)
            by_strat_ndcg[label].append(r.ndcg)
            by_strat_noise[label].append(r.contamination_rate)
            by_strat_lat[label].append(r.latency_p50_ms)

        rows = []
        for strat, scores in by_strat.items():
            avg_r = statistics.mean(by_strat_recall[strat])
            avg_p = statistics.mean(by_strat_prec[strat]) if by_strat_prec[strat] else 0.0
            f1 = (2 * avg_r * avg_p / (avg_r + avg_p)) if (avg_r + avg_p) > 0 else 0.0
            rows.append(
                {
                    "retrieval_strategy": strat,
                    "avg_composite": round(statistics.mean(scores), 4),
                    "avg_recall": round(avg_r, 4),
                    "avg_precision": round(avg_p, 4),
                    "avg_f1": round(f1, 4),
                    "avg_mrr": round(statistics.mean(by_strat_mrr[strat]), 4) if by_strat_mrr[strat] else 0.0,
                    "avg_ndcg": round(statistics.mean(by_strat_ndcg[strat]), 4) if by_strat_ndcg[strat] else 0.0,
                    "avg_noise": round(statistics.mean(by_strat_noise[strat]), 4),
                    "avg_latency_p50_ms": round(statistics.mean(by_strat_lat[strat]), 1) if by_strat_lat[strat] else 0.0,
                    "best_composite": round(max(scores), 4),
                    "runs": len(scores),
                }
            )

        return sorted(rows, key=lambda x: x["avg_composite"], reverse=True)

    def rank_by_decay_policy(self) -> list[dict]:
        """Average composite score per decay policy, ranked best→worst."""
        by_policy: dict[str, list[float]] = defaultdict(list)
        by_policy_recall: dict[str, list[float]] = defaultdict(list)
        by_policy_noise: dict[str, list[float]] = defaultdict(list)

        for r in self._successful:
            by_policy[r.decay_policy].append(r.composite_score())
            by_policy_recall[r.decay_policy].append(r.recall_at_k)
            by_policy_noise[r.decay_policy].append(r.contamination_rate)

        rows = []
        for policy, scores in by_policy.items():
            rows.append(
                {
                    "decay_policy": policy,
                    "avg_composite": round(statistics.mean(scores), 4),
                    "avg_recall": round(statistics.mean(by_policy_recall[policy]), 4),
                    "avg_noise": round(statistics.mean(by_policy_noise[policy]), 4),
                    "best_composite": round(max(scores), 4),
                    "runs": len(scores),
                }
            )

        return sorted(rows, key=lambda x: x["avg_composite"], reverse=True)

    def lambda_sweep_for(self, memory_type: str, strategy: str, policy: str) -> list[dict]:
        """Show how composite score changes as lambda increases for a fixed combination."""
        matching = [
            r
            for r in self._successful
            if r.memory_type == memory_type
            and r.retrieval_strategy == strategy
            and r.decay_policy == policy
        ]
        rows = []
        for r in sorted(matching, key=lambda x: x.lambda_value):
            rows.append(
                {
                    "lambda": r.lambda_value,
                    "recall_at_k": round(r.recall_at_k, 4),
                    "noise_ratio": round(r.contamination_rate, 4),
                    "composite_score": round(r.composite_score(), 4),
                }
            )
        return rows

    def top_n(self, n: int = 10) -> list[dict]:
        """Top N cells by composite score — O(N log n) heap instead of O(N log N) sort."""
        top = heapq.nlargest(n, self._successful, key=lambda r: r.composite_score())
        return [r.to_dict() for r in top]

    def build_grid_table(self) -> list[dict]:
        """Build a flat table of all cells suitable for CSV export.

        Column order is intentional: configuration dimensions first, then
        metrics. Every row fully describes its configuration — no dimension
        is implicit or recoverable only from context.

        For large runs (10K+ cells), use iter_grid_rows() to stream rows
        directly to a writer without materializing the full list.
        """
        return list(self.iter_grid_rows())

    def iter_grid_rows(self):
        """Generator version of build_grid_table() — O(1) peak memory regardless of N cells."""
        import os as _os
        top_k = int(_os.environ.get("BENCHMARK_RECALL_K", "10"))
        for r in self._results:
            row = {
                # ── Configuration dimensions (fully self-describing) ───────
                "cell_id": r.cell_id,
                "study_phase": getattr(r, "study_phase", ""),
                "memory_type": r.memory_type,
                "retrieval_strategy": r.retrieval_strategy,
                "embedding_model": getattr(r, "embedding_model", ""),
                "embedding_backend": getattr(r, "embedding_backend", ""),
                "bm25_weight": getattr(r, "bm25_weight", ""),
                "semantic_weight": getattr(r, "semantic_weight", ""),
                "reranker_model": getattr(r, "reranker_model", "none"),
                "decay_policy": r.decay_policy,
                "lambda": r.lambda_value,
                "pruning_threshold": r.pruning_threshold,
                "archival_floor": getattr(r, "archival_floor", 0.65),
                "top_k": getattr(r, "top_k", top_k),
                "workload_profile": r.workload_profile,
                "seed": r.seed,
                # ── Metrics ───────────────────────────────────────────────
                "recall_at_k": round(r.recall_at_k, 4),
                "precision_at_k": round(r.precision_at_k, 4),
                "mrr": round(r.mrr, 4),
                "ndcg": round(r.ndcg, 4),
                "precision_at_1": round(r.precision_at_1, 4),
                "contamination_rate": round(r.contamination_rate, 4),
                "temporal_accuracy": round(r.temporal_accuracy, 4),
                "composite_score": round(r.composite_score(), 4),
                "total_queries": r.total_queries,
                "correct_recalls": r.correct_recalls,
                # ── Latency ───────────────────────────────────────────────
                "latency_p50_ms": round(r.latency_p50_ms, 3),
                "latency_p90_ms": round(r.latency_p90_ms, 3),
                "latency_p99_ms": round(r.latency_p99_ms, 3),
                "latency_mean_ms": round(r.latency_mean_ms, 3),
                # ── Resources ─────────────────────────────────────────────
                "peak_ram_mb": round(r.peak_ram_mb, 2),
                "avg_ram_mb": round(r.avg_ram_mb, 2),
                "peak_cpu_percent": round(r.peak_cpu_percent, 2),
                "avg_cpu_percent": round(r.avg_cpu_percent, 2),
                "duration_seconds": round(r.duration_seconds, 3),
                "disk_write_mb": round(r.disk_write_mb, 2),
                "total_cost_usd": round(r.total_cost, 6),
                # ── Run metadata ──────────────────────────────────────────
                "success": r.success,
                "error": r.error_message[:100] if r.error_message else "",
                "platform": r.platform,
            }
            yield row

    def summary(self) -> dict:
        """High-level summary of the full matrix run."""
        best = self.best_overall()
        return {
            "total_cells": self.total,
            "successful": self.success_count,
            "failed": self.failure_count,
            "best_config": best.to_dict() if best else None,
            "memory_type_ranking": self.rank_by_memory_type(),
            "retrieval_strategy_ranking": self.rank_by_retrieval_strategy(),
            "decay_policy_ranking": self.rank_by_decay_policy(),
            "top_10": self.top_n(10),
        }


class MatrixReporter:
    """Writes aggregated matrix results to disk in multiple formats."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_all(self, aggregator: MatrixAggregator, run_id: str) -> dict[str, str]:
        """Write JSON, CSV, and text summary with datetime-stamped filenames.

        Returns dict of written file paths.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}
        paths["summary_json"] = self._write_summary_json(aggregator, run_id, ts)
        paths["grid_csv"] = self._write_grid_csv(aggregator, run_id, ts)
        paths["text_report"] = self._write_text_report(aggregator, run_id, ts)
        return paths

    def _write_summary_json(self, agg: MatrixAggregator, run_id: str, ts: str) -> str:
        path = self._output_dir / f"matrix_{ts}_{run_id}_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(agg.summary(), f, indent=2)
        return str(path)

    def _write_grid_csv(self, agg: MatrixAggregator, run_id: str, ts: str) -> str:
        path = self._output_dir / f"matrix_{ts}_{run_id}_grid.csv"
        rows = agg.build_grid_table()
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return str(path)

    def _write_text_report(self, agg: MatrixAggregator, run_id: str, ts: str) -> str:
        path = self._output_dir / f"matrix_{ts}_{run_id}_report.txt"
        lines = self._format_text_report(agg, run_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(path)

    def _format_text_report(self, agg: MatrixAggregator, run_id: str) -> list[str]:
        sep = "=" * 72
        lines = [
            sep,
            "AGENTIC MEMORY BENCHMARK — MATRIX RESULTS",
            f"Run ID:       {run_id}",
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Cells:        {agg.success_count}/{agg.total} successful",
            sep,
        ]

        # Best config
        best = agg.best_overall()
        if best:
            lines += [
                "",
                "BEST CONFIGURATION",
                f"  Memory type:        {best.memory_type}",
                f"  Retrieval strategy: {best.retrieval_strategy}",
                f"  Decay policy:       {best.decay_policy} (λ={best.lambda_value:.2f})",
                f"  Recall@K:           {best.recall_at_k:.4f}",
                f"  Correct recalls:    {best.correct_recalls}/{best.total_queries} queries",
                f"  Precision@K:        {best.precision_at_k:.4f}",
                f"  MRR:                {best.mrr:.4f}",
                f"  NDCG:               {best.ndcg:.4f}",
                f"  Composite Score:    {best.composite_score():.4f}",
                f"  Peak RAM:           {best.peak_ram_mb:.1f} MB",
                f"  Duration:           {best.duration_seconds:.1f}s",
            ]

        # Memory type ranking
        lines += ["", "MEMORY TYPE RANKING (by avg composite score)"]
        for i, row in enumerate(agg.rank_by_memory_type(), 1):
            lines.append(
                f"  {i}. {row['memory_type']:12s}  composite={row['avg_composite']:.4f}"
                f"  recall={row['avg_recall']:.4f}  noise={row['avg_noise']:.4f}"
            )

        # Retrieval strategy ranking
        lines += ["", "RETRIEVAL STRATEGY RANKING (by avg composite score)"]
        for i, row in enumerate(agg.rank_by_retrieval_strategy(), 1):
            lines.append(
                f"  {i}. {row['retrieval_strategy']:12s}  composite={row['avg_composite']:.4f}"
                f"  recall={row['avg_recall']:.4f}  noise={row['avg_noise']:.4f}"
            )

        # Decay policy ranking
        lines += ["", "DECAY POLICY RANKING (by avg composite score)"]
        for i, row in enumerate(agg.rank_by_decay_policy(), 1):
            lines.append(
                f"  {i}. {row['decay_policy']:12s}  composite={row['avg_composite']:.4f}"
                f"  recall={row['avg_recall']:.4f}  noise={row['avg_noise']:.4f}"
            )

        # Top 10
        lines += ["", "TOP 10 CONFIGURATIONS"]
        for i, d in enumerate(agg.top_n(10), 1):
            m = d["metrics"]
            r = d["resources"]
            lat = d.get("latency", {})
            lines.append(
                f"  {i:2d}. {d['memory_type']:12s} × {d['retrieval_strategy']:10s} × "
                f"{d['decay_policy']:12s}(λ={d['lambda_value']:.2f})"
                f"  score={m['composite_score']:.4f}"
                f"  recall={m['recall_at_k']:.4f}"
                f"  hits={m.get('correct_recalls', 0)}"
                f"  prec@k={m.get('precision_at_k', 0):.4f}"
                f"  mrr={m.get('mrr', 0):.4f}"
                f"  p50={lat.get('p50_ms', 0):.1f}ms"
                f"  ram={r['peak_ram_mb']:.0f}MB"
            )

        if agg.failure_count > 0:
            lines += ["", f"FAILURES: {agg.failure_count} cells did not complete successfully"]

        lines += self._format_variance_warning(agg)
        lines += ["", sep]
        return lines

    def _format_variance_warning(self, agg: MatrixAggregator) -> list[str]:
        """Emit variance diagnostics to help identify saturation problems."""
        successful = agg._successful if hasattr(agg, "_successful") else []

        if not successful:
            return []

        # Single Welford pass: composite mean/variance + recall/temporal min/max
        # in one loop instead of 4 separate list comprehensions + min/max scans.
        n = 0
        _mean = 0.0
        _m2 = 0.0
        _score_min = float("inf")
        _score_max = float("-inf")
        _recall_min = float("inf")
        _recall_max = float("-inf")
        _temporal_min = float("inf")
        _temporal_max = float("-inf")
        for r in successful:
            s = r.composite_score()
            n += 1
            delta = s - _mean
            _mean += delta / n
            _m2 += delta * (s - _mean)
            if s < _score_min:
                _score_min = s
            if s > _score_max:
                _score_max = s
            rec = r.recall_at_k
            if rec < _recall_min:
                _recall_min = rec
            if rec > _recall_max:
                _recall_max = rec
            tmp = r.temporal_accuracy
            if tmp < _temporal_min:
                _temporal_min = tmp
            if tmp > _temporal_max:
                _temporal_max = tmp

        if n < 2:
            return []

        mean = _mean
        std_dev = (_m2 / n) ** 0.5
        score_range = _score_max - _score_min
        recall_range = _recall_max - _recall_min
        temporal_range = _temporal_max - _temporal_min

        lines = ["", "VARIANCE DIAGNOSTICS"]
        lines.append(f"  Composite score mean:    {mean:.4f}")
        lines.append(f"  Composite score std dev: {std_dev:.4f}")
        lines.append(f"  Composite score range:   {score_range:.4f} (max - min)")
        lines.append(f"  Recall@K range:          {recall_range:.4f}")

        lines.append(f"  Temporal accuracy range: {temporal_range:.4f}")

        # Warnings
        warnings = []
        if score_range < 0.05:
            warnings.append(
                "⚠️  SATURATION: Composite score range < 0.05. Strategies cannot be differentiated."
            )
            warnings.append(
                "   Fix: Increase simulation horizon (--days 30+), add paraphrased queries, or verify evaluators."
            )
        if recall_range < 0.01:
            warnings.append(
                "⚠️  RECALL SATURATION: All strategies achieve similar recall. Dataset may lack variation."
            )
        if temporal_range < 0.01:
            warnings.append(
                "⚠️  TEMPORAL SATURATION: Temporal accuracy uniform. Need longer time horizon."
            )
        # Memory-type invariance check — if recall is identical across memory
        # types for each (strategy, decay) pair, note it is expected behavior
        # when all memories are indexed into a single store per cell.
        from collections import defaultdict

        type_recalls: dict[str, set[float]] = defaultdict(set)
        for r in successful:
            key = f"{r.retrieval_strategy}_{r.decay_policy}_{r.lambda_value}"
            type_recalls[key].add(round(r.recall_at_k, 4))
        invariant_pairs = sum(1 for v in type_recalls.values() if len(v) == 1)
        if invariant_pairs == len(type_recalls) and len(type_recalls) > 1:
            warnings.append(
                "ℹ️  MEMORY TYPE INVARIANCE: All memory types produce identical recall "
                "for each (strategy, decay) pair. This is expected — the dataset does not "
                "assign memories to specific module types. Differentiation axis: strategy × decay."
            )

        if warnings:
            lines += ["", "⚠️  BENCHMARK HEALTH WARNINGS"] + warnings
        else:
            lines.append("  ✅ Variance looks healthy. Benchmark has discriminative power.")

        return lines
