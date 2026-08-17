"""Parallel matrix scheduler — runs benchmark cells concurrently.

Cross-platform process-based parallelism (macOS + Windows).
Each MatrixCell runs in its own isolated subprocess.

Design:
- Uses ProcessPoolExecutor for CPU isolation and GIL bypass
- Each cell gets its own seed, config snapshot, and run_id
- Results are collected as they complete (not in order)
- Total concurrency defaults to cpu_count - 1
- On Windows: uses 'spawn' start method (Python default on Windows)
- On macOS/Linux: uses 'spawn' to avoid fork-safety issues

Each worker:
1. Builds BenchmarkConfig from MatrixCell
2. Wires memory modules via factory
3. Runs benchmark with resource tracking
4. Returns serializable MatrixRunResult
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path


@dataclass
class MatrixRunResult:
    """Result of a single matrix cell benchmark run."""

    cell_id: str
    run_id: str
    memory_type: str
    retrieval_strategy: str
    decay_policy: str
    lambda_value: float
    pruning_threshold: float
    workload_profile: str
    seed: int

    # Metrics
    recall_at_k: float = 0.0
    contamination_rate: float = 0.0
    precision_at_k: float = 0.0
    temporal_accuracy: float = 0.0
    module_accuracy: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    precision_at_1: float = 0.0
    total_queries: int = 0
    correct_recalls: int = 0

    # Resources
    peak_ram_mb: float = 0.0
    avg_ram_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    duration_seconds: float = 0.0
    disk_write_mb: float = 0.0

    # Latency percentiles (per-query retrieval)
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0

    # Cost
    total_cost: float = 0.0

    # Status
    success: bool = True
    error_message: str = ""
    platform: str = ""

    # Debug: which retrieval class was actually resolved
    resolved_retriever_class: str = ""

    @staticmethod
    def from_csv_row(row: dict) -> "MatrixRunResult":
        """Reconstruct a MatrixRunResult from a grid CSV row dict."""
        return MatrixRunResult(
            cell_id=row.get("cell_id", ""),
            run_id=row.get("run_id", ""),
            memory_type=row.get("memory_type", ""),
            retrieval_strategy=row.get("retrieval_strategy", ""),
            decay_policy=row.get("decay_policy", ""),
            lambda_value=float(row.get("lambda", 0.0)),
            pruning_threshold=float(row.get("pruning_threshold", 0.15)),
            workload_profile=row.get("workload_profile", ""),
            seed=int(row.get("seed", 42)),
            recall_at_k=float(row.get("recall_at_k", 0.0)),
            contamination_rate=float(row.get("contamination_rate", 0.0)),
            precision_at_k=float(row.get("precision_at_k", 0.0)),
            temporal_accuracy=float(row.get("temporal_accuracy", 0.0)),
            mrr=float(row.get("mrr", 0.0)),
            ndcg=float(row.get("ndcg", 0.0)),
            precision_at_1=float(row.get("precision_at_1", 0.0)),
            total_queries=int(row.get("total_queries", 0)),
            correct_recalls=int(row.get("correct_recalls", 0)),
            peak_ram_mb=float(row.get("peak_ram_mb", 0.0)),
            avg_ram_mb=float(row.get("avg_ram_mb", 0.0)),
            peak_cpu_percent=float(row.get("peak_cpu_percent", 0.0)),
            avg_cpu_percent=float(row.get("avg_cpu_percent", 0.0)),
            duration_seconds=float(row.get("duration_seconds", 0.0)),
            disk_write_mb=float(row.get("disk_write_mb", 0.0)),
            total_cost=float(row.get("total_cost_usd", 0.0)),
            success=str(row.get("success", "True")).lower() in ("true", "1"),
            error_message=row.get("error", ""),
            platform=row.get("platform", ""),
        )

    # Default composite score weights — single source of truth.
    # Rationale: Recall (40%) = primary retrieval coverage; Precision (25%) = set
    # cleanliness; MRR (20%) = ranking quality; Temporal (15%) = time-window accuracy.
    # Run sensitivity analysis with composite_score_weighted() before relying on
    # rankings that are within 0.01 of each other.
    COMPOSITE_WEIGHTS: ClassVar[dict] = {
        "recall":    0.40,
        "precision": 0.25,
        "mrr":       0.20,
        "temporal":  0.15,
    }

    def composite_score(self) -> float:
        """Weighted composite score for ranking using COMPOSITE_WEIGHTS.

        composite = recall_gate × (w_R × Recall@K + w_P × Precision@K
                                  + w_M × MRR + w_T × TemporalAccuracy)

        Recall gate: if Recall@K < 0.01 → returns 0.0 to prevent an empty
        store with perfect temporal accuracy from outranking a working system.

        Range: [0.0, 1.0]. Higher is better.
        Use composite_score_weighted() for sensitivity analysis.
        """
        if self.recall_at_k < 0.01:
            return 0.0
        w = self.COMPOSITE_WEIGHTS
        return (
            w["recall"]    * self.recall_at_k
            + w["precision"] * self.precision_at_k
            + w["mrr"]       * self.mrr
            + w["temporal"]  * self.temporal_accuracy
        )

    def composite_score_weighted(self, weights: dict) -> float:
        """Composite score with caller-supplied weights for sensitivity analysis.

        Args:
            weights: dict with keys 'recall', 'precision', 'mrr', 'temporal'.
                     Values must sum to 1.0. Missing keys default to 0.0.

        Example — recall-only:
            result.composite_score_weighted({"recall": 1.0})
        Example — equal weights:
            result.composite_score_weighted({k: 0.25 for k in COMPOSITE_WEIGHTS})
        """
        if self.recall_at_k < 0.01:
            return 0.0
        return (
            weights.get("recall",    0.0) * self.recall_at_k
            + weights.get("precision", 0.0) * self.precision_at_k
            + weights.get("mrr",       0.0) * self.mrr
            + weights.get("temporal",  0.0) * self.temporal_accuracy
        )

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "run_id": self.run_id,
            "memory_type": self.memory_type,
            "retrieval_strategy": self.retrieval_strategy,
            "decay_policy": self.decay_policy,
            "lambda_value": self.lambda_value,
            "pruning_threshold": self.pruning_threshold,
            "workload_profile": self.workload_profile,
            "seed": self.seed,
            "metrics": {
                "recall_at_k": round(self.recall_at_k, 4),
                "precision_at_k": round(self.precision_at_k, 4),
                "contamination_rate": round(self.contamination_rate, 4),
                "temporal_accuracy": round(self.temporal_accuracy, 4),
                "mrr": round(self.mrr, 4),
                "ndcg": round(self.ndcg, 4),
                "precision_at_1": round(self.precision_at_1, 4),
                "composite_score": round(self.composite_score(), 4),
                "total_queries": self.total_queries,
                "correct_recalls": self.correct_recalls,
            },
            "resources": {
                "peak_ram_mb": round(self.peak_ram_mb, 2),
                "avg_ram_mb": round(self.avg_ram_mb, 2),
                "peak_cpu_percent": round(self.peak_cpu_percent, 2),
                "avg_cpu_percent": round(self.avg_cpu_percent, 2),
                "duration_seconds": round(self.duration_seconds, 3),
                "disk_write_mb": round(self.disk_write_mb, 2),
                "platform": self.platform,
            },
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 3),
                "p90_ms": round(self.latency_p90_ms, 3),
                "p99_ms": round(self.latency_p99_ms, 3),
                "mean_ms": round(self.latency_mean_ms, 3),
            },
            "cost": {
                "total_cost_usd": round(self.total_cost, 6),
            },
            "status": {
                "success": self.success,
                "error_message": self.error_message,
                "resolved_retriever_class": self.resolved_retriever_class,
            },
        }


def _run_cell_worker(
    cell_dict: dict,
    gold_dataset_path: str,
    output_dir: str,
    evaluation_horizon: int,
) -> dict:
    """Worker function executed in a subprocess.

    Delegates all composition to BenchmarkComposer — the same path used by
    the CLI. This ensures CLI and matrix produce identical results.

    Must be a module-level function (not a closure) for pickling.

    Args:
        cell_dict: Serialized MatrixCell.to_summary_dict()
        gold_dataset_path: Path to the gold dataset JSON file.
        output_dir: Directory to write per-cell results.
        evaluation_horizon: Number of dataset days to replay for this run.

    Returns:
        Serialized MatrixRunResult.to_dict()
    """
    import logging
    import sys
    import time

    # Re-import benchmark inside subprocess (required for spawn method)
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Suppress verbose INFO logs in worker processes
    logging.disable(logging.INFO)

    from benchmark.application.composer import BenchmarkComposer
    from benchmark.config.loader import load_config_from_dict
    from benchmark.gold.oracle import GoldOracle
    from benchmark.resources.tracker import ResourceTracker
    from benchmark.workload.matrix import DecaySpec, MatrixCell

    run_id = uuid.uuid4().hex[:12]
    start = time.monotonic()

    # Reconstruct MatrixCell from serialized dict
    decay = DecaySpec(
        policy=cell_dict["decay_policy"],
        lambda_value=cell_dict["lambda"],
        pruning_threshold=cell_dict["pruning_threshold"],
    )
    cell = MatrixCell(
        memory_type=cell_dict["memory_type"],
        retrieval_strategy=cell_dict["retrieval_strategy"],
        decay=decay,
        workload_profile=cell_dict["workload_profile"],
        seed=cell_dict["seed"],
    )

    result = MatrixRunResult(
        cell_id=cell.cell_id,
        run_id=run_id,
        memory_type=cell.memory_type,
        retrieval_strategy=cell.retrieval_strategy,
        decay_policy=decay.policy,
        lambda_value=decay.lambda_value,
        pruning_threshold=decay.pruning_threshold,
        workload_profile=cell.workload_profile,
        seed=cell.seed,
        platform=sys.platform,
    )

    normalization_meta: dict = {"applied": False}

    try:
        config_dict = cell.to_config_dict(evaluation_horizon)
        config = load_config_from_dict(config_dict)

        # Use BenchmarkComposer — single composition root shared with CLI.
        # No manual registry bootstrap, strategy resolution, or policy wiring.
        composer = BenchmarkComposer()
        composed = composer.compose(
            config=config,
            dataset_path=Path(gold_dataset_path),
            allow_strategy_fallback=True,
        )

        result.resolved_retriever_class = composed.run_plan.effective_strategy

        # Capture normalization metadata from the oracle
        gold_oracle = GoldOracle()
        gold_oracle.load_dataset(Path(gold_dataset_path))
        normalization_meta = gold_oracle.get_normalization_metadata(
            composed.run_plan.dataset_fingerprint
        ) or {"applied": False}
        # Fallback: use the run_plan normalization info
        if not normalization_meta.get("applied"):
            normalization_meta = {
                "applied": composed.run_plan.normalization_applied,
                "delta_days": composed.run_plan.normalization_delta_days,
            }

        # Track resources while running
        with ResourceTracker(interval_seconds=0.5) as res_tracker:
            bench_result = composed.runner.run(composed.scenarios)

        resource_report = res_tracker.report()

        # Extract metrics from result
        if bench_result.scenario_results:
            sr = bench_result.scenario_results[0]
            result.recall_at_k = sr.recall_at_k
            result.contamination_rate = sr.contamination_rate
            result.precision_at_k = sr.precision_at_k
            result.temporal_accuracy = sr.temporal_accuracy
            result.module_accuracy = sr.module_accuracy
            result.mrr = sr.mrr
            result.ndcg = sr.ndcg
            result.precision_at_1 = sr.precision_at_1
            result.total_queries = sr.total_queries
            result.correct_recalls = sr.correct_recalls
            result.latency_p50_ms = sr.latency_p50_ms
            result.latency_p90_ms = sr.latency_p90_ms
            result.latency_p99_ms = sr.latency_p99_ms
            result.latency_mean_ms = sr.latency_mean_ms

        result.peak_ram_mb = resource_report.peak_ram_mb
        result.avg_ram_mb = resource_report.avg_ram_mb
        result.peak_cpu_percent = resource_report.peak_cpu_percent
        result.avg_cpu_percent = resource_report.avg_cpu_percent
        result.duration_seconds = resource_report.duration_seconds
        result.disk_write_mb = resource_report.total_disk_write_mb
        result.total_cost = (
            bench_result.cost_summary.total_cost if bench_result.cost_summary else 0.0
        )
        result.success = True

    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
        result.duration_seconds = time.monotonic() - start

    result_dict = result.to_dict()
    result_dict["_timestamp_normalization"] = normalization_meta
    return result_dict


class MatrixScheduler:
    """Executes matrix cells in parallel using process-based workers.

    Cross-platform: uses 'spawn' start method for Windows compatibility.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        output_dir: str = "data/output",
    ):
        """Initialize the scheduler.

        Args:
            max_workers: Max parallel processes. Defaults to cpu_count - 1.
            output_dir: Directory to write results to.
        """
        cpu_count = os.cpu_count() or 2
        # Cap at 50% of cores to leave headroom for Ollama and other services
        default_workers = max(1, cpu_count // 2)
        self._max_workers = max_workers or default_workers
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Populated after run() by the first successful cell's normalization info.
        self.normalization_meta: dict = {"applied": False}

    def run(
        self,
        cells: list,
        gold_dataset_path: str,
        evaluation_horizon: int,
        progress_callback=None,
    ) -> list[MatrixRunResult]:
        """Execute all cells in parallel.

        Args:
            cells: List of MatrixCells to run.
            gold_dataset_path: Path to gold dataset.
            evaluation_horizon: Number of dataset days to replay per run.
            progress_callback: Optional callable(completed, total, result) for progress.

        Returns:
            List of MatrixRunResults (all cells, success or fail).
        """
        total = len(cells)
        results: list[MatrixRunResult] = []

        # Serialize cells for pickling across processes
        cell_dicts = [c.to_summary_dict() for c in cells]

        # Use 'spawn' context for cross-platform compatibility (required on Windows)
        ctx = multiprocessing.get_context("spawn")

        print(
            f"  Running {total} cells with {self._max_workers} parallel workers "
            f"(platform: {sys.platform})"
        )

        with ProcessPoolExecutor(max_workers=self._max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_cell_worker,
                    cell_dict,
                    gold_dataset_path,
                    str(self._output_dir),
                    evaluation_horizon,
                ): cell_dict
                for cell_dict in cell_dicts
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                cell_dict = futures[future]

                try:
                    result_dict = future.result(timeout=300)
                    # Capture normalization metadata from the first cell that has it.
                    if not self.normalization_meta.get("applied", False):
                        norm = result_dict.get("_timestamp_normalization", {})
                        if norm.get("applied", False):
                            self.normalization_meta = norm
                    result = self._dict_to_result(result_dict)
                except Exception as e:
                    # Worker crashed entirely
                    result = MatrixRunResult(
                        cell_id=cell_dict.get("cell_id", "unknown"),
                        run_id=uuid.uuid4().hex[:12],
                        memory_type=cell_dict.get("memory_type", "?"),
                        retrieval_strategy=cell_dict.get("retrieval_strategy", "?"),
                        decay_policy=cell_dict.get("decay_policy", "?"),
                        lambda_value=cell_dict.get("lambda", 0.0),
                        pruning_threshold=cell_dict.get("pruning_threshold", 0.3),
                        workload_profile=cell_dict.get("workload_profile", "?"),
                        seed=cell_dict.get("seed", 42),
                        success=False,
                        error_message=str(e),
                    )

                results.append(result)

                if progress_callback:
                    progress_callback(completed, total, result)
                else:
                    status = "✓" if result.success else "✗"
                    print(
                        f"  [{completed:3d}/{total}] {status} "
                        f"{result.memory_type:12s} × {result.retrieval_strategy:10s} × "
                        f"{result.decay_policy:12s}(λ={result.lambda_value:.4f}) "
                        f"recall={result.recall_at_k:.3f} prec@k={result.precision_at_k:.3f}"
                    )

        return results

    @staticmethod
    def _dict_to_result(d: dict) -> MatrixRunResult:
        """Reconstruct MatrixRunResult from dict returned by worker."""
        m = d.get("metrics", {})
        r = d.get("resources", {})
        c = d.get("cost", {})
        s = d.get("status", {})
        lat = d.get("latency", {})
        return MatrixRunResult(
            cell_id=d.get("cell_id", ""),
            run_id=d.get("run_id", ""),
            memory_type=d.get("memory_type", ""),
            retrieval_strategy=d.get("retrieval_strategy", ""),
            decay_policy=d.get("decay_policy", ""),
            lambda_value=d.get("lambda_value", 0.0),
            pruning_threshold=d.get("pruning_threshold", 0.3),
            workload_profile=d.get("workload_profile", ""),
            seed=d.get("seed", 42),
            recall_at_k=m.get("recall_at_k", 0.0),
            contamination_rate=m.get("contamination_rate", m.get("false_positive_rate", 0.0)),
            precision_at_k=m.get(
                "precision_at_k",
                1.0 - m.get("contamination_rate", m.get("false_positive_rate", 0.0)),
            ),
            temporal_accuracy=m.get("temporal_accuracy", 0.0),
            module_accuracy=m.get("module_accuracy", 1.0),
            mrr=m.get("mrr", 0.0),
            ndcg=m.get("ndcg", 0.0),
            precision_at_1=m.get("precision_at_1", 0.0),
            total_queries=m.get("total_queries", 0),
            correct_recalls=m.get("correct_recalls", 0),
            peak_ram_mb=r.get("peak_ram_mb", 0.0),
            avg_ram_mb=r.get("avg_ram_mb", 0.0),
            peak_cpu_percent=r.get("peak_cpu_percent", 0.0),
            avg_cpu_percent=r.get("avg_cpu_percent", 0.0),
            duration_seconds=r.get("duration_seconds", 0.0),
            disk_write_mb=r.get("disk_write_mb", 0.0),
            latency_p50_ms=lat.get("p50_ms", 0.0),
            latency_p90_ms=lat.get("p90_ms", 0.0),
            latency_p99_ms=lat.get("p99_ms", 0.0),
            latency_mean_ms=lat.get("mean_ms", 0.0),
            total_cost=c.get("total_cost_usd", 0.0),
            success=s.get("success", False),
            error_message=s.get("error_message", ""),
            resolved_retriever_class=s.get("resolved_retriever_class", ""),
            platform=r.get("platform", sys.platform),
        )
