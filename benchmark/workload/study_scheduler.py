"""Study scheduler — runs StudyCells through the existing worker pipeline.

Thin wrapper around MatrixScheduler logic that accepts StudyCell objects.
The worker function is extended to accept embedding_model, bm25_weight,
and reranker_model from the cell's to_summary_dict().
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from benchmark.resources.hw_probe import cpu_worker_count as _hw_worker_count
from benchmark.workload.scheduler import MatrixRunResult


def _run_study_cell_worker(
    cell_dict: dict,
    gold_dataset_path: str,
    output_dir: str,
    evaluation_horizon: int,
    gold_dataset_cache: dict | None = None,
) -> dict:
    """Worker function for study cells — runs inside a subprocess.

    Identical flow to _run_cell_worker but reconstructs a StudyCell
    (which produces richer config dicts with embedding / hybrid / reranker blocks).
    """
    import logging
    import time

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logging.disable(logging.INFO)

    from benchmark.application.composer import BenchmarkComposer
    from benchmark.config.loader import load_config_from_dict
    from benchmark.resources.tracker import ResourceTracker
    from benchmark.workload.matrix import DecaySpec
    from benchmark.workload.study_matrix import StudyCell

    run_id = uuid.uuid4().hex[:12]
    start = time.monotonic()

    _archival_floor_raw = cell_dict.get("archival_floor", 0.65)
    _archival_floor: float | None = (
        None if _archival_floor_raw is None
        else float(_archival_floor_raw) if str(_archival_floor_raw) not in ("", "None", "none")
        else None
    )
    decay = DecaySpec(
        policy=cell_dict["decay_policy"],
        lambda_value=cell_dict["lambda"],
        pruning_threshold=cell_dict["pruning_threshold"],
        ranking_alpha=float(cell_dict.get("ranking_alpha", 0.0)),
        archival_floor=_archival_floor,
        archival_day_threshold=int(cell_dict.get("archival_day_threshold", 90)),
        tiered_working_days=int(cell_dict.get("tiered_working_days", 7)),
    )
    cell = StudyCell(
        memory_type=cell_dict["memory_type"],
        retrieval_strategy=cell_dict["retrieval_strategy"],
        decay=decay,
        workload_profile=cell_dict["workload_profile"],
        embedding_model=cell_dict.get("embedding_model", ""),
        embedding_backend=cell_dict.get("embedding_backend", "sentence-transformers"),
        bm25_weight=cell_dict.get("bm25_weight", 0.0),
        reranker_model=cell_dict.get("reranker_model", "none"),
        ollama_base_url=cell_dict.get("ollama_base_url", ""),
        seed=cell_dict.get("seed", 42),
        study_phase=cell_dict.get("study_phase", "general"),
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

        # Wire LLM judge if configured via environment variables.
        # BENCHMARK_JUDGE_MODEL + BENCHMARK_JUDGE_BASE_URL (or BENCHMARK_LLM_BASE_URL).
        _answer_evaluator = None
        _judge_model = os.environ.get("BENCHMARK_JUDGE_MODEL", "")
        _judge_url = os.environ.get("BENCHMARK_JUDGE_BASE_URL") or os.environ.get("BENCHMARK_LLM_BASE_URL", "")

        # VRAM guard: Qwen3-4B embedding (7.6 GB fp16) + gemma4:12b judge (8 GB Q4)
        # exceed 15.5 GB together. Embedding encode happens before queries; Ollama judge
        # runs after. When both are active, PyTorch must release the embedding model
        # from VRAM before Ollama loads the judge — set BENCHMARK_JUDGE_UNLOAD_EMBED=1
        # to trigger this. For smaller embed+judge combos the guard is a no-op.
        _LARGE_EMBED_MB = 5000  # threshold above which we unload before judge call
        _embed_size = {
            "Qwen/Qwen3-Embedding-4B": 7600,
        }.get(cell_dict.get("embedding_model", ""), 0)
        _unload_embed_for_judge = (
            _embed_size >= _LARGE_EMBED_MB
            and _judge_model
            and os.environ.get("BENCHMARK_JUDGE_UNLOAD_EMBED", "auto") != "0"
        )
        if _unload_embed_for_judge:
            os.environ["BENCHMARK_JUDGE_UNLOAD_EMBED"] = "1"
        else:
            os.environ.pop("BENCHMARK_JUDGE_UNLOAD_EMBED", None)

        if _judge_model and _judge_url:
            try:
                from benchmark.judge.evaluator import EndToEndEvaluator
                _answer_evaluator = EndToEndEvaluator(judge_method="llm_judge")
            except Exception:
                pass  # judge unavailable — benchmark still runs without it

        composer = BenchmarkComposer()
        # Use pre-loaded dataset if available — avoids re-parsing 2.7 MB JSON per cell
        _dataset_override = (gold_dataset_cache or {}).get("dataset")
        composed = composer.compose(
            config=config,
            dataset_path=Path(gold_dataset_path) if _dataset_override is None else None,
            dataset_override=_dataset_override,
            allow_strategy_fallback=True,
            answer_evaluator=_answer_evaluator,
        )

        result.resolved_retriever_class = composed.run_plan.effective_strategy

        with ResourceTracker(interval_seconds=0.5) as res_tracker:
            bench_result = composed.runner.run(composed.scenarios)

        resource_report = res_tracker.report()

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
    # Attach all study dimensions for aggregation — every field must be present
    # so CSV rows are fully self-describing (no implicit values).
    result_dict["_study"] = {
        "embedding_model": cell.embedding_model,
        "embedding_backend": cell.embedding_backend,
        "bm25_weight": cell.bm25_weight,
        "semantic_weight": cell.semantic_weight,
        "reranker_model": cell.reranker_model,
        "study_phase": cell.study_phase,
        "top_k": cell_dict.get("top_k", 10),
        "archival_floor": cell.decay.archival_floor,
    }
    result_dict["_timestamp_normalization"] = normalization_meta
    return result_dict


class StudyRunResult(MatrixRunResult):
    """MatrixRunResult extended with study-specific dimensions.

    Default values are honest: 0.0 for numeric fields (not 0.5 which would
    imply a specific BM25 weight was used), empty string for model names.
    The actual configured values are always written by StudyCell.to_summary_dict().
    """

    embedding_model: str = ""
    embedding_backend: str = ""     # sentence-transformers | ollama | none
    bm25_weight: float = 0.0        # meaningful only for hybrid
    semantic_weight: float = 0.0    # 1 - bm25_weight for hybrid
    reranker_model: str = "none"
    study_phase: str = "general"
    top_k: int = 10
    archival_floor: float | None = 0.65   # Phase 4b sweep dimension

    def __init__(self, **kwargs):
        study_kwargs = {
            k: kwargs.pop(k, v)
            for k, v in [
                ("embedding_model", ""),
                ("embedding_backend", ""),
                ("bm25_weight", 0.0),
                ("semantic_weight", 0.0),
                ("reranker_model", "none"),
                ("study_phase", "general"),
                ("top_k", 10),
                ("archival_floor", 0.65),
            ]
        }
        super().__init__(**kwargs)
        for k, v in study_kwargs.items():
            object.__setattr__(self, k, v)

    @staticmethod
    def from_csv_row(row: dict, source_run_id: str = "") -> StudyRunResult:
        """Reconstruct a StudyRunResult from a grid CSV row dict.
        Uses empty string / 0.0 when fields are absent — never fabricates values.
        """
        base = MatrixRunResult.from_csv_row(row)
        if source_run_id:
            object.__setattr__(base, "run_id", source_run_id + ":" + base.run_id)
        raw_bm25w = row.get("bm25_weight", "")
        bm25_weight = float(raw_bm25w) if raw_bm25w != "" else 0.0
        raw_semw = row.get("semantic_weight", "")
        semantic_weight = float(raw_semw) if raw_semw != "" else round(1.0 - bm25_weight, 2)
        raw_topk = row.get("top_k", "")
        top_k = int(raw_topk) if raw_topk != "" else 10
        raw_floor = row.get("archival_floor", "0.65")
        archival_floor: float | None = (
            None if str(raw_floor).strip() in ("", "None", "none")
            else float(raw_floor)
        )
        return StudyRunResult(
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            embedding_model=row.get("embedding_model", ""),
            embedding_backend=row.get("embedding_backend", ""),
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            reranker_model=row.get("reranker_model", "none"),
            study_phase=row.get("study_phase", "general"),
            top_k=top_k,
            archival_floor=archival_floor,
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["study"] = {
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "bm25_weight": self.bm25_weight,
            "semantic_weight": self.semantic_weight,
            "reranker_model": self.reranker_model,
            "study_phase": self.study_phase,
            "top_k": self.top_k,
            "archival_floor": self.archival_floor,
        }
        return d


# Strategies safe to run concurrently in the ThreadPoolExecutor.
# Criterion: must NOT load torch/sentence-transformers (those are not thread-safe
# on MPS/CUDA — multiple threads racing for the same device context causes corruption).
# "recency" = return K most-recent memories, pure Python, no model loading.
# "bm25"    = rank_bm25 + numpy, GIL is released, safe in threads.
# bm25l is also thread-safe: pure Python/numpy, no GPU.
# colbert and adaptive use sentence-transformers — must run in-process.
_STRATEGIES_SAFE_IN_THREADPOOL = {"bm25", "bm25l", "recency"}


class CellCheckpointer:
    """Appends each completed cell result to a live CSV immediately on completion.

    Thread-safe: BM25 cells run in a ThreadPoolExecutor so writes are locked.
    The file is created on the first completed cell; subsequent calls append.
    This means a crashed or interrupted run leaves a recoverable partial CSV.
    """

    _FIELDS = [
        "completed_at", "study_phase", "memory_type", "retrieval_strategy",
        "embedding_model", "embedding_backend", "bm25_weight", "reranker_model",
        "recall_at_k", "precision_at_k", "mrr", "ndcg",
        "latency_p50_ms", "latency_p90_ms", "duration_seconds", "peak_ram_mb",
        "success", "error_message",
    ]

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self._path = output_dir / f"progress_{run_id}.csv"
        self._lock = threading.Lock()
        self._header_written = False
        # Keep the file handle open for the lifetime of the run.
        # open()/close() on every write causes a Win32 FlushFileBuffers (~100ms) on
        # Windows; for a 500-cell sweep that adds ~50s of pure I/O overhead.
        # We flush after each write for crash-safety instead of relying on close().
        self._file = open(self._path, "a", newline="", encoding="utf-8", buffering=1)  # noqa: SIM115

    def __del__(self) -> None:
        try:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
        except Exception:
            pass

    @property
    def path(self) -> Path:
        return self._path

    def on_cell_done(self, completed: int, total: int, result: StudyRunResult) -> None:
        row = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "study_phase": result.study_phase,
            "memory_type": result.memory_type,
            "retrieval_strategy": result.retrieval_strategy,
            "embedding_model": result.embedding_model,
            "embedding_backend": result.embedding_backend,
            "bm25_weight": result.bm25_weight,
            "reranker_model": result.reranker_model,
            "recall_at_k": round(result.recall_at_k, 4),
            "precision_at_k": round(result.precision_at_k, 4),
            "mrr": round(result.mrr, 4),
            "ndcg": round(result.ndcg, 4),
            "latency_p50_ms": round(result.latency_p50_ms, 2),
            "latency_p90_ms": round(result.latency_p90_ms, 2),
            "duration_seconds": round(result.duration_seconds, 2),
            "peak_ram_mb": round(result.peak_ram_mb, 1),
            "success": result.success,
            "error_message": next(iter((result.error_message or "").splitlines()), "")[:120],
        }
        with self._lock:
            writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
            self._file.flush()  # crash-safe: data on disk after each cell


def _needs_direct_execution(retrieval_strategy: str) -> bool:
    """Return True for strategies that must run in-process (not in the thread pool).

    Embedding-based strategies (semantic, hybrid, llm_rerank) load torch models
    onto MPS/CUDA. Multiple threads competing for the same device context cause
    corruption — these must run sequentially in the main process.
    BM25 and recency are pure Python/numpy and are thread-safe.
    """
    return retrieval_strategy not in _STRATEGIES_SAFE_IN_THREADPOOL


class StudyScheduler:
    """Runs StudyCells in parallel using the study worker.

    Strategies that use sentence-transformers or API embeddings (embeddings,
    hybrid, api_embeddings, llm_rerank) are run in-process sequentially to
    avoid a SIGSEGV in loky spawn workers on Python 3.13 / macOS ARM.
    BM25-only cells still run in parallel.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        output_dir: str = "data/output",
        run_id: str | None = None,
    ):
        _cpu = os.cpu_count() or 4
        _max_safe = max(1, _cpu - 1)  # leave 1 core for OS scheduler
        # Use all available cores by default — BM25/recency don't touch the GPU
        # so there is no reason to reserve cores for GPU headroom.
        # Hard cap at cpu_count - 1 regardless of what the caller requests.
        requested = max_workers or _hw_worker_count()
        self._max_workers = min(requested, _max_safe)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.normalization_meta: dict = {"applied": False}
        _rid = run_id or uuid.uuid4().hex[:12]
        self.checkpointer = CellCheckpointer(self._output_dir, _rid)

    def run(
        self,
        cells: list,
        gold_dataset_path: str,
        evaluation_horizon: int,
        progress_callback=None,
    ) -> list[StudyRunResult]:
        # Use the built-in checkpointer unless the caller supplies its own callback.
        if progress_callback is None:
            progress_callback = self.checkpointer.on_cell_done

        total = len(cells)
        results: list[StudyRunResult] = []
        cell_dicts = [c.to_summary_dict() for c in cells]

        # Load and normalise the gold dataset ONCE for all cells in this phase.
        # Without this, every cell re-parses the 2.7 MB JSON + runs timestamp
        # normalisation + validation — wasted work on every one of 96 cells.
        _gold_cache: dict = {}
        try:
            from benchmark.gold.oracle import GoldOracle
            _oracle = GoldOracle()
            _gold_cache["dataset"] = _oracle.load_dataset(
                Path(gold_dataset_path),
                scenario_name="study",
            )
            self.normalization_meta = _oracle.get_normalization_metadata("study")
        except Exception:
            pass  # fall back to per-cell loading if oracle unavailable

        # BM25-only cells run in a ThreadPoolExecutor (safe: no torch, no CUDA context).
        # ThreadPool avoids the CUDA-context-in-spawn crash while still parallelising
        # pure-Python BM25 work (the GIL releases during numpy/rank-bm25 operations).
        # Embedding/hybrid cells run in-process sequentially (they own the GPU).
        parallel_dicts = [cd for cd in cell_dicts if not _needs_direct_execution(cd.get("retrieval_strategy", "bm25"))]
        direct_dicts   = [cd for cd in cell_dicts if _needs_direct_execution(cd.get("retrieval_strategy", "bm25"))]

        # Cap thread workers at actual parallel work — no point spawning more
        # threads than there are BM25 cells to run.
        effective_workers = min(self._max_workers, len(parallel_dicts)) if parallel_dicts else 1

        mode_desc = []
        if parallel_dicts:
            mode_desc.append(
                f"{len(parallel_dicts)} parallel (BM25/recency, {effective_workers} threads)"
            )
        if direct_dicts:
            mode_desc.append(f"{len(direct_dicts)} sequential (embedding/hybrid)")
        print(
            f"  Running {total} study cells: {', '.join(mode_desc)} "
            f"(platform: {sys.platform})"
        )

        completed = 0
        _cell_times: list[float] = []  # track per-cell durations for ETA

        # ── In-process path for embedding/hybrid strategies ───────────────────
        for cell_dict in direct_dicts:
            completed += 1
            _t_cell = time.monotonic()
            try:
                result_dict = _run_study_cell_worker(
                    cell_dict, gold_dataset_path, str(self._output_dir), evaluation_horizon,
                    gold_dataset_cache=_gold_cache,
                )
                if not self.normalization_meta.get("applied", False):
                    norm = result_dict.get("_timestamp_normalization", {})
                    if norm.get("applied", False):
                        self.normalization_meta = norm
                result = self._dict_to_result(result_dict)
            except Exception as e:
                result = self._make_error_result(cell_dict, str(e))

            _cell_times.append(time.monotonic() - _t_cell)
            results.append(result)
            self._print_result(completed, total, result, _cell_times, direct_dicts)
            if progress_callback:
                progress_callback(completed, total, result)

        if not parallel_dicts:
            return results

        # ── Thread pool for BM25 + recency strategies ─────────────────────────
        # Threads are correct here: rank_bm25 and numpy release the GIL during
        # computation so multiple cells genuinely run in parallel. ProcessPoolExecutor
        # would bypass the GIL entirely but macOS spawn costs ~2s per process —
        # with only 3-6 cells that overhead is larger than the GIL savings.
        # CPU utilisation will look modest (~20-30% per active core) because the
        # bottleneck is memory bandwidth (indexing + querying 5879 docs), not FLOPS.
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import as_completed as tpas_completed

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _run_study_cell_worker,
                    cell_dict,
                    gold_dataset_path,
                    str(self._output_dir),
                    evaluation_horizon,
                    _gold_cache,
                ): cell_dict
                for cell_dict in parallel_dicts
            }

            for future in tpas_completed(futures):
                completed += 1
                cell_dict = futures[future]
                try:
                    result_dict = future.result(timeout=300)
                    if not self.normalization_meta.get("applied", False):
                        norm = result_dict.get("_timestamp_normalization", {})
                        if norm.get("applied", False):
                            self.normalization_meta = norm
                    result = self._dict_to_result(result_dict)
                except Exception as e:
                    import traceback as _tb
                    result = self._make_error_result(
                        cell_dict, f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
                    )

                results.append(result)
                self._print_result(completed, total, result)
                if progress_callback:
                    progress_callback(completed, total, result)

        return results

    def _make_error_result(self, cell_dict: dict, error: str) -> StudyRunResult:
        bm25w = cell_dict.get("bm25_weight", 0.0)
        return StudyRunResult(
            cell_id=cell_dict.get("cell_id", "unknown"),
            run_id=uuid.uuid4().hex[:12],
            memory_type=cell_dict.get("memory_type", "?"),
            retrieval_strategy=cell_dict.get("retrieval_strategy", "?"),
            decay_policy=cell_dict.get("decay_policy", "?"),
            lambda_value=cell_dict.get("lambda", 0.0),
            pruning_threshold=cell_dict.get("pruning_threshold", 0.3),
            workload_profile=cell_dict.get("workload_profile", "?"),
            seed=cell_dict.get("seed", 42),
            embedding_model=cell_dict.get("embedding_model", ""),
            embedding_backend=cell_dict.get("embedding_backend", ""),
            bm25_weight=bm25w,
            semantic_weight=round(1.0 - bm25w, 2),
            reranker_model=cell_dict.get("reranker_model", "none"),
            study_phase=cell_dict.get("study_phase", "general"),
            top_k=cell_dict.get("top_k", 10),
            archival_floor=cell_dict.get("archival_floor", 0.65),
            success=False,
            error_message=error,
        )

    def _print_result(
        self,
        completed: int,
        total: int,
        result: StudyRunResult,
        cell_times: list[float] | None = None,
        all_cells: list | None = None,
    ) -> None:
        status = "✓" if result.success else "✗"
        uses_embedding = result.retrieval_strategy not in ("bm25",)
        if uses_embedding:
            _backend = result.embedding_backend or ""
            _tag = {"sentence-transformers": "local", "api": "api", "none": ""}.get(_backend, _backend)
            _model = (result.embedding_model or "—").split("/")[-1]
            embed_str = f"embed={_model}[{_tag}]".ljust(27) + " "
        else:
            embed_str = " " * 28

        # ETA based on rolling average of completed cell times
        eta_str = ""
        if cell_times and len(cell_times) >= 1:
            avg_sec = sum(cell_times) / len(cell_times)
            remaining = total - completed
            eta_sec = avg_sec * remaining
            if eta_sec > 0:
                eta_str = f"  ETA ~{eta_sec:.0f}s"

        print(
            f"  [{completed:3d}/{total}] {status} "
            f"{result.memory_type:12s} × {result.retrieval_strategy:16s} "
            f"{embed_str}"
            f"recall={result.recall_at_k:.3f} mrr={result.mrr:.3f} "
            f"p50={result.latency_p50_ms:.1f}ms"
            f"{eta_str}"
        )
        if not result.success and result.error_message:
            first_line = result.error_message.strip().splitlines()[-1]
            print(f"           ERROR: {first_line}")

    @staticmethod
    def _dict_to_result(d: dict) -> StudyRunResult:
        from benchmark.workload.scheduler import MatrixScheduler

        base = MatrixScheduler._dict_to_result(d)
        study = d.get("_study", {})

        result = StudyRunResult(
            cell_id=base.cell_id,
            run_id=base.run_id,
            memory_type=base.memory_type,
            retrieval_strategy=base.retrieval_strategy,
            decay_policy=base.decay_policy,
            lambda_value=base.lambda_value,
            pruning_threshold=base.pruning_threshold,
            workload_profile=base.workload_profile,
            seed=base.seed,
            recall_at_k=base.recall_at_k,
            contamination_rate=base.contamination_rate,
            precision_at_k=base.precision_at_k,
            temporal_accuracy=base.temporal_accuracy,
            module_accuracy=base.module_accuracy,
            mrr=base.mrr,
            ndcg=base.ndcg,
            precision_at_1=base.precision_at_1,
            total_queries=base.total_queries,
            correct_recalls=base.correct_recalls,
            peak_ram_mb=base.peak_ram_mb,
            avg_ram_mb=base.avg_ram_mb,
            peak_cpu_percent=base.peak_cpu_percent,
            avg_cpu_percent=base.avg_cpu_percent,
            duration_seconds=base.duration_seconds,
            disk_write_mb=base.disk_write_mb,
            latency_p50_ms=base.latency_p50_ms,
            latency_p90_ms=base.latency_p90_ms,
            latency_p99_ms=base.latency_p99_ms,
            latency_mean_ms=base.latency_mean_ms,
            total_cost=base.total_cost,
            success=base.success,
            error_message=base.error_message,
            platform=base.platform,
            resolved_retriever_class=base.resolved_retriever_class,
            embedding_model=study.get("embedding_model", ""),
            embedding_backend=study.get("embedding_backend", ""),
            bm25_weight=study.get("bm25_weight", 0.0),
            semantic_weight=study.get("semantic_weight", 0.0),
            reranker_model=study.get("reranker_model", "none"),
            study_phase=study.get("study_phase", "general"),
            top_k=study.get("top_k", 10),
            archival_floor=study.get("archival_floor", 0.65),
        )
        return result
