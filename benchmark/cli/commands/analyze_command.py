"""Analyze command — full benchmark analysis with plots in one command.

Runs all strategies, sweeps all parameters, generates comparison plots.
This is the primary deliverable of the benchmark tool.

Usage:
    memtuner analyze -d data/input/locomo10.json
    memtuner analyze -d data/input/locomo10.json --with-llm-judge
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import click

from benchmark.analysis.multi_agent import run_interference_test
from benchmark.application.composer import BenchmarkComposer
from benchmark.cli.provenance import (
    build_run_metadata,
    collect_environment_metadata,
    serialize_pack_metadata,
)
from benchmark.config.schema import BenchmarkConfig
from benchmark.gold.oracle import GoldOracle
from benchmark.memory.long_term.entity_store import EntityStore
from benchmark.memory.long_term.episodic_store import EpisodicStore
from benchmark.memory.long_term.preference_store import PreferenceStore
from benchmark.memory.long_term.semantic_store import SemanticStore
from benchmark.observability.logger import get_logger, log_decision
from benchmark.resources.tracker import ResourceTracker

logger = get_logger(__name__)


def _parse_model_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _strategy_allowlist() -> set[str] | None:
    configured = _parse_model_list(os.environ.get("BENCHMARK_ANALYZE_STRATEGIES"))
    if not configured:
        return None
    return {strategy for strategy in configured}


def _memory_type_allowlist() -> set[str] | None:
    configured = _parse_model_list(os.environ.get("BENCHMARK_ANALYZE_MEMORY_TYPES"))
    if not configured:
        return None
    return {memory_type for memory_type in configured}


def _is_truthy_env(name: str) -> bool:
    raw_value = os.environ.get(name, "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _label_for_model(model_id: str) -> str:
    labels = {
        "all-MiniLM-L6-v2": "MiniLM (22M, 384d)",
        "BAAI/bge-base-en-v1.5": "BGE-Base (110M, 768d)",
        "BAAI/bge-m3": "BGE-M3 (1.5B, 1024d)",
    }
    return labels.get(model_id, f"Custom ({model_id})")

def _analysis_defaults_config(pack: str | None):
    del pack  # per-pack config files (locomo.yaml, longmemeval.yaml) were removed;
    # this only needs default provider settings, which BenchmarkConfig's own
    # pydantic defaults already supply — no file to keep in sync with the schema.
    return BenchmarkConfig()


def _resolve_pack_data_dir(pack_name: str, pack_instance, explicit_data_dir: str | None) -> Path:
    if explicit_data_dir:
        return Path(explicit_data_dir)

    candidate_dir = Path("data") / pack_name
    if pack_instance.validate_data(candidate_dir):
        return candidate_dir

    fallback_dir = Path("data")
    if pack_instance.validate_data(fallback_dir):
        return fallback_dir

    return candidate_dir


def _embedding_candidates(local_size_threshold_mb: int = 100) -> list[tuple[str, str]]:
    from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy

    embedding_models_env = _parse_model_list(os.environ.get("BENCHMARK_EMBEDDING_MODELS"))
    if embedding_models_env:
        filtered: list[tuple[str, str]] = []
        for model_id in embedding_models_env:
            size_mb = EmbeddingsStrategy.known_model_size_mb(model_id)
            if size_mb is None or size_mb <= local_size_threshold_mb:
                filtered.append((model_id, _label_for_model(model_id)))
        return filtered

    default_model = os.environ.get("BENCHMARK_EMBEDDING_MODEL")
    if default_model:
        size_mb = EmbeddingsStrategy.known_model_size_mb(default_model)
        if size_mb is None or size_mb <= local_size_threshold_mb:
            return [(default_model, _label_for_model(default_model))]

    return []


def _is_local_environment_block(exc_message: str) -> bool:
    lowered = exc_message.lower()
    return "operation not permitted" in lowered or "permission denied" in lowered


def _api_embeddings_available() -> bool:
    """Return True if the api_embeddings strategy can be used.

    Requires: openai package installed + BENCHMARK_OPENAI_BASE_URL set.
    """
    if os.environ.get("BENCHMARK_OPENAI_BASE_URL"):
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            pass
    return False


def _api_embeddings_models() -> list[str]:
    """Return API embedding model candidates from env."""
    models_env = _parse_model_list(os.environ.get("BENCHMARK_OPENAI_EMBEDDING_MODELS"))
    if models_env:
        return models_env
    default = os.environ.get("BENCHMARK_OPENAI_EMBEDDING_MODEL")
    if default:
        return [default]
    return []


def _validated_provider_settings(config) -> dict[str, Any]:
    retrieval_config = config.benchmark.retrieval
    reranker_config = config.benchmark.reranker

    api_settings: dict[str, Any] = {}
    if retrieval_config.api_embeddings.model_name:
        api_settings["model_name"] = retrieval_config.api_embeddings.model_name
    if retrieval_config.api_embeddings.base_url:
        api_settings["base_url"] = retrieval_config.api_embeddings.base_url
    if retrieval_config.api_embeddings.api_key:
        api_settings["api_key"] = retrieval_config.api_embeddings.api_key
    api_settings["timeout"] = retrieval_config.api_embeddings.timeout
    api_settings["batch_size"] = retrieval_config.api_embeddings.batch_size

    return {
        "api_embeddings": api_settings,
        "reranker": {
            "strategy": reranker_config.strategy,
            "model_name": reranker_config.model_name,
            "api_provider_order": reranker_config.api_provider_order,
            "local_size_threshold_mb": reranker_config.local_size_threshold_mb,
        },
    }


def _strategy_retrieval_overrides(
    strategy_name: str,
) -> dict[str, Any] | None:
    if strategy_name == "embeddings":
        embedding_models = _embedding_candidates()
        if not embedding_models:
            return None
        return {"embeddings": {"model_name": embedding_models[0][0]}}

    if strategy_name == "hybrid":
        embedding_models = _embedding_candidates()
        if not embedding_models:
            return None
        return {
            "hybrid": {
                "strategies": ["bm25", "embeddings"],
                "confidence_threshold": 0.5,
                "bm25_weight": 0.5,
            },
            "embeddings": {"model_name": embedding_models[0][0]},
        }

    if strategy_name == "api_embeddings":
        api_models = _api_embeddings_models()
        base_url = os.environ.get("BENCHMARK_OPENAI_BASE_URL")
        if not api_models or not base_url:
            return None
        overrides: dict[str, Any] = {
            "model_name": api_models[0],
            "base_url": base_url,
        }
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            overrides["api_key"] = api_key
        return {"api_embeddings": overrides}

    return None


def _parse_comparison_models(raw: str | None, defaults: list[str]) -> list[str]:
    configured = _parse_model_list(raw)
    return configured or list(defaults)


def _log_provider_candidates(provider: str, models: list[str], base_url: str | None = None) -> None:
    log_decision(
        logger,
        "Embedding comparison candidates resolved",
        provider=provider,
        models=models,
        base_url=base_url,
    )


def _write_tagged_json(output_dir: Path, tag: str, payload: dict[str, Any]) -> str:
    path = output_dir / f"{tag}.json"
    with path.open("w") as fh:
        import json

        json.dump(payload, fh, indent=2)
    return str(path)


def _artifact_entry(tag: str, artifact_type: str, path: str, description: str) -> dict[str, str]:
    return {
        "tag": tag,
        "type": artifact_type,
        "path": path,
        "description": description,
    }


@click.command("analyze")
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=False,
    help="Gold dataset JSON (required unless --pack is used)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/output",
    help="Output directory for plots and report",
)
@click.option("--pack", type=click.Choice(["longmemeval", "locomo", "private"]), default=None)
@click.option("--data-dir", type=click.Path(exists=True), default=None)
@click.option("--max-queries", type=int, default=None, help="Limit queries for faster runs")
@click.option(
    "--with-llm-judge",
    is_flag=True,
    default=False,
    help="Enable LLM-as-judge (requires BENCHMARK_LLM_BASE_URL)",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
def analyze_benchmark(
    dataset: str,
    output: str,
    pack: str | None,
    data_dir: str | None,
    max_queries: int | None,
    with_llm_judge: bool,
    seed: int,
) -> None:
    """Run full benchmark analysis: all strategies, parameter sweeps, and plots.

    Produces:
    - Strategy comparison (BM25 vs Embeddings vs Hybrid)
    - Decay parameter sweep with plots
    - Memory type comparison
    - Scaling analysis
    - Multi-agent isolation test
    - (Optional) LLM-as-judge end-to-end evaluation

    All results saved as JSON + PNG plots in the output directory.
    """
    import json
    import logging
    import math
    import time
    from datetime import UTC, datetime

    logging.disable(logging.INFO)

    from benchmark.config.loader import load_config_from_dict
    from benchmark.memory.strategies.bm25_strategy import BM25Strategy
    from benchmark.models.memory_event import MemoryEvent
    from benchmark.models.query import ReadQuery, ReadQueryContext

    strategy_results: list[dict[str, Any]] = []
    strategy_failures: list[dict[str, str]] = []
    embedding_model_results: list[dict[str, Any]] = []
    api_embedding_model_results: list[dict[str, Any]] = []
    reranker_model_results: list[dict[str, Any]] = []
    memory_results: list[dict[str, Any]] = []
    decay_results: list[dict[str, Any]] = []
    artifact_manifest: list[dict[str, str]] = []
    provider_failures: dict[str, list[dict[str, str]]] = {
        "strategy_comparison": strategy_failures,
        "api_embedding_model_comparison": [],
        "embedding_model_comparison": [],
        "reranker_model_comparison": [],
        "analyze_runtime": [],
    }
    first_run_plan = None
    first_config_snapshot = None
    decay_response = {"formula": "exp(-lambda * age_days)", "age_days": [], "curves": []}
    interference = None
    gold_dataset = None
    pack_metadata = None
    resolved_dataset_path: str | None = dataset
    resource_report = None
    effective_config_path: str | None = None
    environment_path: str | None = None
    run_metadata_path: str | None = None

    def _ensure_partial_protocol_artifacts() -> tuple[str | None, str | None, str | None]:
        nonlocal effective_config_path, environment_path, run_metadata_path

        if first_config_snapshot is not None and effective_config_path is None:
            effective_config_path = _write_tagged_json(
                output_dir,
                "effective_config",
                first_config_snapshot,
            )
            artifact_manifest.append(
                _artifact_entry(
                    "effective_config",
                    "json",
                    effective_config_path,
                    "Effective validated benchmark config used for the first successful composed analysis run.",
                )
            )

        if environment_path is None:
            environment_path = _write_tagged_json(
                output_dir,
                "environment",
                collect_environment_metadata(),
            )
            artifact_manifest.append(
                _artifact_entry(
                    "environment",
                    "json",
                    environment_path,
                    "Execution environment snapshot including Python, platform, and git revision details when available.",
                )
            )

        if run_metadata_path is None:
            run_metadata_path = _write_tagged_json(
                output_dir,
                "run_metadata",
                build_run_metadata(
                    command_name="memtuner analyze",
                    dataset_path=resolved_dataset_path,
                    output_dir=output_dir,
                    pack_name=pack,
                    pack_metadata=pack_metadata,
                    max_queries=max_queries,
                    seed=seed,
                    with_llm_judge=with_llm_judge,
                    run_plan=first_run_plan,
                    resource_report=resource_report,
                ),
            )
            artifact_manifest.append(
                _artifact_entry(
                    "run_metadata",
                    "json",
                    run_metadata_path,
                    "Protocol-oriented run metadata including dataset fingerprint, run plan, pack identity, and resource usage summary.",
                )
            )

        return effective_config_path, environment_path, run_metadata_path

    def _write_partial_report(runtime_error: str) -> None:
        if gold_dataset is None:
            return
        partial_effective_config_path, partial_environment_path, partial_run_metadata_path = (
            _ensure_partial_protocol_artifacts()
        )
        report = {
            "dataset": {
                "scenario": gold_dataset.scenario,
                "queries": len(gold_dataset.queries),
                "memories": gold_dataset.total_conversation_turns,
                "users": len(gold_dataset.user_ids),
            },
            "run_plan": first_run_plan,
            "effective_config_artifact": partial_effective_config_path,
            "environment_artifact": partial_environment_path,
            "run_metadata_artifact": partial_run_metadata_path,
            "strategy_comparison": strategy_results,
            "embedding_model_comparison": embedding_model_results,
            "api_embedding_model_comparison": api_embedding_model_results,
            "reranker_model_comparison": reranker_model_results,
            "provider_failures": provider_failures,
            "memory_type_comparison": memory_results,
            "decay_sweep": decay_results,
            "decay_response": decay_response,
            "isolation": (
                {"rate": interference.isolation_rate, "leaks": interference.leaked_queries}
                if interference is not None
                else None
            ),
            "artifact_manifest": artifact_manifest,
            "llm_judge": {
                "enabled": with_llm_judge,
                "method": "llm_judge" if with_llm_judge else None,
            },
            "status": "failed",
            "runtime_error": runtime_error,
        }
        report_path = output_dir / "benchmark_report.json"
        with report_path.open("w") as fh:
            json.dump(report, fh, indent=2)

    judge_evaluator = None
    if with_llm_judge:
        from benchmark.judge.evaluator import EndToEndEvaluator

        judge_evaluator = EndToEndEvaluator(judge_method="llm_judge")
        if not judge_evaluator.is_llm_available():
            raise click.ClickException(
                "LLM endpoint is unavailable. Set BENCHMARK_LLM_BASE_URL and "
                "verify the OpenAI-compatible /models endpoint."
            )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analyze_startup_sentinel.txt").write_text("analyze-started\n", encoding="utf-8")
    progress_path = output_dir / "analyze_progress.log"

    def _append_progress(message: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        with progress_path.open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(f"{timestamp} {message}\n")

    _append_progress("startup")
    analysis_defaults = _analysis_defaults_config(pack)
    validated_provider_settings = _validated_provider_settings(analysis_defaults)

    try:

        # Load dataset
        click.echo("📊 Loading dataset...")
        _append_progress("dataset_load_start")
        gold_dataset = None
        if pack:
            from benchmark.packs.registry import PackRegistry

            pack_instance = PackRegistry.get(pack)
            resolved_dir = _resolve_pack_data_dir(pack, pack_instance, data_dir)
            resolved_dataset_path = str(resolved_dir)
            pack_instance.load(resolved_dir)
            pack_metadata = serialize_pack_metadata(pack_instance.metadata())
            gold_dataset = pack_instance.to_gold_dataset(
                max_queries=max_queries, seed=seed, evaluation_horizon=100
            )
        else:
            if not dataset:
                raise click.UsageError("--dataset is required unless --pack is used")
            oracle = GoldOracle()
            gold_dataset = oracle.load_dataset(Path(dataset))

        if with_llm_judge and not any(query.gold_answer for query in gold_dataset.queries):
            raise click.ClickException(
                "LLM judging requires gold_answer on at least one dataset query. "
                "Add it to the gold dataset or use a supported pack."
            )

        click.echo(
            f"   Queries: {len(gold_dataset.queries)}, "
            f"Memories: {gold_dataset.total_conversation_turns}"
        )
        _append_progress("dataset_loaded")

        _ensure_partial_protocol_artifacts()

        def _base_config(strategy_name: str, retrieval_overrides: dict | None = None) -> dict:
            benchmark_config = {
                "evaluation_horizon": 30,
                "seed": seed,
                "scenarios": ["delayed_recall"],
                "retrieval_strategy": strategy_name,
            }
            if retrieval_overrides:
                benchmark_config["retrieval"] = retrieval_overrides

            return {
                "memory": {"enabled": {"short_term": [], "long_term": ["episodic_store"]}},
                "policies": {
                    "module_policies": {
                        "episodic_store": {
                            "decay": {"type": "exponential", "lambda": 0.0},
                            "pruning": {"strategy": "score_threshold", "threshold": 0.01},
                        }
                    }
                },
                "benchmark": benchmark_config,
                "observability": {
                    "exporter": "none",
                    "endpoint": "http://localhost:4317",
                    "log_level": "ERROR",
                },
                "answering": {"enabled": False, "model": "", "max_tokens": 500},
            }

        # =========================================================================
        # 1. STRATEGY COMPARISON
        # =========================================================================
        click.echo("\n🔬 1/5 Strategy Comparison...")
        _append_progress("strategy_comparison_start")

        strategies_available = ["bm25"]
        strategy_allowlist = _strategy_allowlist()
        allowlist_only_bm25 = (
            strategy_allowlist is not None and strategy_allowlist.issubset({"bm25"})
        )
        if not allowlist_only_bm25:
            try:
                from benchmark.memory.strategies.embeddings_strategy import EmbeddingsStrategy

                if EmbeddingsStrategy.is_available() and _embedding_candidates():
                    strategies_available.append("embeddings")
                    strategies_available.append("hybrid")
            except Exception:
                pass
            try:
                from benchmark.memory.strategies.llm_rerank_strategy import LLMRerankStrategy

                if LLMRerankStrategy.is_available():
                    strategies_available.append("llm_rerank")
            except Exception:
                pass

            if _api_embeddings_available() and _strategy_retrieval_overrides("api_embeddings"):
                strategies_available.append("api_embeddings")

        if strategy_allowlist is not None:
            strategies_available = [
                strategy_name
                for strategy_name in strategies_available
                if strategy_name in strategy_allowlist
            ]
            if not strategies_available:
                raise click.ClickException(
                    "BENCHMARK_ANALYZE_STRATEGIES filtered out all analyze strategies"
                )

        with ResourceTracker() as resource_tracker:
            for strategy_name in strategies_available:
                _append_progress(f"strategy_start:{strategy_name}")
                retrieval_overrides = _strategy_retrieval_overrides(strategy_name)
                config_dict = _base_config(strategy_name, retrieval_overrides)
                try:
                    config = load_config_from_dict(config_dict)
                    _append_progress(f"strategy_config_loaded:{strategy_name}")
                    composer = BenchmarkComposer()
                    _append_progress(f"strategy_composer_ready:{strategy_name}")
                    start = time.monotonic()
                    composed = composer.compose(
                        config=config,
                        dataset_override=gold_dataset,
                        answer_evaluator=judge_evaluator,
                        allow_strategy_fallback=True,
                    )
                    _append_progress(f"strategy_composed:{strategy_name}")
                    result = composed.runner.run(composed.scenarios)
                    _append_progress(f"strategy_run_done:{strategy_name}")
                    elapsed = time.monotonic() - start
                    if first_run_plan is None:
                        first_run_plan = composed.run_plan.to_dict()
                        first_config_snapshot = config.model_dump(mode="json")
                    sr = result.scenario_results[0]
                    entry = {
                        "strategy": strategy_name,
                        "recall": sr.recall_at_k,
                        "precision": sr.precision_at_k,
                        "contamination": sr.contamination_rate,
                        "temporal": sr.temporal_accuracy,
                        "mrr": sr.mrr,
                        "ndcg": sr.ndcg,
                        "time": elapsed,
                        "ms_per_query": elapsed * 1000 / max(sr.total_queries, 1),
                    }
                    if with_llm_judge:
                        entry["llm_judge_score"] = sr.llm_judge_score
                        entry["llm_judge_queries"] = sr.llm_judge_queries
                    strategy_results.append(entry)
                    _append_progress(f"strategy_done:{strategy_name}")
                    click.echo(
                        f"   {strategy_name:<15} Recall={sr.recall_at_k:.1%} "
                        f"Prec={sr.precision_at_k:.1%} ({elapsed:.1f}s)"
                    )
                except Exception as exc:
                    strategy_failures.append(
                        {
                            "strategy": strategy_name,
                            "error": str(exc),
                        }
                    )
                    _append_progress(f"strategy_failed:{strategy_name}:{exc}")
                    click.echo(f"   {strategy_name:<15} FAILED: {exc}")

            resource_report = resource_tracker.report()

        if with_llm_judge and not strategy_results:
            raise click.ClickException(
                "LLM judging produced no successful strategy results. "
                "Check BENCHMARK_LLM_MODEL and endpoint logs."
            )

        # =========================================================================
        # 1b. API EMBEDDING MODEL COMPARISON (if BENCHMARK_OPENAI_BASE_URL set)
        # =========================================================================
        if _api_embeddings_available() and not _is_truthy_env(
            "BENCHMARK_ANALYZE_SKIP_API_EMBEDDING_COMPARISON"
        ):
            _append_progress("section_start:api_embedding_comparison")
            api_models = _api_embeddings_models()
            click.echo("\n🔬 1b. API Embedding Model Comparison...")
            if not api_models:
                click.echo("   SKIPPED: no models configured (set BENCHMARK_OPENAI_EMBEDDING_MODELS)")
            for model_id in api_models:
                try:
                    api_overrides: dict[str, Any] = {"model_name": model_id}
                    base_url = os.environ.get("BENCHMARK_OPENAI_BASE_URL")
                    if base_url:
                        api_overrides["base_url"] = base_url
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if api_key:
                        api_overrides["api_key"] = api_key
                    config_dict = _base_config("api_embeddings", {"api_embeddings": api_overrides})
                    config = load_config_from_dict(config_dict)
                    composer = BenchmarkComposer()
                    start = time.monotonic()
                    composed = composer.compose(
                        config=config,
                        dataset_override=gold_dataset,
                        allow_strategy_fallback=True,
                    )
                    result = composed.runner.run(composed.scenarios)
                    elapsed = time.monotonic() - start
                    sr = result.scenario_results[0]
                    entry = {
                        "model": model_id,
                        "label": model_id,
                        "recall": sr.recall_at_k,
                        "precision": sr.precision_at_k,
                        "mrr": sr.mrr,
                        "ndcg": sr.ndcg,
                        "contamination": sr.contamination_rate,
                        "ms_per_query": elapsed * 1000 / max(sr.total_queries, 1),
                    }
                    api_embedding_model_results.append(entry)
                    click.echo(
                        f"   {model_id:<40} Recall={sr.recall_at_k:.1%} "
                        f"MRR={sr.mrr:.3f} ({elapsed:.1f}s)"
                    )
                except Exception as exc:
                    provider_failures["api_embedding_model_comparison"].append(
                        {"model": model_id, "error": str(exc)}
                    )
                    click.echo(f"   {model_id:<40} FAILED: {exc}")

        # =========================================================================
        # 1c. LOCAL EMBEDDING MODEL COMPARISON (if sentence-transformers available)
        # =========================================================================
        embedding_models: list[tuple[str, str]] = []
        local_embedding_skip = _is_truthy_env("BENCHMARK_ANALYZE_SKIP_LOCAL_EMBEDDING_COMPARISON")
        if not local_embedding_skip and find_spec("sentence_transformers") is not None:
            local_size_threshold_mb = (
                analysis_defaults.benchmark.reranker.local_size_threshold_mb
            )
            embedding_models = _embedding_candidates(
                local_size_threshold_mb=local_size_threshold_mb
            )

        if (
            not local_embedding_skip
            and find_spec("sentence_transformers") is not None
            and embedding_models
        ):

            _append_progress("section_start:local_embedding_comparison")

            click.echo("\n🔬 1c. Local Embedding Model Comparison...")
            for model_id, model_label in embedding_models:
                try:
                    config_dict = _base_config(
                        "embeddings",
                        {
                            "embeddings": {
                                "model_name": model_id,
                                "cache_dir": analysis_defaults.benchmark.retrieval.embeddings.cache_dir,
                            }
                        },
                    )
                    config = load_config_from_dict(config_dict)
                    composer = BenchmarkComposer()
                    start = time.monotonic()
                    composed = composer.compose(
                        config=config,
                        dataset_override=gold_dataset,
                        allow_strategy_fallback=True,
                    )
                    result = composed.runner.run(composed.scenarios)
                    elapsed = time.monotonic() - start
                    sr = result.scenario_results[0]
                    entry = {
                        "model": model_id,
                        "label": model_label,
                        "recall": sr.recall_at_k,
                        "precision": sr.precision_at_k,
                        "mrr": sr.mrr,
                        "ndcg": sr.ndcg,
                        "contamination": sr.contamination_rate,
                        "ms_per_query": elapsed * 1000 / max(sr.total_queries, 1),
                    }
                    embedding_model_results.append(entry)
                    click.echo(
                        f"   {model_label:<25} Recall={sr.recall_at_k:.1%} "
                        f"MRR={sr.mrr:.3f} ({elapsed:.1f}s)"
                    )
                except Exception as exc:
                    provider_failures["embedding_model_comparison"].append(
                        {
                            "model": model_id,
                            "label": model_label,
                            "error": str(exc),
                        }
                    )
                    click.echo(f"   {model_label:<25} FAILED: {exc}")

        # =========================================================================
        # 1d. RERANKER MODEL COMPARISON
        # =========================================================================
        reranker_model = analysis_defaults.benchmark.reranker.model_name
        if reranker_model and not _is_truthy_env("BENCHMARK_ANALYZE_SKIP_RERANKER_COMPARISON"):
            _append_progress("section_start:reranker_comparison")
            click.echo("\n🔬 1d. Reranker Comparison...")
            reranker_strategy = analysis_defaults.benchmark.reranker.strategy
            reranker_provider_order = analysis_defaults.benchmark.reranker.api_provider_order
            try:
                rerank_overrides: dict[str, Any] = {"bm25": {}}
                config_dict = _base_config("llm_rerank", rerank_overrides)
                config_dict["benchmark"]["reranker"] = dict(validated_provider_settings["reranker"])
                config = load_config_from_dict(config_dict)
                composer = BenchmarkComposer()
                _append_progress("reranker_model_testing:compose_start")
                start = time.monotonic()
                composed = composer.compose(
                    config=config,
                    dataset_override=gold_dataset,
                    allow_strategy_fallback=True,
                )
                _append_progress("reranker_model_testing:runner_start")
                result = composed.runner.run(composed.scenarios)
                _append_progress("reranker_model_testing:runner_done")
                elapsed = time.monotonic() - start
                sr = result.scenario_results[0]
                reranker_model_results.append(
                    {
                        "model": reranker_model,
                        "strategy": reranker_strategy,
                        "provider_order": reranker_provider_order,
                        "recall": sr.recall_at_k,
                        "precision": sr.precision_at_k,
                        "mrr": sr.mrr,
                        "ndcg": sr.ndcg,
                        "contamination": sr.contamination_rate,
                        "ms_per_query": elapsed * 1000 / max(sr.total_queries, 1),
                    }
                )
                click.echo(
                    f"   {reranker_model:<35} Recall={sr.recall_at_k:.1%} "
                    f"MRR={sr.mrr:.3f} ({elapsed:.1f}s)"
                )
            except Exception as exc:
                _append_progress(f"reranker_model_testing:failed:{type(exc).__name__}")
                provider_failures["reranker_model_comparison"].append(
                    {
                        "model": reranker_model,
                        "strategy": reranker_strategy,
                        "error": str(exc),
                    }
                )
                click.echo(f"   {reranker_model:<35} FAILED: {exc}")

        _append_progress("section_done:reranker_comparison")

        # =========================================================================
        # 2. MEMORY TYPE COMPARISON
        # =========================================================================
        if not _is_truthy_env("BENCHMARK_ANALYZE_SKIP_MEMORY_TYPE_COMPARISON"):
            click.echo("\n🧠 2/5 Memory Type Comparison...")
            _append_progress("section_start:memory_type_comparison")

        if _is_truthy_env("BENCHMARK_ANALYZE_SKIP_MEMORY_TYPE_COMPARISON"):
            memory_results = []
        else:
            memory_types = [
                ("episodic_store", EpisodicStore),
                ("entity_store", EntityStore),
                ("preference_store", PreferenceStore),
                ("semantic_store", SemanticStore),
            ]
            memory_type_allowlist = _memory_type_allowlist()
            if memory_type_allowlist is not None:
                memory_types = [
                    (module_name, store_class)
                    for module_name, store_class in memory_types
                    if module_name in memory_type_allowlist
                ]
            trace_memory_queries = _is_truthy_env(
                "BENCHMARK_ANALYZE_MEMORY_TYPE_QUERY_TRACE"
            )
            ts = datetime(2026, 1, 1, tzinfo=UTC)

            for module_name, store_class in memory_types:
                _append_progress(f"memory_type_start:{module_name}")
                try:
                    store = store_class(retrieval_strategy=BM25Strategy())
                    _append_progress(f"memory_type_store_created:{module_name}")
                    for de in gold_dataset.events:
                        for ge in de.memory_events:
                            mem = MemoryEvent(
                                id=ge.id,
                                user_id=ge.user_id,
                                type=ge.type,
                                content=ge.content,
                                timestamp=ts,
                                importance=ge.importance,
                                entities=ge.entities,
                                task_id=ge.task_id,
                            )
                            store.write_on_day(mem, de.day)
                    _append_progress(f"memory_type_populated:{module_name}")

                    recalls = []
                    _append_progress(f"memory_type_query_start:{module_name}")
                    for idx, q in enumerate(gold_dataset.queries):
                        if trace_memory_queries:
                            _append_progress(f"memory_type_query:{module_name}:{idx}")
                        query = ReadQuery(
                            query=q.query,
                            top_k=10,
                            context=ReadQueryContext(
                                dataset_day=q.day,
                                task_id=q.task_id,
                                user_id=q.user_id,
                            ),
                        )
                        response = store.read(query)
                        found = set(m.memory_id for m in response.retrieved_memories) & set(
                            q.expected.memory_ids
                        )
                        recalls.append(
                            len(found) / len(q.expected.memory_ids) if q.expected.memory_ids else 0
                        )
                    _append_progress(f"memory_type_query_done:{module_name}")

                    avg_recall = sum(recalls) / len(recalls)
                    memory_results.append(
                        {
                            "module": module_name,
                            "recall": avg_recall,
                            "memories_stored": store.count(),
                        }
                    )
                    _append_progress(f"memory_type_done:{module_name}")
                    click.echo(
                        f"   {module_name:<20} Recall={avg_recall:.1%} ({store.count()} stored)"
                    )
                except Exception as exc:
                    provider_failures["analyze_runtime"].append(
                        {"section": "memory_type_comparison", "module": module_name, "error": str(exc)}
                    )
                    _append_progress(f"memory_type_failed:{module_name}")
                    click.echo(f"   {module_name:<20} FAILED: {exc}")

        # =========================================================================
        # 3. DECAY PARAMETER SWEEP
        # =========================================================================
        if not _is_truthy_env("BENCHMARK_ANALYZE_SKIP_DECAY_SWEEP"):
            click.echo("\n⏳ 3/5 Decay + Pruning Sweep...")
            _append_progress("section_start:decay_sweep")

        if _is_truthy_env("BENCHMARK_ANALYZE_SKIP_DECAY_SWEEP"):
            decay_results = []
        else:
            for lam in [0.0, 0.01, 0.05, 0.10, 0.20]:
                for threshold in [0.01, 0.15, 0.35]:
                    config_dict = {
                        "memory": {"enabled": {"short_term": [], "long_term": ["episodic_store"]}},
                        "policies": {
                            "module_policies": {
                                "episodic_store": {
                                    "decay": {"type": "exponential", "lambda": lam},
                                    "pruning": {"strategy": "score_threshold", "threshold": threshold},
                                }
                            }
                        },
                        "benchmark": {
                            "evaluation_horizon": 30,
                            "seed": seed,
                            "scenarios": ["delayed_recall"],
                            "retrieval_strategy": "bm25",
                        },
                        "observability": {
                            "exporter": "none",
                            "endpoint": "http://localhost:4317",
                            "log_level": "ERROR",
                        },
                        "answering": {"enabled": False, "model": "", "max_tokens": 500},
                    }
                    config = load_config_from_dict(config_dict)
                    composer = BenchmarkComposer()
                    composed = composer.compose(config=config, dataset_override=gold_dataset, allow_strategy_fallback=True)
                    result = composed.runner.run(composed.scenarios)
                    sr = result.scenario_results[0]
                    decay_results.append(
                        {
                            "lambda": lam,
                            "threshold": threshold,
                            "recall": sr.recall_at_k,
                            "precision": sr.precision_at_k,
                        }
                    )

            click.echo(f"   Tested {len(decay_results)} combinations (lambda x threshold)")

            decay_response = {
                "formula": "exp(-lambda * age_days)",
                "age_days": list(range(0, 181, 10)),
                "curves": [
                    {
                        "lambda": lambda_value,
                        "relative_weights": [math.exp(-lambda_value * age) for age in range(0, 181, 10)],
                    }
                    for lambda_value in [0.0, 0.01, 0.05, 0.10, 0.20]
                ],
            }

        # =========================================================================
        # 4. MULTI-AGENT ISOLATION
        # =========================================================================
        if _is_truthy_env("BENCHMARK_ANALYZE_SKIP_ISOLATION_TEST"):
            interference = None
        else:
            click.echo("\n🔒 4/5 Multi-Agent Isolation Test...")
            _append_progress("section_start:isolation_test")
            isolation_store = EpisodicStore(retrieval_strategy=BM25Strategy())
            interference = run_interference_test(
                isolation_store, num_users=10, memories_per_user=50, queries_per_user=10
            )
            click.echo(
                f"   Isolation: {interference.isolation_rate:.0%} "
                f"({interference.leaked_queries} leaks / {interference.total_queries} queries)"
            )

        # =========================================================================
        # 5. GENERATE PLOTS
        # =========================================================================
        if _is_truthy_env("BENCHMARK_ANALYZE_SKIP_PLOTS"):
            _append_progress("section_start:plot_generation_skipped")
        else:
            click.echo("\n📈 5/5 Generating plots...")
            _append_progress("section_start:plot_generation")
        if not _is_truthy_env("BENCHMARK_ANALYZE_SKIP_PLOTS"):
            if find_spec("matplotlib") is None:
                click.echo("   ⚠ matplotlib not installed — skipping plots (pip install matplotlib)")
            else:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                fig, axes = plt.subplots(4, 2, figsize=(16, 20))
                fig.suptitle(
                    f"Agentic Memory Benchmark Analysis\n"
                    f"({len(gold_dataset.queries)} queries, "
                    f"{gold_dataset.total_conversation_turns} memories)",
                    fontsize=13,
                    fontweight="bold",
                )

                # Panel 1: Strategy comparison
                ax = axes[0, 0]
                names = [r["strategy"] for r in strategy_results]
                recalls_s = [r["recall"] * 100 for r in strategy_results]
                precs_s = [r["precision"] * 100 for r in strategy_results]
                x = range(len(names))
                width = 0.35
                ax.bar([i - width / 2 for i in x], recalls_s, width, label="Recall@K", color="steelblue")
                ax.bar([i + width / 2 for i in x], precs_s, width, label="Precision@K", color="coral")
                ax.set_xticks(list(x))
                ax.set_xticklabels(names)
                ax.set_ylabel("%")
                ax.set_title("Strategy Comparison")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)

                # Panel 2: Memory type comparison
                ax = axes[0, 1]
                mod_names = [r["module"].replace("_store", "") for r in memory_results]
                mod_recalls = [r["recall"] * 100 for r in memory_results]
                colors = ["steelblue", "forestgreen", "coral", "mediumpurple"]
                ax.barh(mod_names, mod_recalls, color=colors[: len(mod_names)])
                ax.set_xlabel("Recall@K (%)")
                ax.set_title("Memory Type Comparison")
                ax.grid(axis="x", alpha=0.3)

                # Panel 3: Theoretical decay weight versus memory age.
                # The decay sweep currently uses exponential decay, so this shows
                # exactly how lambda changes the weight before retrieval ranking.
                ax = axes[1, 0]
                ages = np.arange(0, 181)
                for lambda_value in [0.0, 0.01, 0.05, 0.10, 0.20]:
                    weights = np.exp(-lambda_value * ages)
                    ax.plot(ages, weights, label=f"lambda={lambda_value:g}")
                ax.set_xlabel("Memory age (simulated days)")
                ax.set_ylabel("Relative decay weight")
                ax.set_title("Decay Weight versus Memory Age")
                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

                # Build response surfaces for every lambda x pruning combination.
                lambdas_unique = sorted(set(r["lambda"] for r in decay_results))
                thresholds_unique = sorted(set(r["threshold"] for r in decay_results))
                recall_heatmap = np.zeros((len(thresholds_unique), len(lambdas_unique)))
                precision_heatmap = np.zeros_like(recall_heatmap)
                for r in decay_results:
                    li = lambdas_unique.index(r["lambda"])
                    ti = thresholds_unique.index(r["threshold"])
                    recall_heatmap[ti, li] = r["recall"] * 100
                    precision_heatmap[ti, li] = r["precision"] * 100

                def draw_heatmap(ax, values, title):
                    image = ax.imshow(values, aspect="auto", cmap="YlOrRd_r", origin="lower")
                    ax.set_xticks(range(len(lambdas_unique)))
                    ax.set_xticklabels([f"{value:.2f}" for value in lambdas_unique])
                    ax.set_yticks(range(len(thresholds_unique)))
                    ax.set_yticklabels([f"{value:.2f}" for value in thresholds_unique])
                    ax.set_xlabel("Decay lambda")
                    ax.set_ylabel("Pruning threshold")
                    ax.set_title(title)
                    for ti in range(len(thresholds_unique)):
                        for li in range(len(lambdas_unique)):
                            ax.text(li, ti, f"{values[ti, li]:.1f}", ha="center", va="center", fontsize=8)
                    plt.colorbar(image, ax=ax, label="Percent")

                # Panel 4: Recall response surface.
                draw_heatmap(axes[1, 1], recall_heatmap, "Recall@K (%) by Decay x Pruning")

                # Panel 5: Precision response surface.
                draw_heatmap(axes[2, 0], precision_heatmap, "Precision@K (%) by Decay x Pruning")

                # Panel 6: Combined response: every point is one tested configuration.
                ax = axes[2, 1]
                for r in decay_results:
                    ax.scatter(r["recall"] * 100, r["precision"] * 100, s=55, alpha=0.85)
                    ax.annotate(
                        f"l={r['lambda']:g}, p={r['threshold']:g}",
                        (r["recall"] * 100, r["precision"] * 100),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
                ax.set_xlabel("Recall@K (%)")
                ax.set_ylabel("Precision@K (%)")
                ax.set_title("Decay x Pruning Recall-Precision Tradeoff")
                ax.grid(alpha=0.3)

                # Panel 7: Latency comparison.
                ax = axes[3, 0]
                if strategy_results:
                    lat_names = [r["strategy"] for r in strategy_results]
                    lat_values = [r["ms_per_query"] for r in strategy_results]
                    bars = ax.bar(lat_names, lat_values, color="steelblue")
                    ax.set_ylabel("Latency (ms/query)")
                    ax.set_title("Query Latency by Strategy")
                    ax.grid(axis="y", alpha=0.3)
                    for bar, value in zip(bars, lat_values, strict=True):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"{value:.1f}",
                            ha="center",
                            fontsize=9,
                        )

                # Panel 8: Contamination comparison.
                ax = axes[3, 1]
                if strategy_results:
                    contamination_values = [r["contamination"] * 100 for r in strategy_results]
                    ax.bar(lat_names, contamination_values, color="darkorange")
                    ax.set_ylabel("Contamination (%)")
                    ax.set_title("Retrieved Noise by Strategy")
                    ax.grid(axis="y", alpha=0.3)

                plt.tight_layout()
                plot_path = output_dir / "benchmark_analysis.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                click.echo(f"   ✓ Plot saved: {plot_path}")
                artifact_manifest.append(
                    _artifact_entry(
                        "overview_benchmark_analysis",
                        "image",
                        str(plot_path),
                        "Combined benchmark overview covering strategy, memory, decay, latency, and contamination panels.",
                    )
                )

                if strategy_results:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(strategy_results))
                    recalls = [row["recall"] * 100 for row in strategy_results]
                    precisions = [row["precision"] * 100 for row in strategy_results]
                    ax.bar(x - 0.18, recalls, width=0.36, label="Recall@K", color="steelblue")
                    ax.bar(x + 0.18, precisions, width=0.36, label="Precision@K", color="coral")
                    ax.set_xticks(x)
                    ax.set_xticklabels([row["strategy"] for row in strategy_results], rotation=15)
                    ax.set_ylabel("Percent")
                    ax.set_title("Strategy Recall vs Precision")
                    ax.legend()
                    ax.grid(axis="y", alpha=0.3)
                    tagged_path = output_dir / "strategy_recall_precision.png"
                    fig.tight_layout()
                    fig.savefig(tagged_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    artifact_manifest.append(
                        _artifact_entry(
                            "strategy_recall_precision",
                            "image",
                            str(tagged_path),
                            "Compares recall and precision for each retrieval backend used in the benchmark run.",
                        )
                    )

                if embedding_model_results or api_embedding_model_results:
                    fig, ax = plt.subplots(figsize=(11, 6))
                    backend_rows = [
                        *[("local", row) for row in embedding_model_results],
                        *[("api", row) for row in api_embedding_model_results],
                    ]
                    labels = [f"{provider}:{row['label']}" for provider, row in backend_rows]
                    recalls = [row["recall"] * 100 for _, row in backend_rows]
                    latency = [row["ms_per_query"] for _, row in backend_rows]
                    ax.scatter(latency, recalls, s=80, c=range(len(labels)), cmap="viridis")
                    for idx, label in enumerate(labels):
                        ax.annotate(
                            label,
                            (latency[idx], recalls[idx]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                        )
                    ax.set_xlabel("Latency (ms/query)")
                    ax.set_ylabel("Recall@K (%)")
                    ax.set_title("Embedding Backend and Model Sweep")
                    ax.grid(alpha=0.3)
                    tagged_path = output_dir / "embedding_backend_sweep.png"
                    fig.tight_layout()
                    fig.savefig(tagged_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    artifact_manifest.append(
                        _artifact_entry(
                            "embedding_backend_sweep",
                            "image",
                            str(tagged_path),
                            "Shows recall-latency tradeoffs across local, HF, and Ollama embedding model comparisons.",
                        )
                    )

                if decay_results:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for threshold in sorted(set(row["threshold"] for row in decay_results)):
                        threshold_rows = [row for row in decay_results if row["threshold"] == threshold]
                        threshold_rows.sort(key=lambda row: row["lambda"])
                        ax.plot(
                            [row["lambda"] for row in threshold_rows],
                            [row["recall"] * 100 for row in threshold_rows],
                            marker="o",
                            label=f"threshold={threshold:.2f}",
                        )
                    ax.set_xlabel("Decay lambda")
                    ax.set_ylabel("Recall@K (%)")
                    ax.set_title("Decay Sweep Cliff Detection")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    tagged_path = output_dir / "decay_cliff_sweep.png"
                    fig.tight_layout()
                    fig.savefig(tagged_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    artifact_manifest.append(
                        _artifact_entry(
                            "decay_cliff_sweep",
                            "image",
                            str(tagged_path),
                            "Highlights recall degradation and cliff points across lambda and pruning threshold combinations.",
                        )
                    )

                if reranker_model_results:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    labels = [row["model"] for row in reranker_model_results]
                    recalls = [row["recall"] * 100 for row in reranker_model_results]
                    latency = [row["ms_per_query"] for row in reranker_model_results]
                    ax.bar(labels, recalls, color="darkgreen")
                    for idx, value in enumerate(latency):
                        ax.text(idx, recalls[idx] + 0.5, f"{value:.1f} ms/q", ha="center", fontsize=8)
                    ax.set_ylabel("Recall@K (%)")
                    ax.set_title("Reranker Provider Comparison")
                    ax.grid(axis="y", alpha=0.3)
                    tagged_path = output_dir / "reranker_provider_comparison.png"
                    fig.tight_layout()
                    fig.savefig(tagged_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    artifact_manifest.append(
                        _artifact_entry(
                            "reranker_provider_comparison",
                            "image",
                            str(tagged_path),
                            "Summarizes the configured reranker model result and latency under the selected provider routing policy.",
                        )
                    )


        # =========================================================================
        # SAVE JSON REPORT
        # =========================================================================
        _append_progress("section_start:write_reports")
        (
            effective_config_path,
            environment_path,
            run_metadata_path,
        ) = _ensure_partial_protocol_artifacts()

        report = {
            "dataset": {
                "scenario": gold_dataset.scenario,
                "queries": len(gold_dataset.queries),
                "memories": gold_dataset.total_conversation_turns,
                "users": len(gold_dataset.user_ids),
            },
            "run_plan": first_run_plan,
            "effective_config_artifact": effective_config_path if first_config_snapshot is not None else None,
            "environment_artifact": environment_path,
            "run_metadata_artifact": run_metadata_path,
            "strategy_comparison": strategy_results,
            "embedding_model_comparison": embedding_model_results,
            "api_embedding_model_comparison": api_embedding_model_results,
            "reranker_model_comparison": reranker_model_results,
            "provider_failures": provider_failures,
            "memory_type_comparison": memory_results,
            "decay_sweep": decay_results,
            "decay_response": decay_response,
            "isolation": {"rate": interference.isolation_rate, "leaks": interference.leaked_queries},
            "artifact_manifest": artifact_manifest,
            "llm_judge": {
                "enabled": with_llm_judge,
                "method": "llm_judge" if with_llm_judge else None,
            },
            "status": "completed",
        }
        report_path = output_dir / "benchmark_report.json"
        with report_path.open("w") as fh:
            json.dump(report, fh, indent=2)

        manifest_path = _write_tagged_json(
            output_dir,
            "artifact_manifest",
            {
                "artifacts": artifact_manifest,
                "summary": "Tagged images and JSON artifacts emitted by memtuner analyze.",
            },
        )

        click.echo(f"\n💾 Report saved: {report_path}")
        click.echo(f"📊 Plots saved: {output_dir}/benchmark_analysis.png")
        click.echo(f"🗂 Artifact manifest saved: {manifest_path}")

        # Print final summary
        click.echo(f"\n{'═' * 60}")
        click.echo("  ANALYSIS COMPLETE")
        click.echo(f"{'═' * 60}")
        if strategy_results:
            best = max(strategy_results, key=lambda r: r["recall"])
            click.echo(f"  Best strategy: {best['strategy']} (Recall={best['recall']:.1%})")
        if memory_results:
            best_m = max(memory_results, key=lambda r: r["recall"])
            click.echo(f"  Best memory type: {best_m['module']} (Recall={best_m['recall']:.1%})")
            click.echo(f"  Isolation: {'✓ PASS' if interference.leaked_queries == 0 else '✗ FAIL'}")
        click.echo(f"{'═' * 60}")
    except Exception as exc:
        _write_partial_report(str(exc))
        raise

