"""Benchmark run result data model.

Represents the complete output of a benchmark run.
This is a pure data class — no business logic.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ScenarioMetrics(BaseModel):
    """Metrics for a single scenario within a benchmark run.

    Attributes:
        scenario_name: Name of the scenario.
        recall_at_k: Recall@K metric (0.0 to 1.0).
        contamination_rate: Contamination rate (0.0 to 1.0).
        precision_at_k: Precision@K metric (0.0 to 1.0).
        temporal_accuracy: Temporal accuracy metric (0.0 to 1.0).
        module_accuracy: Module accuracy metric (0.0 to 1.0).
        mrr: Mean Reciprocal Rank (0.0 to 1.0).
        ndcg: Normalized Discounted Cumulative Gain (0.0 to 1.0).
        precision_at_1: Precision@1 (0.0 to 1.0).
        memory_survival_rates: Survival rate per simulated day.
        total_queries: Number of queries executed.
        correct_recalls: Number of queries with at least one correct result.
    """

    scenario_name: str = Field(..., description="Scenario name")
    recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Recall@K")
    contamination_rate: float = Field(..., ge=0.0, le=1.0, description="Contamination rate")
    precision_at_k: float = Field(default=0.0, ge=0.0, le=1.0, description="Precision@K")
    temporal_accuracy: float = Field(..., ge=0.0, le=1.0, description="Temporal accuracy")
    module_accuracy: float = Field(default=1.0, ge=0.0, le=1.0, description="Module accuracy")
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg: float = Field(default=0.0, ge=0.0, le=1.0, description="NDCG@K")
    precision_at_1: float = Field(default=0.0, ge=0.0, le=1.0, description="Precision@1")
    llm_judge_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mean normalized LLM-as-judge score when enabled",
    )
    llm_judge_queries: int = Field(
        default=0,
        ge=0,
        description="Queries evaluated by the LLM judge",
    )
    memory_survival_rates: dict[int, float] = Field(
        default_factory=dict,
        description="Survival rate per simulated day",
    )
    total_queries: int = Field(..., ge=0, description="Total queries executed")
    correct_recalls: int = Field(..., ge=0, description="Queries with correct results")

    # Latency percentiles (per-query retrieval latency in milliseconds)
    latency_p50_ms: float = Field(default=0.0, ge=0.0, description="Latency P50 in ms")
    latency_p90_ms: float = Field(default=0.0, ge=0.0, description="Latency P90 in ms")
    latency_p99_ms: float = Field(default=0.0, ge=0.0, description="Latency P99 in ms")
    latency_mean_ms: float = Field(default=0.0, ge=0.0, description="Mean latency in ms")

    model_config = {"frozen": True}


class CostSummary(BaseModel):
    """Cost breakdown for a benchmark run.

    Attributes:
        total_token_cost: Total cost from LLM tokens in USD.
        total_storage_cost: Total cost from storage operations in USD.
        total_cost: Sum of all costs in USD.
        cost_per_correct_recall: Cost per query with correct results in USD.
    """

    total_token_cost: float = Field(default=0.0, ge=0.0, description="Token cost in USD")
    total_storage_cost: float = Field(default=0.0, ge=0.0, description="Storage cost in USD")
    total_cost: float = Field(default=0.0, ge=0.0, description="Total cost in USD")
    cost_per_correct_recall: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per correct recall in USD",
    )

    model_config = {"frozen": True}


class BenchmarkRunResult(BaseModel):
    """Complete output of a benchmark run.

    Attributes:
        run_id: Unique identifier for this benchmark run.
        config_hash: Hash of the config used for reproducibility.
        started_at: Timestamp when the run started.
        completed_at: Timestamp when the run completed.
        seed: Random seed used for determinism.
        memory_modules_enabled: List of enabled memory module names.
        scenario_results: Per-scenario metric results.
        cost_summary: Cost breakdown.
        aggregate_recall_at_k: Weighted average recall across scenarios.
        aggregate_temporal_accuracy: Weighted average temporal accuracy.
        aggregate_contamination_rate: Weighted average contamination rate.
    """

    run_id: str = Field(..., description="Unique run identifier")
    config_hash: str = Field(..., description="Config hash for reproducibility")
    started_at: datetime = Field(..., description="Run start timestamp")
    completed_at: datetime = Field(..., description="Run completion timestamp")
    seed: int = Field(..., description="Random seed used")
    memory_modules_enabled: list[str] = Field(
        default_factory=list,
        description="Enabled memory modules",
    )
    scenario_results: list[ScenarioMetrics] = Field(
        default_factory=list,
        description="Per-scenario results",
    )
    cost_summary: CostSummary = Field(
        default_factory=CostSummary,
        description="Cost breakdown",
    )
    aggregate_recall_at_k: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted average recall",
    )
    aggregate_temporal_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted average temporal accuracy",
    )
    aggregate_contamination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted average contamination rate",
    )

    model_config = {"frozen": True}
