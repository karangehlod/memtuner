"""Configuration schema using pydantic for validation.

Defines the complete configuration structure for benchmark runs.
Validation happens at load time — fail fast on invalid configs.
Unknown fields are rejected immediately (extra="forbid").
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DecayType(str, Enum):
    """Supported decay function types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    TIERED = "tiered"
    STEP = "step"


class PruningStrategy(str, Enum):
    """Supported pruning strategies."""

    SCORE_THRESHOLD = "score_threshold"
    AGE_BASED = "age_based"
    CAPACITY_BASED = "capacity_based"


class DecayConfig(BaseModel):
    """Configuration for a decay policy.

    Attributes:
        type: The decay function type.
        lambda_factor: Decay rate parameter (for exponential decay).
        ranking_alpha: How strongly decay affects ranking (0.0–1.0).
            0.0 = decay is applied only as a post-ranking score multiplier
                  (no effect on which memories appear in top-K).
            1.0 = full decay penalty in ranking (recent memories strongly preferred).
            Default 0.5 gives moderate recency bias so lambda sweeps produce
            measurably different recall values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: DecayType = Field(default=DecayType.EXPONENTIAL, description="Decay function type")
    lambda_factor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        alias="lambda",
        description="Decay rate parameter",
    )
    ranking_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Decay influence on ranking (0=post-hoc only, 1=full recency bias)",
    )
    archival_floor: float | None = Field(
        default=0.65,
        description=(
            "Minimum decay factor for memories older than archival_day_threshold. "
            "None = no floor (full decay to zero allowed). "
            "0.65 (default) = old memories retain at least 65% weight."
        ),
    )
    archival_day_threshold: int = Field(
        default=90,
        ge=1,
        description="Age in days at which the archival floor kicks in.",
    )
    tiered_working_days: int = Field(
        default=7,
        ge=0,
        description="Tiered policy: working memory window (no decay). Only used when type=tiered.",
    )


class PruningConfig(BaseModel):
    """Configuration for a pruning policy.

    Attributes:
        strategy: The pruning strategy to use.
        threshold: Score threshold for pruning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: PruningStrategy = Field(
        default=PruningStrategy.SCORE_THRESHOLD,
        description="Pruning strategy",
    )
    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Score threshold for pruning",
    )


class ModulePolicyConfig(BaseModel):
    """Policy configuration for a specific memory module.

    Attributes:
        decay: Decay policy configuration.
        pruning: Pruning policy configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decay: DecayConfig = Field(default_factory=DecayConfig, description="Decay policy")
    pruning: PruningConfig = Field(default_factory=PruningConfig, description="Pruning policy")


class MemorySelectionConfig(BaseModel):
    """Configuration for which memory modules are enabled.

    Attributes:
        short_term: List of enabled short-term memory module names.
        long_term: List of enabled long-term memory module names.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    short_term: list[str] = Field(default_factory=list, description="Enabled STM modules")
    long_term: list[str] = Field(default_factory=list, description="Enabled LTM modules")


class MemoryConfig(BaseModel):
    """Top-level memory configuration.

    Attributes:
        enabled: Which memory modules are enabled.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: MemorySelectionConfig = Field(
        default_factory=MemorySelectionConfig,
        description="Enabled memory modules",
    )


class PoliciesConfig(BaseModel):
    """Top-level policies configuration.

    Maps module names to their policy configs.

    Attributes:
        module_policies: Per-module policy configurations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    module_policies: dict[str, ModulePolicyConfig] = Field(
        default_factory=dict,
        description="Per-module policy configurations",
    )


class EmbeddingsRetrievalConfig(BaseModel):
    """Configuration for local sentence-transformer embeddings retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str | None = Field(default=None, description="Embedding model identifier")
    comparison_models: str | None = Field(
        default=None,
        description="Comma-separated local embedding model identifiers for analysis comparison",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Optional cache directory for sentence-transformer downloads",
    )


class ApiEmbeddingsRetrievalConfig(BaseModel):
    """Configuration for API-based embeddings (OpenAI-compatible endpoint).

    Works with any provider that exposes a /v1/embeddings endpoint:
    OpenAI, Ollama (base_url=http://localhost:11434/v1), vLLM, HF TEI, Together, etc.
    base_url and api_key default to BENCHMARK_OPENAI_BASE_URL / OPENAI_API_KEY env vars.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str | None = Field(default=None, description="Embedding model name")
    base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL. Reads BENCHMARK_OPENAI_BASE_URL if None.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key. Reads OPENAI_API_KEY if None.",
    )
    timeout: float = Field(default=60.0, gt=0.0, le=600.0, description="Request timeout seconds")
    batch_size: int = Field(default=512, ge=1, le=2048, description="Texts per API call")


class HybridRetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategies: list[str] = Field(
        default_factory=lambda: ["bm25", "embeddings"],
        min_length=1,
        description="Ordered retrieval strategies to fuse inside hybrid retrieval",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for hybrid retrieval",
    )
    bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative BM25 weight in reciprocal rank fusion",
    )


class EmptyRetrievalConfig(BaseModel):
    """Marker config for strategies with no validated retrieval parameters yet."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalConfig(BaseModel):
    """Validated retrieval-strategy configuration passed through composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embeddings: EmbeddingsRetrievalConfig = Field(
        default_factory=EmbeddingsRetrievalConfig,
        description="Local embeddings retrieval settings",
    )
    bm25: EmptyRetrievalConfig = Field(
        default_factory=EmptyRetrievalConfig,
        description="BM25 retrieval settings",
    )
    hybrid: HybridRetrievalConfig = Field(
        default_factory=HybridRetrievalConfig,
        description="Hybrid retrieval settings",
    )
    api_embeddings: ApiEmbeddingsRetrievalConfig = Field(
        default_factory=ApiEmbeddingsRetrievalConfig,
        description="API-based embeddings (OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, HF TEI)",
    )
    database: EmptyRetrievalConfig = Field(
        default_factory=EmptyRetrievalConfig,
        description="Database retrieval settings",
    )
    llm: EmptyRetrievalConfig = Field(
        default_factory=EmptyRetrievalConfig,
        description="LLM retrieval settings",
    )


class RerankerConfig(BaseModel):
    """Configuration for local and API-backed reranker selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str = Field(
        default="local_overlap",
        description="Reranker strategy: local_overlap or adaptive_api",
    )
    model_name: str | None = Field(default=None, description="Reranker model identifier")
    api_provider_order: list[str] = Field(
        default_factory=list,
        description="Reserved for future API reranker providers.",
    )
    local_size_threshold_mb: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Rerankers above this size should use an API backend",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Optional local cache directory for reranker downloads",
    )
    reranker_top_n: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of BM25 candidates passed to the reranker. "
            "0 = full corpus (no limit). "
            "Non-zero values (e.g. 20, 50, 100) cap pre-rerank candidate depth."
        ),
    )


class BenchmarkScopeConfig(BaseModel):
    """Configuration for benchmark scope.

    Attributes:
        evaluation_horizon: Number of dataset days to evaluate (replay horizon).
        seed: Random seed for deterministic execution.
        scenarios: List of scenario names to run.
        retrieval_strategy: Retrieval strategy to use for memory modules.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_horizon: int = Field(default=14, ge=1, le=365, description="Number of dataset days to evaluate (replay horizon)")
    seed: int = Field(default=42, description="Random seed for determinism")
    scenarios: list[str] = Field(
        default_factory=lambda: ["delayed_recall"],
        description="Scenarios to run",
    )
    retrieval_strategy: str = Field(
        default="bm25",
        description=(
            "Retrieval strategy: bm25, embeddings, api_embeddings, "
            "llm, database, or hybrid"
        ),
    )
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig,
        description="Optional reranker selection and provider policy",
    )
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Validated retrieval strategy configuration",
    )


class ObservabilityConfig(BaseModel):
    """Configuration for observability/OTel.

    Attributes:
        exporter: OTel exporter type.
        endpoint: OTel collector endpoint.
        log_level: Logging level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exporter: str = Field(default="otlp", description="OTel exporter type")
    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTel collector endpoint",
    )
    log_level: str = Field(default="INFO", description="Logging level")


class AnsweringConfig(BaseModel):
    """Configuration for optional RAG/LLM answering.

    Attributes:
        enabled: Whether the answering system is active.
        model: The LLM model to use.
        max_tokens: Maximum tokens for completion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False, description="Enable answering system")
    model: str = Field(default="", description="LLM model name — set via BENCHMARK_OPENAI_BASE_URL endpoint")
    max_tokens: int = Field(default=500, ge=1, le=4096, description="Max completion tokens")


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration.

    This is the top-level config object loaded from YAML.
    Validated at load time — fail fast on invalid configs.

    Attributes:
        memory: Memory module selection.
        policies: Per-module policy configurations.
        benchmark: Benchmark scope settings.
        observability: OTel configuration.
        answering: Optional answering configuration.
    """

    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory module selection",
    )
    policies: PoliciesConfig = Field(
        default_factory=PoliciesConfig,
        description="Policy configurations",
    )
    benchmark: BenchmarkScopeConfig = Field(
        default_factory=BenchmarkScopeConfig,
        description="Benchmark scope",
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Observability settings",
    )
    answering: AnsweringConfig = Field(
        default_factory=AnsweringConfig,
        description="Answering system settings",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)
