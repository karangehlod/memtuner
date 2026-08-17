"""Unit tests for config schema, loader, and defaults."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmark.config.loader import load_config_from_dict, load_config_from_path
from benchmark.cli.commands.analyze_command import (
    _api_embeddings_models,
    _validated_provider_settings,
)
from benchmark.config.schema import (
    BenchmarkConfig,
    BenchmarkScopeConfig,
    DecayConfig,
    DecayType,
    PruningConfig,
    PruningStrategy,
)
from benchmark.exceptions.config_errors import ConfigLoadError, ConfigValidationError
from benchmark.factory.resolver import ConfigResolver

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.unit
class TestBenchmarkConfig:
    """Tests for the BenchmarkConfig pydantic model."""

    def test_default_config_is_valid(self) -> None:
        config = BenchmarkConfig()
        assert config.benchmark.seed == 42
        assert config.benchmark.evaluation_horizon == 14

    def test_custom_config_values(self) -> None:
        config = BenchmarkConfig(
            benchmark=BenchmarkScopeConfig(seed=123, evaluation_horizon=30)
        )
        assert config.benchmark.seed == 123
        assert config.benchmark.evaluation_horizon == 30

    def test_decay_config_defaults(self) -> None:
        decay = DecayConfig()
        assert decay.type == DecayType.EXPONENTIAL
        assert decay.lambda_factor == 0.05

    def test_pruning_config_defaults(self) -> None:
        pruning = PruningConfig()
        assert pruning.strategy == PruningStrategy.SCORE_THRESHOLD
        assert pruning.threshold == 0.35

    def test_invalid_evaluation_horizon_rejected(self) -> None:
        with pytest.raises(Exception):
            BenchmarkScopeConfig(evaluation_horizon=0)

    def test_invalid_seed_type_rejected(self) -> None:
        with pytest.raises(Exception):
            BenchmarkScopeConfig(seed="not_a_number")  # type: ignore[arg-type]


@pytest.mark.unit
class TestConfigLoader:
    """Tests for the config loader functions."""

    def test_load_default_yaml(self) -> None:
        config_path = CONFIGS_DIR / "default.yaml"
        if config_path.exists():
            config = load_config_from_path(config_path)
            assert isinstance(config, BenchmarkConfig)

    def test_load_from_dict(self) -> None:
        data = {
            "memory": {
                "enabled": {
                    "short_term": ["episodic_buffer"],
                    "long_term": ["episodic_store"],
                }
            },
            "benchmark": {"seed": 99, "evaluation_horizon": 7},
        }
        config = load_config_from_dict(data)
        assert config.benchmark.seed == 99
        assert config.benchmark.evaluation_horizon == 7

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(ConfigLoadError):
            load_config_from_path(Path("/nonexistent/config.yaml"))

    def test_load_empty_dict(self) -> None:
        config = load_config_from_dict({})
        assert isinstance(config, BenchmarkConfig)

    def test_load_from_dict_with_policies(self) -> None:
        data = {
            "policies": {
                "module_policies": {
                    "episodic_store": {
                        "decay": {"type": "exponential", "lambda": 0.1},
                        "pruning": {"strategy": "score_threshold", "threshold": 0.5},
                    }
                }
            }
        }
        config = load_config_from_dict(data)
        assert "episodic_store" in config.policies.module_policies

    def test_retrieval_env_defaults_are_hydrated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any local .env pollution
        monkeypatch.delenv("BENCHMARK_EMBEDDING_MODELS", raising=False)

        monkeypatch.setenv("BENCHMARK_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        monkeypatch.setenv("BENCHMARK_EMBED_CACHE_DIR", "/tmp/embed-cache")
        monkeypatch.setenv("BENCHMARK_RERANKER_STRATEGY", "adaptive_api")
        monkeypatch.setenv("BENCHMARK_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
        monkeypatch.setenv("BENCHMARK_RERANKER_LOCAL_SIZE_THRESHOLD_MB", "100")
        monkeypatch.setenv("BENCHMARK_RERANKER_API_PROVIDER_ORDER", "hf_inference,ollama")

        config = load_config_from_dict({"benchmark": {"retrieval_strategy": "embeddings"}})

        assert config.benchmark.retrieval.embeddings.model_name == "BAAI/bge-base-en-v1.5"
        assert config.benchmark.retrieval.embeddings.cache_dir == "/tmp/embed-cache"
        assert config.benchmark.reranker.strategy == "adaptive_api"
        assert config.benchmark.reranker.model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert config.benchmark.reranker.local_size_threshold_mb == 100
        assert config.benchmark.reranker.api_provider_order == ["hf_inference", "ollama"]

    def test_explicit_retrieval_config_overrides_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clear any local .env pollution
        monkeypatch.delenv("BENCHMARK_EMBEDDING_MODELS", raising=False)

        monkeypatch.setenv("BENCHMARK_EMBEDDING_MODEL", "env-model")

        config = load_config_from_dict(
            {
                "benchmark": {
                    "retrieval_strategy": "embeddings",
                    "retrieval": {
                        "embeddings": {
                            "model_name": "config-model",
                        }
                    },
                }
            }
        )

        assert config.benchmark.retrieval.embeddings.model_name == "config-model"

    def test_explicit_empty_retrieval_blocks_are_valid(self) -> None:
        config = load_config_from_dict(
            {
                "benchmark": {
                    "retrieval_strategy": "bm25",
                    "retrieval": {
                        "bm25": {},
                        "hybrid": {},
                        "database": {},
                        "llm": {},
                    },
                }
            }
        )

        assert isinstance(config, BenchmarkConfig)

    def test_hybrid_retrieval_config_is_valid(self) -> None:
        config = load_config_from_dict(
            {
                "benchmark": {
                    "retrieval_strategy": "hybrid",
                    "retrieval": {
                        "hybrid": {
                            "strategies": ["bm25", "embeddings"],
                            "confidence_threshold": 0.8,
                            "bm25_weight": 0.7,
                        },
                        "embeddings": {
                            "model_name": "all-MiniLM-L6-v2",
                        },
                    },
                }
            }
        )

        assert config.benchmark.retrieval.hybrid.strategies == ["bm25", "embeddings"]
        assert config.benchmark.retrieval.hybrid.confidence_threshold == 0.8
        assert config.benchmark.retrieval.hybrid.bm25_weight == 0.7

    def test_hybrid_retrieval_config_resolver_builds_nested_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear .env variables that might be loaded from local .env file
        monkeypatch.delenv("BENCHMARK_EMBEDDING_MODELS", raising=False)

        config = load_config_from_dict(
            {
                "benchmark": {
                    "retrieval_strategy": "hybrid",
                    "retrieval": {
                        "hybrid": {
                            "strategies": ["bm25", "embeddings"],
                            "confidence_threshold": 0.9,
                            "bm25_weight": 0.6,
                        },
                        "embeddings": {
                            "model_name": "test-model",
                        },
                    },
                }
            }
        )
        registry = Mock()
        strategy_registry = Mock()
        strategy_registry.resolve.side_effect = [
            "bm25-instance",
            "embeddings-instance",
            "hybrid-instance",
        ]

        resolver = ConfigResolver(registry=registry, strategy_registry=strategy_registry)
        strategy = resolver.resolve_retrieval_strategy(config, "hybrid")

        assert strategy == "hybrid-instance"
        resolve_calls = strategy_registry.resolve.call_args_list
        assert resolve_calls[0].args == ("bm25",)
        assert resolve_calls[1].args == ("embeddings",)
        assert resolve_calls[1].kwargs == {"model_name": "test-model"}
        assert resolve_calls[2].args == ("hybrid",)
        assert resolve_calls[2].kwargs["strategies"] == ["bm25", "embeddings"]
        assert resolve_calls[2].kwargs["confidence_threshold"] == 0.9
        assert resolve_calls[2].kwargs["bm25_weight"] == 0.6
        assert resolve_calls[2].kwargs["bm25_strategy"] == "bm25-instance"
        assert resolve_calls[2].kwargs["embeddings_strategy"] == "embeddings-instance"

    def test_reranker_resolver_builds_provider_kwargs(self) -> None:
        config = load_config_from_dict(
            {
                "benchmark": {
                    "retrieval_strategy": "llm_rerank",
                    "reranker": {
                        "strategy": "adaptive_api",
                        "model_name": "BAAI/bge-reranker-base",
                        "api_provider_order": ["hf_inference", "ollama"],
                        "local_size_threshold_mb": 100,
                        "reranker_top_n": 50,
                    },
                }
            }
        )

        resolver = ConfigResolver(Mock())

        kwargs = resolver._build_retrieval_strategy_kwargs(
            "llm_rerank",
            config.benchmark.retrieval,
            config,
        )

        assert kwargs == {
            "reranker_strategy": "adaptive_api",
            "model_name": "BAAI/bge-reranker-base",
            "api_provider_order": ["hf_inference", "ollama"],
            "local_size_threshold_mb": 100,
            "reranker_top_n": 50,
        }

    def test_reranker_analysis_config_loads_without_llm_rerank_retrieval_block(self) -> None:
        config = load_config_from_dict(
            {
                "memory": {"enabled": {"short_term": [], "long_term": ["episodic_store"]}},
                "policies": {
                    "module_policies": {
                        "episodic_store": {
                            "decay": {"type": "exponential", "lambda": 0.0},
                            "pruning": {"strategy": "score_threshold", "threshold": 0.01},
                        }
                    }
                },
                "benchmark": {
                    "evaluation_horizon": 30,
                    "seed": 42,
                    "scenarios": ["delayed_recall"],
                    "retrieval_strategy": "llm_rerank",
                    "retrieval": {
                        "bm25": {},
                    },
                    "reranker": {
                        "strategy": "adaptive_api",
                        "model_name": "BAAI/bge-reranker-base",
                        "api_provider_order": ["hf_inference", "ollama"],
                        "local_size_threshold_mb": 100,
                    },
                },
                "observability": {
                    "exporter": "none",
                    "endpoint": "http://localhost:4317",
                    "log_level": "ERROR",
                },
                "answering": {"enabled": False, "model": "gpt-4o", "max_tokens": 500},
            }
        )

        retrieval_dump = config.benchmark.retrieval.model_dump(mode="python")

        assert "llm_rerank" not in retrieval_dump
        assert config.benchmark.reranker.model_name == "BAAI/bge-reranker-base"

    def test_reranker_analysis_config_rejects_llm_rerank_retrieval_block(self) -> None:
        with pytest.raises(ConfigValidationError):
            load_config_from_dict(
                {
                    "benchmark": {
                        "retrieval_strategy": "llm_rerank",
                        "retrieval": {
                            "bm25": {},
                            "llm_rerank": {},
                        },
                        "reranker": {
                            "strategy": "adaptive_api",
                            "model_name": "BAAI/bge-reranker-base",
                        },
                    }
                }
            )


@pytest.mark.unit
class TestAnalyzeProviderDiscovery:
    def test_validated_provider_settings_prefer_loaded_config_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BENCHMARK_OPENAI_BASE_URL", "https://env.invalid/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        config = load_config_from_dict(
            {
                "benchmark": {
                    "retrieval": {
                        "api_embeddings": {
                            "model_name": "text-embedding-3-small",
                            "base_url": "http://localhost:11434/v1",
                            "api_key": "config-key",
                            "timeout": 45.0,
                            "batch_size": 128,
                        },
                    },
                    "reranker": {
                        "strategy": "adaptive_api",
                        "model_name": "BAAI/bge-reranker-base",
                        "api_provider_order": ["hf_inference", "ollama"],
                        "local_size_threshold_mb": 100,
                    },
                }
            }
        )

        settings = _validated_provider_settings(config)

        assert settings["api_embeddings"]["model_name"] == "text-embedding-3-small"
        assert settings["api_embeddings"]["base_url"] == "http://localhost:11434/v1"
        assert settings["api_embeddings"]["api_key"] == "config-key"
        assert settings["api_embeddings"]["timeout"] == 45.0
        assert settings["api_embeddings"]["batch_size"] == 128
        assert settings["reranker"]["model_name"] == "BAAI/bge-reranker-base"

    def test_api_embeddings_models_use_configured_env_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BENCHMARK_OPENAI_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv(
            "BENCHMARK_OPENAI_EMBEDDING_MODELS",
            "text-embedding-3-small,text-embedding-3-large",
        )

        candidates = _api_embeddings_models()

        assert candidates == ["text-embedding-3-small", "text-embedding-3-large"]

    def test_api_embeddings_models_empty_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BENCHMARK_OPENAI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("BENCHMARK_OPENAI_EMBEDDING_MODELS", raising=False)

        assert _api_embeddings_models() == []
