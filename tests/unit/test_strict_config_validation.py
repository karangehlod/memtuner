"""Unit tests for strict configuration validation.

Verifies that unknown fields are rejected at load time and
deprecated nesting structures produce clear error messages.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.config.schema import (
    BenchmarkConfig,
    DecayConfig,
    PruningConfig,
)


@pytest.mark.unit
class TestStrictConfigValidation:
    """Verify that extra="forbid" rejects unknown fields."""

    def test_valid_default_config_succeeds(self) -> None:
        """A default config with no extra fields should load fine."""
        config = BenchmarkConfig()
        assert config.benchmark.seed == 42

    def test_unknown_top_level_field_rejected(self) -> None:
        """Unknown top-level fields raise ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(unknown_field="bad")

    def test_unknown_benchmark_field_rejected(self) -> None:
        """Unknown fields in benchmark section raise ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                benchmark={"evaluation_horizon": 14, "seed": 42, "name": "should fail"}
            )

    def test_legacy_recall_k_rejected(self) -> None:
        """The legacy 'recall_k' field in benchmark section is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                benchmark={"evaluation_horizon": 14, "seed": 42, "recall_k": 10}
            )

    def test_legacy_metrics_list_rejected(self) -> None:
        """The legacy 'metrics' field in benchmark section is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                benchmark={
                    "evaluation_horizon": 14,
                    "seed": 42,
                    "metrics": ["recall_at_k", "mrr"],
                }
            )

    def test_legacy_gold_dataset_rejected(self) -> None:
        """The legacy 'gold_dataset' field in benchmark section is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                benchmark={
                    "evaluation_horizon": 14,
                    "seed": 42,
                    "gold_dataset": "data/locomo10.json",
                }
            )

    def test_deprecated_memory_nesting_rejected(self) -> None:
        """Using memory.short_term instead of memory.enabled.short_term is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                memory={"short_term": ["episodic_buffer"], "long_term": ["episodic_store"]}
            )

    def test_correct_memory_nesting_accepted(self) -> None:
        """The correct memory.enabled.short_term nesting works."""
        config = BenchmarkConfig(
            memory={
                "enabled": {
                    "short_term": ["episodic_buffer"],
                    "long_term": ["episodic_store"],
                }
            }
        )
        assert config.memory.enabled.short_term == ["episodic_buffer"]

    def test_legacy_observability_enabled_rejected(self) -> None:
        """The legacy 'enabled' field in observability is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                observability={"enabled": True, "exporter": "otlp"}
            )

    def test_legacy_observability_trace_sample_rate_rejected(self) -> None:
        """The legacy 'trace_sample_rate' field in observability is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(
                observability={"trace_sample_rate": 1.0, "exporter": "otlp"}
            )

    def test_unknown_cost_section_rejected(self) -> None:
        """A top-level 'cost' section is rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchmarkConfig(cost={"enabled": True, "model": "gpt-4o"})

    def test_unknown_decay_field_rejected(self) -> None:
        """Unknown fields in DecayConfig are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DecayConfig(type="exponential", **{"lambda": 0.05}, unknown="bad")

    def test_unknown_pruning_field_rejected(self) -> None:
        """Unknown fields in PruningConfig are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PruningConfig(strategy="score_threshold", threshold=0.3, unknown="bad")

    def test_valid_full_config_accepted(self) -> None:
        """A fully-specified valid config loads without error."""
        config = BenchmarkConfig(
            memory={
                "enabled": {
                    "short_term": ["episodic_buffer"],
                    "long_term": ["episodic_store", "preference_store"],
                }
            },
            policies={
                "module_policies": {
                    "episodic_store": {
                        "decay": {"type": "exponential", "lambda": 0.05},
                        "pruning": {"strategy": "score_threshold", "threshold": 0.35},
                    }
                }
            },
            benchmark={
                "evaluation_horizon": 14,
                "seed": 42,
                "scenarios": ["delayed_recall"],
                "retrieval_strategy": "bm25",
            },
            observability={
                "exporter": "none",
                "endpoint": "http://localhost:4317",
                "log_level": "INFO",
            },
            answering={"enabled": False, "model": "gpt-4o", "max_tokens": 500},
        )
        assert config.benchmark.evaluation_horizon == 14
        assert config.benchmark.retrieval_strategy == "bm25"
