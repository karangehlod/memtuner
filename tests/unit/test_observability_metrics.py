"""Unit tests for the OTel metrics facade."""

from __future__ import annotations

import pytest

from benchmark.observability.metrics import initialize_metrics, record_metric
from benchmark.observability.schemas import (
    METRIC_CONTAMINATION_RATE,
    METRIC_COST_PER_CORRECT_RECALL,
    METRIC_LATENCY_MS,
    METRIC_MEMORY_SURVIVAL_RATE,
    METRIC_RECALL_AT_K,
    METRIC_TEMPORAL_ACCURACY,
)


@pytest.mark.unit
class TestMetricsFacade:
    """Tests for the OTel metrics initialization and recording."""

    def test_initialize_metrics_returns_meter(self) -> None:
        meter = initialize_metrics("test-service")
        assert meter is not None

    def test_record_metric_recall_at_k(self) -> None:
        initialize_metrics("test-service")
        # Should not raise
        record_metric(METRIC_RECALL_AT_K, 0.85)

    def test_record_metric_contamination_rate(self) -> None:
        initialize_metrics("test-service")
        record_metric(METRIC_CONTAMINATION_RATE, 0.1)

    def test_record_metric_temporal_accuracy(self) -> None:
        initialize_metrics("test-service")
        record_metric(METRIC_TEMPORAL_ACCURACY, 0.92)

    def test_record_metric_latency_ms(self) -> None:
        initialize_metrics("test-service")
        record_metric(METRIC_LATENCY_MS, 15.5)

    def test_record_metric_cost_per_correct_recall(self) -> None:
        initialize_metrics("test-service")
        record_metric(METRIC_COST_PER_CORRECT_RECALL, 0.017)

    def test_record_metric_memory_survival_rate(self) -> None:
        initialize_metrics("test-service")
        record_metric(METRIC_MEMORY_SURVIVAL_RATE, 0.8)

    def test_record_metric_with_attributes(self) -> None:
        initialize_metrics("test-service")
        record_metric(
            METRIC_RECALL_AT_K,
            0.85,
            attributes={"scenario": "delayed_recall"},
        )

    def test_record_unknown_metric_raises_value_error(self) -> None:
        initialize_metrics("test-service")
        with pytest.raises(ValueError, match="Unknown metric name"):
            record_metric("nonexistent.metric", 1.0)

    def test_record_metric_auto_initializes_if_needed(self) -> None:
        """If instruments not created yet, record_metric initializes them."""
        # This tests the auto-init branch inside record_metric
        record_metric(METRIC_RECALL_AT_K, 0.5)

    def test_all_gauge_metrics_can_be_recorded(self) -> None:
        initialize_metrics("test-all")
        gauge_metrics = [
            METRIC_RECALL_AT_K,
            METRIC_CONTAMINATION_RATE,
            METRIC_TEMPORAL_ACCURACY,
            METRIC_COST_PER_CORRECT_RECALL,
            METRIC_MEMORY_SURVIVAL_RATE,
        ]
        for metric_name in gauge_metrics:
            record_metric(metric_name, 0.5)

    def test_histogram_metric_latency_ms(self) -> None:
        initialize_metrics("test-histogram")
        record_metric(METRIC_LATENCY_MS, 100.0)
        record_metric(METRIC_LATENCY_MS, 200.0)
