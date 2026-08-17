"""OTel metrics facade.

Provides a simplified interface for recording benchmark metrics.
This module contains NO business logic — it is a pure observability facade.
"""

from __future__ import annotations

from opentelemetry import metrics

from benchmark.observability.schemas import (
    ALL_METRIC_NAMES,
    METRIC_CONTAMINATION_RATE,
    METRIC_COST_PER_CORRECT_RECALL,
    METRIC_LATENCY_MS,
    METRIC_MEMORY_SURVIVAL_RATE,
    METRIC_RECALL_AT_K,
    METRIC_TEMPORAL_ACCURACY,
    SCHEMA_VERSION,
)

_meter: metrics.Meter | None = None
_instruments: dict[str, metrics.Instrument] = {}


def initialize_metrics(service_name: str = "agentic-memory-benchmark") -> metrics.Meter:
    """Initialize the global OTel meter and create all metric instruments.

    Args:
        service_name: The service name for metric identification.

    Returns:
        The initialized Meter instance.
    """
    global _meter
    _meter = metrics.get_meter(service_name, SCHEMA_VERSION)

    _instruments[METRIC_RECALL_AT_K] = _meter.create_gauge(
        name=METRIC_RECALL_AT_K,
        unit="ratio",
        description="Recall@K metric",
    )
    _instruments[METRIC_CONTAMINATION_RATE] = _meter.create_gauge(
        name=METRIC_CONTAMINATION_RATE,
        unit="ratio",
        description="Contamination rate",
    )
    _instruments[METRIC_TEMPORAL_ACCURACY] = _meter.create_gauge(
        name=METRIC_TEMPORAL_ACCURACY,
        unit="ratio",
        description="Temporal accuracy",
    )
    _instruments[METRIC_LATENCY_MS] = _meter.create_histogram(
        name=METRIC_LATENCY_MS,
        unit="ms",
        description="Operation latency",
    )
    _instruments[METRIC_COST_PER_CORRECT_RECALL] = _meter.create_gauge(
        name=METRIC_COST_PER_CORRECT_RECALL,
        unit="USD",
        description="Cost per correct recall",
    )
    _instruments[METRIC_MEMORY_SURVIVAL_RATE] = _meter.create_gauge(
        name=METRIC_MEMORY_SURVIVAL_RATE,
        unit="ratio",
        description="Memory survival rate",
    )

    return _meter


def record_metric(name: str, value: float, attributes: dict[str, str] | None = None) -> None:
    """Record a metric value.

    Args:
        name: The metric name (must be from schemas.py constants).
        value: The metric value.
        attributes: Optional metric attributes/labels.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    if name not in ALL_METRIC_NAMES:
        raise ValueError(f"Unknown metric name: '{name}'. Valid: {sorted(ALL_METRIC_NAMES)}")

    if name not in _instruments:
        initialize_metrics()

    instrument = _instruments.get(name)
    if instrument is None:
        return

    if hasattr(instrument, "set"):
        instrument.set(value, attributes=attributes)  # type: ignore[union-attr]
    elif hasattr(instrument, "record"):
        instrument.record(value, attributes=attributes)  # type: ignore[union-attr]
