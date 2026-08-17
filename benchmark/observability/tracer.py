"""OTel trace facade.

Provides a simplified interface for creating and managing OTel spans.
This module contains NO business logic — it is a pure observability facade.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode, Tracer

from benchmark.observability.schemas import SCHEMA_VERSION

_tracer: Tracer | None = None


def initialize_tracer(service_name: str = "agentic-memory-benchmark") -> Tracer:
    """Initialize the global OTel tracer.

    Args:
        service_name: The service name for trace identification.

    Returns:
        The initialized Tracer instance.
    """
    global _tracer
    _tracer = trace.get_tracer(service_name, SCHEMA_VERSION)
    return _tracer


def get_tracer() -> Tracer:
    """Get the current tracer, initializing if needed.

    Returns:
        The active Tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = initialize_tracer()
    return _tracer


@contextmanager
def create_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Create an OTel span with the given name and attributes.

    Usage:
        with create_span("memory.read", {"module": "episodic"}) as span:
            result = do_work()
            span.set_attribute("result_count", len(result))

    Args:
        name: The span name (must be from schemas.py constants).
        attributes: Optional span attributes.

    Yields:
        The active Span object.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        try:
            yield span
        except Exception as error:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            raise
