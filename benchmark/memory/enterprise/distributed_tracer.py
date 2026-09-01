
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class Span:
    span_id: str
    operation: str
    start_time: float = 0.0
    end_time: float = 0.0
    tags: dict[str, Any] = field(default_factory=dict)

class DistributedTracer:
    """Request-level distributed tracing."""

    def __init__(self):
        self._traces: dict[str, list] = {}
        self._spans: dict[str, Span] = {}

    def start_trace(self, trace_id: str, operation: str) -> Span:
        import time
        span = Span(trace_id, operation, start_time=time.time())
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(span)
        return span

    def add_span(self, trace_id: str, span: Span) -> None:
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(span)
        self._spans[span.span_id] = span

    def finish_trace(self, trace_id: str) -> dict[str, Any]:
        if trace_id not in self._traces:
            return {}
        return {'trace_id': trace_id, 'spans': len(self._traces[trace_id])}

    def query_traces(self, query: dict | None = None) -> list:
        return list(self._traces.keys())

    def get_trace_stats(self) -> dict[str, Any]:
        return {
            'total_traces': len(self._traces),
            'total_spans': len(self._spans),
            'avg_spans_per_trace': len(self._spans) / max(1, len(self._traces)),
        }
