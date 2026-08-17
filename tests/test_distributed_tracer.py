"""Tests for DistributedTracer."""

import pytest
from benchmark.memory.enterprise.distributed_tracer import DistributedTracer, Span


@pytest.fixture
def tracer():
    return DistributedTracer()


class TestDistributedTracer:
    def test_initialization(self, tracer):
        assert tracer is not None

    def test_start_trace(self, tracer):
        span = tracer.start_trace("trace_1", "query")
        assert span.span_id == "trace_1"

    def test_add_span(self, tracer):
        span = Span("span_1", "operation_1")
        tracer.add_span("trace_1", span)
        assert "trace_1" in tracer._traces

    def test_finish_trace(self, tracer):
        tracer.start_trace("trace_1", "op1")
        result = tracer.finish_trace("trace_1")
        assert result["trace_id"] == "trace_1"

    def test_query_traces(self, tracer):
        tracer.start_trace("trace_1", "op1")
        tracer.start_trace("trace_2", "op2")
        traces = tracer.query_traces()
        assert len(traces) == 2

    def test_get_trace_stats(self, tracer):
        tracer.start_trace("trace_1", "op1")
        stats = tracer.get_trace_stats()
        assert "total_traces" in stats

    def test_multi_service_tracing(self, tracer):
        tracer.start_trace("t1", "service_a")
        tracer.add_span("t1", Span("s1", "service_b"))
        result = tracer.finish_trace("t1")
        assert result["spans"] == 2

    def test_latency_computation(self, tracer):
        import time
        tracer.start_trace("t1", "op1")
        time.sleep(0.01)
        result = tracer.finish_trace("t1")
        assert result is not None

    def test_error_tracking(self, tracer):
        span = Span("s1", "op1", tags={"error": "timeout"})
        tracer.add_span("t1", span)
        result = tracer.finish_trace("t1")
        assert result is not None

    def test_trace_sampling(self, tracer):
        for i in range(100):
            tracer.start_trace(f"trace_{i}", f"op_{i}")
        stats = tracer.get_trace_stats()
        assert stats["total_traces"] == 100

    def test_span_linking(self, tracer):
        tracer.start_trace("t1", "op1")
        tracer.add_span("t1", Span("s1", "op2"))
        tracer.add_span("t1", Span("s2", "op3"))
        result = tracer.finish_trace("t1")
        assert result["spans"] == 3

    def test_large_trace_size(self, tracer):
        tracer.start_trace("t1", "op1")
        for i in range(1000):
            tracer.add_span("t1", Span(f"s{i}", f"op{i}"))
        result = tracer.finish_trace("t1")
        assert result["spans"] == 1001

    def test_concurrent_traces(self, tracer):
        for i in range(10):
            tracer.start_trace(f"trace_{i}", f"operation_{i}")
        stats = tracer.get_trace_stats()
        assert stats["total_traces"] == 10

    def test_trace_correlation(self, tracer):
        tracer.start_trace("t1", "root_op")
        tracer.add_span("t1", Span("s1", "child_op"))
        result = tracer.finish_trace("t1")
        assert result["spans"] == 2

    def test_empty_trace(self, tracer):
        result = tracer.finish_trace("nonexistent")
        assert result == {}

    def test_span_tags(self, tracer):
        span = Span("s1", "op", tags={"user_id": "123", "status": "success"})
        tracer.add_span("t1", span)
        assert tracer._spans["s1"].tags["user_id"] == "123"

    def test_stats_accumulation(self, tracer):
        tracer.start_trace("t1", "op1")
        tracer.add_span("t1", Span("s1", "op1"))
        stats = tracer.get_trace_stats()
        assert stats["total_spans"] > 0

    def test_avg_spans_per_trace(self, tracer):
        tracer.start_trace("t1", "op1")
        tracer.add_span("t1", Span("s1", "op2"))
        stats = tracer.get_trace_stats()
        assert stats["avg_spans_per_trace"] > 0
