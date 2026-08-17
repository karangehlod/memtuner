"""Unit tests for the observability module (logger + tracer)."""

from __future__ import annotations

import json
import logging

import pytest

from benchmark.observability.logger import StructuredFormatter, get_logger, log_decision
from benchmark.observability.tracer import create_span, get_tracer, initialize_tracer


@pytest.mark.unit
class TestStructuredFormatter:
    """Tests for the StructuredFormatter log formatter."""

    def test_format_basic_message(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="benchmark.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "benchmark.test"

    def test_format_includes_exception_info(self) -> None:
        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="benchmark.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "test error"

    def test_format_includes_benchmark_data(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="benchmark.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Decision made",
            args=(),
            exc_info=None,
        )
        record.benchmark_data = {"key": "value"}  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert "data" in data
        assert data["data"]["key"] == "value"


@pytest.mark.unit
class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_logger_with_handler(self) -> None:
        logger = get_logger("test_module")
        assert logger.name == "benchmark.test_module"
        assert len(logger.handlers) >= 1

    def test_idempotent_handler_setup(self) -> None:
        logger = get_logger("test_idempotent")
        handler_count = len(logger.handlers)
        get_logger("test_idempotent")
        assert len(logger.handlers) == handler_count


@pytest.mark.unit
class TestLogDecision:
    """Tests for the log_decision helper."""

    def test_log_decision_without_data(self) -> None:
        logger = get_logger("test_decision")
        # Should not raise
        log_decision(logger, "Simple decision")

    def test_log_decision_with_data(self) -> None:
        logger = get_logger("test_decision_data")
        log_decision(logger, "Complex decision", key="value", count=42)


@pytest.mark.unit
class TestTracerFacade:
    """Tests for the OTel tracer facade."""

    def test_initialize_tracer_returns_tracer(self) -> None:
        tracer = initialize_tracer("test-service")
        assert tracer is not None

    def test_get_tracer_auto_initializes(self) -> None:
        tracer = get_tracer()
        assert tracer is not None

    def test_create_span_context_manager(self) -> None:
        initialize_tracer("test-service")
        with create_span("test.operation") as span:
            assert span is not None

    def test_create_span_with_attributes(self) -> None:
        initialize_tracer("test-service")
        with create_span("test.operation", attributes={"key": "value"}) as span:
            assert span is not None

    def test_create_span_records_exception(self) -> None:
        initialize_tracer("test-service")
        with pytest.raises(RuntimeError, match="test error"), create_span("test.failing"):
            raise RuntimeError("test error")
