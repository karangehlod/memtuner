"""Structured logging facade.

Provides JSON-structured logging with automatic trace context injection.
All logs include trace_id and span_id when available.
NO print() calls should exist anywhere in the codebase — use this module.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from opentelemetry import trace


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter that includes OTel trace context.

    Every log entry includes:
    - timestamp
    - level
    - message
    - trace_id (if available)
    - span_id (if available)
    - Any extra fields passed via the `extra` dict
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as structured JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        span_context = trace.get_current_span().get_span_context()
        if span_context.trace_id != 0:
            log_entry["trace_id"] = format(span_context.trace_id, "032x")
            log_entry["span_id"] = format(span_context.span_id, "016x")

        if hasattr(record, "benchmark_data"):
            log_entry["data"] = record.benchmark_data  # type: ignore[attr-defined]

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for the given module.

    Args:
        name: The logger name (typically __name__ of the calling module).

    Returns:
        A configured Logger with structured JSON output.
    """
    logger = logging.getLogger(f"benchmark.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def log_decision(
    logger: logging.Logger,
    message: str,
    **data: Any,
) -> None:
    """Log a benchmark decision with structured data.

    Every decision point in the benchmark should use this function
    to ensure consistent structured logging with trace context.

    Args:
        logger: The logger instance.
        message: Human-readable decision description.
        **data: Structured data to include in the log entry.
    """
    extra = {"benchmark_data": data} if data else {}
    logger.info(message, extra=extra)
