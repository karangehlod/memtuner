"""Memory-related exceptions.

Raised during memory write, read, or factory resolution operations.
"""

from __future__ import annotations

from benchmark.exceptions.config_errors import BenchmarkError


class MemoryWriteError(BenchmarkError):
    """Raised when a memory write operation fails.

    Examples:
        - Storage backend unavailable
        - Invalid memory event data that passed validation
        - Write timeout
    """


class MemoryReadError(BenchmarkError):
    """Raised when a memory read operation fails.

    Examples:
        - Storage backend unavailable
        - Query execution failure
        - Read timeout
    """


class RegistryResolutionError(BenchmarkError):
    """Raised when the factory registry cannot resolve an implementation.

    Examples:
        - Unknown module name in config
        - Implementation class not registered
        - Constructor arguments mismatch
    """
