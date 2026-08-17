"""Configuration-related exceptions.

Raised when config files cannot be loaded, parsed, or validated.
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base exception for all benchmark errors.

    All custom exceptions in this project inherit from this class
    to enable catch-all handling at the CLI boundary.
    """


class ConfigLoadError(BenchmarkError):
    """Raised when a configuration file cannot be loaded or parsed.

    Examples:
        - File not found
        - Invalid YAML syntax
        - File permission errors
    """


class ConfigValidationError(BenchmarkError):
    """Raised when a configuration file fails schema validation.

    Examples:
        - Missing required fields
        - Invalid field values
        - Unknown memory module names
    """
