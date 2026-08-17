"""Input validation utilities for security hardening.

Provides validation functions for paths, strings, and config values
to prevent path traversal, injection, and oversized inputs.
"""

from __future__ import annotations

from pathlib import Path

from benchmark.exceptions.config_errors import ConfigValidationError

# Maximum file size for config files (1 MB)
MAX_CONFIG_FILE_SIZE_BYTES: int = 1_048_576

# Maximum file size for gold datasets (10 MB)
MAX_GOLD_DATASET_SIZE_BYTES: int = 10_485_760

# Maximum length for string fields
MAX_STRING_FIELD_LENGTH: int = 10_000

# Allowed config file extensions
ALLOWED_CONFIG_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml"})

# Allowed dataset file extensions
ALLOWED_DATASET_EXTENSIONS: frozenset[str] = frozenset({".json"})


def validate_file_path_safe(
    file_path: Path,
    allowed_extensions: frozenset[str],
    max_size_bytes: int,
    base_directory: Path | None = None,
) -> Path:
    """Validate a file path for security concerns.

    Checks:
    - File exists and is a regular file
    - Extension is in the allowed set
    - File size is within limits
    - Path does not escape the base directory (if provided)

    Args:
        file_path: The file path to validate.
        allowed_extensions: Set of allowed file extensions.
        max_size_bytes: Maximum allowed file size in bytes.
        base_directory: Optional base directory for path traversal check.

    Returns:
        The resolved (absolute) path.

    Raises:
        ConfigValidationError: If any validation check fails.
    """
    resolved = file_path.resolve()

    if base_directory is not None:
        resolved_base = base_directory.resolve()
        if not str(resolved).startswith(str(resolved_base)):
            raise ConfigValidationError(
                f"Path traversal detected: {file_path} escapes {base_directory}"
            )

    if not resolved.exists():
        raise ConfigValidationError(f"File not found: {resolved}")

    if not resolved.is_file():
        raise ConfigValidationError(f"Path is not a regular file: {resolved}")

    if resolved.suffix.lower() not in allowed_extensions:
        raise ConfigValidationError(
            f"Disallowed file extension '{resolved.suffix}'. Allowed: {sorted(allowed_extensions)}"
        )

    file_size = resolved.stat().st_size
    if file_size > max_size_bytes:
        raise ConfigValidationError(
            f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)"
        )

    return resolved


def validate_config_path(config_path: Path) -> Path:
    """Validate a configuration file path.

    Args:
        config_path: Path to the config file.

    Returns:
        Resolved, validated path.

    Raises:
        ConfigValidationError: If validation fails.
    """
    return validate_file_path_safe(
        file_path=config_path,
        allowed_extensions=ALLOWED_CONFIG_EXTENSIONS,
        max_size_bytes=MAX_CONFIG_FILE_SIZE_BYTES,
    )


def validate_dataset_path(dataset_path: Path) -> Path:
    """Validate a gold dataset file path.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        Resolved, validated path.

    Raises:
        ConfigValidationError: If validation fails.
    """
    return validate_file_path_safe(
        file_path=dataset_path,
        allowed_extensions=ALLOWED_DATASET_EXTENSIONS,
        max_size_bytes=MAX_GOLD_DATASET_SIZE_BYTES,
    )


def validate_output_directory(output_path: Path) -> Path:
    """Validate and prepare an output directory.

    Creates the directory if it doesn't exist. Ensures it's a directory.

    Args:
        output_path: Desired output directory path.

    Returns:
        Resolved, validated directory path.

    Raises:
        ConfigValidationError: If the path exists but is not a directory.
    """
    resolved = output_path.resolve()

    if resolved.exists() and not resolved.is_dir():
        raise ConfigValidationError(f"Output path exists but is not a directory: {resolved}")

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def sanitize_run_id(run_id: str) -> str:
    """Sanitize a run ID to prevent injection.

    Only allows alphanumeric characters, hyphens, and underscores.

    Args:
        run_id: The run ID to sanitize.

    Returns:
        The sanitized run ID.

    Raises:
        ConfigValidationError: If the run ID contains invalid characters.
    """
    if not run_id:
        raise ConfigValidationError("Run ID cannot be empty")

    if len(run_id) > 128:
        raise ConfigValidationError(f"Run ID too long: {len(run_id)} characters (max 128)")

    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if not all(char in allowed_chars for char in run_id):
        invalid = set(run_id) - allowed_chars
        raise ConfigValidationError(f"Run ID contains invalid characters: {invalid}")

    return run_id
