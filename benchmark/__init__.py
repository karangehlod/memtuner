"""MemTuner — adaptive benchmarking for AI agent memory retrieval."""

from __future__ import annotations


def _read_version() -> str:
    # 1. Installed package metadata (pip install / editable install)
    try:
        from importlib.metadata import version
        return version("memtuner")
    except Exception:
        pass
    # 2. Source tree without install — parse pyproject.toml directly.
    # This keeps version in ONE place (pyproject.toml) with no stale fallback.
    try:
        import re
        from pathlib import Path
        _pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        _m = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), re.M)
        if _m:
            return _m.group(1)
    except Exception:
        pass
    return "unknown"


__version__: str = _read_version()
