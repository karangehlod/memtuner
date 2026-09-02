from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from benchmark.cli.main import _load_repo_env


@pytest.mark.unit
def test_load_repo_env_reads_repository_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    benchmark_package = repo_root / "benchmark" / "cli"
    benchmark_package.mkdir(parents=True)
    env_path = repo_root / ".env"
    env_path.write_text(
        "BENCHMARK_HF_API_TOKEN=test-token\nBENCHMARK_HF_API_BASE_URL=https://api-inference.huggingface.co\n",
        encoding="utf-8",
    )

    fake_main = repo_root / "benchmark" / "cli" / "main.py"
    fake_main.write_text("", encoding="utf-8")

    # Clean up environment before test
    monkeypatch.delenv("BENCHMARK_HF_API_TOKEN", raising=False)
    monkeypatch.delenv("BENCHMARK_HF_API_BASE_URL", raising=False)
    # Also clean up any vars from local .env that might interfere
    monkeypatch.delenv("BENCHMARK_EMBEDDING_MODELS", raising=False)

    monkeypatch.setattr("benchmark.cli.main.__file__", str(fake_main))
    monkeypatch.setitem(sys.modules, "dotenv", None)

    _load_repo_env()

    assert os.environ["BENCHMARK_HF_API_TOKEN"] == "test-token"
    assert os.environ["BENCHMARK_HF_API_BASE_URL"] == "https://api-inference.huggingface.co"

    # Clean up after test to prevent environment pollution in subsequent tests
    monkeypatch.delenv("BENCHMARK_HF_API_TOKEN", raising=False)
    monkeypatch.delenv("BENCHMARK_HF_API_BASE_URL", raising=False)
