"""MemTuner configuration loader.

Priority (highest to lowest):
  1. Environment variables  (BENCHMARK_* in .env or shell)
  2. configs/benchmark_config.yaml
  3. Built-in defaults

Usage:
    from scripts.config import cfg

    cfg.datasets.name(11873)        # → "SQuAD"
    cfg.composite.weights           # → {"recall": 0.40, ...}
    cfg.composite.recall_gate       # → 0.01
    cfg.colors.for_strategy("hybrid") # → "#2a78d6"
    cfg.reporting.output_dir        # → Path("data/output")
    cfg.reporting.plot_dpi          # → 150 (or BENCHMARK_DPI env override)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Load .env before anything else (python-dotenv is a core dependency)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)  # shell vars win over .env
except ImportError:
    pass  # python-dotenv not installed — rely on pre-set shell vars


# ── Config path resolution ─────────────────────────────────────────────────────

def _find_config() -> Path:
    """Find benchmark_config.yaml relative to this file or via env var."""
    env_path = os.environ.get("BENCHMARK_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"BENCHMARK_CONFIG={env_path!r} not found")
    default = Path(__file__).resolve().parent.parent / "configs" / "benchmark_config.yaml"
    if default.exists():
        return default
    raise FileNotFoundError(
        f"benchmark_config.yaml not found at {default}. "
        "Set BENCHMARK_CONFIG env var or create the file."
    )


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Sub-config classes ─────────────────────────────────────────────────────────

@dataclass
class DatasetsConfig:
    query_count_to_name: dict[int, str]
    display_order: list[str]

    def name(self, query_count: int, fallback: str | None = None) -> str:
        """Return dataset name for a given total_queries value."""
        default = fallback or f"unknown_{query_count}"
        return self.query_count_to_name.get(query_count, default)

    def query_count(self, name: str) -> int | None:
        """Reverse lookup: name → query count."""
        return next((k for k, v in self.query_count_to_name.items() if v == name), None)


@dataclass
class CompositeConfig:
    weights:            dict[str, float]
    recall_gate:        float
    significance_margin: float

    @property
    def formula_str(self) -> str:
        w = self.weights
        return (
            f"gate(R@10≥{self.recall_gate}) × "
            f"({w['recall']}·R@10 + {w['precision']}·P@10 + {w['mrr']}·MRR + {w['temporal']}·TA)"
        )

    def score(self, recall: float, precision: float, mrr: float, temporal: float) -> float:
        if recall < self.recall_gate:
            return 0.0
        w = self.weights
        return w["recall"] * recall + w["precision"] * precision + w["mrr"] * mrr + w["temporal"] * temporal


@dataclass
class EvaluationConfig:
    recall_k:              int
    bootstrap_resamples:   int
    early_stop_min_delta:  float
    early_stop_patience:   int


@dataclass
class ReportingConfig:
    _output_dir: str
    _plots_dir:  str
    plot_dpi:    int
    max_phase_tokens_in_filename: int
    _project_root: Path

    @property
    def output_dir(self) -> Path:
        p = Path(self._output_dir)
        return p if p.is_absolute() else self._project_root / p

    @property
    def plots_dir(self) -> Path:
        p = Path(self._plots_dir)
        return p if p.is_absolute() else self._project_root / p


@dataclass
class ColorsConfig:
    _palette:         dict[str, str]   # name → hex
    _strategy_colors: dict[str, str]   # strategy → palette key
    _decay_colors:    dict[str, str]   # decay → palette key
    _memory_colors:   dict[str, str]   # memory type → palette key

    def hex(self, name: str) -> str:
        """Return hex colour for a palette name."""
        return self._palette.get(name, "#898781")

    def for_strategy(self, strategy: str) -> str:
        key = self._strategy_colors.get(strategy, "gray")
        return self.hex(key)

    def for_decay(self, decay: str) -> str:
        key = self._decay_colors.get(decay, "gray")
        return self.hex(key)

    def for_memory(self, memory_type: str) -> str:
        key = self._memory_colors.get(memory_type, "gray")
        return self.hex(key)

    @property
    def strategy_map(self) -> dict[str, str]:
        """Return {strategy_name: hex_color} mapping."""
        return {s: self.for_strategy(s) for s in self._strategy_colors}

    @property
    def decay_map(self) -> dict[str, str]:
        return {d: self.for_decay(d) for d in self._decay_colors}

    @property
    def palette(self) -> dict[str, str]:
        return dict(self._palette)


@dataclass
class BenchmarkConfig:
    datasets:    DatasetsConfig
    composite:   CompositeConfig
    evaluation:  EvaluationConfig
    reporting:   ReportingConfig
    colors:      ColorsConfig
    phase_abbreviations: dict[str, str]


# ── Builder ────────────────────────────────────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _build(raw: dict, project_root: Path) -> BenchmarkConfig:
    ds_raw = raw.get("datasets", {})
    # query_count_to_name: YAML keys are ints; after yaml.safe_load they may be ints already
    qctn: dict[int, str] = {int(k): str(v) for k, v in ds_raw.get("query_count_to_name", {}).items()}
    datasets = DatasetsConfig(
        query_count_to_name=qctn,
        display_order=ds_raw.get("display_order", list(qctn.values())),
    )

    comp_raw = raw.get("composite", {})
    w_raw    = comp_raw.get("weights", {})
    composite = CompositeConfig(
        weights={
            "recall":    _env_float("BENCHMARK_COMPOSITE_W_RECALL",    w_raw.get("recall",    0.40)),
            "precision": _env_float("BENCHMARK_COMPOSITE_W_PRECISION", w_raw.get("precision", 0.25)),
            "mrr":       _env_float("BENCHMARK_COMPOSITE_W_MRR",       w_raw.get("mrr",       0.20)),
            "temporal":  _env_float("BENCHMARK_COMPOSITE_W_TEMPORAL",  w_raw.get("temporal",  0.15)),
        },
        recall_gate        =_env_float("BENCHMARK_COMPOSITE_RECALL_GATE",    comp_raw.get("recall_gate",         0.01)),
        significance_margin=_env_float("BENCHMARK_COMPOSITE_SIG_MARGIN",     comp_raw.get("significance_margin", 0.01)),
    )

    ev_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        recall_k             =_env_int  ("BENCHMARK_RECALL_K",              ev_raw.get("recall_k",             10)),
        bootstrap_resamples  =_env_int  ("BENCHMARK_BOOTSTRAP_RESAMPLES",   ev_raw.get("bootstrap_resamples",  1000)),
        early_stop_min_delta =_env_float("BENCHMARK_EARLY_STOP_MIN_DELTA",  ev_raw.get("early_stop_min_delta", 0.005)),
        early_stop_patience  =_env_int  ("BENCHMARK_EARLY_STOP_PATIENCE",   ev_raw.get("early_stop_patience",  3)),
    )

    rep_raw = raw.get("reporting", {})
    reporting = ReportingConfig(
        _output_dir =_env_str("BENCHMARK_OUTPUT_DIR",  rep_raw.get("output_dir", "data/output")),
        _plots_dir  =_env_str("BENCHMARK_PLOTS_DIR",   rep_raw.get("plots_dir",  "data/output/plots")),
        plot_dpi    =_env_int("BENCHMARK_DPI",          rep_raw.get("plot_dpi",   150)),
        max_phase_tokens_in_filename=rep_raw.get("max_phase_tokens_in_filename", 4),
        _project_root=project_root,
    )

    col_raw  = raw.get("colors", {})
    # Build flat palette from the YAML (strip non-hex keys like strategy_colors etc.)
    palette  = {k: v for k, v in col_raw.items() if isinstance(v, str) and v.startswith("#")}
    colors = ColorsConfig(
        _palette        =palette,
        _strategy_colors=col_raw.get("strategy_colors", {}),
        _decay_colors   =col_raw.get("decay_colors",    {}),
        _memory_colors  =col_raw.get("memory_colors",   {}),
    )

    phase_abbrev: dict[str, str] = raw.get("phase_abbreviations", {})

    return BenchmarkConfig(
        datasets=datasets,
        composite=composite,
        evaluation=evaluation,
        reporting=reporting,
        colors=colors,
        phase_abbreviations=phase_abbrev,
    )


# ── Module-level singleton ─────────────────────────────────────────────────────

def load_config(config_path: Path | None = None) -> BenchmarkConfig:
    """Load config from YAML + env overrides. Cached after first call."""
    path         = config_path or _find_config()
    project_root = path.parent.parent  # configs/ → project root
    raw          = _load_yaml(path)
    return _build(raw, project_root)


# Lazy singleton — loaded on first import of `cfg`
class _Proxy:
    _cfg: BenchmarkConfig | None = None

    def _ensure(self) -> BenchmarkConfig:
        if self._cfg is None:
            self._cfg = load_config()
        return self._cfg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure(), name)

    def reload(self, config_path: Path | None = None) -> None:
        """Force reload — useful in tests or after YAML edits."""
        self._cfg = load_config(config_path)


cfg = _Proxy()
