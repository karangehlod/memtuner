"""Benchmark Packs — Packaged, versioned open datasets and adapters.

This module provides:
- BenchmarkPack interface (base class for all packs)
- PackRegistry (discovers and resolves packs)
- LongMemEval and LoCoMo adapters
- Private/custom data adapter
"""

from benchmark.packs.base import BenchmarkPack
from benchmark.packs.registry import PackRegistry

__all__ = ["BenchmarkPack", "PackRegistry"]
