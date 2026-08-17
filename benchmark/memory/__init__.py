"""Segregated memory interfaces.

Provides MemoryWriter, MemoryReader, and LifecyclePolicy ABCs.
"""

from benchmark.memory.interfaces.lifecycle import LifecyclePolicy
from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter

__all__ = [
    "LifecyclePolicy",
    "MemoryReader",
    "MemoryWriter",
]
