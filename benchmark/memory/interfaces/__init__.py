"""Segregated memory interfaces (ABCs).

These interfaces define the contracts that all memory modules must follow.
No default implementations. No utility methods.
"""

from benchmark.memory.interfaces.lifecycle import LifecyclePolicy
from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.memory.interfaces.writer import MemoryWriter

__all__ = [
    "LifecyclePolicy",
    "MemoryReader",
    "MemoryWriter",
    "RetrievalStrategy",
]
