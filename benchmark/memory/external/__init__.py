"""External memory-system adapters (arena backends).

Adapters in this package wrap third-party memory systems (Mem0, Zep, Letta)
behind the same MemoryWriter/MemoryReader contract as MemTuner's own stores,
so the benchmark harness can run them through an identical protocol.

Design rules (docs/ARENA_PLAN.md, Phase 1):
- Vendor-recommended defaults, committed verbatim in configs/arena/.
- Fresh namespace per benchmark cell; teardown verified.
- External systems rewrite memories at ingest, so gold-ID metrics rely on
  provenance metadata threaded through the vendor API; when the vendor
  cannot preserve it, cells must run judge-primary scoring instead.
"""

from benchmark.memory.external.graphiti_store import GraphitiStore
from benchmark.memory.external.mem0_store import Mem0Store
from benchmark.memory.external.zep_store import ZepStore

__all__ = ["GraphitiStore", "Mem0Store", "ZepStore"]
