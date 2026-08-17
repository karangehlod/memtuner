"""Leakage checker for benchmark evaluation.

Detects if evaluation queries appear verbatim (or near-verbatim) in the
indexed memory corpus — which would mean the benchmark is testing
memorization, not retrieval generalization.
"""

import re
from dataclasses import dataclass, field


@dataclass
class LeakageReport:
    leakage_rate: float
    leaked_count: int
    total_queries: int
    total_memories: int
    leaked_queries: list[dict]
    is_clean: bool

    def summary(self) -> str:
        status = "CLEAN" if self.is_clean else "WARNING: LEAKAGE DETECTED"
        return (
            f"Leakage check: {status} | "
            f"{self.leaked_count}/{self.total_queries} queries leaked "
            f"({self.leakage_rate*100:.1f}%) | "
            f"{self.total_memories} memories checked"
        )


def _normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower())


class LeakageChecker:
    """Detects query-in-corpus leakage before running evaluation.

    A query is 'leaked' if any min_overlap_chars-length substring of its
    normalized text appears in any memory content.

    Algorithm: O(total_memory_chars + n_queries × max_query_len)
    - Build a set of all min_n-grams from memory contents (one pass).
    - For each query, slide a window of width min_n over it and probe the set.
    This replaces the O(queries × memories × query_len²) nested loop.
    """

    def __init__(self, min_overlap_chars: int = 20):
        self.min_overlap_chars = min_overlap_chars
        self.leaked_queries: list[dict] = []
        self.total_queries: int = 0
        self.total_memories: int = 0

    def check(self, queries: list[str], memory_contents: dict[str, str]) -> "LeakageReport":
        """Check for leakage between queries and memory contents.

        Args:
            queries: List of query strings to check.
            memory_contents: Dict mapping memory_id -> content string.

        Returns:
            LeakageReport with leakage_rate, leaked_queries, is_clean.
        """
        min_n = self.min_overlap_chars
        self.total_queries = len(queries)
        self.total_memories = len(memory_contents)
        self.leaked_queries = []

        # Build ngram → memory_id index in a single pass over all memory content.
        # Using the first memory_id that contains each ngram is enough for reporting.
        ngram_to_memory: dict[str, str] = {}
        for memory_id, content in memory_contents.items():
            nc = _normalize(content)
            for i in range(len(nc) - min_n + 1):
                ng = nc[i:i + min_n]
                if ng not in ngram_to_memory:
                    ngram_to_memory[ng] = memory_id

        # For each query, probe the ngram set with a sliding window.
        for query in queries:
            norm_q = _normalize(query)
            if len(norm_q) < min_n:
                continue
            for i in range(len(norm_q) - min_n + 1):
                fragment = norm_q[i:i + min_n]
                memory_id = ngram_to_memory.get(fragment)
                if memory_id is not None:
                    self.leaked_queries.append({
                        "query": query,
                        "memory_id": memory_id,
                        "overlap": fragment,
                        "overlap_chars": min_n,
                    })
                    break  # one match per query is enough

        leakage_rate = len(self.leaked_queries) / max(1, self.total_queries)
        return LeakageReport(
            leakage_rate=leakage_rate,
            leaked_count=len(self.leaked_queries),
            total_queries=self.total_queries,
            total_memories=self.total_memories,
            leaked_queries=self.leaked_queries,
            is_clean=leakage_rate < 0.01,
        )
