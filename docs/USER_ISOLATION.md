# User Isolation Guarantee

## What "User Isolation" Means in This Benchmark

During a multi-user benchmark run, the store holds memories from many
simulated users simultaneously. The **user isolation guarantee** is:

> **A query from user A will NEVER return a memory that was written for user B.**

This is a hard security boundary, not a soft filter. It is enforced at the
lowest possible layer (the memory store itself), not at the orchestrator or CLI.

---

## Where the Boundary Is Enforced

### Layer 1 — Memory Store (`BaseLongTermStore._filter_candidates`)

Every `read()` call passes through `_filter_candidates()` before any scoring,
indexing, or retrieval strategy is invoked. The **first** operation inside that
method is a hard user_id equality check:

```python
# benchmark/memory/long_term/base_store.py
def _filter_candidates(self, query: ReadQuery) -> dict[str, MemoryEvent]:
    # Step 1 — USER ISOLATION (always runs first, cannot be skipped)
    requesting_user = query.context.user_id
    candidates = {
        mid: event
        for mid, event in self._memories.items()
        if event.user_id == requesting_user     # ← hard boundary
    }
    # Step 2 — type filter (operates on already-scoped candidates)
    # Step 3 — importance threshold (operates on already-scoped candidates)
    ...
```

This candidate dict is then passed to ALL downstream paths:
- The retrieval strategy (BM25, embeddings, hybrid, pgvector)
- The fallback `_compute_relevance_score()` path
- The `_score_candidates()` loop (which also rechecks user_id as defense-in-depth)

Because the strategy receives only the requesting user's memories as the
index, it is structurally impossible for it to return a memory belonging to
another user.

### Layer 2 — Fallback Scoring (`_score_candidates`)

The fallback path contains a second, redundant `user_id` check:

```python
for memory_id, event in candidates.items():
    if event.user_id != user_id:      # defense-in-depth
        continue
```

This second check costs nothing and ensures that even if `_filter_candidates`
were somehow bypassed, the fallback path would still not return cross-user data.

### Layer 3 — Event Model (`MemoryEvent.user_id`)

Every `MemoryEvent` carries an immutable `user_id` field (pydantic frozen model).
It is set at write time and cannot be changed after the fact. This makes it
impossible to retroactively re-assign a memory to a different user.

### Layer 4 — Query Context (`ReadQueryContext.user_id`)

Every `ReadQuery` carries the requesting user's ID in its context. The
orchestrator sets this from the gold dataset's `GoldQuery.user_id`, which is
derived directly from the original conversation data — it is never inferred or
defaulted from the config.

---

## Data Model

```
MemoryEvent
├── id          "M-0042"
├── user_id     "user-0017"       ← owner, set at write time, immutable
├── type        EPISODIC
├── content     "..."
└── ...

ReadQuery
├── query       "What did I say about..."
├── top_k       10
├── context
│   ├── user_id  "user-0017"     ← requester, checked against MemoryEvent.user_id
│   ├── task_id  "..."
│   └── simulated_day  5
└── filters
    ├── memory_types  [...]
    └── min_importance  0.0
```

---

## What Is NOT Isolated

The user isolation guarantee applies to **memory retrieval** only. The following
are shared across all users within a single benchmark run:

| Shared resource | Why it is safe |
|-----------------|----------------|
| The in-memory dict of all events | Read through `_filter_candidates` — never exposed directly |
| Metric aggregations (Recall@K, FPR) | Computed per-query, per-user. Aggregate averages do not reveal individual data |
| Decay and pruning | Applied per-memory by ID; policies operate on scores, not content |
| Retrieval strategy index | Index built from user-scoped candidate pool only (see Layer 1) |
| Cost tracking | Counts operations, not memory content |
| OTel traces | Log metadata (user_id, event_id) but never memory content |

---

## Verification

The isolation guarantee is verified by `tests/unit/test_user_isolation.py`:

```
tests/unit/test_user_isolation.py
├── TestUserIsolationFallbackPath        (8 tests — fallback scoring path)
│   ├── test_user_sees_only_own_memories
│   ├── test_user_b_cannot_see_user_a_memories
│   ├── test_overlapping_memory_ids_no_leakage
│   ├── test_empty_store_returns_empty
│   ├── test_user_with_no_memories_gets_nothing
│   ├── test_many_users_strict_isolation    (10 users × 5 memories each)
│   ├── test_after_prune_isolation_still_holds
│   └── test_write_on_day_preserves_isolation
│
├── TestUserIsolationFilterCandidates    (2 tests — candidate pool boundary)
│   ├── test_filter_candidates_user_scope_is_first
│   └── test_filter_candidates_type_filter_within_user_scope
│
└── TestUserIsolationAllStores           (4 tests — all four store types)
    ├── EpisodicStore
    ├── SemanticStore
    ├── PreferenceStore
    └── EntityStore
```

Run them with:

```bash
# macOS / Linux
python3 -m pytest tests/unit/test_user_isolation.py -v

# Windows (PowerShell)
python -m pytest tests/unit/test_user_isolation.py -v
```

Expected output:

```
14 passed in 0.18s
```

---

## Reporting Isolation in Benchmark Results

Every benchmark run result JSON includes the user ID per query evaluation:

```json
{
  "query": "What time did I say...",
  "user_id": "user-0017",
  "retrieved_memory_ids": ["M-0042", "M-0043"],
  "expected_memory_ids": ["M-0042"],
  "recall_at_k": 1.0
}
```

The run-level summary shows per-user aggregate stats when multiple users are
benchmarked. No user's content is ever included in another user's result.

---

## Threat Model

| Threat | Status | Mitigation |
|--------|--------|------------|
| User A's query retrieves User B's memories | **BLOCKED** | `_filter_candidates` scopes to `user_id` first |
| Retrieval strategy indexes User B's memories for User A's query | **BLOCKED** | Strategy receives only the user-scoped candidate dict |
| Memory content appears in metrics or logs | **BLOCKED** | Metrics operate on IDs and scores only; logs write user_id + event_id, not content |
| User B prunes User A's memories via lifecycle policy | **BLOCKED** | Pruning operates on memory IDs supplied by the policy, which receives only scores (not user_ids); no cross-user ID can be flagged |
| Config `user_id` default leaks across users | **BLOCKED** | Gold dataset always sets explicit user_id per event and query; "user-default" never used in multi-user scenarios |

---

## OWASP Relevance

This control addresses **OWASP Top 10 A01:2021 — Broken Access Control**.

The specific sub-risk is *Insecure Direct Object Reference (IDOR)*: a query
using one user's credentials (user_id in query context) could in principle
retrieve memory objects owned by a different user if the store did not enforce
ownership at retrieval time.

The fix is enforced at the data access layer (the store), not at the API or
orchestration layer, which is the OWASP-recommended location for access
control enforcement.
