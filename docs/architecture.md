# Architecture & Design Guide

> Complete technical documentation for the Agentic Memory Benchmark.
> Covers storage design, user isolation, memory type comparison, extensibility,
> and the data behind every benchmark run.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [What It Can Do](#2-what-it-can-do)
3. [Storage Architecture](#3-storage-architecture)
4. [Memory Type Taxonomy](#4-memory-type-taxonomy)
5. [User Isolation & Security](#5-user-isolation--security)
6. [Short-Term vs Long-Term: Head-to-Head Comparison](#6-short-term-vs-long-term-head-to-head-comparison)
7. [Data Behind Each Run](#7-data-behind-each-run)
8. [How the Benchmark Executes](#8-how-the-benchmark-executes)
9. [How to Replace Any Component](#9-how-to-replace-any-component)
10. [Database & Persistence Adapter Guide](#10-database--persistence-adapter-guide)
11. [Confidence Scoring Model](#11-confidence-scoring-model)
12. [Cost Model](#12-cost-model)
13. [Observability & Tracing](#13-observability--tracing)
14. [FAQ & Honest Limitations](#14-faq--honest-limitations)

---

## 1. What This Project Is

The Agentic Memory Benchmark is a **framework for measuring how well AI agent
memory systems work**. It is NOT a memory system itself — it is a **test harness**
with pluggable reference implementations.

```
┌──────────────────────────────────────────────────────┐
│                    What We Are                       │
├──────────────────────────────────────────────────────┤
│ ✅ A reproducible benchmark framework                │
│ ✅ A set of memory module reference implementations  │
│ ✅ A pluggable architecture where any piece swaps out│
│ ✅ A measurement tool: recall, FPR, temporal, cost   │
│ ✅ Multi-user, multi-turn conversation simulation    │
├──────────────────────────────────────────────────────┤
│                  What We Are NOT                     │
├──────────────────────────────────────────────────────┤
│ ❌ A production memory database                      │
│ ❌ A vector search engine                            │
│ ❌ A replacement for pgvector / Pinecone / Weaviate  │
│ ❌ A real LLM inference pipeline                     │
└──────────────────────────────────────────────────────┘
```

The reference implementations use **in-memory Python dictionaries** with
`SequenceMatcher`-based text similarity. This is **intentional** — the
benchmark measures the *structure* of memory (decay, retrieval, isolation,
cost), not raw embedding quality. Any component can be swapped for a
production backend (see [Section 9](#9-how-to-replace-any-component)).

---

## 2. What It Can Do

### Metrics It Computes

| Metric | Formula | What It Answers |
|--------|---------|-----------------|
| **Recall@K** | `\|Retrieved ∩ Gold\| / \|Gold\|` | "Did we find the right memories?" |
| **False Positive Rate** | `\|Retrieved \ Gold\| / \|Retrieved\|` | "How much noise crept in?" |
| **Temporal Accuracy** | Fraction within expected time window | "Did we return memories from the right time?" |
| **Memory Survival Rate** | `\|Alive(day)\| / \|Injected\|` | "How reliably do memories persist?" |
| **Confidence Score** | `score * 0.6 + decay_factor * 0.4` | "How certain is each retrieval?" |
| **Cost per Correct Recall** | `total_cost / correct_recalls` | "What does each correct answer cost?" |

### Capabilities

- Run **any combination** of 7 memory modules against 3 gold-truth scenarios.
- Simulate **multi-user, multi-turn conversations** with follow-ups and contradictions.
- Track **token costs** (GPT-4o pricing) and **storage costs** per operation.
- Generate **JSON, CSV, and text reports** with full metric breakdowns.
- **Compare two runs** side-by-side (A/B testing memory configurations).
- Browse results in a **web dashboard** (FastAPI explorer).
- Full **OpenTelemetry** tracing — every read, write, and decision is a span.
- **Deterministic replay** — same config + seed = identical results, always.

---

## 3. Storage Architecture

### Current: In-Memory Reference Implementation

All 7 memory modules store data in **Python dictionaries** scoped by memory ID.
There is no external database. This is the reference implementation.

```
┌───────────────────────────────────────────────────────────┐
│                   Storage Schema (In-Memory)              │
│                                                           │
│  EpisodicStore._memories: dict[str, MemoryEvent]          │
│  EpisodicStore._creation_days: dict[str, int]             │
│                                                           │
│  Each MemoryEvent contains:                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ id:         "M-001"                                 │  │
│  │ user_id:    "user-alice"                            │  │
│  │ type:       MemoryType.EPISODIC                     │  │
│  │ content:    "User prefers Postgres for vectors"     │  │
│  │ timestamp:  2026-01-01T00:00:00Z                    │  │
│  │ importance: 0.85                                    │  │
│  │ entities:   ["user", "postgres"]                    │  │
│  │ task_id:    "db_selection"                          │  │
│  │ metadata:   {}                                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  Storage Operations:                                      │
│  • write(event)        → O(1) dict insertion              │
│  • write_on_day(e, d)  → O(1) dict insertion + day tag    │
│  • read(query)         → O(n) linear scan + similarity    │
│  • prune(ids)          → O(k) dict deletion               │
│  • count()             → O(1) len()                       │
│  • clear()             → O(1) dict.clear()                │
└───────────────────────────────────────────────────────────┘
```

### Storage Per Module Type

| Module | Container | Keying Strategy | Capacity | Eviction |
|--------|-----------|----------------|----------|----------|
| **EpisodicBuffer** | `deque[MemoryEvent]` | FIFO position | Fixed (default 50) | Oldest evicted |
| **ContextBuffer** | `dict[task_id, list[MemoryEvent]]` | Task-scoped lists | Unbounded per task | Manual `clear_task()` |
| **Scratchpad** | `dict[id, MemoryEvent]` | Memory ID | Unbounded | Manual `clear()` |
| **EpisodicStore** | `dict[id, MemoryEvent]` + `dict[id, int]` | Memory ID + day | Unbounded | Policy-driven pruning |
| **PreferenceStore** | `dict[id, MemoryEvent]` + `dict[id, int]` | Memory ID + day | Unbounded | Policy-driven pruning |
| **SemanticStore** | `dict[id, MemoryEvent]` + `dict[id, int]` | Memory ID + day | Unbounded | Manual `remove()` |
| **EntityStore** | `dict[id, MemoryEvent]` + `dict[id, int]` | Memory ID + day | Unbounded | Manual `remove()` |

### Why In-Memory?

The benchmark needs:

1. **Determinism** — same seed produces identical results. External DBs add non-determinism (network, query planner, caching).
2. **Speed** — 445 tests in <1 second. A DB would add orders of magnitude.
3. **Zero dependencies** — no Docker, no migrations, no connection strings.
4. **Measurability** — we control every variable. Decay, similarity, and scoring are transparent.

For production use, see [Section 10: Database Adapter Guide](#10-database--persistence-adapter-guide).

---

## 4. Memory Type Taxonomy

```
Memory Types (MemoryType enum)
├── EPISODIC    → "What happened" — events, conversations, actions
├── SEMANTIC    → "What is known" — facts, definitions, knowledge
├── PREFERENCE  → "What the user wants" — settings, choices, styles
└── ENTITY      → "Who/what exists" — people, orgs, products
```

### Memory Type × Module Mapping

Any memory type can go into any module, but certain pairings are natural:

| Memory Type | Best Short-Term Module | Best Long-Term Module | Why |
|-------------|----------------------|----------------------|-----|
| `EPISODIC` | EpisodicBuffer | EpisodicStore | Buffer captures recent events; store applies decay for long retention |
| `SEMANTIC` | Scratchpad | SemanticStore | Scratch for working facts; store for persistent knowledge with slow decay |
| `PREFERENCE` | ContextBuffer | PreferenceStore | Context for current-session prefs; store for persistent prefs with task-boost |
| `ENTITY` | EpisodicBuffer | EntityStore | Buffer for recent mentions; store applies entity-name boosting |

### Per-Module Feature Matrix

| Feature | EpisodicBuffer | ContextBuffer | Scratchpad | EpisodicStore | PreferenceStore | SemanticStore | EntityStore |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Decay | ✗ | ✗ | ✗ | ✓ (exp) | ✓ (exp) | ✓ (exp) | ✓ (exp) |
| Day-tagged write | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Task scoping | ✗ | ✓ | ✗ | ✗ | ✓ (boost) | ✗ | ✗ |
| Entity boosting | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| User isolation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Confidence score | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Capacity limit | ✓ (FIFO) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Temperature tiers | HOT only | HOT only | HOT only | HOT/WARM/COLD | HOT/WARM/COLD | HOT/WARM/COLD | HOT/WARM/COLD |
| Pruning | Auto (FIFO) | Manual | Manual | Policy-driven | Policy-driven | Manual | Manual |
| Keyword boost | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## 5. User Isolation & Security

### How User Isolation Works

Every memory event carries a `user_id` field. Every read query carries a
`user_id` in its context. At read time, **every module filters by user_id
before scoring**.

```python
# Inside every memory module's read/score method:
for memory_id, event in self._memories.items():
    if event.user_id != user_id:
        continue  # Skip — this memory belongs to a different user
    # ... score and return
```

### Isolation Guarantees

| Property | Guarantee | Enforced By |
|----------|-----------|-------------|
| Read isolation | User A cannot retrieve User B's memories | Every `read()` method filters by `user_id` |
| Write isolation | User A's writes are tagged with User A's ID | `MemoryEvent.user_id` field (set at injection) |
| No cross-user scoring | Similarity scoring never considers other users' data | Filter happens before scoring loop |
| Contract tested | Formal contract tests verify isolation across all 7 modules | `TestUserIsolationContract` in `tests/contract/` |

### Security Boundaries (Honest Assessment)

```
┌──────────────────────────────────────────────────────────┐
│              Current Security Boundary                   │
├──────────────────────────────────────────────────────────┤
│ ✅ Application-level user_id filtering on every read     │
│ ✅ Contract tests ensure no module leaks cross-user data │
│ ✅ Immutable (frozen) pydantic models prevent tampering  │
│ ✅ user_id flows from gold dataset → event → query → filter│
│                                                          │
│ ❌ No authentication — user_id is a plain string         │
│ ❌ No encryption at rest — in-memory Python dicts        │
│ ❌ No row-level security — no database layer             │
│ ❌ No access audit log — only OTel spans                 │
│ ❌ No tenant isolation — single Python process           │
└──────────────────────────────────────────────────────────┘
```

**This is appropriate for a benchmark tool.** For production multi-tenant
systems, you would replace the in-memory stores with a database adapter
that enforces row-level security (see [Section 10](#10-database--persistence-adapter-guide)).

---

## 6. Short-Term vs Long-Term: Head-to-Head Comparison

This section explains **what would happen** when you run the same query
against different module combinations, and **why the results differ**.

### Same Query, Different Modules

Consider the query: `"Which database did Alice prefer?"` on simulated day 10.
Memory `M-001` was injected on day 0 with content `"User prefers Postgres
over Pinecone for vector storage"`, importance=0.85, user=`user-alice`.

#### Short-Term: EpisodicBuffer

```
Scoring: text_similarity * 0.6 + keyword_overlap * 0.4
Decay:   None (always fresh)
Tier:    Always HOT
Result:  score=0.42, confidence=0.65, decay_factor=1.0

Behavior: If M-001 is still in the buffer (hasn't been evicted by
newer events), it scores moderately via text similarity. No decay
penalty. But if 50+ events were injected since day 0, M-001 was
already FIFO-evicted and won't appear at all.
```

#### Short-Term: ContextBuffer

```
Scoring: text_similarity (SequenceMatcher)
Decay:   None
Tier:    Always HOT
Scope:   Task-scoped — only returns memories with matching task_id

Behavior: Only returns M-001 if the query's task_id matches
"db_selection". If queried with task_id="cache_selection", returns
nothing — even though the content matches.
```

#### Long-Term: EpisodicStore

```
Scoring: text_similarity * importance * decay_factor
Decay:   e^(-0.05 * 10) = 0.607
Tier:    WARM (0.3 < 0.607 < 0.7)
Result:  score=0.42 * 0.85 * 0.607 = 0.217, confidence=0.37

Behavior: M-001 is always present (no FIFO eviction). But after 10
days, decay reduces both score and confidence. The memory is WARM,
not HOT. After ~25 days it becomes COLD. After ~40 days it may
be pruned if a score-threshold policy is active.
```

#### Long-Term: PreferenceStore

```
Scoring: text_similarity * importance * decay_factor * task_boost
Decay:   e^(-0.05 * 10) = 0.607
Boost:   1.2x if task_id matches, 1.0x otherwise
Tier:    WARM
Result:  score = min(1.0, 0.42 * 0.85 * 0.607 * 1.2) = 0.260

Behavior: Same decay as episodic, but with a 20% task-affinity
boost. Preferences for the same task rank higher. This models
the real-world pattern where "Alice prefers Postgres for DB work"
is more relevant when the current task is also DB-related.
```

#### Long-Term: SemanticStore

```
Scoring: text_similarity * decay_factor (no importance weighting)
Decay:   e^(-0.03 * 10) = 0.741 (slower decay than episodic)
Tier:    HOT (0.741 > 0.7 — semantic uses 0.8/0.4 thresholds)

Behavior: Semantic facts decay slower because factual knowledge
tends to be more stable. "Postgres is a relational database" stays
relevant longer than "Alice said she prefers it last Tuesday."
```

#### Long-Term: EntityStore

```
Scoring: (text_similarity + entity_boost) * decay_factor
Entity boost: 0.15 per matched entity in query text
Decay:   e^(-0.03 * 10) = 0.741

Behavior: If entities=["postgres", "pinecone"] and query mentions
"postgres", the score gets +0.15 before decay. Entity-aware
retrieval outperforms pure text similarity for entity-centric
queries like "Tell me about Alice's database preferences."
```

### Comparison Table

| Module | Score | Confidence | Tier | Decay | Survives Day 50? |
|--------|-------|------------|------|-------|------------------|
| EpisodicBuffer | 0.42 | 0.65 | HOT | None | ❌ (FIFO evicted) |
| ContextBuffer | 0.42 | 0.65 | HOT | None | ✓ (if task active) |
| Scratchpad | 0.42 | 0.65 | HOT | None | ✓ (until cleared) |
| EpisodicStore | 0.22 | 0.37 | WARM | 0.607 | ⚠️ (near threshold) |
| PreferenceStore | 0.26 | 0.40 | WARM | 0.607 | ⚠️ (task boost helps) |
| SemanticStore | 0.31 | 0.48 | HOT | 0.741 | ✓ (slow decay) |
| EntityStore | 0.34 | 0.50 | HOT | 0.741 | ✓ (entity boost) |

**Key insight**: Short-term modules are fast and simple but lose memories.
Long-term modules apply decay and sophisticated scoring but retain memories
indefinitely until pruned. The benchmark measures this tradeoff quantitatively.

---

## 7. Data Behind Each Run

When you execute `benchmark run`, here's exactly what data flows through the
system and what gets produced.

### Input Data

```
configs/locomo.yaml           → Which modules, policies, scenarios, seed
benchmark/gold/datasets/*.json → Ground truth: events to inject, queries to run,
                                 expected results to compare against
```

### Gold Dataset Structure

```json
{
  "schema_version": "1.0",
  "scenario": "delayed_recall",
  "description": "Tests memory retention over time with multi-user conversations",
  "user_ids": ["user-alice", "user-bob"],
  "total_conversation_turns": 12,
  "events": [
    {
      "day": 0,
      "memory_events": [
        {
          "id": "M-001",
          "user_id": "user-alice",
          "type": "preference",
          "content": "User prefers Postgres over Pinecone for vector storage",
          "importance": 0.85,
          "entities": ["user", "postgres", "pinecone", "database"],
          "task_id": "db_selection",
          "conversation_turn": 1
        }
      ]
    }
  ],
  "queries": [
    {
      "day": 3,
      "query": "Which database did Alice prefer?",
      "task_id": "db_selection",
      "user_id": "user-alice",
      "is_followup": false,
      "expected": {
        "memory_ids": ["M-001"],
        "acceptable_modules": ["preference_store", "episodic_store"],
        "temporal_window": { "not_before_day": 0, "not_after_day": 0 }
      }
    }
  ],
  "evaluation_criteria": { "recall_k": 5, "temporal_tolerance_days": 1 }
}
```

### Runtime Data Flow

```
Day 0:
  INJECT → M-001 (Alice, preference) → written to ALL enabled modules
  INJECT → M-002 (Alice, episodic)   → written to ALL enabled modules
  INJECT → M-003 (Bob, preference)   → written to ALL enabled modules

Day 3:
  QUERY  → "Which database did Alice prefer?" (user=alice)
         → Read from ALL enabled modules (filtered by user_id=alice)
         → Merge results by score (deduplicate)
         → Compare retrieved IDs vs expected [M-001]
         → Compute: Recall@K, FPR, Temporal Accuracy
         → Record token cost estimate + storage read cost
```

### Output Data

```json
{
  "run_id": "run-2026-01-01T00:00:00Z-a1b2c3",
  "config_hash": "sha256:abc123...",
  "seed": 42,
  "started_at": "2026-01-01T00:00:00Z",
  "completed_at": "2026-01-01T00:00:02Z",
  "memory_modules_enabled": ["episodic_buffer", "episodic_store"],
  "scenario_results": [
    {
      "scenario_name": "delayed_recall",
      "recall_at_k": 0.75,
      "false_positive_rate": 0.20,
      "temporal_accuracy": 0.80,
      "memory_survival_rates": { "0": 1.0, "7": 0.9, "14": 0.7 },
      "total_queries": 8,
      "correct_recalls": 6
    }
  ],
  "cost_summary": {
    "total_token_cost": 0.0042,
    "total_storage_cost": 0.00006,
    "total_cost": 0.00426,
    "cost_per_correct_recall": 0.00071
  },
  "aggregate_recall_at_k": 0.75,
  "aggregate_temporal_accuracy": 0.80,
  "aggregate_false_positive_rate": 0.20
}
```

---

## 8. How the Benchmark Executes

### Execution Flow

```
benchmark run --config configs/locomo.yaml
    │
    ▼
┌─────────────────┐
│  CLI (Click)    │  Parses args, loads config, validates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Orchestrator   │  Creates TimeProvider, GoldOracle, CostTracker
│  (thin coord.)  │  Resolves memory modules via Factory Registry
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ ScenarioRunner  │────▶│  Gold Oracle  │  Loads datasets, provides truth
│                 │     └──────────────┘
│  For each day:  │     ┌──────────────┐
│  1. Inject      │────▶│ MemoryWriter │  Write events to all modules
│  2. Query       │────▶│ MemoryReader │  Read from all modules
│  3. Evaluate    │────▶│  Evaluators  │  Recall, FPR, Temporal, Reliability
│  4. Cost Track  │────▶│ CostTracker  │  Token + storage costs
│  5. Advance Day │────▶│ TimeProvider │  Deterministic clock
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│   Reporting     │  JSON, CSV, Text summary
│   Explorer      │  Web dashboard (FastAPI)
│   Comparator    │  A/B run comparison
└─────────────────┘
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| All modules receive every event | Fair comparison — same input, different algorithms |
| Query reads from ALL modules, merges by score | Measures which module surfaces the right answer |
| user_id filters at module level, not orchestrator | Each module is responsible for its own isolation |
| Deterministic clock, no `time.time()` | Reproducible results across machines |
| Cost estimated, not measured | No real LLM calls — estimates based on token count heuristics |

---

## 9. How to Replace Any Component

The architecture is designed for **every component to be replaceable** without
touching other modules. This section is a step-by-step guide for each.

### Replace a Memory Module

**Example**: Swap `EpisodicStore` for a pgvector-backed implementation.

```python
# Step 1: Create your implementation (implements the same interfaces)
# benchmark/memory/long_term/pgvector_episodic_store.py

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import ReadResponse


class PgVectorEpisodicStore(MemoryWriter, MemoryReader):
    """PostgreSQL + pgvector backed episodic store."""

    def __init__(self, connection_string: str, **kwargs):
        self._connection_string = connection_string
        # ... setup connection pool

    def write(self, event: MemoryEvent) -> None:
        # INSERT INTO memories (id, user_id, content, embedding, ...) ...
        pass

    def read(self, query: ReadQuery) -> ReadResponse:
        # SELECT * FROM memories
        # WHERE user_id = :user_id
        # ORDER BY embedding <=> :query_embedding
        # LIMIT :top_k
        pass
```

```python
# Step 2: Register in the factory
# benchmark/factory/defaults.py

from benchmark.memory.long_term.pgvector_episodic_store import PgVectorEpisodicStore

registry.register("pgvector_episodic_store", PgVectorEpisodicStore)
```

```yaml
# Step 3: Enable in config
memory:
  enabled:
    long_term: [pgvector_episodic_store]
```

```bash
# Step 4: Run contract tests to verify compliance
pytest tests/contract/ -k "MemoryReader or MemoryWriter"
```

**That's it.** The orchestrator never imports your class directly. The factory
resolves it by name. Contract tests verify interface compliance.

### Replace the Similarity Engine

Currently: `difflib.SequenceMatcher` (pure Python, no dependencies).

To use embeddings:

```python
class EmbeddingEpisodicStore(MemoryWriter, MemoryReader):
    def __init__(self, embedding_model: str = "text-embedding-3-small", **kwargs):
        self._embeddings: dict[str, list[float]] = {}
        # ... setup embedding client

    def write(self, event: MemoryEvent) -> None:
        embedding = self._embed(event.content)
        self._embeddings[event.id] = embedding
        self._memories[event.id] = event

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        # Replace SequenceMatcher with cosine similarity
        pass
```

### Replace the Decay Policy

```python
# benchmark/memory/policies/custom_decay.py
from benchmark.memory.interfaces.lifecycle import LifecyclePolicy

class HalfLifeDecayPolicy(LifecyclePolicy):
    """Decay based on half-life model instead of exponential."""

    def __init__(self, half_life_days: int = 7, threshold: float = 0.1):
        self._half_life = half_life_days
        self._threshold = threshold

    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]:
        flagged = []
        for memory_id, score in memory_scores.items():
            decayed = score * (0.5 ** (day / self._half_life))
            if decayed < self._threshold:
                flagged.append(memory_id)
        return flagged
```

### Replace the Cost Model

```python
# benchmark/cost/custom_pricing.py
from benchmark.cost.tracker import CostEntry
from benchmark.models.answer import TokenUsage

class AzureOpenAICostCalculator:
    """Cost calculator using Azure OpenAI pricing."""

    PRICING = {
        "gpt-4o": {"prompt": 0.003, "completion": 0.012},
    }

    def compute_cost(self, usage: TokenUsage, model: str) -> CostEntry:
        pricing = self.PRICING.get(model, {"prompt": 0.01, "completion": 0.03})
        total = (usage.prompt / 1000) * pricing["prompt"] + \
                (usage.completion / 1000) * pricing["completion"]
        return CostEntry(source="azure_llm_tokens", amount_usd=total)
```

### Replace the Evaluation Engine

```python
# benchmark/evaluation/custom_metric.py
from benchmark.evaluation.base import MetricEvaluator, EvaluationResult

class NDCGEvaluator(MetricEvaluator):
    """Normalized Discounted Cumulative Gain evaluator."""

    def evaluate(self, retrieved_ids: list[str], expected_ids: list[str]) -> EvaluationResult:
        # Compute NDCG@K
        pass

    def metric_name(self) -> str:
        return "benchmark.ndcg_at_k"
```

### Replaceability Matrix

| Component | Interface/ABC | Register In | Config Key | Contract Test |
|-----------|--------------|-------------|------------|---------------|
| Memory Module | `MemoryWriter` + `MemoryReader` | `MemoryModuleRegistry` | `memory.enabled.*` | `TestMemoryWriterContract`, `TestMemoryReaderContract` |
| Lifecycle Policy | `LifecyclePolicy` | Direct instantiation | `policies.module_policies.*` | `TestLifecyclePolicyContract` |
| Cost Calculator | Convention (returns `CostEntry`) | `ScenarioRunner.__init__` | N/A (code swap) | Unit test |
| Evaluator | `MetricEvaluator` | `ScenarioRunner.__init__` | N/A (code swap) | Unit test |
| Time Provider | `TimeProvider` | `Orchestrator.__init__` | N/A | Unit test |
| Gold Oracle | Direct class | `Orchestrator.__init__` | N/A | Unit test |
| Reporter | Convention | `CLI report command` | `--format` flag | Unit test |

---

## 10. Database & Persistence Adapter Guide

To replace the in-memory stores with a real database, implement the same
interfaces. Here's an architectural blueprint.

### PostgreSQL + pgvector Schema

```sql
-- Memory events table with per-user isolation
CREATE TABLE memory_events (
    id          VARCHAR(64)  PRIMARY KEY,
    user_id     VARCHAR(128) NOT NULL,
    type        VARCHAR(32)  NOT NULL CHECK (type IN ('episodic','semantic','preference','entity')),
    content     TEXT         NOT NULL,
    embedding   vector(1536),           -- pgvector embedding column
    importance  FLOAT        NOT NULL CHECK (importance BETWEEN 0 AND 1),
    entities    JSONB        DEFAULT '[]',
    task_id     VARCHAR(128) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    injected_day INTEGER     NOT NULL DEFAULT 0,
    metadata    JSONB        DEFAULT '{}'
);

-- Enforce user isolation at the row level
CREATE POLICY user_isolation ON memory_events
    USING (user_id = current_setting('app.current_user_id'));

ALTER TABLE memory_events ENABLE ROW LEVEL SECURITY;

-- Index for vector similarity search per user
CREATE INDEX idx_memories_user_embedding
    ON memory_events USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for task-scoped queries
CREATE INDEX idx_memories_user_task
    ON memory_events (user_id, task_id);

-- Index for temporal queries
CREATE INDEX idx_memories_user_day
    ON memory_events (user_id, injected_day);
```

### Query Pattern (Equivalent to Current read())

```sql
-- Equivalent of EpisodicStore.read() with user isolation and decay
SELECT
    id,
    content,
    importance,
    injected_day,
    1 - (embedding <=> :query_embedding) AS similarity,
    EXP(-0.05 * (:current_day - injected_day)) AS decay_factor,
    (1 - (embedding <=> :query_embedding)) * importance
        * EXP(-0.05 * (:current_day - injected_day)) AS combined_score
FROM memory_events
WHERE user_id = :user_id
  AND type = 'episodic'
ORDER BY combined_score DESC
LIMIT :top_k;
```

### Redis Schema (for Short-Term)

```
# Key pattern: stm:{user_id}:{module}:{memory_id}
# TTL: auto-expire for short-term memories

SET stm:user-alice:episodic_buffer:M-001 '{"content":"...","importance":0.85}' EX 3600

# Sorted set for top-K retrieval
ZADD stm:user-alice:episodic_buffer:scores 0.85 M-001
```

### Security in Database Mode

| Layer | In-Memory (Current) | PostgreSQL (Recommended) |
|-------|--------------------|-----------------------|
| Authentication | None (trust) | mTLS + SCRAM-SHA-256 |
| Authorization | user_id string filter | Row-Level Security policies |
| Encryption at rest | None | TDE / LUKS / Cloud KMS |
| Encryption in transit | N/A (local) | TLS 1.3 |
| Audit | OTel spans | pg_audit + OTel |
| Tenant isolation | Python filter | RLS + connection pooler |

---

## 11. Confidence Scoring Model

Every retrieved memory includes a `confidence` score reflecting how certain
the system is that this result is relevant.

### Formula

```
confidence = clamp(score * 0.6 + decay_factor * 0.4, 0.0, 1.0)
```

| Component | Weight | What It Captures |
|-----------|--------|-----------------|
| `score` | 60% | How relevant is this memory to the query? |
| `decay_factor` | 40% | How fresh is this memory? |

### Why This Split?

- A memory that matches the query perfectly but is very old should have
  **moderate** confidence (high relevance, low freshness).
- A very recent memory with low relevance should also have **moderate**
  confidence.
- Only a **relevant AND fresh** memory gets **high** confidence.

### Short-Term Modules

Short-term modules always use `decay_factor=1.0` (no decay), so:

```
confidence = score * 0.6 + 0.4
```

This means short-term memories have a confidence floor of 0.4, reflecting
that recency itself provides some confidence.

### Confidence Over Time (EpisodicStore, λ=0.05)

| Day | Decay Factor | Score=0.8 Confidence | Score=0.4 Confidence |
|-----|-------------|---------------------|---------------------|
| 0 | 1.000 | 0.88 | 0.64 |
| 5 | 0.779 | 0.79 | 0.55 |
| 10 | 0.607 | 0.72 | 0.48 |
| 20 | 0.368 | 0.63 | 0.39 |
| 30 | 0.223 | 0.57 | 0.33 |

---

## 12. Cost Model

The benchmark tracks two categories of cost:

### Token Costs (LLM API)

| Model | Prompt (per 1K tokens) | Completion (per 1K tokens) |
|-------|----------------------|--------------------------|
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| gpt-3.5-turbo | $0.0005 | $0.0015 |

Token counts are estimated (not from real API calls):
- Prompt tokens ≈ `word_count(query) × 2`
- Completion tokens ≈ `retrieved_count × 10`

### Storage Costs

| Operation | Cost per Operation |
|-----------|-------------------|
| Read | $0.000001 ($1 per million) |
| Write | $0.000005 ($5 per million) |

### Cost per Correct Recall

```
cost_per_correct = total_cost / correct_recalls
```

This is the key efficiency metric. A system that costs $0.10 per correct
recall is 10× more expensive than one at $0.01 — even if both have the
same Recall@K.

---

## 13. Observability & Tracing

Every operation emits structured data via OpenTelemetry:

### Span Hierarchy

```
benchmark.run
  └── scenario.run (per scenario)
        └── simulated_day (per day)
              ├── memory.write (per event injection)
              └── query (per gold query)
                    └── memory.read (per module)
```

### Structured Log Fields

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | string | OTel trace ID for correlation |
| `run_id` | string | Benchmark run identifier |
| `scenario` | string | Current scenario name |
| `simulated_day` | int | Current simulated day |
| `event_id` | string | Memory event being processed |
| `user_id` | string | User context for the operation |
| `decision` | string | What was decided and why |

### Metric Names (Fixed & Versioned)

| Metric Name | Type | Unit |
|-------------|------|------|
| `benchmark.recall_at_k` | Gauge | ratio (0-1) |
| `benchmark.false_positive_rate` | Gauge | ratio (0-1) |
| `benchmark.temporal_accuracy` | Gauge | ratio (0-1) |
| `benchmark.memory_survival_rate` | Gauge | ratio (0-1) |
| `benchmark.cost_total_usd` | Counter | USD |
| `benchmark.latency_ms` | Histogram | milliseconds |

---

## 14. FAQ & Honest Limitations

### Q: Is this a production memory system?

**No.** This is a benchmark framework with reference implementations. The
in-memory stores are designed for measurement, not for serving real users.
They exist to provide a controlled, deterministic baseline against which
you can measure your own memory system.

### Q: Can I use this to decide which memory architecture to use in production?

**Yes, with caveats.** The benchmark tells you:
- Which memory *structure* (episodic vs. semantic vs. preference) works best
  for your access patterns.
- How decay parameters affect recall over time.
- What the cost profile looks like for different configurations.

It does NOT tell you about real-world latency, embedding quality, or scale
behavior. For that, you need a production-grade implementation behind the
same interfaces.

### Q: How do I add a real database?

Implement `MemoryWriter` and `MemoryReader`, register in the factory, enable
in config. See [Section 10](#10-database--persistence-adapter-guide) for the
full PostgreSQL/Redis blueprint with row-level security.

### Q: Why `SequenceMatcher` instead of embeddings?

Determinism. `SequenceMatcher` produces identical results on every machine,
every run. Embedding models can vary by version, quantization, and hardware.
For benchmarking the *structure* of memory, deterministic similarity is more
valuable than embedding accuracy.

### Q: Is the user isolation real security?

It's application-level filtering — appropriate for a benchmark tool. For
production, layer database-level RLS, authentication, and encryption.
The `user_id` filter pattern is correct; only the enforcement layer changes.

### Q: Can I benchmark my own memory system?

Yes. Implement the interfaces, register your module, and run:

```bash
benchmark run --config my_config.yaml --output-dir outputs/
benchmark compare --baseline outputs/reference.json --candidate outputs/mine.json
```

### Q: What happens if I run the same config twice?

**Identical results.** Same config + same seed = same metrics, same costs,
same report. This is enforced by:
- Deterministic `SimulatedClock` (no `time.time()`)
- Fixed decay formulas (no randomness)
- Deterministic similarity scoring
- Explicit seed in config

### Limitations We Acknowledge

| Limitation | Impact | Mitigation Path |
|-----------|--------|-----------------|
| In-memory only | No persistence, no real DB | Implement DB adapter (Section 10) |
| SequenceMatcher similarity | Low quality vs embeddings | Swap for embedding module (Section 9) |
| Estimated token costs | Not real API costs | Wire real LLM client into cost tracker |
| Single-process | No concurrency testing | Add async adapter + load testing |
| No real LLM evaluation | Can't judge answer quality | Enable `answering` module with real model |
| No encryption | Benchmark-appropriate only | Add DB-level encryption for production |

---

*This document is the authoritative reference for the Agentic Memory Benchmark
architecture. For code-level API docs, see [api_reference.md](api_reference.md).
For the original system blueprint, see [docs/architecture.md](architecture.md).*
