# API Reference

This document provides a reference for the key public APIs in the Agentic Memory
Benchmark tool.

## Table of Contents

- [Models](#models)
- [Memory Interfaces](#memory-interfaces)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [CLI](#cli)
- [Explorer](#explorer)

---

## Models

All models are in `benchmark/models/` — pure data classes (pydantic) with zero
logic and zero dependencies.

### MemoryEvent

```python
from benchmark.models.memory_event import MemoryEvent, MemoryType

event = MemoryEvent(
    id="M-001",
    type=MemoryType.EPISODIC,
    content="User prefers Postgres for vector storage",
    timestamp=datetime.now(timezone.utc),
    importance=0.85,
    entities=["user", "postgres"],
    task_id="db_selection",
)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique memory identifier |
| `user_id` | `str` | User this memory belongs to (default: `"user-default"`) |
| `type` | `MemoryType` | Category: EPISODIC, SEMANTIC, PREFERENCE, ENTITY |
| `content` | `str` | Human-readable content (min 1 char) |
| `timestamp` | `datetime` | Creation timestamp |
| `importance` | `float` | Score between 0.0 and 1.0 |
| `entities` | `list[str]` | Entities mentioned |
| `task_id` | `str` | Related task identifier |
| `metadata` | `dict[str, Any]` | Extensible metadata |

### ReadQuery

```python
from benchmark.models.query import ReadQuery, ReadQueryContext

query = ReadQuery(
    query="Which database does the user prefer?",
    top_k=5,
    context=ReadQueryContext(
        simulated_day=3,
        task_id="db_selection",
        user_id="user-alice",
    ),
)
```

**ReadQueryContext Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `simulated_day` | `int` | Current simulated day (≥0) |
| `task_id` | `str` | Related task identifier |
| `user_id` | `str` | User executing this query (default: `"user-default"`) |

### ReadResponse

```python
from benchmark.models.response import ReadResponse, RetrievedMemory, MemoryTier
```

**RetrievedMemory Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `memory_id` | `str` | Unique memory identifier |
| `source_module` | `str` | Which memory module returned this |
| `score` | `float` | Relevance score (0.0–1.0) |
| `confidence` | `float` | Retrieval confidence (0.0=uncertain, 1.0=certain) |
| `timestamp` | `datetime` | Original creation timestamp |
| `tier` | `MemoryTier` | HOT, WARM, or COLD |
| `decay_factor` | `float` | Current decay multiplier |

### BenchmarkRunResult

Complete output of a benchmark run, including per-scenario metrics, cost
summary, and aggregate scores.

---

## Memory Interfaces

All memory modules implement segregated interfaces from
`benchmark/memory/interfaces/`.

### MemoryWriter

```python
from benchmark.memory.interfaces.writer import MemoryWriter

class MemoryWriter(ABC):
    @abstractmethod
    def write(self, event: MemoryEvent) -> None: ...
```

### MemoryReader

```python
from benchmark.memory.interfaces.reader import MemoryReader

class MemoryReader(ABC):
    @abstractmethod
    def read(self, query: ReadQuery) -> ReadResponse: ...
```

### LifecyclePolicy

```python
from benchmark.memory.interfaces.lifecycle import LifecyclePolicy

class LifecyclePolicy(ABC):
    @abstractmethod
    def apply(self, day: int, memory_scores: dict[str, float]) -> list[str]: ...
```

### Available Implementations

| Module | Type | Description |
|--------|------|-------------|
| `EpisodicBuffer` | Short-term | Fixed-capacity FIFO buffer |
| `ContextBuffer` | Short-term | Task-scoped context memory |
| `Scratchpad` | Short-term | Temporary working memory |
| `EpisodicStore` | Long-term | Episodic memory with decay |
| `SemanticStore` | Long-term | Factual/semantic knowledge |
| `PreferenceStore` | Long-term | User preference memory |
| `EntityStore` | Long-term | Named entity memory |

---

## Configuration

Configuration is loaded from YAML and validated with pydantic.

### Loading Config

```python
from benchmark.config.loader import load_config_from_path
from pathlib import Path

config = load_config_from_path(Path("configs/locomo.yaml"))
```

### BenchmarkConfig Schema

```yaml
memory:
  enabled:
    short_term: ["episodic_buffer"]
    long_term: ["episodic_store"]

policies:
  module_policies:
    episodic_store:
      decay:
        type: exponential
        lambda: 0.05
      pruning:
        strategy: score_threshold
        threshold: 0.35

benchmark:
  evaluation_horizon: 14
  seed: 42
  scenarios: ["delayed_recall"]

observability:
  exporter: otlp
  endpoint: "http://localhost:4317"
  log_level: INFO

answering:
  enabled: false
  model: gpt-4o
  max_tokens: 500
```

---

## Evaluation

### Metric Evaluators

All evaluators implement the `MetricEvaluator` interface:

```python
from benchmark.evaluation.base import MetricEvaluator, EvaluationResult

class MetricEvaluator(ABC):
    @abstractmethod
    def evaluate(self, retrieved_ids: list[str], expected_ids: list[str]) -> EvaluationResult: ...
    
    @abstractmethod
    def metric_name(self) -> str: ...
```

### Available Evaluators

| Evaluator | Metric | Formula |
|-----------|--------|---------|
| `RecallEvaluator` | Recall@K | \|Retrieved ∩ Gold\| / \|Gold\| |
| `FalsePositiveEvaluator` | FP Rate | \|Retrieved \ Gold\| / \|Retrieved\| |
| `TemporalAccuracyEvaluator` | Temporal Accuracy | Fraction within time window |
| `ReliabilityCurveEvaluator` | Survival Rate | \|Alive(day)\| / \|Injected\| |

---

## CLI

The CLI is the primary interface for the benchmark tool.

### Commands

```bash
# Initialize project with default config
benchmark init --output configs/my_config.yaml

# Validate a config file
benchmark validate --config configs/locomo.yaml

# Run a benchmark
benchmark run --config configs/locomo.yaml --gold-dataset data/locomo10.json --output-dir outputs/

# Generate a report from results
benchmark report --results-dir outputs/ --format json

# Compare two benchmark runs
benchmark compare --baseline outputs/run_a.json --candidate outputs/run_b.json

# Launch the explorer dashboard
benchmark explore --results-dir outputs/ --port 8501
```

---

## Explorer

The explorer provides a read-only web dashboard for browsing results.

### Programmatic Usage

```python
from benchmark.explorer.server import create_explorer_app
from pathlib import Path

app = create_explorer_app(Path("outputs/"))
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML dashboard |
| GET | `/api/runs` | List all loaded runs |
| GET | `/api/runs/{run_id}` | Get run details |
| GET | `/api/metrics/{metric_name}` | Metric series across runs |
| POST | `/api/reload` | Reload results from disk |

---

## Factory Registry

The factory registry provides dependency inversion — the orchestrator never
imports concrete implementations.

```python
from benchmark.factory.registry import MemoryModuleRegistry

registry = MemoryModuleRegistry()
registry.register("episodic_store", EpisodicStore)

# Later, resolve by name:
store = registry.resolve("episodic_store", decay_lambda=0.05)
```

---

## Observability

### Structured Logging

```python
from benchmark.observability.logger import get_logger, log_decision

logger = get_logger(__name__)
log_decision(logger, "Memory pruning decision", pruned_count=5, day=7)
```

### OTel Tracing

```python
from benchmark.observability.tracer import create_span

with create_span("memory.read", attributes={"module": "episodic"}) as span:
    result = store.read(query)
    span.set_attribute("result_count", len(result.retrieved_memories))
```

### OTel Metrics

```python
from benchmark.observability.metrics import record_metric
from benchmark.observability.schemas import METRIC_RECALL_AT_K

record_metric(METRIC_RECALL_AT_K, 0.85, attributes={"scenario": "delayed_recall"})
```
