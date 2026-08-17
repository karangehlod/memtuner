# Custom Data Guide

This guide explains how to benchmark your own memory data using the private pack adapter.

## Quick Start

1. Create your data directory:
```bash
mkdir -p data/my_data
```

2. Create `events.jsonl` — one memory event per line:
```jsonl
{"memory_id": "evt-001", "user_id": "user-1", "type": "episodic", "content": "User discussed project deadline for Friday", "day": 0, "importance": 0.7, "entities": ["project"], "task_id": "task-1"}
{"memory_id": "evt-002", "user_id": "user-1", "type": "preference", "content": "User prefers dark mode in all editors", "day": 1, "importance": 0.9, "entities": [], "task_id": "task-2"}
{"memory_id": "evt-003", "user_id": "user-1", "type": "semantic", "content": "The API uses REST with JSON payloads", "day": 2, "importance": 0.6, "entities": ["API"], "task_id": "task-1"}
```

3. Create `queries.jsonl` — one query per line:
```jsonl
{"query_id": "q-001", "user_id": "user-1", "query_text": "What was the project deadline?", "day": 10, "expected_memory_ids": ["evt-001"], "acceptable_modules": ["episodic_store"], "task_id": "task-1"}
{"query_id": "q-002", "user_id": "user-1", "query_text": "Does the user prefer light or dark mode?", "day": 10, "expected_memory_ids": ["evt-002"], "acceptable_modules": ["preference_store"]}
```

4. Run the benchmark:
```bash
benchmark run -c configs/locomo.yaml --pack private --data-dir data/my_data
```

## Field Reference

### events.jsonl

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `memory_id` | string | **Yes** | Unique identifier for this memory event |
| `user_id` | string | No | User who owns this memory (default: `default-user`) |
| `type` | string | No | One of: `episodic`, `semantic`, `preference`, `entity` (default: `episodic`) |
| `content` | string | No | The actual memory text content |
| `day` | int | No | Simulated day number, 0-indexed (default: 0) |
| `importance` | float | No | Relevance score 0.0–1.0 (default: 0.5) |
| `entities` | list[str] | No | Entity names mentioned in this memory |
| `task_id` | string | No | Task/session grouping identifier |
| `turn` | int | No | Conversation turn number (default: 0) |

### queries.jsonl

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string | **Yes** | Unique identifier for this query |
| `user_id` | string | No | User executing the query (default: `default-user`) |
| `query_text` | string | No | The natural language question |
| `day` | int | No | Day to execute query (default: last simulated day) |
| `expected_memory_ids` | list[str] | No | Memory IDs that should be retrieved |
| `acceptable_modules` | list[str] | No | Which modules should retrieve (default: `["episodic_store", "semantic_store"]`) |
| `task_id` | string | No | Task grouping |
| `earliest_day` | int | No | Earliest acceptable retrieval day |
| `latest_day` | int | No | Latest acceptable retrieval day |

## Tips

- **Evidence linking**: Each query's `expected_memory_ids` should reference `memory_id` values from your events. This is how recall and false positive rate are measured.
- **Multi-user**: Use different `user_id` values to test user isolation.
- **Temporal spread**: Spread events across multiple `day` values to test temporal decay.
- **Memory types**: Use different `type` values to test different memory modules.
- **Importance scores**: Higher importance (closer to 1.0) events should be recalled more reliably.

## Using with Workload Profiles

```bash
# Quick smoke test (14 days)
benchmark run -c configs/profiles/low_qpd.yaml --pack private --data-dir data/my_data

# Standard evaluation (50 days)
benchmark run -c configs/profiles/medium_qpd.yaml --pack private --data-dir data/my_data

# Stress test (100 days)
benchmark run -c configs/profiles/high_qpd.yaml --pack private --data-dir data/my_data
```

## Converting Existing Data

If you have conversation logs in another format, write a script to convert them:

```python
import json

# Your source data
conversations = [...]

with open("data/my_data/events.jsonl", "w") as f:
    for i, msg in enumerate(conversations):
        event = {
            "memory_id": f"evt-{i:04d}",
            "user_id": "user-1",
            "type": "episodic",
            "content": msg["text"],
            "day": msg["session_day"],
            "importance": 0.5,
        }
        f.write(json.dumps(event) + "\n")
```
