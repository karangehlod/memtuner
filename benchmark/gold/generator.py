from __future__ import annotations

import json
import random
from dataclasses import dataclass

from benchmark.gold.schema import (
    GoldDataset,
    GoldDayEvents,
    GoldExpectedResult,
    GoldMemoryEvent,
    GoldQuery,
    TemporalWindow,
)
from benchmark.models.memory_event import MemoryType
from benchmark.observability.tracer import create_span
from benchmark.time.provider import TimeProvider


@dataclass(frozen=True)
class GoldGeneratorConfig:
    seed: int
    users: int = 10
    days: int = 7
    events_per_day: int = 10
    scenario: str = "generated"


class GoldGenerator:
    """Deterministic gold dataset generator.

    Uses an explicit seed and the TimeProvider for day stamping.
    Produces `GoldDataset` instances serializable to the project's schema.
    """

    def __init__(self, config: GoldGeneratorConfig, time_provider: TimeProvider):
        self._config = config
        self._time_provider = time_provider
        self._rand = random.Random(config.seed)

    def generate(self) -> GoldDataset:
        with create_span("gold.generator.generate") as span:
            users = [f"user-{i}" for i in range(self._config.users)]
            events: list[GoldDayEvents] = []
            for day in range(self._config.days):
                day_events: list[GoldMemoryEvent] = []
                for eidx in range(self._config.events_per_day):
                    mid = f"M-{day:03d}-{eidx:04d}"
                    user = self._rand.choice(users)
                    mem_type = self._rand.choice(list(MemoryType))
                    content = self._generate_content(user, day, eidx, mem_type.value)
                    user_idx = int(user.split("-")[1]) % len(self._USER_TOOLS)
                    tool_name = self._USER_TOOLS[user_idx][0].lower()
                    mem = GoldMemoryEvent(
                        id=mid,
                        user_id=user,
                        type=mem_type,
                        content=content,
                        importance=round(self._rand.uniform(0.1, 1.0), 2),
                        entities=[user, tool_name],
                        task_id="generated_task",
                        conversation_turn=eidx % 5,
                    )
                    day_events.append(mem)
                events.append(GoldDayEvents(day=day, memory_events=day_events))

            queries = self._generate_queries(users, events)

            dataset = GoldDataset(
                scenario=self._config.scenario,
                description=f"Generated dataset seed={self._config.seed}",
                user_ids=users,
                total_conversation_turns=self._config.events_per_day * self._config.days,
                events=events,
                queries=queries,
            )
            span.set_attribute("generated_days", len(events))
            span.set_attribute("generated_users", len(users))
            return dataset

    # User-specific vocabulary pools — gives each user a distinct signal
    _USER_TOOLS: list[list[str]] = [
        ["Redis", "in-memory caching", "sub-millisecond latency"],
        ["Postgres", "relational data", "ACID compliance"],
        ["Pinecone", "vector embeddings", "ANN search"],
        ["Weaviate", "semantic search", "hybrid retrieval"],
        ["Qdrant", "Rust-based", "high-throughput vectors"],
        ["MongoDB", "document store", "flexible schema"],
        ["Cassandra", "wide-column", "high write throughput"],
        ["Chroma", "local embeddings", "LangChain integration"],
        ["Milvus", "cloud-native", "GPU acceleration"],
        ["Elasticsearch", "full-text search", "aggregation pipelines"],
    ]

    _MEMORY_TEMPLATES: dict[str, list[str]] = {
        "preference": [
            "{user} always chooses {tool} for {feature} tasks.",
            "{user} prefers {tool} because of {feature}.",
            "{user} uses {tool} for all {feature} workloads.",
        ],
        "episodic": [
            "Today {user} set up {tool} with {feature} configuration.",
            "{user} migrated the pipeline to {tool} to get {feature}.",
            "{user} benchmarked {tool} and confirmed {feature}.",
        ],
        "semantic": [
            "{user} knows that {tool} excels at {feature}.",
            "{user} documented: {tool} is best for {feature} use-cases.",
            "{user} learned that {tool} provides {feature}.",
        ],
        "entity": [
            "{user} recommended {tool} to the team for {feature}.",
            "{user} referenced {tool} as a solution for {feature}.",
        ],
    }

    def _generate_content(self, user: str, day: int, eidx: int, mem_type: str) -> str:
        """Generate user-specific, type-aware content with distinct vocabulary."""
        # Each user gets a consistent tool preference based on user index
        user_idx = int(user.split("-")[1]) % len(self._USER_TOOLS)
        tool_pool = self._USER_TOOLS[user_idx]
        tool = tool_pool[0]
        feature = self._rand.choice(tool_pool[1:])

        templates = self._MEMORY_TEMPLATES.get(mem_type, self._MEMORY_TEMPLATES["semantic"])
        template = self._rand.choice(templates)
        base = template.format(user=user, tool=tool, feature=feature)
        return f"{base} (day {day}, turn {eidx})"

    def _generate_queries(self, users: list[str], events: list) -> list[GoldQuery]:
        """Generate one query per user using only memory IDs that belong to that user.

        BUG-001 FIX: Previously this hardcoded M-{last_day}-000{0,1,2} for every
        user. Those IDs are randomly owned — user-0's query expected memories
        owned by user-7, making correct recall mathematically impossible.
        """
        # Build per-user index of their own memory IDs, preserving insertion order
        user_memory_ids: dict[str, list[str]] = {u: [] for u in users}
        for day_events in events:
            for event in day_events.memory_events:
                user_memory_ids[event.user_id].append(event.id)

        last_day = self._config.days - 1
        window_start = max(0, last_day - 2)

        # Build per-user index of recent memory IDs (last 3 days only)
        user_recent_ids: dict[str, list[str]] = {u: [] for u in users}
        for day_events in events:
            if day_events.day >= window_start:
                for event in day_events.memory_events:
                    user_recent_ids[event.user_id].append(event.id)

        queries: list[GoldQuery] = []
        for user in users:
            recent = user_recent_ids[user]
            if not recent:
                # Fall back to any memory owned by this user
                recent = user_memory_ids[user]
            if not recent:
                continue  # user has no memories — skip

            # Use up to 3 memories that this user owns
            expected_ids = recent[: min(3, len(recent))]

            # Determine the tool this user is associated with for the query text
            user_idx = int(user.split("-")[1]) % len(self._USER_TOOLS)
            tool = self._USER_TOOLS[user_idx][0]

            expected = GoldExpectedResult(
                memory_ids=expected_ids,
                acceptable_modules=["episodic_store", "preference_store", "semantic_store"],
                temporal_window=TemporalWindow(
                    not_before_day=window_start,
                    not_after_day=last_day,
                ),
            )

            q = GoldQuery(
                day=last_day,
                query=f"What does {user} know about {tool}?",
                task_id="generated_task",
                user_id=user,
                is_followup=False,
                expected=expected,
            )
            queries.append(q)
        return queries


def write_dataset_to_file(dataset: GoldDataset, path: str) -> None:
    # Using tracer span as a context manager
    with create_span("gold.generator.write") as span:
        data = dataset.model_dump() if hasattr(dataset, "model_dump") else dataset.dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        span.set_attribute("output_path", path)


# CLI entrypoint (click)
import click


@click.command()
@click.option("--seed", default=42, help="Random seed (deterministic)")
@click.option("--users", default=10, help="Number of distinct users to simulate")
@click.option("--days", default=7, help="Number of simulated days")
@click.option("--events-per-day", default=10, help="Events to inject per day")
@click.option("--out", default="benchmark/gold/datasets/generated.json", help="Output path")
@click.option("--dry-run", is_flag=True, help="Don't write file, just validate")
@click.option("--validate", is_flag=True, help="Validate output against schema")
def generate_gold(
    seed: int, users: int, days: int, events_per_day: int, out: str, dry_run: bool, validate: bool
):
    from benchmark.time.simulated_clock import SimulatedClock

    clock = SimulatedClock()
    cfg = GoldGeneratorConfig(seed=seed, users=users, days=days, events_per_day=events_per_day)
    gen = GoldGenerator(cfg, clock)
    ds = gen.generate()
    if validate:
        # Pydantic validation already performed; re-construct to ensure
        GoldDataset.model_validate(ds.model_dump() if hasattr(ds, "model_dump") else ds.dict())
    if dry_run:
        click.echo("Dry run OK - dataset valid")
        return
    write_dataset_to_file(ds, out)
    click.echo(f"Wrote dataset to {out}")
