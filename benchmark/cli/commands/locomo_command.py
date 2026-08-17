"""CLI command to convert LoCoMo dataset for benchmarking.

Provides commands to:
- Convert the full dataset to our gold format
- Create small test subsets for validation
- Show dataset statistics
"""

from __future__ import annotations

from pathlib import Path

import click

from benchmark.gold.locomo_loader import (
    LoCoMoLoader,
    convert_locomo_to_gold,
    create_test_subset,
)


@click.group("locomo")
def locomo_group() -> None:
    """LoCoMo dataset commands."""


@locomo_group.command("convert")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for the converted gold dataset.",
)
@click.option(
    "--scenario-name",
    "-s",
    default="locomo",
    help="Scenario name for the dataset.",
)
@click.option(
    "--subset",
    "-n",
    type=int,
    default=None,
    help="Convert only the first N conversations (for testing).",
)
def convert_command(
    input_path: Path,
    output: Path | None,
    scenario_name: str,
    subset: int | None,
) -> None:
    """Convert a LoCoMo JSON file to our gold dataset format.

    INPUT_PATH is the path to locomo10.json or similar.
    """
    if output is None:
        stem = input_path.stem
        output = input_path.parent / f"{stem}_gold.json"

    click.echo(f"Converting {input_path.name} → {output.name}")
    click.echo(f"  Scenario: {scenario_name}")
    if subset:
        click.echo(f"  Subset: first {subset} conversations")

    dataset = convert_locomo_to_gold(
        input_path=input_path,
        output_path=output,
        scenario_name=scenario_name,
        subset_size=subset,
    )

    click.echo("\nConversion complete:")
    click.echo(f"  Queries:        {len(dataset.queries)}")
    click.echo(f"  Days:           {len(dataset.events)}")
    click.echo(f"  Users:          {len(dataset.user_ids)}")
    click.echo(f"  Total memories: {dataset.total_conversation_turns}")
    click.echo(f"  Output:         {output}")


@locomo_group.command("test-subset")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for the test subset.",
)
@click.option(
    "--conversations",
    "-n",
    type=int,
    default=3,
    help="Number of conversations to include.",
)
@click.option(
    "--max-sessions",
    type=int,
    default=10,
    help="Max sessions per conversation.",
)
def test_subset_command(
    input_path: Path,
    output: Path | None,
    conversations: int,
    max_sessions: int,
) -> None:
    """Create a small test subset from a LoCoMo file."""
    if output is None:
        output = input_path.parent / "locomo_test_subset.json"

    click.echo(f"Creating test subset from {input_path.name}")
    click.echo(f"  Conversations: {conversations}")
    click.echo(f"  Max sessions:  {max_sessions}")

    dataset = create_test_subset(
        input_path=input_path,
        output_path=output,
        max_conversations=conversations,
        max_sessions=max_sessions,
    )

    click.echo("\nTest subset created:")
    click.echo(f"  Queries:        {len(dataset.queries)}")
    click.echo(f"  Days:           {len(dataset.events)}")
    click.echo(f"  Total memories: {dataset.total_conversation_turns}")
    click.echo(f"  Output:         {output}")


@locomo_group.command("stats")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
def stats_command(input_path: Path) -> None:
    """Show statistics about a LoCoMo dataset file."""
    import json

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        else:
            data = [data]

    click.echo(f"Dataset: {input_path.name}")
    click.echo(f"  Total conversations: {len(data)}")

    total_sessions = 0
    total_turns = 0
    total_qa = 0

    for sample in data:
        conversation = sample.get("conversation", {})
        sessions = [
            k
            for k in conversation.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        ]
        total_sessions += len(sessions)
        for sk in sessions:
            turns = conversation.get(sk, [])
            if isinstance(turns, list):
                total_turns += len(turns)

        qa = sample.get("qa", [])
        total_qa += len(qa)

    click.echo(f"  Total sessions:      {total_sessions}")
    click.echo(f"  Total turns:         {total_turns}")
    click.echo(f"  Total QA pairs:      {total_qa}")

    # QA category breakdown
    adapter = LoCoMoLoader()
    categories = adapter.get_category_distribution(data)
    click.echo("\n  QA categories:")
    for cat, count in sorted(categories.items()):
        click.echo(f"    {cat:25s}: {count:4d}")

    difficulty = adapter.get_difficulty_distribution(data)
    click.echo("\n  Difficulty distribution:")
    for level, count in sorted(difficulty.items()):
        click.echo(f"    {level:10s}: {count:4d}")
