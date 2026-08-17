"""LongMemEval adapter compatibility shim.

Re-exports the LongMemEval pack adapter under the old import path
for backward compatibility with existing tests and scripts.

The canonical implementation lives at benchmark.packs.longmemeval.adapter.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.gold.schema import GoldEvaluationCriteria
from benchmark.packs.longmemeval.adapter import LongMemEvalPack


# Alias for backward compatibility
class LongMemEvalAdapter(LongMemEvalPack):
    """Backward-compatible adapter wrapping LongMemEvalPack.

    Adds the `load_and_convert` method used by older tests.
    """

    def load_and_convert(self, data_path: Path, scenario_name: str) -> Any:
        """Load data from a path and convert to GoldDataset.

        Args:
            data_path: Path to the LongMemEval JSON file.
            scenario_name: Name for the scenario.

        Returns:
            A GoldDataset instance.
        """
        if data_path.is_file():
            with data_path.open() as fh:
                self._data = json.load(fh)
            self._loaded = True
        else:
            self.load(data_path.parent)
        dataset = self.to_gold_dataset(seed=42, evaluation_horizon=180)
        return self._apply_legacy_test_compat(dataset)

    def _apply_legacy_test_compat(self, dataset: Any) -> Any:
        question_type_by_id = {
            item["question_id"]: item.get("question_type", "")
            for item in self._data
            if item.get("question_id")
        }

        updated_queries = []
        for query in dataset.queries:
            question_type = question_type_by_id.get(query.task_id, "")
            updated_queries.append(
                query.model_copy(
                    update={
                        "task_id": f"lme_{query.task_id}",
                        "is_followup": question_type in {"knowledge-update", "multi-session"},
                    }
                )
            )

        return dataset.model_copy(
            update={
                "queries": updated_queries,
                "evaluation_criteria": GoldEvaluationCriteria(
                    recall_k=10,
                    temporal_tolerance_days=7,
                ),
            }
        )

    def get_difficulty_distribution(self, raw_data: list) -> dict[str, int]:
        """Get distribution of difficulty levels from raw data.

        Args:
            raw_data: List of LongMemEval instances.

        Returns:
            Dict mapping difficulty level to count.
        """
        dist: dict[str, int] = {}
        for item in raw_data:
            level = item.get("haystack_level", "unknown")
            dist[level] = dist.get(level, 0) + 1
        return dist


def _parse_date_to_day(date_str: str, reference_date: str = "2023-01-01") -> int:
    """Parse a date string to a day number relative to a reference date.

    Args:
        date_str: Date in YYYY-MM-DD format.
        reference_date: Reference date for day 0.

    Returns:
        Number of days since reference_date.
    """
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
        reference = datetime.strptime(reference_date, "%Y-%m-%d")
        return (target - reference).days
    except (ValueError, TypeError):
        return 0


def convert_longmemeval_to_gold(
    input_path: Path,
    output_path: Path,
    scenario_name: str = "longmemeval",
) -> Any:
    """Convert a LongMemEval JSON file to gold dataset format.

    Args:
        input_path: Path to the LongMemEval oracle JSON.
        output_path: Path to write the gold dataset.
        scenario_name: Name for the scenario.

    Returns:
        The generated GoldDataset.
    """
    pack = LongMemEvalPack()
    if input_path.is_file():
        with input_path.open() as fh:
            pack._data = json.load(fh)
        pack._loaded = True
    else:
        pack.load(input_path.parent)
    dataset = pack.to_gold_dataset(seed=42, evaluation_horizon=180)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(dataset.model_dump(mode="json"), fh, indent=2)

    return dataset


def create_test_subset(
    input_path: Path,
    output_path: Path,
    questions_per_type: int = 5,
) -> Any:
    """Create a small test subset from LongMemEval data.

    Args:
        input_path: Path to the LongMemEval oracle JSON.
        output_path: Path to write the subset.
        questions_per_type: Number of questions per type.

    Returns:
        The generated GoldDataset subset.
    """
    pack = LongMemEvalPack()
    if input_path.is_file():
        with input_path.open() as fh:
            pack._data = json.load(fh)
        pack._loaded = True
    else:
        pack.load(input_path.parent)
    dataset = pack.to_gold_dataset(max_queries=questions_per_type * 5, seed=42, evaluation_horizon=30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(dataset.model_dump(mode="json"), fh, indent=2)

    return dataset


def _generate_memory_id(question_id: str, session_idx: int, turn_idx: int) -> str:
    """Generate deterministic memory ID from question, session, and turn.

    Re-exported from packs.longmemeval.adapter for backward compatibility.

    Args:
        question_id: The question identifier.
        session_idx: Session index.
        turn_idx: Turn index within the session.

    Returns:
        A deterministic memory ID string.
    """
    from benchmark.packs.longmemeval.adapter import (
        _generate_memory_id as _pack_generate_memory_id,
    )

    return _pack_generate_memory_id(question_id, session_idx, turn_idx)
