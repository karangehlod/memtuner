"""Evaluation context — rich context object for all evaluators.

Replaces the (retrieved_ids, expected_ids) pair with a richer context
that carries source modules, creation days, temporal windows, and
conversation metadata. Each evaluator picks the fields it needs.

This follows OCP: new evaluators can use new context fields without
changing existing evaluators or the evaluator interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationContext:
    """Rich context for metric evaluation.

    Carries all data needed by any evaluator. Evaluators read only
    the fields they need and ignore the rest (ISP via data selection).

    Attributes:
        retrieved_ids: Memory IDs returned by the memory system (ordered by score).
        expected_ids: Memory IDs expected from the gold dataset.
        retrieved_source_modules: Mapping of memory_id → source module name.
        retrieved_creation_days: Mapping of memory_id → injection day.
        acceptable_modules: Gold-specified acceptable source modules (empty = any).
        temporal_window: Expected (not_before_day, not_after_day) or None.
        is_followup: Whether the query is a follow-up in a conversation.
        references_turn: The conversation turn this query follows up on.
    """

    retrieved_ids: list[str]
    expected_ids: list[str]
    retrieved_source_modules: dict[str, str] = field(default_factory=dict)
    retrieved_creation_days: dict[str, int] = field(default_factory=dict)
    acceptable_modules: list[str] = field(default_factory=list)
    temporal_window: tuple[int, int] | None = None
    is_followup: bool = False
    references_turn: int | None = None
    corpus_size: int = 5879
