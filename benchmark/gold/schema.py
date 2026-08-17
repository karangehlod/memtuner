"""Gold dataset schema for validation.

Defines the pydantic models for gold dataset files.
Gold datasets are versioned, immutable, and scenario-specific.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from benchmark.models.memory_event import MemoryType


class GoldMemoryEvent(BaseModel):
    """A memory event within a gold dataset.

    Attributes:
        id: Unique memory identifier.
        user_id: Identifier of the user this memory belongs to.
        type: The memory type category.
        content: Human-readable memory content.
        importance: Importance score.
        entities: List of entities mentioned.
        task_id: Related task identifier.
        conversation_turn: Optional turn number within a conversation.
    """

    id: str = Field(..., description="Unique memory identifier")
    user_id: str = Field(default="user-default", description="User identifier")
    type: MemoryType = Field(..., description="Memory type category")
    content: str = Field(..., min_length=1, description="Memory content")
    importance: float = Field(..., ge=0.0, le=1.0, description="Importance score")
    entities: list[str] = Field(default_factory=list, description="Entities")
    task_id: str = Field(..., description="Task identifier")
    conversation_turn: int = Field(default=0, ge=0, description="Turn number in conversation")

    model_config = {"frozen": True}


class GoldDayEvents(BaseModel):
    """Events to inject on a specific simulated day.

    Attributes:
        day: The simulated day to inject these events.
        memory_events: List of memory events for this day.
    """

    day: int = Field(..., ge=0, description="Simulated day")
    memory_events: list[GoldMemoryEvent] = Field(
        ...,
        min_length=1,
        description="Memory events for this day",
    )

    model_config = {"frozen": True}


class TemporalWindow(BaseModel):
    """Expected temporal window for a gold query result.

    Attributes:
        not_before_day: Earliest acceptable day for the retrieved memory.
        not_after_day: Latest acceptable day for the retrieved memory.
    """

    not_before_day: int = Field(..., ge=0, description="Earliest acceptable day")
    not_after_day: int = Field(..., ge=0, description="Latest acceptable day")

    model_config = {"frozen": True}


class GoldExpectedResult(BaseModel):
    """Expected results for a gold query.

    Attributes:
        memory_ids: List of memory IDs that should be retrieved.
        acceptable_modules: Memory modules that are acceptable sources.
        temporal_window: Expected temporal window for results.
    """

    memory_ids: list[str] = Field(..., min_length=1, description="Expected memory IDs")
    acceptable_modules: list[str] = Field(
        default_factory=list,
        description="Acceptable source modules",
    )
    temporal_window: TemporalWindow | None = Field(
        default=None,
        description="Temporal window constraint",
    )

    model_config = {"frozen": True}


class GoldQuery(BaseModel):
    """A query within a gold dataset with expected results.

    Attributes:
        day: The simulated day this query is executed.
        query: The natural language query string.
        task_id: Related task identifier.
        user_id: The user executing this query.
        expected: Expected results for evaluation.
        is_followup: Whether this is a follow-up to a previous query.
        references_turn: Optional turn number this follows up on.
    """

    day: int = Field(..., ge=0, description="Query execution day")
    query: str = Field(..., min_length=1, description="Query string")
    task_id: str = Field(..., description="Task identifier")
    user_id: str = Field(default="user-default", description="User executing query")
    expected: GoldExpectedResult = Field(..., description="Expected results")
    gold_answer: str | None = Field(
        default=None,
        description="Optional canonical answer used by answer-quality judges",
    )
    is_followup: bool = Field(default=False, description="Is this a follow-up query")
    references_turn: int | None = Field(
        default=None,
        description="Conversation turn this query follows up on",
    )

    model_config = {"frozen": True}


class GoldEvaluationCriteria(BaseModel):
    """Evaluation criteria for a gold dataset.

    Attributes:
        recall_k: The K value for Recall@K computation.
        temporal_tolerance_days: Tolerance in days for temporal accuracy.
    """

    recall_k: int = Field(default=5, ge=1, le=100, description="K for Recall@K")
    temporal_tolerance_days: int = Field(
        default=1,
        ge=0,
        description="Temporal tolerance in days",
    )

    model_config = {"frozen": True}


class GoldDataset(BaseModel):
    """Complete gold dataset for a benchmark scenario.

    Gold datasets are the source of truth for evaluation.
    They are versioned, immutable, and scenario-specific.

    Attributes:
        schema_version: Version of the gold dataset schema.
        scenario: Name of the scenario this dataset belongs to.
        description: Human-readable description of the scenario.
        user_ids: List of distinct user IDs simulated in this dataset.
        total_conversation_turns: Total conversation turns across all users.
        events: Memory events organized by simulated day.
        queries: Evaluation queries with expected results.
        evaluation_criteria: Criteria for metric computation.
    """

    schema_version: str = Field(default="1.0", description="Schema version")
    scenario: str = Field(..., min_length=1, description="Scenario name")
    description: str = Field(..., min_length=1, description="Scenario description")
    user_ids: list[str] = Field(
        default_factory=lambda: ["user-default"],
        description="Distinct user IDs in this dataset",
    )
    total_conversation_turns: int = Field(
        default=0,
        ge=0,
        description="Total conversation turns across all users",
    )
    events: list[GoldDayEvents] = Field(
        ...,
        min_length=1,
        description="Events by simulated day",
    )
    queries: list[GoldQuery] = Field(
        ...,
        min_length=1,
        description="Evaluation queries",
    )
    evaluation_criteria: GoldEvaluationCriteria = Field(
        default_factory=GoldEvaluationCriteria,
        description="Evaluation criteria",
    )

    model_config = {"frozen": True}
