"""Memory read query data model.

Represents a query to retrieve memories from a memory module.
This is a pure data class — no business logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from benchmark.models.memory_event import MemoryType


class ReadQueryFilters(BaseModel):
    """Optional filters for scoped memory retrieval.

    Attributes:
        memory_types: Restrict retrieval to specific memory types.
        min_importance: Minimum importance threshold.
    """

    memory_types: list[MemoryType] = Field(
        default_factory=list,
        description="Restrict to specific memory types",
    )
    min_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum importance threshold",
    )

    model_config = {"frozen": True}


class ReadQueryContext(BaseModel):
    """Context information for a memory read query.

    Attributes:
        dataset_day: The current day index within the dataset replay.
        task_id: The task this query relates to.
        user_id: The user executing this query.
    """

    dataset_day: int = Field(..., ge=0, description="Current dataset day")
    task_id: str = Field(..., description="Related task identifier")
    user_id: str = Field(default="user-default", description="User executing this query")

    model_config = {"frozen": True}


class ReadQuery(BaseModel):
    """A query to retrieve memories from a memory module.

    Attributes:
        query: The natural language query string.
        top_k: Maximum number of results to return.
        context: Contextual information for the query.
        filters: Optional filters for scoped retrieval.
    """

    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum results to return")
    context: ReadQueryContext = Field(..., description="Query context")
    filters: ReadQueryFilters = Field(
        default_factory=ReadQueryFilters,
        description="Optional retrieval filters",
    )

    model_config = {"frozen": True}
