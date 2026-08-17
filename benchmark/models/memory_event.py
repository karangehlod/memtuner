"""Memory event data model.

Represents a single memory event to be written into a memory module.
This is a pure data class — no business logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Enumeration of supported memory types.

    These are fixed benchmark rules and cannot be configured.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PREFERENCE = "preference"
    ENTITY = "entity"


class MemoryEvent(BaseModel):
    """A single memory event to be written into a memory module.

    Attributes:
        id: Unique memory identifier (e.g., "M-102").
        user_id: Identifier of the user this memory belongs to.
        type: The category of memory.
        content: Human-readable content of the memory.
        timestamp: When this memory was created.
        importance: Importance score between 0.0 and 1.0.
        entities: List of entities mentioned in this memory.
        task_id: Identifier of the task this memory relates to.
        metadata: Extensible metadata dict for future use (OCP).
    """

    id: str = Field(..., description="Unique memory identifier")
    user_id: str = Field(default="user-default", description="User this memory belongs to")
    type: MemoryType = Field(..., description="Category of memory")
    content: str = Field(..., min_length=1, description="Memory content")
    timestamp: datetime = Field(..., description="When this memory was created")
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Importance score between 0.0 and 1.0",
    )
    entities: list[str] = Field(default_factory=list, description="Entities in this memory")
    task_id: str = Field(..., description="Related task identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for future use",
    )

    model_config = {"frozen": True}
