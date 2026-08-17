"""Memory read response data model.

Represents the response from a memory read operation.
This is a pure data class — no business logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    """Temperature tier of a retrieved memory.

    Indicates how recently or frequently the memory has been accessed.
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class RetrievedMemory(BaseModel):
    """A single memory item returned from a read operation.

    Attributes:
        memory_id: The unique identifier of the retrieved memory.
        source_module: Which memory module returned this result.
        score: Relevance score (higher is better, monotonic descending in list).
        confidence: Confidence level of the retrieval (0.0 = uncertain, 1.0 = certain).
        timestamp: Original creation timestamp of the memory.
        tier: Temperature tier of the memory.
        decay_factor: Current decay multiplier applied to this memory.
    """

    memory_id: str = Field(..., description="Unique memory identifier")
    source_module: str = Field(..., description="Memory module that returned this")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence (0=uncertain, 1=certain)",
    )
    timestamp: datetime = Field(..., description="Original creation timestamp")
    tier: MemoryTier = Field(..., description="Temperature tier")
    decay_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Current decay multiplier",
    )

    model_config = {"frozen": True}


class ReadResponse(BaseModel):
    """Response from a memory read operation.

    Attributes:
        retrieved_memories: Ordered list of retrieved memories (best first).
        latency_ms: Time taken for the read operation in milliseconds.
        total_candidates: Total number of candidates considered before top-k filtering.
    """

    retrieved_memories: list[RetrievedMemory] = Field(
        default_factory=list,
        description="Ordered list of retrieved memories",
    )
    latency_ms: float = Field(..., ge=0.0, description="Read latency in milliseconds")
    total_candidates: int = Field(
        default=0,
        ge=0,
        description="Total candidates considered",
    )

    model_config = {"frozen": True}
