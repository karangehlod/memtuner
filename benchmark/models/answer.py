"""Answer request and response data models.

Represents the optional RAG/LLM answering capability.
This is a pure data class — no business logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage breakdown for an LLM call.

    Attributes:
        prompt: Number of tokens in the prompt.
        completion: Number of tokens in the completion.
    """

    prompt: int = Field(..., ge=0, description="Prompt tokens")
    completion: int = Field(..., ge=0, description="Completion tokens")

    model_config = {"frozen": True}

    @property
    def total(self) -> int:
        """Total tokens used."""
        return self.prompt + self.completion


class AnswerRequest(BaseModel):
    """Request for the optional answering system.

    Attributes:
        question: The question to answer using memory context.
        memory_context: List of memory IDs to use as context.
    """

    question: str = Field(..., min_length=1, description="Question to answer")
    memory_context: list[str] = Field(
        default_factory=list,
        description="Memory IDs for context",
    )

    model_config = {"frozen": True}


class AnswerResponse(BaseModel):
    """Response from the optional answering system.

    Attributes:
        answer: The generated answer text.
        tokens_used: Token usage breakdown.
        latency_ms: Time taken for the answer in milliseconds.
        model: The model used for generation.
    """

    answer: str = Field(..., description="Generated answer")
    tokens_used: TokenUsage = Field(..., description="Token usage breakdown")
    latency_ms: float = Field(..., ge=0.0, description="Answer latency in milliseconds")
    model: str = Field(..., description="Model used for generation")

    model_config = {"frozen": True}
