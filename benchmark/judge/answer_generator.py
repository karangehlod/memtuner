"""Answer generator — uses retrieved memories to generate an answer via LLM.

This is the bridge between retrieval and evaluation:
    retrieved_memories → LLM prompt → generated_answer → judge scoring
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.judge.llm_client import LLMClient

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant with access to memory about past conversations.
Use ONLY the provided memories to answer the question. If the memories don't contain enough information, say "I don't have enough information."
Be concise — answer in 1-2 sentences."""

ANSWER_USER_TEMPLATE = """Memories:
{memories}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class GeneratedAnswer:
    """A generated answer with its source memories."""

    answer: str
    query: str
    memories_used: list[str]
    model: str


class AnswerGenerator:
    """Generates answers from retrieved memories using an LLM.

    Pipeline:
        1. Format retrieved memory contents into a prompt
        2. Send to LLM with system instructions
        3. Return the generated answer
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialize with an LLM client.

        Args:
            client: LLM client instance. If None, creates one from env config.
        """
        self._client = client or LLMClient()

    def generate(
        self,
        query: str,
        memory_contents: list[str],
    ) -> GeneratedAnswer:
        """Generate an answer from retrieved memory contents.

        Args:
            query: The user's question.
            memory_contents: List of memory text contents (from retrieval).

        Returns:
            GeneratedAnswer with the LLM's response.
        """
        if not memory_contents:
            return GeneratedAnswer(
                answer="I don't have enough information.",
                query=query,
                memories_used=[],
                model=self._client._config.model,
            )

        # Format memories into a numbered list
        formatted_memories = "\n".join(
            f"  [{i + 1}] {content[:200]}" for i, content in enumerate(memory_contents[:10])
        )

        prompt = ANSWER_USER_TEMPLATE.format(
            memories=formatted_memories,
            question=query,
        )

        answer = self._client.generate(prompt=prompt, system=ANSWER_SYSTEM_PROMPT)

        return GeneratedAnswer(
            answer=answer,
            query=query,
            memories_used=memory_contents[:10],
            model=self._client._config.model,
        )
