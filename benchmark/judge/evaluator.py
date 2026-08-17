"""End-to-end LLM evaluation — retrieval → generation → judging.

This is the full pipeline that measures actual answer quality,
not just retrieval ID matching.

Pipeline:
    1. Query → Memory Store → Retrieved memories (existing benchmark)
    2. Retrieved memory contents → LLM → Generated answer (NEW)
    3. Generated answer vs Gold answer → Judge → Score (NEW)

This closes the gap between "did we find the right IDs?" and
"did we produce the right answer?"
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.judge.answer_generator import AnswerGenerator, GeneratedAnswer
from benchmark.judge.judge import AnswerJudge
from benchmark.judge.llm_client import LLMClient, get_judge_config, get_llm_config


@dataclass(frozen=True)
class EndToEndResult:
    """Complete evaluation result for one query."""

    query: str
    gold_answer: str
    generated_answer: str
    retrieval_recall: float
    judge_score: float
    judge_method: str
    memories_used_count: int


class EndToEndEvaluator:
    """Evaluates the full memory pipeline: retrieve → generate → judge.

    This is what production memory systems are actually measured on:
    "Given the user's question, did the agent produce the right answer?"

    Uses OpenAI-compatible endpoints configured via environment variables:
        BENCHMARK_LLM_BASE_URL, BENCHMARK_LLM_API_KEY, BENCHMARK_LLM_MODEL
        BENCHMARK_JUDGE_BASE_URL, BENCHMARK_JUDGE_API_KEY, BENCHMARK_JUDGE_MODEL
    """

    def __init__(
        self,
        judge_method: str = "token_f1",
        generator_client: LLMClient | None = None,
        judge_client: LLMClient | None = None,
    ) -> None:
        """Initialize the end-to-end evaluator.

        Args:
            judge_method: Scoring method — "token_f1" (fast, no LLM)
                         or "llm_judge" (accurate, needs LLM endpoint).
            generator_client: LLM client for answer generation.
            judge_client: LLM client for answer judging.
        """
        self._judge_method = judge_method
        self._generator = AnswerGenerator(generator_client)
        self._judge = AnswerJudge(judge_client)

    def evaluate_query(
        self,
        query: str,
        gold_answer: str,
        retrieved_memory_contents: list[str],
        retrieval_recall: float,
    ) -> EndToEndResult:
        """Evaluate a single query end-to-end.

        Args:
            query: The user's question.
            gold_answer: Expected answer from gold dataset.
            retrieved_memory_contents: Text contents of retrieved memories.
            retrieval_recall: Pre-computed retrieval recall for reference.

        Returns:
            EndToEndResult with all evaluation data.
        """
        # VRAM guard: when a large embedding model (e.g. Qwen3-4B at 7.6 GB) is
        # active and the judge model (e.g. gemma4:12b at 8 GB Q4) would exceed
        # total VRAM, offload PyTorch embedding cache before the Ollama judge call.
        import os as _os
        if _os.environ.get("BENCHMARK_JUDGE_UNLOAD_EMBED") == "1":
            try:
                from benchmark.memory.strategies.embeddings_strategy import (
                    _MODEL_CACHE, _MODEL_CACHE_ORDER,
                )
                import torch as _t
                for _k in list(_MODEL_CACHE.keys()):
                    try:
                        _MODEL_CACHE[_k].to("cpu")
                    except Exception:
                        pass
                _t.cuda.empty_cache()
            except Exception:
                pass

        # Step 1: Generate answer from memories
        if self._judge_method == "llm_judge":
            generated = self._generator.generate(query, retrieved_memory_contents)
            # Step 2: Judge the answer
            judge_result = self._judge.judge_with_llm(
                question=query,
                expected_answer=gold_answer,
                generated_answer=generated.answer,
            )
        else:
            # Token F1 — doesn't need actual generation, just compare directly
            # Use the memory contents as a proxy for the "answer"
            combined_memories = " ".join(retrieved_memory_contents[:5])
            generated = GeneratedAnswer(
                answer=combined_memories[:500],
                query=query,
                memories_used=retrieved_memory_contents[:5],
                model="token_f1_proxy",
            )
            judge_result = AnswerJudge.judge_with_token_f1(
                question=query,
                expected_answer=gold_answer,
                generated_answer=combined_memories,
            )

        return EndToEndResult(
            query=query,
            gold_answer=gold_answer,
            generated_answer=generated.answer[:200],
            retrieval_recall=retrieval_recall,
            judge_score=judge_result.score,
            judge_method=judge_result.method,
            memories_used_count=len(retrieved_memory_contents),
        )

    def is_llm_available(self) -> bool:
        """Check if LLM endpoint is reachable (for llm_judge mode).

        Returns:
            True if the LLM server responds.
        """
        clients: list[LLMClient] = []
        try:
            clients = [
                LLMClient(get_llm_config()),
                LLMClient(get_judge_config()),
            ]
            return all(client.is_available() for client in clients)
        except Exception:
            return False
        finally:
            for client in clients:
                client.close()
