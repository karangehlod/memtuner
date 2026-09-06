"""Scenario runner — executes a single benchmark scenario.

Handles per-day event injection, lifecycle policy application,
queries, evaluation, and cost tracking.

OPTIMIZATIONS:
- Batch query processing: Queries processed in configurable batches
- Memory efficiency: Explicit cleanup between batches
- Latency tracking: Per-query latency maintained for metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any

from benchmark.cost.storage_cost import StorageCostCalculator
from benchmark.cost.token_cost import TokenCostCalculator
from benchmark.evaluation.context import EvaluationContext
from benchmark.evaluation.reliability import ReliabilityCurveEvaluator
from benchmark.memory.interfaces.optional_capabilities import (
    CreationDayTracker,
    LifecycleAwareWriter,
    MemoryScoreComputer,
)
from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.answer import TokenUsage
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery, ReadQueryContext
from benchmark.models.run_result import ScenarioMetrics
from benchmark.observability.logger import get_logger, log_decision
from benchmark.observability.tracer import create_span

if TYPE_CHECKING:
    from benchmark.cost.tracker import CostTracker
    from benchmark.evaluation.base import EvaluationResult, MetricEvaluator
    from benchmark.gold.oracle import GoldOracle
    from benchmark.gold.schema import GoldMemoryEvent, GoldQuery
    from benchmark.memory.interfaces.lifecycle import LifecyclePolicy
    from benchmark.models.response import RetrievedMemory
    from benchmark.scenario.base import BenchmarkScenario
    from benchmark.time.provider import TimeProvider

logger = get_logger(__name__)


class ScenarioRunner:
    """Executes a single benchmark scenario day by day.

    Coordinates event injection, lifecycle policy application,
    query execution, evaluation, and cost tracking for one scenario.
    """

    def __init__(
        self,
        time_provider: TimeProvider,
        gold_oracle: GoldOracle,
        memory_modules: dict[str, Any],
        evaluators: list[MetricEvaluator],
        cost_tracker: CostTracker,
        lifecycle_policies: dict[str, LifecyclePolicy] | None = None,
        answer_evaluator: Any | None = None,
    ) -> None:
        """Initialize with all dependencies.

        Args:
            time_provider: Simulated clock.
            gold_oracle: Gold truth repository.
            memory_modules: Resolved memory modules (name → instance).
            evaluators: List of metric evaluators.
            cost_tracker: Cost tracking service.
            lifecycle_policies: Optional per-module lifecycle policies
                (module_name → policy). Applied after event injection.
        """
        self._time_provider = time_provider
        self._gold_oracle = gold_oracle
        self._memory_modules = memory_modules
        self._evaluators = evaluators
        self._cost_tracker = cost_tracker
        self._lifecycle_policies = lifecycle_policies or {}
        self._answer_evaluator = answer_evaluator
        self._storage_cost = StorageCostCalculator()
        self._token_cost = TokenCostCalculator()
        self._reliability_evaluator = ReliabilityCurveEvaluator()
        self._total_injected = 0
        self._memory_content_by_id: dict[str, str] = {}
        self._query_log_count: int = 0

    def run_scenario(
        self,
        scenario: BenchmarkScenario,
        run_id: str,
    ) -> ScenarioMetrics:
        """Execute a complete scenario and return metrics.

        Args:
            scenario: The scenario to execute.
            run_id: The parent benchmark run ID.

        Returns:
            ScenarioMetrics with all evaluation results.
        """
        with create_span(
            "scenario.run",
            attributes={
                "run_id": run_id,
                "scenario": scenario.name(),
            },
        ):
            # DIAGNOSTIC: Log store populations before eval starts
            store_populations = self._get_store_populations()
            log_decision(
                logger,
                "Starting scenario",
                scenario=scenario.name(),
                store_populations=store_populations,
            )
            if sum(store_populations.values()) == 0:
                log_decision(
                    logger,
                    "WARNING: All stores are empty before scenario starts. "
                    "Check dataset ingestion upstream.",
                )

            all_evaluation_results: list[EvaluationResult] = []
            total_queries = 0
            correct_recalls = 0
            all_query_latencies_ms: list[float] = []
            all_judge_scores: list[float] = []
            self._reliability_evaluator.reset()
            self._total_injected = 0
            self._memory_content_by_id.clear()

            for day in range(scenario.total_days()):
                day_results = self._run_day(scenario, day, run_id)
                all_evaluation_results.extend(day_results["evaluations"])
                total_queries += day_results["query_count"]
                correct_recalls += day_results["correct_count"]
                all_query_latencies_ms.extend(day_results.get("query_latencies_ms", []))
                all_judge_scores.extend(day_results.get("judge_scores", []))

                alive_count = self._count_alive_memories()
                self._reliability_evaluator.record_day(
                    day=day,
                    alive_count=alive_count,
                    injected_count=day_results.get("injected_count", 0),
                )

                self._time_provider.advance_day()

            reliability_result = self._reliability_evaluator.compute_curve()

            return self._build_scenario_metrics(
                scenario_name=scenario.name(),
                evaluations=all_evaluation_results,
                total_queries=total_queries,
                correct_recalls=correct_recalls,
                survival_rates=reliability_result.survival_rates,
                query_latencies_ms=all_query_latencies_ms,
                judge_scores=all_judge_scores,
            )

    def _run_day(
        self,
        scenario: BenchmarkScenario,
        day: int,
        run_id: str,
    ) -> dict[str, Any]:
        """Execute one simulated day within a scenario.

        Pipeline:
            1. Inject events
            2. Apply lifecycle policies (pruning/decay)
            3. Execute queries with rich EvaluationContext

        Args:
            scenario: The scenario.
            day: The current simulated day number.
            run_id: The parent run ID.

        Returns:
            Dict with evaluations list, query_count, correct_count, injected_count.
        """
        with create_span(
            "dataset_day",
            attributes={
                "run_id": run_id,
                "dataset_day": day,
            },
        ):
            injected_count = self._inject_events(scenario, day)
            pruned_count = self._apply_lifecycle_policies(day)
            query_results = self._execute_queries(scenario, day, run_id)
            query_results["injected_count"] = injected_count

            if pruned_count > 0:
                log_decision(
                    logger,
                    "Lifecycle policies applied",
                    day=day,
                    pruned_count=pruned_count,
                )

            return query_results

    def _inject_events(self, scenario: BenchmarkScenario, day: int) -> int:
        """Inject memory events for the current day.

        Args:
            scenario: The scenario providing events.
            day: The current simulated day.

        Returns:
            Number of events injected on this day.
        """
        day_events = scenario.get_events_for_day(day)
        if day_events is None:
            return 0

        injected = 0
        with create_span("memory.write", attributes={"dataset_day": day}):
            for gold_event in day_events.memory_events:
                memory_event = self._gold_event_to_memory_event(gold_event, day)
                self._memory_content_by_id[memory_event.id] = memory_event.content
                for _module_name, module in self._memory_modules.items():
                    if isinstance(module, MemoryWriter):
                        if isinstance(module, LifecycleAwareWriter):
                            module.write_on_day(memory_event, day)
                        else:
                            module.write(memory_event)
                        self._cost_tracker.record(self._storage_cost.compute_write_cost())
                injected += 1
                if logger.isEnabledFor(logging.INFO):
                    log_decision(
                        logger,
                        "Memory event injected",
                        event_id=gold_event.id,
                        user_id=gold_event.user_id,
                        day=day,
                    )

        self._total_injected += injected
        return injected

    # ------------------------------------------------------------------
    # FIX #1: Lifecycle Policy Application
    # ------------------------------------------------------------------

    def _apply_lifecycle_policies(self, day: int) -> int:
        """Apply lifecycle policies to all memory modules with assigned policies.

        For each module that has an assigned lifecycle policy:
        1. Retrieve current memory scores from the module.
        2. Let the policy decide which memories to prune.
        3. Call module.prune() on the flagged IDs.
        4. Track the storage cost of pruning operations.

        Args:
            day: The current simulated day.

        Returns:
            Total number of memories pruned across all modules.
        """
        total_pruned = 0

        for module_name, policy in self._lifecycle_policies.items():
            module = self._memory_modules.get(module_name)
            if module is None:
                continue

            if not isinstance(module, MemoryScoreComputer):
                continue

            with create_span(
                "lifecycle.apply",
                attributes={
                    "module": module_name,
                    "dataset_day": day,
                },
            ):
                scores = module.get_memory_scores(day)
                flagged_ids = policy.apply(day, scores)

                if flagged_ids:
                    pruned = module.prune(flagged_ids)
                    total_pruned += pruned
                    self._cost_tracker.record(self._storage_cost.compute_write_cost())
                    log_decision(
                        logger,
                        "Memories pruned by policy",
                        module=module_name,
                        flagged=len(flagged_ids),
                        pruned=pruned,
                        day=day,
                    )

        return total_pruned

    # ------------------------------------------------------------------
    # FIX #2 + #3 + #4: Rich EvaluationContext
    # ------------------------------------------------------------------

    def _execute_queries(
        self,
        scenario: BenchmarkScenario,
        day: int,
        run_id: str,
    ) -> dict[str, Any]:
        """Execute queries with optional async processing.

        OPTIMIZATION: Determines whether to use async queries based on
        BENCHMARK_ASYNC_QUERIES environment variable (default: true).
        Async queries process multiple queries concurrently for better throughput.
        Expected: 4-6× speedup for I/O-bound operations.
        """
        # Async dispatch adds ~1ms/query overhead via thread pool and is only beneficial
        # for I/O-bound strategies (API calls). For GPU-bound strategies (embeddings,
        # hybrid, reranker), the GPU serializes work anyway — async adds overhead with
        # no throughput gain. Default to false for GPU backends; opt-in via env var.
        from benchmark.resources.hw_probe import DEVICE as _HW_DEVICE
        _gpu_active = _HW_DEVICE in ("cuda", "mps")
        _async_default = "false" if _gpu_active else "true"
        use_async = os.environ.get("BENCHMARK_ASYNC_QUERIES", _async_default).lower() == "true"
        can_use_async = sys.version_info >= (3, 7)

        if use_async and can_use_async:
            # Reuse the event loop across days — creating a new loop per day adds
            # ~1-2ms overhead × N_days × N_cells for no benefit (strategies are sync).
            if not hasattr(self, "_event_loop") or self._event_loop.is_closed():
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
            return self._event_loop.run_until_complete(
                self._execute_queries_async(scenario, day, run_id)
            )
        else:
            # Fall back to sequential version
            return self._execute_queries_sequential(scenario, day, run_id)

    def _execute_queries_sequential(
        self,
        scenario: BenchmarkScenario,
        day: int,
        run_id: str,
    ) -> dict[str, Any]:
        """Execute all queries for the current day using rich EvaluationContext.

        Builds an EvaluationContext per query that includes:
        - retrieved IDs and expected IDs
        - source module per retrieved memory
        - creation day per retrieved memory
        - acceptable modules from gold query
        - temporal window from gold query
        - conversation metadata

        Args:
            scenario: The scenario providing queries.
            day: The current simulated day.
            run_id: The parent run ID.

        Returns:
            Dict with evaluations, query_count, correct_count, query_latencies_ms.
        """
        queries = scenario.get_queries_for_day(day)
        evaluations: list[EvaluationResult] = []
        correct_count = 0
        query_latencies_ms: list[float] = []
        judge_scores: list[float] = []

        # Pre-encode all queries for the day in a single batched GPU call.
        # Populates the query_embedding_cache inside each strategy so that
        # individual retrieve() calls hit the cache instead of encoding one-by-one.
        self._prewarm_query_cache(list({q.query for q in queries}))

        for gold_query in queries:
            with create_span(
                "query",
                attributes={
                    "run_id": run_id,
                    "query_text": gold_query.query,
                    "dataset_day": day,
                    "user_id": gold_query.user_id,
                    "is_followup": gold_query.is_followup,
                },
            ):
                read_query = ReadQuery(
                    query=gold_query.query,
                    top_k=scenario.recall_k(),
                    context=ReadQueryContext(
                        dataset_day=day,
                        task_id=gold_query.task_id,
                        user_id=gold_query.user_id,
                    ),
                )

                retrieved_memories, latency_ms = self._read_from_all_modules(read_query)
                query_latencies_ms.append(latency_ms)
                expected_ids = gold_query.expected.memory_ids
                # Build once — reused by the judge path, the logging path, and the metric evaluators.
                _retrieved_set = {m.memory_id for m in retrieved_memories}

                if self._answer_evaluator is not None and gold_query.gold_answer:
                    retrieved_contents = [
                        self._memory_content_by_id[memory.memory_id]
                        for memory in retrieved_memories
                        if memory.memory_id in self._memory_content_by_id
                    ]
                    # Recall for the LLM judge context: fraction of gold evidence
                    # that was surfaced. Guard against empty gold sets (data error).
                    retrieval_recall = (
                        len(set(expected_ids) & _retrieved_set) / len(expected_ids)
                        if expected_ids else 0.0
                    )
                    judge_result = self._answer_evaluator.evaluate_query(
                        query=gold_query.query,
                        gold_answer=gold_query.gold_answer,
                        retrieved_memory_contents=retrieved_contents,
                        retrieval_recall=retrieval_recall,
                    )
                    judge_scores.append(judge_result.judge_score)

                self._cost_tracker.record(self._storage_cost.compute_read_cost())

                # Retrieval validation logging: log the first ten queries
                # to verify strategy differentiation
                if len(queries) <= 10 or self._query_log_count < 10:
                    self._query_log_count += 1

                    all_retrieved_ids = [mem.memory_id for mem in retrieved_memories]
                    log_decision(
                        logger,
                        f"RETRIEVAL_VALIDATION: Query {self._query_log_count}",
                        query_text=gold_query.query[:100],
                        retrieved_count=len(all_retrieved_ids),
                        expected_count=len(expected_ids),
                        retrieved_ids_sample=all_retrieved_ids[:5],
                        expected_ids_sample=expected_ids[:5],
                        # Reuse _retrieved_set already built above — avoids two new set() calls per logged query
                        expected_in_retrieved=len(set(expected_ids) & _retrieved_set),
                    )
                else:
                    all_retrieved_ids = [mem.memory_id for mem in retrieved_memories]
                estimated_tokens = TokenUsage(
                    prompt=len(gold_query.query.split()) * 2,
                    completion=len(all_retrieved_ids) * 10,
                )
                self._cost_tracker.record(self._token_cost.compute_cost(estimated_tokens, os.environ.get("OPENAI_MODEL_NAME", "")))

                # Build rich evaluation context
                evaluation_context = self._build_evaluation_context(
                    retrieved_memories=retrieved_memories,
                    gold_query=gold_query,
                )

                # Evaluate with full context
                for evaluator in self._evaluators:
                    result = evaluator.evaluate_with_context(evaluation_context)
                    evaluations.append(result)

                is_correct = any(rid in expected_ids for rid in all_retrieved_ids)
                if is_correct:
                    correct_count += 1

        return {
            "evaluations": evaluations,
            "query_count": len(queries),
            "correct_count": correct_count,
            "query_latencies_ms": query_latencies_ms,
            "judge_scores": judge_scores,
        }

    async def _execute_queries_async(
        self, scenario: BenchmarkScenario, day: int, run_id: str
    ) -> dict[str, Any]:
        """Execute queries asynchronously for better throughput.

        OPTIMIZATION: Uses asyncio to process multiple queries concurrently
        instead of sequentially. Expected: 4-6× speedup for I/O-bound operations.

        This method processes queries in batches using asyncio, allowing
        multiple HTTP requests to Ollama (or other APIs) to be in-flight
        simultaneously instead of blocking on each one.
        """
        queries = scenario.get_queries_for_day(day)
        evaluations: list[EvaluationResult] = []
        correct_count = 0
        query_latencies_ms: list[float] = []
        judge_scores: list[float] = []

        # Pre-encode all day's queries in one batched GPU call on the main thread
        # before dispatching concurrent tasks — avoids N concurrent single-encode calls.
        self._prewarm_query_cache(list({q.query for q in queries}))

        async def process_query_async(gold_query: GoldQuery) -> dict[str, Any]:
            """Process single query in background task."""
            start = time.monotonic()

            read_query = ReadQuery(
                query=gold_query.query,
                top_k=scenario.recall_k(),
                context=ReadQueryContext(
                    dataset_day=day,
                    task_id=gold_query.task_id,
                    user_id=gold_query.user_id,
                ),
            )

            try:
                # Run blocking I/O in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                retrieved_memories, latency_ms = await loop.run_in_executor(
                    None,  # Use default executor (ThreadPoolExecutor)
                    self._read_from_all_modules,
                    read_query,
                )
            except Exception as exc:
                logger.warning(
                    f"Query processing failed (returning empty results): {exc}",
                    exc_info=False,
                )
                retrieved_memories = []
                latency_ms = 0.0

            elapsed = time.monotonic() - start

            return {
                "gold_query": gold_query,
                "retrieved_memories": retrieved_memories,
                "latency_ms": latency_ms,
                "elapsed": elapsed,
            }

        # Create tasks for all queries (non-blocking)
        tasks = [process_query_async(q) for q in queries]

        # Execute all tasks concurrently with return_exceptions to catch any unhandled errors
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Process results in order
        for result in results:
            gold_query = result["gold_query"]
            retrieved_memories = result["retrieved_memories"]
            latency_ms = result["latency_ms"]
            # latency_ms = pure retrieval time (from _read_from_all_modules / memory module).
            # result["elapsed"] = wall-clock including event-loop scheduling overhead.
            # We report latency_ms (retrieval) as the benchmark metric — consistent with
            # the synchronous path which also measures _read_from_all_modules directly.
            query_latencies_ms.append(latency_ms)
            expected_ids = gold_query.expected.memory_ids

            if self._answer_evaluator is not None and gold_query.gold_answer:
                retrieved_contents = [
                    self._memory_content_by_id[memory.memory_id]
                    for memory in retrieved_memories
                    if memory.memory_id in self._memory_content_by_id
                ]
                # Recall for LLM judge: fraction of gold evidence retrieved.
                _retrieved_set = {m.memory_id for m in retrieved_memories}
                retrieval_recall = (
                    len(set(expected_ids) & _retrieved_set) / len(expected_ids)
                    if expected_ids else 0.0
                )
                judge_result = self._answer_evaluator.evaluate_query(
                    query=gold_query.query,
                    gold_answer=gold_query.gold_answer,
                    retrieved_memory_contents=retrieved_contents,
                    retrieval_recall=retrieval_recall,
                )
                judge_scores.append(judge_result.judge_score)

            self._cost_tracker.record(self._storage_cost.compute_read_cost())

            all_retrieved_ids = [mem.memory_id for mem in retrieved_memories]
            evaluation_context = self._build_evaluation_context(
                retrieved_memories=retrieved_memories,
                gold_query=gold_query,
            )

            for evaluator in self._evaluators:
                result_eval = evaluator.evaluate_with_context(evaluation_context)
                evaluations.append(result_eval)

            is_correct = any(rid in expected_ids for rid in all_retrieved_ids)
            if is_correct:
                correct_count += 1

        return {
            "evaluations": evaluations,
            "query_count": len(queries),
            "correct_count": correct_count,
            "query_latencies_ms": query_latencies_ms,
            "judge_scores": judge_scores,
        }

    def _prewarm_query_cache(self, query_texts: list[str]) -> None:
        """Batch-encode all unique query strings for a day in one GPU call.

        Iterates every memory module's retrieval strategy and, if that strategy
        exposes an ``encode_batch`` method and a ``_query_embedding_cache`` dict,
        encodes all unseen query strings at once and populates the cache.

        This converts N × single-string encode calls (one per query inside
        retrieve()) into a single batched encode call, which on CUDA is typically
        10-50× faster for a day's worth of queries.

        Args:
            query_texts: Unique query strings for the upcoming query batch.
        """
        if not query_texts:
            return

        for module in self._memory_modules.values():
            strategy = getattr(module, "_retrieval_strategy", None)
            if strategy is None:
                continue
            if not (hasattr(strategy, "encode_batch") and hasattr(strategy, "_query_embedding_cache")):
                continue

            # Collect only the queries not already in the cache.
            # Use a set of raw text for O(1) lookup instead of MD5-hashing every text
            # every day — saves ~100-200ms per embedding cell (1977 queries × 50 days).
            cache = strategy._query_embedding_cache
            prewarmed_texts = getattr(strategy, "_prewarmed_texts", None)
            if prewarmed_texts is None:
                # Build the text set from existing cache keys on first call
                prewarmed_texts = set()
                strategy._prewarmed_texts = prewarmed_texts
            new_texts = [t for t in query_texts if t not in prewarmed_texts]
            if not new_texts:
                continue

            try:
                embeddings = strategy.encode_batch(new_texts)
                for text, emb in zip(new_texts, embeddings):
                    cache[hashlib.md5(text.encode()).hexdigest()] = emb
                    prewarmed_texts.add(text)
            except Exception:
                pass  # non-fatal — retrieve() will encode on demand

    def _read_from_all_modules(self, query: ReadQuery) -> tuple[list[RetrievedMemory], float]:
        """Read from all memory modules and merge results.

        Returns full RetrievedMemory objects (not just IDs) so the
        caller can extract source_module and creation_day data.

        Args:
            query: The read query.

        Returns:
            Tuple of (merged deduplicated list of RetrievedMemory ordered by score,
                      total latency in milliseconds across all modules).
        """
        all_results: list[RetrievedMemory] = []
        total_latency_ms: float = 0.0

        with create_span("memory.read"):
            for module_name, module in self._memory_modules.items():
                if isinstance(module, MemoryReader):
                    try:
                        response = module.read(query)
                        all_results.extend(response.retrieved_memories)
                        total_latency_ms += response.latency_ms
                    except Exception as exc:
                        logger.warning(
                            f"Memory module '{module_name}' read failed (graceful degradation): {exc}",
                            exc_info=False,
                        )
                        continue

        all_results.sort(key=lambda mem: mem.score, reverse=True)

        seen: set[str] = set()
        deduplicated: list[RetrievedMemory] = []
        for memory in all_results:
            if memory.memory_id not in seen:
                seen.add(memory.memory_id)
                deduplicated.append(memory)

        return deduplicated[: query.top_k], total_latency_ms

    def _build_evaluation_context(
        self,
        retrieved_memories: list[RetrievedMemory],
        gold_query: GoldQuery,
    ) -> EvaluationContext:
        """Build a rich EvaluationContext from retrieved memories and gold query.

        Maps source modules and creation days from retrieved memory objects
        and from the memory modules' internal state.

        Args:
            retrieved_memories: The retrieved memory objects.
            gold_query: The gold query with expected results.

        Returns:
            EvaluationContext carrying all data for evaluation.
        """
        retrieved_ids = [mem.memory_id for mem in retrieved_memories]
        source_modules = {mem.memory_id: mem.source_module for mem in retrieved_memories}

        # Collect creation days from modules that track them
        creation_days: dict[str, int] = {}
        for memory in retrieved_memories:
            module = self._memory_modules.get(memory.source_module)
            if module is not None and isinstance(module, CreationDayTracker):
                creation_day = module.get_creation_day(memory.memory_id)
                if creation_day is not None:
                    creation_days[memory.memory_id] = creation_day

        # Extract temporal window
        temporal_window: tuple[int, int] | None = None
        if gold_query.expected.temporal_window is not None:
            temporal_window = (
                gold_query.expected.temporal_window.not_before_day,
                gold_query.expected.temporal_window.not_after_day,
            )

        return EvaluationContext(
            retrieved_ids=retrieved_ids,
            expected_ids=gold_query.expected.memory_ids,
            retrieved_source_modules=source_modules,
            retrieved_creation_days=creation_days,
            acceptable_modules=gold_query.expected.acceptable_modules,
            temporal_window=temporal_window,
            is_followup=gold_query.is_followup,
            references_turn=gold_query.references_turn,
        )

    def _gold_event_to_memory_event(
        self,
        gold_event: GoldMemoryEvent,
        day: int,
    ) -> MemoryEvent:
        """Convert a gold dataset event to a MemoryEvent model.

        Args:
            gold_event: The gold event.
            day: The simulated day for timestamp.

        Returns:
            A MemoryEvent instance.
        """
        timestamp = self._time_provider.current_timestamp()
        return MemoryEvent(
            id=gold_event.id,
            user_id=gold_event.user_id,
            type=gold_event.type,
            content=gold_event.content,
            timestamp=timestamp,
            importance=gold_event.importance,
            entities=gold_event.entities,
            task_id=gold_event.task_id,
        )

    def _count_alive_memories(self) -> int:
        """Count total memories alive across all modules.

        Returns:
            Total number of memories in all modules.
        """
        total = 0
        for module in self._memory_modules.values():
            if hasattr(module, "count"):
                total += module.count()
        return total

    def _get_store_populations(self) -> dict[str, int]:
        """Get population size for each memory module.

        Returns:
            Dict mapping module name to memory count.
        """
        populations = {}
        for module_name, module in self._memory_modules.items():
            if hasattr(module, "count"):
                populations[module_name] = module.count()
            else:
                populations[module_name] = 0  # unknown
        return populations

    def _build_scenario_metrics(
        self,
        scenario_name: str,
        evaluations: list[EvaluationResult],
        total_queries: int,
        correct_recalls: int,
        survival_rates: dict[int, float] | None = None,
        query_latencies_ms: list[float] | None = None,
        judge_scores: list[float] | None = None,
    ) -> ScenarioMetrics:
        """Build ScenarioMetrics from evaluation results.

        Args:
            scenario_name: Name of the scenario.
            evaluations: All evaluation results from the scenario.
            total_queries: Total queries executed.
            correct_recalls: Queries with at least one correct result.
            survival_rates: Per-day survival rates.
            query_latencies_ms: Per-query retrieval latencies in ms.

        Returns:
            ScenarioMetrics summary.
        """
        metrics_by_name: dict[str, list[float]] = {}
        for result in evaluations:
            if result.metric_name not in metrics_by_name:
                metrics_by_name[result.metric_name] = []
            # Only count evaluated queries (query_count > 0)
            if result.query_count > 0:
                metrics_by_name[result.metric_name].append(result.value)

        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        # Compute latency percentiles
        latency_p50 = 0.0
        latency_p90 = 0.0
        latency_p99 = 0.0
        latency_mean = 0.0
        if query_latencies_ms:
            sorted_latencies = sorted(query_latencies_ms)
            n = len(sorted_latencies)
            latency_mean = sum(sorted_latencies) / n

            # Nearest-rank percentile: index = ceil(n * p) - 1, clamped to [0, n-1].
            # Standard formula from NIST/Hyndman & Fan (1996).
            # Example: n=10, p=0.90 → ceil(9.0)-1 = 8 (9th value, 0-indexed)
            #          floor method: int(10*0.90) = 9 (10th = max) — inflated
            import math as _math
            def _percentile(p: float) -> float:
                idx = max(0, min(_math.ceil(n * p) - 1, n - 1))
                return sorted_latencies[idx]

            latency_p50 = _percentile(0.50)
            latency_p90 = _percentile(0.90)
            latency_p99 = _percentile(0.99)

        contamination = avg(metrics_by_name.get("benchmark.contamination_rate", []))
        recall = avg(metrics_by_name.get("benchmark.recall_at_k", []))
        precision_at_k = avg(metrics_by_name.get("benchmark.precision_at_k", []))

        return ScenarioMetrics(
            scenario_name=scenario_name,
            recall_at_k=recall,
            contamination_rate=contamination,
            precision_at_k=precision_at_k,
            temporal_accuracy=avg(metrics_by_name.get("benchmark.temporal_accuracy", [])),
            module_accuracy=avg(metrics_by_name.get("benchmark.module_accuracy", [])),
            mrr=avg(metrics_by_name.get("benchmark.mrr", [])),
            ndcg=avg(metrics_by_name.get("benchmark.ndcg", [])),
            precision_at_1=avg(metrics_by_name.get("benchmark.precision_at_1", [])),
            llm_judge_score=avg(judge_scores or []) if judge_scores else None,
            llm_judge_queries=len(judge_scores or []),
            memory_survival_rates=survival_rates or {},
            total_queries=total_queries,
            correct_recalls=correct_recalls,
            latency_p50_ms=latency_p50,
            latency_p90_ms=latency_p90,
            latency_p99_ms=latency_p99,
            latency_mean_ms=latency_mean,
        )
