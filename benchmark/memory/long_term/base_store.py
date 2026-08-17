"""Base long-term memory store.

Extracts shared write/read/prune/decay/tier/confidence logic from all
long-term stores. Concrete subclasses override only `_compute_relevance_score()`
to provide module-specific scoring.

This eliminates ~250 lines of duplication across the four long-term stores
and normalizes tier thresholds, confidence formulas, and pruning logic.
"""

from __future__ import annotations

import math
import threading
import time as time_module
from abc import abstractmethod

from benchmark.memory.interfaces.reader import MemoryReader
from benchmark.memory.interfaces.writer import MemoryWriter
from benchmark.models.memory_event import MemoryEvent
from benchmark.models.query import ReadQuery
from benchmark.models.response import MemoryTier, ReadResponse, RetrievedMemory

# Normalized tier thresholds — single source of truth.
_HOT_THRESHOLD: float = 0.7
_WARM_THRESHOLD: float = 0.3


class BaseLongTermStore(MemoryWriter, MemoryReader):
    """Abstract base for all long-term memory stores.

    Provides shared infrastructure for:
    - Day-tagged writes
    - Decay-adjusted scoring
    - User-scoped reads with `ReadQueryFilters` support
    - Tier classification
    - Confidence computation
    - Pruning by memory ID list

    Subclasses implement only `_compute_relevance_score()` to define
    how each module scores a memory against a query.

    Constructor accepts `module_name` to eliminate hardcoded source_module strings.
    """

    def __init__(
        self,
        *,
        module_name: str | None = None,
        decay_type: str = "exponential",
        decay_lambda: float = 0.05,
        pruning_threshold: float = 0.35,
        retrieval_strategy: object | None = None,
        allow_strategy_fallback: bool = False,
        decay_ranking_alpha: float = 0.0,
        archival_floor: float | None = 0.65,
        archival_day_threshold: int = 90,
        tiered_working_days: int = 7,
        **_kwargs: object,
    ) -> None:
        """Initialize the base long-term store.

        Args:
            module_name: Logical name for this module. Falls back to
                the lowercased class name if not provided.
            decay_type: Decay formula — 'exponential', 'linear', 'logarithmic',
                or 'tiered'. Any unknown value defaults to exponential.
            decay_lambda: Decay rate parameter (λ).
            pruning_threshold: Score threshold for pruning eligibility.
            retrieval_strategy: Optional RetrievalStrategy instance for scoring.
            allow_strategy_fallback: Whether read() may silently fall back to
                default scoring if retrieval_strategy.retrieve() fails.
                Default False for benchmark correctness.
            decay_ranking_alpha: How much decay affects ranking (0.0-1.0).
                0.0 = decay is post-ranking only (no effect on which memories are retrieved).
                1.0 = full decay in ranking (newer memories strongly preferred).
                0.5 = moderate recency bias in ranking.
            archival_floor: Minimum decay factor for memories older than archival_day_threshold.
                Set to None to disable (allows full exponential decay).
                Default 0.65 preserves old memories at 65% weight.
            archival_day_threshold: Age in days at which the archival floor kicks in.
                Also the upper boundary of the tiered policy's episodic fade zone.
                Default 90 days. Phase 4b sweeps this parameter.
            tiered_working_days: Tiered policy only: working memory window (no decay).
                Default 7 days. Only used when decay_type='tiered'.
            **_kwargs: Ignored (uniform factory construction).
        """
        from benchmark.observability.logger import get_logger
        logger = get_logger(__name__)

        self._module_name = module_name or type(self).__name__.lower()
        logger.info(f"[TRACE] Initializing {self._module_name}: allow_strategy_fallback={allow_strategy_fallback}")
        self._memories: dict[str, MemoryEvent] = {}
        self._creation_days: dict[str, int] = {}
        self._decay_type = decay_type
        self._decay_lambda = decay_lambda
        self._pruning_threshold = pruning_threshold
        self._retrieval_strategy = retrieval_strategy
        self._allow_strategy_fallback = allow_strategy_fallback
        self._decay_ranking_alpha = max(0.0, min(1.0, decay_ranking_alpha))
        self._archival_floor = archival_floor
        self._archival_day_threshold = max(1, archival_day_threshold)
        self._tiered_working_days = max(0, tiered_working_days)
        # Dirty flag: True means strategy index must be rebuilt before next read
        self._index_dirty: bool = True
        # Lock prevents concurrent threads (async query path) from triggering
        # simultaneous re-encodes when the dirty flag is set.
        self._index_lock: threading.Lock = threading.Lock()
        # Once a strategy permanently fails, disable it for the whole cell run
        self._strategy_permanently_disabled: bool = False
        # DSA: User-scoped index for O(1) user filtering instead of O(N) scan
        self._user_memories: dict[str, set[str]] = {}
        # DSA: Decay factor LRU cache to avoid recomputing exp() repeatedly
        self._decay_cache: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Decay formula
    # ------------------------------------------------------------------

    def _compute_decay_factor(self, days_elapsed: int) -> float:
        """Compute decay factor for a memory age of days_elapsed days.

        Uses memoization cache: same days_elapsed always yields same result
        (deterministic), so cache across queries within a single run.

        Supported formulas:
          exponential:  exp(-λ * t) with archival floor at 90+ days
          linear:       max(0, 1 - λ * t)
          logarithmic:  1 / (1 + λ * t)
          tiered:       class-based decay —
                        [0-7 days]  → 1.0 (working memory, no decay)
                        [7-90 days] → exp(-λ * (t - 7)) (episodic fade)
                        [90+ days]  → 1.0 (archival, no decay — survived long enough)

        All formulas enforce an archival floor of 0.65 for memories older
        than 90 days. Gold memories in the corpus run up to 294 days old;
        continuous decay without a floor zeros them out incorrectly.

        Args:
            days_elapsed: Non-negative age of the memory in simulated days.

        Returns:
            Decay factor in [0, 1].
        """
        # DSA: Memoize decay computation (same λ + same days = same result)
        if days_elapsed in self._decay_cache:
            return self._decay_cache[days_elapsed]

        t = max(0, days_elapsed)
        lam = self._decay_lambda
        if lam == 0.0:
            self._decay_cache[days_elapsed] = 1.0
            return 1.0

        archival_floor = self._archival_floor
        archival_day_threshold = self._archival_day_threshold
        tiered_working_days = self._tiered_working_days

        if self._decay_type == "linear":
            raw = max(0.0, 1.0 - lam * t)
            if archival_floor is not None and t >= archival_day_threshold:
                result = max(archival_floor, raw)
            else:
                result = raw
        elif self._decay_type == "logarithmic":
            raw = 1.0 / (1.0 + lam * t)
            if archival_floor is not None and t >= archival_day_threshold:
                result = max(archival_floor, raw)
            else:
                result = raw
        elif self._decay_type == "tiered":
            if t <= tiered_working_days or t >= archival_day_threshold:
                result = 1.0
            else:
                result = math.exp(-lam * (t - tiered_working_days))
        else:
            # exponential (default)
            raw = math.exp(-lam * t)
            if archival_floor is not None and t >= archival_day_threshold:
                result = max(archival_floor, raw)
            else:
                result = raw

        self._decay_cache[days_elapsed] = result
        return result

    # ------------------------------------------------------------------

    @property
    def module_name(self) -> str:
        """Return the logical name of this memory module."""
        return self._module_name

    # ------------------------------------------------------------------
    # MemoryWriter interface
    # ------------------------------------------------------------------

    def write(self, event: MemoryEvent) -> None:
        """Write a memory event to the store.

        Args:
            event: The memory event to store.
        """
        self._memories[event.id] = event
        self._creation_days[event.id] = 0
        self._index_dirty = True
        # Maintain user index
        uid = event.user_id or "__none__"
        if uid not in self._user_memories:
            self._user_memories[uid] = set()
        self._user_memories[uid].add(event.id)

    def write_on_day(self, event: MemoryEvent, day: int) -> None:
        """Write a memory event tagged with its injection day.

        Args:
            event: The memory event to store.
            day: The simulated day of injection.
        """
        self._memories[event.id] = event
        self._creation_days[event.id] = day
        self._index_dirty = True
        # Maintain user index
        uid = event.user_id or "__none__"
        if uid not in self._user_memories:
            self._user_memories[uid] = set()
        self._user_memories[uid].add(event.id)

    # ------------------------------------------------------------------
    # MemoryReader interface
    # ------------------------------------------------------------------

    def read(self, query: ReadQuery) -> ReadResponse:
        """Retrieve memories with decay-adjusted scores.

        If a retrieval_strategy is configured, uses that for scoring.
        Otherwise falls back to _compute_relevance_score().

        Filters by user_id from query context. Applies `ReadQueryFilters`
        for memory_types and min_importance if provided.

        Args:
            query: The read query.

        Returns:
            ReadResponse with decay-adjusted scored results.
        """
        start_time = time_module.monotonic()
        current_day = query.context.dataset_day
        user_id = query.context.user_id

        candidates = self._filter_candidates(query)

        # Use retrieval strategy if provided and not permanently disabled
        if self._retrieval_strategy and not self._strategy_permanently_disabled:
            # Only re-index when memories have changed since last index.
            # Lock ensures at most one thread re-encodes; subsequent threads
            # see dirty=False after the first finishes and skip re-encoding.
            if self._index_dirty:
                with self._index_lock:
                    if self._index_dirty:  # re-check inside lock
                        _index_attempts = 0
                        _index_success = False
                        while _index_attempts < 3 and not _index_success and self._retrieval_strategy:
                            try:
                                self._retrieval_strategy.index(list(self._memories.values()))
                                self._index_dirty = False
                                _index_success = True
                            except Exception as e:
                                _index_attempts += 1
                                _err_lower = str(e).lower()
                                # 404 = model not found in Ollama — hard config error,
                                # never transient. Raise immediately so the cell is
                                # marked failed rather than silently falling back to
                                # SequenceMatcher and producing misleading recalls.
                                _is_hard_config_error = any(kw in _err_lower for kw in
                                                            ("404", "not found", "model not found"))
                                _is_transient = (not _is_hard_config_error) and any(
                                    kw in _err_lower for kw in
                                    ("connection", "timeout", "timed out", "400",
                                     "tokenize", "refused", "unavailable", "503"))
                                if _is_hard_config_error:
                                    # Surface as hard failure — do not fall back silently
                                    raise RuntimeError(
                                        f"Strategy configuration error (model not available): {str(e)[:200]}"
                                    ) from e
                                if _index_attempts < 3 and _is_transient:
                                    import time as _t
                                    _t.sleep(2 ** _index_attempts)  # 2s, 4s backoff
                                    continue
                                from benchmark.observability.logger import get_logger
                                logger_inst = get_logger(__name__)
                                if not self._allow_strategy_fallback:
                                    raise RuntimeError(
                                        "Retrieval strategy indexing failed and fallback is disabled"
                                    ) from e
                                logger_inst.warning(
                                    f"Strategy indexing failed after {_index_attempts} attempt(s), "
                                    f"disabling for this run. Error: {type(e).__name__}: {str(e)[:120]}",
                                    extra={"error": str(e), "exception_type": type(e).__name__},
                                )
                                self._index_dirty = False
                                self._strategy_permanently_disabled = True
                                self._retrieval_strategy = None
                                break

            # Use strategy to retrieve ranked memories
            # FIX #1: Fetch ALL candidates so ranking (not retrieval) is the
            # sole bottleneck. With ~588 memories per user, fetching all is
            # fast and ensures no gold memory is missed in the candidate pool.
            if self._retrieval_strategy:
                fetch_k = len(candidates)
                try:
                    strategy_results = self._retrieval_strategy.retrieve(
                        query=query.query,
                        top_k=fetch_k,
                        user_id=user_id,
                    )
                except Exception as e:
                    from benchmark.observability.logger import get_logger

                    logger_inst = get_logger(__name__)
                    logger_inst.debug(
                        f"Strategy retrieval failed. allow_strategy_fallback={self._allow_strategy_fallback}",
                        extra={"error": str(e)},
                    )
                    if not self._allow_strategy_fallback:
                        raise RuntimeError(
                            "Retrieval strategy failed during read() and fallback is disabled"
                        ) from e
                    logger_inst.warning(
                        f"Strategy retrieval failed, falling back to default scoring. Error: {type(e).__name__}: {e}",
                        extra={"error": str(e), "exception_type": type(e).__name__},
                    )
                    strategy_results = []
            else:
                strategy_results = []

            # Convert strategy results to scored triples using polymorphic
            # module weighting. Each concrete store (Entity, Preference, etc.)
            # applies its own boosts via _apply_module_weight().
            #
            # Pipeline:
            #   1. Strategy provides raw relevance signal
            #   2. Module-specific weighting (entity boost, task affinity, etc.)
            #   3. Apply decay_ranking_alpha (recency bias in ranking)
            #   4. Sort and truncate to top_k
            #   5. Apply remaining decay as post-hoc adjustment
            scored: list[tuple[MemoryEvent, float, float]] = []
            for result in strategy_results:
                try:
                    memory_id, strategy_score = result
                    if memory_id in candidates:
                        event = candidates[memory_id]
                        creation_day = self._creation_days.get(memory_id, 0)
                        days_elapsed = max(0, current_day - creation_day)
                        decay_factor = self._compute_decay_factor(days_elapsed)
                        # Apply module-specific weighting (polymorphic hook).
                        # Concrete stores override _apply_module_weight() to
                        # inject entity boosts, task affinity, importance, etc.
                        rank_score = self._apply_module_weight(
                            query, event, float(strategy_score), decay_factor
                        )
                        # Apply decay_ranking_alpha: controls how much recency
                        # affects which memories make it into the top-K.
                        # alpha=0 → no effect (pure relevance ranking)
                        # alpha=1 → full decay penalty (newer memories preferred)
                        if self._decay_ranking_alpha > 0:
                            rank_score *= decay_factor**self._decay_ranking_alpha
                        scored.append((event, rank_score, decay_factor))
                except Exception as e:
                    from benchmark.observability.logger import get_logger

                    get_logger(__name__).warning(
                        "Failed to process strategy result",
                        extra={"result": str(result), "error": str(e)},
                    )
                    continue

            # Sort by module-weighted relevance score, truncate to top_k
            scored.sort(key=lambda triple: triple[1], reverse=True)
            top_k_by_relevance = scored[: query.top_k]

            # Apply decay as post-ranking recency adjustment
            top_k = [
                (event, score * decay_factor, decay_factor)
                for event, score, decay_factor in top_k_by_relevance
            ]
        else:
            # Fall back to standard scoring via _compute_relevance_score
            scored = self._score_candidates(query, candidates, current_day, user_id)
            scored.sort(key=lambda triple: triple[1], reverse=True)
            top_k = scored[: query.top_k]

        retrieved = [
            RetrievedMemory(
                memory_id=event.id,
                source_module=self._module_name,
                score=max(0.0, min(score, 1.0)),
                confidence=self._compute_confidence(max(0.0, min(score, 1.0)), decay_factor),
                timestamp=event.timestamp,
                tier=self._compute_tier(decay_factor),
                decay_factor=decay_factor,
            )
            for event, score, decay_factor in top_k
        ]

        elapsed_ms = (time_module.monotonic() - start_time) * 1000.0

        return ReadResponse(
            retrieved_memories=retrieved,
            latency_ms=elapsed_ms,
            total_candidates=len(self._memories),
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def get_memory_scores(self, day: int) -> dict[str, float]:
        """Get current scores for all memories (for lifecycle policy decisions).

        Returns importance × decay_factor so that policies can prune memories
        that have both low relevance AND have decayed significantly.

        Args:
            day: The current simulated day.

        Returns:
            Dictionary mapping memory_id → importance × decay_factor.
        """
        scores: dict[str, float] = {}
        for memory_id, event in self._memories.items():
            creation_day = self._creation_days.get(memory_id, 0)
            days_elapsed = max(0, day - creation_day)
            decay_factor = self._compute_decay_factor(days_elapsed)
            scores[memory_id] = event.importance * decay_factor
        return scores

    def get_creation_day(self, memory_id: str) -> int | None:
        """Return the creation day for a memory, or None if not found.

        Args:
            memory_id: The memory identifier.

        Returns:
            The simulated day the memory was injected, or None.
        """
        return self._creation_days.get(memory_id)

    def prune(self, memory_ids: list[str]) -> int:
        """Remove specified memories from the store.

        Args:
            memory_ids: List of memory IDs to prune.

        Returns:
            Number of memories actually removed.
        """
        removed = 0
        for memory_id in memory_ids:
            if memory_id in self._memories:
                event = self._memories[memory_id]
                uid = event.user_id or "__none__"
                if uid in self._user_memories:
                    self._user_memories[uid].discard(memory_id)
                del self._memories[memory_id]
                self._creation_days.pop(memory_id, None)
                removed += 1
        if removed:
            self._index_dirty = True
        return removed

    def remove(self, memory_id: str) -> None:
        """Remove a single memory by ID.

        Args:
            memory_id: The ID of the memory to remove.
        """
        if memory_id in self._memories:
            event = self._memories[memory_id]
            uid = event.user_id or "__none__"
            if uid in self._user_memories:
                self._user_memories[uid].discard(memory_id)
            del self._memories[memory_id]
            self._creation_days.pop(memory_id, None)
            self._index_dirty = True

    def count(self) -> int:
        """Return number of memories in the store.

        Returns:
            Current count.
        """
        return len(self._memories)

    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        self._creation_days.clear()

    # ------------------------------------------------------------------
    # Template method — subclasses override this
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_relevance_score(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        decay_factor: float,
    ) -> float:
        """Compute the relevance score for a single memory against a query.

        This is the only method subclasses must implement. The base class
        handles decay computation, user filtering, tier/confidence, and
        ReadQueryFilters. Subclasses focus on domain-specific scoring.

        Args:
            query: The full read query (includes query text, context, filters).
            event: The memory event to score.
            decay_factor: Pre-computed exponential decay factor for this event.

        Returns:
            A relevance score (higher is better). Will be clamped to [0, 1].
        """

    def _apply_module_weight(
        self,
        query: ReadQuery,
        event: MemoryEvent,
        strategy_score: float,
        decay_factor: float,
    ) -> float:
        """Apply module-specific weighting to a strategy retrieval score.

        Called when a retrieval strategy provides the base relevance score.
        Subclasses override to inject module-specific signals (importance,
        entity boost, task boost) so different memory types produce
        differentiated rankings even with the same retrieval strategy.

        Default: strategy_score × importance. This preserves the ranking
        signal where gold memories (importance=0.9) are boosted above
        noise (importance avg=0.53). Decay is applied post-ranking.

        Args:
            query: The read query.
            event: The memory event being scored.
            strategy_score: Raw score from the retrieval strategy.
            decay_factor: Pre-computed decay factor for this event.

        Returns:
            Ranking score incorporating module-specific weighting.
            Decay is NOT included here — it is applied post-ranking.
        """
        return strategy_score * event.importance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_candidates(self, query: ReadQuery) -> dict[str, MemoryEvent]:
        """Apply user-scoping and ReadQueryFilters to the memory store.

        USER ISOLATION GUARANTEE: The first filter applied is always user_id.
        No memory belonging to another user is ever included in the candidate set,
        which means neither the retrieval strategy path nor the fallback scoring
        path can surface cross-user data.

        Args:
            query: The read query with optional filters.

        Returns:
            Filtered dict of memory_id → MemoryEvent, strictly scoped to the
            requesting user.
        """
        # --- Step 1: O(1) user scope via pre-built index (security boundary) ---
        requesting_user = query.context.user_id
        user_mem_ids = self._user_memories.get(requesting_user, set())
        candidates: dict[str, MemoryEvent] = {
            mid: self._memories[mid] for mid in user_mem_ids if mid in self._memories
        }

        # --- Step 2: Apply optional type filter ---
        if query.filters.memory_types:
            allowed_types = set(query.filters.memory_types)
            candidates = {
                mid: event for mid, event in candidates.items() if event.type in allowed_types
            }

        # --- Step 3: Apply optional importance threshold ---
        if query.filters.min_importance > 0.0:
            threshold = query.filters.min_importance
            candidates = {
                mid: event for mid, event in candidates.items() if event.importance >= threshold
            }

        return candidates

    def _score_candidates(
        self,
        query: ReadQuery,
        candidates: dict[str, MemoryEvent],
        current_day: int,
        user_id: str,
    ) -> list[tuple[MemoryEvent, float, float]]:
        """Score filtered candidates with decay and user-scoping.

        Args:
            query: The read query.
            candidates: Pre-filtered memory candidates.
            current_day: Current simulated day.
            user_id: The user whose memories to consider.

        Returns:
            List of (event, combined_score, decay_factor) tuples.
        """
        results: list[tuple[MemoryEvent, float, float]] = []

        for memory_id, event in candidates.items():
            creation_day = self._creation_days.get(memory_id, 0)
            days_elapsed = max(0, current_day - creation_day)
            decay_factor = self._compute_decay_factor(days_elapsed)

            score = self._compute_relevance_score(query, event, decay_factor)
            results.append((event, score, decay_factor))

        return results

    @staticmethod
    def _compute_tier(decay_factor: float) -> MemoryTier:
        """Compute the memory tier based on decay factor.

        Uses normalized thresholds — single source of truth.

        Args:
            decay_factor: Current decay multiplier.

        Returns:
            MemoryTier based on decay thresholds.
        """
        if decay_factor > _HOT_THRESHOLD:
            return MemoryTier.HOT
        if decay_factor > _WARM_THRESHOLD:
            return MemoryTier.WARM
        return MemoryTier.COLD

    @staticmethod
    def _compute_confidence(score: float, decay_factor: float) -> float:
        """Compute retrieval confidence from score and decay factor.

        Confidence reflects how certain the system is that this result
        is relevant. Combines relevance score with temporal freshness.

        Args:
            score: The relevance score of the result.
            decay_factor: The current decay multiplier.

        Returns:
            A confidence value between 0.0 and 1.0.
        """
        raw = (score * 0.6) + (decay_factor * 0.4)
        return max(0.0, min(1.0, raw))
