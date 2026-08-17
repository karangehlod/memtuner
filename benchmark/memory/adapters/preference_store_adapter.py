"""Adapter for Preference Store - user preference memory."""

import json
import time
from typing import Any

from benchmark.memory.adapters.memory_adapter import MemoryAdapter, MemoryMetrics, MemoryRegistry
from benchmark.memory.adapters._sys_metrics import percentile as _pct, peak_rss_mb as _rss, cpu_percent_snapshot as _cpu


class PreferenceStoreAdapter(MemoryAdapter):
    """Benchmarks preference store for user preference and prediction.

    Preference memory stores user preferences, interests, and patterns.
    Queries predict user preferences and rank items by preference fit.
    """

    name = "preference_store"

    def __init__(self):
        self.user_preferences: dict[str, dict[str, float]] = {}
        self.preference_history: dict[str, list[dict[str, Any]]] = {}
        self.predictions: list[dict[str, Any]] = []
        self._per_query_results: list[list[dict[str, Any]]] = []
        self.write_times: list[float] = []
        self.query_times: list[float] = []
        self.config: dict[str, Any] = {}
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time: float = 0.0

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize preference store with configuration."""
        self.config = config
        self.user_preferences = {}
        self.preference_history = {}
        self.predictions = []
        self._per_query_results = []
        self.write_times = []
        self.query_times = []
        self.num_writes = 0
        self.num_queries = 0
        self.num_failures = 0
        self.start_time = time.time()

    def write_memory(self, memory: dict[str, Any]) -> None:
        """Write user preference to store.

        Stores preference signals and updates preference profile.
        """
        try:
            start = time.time()

            memory_id = memory.get("id", "")
            user_id = memory.get("user_id", "unknown_user")
            content = memory.get("content", "")
            importance = memory.get("importance", 0.5)

            # Initialize user profile if needed
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
                self.preference_history[user_id] = []

            # Extract preference signals from content
            preferences = self._extract_preferences(content)

            # Update user preference profile (exponential moving average)
            alpha = 0.3  # Learning rate
            for pref_dim, pref_value in preferences.items():
                current_value = self.user_preferences[user_id].get(pref_dim, 0.5)
                updated_value = (1 - alpha) * current_value + alpha * pref_value
                self.user_preferences[user_id][pref_dim] = updated_value

            # Track history for pattern detection
            self.preference_history[user_id].append({
                "memory_id": memory_id,
                "preferences": preferences,
                "timestamp": time.time(),
                "importance": importance,
            })

            elapsed = time.time() - start
            self.write_times.append(elapsed)
            self.num_writes += 1

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to write preference memory: {e}")

    def query_memories(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query preference store for preference prediction.

        Predicts what the user would prefer based on past preferences.
        """
        try:
            start = time.time()

            # Try to find a user with preferences
            user_id = None
            if self.user_preferences:
                user_id = next(iter(self.user_preferences.keys()))
            else:
                user_id = "unknown_user"

            # Get user preference profile
            user_prefs = self.user_preferences.get(user_id, {})

            if not user_prefs:
                # No preferences learned yet, return default
                elapsed = time.time() - start
                self.query_times.append(elapsed)
                self.num_queries += 1
                return []

            # Extract query preferences
            query_prefs = self._extract_preferences(query)

            # Score prediction confidence
            scores: dict[str, tuple[float, str]] = {}

            for pref_dim, pref_value in query_prefs.items():
                if pref_dim in user_prefs:
                    user_value = user_prefs[pref_dim]
                    # Similarity score: 1.0 if match, lower if different
                    similarity = 1.0 - abs(user_value - pref_value)
                    scores[pref_dim] = (similarity, query)

            # Sort by prediction score
            sorted_scores = sorted(
                scores.items(),
                key=lambda x: x[1][0],
                reverse=True
            )[:top_k]

            results = [
                {
                    "preference_dim": dim,
                    "score": score,
                    "user_value": user_prefs.get(dim, 0.5),
                    "query_value": query_prefs.get(dim, 0.5),
                }
                for dim, (score, _) in sorted_scores
            ]

            elapsed = time.time() - start
            self.query_times.append(elapsed)
            self.num_queries += 1
            self.predictions.extend(results)
            self._per_query_results.append(results)

            return results

        except Exception as e:
            self.num_failures += 1
            raise RuntimeError(f"Failed to query preference memories: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Compute performance metrics for preference store."""
        try:
            elapsed_seconds = time.time() - self.start_time

            # Per-query MRR, Recall@K, Precision@K — computed correctly: one
            # reciprocal rank per query at the rank of the first relevant result
            # within that query's ranked list, then macro-averaged across queries.
            # Score-based relevance (threshold=0.5) is used since gold labels are
            # not available at this layer; Pipeline 1 (ScenarioRunner) provides
            # gold-grounded metrics for the leaderboard.
            from benchmark.retrieval.metrics_utils import compute_metric_summary as _cms
            _all_q = [
                [{"doc_id": r.get("preference_dim", r.get("memory_id", "")), "score": r.get("score", 0.0)} for r in qr]
                for qr in self._per_query_results
            ]
            _ms = _cms(_all_q, use_score_estimation=True)
            recall_at_1   = _ms.get("recall_at_1", 0.0)
            recall_at_5   = _ms.get("recall_at_5", 0.0)
            recall_at_10  = _ms["recall_at_10"]
            recall_at_100 = _ms["recall_at_100"]
            mrr           = _ms["mrr"]
            ndcg          = _ms["ndcg"]

            # Efficiency
            avg_write_latency = sum(self.write_times) / len(self.write_times) if self.write_times else 0.0
            avg_query_latency = sum(self.query_times) / len(self.query_times) if self.query_times else 0.0

            # Storage: user profiles + history
            storage_bytes = sum(
                len(json.dumps(prefs).encode())
                for prefs in self.user_preferences.values()
            ) + sum(
                len(json.dumps(h).encode())
                for history in self.preference_history.values()
                for h in history
            )

            # Reliability
            success_rate = 1.0 - (self.num_failures / max(1, self.num_writes + self.num_queries))

            return MemoryMetrics(
                recall_at_1=min(1.0, recall_at_1),
                recall_at_5=min(1.0, recall_at_5),
                recall_at_10=min(1.0, recall_at_10),
                recall_at_100=min(1.0, recall_at_100),
                mrr=min(1.0, mrr),
                ndcg=min(1.0, ndcg),
                write_latency_ms=avg_write_latency * 1000,
                query_latency_ms=avg_query_latency * 1000,
                storage_bytes=float(storage_bytes),
                success_rate=success_rate,
                error_count=self.num_failures,
                dataset_name=self.config.get("dataset_name", "unknown"),
                num_memories=self.num_writes,
                num_queries=self.num_queries,
                elapsed_seconds=elapsed_seconds,
                query_latency_p50_ms=_pct(self.query_times, 50) * 1000,
                query_latency_p95_ms=_pct(self.query_times, 95) * 1000,
                index_build_ms=sum(self.write_times) * 1000,
                peak_rss_mb=_rss(),
                cpu_percent=_cpu(),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to compute preference store metrics: {e}")

    def teardown(self) -> None:
        """Clean up resources."""
        self.user_preferences.clear()
        self.preference_history.clear()
        self.predictions.clear()
        self._per_query_results.clear()
        self.write_times.clear()
        self.query_times.clear()

    @staticmethod
    def _extract_preferences(text: str) -> dict[str, float]:
        """Extract preference dimensions from text.

        Returns dict mapping preference dimensions to scores (0-1).
        """
        if not text:
            return {}

        text_lower = text.lower()
        preferences = {}

        # Simple preference signals
        preference_keywords = {
            "positive_sentiment": ["good", "great", "love", "like", "best", "excellent", "awesome"],
            "negative_sentiment": ["bad", "hate", "dislike", "worst", "terrible", "awful"],
            "technical": ["algorithm", "efficient", "performance", "speed", "optimization"],
            "social": ["friend", "people", "social", "community", "group", "team"],
            "creative": ["art", "creative", "design", "music", "culture", "style"],
        }

        for dimension, keywords in preference_keywords.items():
            # Count keyword occurrences
            count = sum(text_lower.count(kw) for kw in keywords)
            # Normalize to 0-1 score
            score = min(count / 3.0, 1.0)
            if count > 0:
                preferences[dimension] = score

        return preferences or {"neutral": 0.5}


# Auto-register on import
MemoryRegistry.register("preference_store", PreferenceStoreAdapter)
