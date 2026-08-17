"""Personalized memory ranking based on user preferences."""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


@dataclass
class UserProfile:
    """User preference profile."""
    user_id: str
    preferred_topics: list[str] = field(default_factory=list)
    content_type_preferences: dict[str, float] = field(default_factory=dict)
    recency_preference: float = 0.5  # 0=prefer old, 1=prefer new
    depth_preference: float = 0.5  # 0=prefer summary, 1=prefer detailed
    diversity_factor: float = 0.1  # How much to boost novel items


@dataclass
class RankingResult:
    """Result of ranking operation."""
    memory_id: str
    base_score: float
    personalization_boost: float
    final_score: float
    rank: int


class PersonalizedMemoryRanker:
    """Rank memory results based on user preferences."""

    def __init__(self, user_profile: UserProfile):
        """Initialize ranker with user profile.

        Args:
            user_profile: User preference configuration
        """
        self.user_profile = user_profile

        # Feedback history: memory_id → (explicit_score, implicit_count)
        self._explicit_feedback: dict[str, float] = defaultdict(lambda: 0.5)
        self._implicit_feedback: dict[str, int] = defaultdict(int)

        # Seen memories for diversity tracking
        self._recently_seen: dict[str, int] = defaultdict(int)

    def rank(
        self,
        results: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Rank results based on user preferences.

        Args:
            results: List of search results (base ranked)
            query: Original query

        Returns:
            Re-ranked results list (highest personalized score first)

        Raises:
            ValueError: If results invalid
        """
        if not results:
            return []

        # Compute personalized scores
        scores = {}
        for i, result in enumerate(results):
            memory_id = result.get("id", f"result_{i}")
            base_score = result.get("score", 0.5)

            # Compute personalization boost
            boost = self._compute_personalization_boost(result, query)

            # Final score
            final_score = (base_score * 0.7) + (boost * 0.3)
            scores[memory_id] = (final_score, i)

        # Sort by personalized score (descending)
        sorted_results = sorted(
            results,
            key=lambda r: scores.get(r.get("id"), (0.0, 0))[0],
            reverse=True,
        )

        # Update seen memories for diversity
        for result in sorted_results:
            mem_id = result.get("id", "unknown")
            self._recently_seen[mem_id] += 1

        return sorted_results

    def add_preference(self, memory_id: str, score: float) -> None:
        """Record explicit user preference.

        Args:
            memory_id: Memory identifier
            score: Preference score (0-1, 1=high preference)

        Raises:
            ValueError: If score invalid
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score must be [0,1], got {score}")

        self._explicit_feedback[memory_id] = score

    def add_feedback(
        self,
        query: str,
        memory_id: str,
        helpful: bool,
    ) -> None:
        """Record implicit feedback from user interaction.

        Args:
            query: Query that returned this result
            memory_id: Memory identifier
            helpful: Whether user found it helpful
        """
        if helpful:
            self._implicit_feedback[memory_id] += 1
        else:
            self._implicit_feedback[memory_id] = max(0, self._implicit_feedback[memory_id] - 1)

    def update_profile(self, new_profile: UserProfile) -> None:
        """Update user profile.

        Args:
            new_profile: Updated profile
        """
        self.user_profile = new_profile

    def get_preference_score(self, memory_id: str) -> float:
        """Get combined preference score for memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Preference score (0-1)
        """
        explicit = self._explicit_feedback.get(memory_id, 0.5)
        implicit = min(1.0, self._implicit_feedback.get(memory_id, 0) / 10.0)

        # Weight explicit higher
        combined = (explicit * 0.7) + (implicit * 0.3)
        return min(1.0, max(0.0, combined))

    def get_user_interests(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Get user's top interests based on preferences.

        Args:
            top_k: Number of interests to return

        Returns:
            List of (topic, score) tuples

        Raises:
            ValueError: If top_k invalid
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        interests = []

        # Add preferred topics
        for topic in self.user_profile.preferred_topics[:top_k]:
            interests.append((topic, 1.0))

        # Add frequently helpful memories as derived interests
        sorted_implicit = sorted(
            self._implicit_feedback.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for mem_id, count in sorted_implicit[:top_k - len(interests)]:
            score = min(1.0, count / 10.0)
            interests.append((mem_id, score))

        return interests[:top_k]

    def reset_feedback(self) -> None:
        """Reset all feedback history."""
        self._explicit_feedback.clear()
        self._implicit_feedback.clear()
        self._recently_seen.clear()

    def get_ranking_statistics(self) -> dict[str, Any]:
        """Get statistics about ranking patterns.

        Returns:
            Dict with ranking metrics
        """
        if not self._explicit_feedback:
            avg_explicit = 0.5
        else:
            avg_explicit = np.mean(list(self._explicit_feedback.values()))

        if not self._implicit_feedback:
            avg_implicit = 0.0
        else:
            avg_implicit = np.mean(list(self._implicit_feedback.values()))

        return {
            "avg_explicit_preference": avg_explicit,
            "avg_implicit_feedback": avg_implicit,
            "total_explicit_scores": len(self._explicit_feedback),
            "total_implicit_feedbacks": sum(self._implicit_feedback.values()),
            "unique_memories_ranked": len(self._recently_seen),
        }

    # Private helper methods

    def _compute_personalization_boost(
        self,
        result: dict[str, Any],
        query: str,
    ) -> float:
        """Compute personalization boost for result."""
        boost = 0.0

        # Topic preference boost
        topics_boost = self._compute_topic_boost(result)
        boost += topics_boost * 0.3

        # Content type preference boost
        type_boost = self._compute_content_type_boost(result)
        boost += type_boost * 0.2

        # Preference history boost
        mem_id = result.get("id", "unknown")
        pref_boost = self.get_preference_score(mem_id)
        boost += pref_boost * 0.3

        # Diversity boost
        diversity_boost = self._compute_diversity_boost(mem_id)
        boost += diversity_boost * 0.2

        return min(1.0, max(0.0, boost))

    def _compute_topic_boost(self, result: dict[str, Any]) -> float:
        """Compute boost based on topic preferences."""
        if not self.user_profile.preferred_topics:
            return 0.0

        content = str(result.get("content", "")).lower()

        for topic in self.user_profile.preferred_topics:
            if topic.lower() in content:
                return 1.0

        return 0.0

    def _compute_content_type_boost(self, result: dict[str, Any]) -> float:
        """Compute boost based on content type preference."""
        if not self.user_profile.content_type_preferences:
            return 0.5

        content_type = result.get("type", "unknown")
        pref = self.user_profile.content_type_preferences.get(content_type, 0.5)

        return pref

    def _compute_diversity_boost(self, memory_id: str) -> float:
        """Compute boost for novel/diverse items."""
        seen_count = self._recently_seen.get(memory_id, 0)

        # Penalize frequently seen items
        diversity_penalty = min(1.0, seen_count * 0.1)
        boost = max(0.0, 1.0 - diversity_penalty)

        # Apply diversity factor
        return boost * self.user_profile.diversity_factor

    def _extract_topics(self, content: str) -> list[str]:
        """Extract topics from content."""
        # Simple extraction: split by common delimiters
        words = content.lower().split()
        return [w for w in words if len(w) > 3]
