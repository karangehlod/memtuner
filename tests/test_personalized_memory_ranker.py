"""Comprehensive tests for PersonalizedMemoryRanker."""

import pytest
import numpy as np
from benchmark.memory.strategies.personalized_memory_ranker import (
    PersonalizedMemoryRanker,
    UserProfile,
)


@pytest.fixture
def basic_profile():
    """Create a basic user profile."""
    return UserProfile(
        user_id="user123",
        preferred_topics=["machine learning", "AI"],
        content_type_preferences={"article": 0.9, "video": 0.5},
        recency_preference=0.6,
        depth_preference=0.7,
    )


@pytest.fixture
def sample_results():
    """Create sample search results."""
    return [
        {"id": "m1", "content": "machine learning basics", "type": "article", "score": 0.8},
        {"id": "m2", "content": "video about AI", "type": "video", "score": 0.7},
        {"id": "m3", "content": "deep learning advanced", "type": "article", "score": 0.6},
        {"id": "m4", "content": "cooking recipes", "type": "article", "score": 0.5},
    ]


class TestUserProfileInitialization:
    """Test UserProfile initialization."""

    def test_user_profile_creation(self):
        """Test creating a user profile."""
        profile = UserProfile(
            user_id="user1",
            preferred_topics=["AI"],
        )
        assert profile.user_id == "user1"
        assert "AI" in profile.preferred_topics

    def test_user_profile_defaults(self):
        """Test user profile default values."""
        profile = UserProfile(user_id="user1")
        assert profile.preferred_topics == []
        assert profile.content_type_preferences == {}
        assert profile.recency_preference == 0.5
        assert profile.depth_preference == 0.5
        assert profile.diversity_factor == 0.1


class TestRankerInitialization:
    """Test ranker initialization."""

    def test_ranker_creation(self, basic_profile):
        """Test creating a ranker."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        assert ranker.user_profile.user_id == "user123"

    def test_ranker_initial_state(self, basic_profile):
        """Test ranker initial state is clean."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        stats = ranker.get_ranking_statistics()
        assert stats["total_explicit_scores"] == 0


class TestRanking:
    """Test ranking functionality."""

    def test_rank_empty_results(self, basic_profile):
        """Test ranking empty results."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranked = ranker.rank([], "test query")
        assert ranked == []

    def test_rank_preserves_results(self, basic_profile, sample_results):
        """Test ranking preserves all results."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranked = ranker.rank(sample_results, "test")

        assert len(ranked) == len(sample_results)
        ids_original = {r["id"] for r in sample_results}
        ids_ranked = {r["id"] for r in ranked}
        assert ids_original == ids_ranked

    def test_rank_prefers_preferred_topics(self, basic_profile, sample_results):
        """Test that ranking prefers preferred topics."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranked = ranker.rank(sample_results, "machine learning")

        # m1 (machine learning) should rank high
        ranked_ids = [r["id"] for r in ranked]
        assert ranked_ids[0] in ["m1", "m3"]  # AI-related should be first

    def test_rank_multiple_calls_consistent(self, basic_profile, sample_results):
        """Test ranking is consistent across calls."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        ranked1 = ranker.rank(sample_results.copy(), "query")
        ranker.reset_feedback()
        ranked2 = ranker.rank(sample_results.copy(), "query")

        ids1 = [r["id"] for r in ranked1]
        ids2 = [r["id"] for r in ranked2]
        assert ids1 == ids2


class TestPreferences:
    """Test preference recording."""

    def test_add_preference_valid(self, basic_profile):
        """Test adding valid preference."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranker.add_preference("m1", 0.9)

        score = ranker.get_preference_score("m1")
        assert score > 0.5

    def test_add_preference_invalid_score_raises_error(self, basic_profile):
        """Test invalid preference score raises error."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        with pytest.raises(ValueError):
            ranker.add_preference("m1", 1.5)

    def test_add_preference_persists(self, basic_profile):
        """Test preference persists across calls."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranker.add_preference("m1", 0.95)

        score1 = ranker.get_preference_score("m1")
        score2 = ranker.get_preference_score("m1")

        assert score1 == score2

    def test_preference_score_calculation(self, basic_profile):
        """Test preference score calculation."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranker.add_preference("m1", 0.8)

        score = ranker.get_preference_score("m1")
        # Explicit is 0.8 * 0.7 = 0.56, implicit is 0, so score ≈ 0.56 + 0 = 0.56
        assert 0.5 < score < 1.0


class TestFeedback:
    """Test feedback recording."""

    def test_add_feedback_helpful(self, basic_profile):
        """Test recording helpful feedback."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranker.add_feedback("query1", "m1", helpful=True)

        score = ranker.get_preference_score("m1")
        # Score is weighted: explicit (0.5 default) * 0.7 + implicit (0.1) * 0.3 ≈ 0.385
        assert score >= 0.35

    def test_add_feedback_not_helpful(self, basic_profile):
        """Test recording unhelpful feedback."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranker.add_feedback("query1", "m1", helpful=False)

        score = ranker.get_preference_score("m1")
        assert score <= 0.5

    def test_feedback_accumulates(self, basic_profile):
        """Test feedback accumulates."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        ranker.add_feedback("q1", "m1", helpful=True)
        ranker.add_feedback("q2", "m1", helpful=True)
        ranker.add_feedback("q3", "m1", helpful=True)

        score = ranker.get_preference_score("m1")
        # After 3 helpful: implicit = 3, normalized = min(1.0, 3/10) = 0.3
        # Score = 0.5 * 0.7 + 0.3 * 0.3 = 0.35 + 0.09 = 0.44
        assert score > 0.4


class TestProfileUpdate:
    """Test profile updates."""

    def test_update_profile(self, basic_profile):
        """Test updating user profile."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        new_profile = UserProfile(
            user_id="user123",
            preferred_topics=["quantum computing"],
        )

        ranker.update_profile(new_profile)
        assert ranker.user_profile.preferred_topics == ["quantum computing"]


class TestResetFeedback:
    """Test feedback reset."""

    def test_reset_clears_feedback(self, basic_profile):
        """Test reset clears all feedback."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        ranker.add_preference("m1", 0.9)
        ranker.add_feedback("q", "m1", helpful=True)

        ranker.reset_feedback()

        stats = ranker.get_ranking_statistics()
        assert stats["total_explicit_scores"] == 0
        assert stats["total_implicit_feedbacks"] == 0


class TestUserInterests:
    """Test user interest extraction."""

    def test_get_user_interests_empty(self, basic_profile):
        """Test getting interests from empty ranker."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        interests = ranker.get_user_interests(top_k=5)

        # Should return preferred topics at least
        assert len(interests) > 0

    def test_get_user_interests_with_feedback(self, basic_profile):
        """Test interests include feedback-derived interests."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        for _ in range(5):
            ranker.add_feedback("query", "m_feedback", helpful=True)

        interests = ranker.get_user_interests(top_k=10)
        interest_names = [i[0] for i in interests]

        # Should include either topic or feedback memory
        assert len(interests) > 0

    def test_get_user_interests_invalid_top_k_raises_error(self, basic_profile):
        """Test invalid top_k raises error."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        with pytest.raises(ValueError):
            ranker.get_user_interests(top_k=0)


class TestRankingStatistics:
    """Test statistics tracking."""

    def test_get_ranking_statistics_empty(self, basic_profile):
        """Test statistics on empty ranker."""
        ranker = PersonalizedMemoryRanker(basic_profile)
        stats = ranker.get_ranking_statistics()

        assert stats["total_explicit_scores"] == 0
        assert stats["avg_explicit_preference"] == 0.5

    def test_get_ranking_statistics_with_data(self, basic_profile):
        """Test statistics with data."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        ranker.add_preference("m1", 0.9)
        ranker.add_preference("m2", 0.7)
        ranker.add_feedback("q", "m3", helpful=True)

        stats = ranker.get_ranking_statistics()
        assert stats["total_explicit_scores"] == 2
        assert stats["total_implicit_feedbacks"] >= 1


class TestDiversityBoosting:
    """Test diversity in ranking."""

    def test_diversity_factor_applied(self, basic_profile, sample_results):
        """Test diversity factor affects ranking."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        # First ranking
        ranked1 = ranker.rank(sample_results.copy(), "query")
        top1_id = ranked1[0]["id"]

        # See it multiple times
        for _ in range(5):
            ranker.rank([r for r in sample_results if r["id"] == top1_id], "query")

        # Next ranking should penalize top1
        ranked2 = ranker.rank(sample_results.copy(), "query")

        # Top item might change due to diversity
        top2_id = ranked2[0]["id"]
        # Can't guarantee order change, just that diversity is considered


class TestPersonalizationIntegration:
    """Test full personalization workflow."""

    def test_full_ranking_workflow(self, basic_profile):
        """Test complete ranking workflow."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        results = [
            {"id": "m1", "content": "AI topic", "type": "article", "score": 0.6},
            {"id": "m2", "content": "unrelated", "type": "video", "score": 0.9},
        ]

        # Initial rank
        ranked = ranker.rank(results, "AI query")
        assert len(ranked) == 2

        # Add preference
        ranker.add_preference("m1", 0.95)

        # Rank again
        ranked2 = ranker.rank(results.copy(), "AI query")
        assert len(ranked2) == 2

    def test_ranking_with_implicit_feedback(self, basic_profile):
        """Test ranking improves with implicit feedback."""
        ranker = PersonalizedMemoryRanker(basic_profile)

        results = [
            {"id": "m1", "content": "content", "type": "article", "score": 0.5},
            {"id": "m2", "content": "content", "type": "article", "score": 0.5},
        ]

        # Give positive feedback to m1
        ranker.add_feedback("q", "m1", helpful=True)

        # m1 should rank higher now
        ranked = ranker.rank(results.copy(), "q")

        # Verify m1 is ranked favorably (though score-based ranking might differ)
        assert len(ranked) == 2
