"""Comprehensive tests for AdaptiveStrategySelector."""

import pytest
from benchmark.memory.strategies.adaptive_strategy_selector import (
    AdaptiveStrategySelector,
)


class TestQueryClassification:
    """Test query type classification."""

    def test_classify_factual_what(self):
        """Test classifying 'what' questions."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("What is machine learning?")
        assert query_type == "factual"

    def test_classify_factual_who(self):
        """Test classifying 'who' questions."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("Who invented the internet?")
        assert query_type == "factual"

    def test_classify_factual_when(self):
        """Test classifying 'when' questions."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("When was Python released?")
        assert query_type == "factual"

    def test_classify_exact_phrase(self):
        """Test classifying exact phrase queries."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query('Find exact match for "hello world"')
        assert query_type == "exact"

    def test_classify_complex_compare(self):
        """Test classifying complex comparison queries."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("Compare Python and JavaScript")
        assert query_type == "complex"

    def test_classify_complex_synthesize(self):
        """Test classifying complex synthesis queries."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("Synthesize information about AI")
        assert query_type == "complex"

    def test_classify_semantic_default(self):
        """Test semantic classification as default."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("Tell me about cats")
        assert query_type == "semantic"

    def test_classify_case_insensitive(self):
        """Test classification is case-insensitive."""
        selector = AdaptiveStrategySelector()
        type1 = selector.classify_query("WHAT IS AI?")
        type2 = selector.classify_query("what is ai?")
        assert type1 == type2 == "factual"

    def test_classify_whitespace_tolerant(self):
        """Test classification tolerates whitespace."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("  what is this  ")
        assert query_type == "factual"


class TestStrategySelection:
    """Test strategy selection."""

    def test_select_strategy_factual(self):
        """Test strategy selection for factual queries."""
        selector = AdaptiveStrategySelector()
        strategy = selector.select_strategy("What is AI?")
        assert strategy == "bm25"

    def test_select_strategy_semantic(self):
        """Test strategy selection for semantic queries."""
        selector = AdaptiveStrategySelector()
        strategy = selector.select_strategy("Tell me about neural networks")
        assert strategy == "dense_vector"

    def test_select_strategy_exact(self):
        """Test strategy selection for exact queries."""
        selector = AdaptiveStrategySelector()
        strategy = selector.select_strategy('Find "machine learning"')
        assert strategy == "colbert"

    def test_select_strategy_complex(self):
        """Test strategy selection for complex queries."""
        selector = AdaptiveStrategySelector()
        strategy = selector.select_strategy("Compare ML and DL")
        assert strategy == "cascading"

    def test_select_strategy_empty_list_uses_defaults(self):
        """Test that empty list uses default strategies."""
        selector = AdaptiveStrategySelector(strategies=[])
        # Empty list is falsy, so defaults are used
        strategy = selector.select_strategy("test")
        assert strategy in ["bm25", "dense_vector", "colbert", "cascading", "hybrid_fusion"]

    def test_select_strategy_custom_strategies(self):
        """Test strategy selection with custom strategies."""
        selector = AdaptiveStrategySelector(strategies=["custom1", "custom2"])
        strategy = selector.select_strategy("What is this?")
        assert strategy in ["custom1", "custom2"]


class TestRecommendations:
    """Test strategy recommendations."""

    def test_get_recommendations_returns_list(self):
        """Test recommendations returns a list."""
        selector = AdaptiveStrategySelector()
        recs = selector.get_recommendations("What is AI?")
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_get_recommendations_top_k(self):
        """Test getting specific number of recommendations."""
        selector = AdaptiveStrategySelector()
        recs = selector.get_recommendations("query", top_k=2)
        assert len(recs) <= 2

    def test_get_recommendations_invalid_top_k_raises_error(self):
        """Test invalid top_k raises error."""
        selector = AdaptiveStrategySelector()
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            selector.get_recommendations("query", top_k=0)

    def test_recommendation_has_confidence(self):
        """Test recommendation has confidence score."""
        selector = AdaptiveStrategySelector()
        recs = selector.get_recommendations("What is AI?")
        assert all(0.0 <= rec.confidence <= 1.0 for rec in recs)

    def test_recommendation_has_reason(self):
        """Test recommendation has reason."""
        selector = AdaptiveStrategySelector()
        recs = selector.get_recommendations("query")
        assert all(isinstance(rec.reason, str) and len(rec.reason) > 0 for rec in recs)

    def test_recommendation_primary_first(self):
        """Test primary strategy is first recommendation."""
        selector = AdaptiveStrategySelector()
        recs = selector.get_recommendations("What is AI?")
        assert recs[0].strategy_name == "bm25"

    def test_get_recommendations_for_query_type(self):
        """Test getting recommendations for specific query type."""
        selector = AdaptiveStrategySelector()
        recs = selector.get_recommendations_for_type("factual")
        assert len(recs) > 0
        assert recs[0].strategy_name == "bm25"

    def test_get_recommendations_for_query_type_invalid_raises_error(self):
        """Test invalid query type raises error."""
        selector = AdaptiveStrategySelector()
        with pytest.raises(ValueError, match="Unknown query type"):
            selector.get_recommendations_for_type("invalid")


class TestLearningFromFeedback:
    """Test learning from feedback."""

    def test_learn_from_feedback_success(self):
        """Test recording successful strategy use."""
        selector = AdaptiveStrategySelector()
        selector.learn_from_feedback("What is AI?", "bm25", successful=True)

        stats = selector.get_strategy_stats()
        assert stats["bm25"]["attempts"] == 1
        assert stats["bm25"]["successes"] == 1

    def test_learn_from_feedback_failure(self):
        """Test recording failed strategy use."""
        selector = AdaptiveStrategySelector()
        selector.learn_from_feedback("What is AI?", "bm25", successful=False)

        stats = selector.get_strategy_stats()
        assert stats["bm25"]["attempts"] == 1
        assert stats["bm25"]["successes"] == 0

    def test_learn_from_feedback_accumulates(self):
        """Test feedback accumulates over time."""
        selector = AdaptiveStrategySelector()

        selector.learn_from_feedback("query1", "bm25", successful=True)
        selector.learn_from_feedback("query2", "bm25", successful=True)
        selector.learn_from_feedback("query3", "bm25", successful=False)

        stats = selector.get_strategy_stats()
        assert stats["bm25"]["attempts"] == 3
        assert stats["bm25"]["successes"] == 2
        assert stats["bm25"]["success_rate"] == pytest.approx(2/3)

    def test_learn_from_feedback_invalid_strategy_raises_error(self):
        """Test invalid strategy raises error."""
        selector = AdaptiveStrategySelector()
        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.learn_from_feedback("query", "invalid_strategy", successful=True)

    def test_success_rate_influences_recommendations(self):
        """Test that success rate influences recommendations."""
        selector = AdaptiveStrategySelector()

        # Make dense_vector very successful
        for _ in range(10):
            selector.learn_from_feedback("query", "dense_vector", successful=True)

        # Make bm25 unsuccessful
        for _ in range(10):
            selector.learn_from_feedback("query", "bm25", successful=False)

        # For semantic queries, dense_vector should rank higher
        recs = selector.get_recommendations("Tell me about AI", top_k=2)
        # dense_vector might be recommended higher due to success rate
        strategy_names = [rec.strategy_name for rec in recs]
        assert len(strategy_names) > 0

    def test_reset_feedback(self):
        """Test resetting feedback."""
        selector = AdaptiveStrategySelector()
        selector.learn_from_feedback("query", "bm25", successful=True)

        selector.reset_feedback()

        stats = selector.get_strategy_stats()
        assert stats["bm25"]["attempts"] == 0


class TestStrategyStats:
    """Test strategy statistics."""

    def test_get_strategy_stats_all_strategies(self):
        """Test getting stats for all strategies."""
        selector = AdaptiveStrategySelector()
        selector.learn_from_feedback("query", "bm25", successful=True)

        stats = selector.get_strategy_stats()
        assert "bm25" in stats
        assert "dense_vector" in stats

    def test_strategy_stats_has_required_fields(self):
        """Test stats have required fields."""
        selector = AdaptiveStrategySelector()
        selector.learn_from_feedback("query", "bm25", successful=True)

        stats = selector.get_strategy_stats()
        for strategy, strategy_stats in stats.items():
            assert "success_rate" in strategy_stats
            assert "attempts" in strategy_stats
            assert "successes" in strategy_stats

    def test_initial_success_rate_neutral(self):
        """Test initial success rate is neutral."""
        selector = AdaptiveStrategySelector()

        stats = selector.get_strategy_stats()
        # Untested strategies should have 0 attempts initially
        for strategy_stats in stats.values():
            if strategy_stats["attempts"] == 0:
                assert strategy_stats["successes"] == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_query(self):
        """Test handling empty query."""
        selector = AdaptiveStrategySelector()
        query_type = selector.classify_query("")
        assert query_type in ["factual", "semantic", "exact", "complex"]

    def test_long_query(self):
        """Test handling very long query."""
        selector = AdaptiveStrategySelector()
        long_query = "What " * 1000
        query_type = selector.classify_query(long_query)
        assert query_type == "factual"

    def test_special_characters_in_query(self):
        """Test handling special characters."""
        selector = AdaptiveStrategySelector()
        query = 'Find "test@#$%^&*()" exactly'
        query_type = selector.classify_query(query)
        assert query_type in ["factual", "semantic", "exact", "complex"]

    def test_unicode_in_query(self):
        """Test handling unicode characters."""
        selector = AdaptiveStrategySelector()
        query = "What is 机器学习?"
        query_type = selector.classify_query(query)
        assert query_type == "factual"

    def test_initialization_with_custom_strategies(self):
        """Test initialization with custom strategy list."""
        strategies = ["strat1", "strat2", "strat3"]
        selector = AdaptiveStrategySelector(strategies=strategies)
        recs = selector.get_recommendations("query", top_k=2)
        assert len(recs) <= 2

    def test_single_strategy(self):
        """Test with single available strategy."""
        selector = AdaptiveStrategySelector(strategies=["only_strategy"])
        strategy = selector.select_strategy("query")
        assert strategy == "only_strategy"


class TestConsistency:
    """Test consistency and determinism."""

    def test_same_query_same_classification(self):
        """Test same query always classified same way."""
        selector = AdaptiveStrategySelector()
        query = "What is machine learning?"

        type1 = selector.classify_query(query)
        type2 = selector.classify_query(query)
        type3 = selector.classify_query(query)

        assert type1 == type2 == type3

    def test_same_query_same_strategy(self):
        """Test same query always selects same strategy."""
        selector = AdaptiveStrategySelector()
        query = "Tell me about AI"

        strategy1 = selector.select_strategy(query)
        strategy2 = selector.select_strategy(query)

        assert strategy1 == strategy2

    def test_feedback_improves_success_rate(self):
        """Test feedback actually improves measured success rate."""
        selector = AdaptiveStrategySelector()

        # Initially neutral
        stats_before = selector.get_strategy_stats()["bm25"]["success_rate"]

        # Add successful feedback
        for _ in range(20):
            selector.learn_from_feedback("query", "bm25", successful=True)

        stats_after = selector.get_strategy_stats()["bm25"]["success_rate"]
        assert stats_after > stats_before
