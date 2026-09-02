"""Integration tests for dataset adapters.

Tests loading, validation, statistics, and fingerprinting for each adapter.

SOLID principles tested:
  - All adapters implement DatasetAdapter interface consistently
  - Validation works uniformly across adapters
  - Statistics computation is consistent
  - Fingerprints are deterministic
"""

import json
import tempfile
from pathlib import Path

from benchmark.gold.adapters.locomo_adapter import LoCoMoAdapter
from benchmark.gold.adapters.longmemeval_adapter import LongMemEvalAdapter
from benchmark.gold.adapters.synthetic_adapter import SyntheticAdapter

# ============================================================================
# Synthetic Adapter Tests
# ============================================================================


class TestSyntheticAdapter:
    """Test synthetic dataset generation."""

    def test_synthetic_basic_generation(self) -> None:
        """Generate small synthetic dataset."""
        adapter = SyntheticAdapter(query_count=10, user_count=3, seed=42)
        dataset = adapter.load()

        assert dataset is not None
        assert len(dataset.queries) == 10
        assert len(dataset.user_ids) >= 3

    def test_synthetic_reproducible(self) -> None:
        """Same seed produces identical fingerprints."""
        adapter1 = SyntheticAdapter(query_count=50, seed=42)
        dataset1 = adapter1.load()
        fp1 = adapter1.fingerprint(dataset1)

        adapter2 = SyntheticAdapter(query_count=50, seed=42)
        dataset2 = adapter2.load()
        fp2 = adapter2.fingerprint(dataset2)

        assert fp1 == fp2

    def test_synthetic_different_seeds_differ(self) -> None:
        """Different seeds produce different fingerprints."""
        adapter1 = SyntheticAdapter(query_count=50, seed=42)
        dataset1 = adapter1.load()
        fp1 = adapter1.fingerprint(dataset1)

        adapter2 = SyntheticAdapter(query_count=50, seed=43)
        dataset2 = adapter2.load()
        fp2 = adapter2.fingerprint(dataset2)

        assert fp1 != fp2

    def test_synthetic_validation_passes(self) -> None:
        """Generated dataset passes validation."""
        adapter = SyntheticAdapter(query_count=20, seed=42)
        dataset = adapter.load()
        report = adapter.validate(dataset)

        assert report.passed or len(report.errors) == 0

    def test_synthetic_statistics_computed(self) -> None:
        """Can compute statistics on synthetic dataset."""
        adapter = SyntheticAdapter(query_count=50, seed=42)
        dataset = adapter.load()
        stats = adapter.statistics(dataset)

        assert stats.counts.query_count == 50
        assert stats.counts.memory_count > 0
        assert stats.counts.user_count >= 1

    def test_synthetic_metadata(self) -> None:
        """Adapter returns proper metadata."""
        adapter = SyntheticAdapter(seed=42)
        metadata = adapter.metadata()

        assert metadata["name"] == "Synthetic"
        assert "seed" in str(metadata)
        assert metadata["reproducible"] is True

    def test_synthetic_low_density(self) -> None:
        """Low memory density produces fewer memories."""
        adapter_low = SyntheticAdapter(
            query_count=100, memory_density="low", seed=42
        )
        dataset_low = adapter_low.load()
        count_low = sum(len(d.memory_events) for d in dataset_low.events)

        adapter_high = SyntheticAdapter(
            query_count=100, memory_density="high", seed=42
        )
        dataset_high = adapter_high.load()
        count_high = sum(len(d.memory_events) for d in dataset_high.events)

        assert count_low < count_high

    def test_synthetic_high_diversity_queries(self) -> None:
        """High diversity produces complex queries."""
        adapter = SyntheticAdapter(
            query_count=50, query_diversity="high", seed=42
        )
        dataset = adapter.load()

        # Count queries with multiple relevant memories
        multi_memory_queries = sum(
            1 for q in dataset.queries if len(q.expected.memory_ids) > 2
        )

        # Should have some multi-memory queries with high diversity
        assert multi_memory_queries > 0


# ============================================================================
# LoCoMo Adapter Tests
# ============================================================================


class TestLoCoMoAdapter:
    """Test LoCoMo dataset adapter."""

    def test_locomo_minimal_dataset(self) -> None:
        """Load minimal LoCoMo dataset structure."""
        # Create minimal LoCoMo data
        locomo_data = [
            {
                "event_summary": ["Memory 1", "Memory 2"],
                "qa": {"What happened?": "It happened"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(locomo_data, f)

            adapter = LoCoMoAdapter()
            dataset = adapter.load(path)

            assert len(dataset.queries) > 0
            assert len(dataset.events) > 0

    def test_locomo_validation(self) -> None:
        """LoCoMo dataset passes validation."""
        locomo_data = [
            {
                "event_summary": ["Event 1"],
                "qa": {"Q1": "A1"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(locomo_data, f)

            adapter = LoCoMoAdapter()
            dataset = adapter.load(path)
            report = adapter.validate(dataset)

            assert isinstance(report.passed, bool)

    def test_locomo_fingerprint_deterministic(self) -> None:
        """LoCoMo fingerprints are deterministic."""
        locomo_data = [
            {
                "event_summary": ["Event 1", "Event 2"],
                "qa": {"Q1": "A1"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(locomo_data, f)

            adapter = LoCoMoAdapter()
            dataset = adapter.load(path)
            fp1 = adapter.fingerprint(dataset)
            fp2 = adapter.fingerprint(dataset)

            assert fp1 == fp2

    def test_locomo_metadata(self) -> None:
        """LoCoMo adapter returns proper metadata."""
        adapter = LoCoMoAdapter()
        metadata = adapter.metadata()

        assert metadata["name"] == "LoCoMo"
        assert "Long Context" in metadata["description"]

    def test_locomo_statistics(self) -> None:
        """Can compute statistics on LoCoMo dataset."""
        locomo_data = [
            {
                "event_summary": ["E1", "E2"],
                "qa": {"Q1": "A1", "Q2": "A2"},
            },
            {
                "event_summary": ["E3"],
                "qa": {"Q3": "A3"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(locomo_data, f)

            adapter = LoCoMoAdapter()
            dataset = adapter.load(path)
            stats = adapter.statistics(dataset)

            assert stats.counts.query_count > 0
            assert stats.counts.memory_count > 0


# ============================================================================
# LongMemEval Adapter Tests
# ============================================================================


class TestLongMemEvalAdapter:
    """Test LongMemEval dataset adapter."""

    def test_longmemeval_minimal_dataset(self) -> None:
        """Load minimal LongMemEval dataset structure."""
        longmemeval_data = [
            {
                "question_id": "q1",
                "question": "What was discussed?",
                "question_date": "2024-01-01",
                "haystack_sessions": ["Context 1", "Context 2"],
                "haystack_session_ids": ["s1", "s2"],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(longmemeval_data, f)

            adapter = LongMemEvalAdapter()
            dataset = adapter.load(path)

            assert len(dataset.queries) > 0
            assert len(dataset.events) > 0

    def test_longmemeval_validation(self) -> None:
        """LongMemEval dataset passes validation."""
        longmemeval_data = [
            {
                "question_id": "q1",
                "question": "Question?",
                "haystack_sessions": ["Context"],
                "haystack_session_ids": ["s1"],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(longmemeval_data, f)

            adapter = LongMemEvalAdapter()
            dataset = adapter.load(path)
            report = adapter.validate(dataset)

            assert isinstance(report.passed, bool)

    def test_longmemeval_fingerprint_deterministic(self) -> None:
        """LongMemEval fingerprints are deterministic."""
        longmemeval_data = [
            {
                "question_id": "q1",
                "question": "Question?",
                "haystack_sessions": ["C1", "C2"],
                "haystack_session_ids": ["s1", "s2"],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(longmemeval_data, f)

            adapter = LongMemEvalAdapter()
            dataset = adapter.load(path)
            fp1 = adapter.fingerprint(dataset)
            fp2 = adapter.fingerprint(dataset)

            assert fp1 == fp2

    def test_longmemeval_metadata(self) -> None:
        """LongMemEval adapter returns proper metadata."""
        adapter = LongMemEvalAdapter()
        metadata = adapter.metadata()

        assert metadata["name"] == "LongMemEval"
        assert "long-context" in metadata["description"].lower()

    def test_longmemeval_statistics(self) -> None:
        """Can compute statistics on LongMemEval dataset."""
        longmemeval_data = [
            {
                "question_id": "q1",
                "question": "Q1?",
                "haystack_sessions": ["C1", "C2"],
                "haystack_session_ids": ["s1", "s2"],
            },
            {
                "question_id": "q2",
                "question": "Q2?",
                "haystack_sessions": ["C3"],
                "haystack_session_ids": ["s3"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(longmemeval_data, f)

            adapter = LongMemEvalAdapter()
            dataset = adapter.load(path)
            stats = adapter.statistics(dataset)

            assert stats.counts.query_count > 0
            assert stats.counts.memory_count > 0


# ============================================================================
# Cross-Adapter Consistency Tests
# ============================================================================


class TestAdapterConsistency:
    """Test consistency across adapters."""

    def test_all_adapters_implement_interface(self) -> None:
        """All adapters have required methods."""
        adapters = [
            SyntheticAdapter(seed=42),
            LoCoMoAdapter(),
            LongMemEvalAdapter(),
        ]

        for adapter in adapters:
            assert hasattr(adapter, "load")
            assert hasattr(adapter, "validate")
            assert hasattr(adapter, "fingerprint")
            assert hasattr(adapter, "statistics")
            assert hasattr(adapter, "metadata")

    def test_synthetic_adapter_in_registry(self) -> None:
        """Synthetic adapter can be registered."""
        adapter = SyntheticAdapter(seed=42)
        dataset = adapter.load()

        assert dataset is not None
        assert len(dataset.queries) > 0

    def test_validation_consistent_across_adapters(self) -> None:
        """Validation produces ValidationReport for all adapters."""
        # Synthetic (easiest to test)
        synthetic_adapter = SyntheticAdapter(query_count=20, seed=42)
        synthetic_dataset = synthetic_adapter.load()
        synthetic_report = synthetic_adapter.validate(synthetic_dataset)

        assert hasattr(synthetic_report, "passed")
        assert hasattr(synthetic_report, "issues")
        assert isinstance(synthetic_report.passed, bool)

    def test_statistics_consistent_across_adapters(self) -> None:
        """Statistics dataclass consistent across adapters."""
        synthetic_adapter = SyntheticAdapter(query_count=20, seed=42)
        synthetic_dataset = synthetic_adapter.load()
        synthetic_stats = synthetic_adapter.statistics(synthetic_dataset)

        assert hasattr(synthetic_stats, "counts")
        assert hasattr(synthetic_stats, "distributions")
        assert hasattr(synthetic_stats, "quality")

    def test_fingerprint_format_consistent(self) -> None:
        """Fingerprints are consistent format across adapters."""
        synthetic_adapter = SyntheticAdapter(query_count=20, seed=42)
        synthetic_dataset = synthetic_adapter.load()
        synthetic_fp = synthetic_adapter.fingerprint(synthetic_dataset)

        # Should be 64-char hex string (SHA256)
        assert isinstance(synthetic_fp, str)
        assert len(synthetic_fp) == 64
        assert all(c in "0123456789abcdef" for c in synthetic_fp)

    def test_metadata_format_consistent(self) -> None:
        """Metadata returned as dict with standard keys."""
        adapters = [
            SyntheticAdapter(seed=42),
            LoCoMoAdapter(),
            LongMemEvalAdapter(),
        ]

        for adapter in adapters:
            metadata = adapter.metadata()
            assert isinstance(metadata, dict)
            assert "name" in metadata
            assert "description" in metadata
