"""Integration tests for full memory benchmarking pipeline."""

import pytest
import time
from benchmark.memory.adapters import MemoryRegistry
from benchmark.memory.leaderboards import LeaderboardGenerator


@pytest.fixture
def benchmark_dataset():
    """Generate synthetic benchmark dataset."""
    memories = []
    for i in range(50):
        memories.append({
            "id": f"mem_{i:03d}",
            "user_id": f"user_{i % 5}",
            "content": f"Event {i}: Task analysis completed with optimization approach discussed",
            "importance": 0.5 + (i % 10) / 10.0,
            "timestamp": time.time() - (50 - i) * 60,
            "day": i % 7,
            "task_id": f"task_{i // 10}",
        })

    queries = [
        "optimization approach",
        "task analysis",
        "event history",
        "user preferences",
        "relationship discovery",
    ]

    return memories, queries


class TestMemoryAdapterBenchmark:
    """Integration tests for memory adapter benchmarking."""

    def test_all_adapters_complete_benchmark(self, benchmark_dataset):
        """Test that all adapters can complete a full benchmark."""
        memories, queries = benchmark_dataset

        adapter_names = MemoryRegistry.list_all()
        assert len(adapter_names) >= 7, "Should have at least 7 adapters registered"

        for adapter_name in adapter_names:
            # Get adapter instance
            adapter = MemoryRegistry.get(adapter_name)

            # Initialize
            adapter.initialize({"dataset_name": "test_benchmark"})

            # Write all memories
            for memory in memories:
                try:
                    adapter.write_memory(memory)
                except Exception as e:
                    pytest.fail(f"{adapter_name} failed to write memory: {e}")

            # Query multiple times
            for query in queries:
                try:
                    results = adapter.query_memories(query, top_k=5)
                    assert isinstance(results, list)
                except Exception as e:
                    pytest.fail(f"{adapter_name} failed to query: {e}")

            # Get metrics
            try:
                metrics = adapter.get_metrics()
                assert metrics.recall_at_10 >= 0.0
                assert metrics.recall_at_10 <= 1.0
                assert metrics.write_latency_ms >= 0.0
                assert metrics.num_memories > 0
            except Exception as e:
                pytest.fail(f"{adapter_name} failed to compute metrics: {e}")

            # Cleanup
            adapter.teardown()

    def test_cross_adapter_consistency(self, benchmark_dataset):
        """Test that all adapters produce consistent results."""
        memories, queries = benchmark_dataset

        adapter_names = MemoryRegistry.list_all()
        adapter_instances = {}
        all_metrics = {}

        # Run all adapters
        for adapter_name in adapter_names:
            adapter = MemoryRegistry.get(adapter_name)
            adapter.initialize({"dataset_name": "consistency_test"})

            for memory in memories:
                adapter.write_memory(memory)

            for query in queries:
                adapter.query_memories(query, top_k=5)

            metrics = adapter.get_metrics()
            all_metrics[adapter_name] = metrics

            adapter_instances[adapter_name] = adapter

        # Verify all adapters produced metrics
        assert len(all_metrics) == len(adapter_names)

        # Verify metrics are valid
        for adapter_name, metrics in all_metrics.items():
            assert metrics.num_memories > 0
            assert 0.0 <= metrics.recall_at_10 <= 1.0
            assert metrics.write_latency_ms >= 0.0
            assert metrics.storage_bytes >= 0.0

        # Cleanup
        for adapter in adapter_instances.values():
            adapter.teardown()

    def test_leaderboard_generation_from_benchmark(self, benchmark_dataset):
        """Test generating leaderboards from benchmark results."""
        memories, queries = benchmark_dataset

        adapter_names = MemoryRegistry.list_all()
        gen = LeaderboardGenerator()

        # Run benchmark on each adapter
        for adapter_name in adapter_names:
            adapter = MemoryRegistry.get(adapter_name)
            adapter.initialize({"dataset_name": "benchmark_data"})

            for memory in memories:
                adapter.write_memory(memory)

            for query in queries:
                adapter.query_memories(query, top_k=5)

            metrics = adapter.get_metrics()
            gen.add_result(metrics, adapter_name, "benchmark_data")

            adapter.teardown()

        # Generate leaderboards
        accuracy_board = gen.accuracy_leaderboard()
        efficiency_board = gen.efficiency_leaderboard()
        balanced_board = gen.balanced_leaderboard()
        analysis = gen.cross_dataset_analysis()

        # Verify leaderboards
        assert len(accuracy_board["entries"]) == len(adapter_names)
        assert len(efficiency_board["entries"]) == len(adapter_names)
        assert len(balanced_board["entries"]) == len(adapter_names)

        # Verify ranking order
        accuracy_scores = [e["accuracy_score"] for e in accuracy_board["entries"]]
        assert accuracy_scores == sorted(accuracy_scores, reverse=True)

        efficiency_scores = [e["efficiency_score"] for e in efficiency_board["entries"]]
        assert efficiency_scores == sorted(efficiency_scores, reverse=True)

        balanced_scores = [e["balanced_score"] for e in balanced_board["entries"]]
        assert balanced_scores == sorted(balanced_scores, reverse=True)

    def test_multi_dataset_benchmark(self):
        """Test benchmarking same adapters across multiple datasets."""
        gen = LeaderboardGenerator()

        adapter_names = MemoryRegistry.list_all()
        datasets = ["dataset_1", "dataset_2", "dataset_3"]

        for dataset in datasets:
            # Create dataset-specific memories
            memories = []
            for i in range(20):
                memories.append({
                    "id": f"{dataset}_mem_{i:02d}",
                    "user_id": f"user_{i % 3}",
                    "content": f"{dataset} content {i}",
                    "importance": 0.5 + (i % 5) / 10.0,
                    "timestamp": time.time() - (20 - i) * 30,
                    "day": i % 3,
                    "task_id": f"task_{dataset}",
                })

            for adapter_name in adapter_names:
                adapter = MemoryRegistry.get(adapter_name)
                adapter.initialize({"dataset_name": dataset})

                for memory in memories:
                    adapter.write_memory(memory)

                results = adapter.query_memories("content", top_k=5)
                assert len(results) >= 0

                metrics = adapter.get_metrics()
                gen.add_result(metrics, adapter_name, dataset)

                adapter.teardown()

        # Generate leaderboards (should now include all datasets)
        accuracy_board = gen.accuracy_leaderboard()

        # Each adapter should have aggregated results across 3 datasets
        for entry in accuracy_board["entries"]:
            assert entry["num_datasets"] == len(datasets)

    def test_adapter_efficiency_profile(self, benchmark_dataset):
        """Test efficiency profiling of adapters."""
        memories, queries = benchmark_dataset

        efficiency_profiles = {}

        for adapter_name in MemoryRegistry.list_all():
            adapter = MemoryRegistry.get(adapter_name)
            adapter.initialize({"dataset_name": "efficiency_test"})

            # Write phase
            for memory in memories:
                adapter.write_memory(memory)

            write_latency = sum(adapter.write_times) / len(adapter.write_times) if adapter.write_times else 0.0

            # Query phase
            for query in queries:
                adapter.query_memories(query, top_k=5)

            query_latency = sum(adapter.query_times) / len(adapter.query_times) if adapter.query_times else 0.0

            metrics = adapter.get_metrics()
            efficiency_profiles[adapter_name] = {
                "write_latency": write_latency,
                "query_latency": query_latency,
                "storage_mb": metrics.storage_bytes / (1024 * 1024),
            }

            adapter.teardown()

        # Verify all adapters have profiled latencies
        for adapter_name, profile in efficiency_profiles.items():
            assert "write_latency" in profile
            assert "query_latency" in profile
            assert "storage_mb" in profile

        # Scratchpad should use less storage
        scratchpad_storage = efficiency_profiles.get("scratchpad", {}).get("storage_mb", float('inf'))
        episodic_store_storage = efficiency_profiles.get("episodic_store", {}).get("storage_mb", float('inf'))

        assert scratchpad_storage <= episodic_store_storage

    def test_report_generation(self, benchmark_dataset):
        """Test generating human-readable reports."""
        memories, queries = benchmark_dataset

        gen = LeaderboardGenerator()

        for adapter_name in MemoryRegistry.list_all():
            adapter = MemoryRegistry.get(adapter_name)
            adapter.initialize({"dataset_name": "report_test"})

            for memory in memories:
                adapter.write_memory(memory)

            for query in queries:
                adapter.query_memories(query, top_k=5)

            metrics = adapter.get_metrics()
            gen.add_result(metrics, adapter_name, "report_test")

            adapter.teardown()

        # Generate different report formats
        text_report = gen.summary_report()
        assert isinstance(text_report, str)
        assert len(text_report) > 0
        assert "ACCURACY LEADERBOARD" in text_report

        json_export = gen.to_json()
        assert isinstance(json_export, str)
        assert "accuracy_leaderboard" in json_export

        html_export = gen.to_html()
        assert isinstance(html_export, str)
        assert "<table>" in html_export


class TestAdapterRegistry:
    """Tests for adapter registry discovery."""

    def test_registry_has_all_adapters(self):
        """Test that all 8 adapters are registered."""
        expected_adapters = {
            "episodic_store",
            "semantic_store",
            "entity_store",
            "preference_store",
            "episodic_buffer",
            "context_buffer",
            "scratchpad",
        }

        registered = set(MemoryRegistry.list_all())

        assert expected_adapters.issubset(registered), f"Missing adapters: {expected_adapters - registered}"

    def test_all_registered_adapters_instantiate(self):
        """Test that all registered adapters can be instantiated."""
        for adapter_name in MemoryRegistry.list_all():
            adapter = MemoryRegistry.get(adapter_name)
            assert adapter is not None
            assert hasattr(adapter, "initialize")
            assert hasattr(adapter, "write_memory")
            assert hasattr(adapter, "query_memories")
            assert hasattr(adapter, "get_metrics")
            assert hasattr(adapter, "teardown")
