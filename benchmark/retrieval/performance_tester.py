"""Performance testing suite for retrieval strategies on larger datasets."""

import time
import random
import string
from typing import Any

import numpy as np

from benchmark.retrieval.benchmark_orchestrator import RetrievalBenchmarkOrchestrator


class PerformanceTester:
    """Test retrieval strategies on various dataset sizes and characteristics."""

    def __init__(self):
        self.orchestrator = RetrievalBenchmarkOrchestrator()
        self.results = {}

    def generate_large_dataset(
        self,
        num_docs: int,
        avg_doc_length: int = 100,
        vocab_size: int = 5000,
        dataset_name: str = "large",
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Generate synthetic large dataset for benchmarking."""
        vocab = [
            "".join(random.choices(string.ascii_lowercase, k=5))
            for _ in range(vocab_size)
        ]

        docs = []
        for i in range(num_docs):
            # Generate document with varied length
            doc_length = max(
                10,
                int(np.random.normal(avg_doc_length, avg_doc_length * 0.3))
            )
            words = random.choices(vocab, k=doc_length)
            content = " ".join(words)
            docs.append({"id": f"doc_{i}", "content": content})

        # Generate queries
        queries = []
        for _ in range(max(10, num_docs // 100)):  # 1% of docs as queries
            query_length = random.randint(2, 5)
            words = random.choices(vocab, k=query_length)
            queries.append(" ".join(words))

        return docs, queries

    def test_scalability(self, dataset_sizes: list[int]) -> dict[str, Any]:
        """Test how strategies scale with increasing dataset size."""
        print("=" * 60)
        print("SCALABILITY TEST - Varying dataset size")
        print("=" * 60)

        results = {}
        strategies = ["bm25", "tfidf", "boolean", "dense_vector", "ann", "quantized"]

        for size in dataset_sizes:
            print(f"\nTesting with {size} documents...")
            docs, queries = self.generate_large_dataset(size, dataset_name=f"size_{size}")

            dataset_results = {}

            for strategy in strategies:
                try:
                    print(f"  {strategy}...", end="", flush=True)

                    start = time.time()
                    result = self.orchestrator.benchmark_strategy(
                        strategy,
                        docs,
                        queries,
                        f"perf_size_{size}",
                        top_k=10,
                    )
                    elapsed = time.time() - start

                    if result["status"] == "success":
                        metrics = {
                            "latency_ms": result["query_latency_ms"],
                            "index_mb": result["index_size_mb"],
                            "recall": result["recall_at_10"],
                        }
                        dataset_results[strategy] = metrics
                        print(f" OK ({result['query_latency_ms']:.2f}ms)")
                    else:
                        print(f" FAILED")
                        dataset_results[strategy] = {"status": "failed"}

                except Exception as e:
                    print(f" ERROR: {e}")
                    dataset_results[strategy] = {"status": "error", "error": str(e)}

            results[size] = dataset_results

        return results

    def test_document_length_variation(self) -> dict[str, Any]:
        """Test strategies with varying document lengths."""
        print("=" * 60)
        print("DOCUMENT LENGTH VARIATION TEST")
        print("=" * 60)

        results = {}
        strategies = ["bm25", "bm25l", "dense_vector", "colbert"]

        doc_length_scenarios = {
            "short": (50, "Short docs (~50 words)"),
            "medium": (200, "Medium docs (~200 words)"),
            "long": (1000, "Long docs (~1000 words)"),
            "very_long": (5000, "Very long docs (~5000 words)"),
            "mixed": None,  # Special case
        }

        for scenario, (length, description) in doc_length_scenarios.items():
            print(f"\n{description}...")

            if scenario == "mixed":
                # Mix of short, medium, long documents
                docs_short, _ = self.generate_large_dataset(100, avg_doc_length=50)
                docs_medium, _ = self.generate_large_dataset(100, avg_doc_length=200)
                docs_long, _ = self.generate_large_dataset(100, avg_doc_length=1000)
                docs = docs_short + docs_medium + docs_long
                for i, doc in enumerate(docs):
                    doc["id"] = f"doc_{i}"
            else:
                docs, _ = self.generate_large_dataset(300, avg_doc_length=length)

            _, queries = self.generate_large_dataset(100, avg_doc_length=5)

            scenario_results = {}

            for strategy in strategies:
                try:
                    print(f"  {strategy}...", end="", flush=True)

                    result = self.orchestrator.benchmark_strategy(
                        strategy,
                        docs,
                        queries,
                        f"perf_length_{scenario}",
                        top_k=10,
                    )

                    if result["status"] == "success":
                        metrics = {
                            "latency_ms": result["query_latency_ms"],
                            "index_mb": result["index_size_mb"],
                            "recall": result["recall_at_10"],
                        }
                        scenario_results[strategy] = metrics
                        print(f" OK ({result['recall_at_10']:.3f})")
                    else:
                        print(f" FAILED")
                        scenario_results[strategy] = {"status": "failed"}

                except Exception as e:
                    print(f" ERROR")
                    scenario_results[strategy] = {"status": "error"}

            results[scenario] = scenario_results

        return results

    def test_vocabulary_size(self) -> dict[str, Any]:
        """Test strategies with varying vocabulary sizes."""
        print("=" * 60)
        print("VOCABULARY SIZE TEST")
        print("=" * 60)

        results = {}
        strategies = ["bm25", "tfidf", "dense_vector", "colbert"]

        vocab_sizes = {
            "small": 1000,
            "medium": 10000,
            "large": 100000,
        }

        for vocab_name, vocab_size in vocab_sizes.items():
            print(f"\nVocabulary size: {vocab_size}...")

            docs, queries = self.generate_large_dataset(
                500,
                avg_doc_length=100,
                vocab_size=vocab_size,
            )

            vocab_results = {}

            for strategy in strategies:
                try:
                    print(f"  {strategy}...", end="", flush=True)

                    result = self.orchestrator.benchmark_strategy(
                        strategy,
                        docs,
                        queries,
                        f"perf_vocab_{vocab_name}",
                        top_k=10,
                    )

                    if result["status"] == "success":
                        metrics = {
                            "latency_ms": result["query_latency_ms"],
                            "index_mb": result["index_size_mb"],
                            "recall": result["recall_at_10"],
                        }
                        vocab_results[strategy] = metrics
                        print(f" OK")
                    else:
                        print(f" FAILED")
                        vocab_results[strategy] = {"status": "failed"}

                except Exception as e:
                    print(f" ERROR")
                    vocab_results[strategy] = {"status": "error"}

            results[vocab_name] = vocab_results

        return results

    def test_query_complexity(self) -> dict[str, Any]:
        """Test strategies with varying query complexity."""
        print("=" * 60)
        print("QUERY COMPLEXITY TEST")
        print("=" * 60)

        results = {}
        strategies = ["bm25", "dense_vector", "hybrid_fusion", "chainsearch"]

        # Generate base dataset
        docs, _ = self.generate_large_dataset(500, avg_doc_length=200)

        query_types = {
            "simple": {
                "description": "1-2 word queries",
                "generator": lambda: " ".join(random.choices(["query", "test", "search"], k=random.randint(1, 2))),
            },
            "moderate": {
                "description": "3-5 word queries",
                "generator": lambda: " ".join(random.choices(["query", "test", "search", "document", "find"], k=random.randint(3, 5))),
            },
            "complex": {
                "description": "6+ word phrases",
                "generator": lambda: " ".join(random.choices(["query", "test", "search", "document", "find", "retrieve", "rank"], k=random.randint(6, 10))),
            },
        }

        for complexity, config in query_types.items():
            print(f"\n{config['description']}...")

            queries = [config["generator"]() for _ in range(20)]

            complexity_results = {}

            for strategy in strategies:
                try:
                    print(f"  {strategy}...", end="", flush=True)

                    result = self.orchestrator.benchmark_strategy(
                        strategy,
                        docs,
                        queries,
                        f"perf_complexity_{complexity}",
                        top_k=10,
                    )

                    if result["status"] == "success":
                        metrics = {
                            "latency_ms": result["query_latency_ms"],
                            "index_mb": result["index_size_mb"],
                            "recall": result["recall_at_10"],
                        }
                        complexity_results[strategy] = metrics
                        print(f" OK")
                    else:
                        print(f" FAILED")
                        complexity_results[strategy] = {"status": "failed"}

                except Exception as e:
                    print(f" ERROR")
                    complexity_results[strategy] = {"status": "error"}

            results[complexity] = complexity_results

        return results

    def print_summary(self, results: dict[str, Any], test_name: str):
        """Print test results summary."""
        print(f"\n{'=' * 60}")
        print(f"SUMMARY - {test_name}")
        print(f"{'=' * 60}\n")

        for category, category_results in results.items():
            print(f"{category}:")
            for strategy, metrics in category_results.items():
                if isinstance(metrics, dict) and "status" not in metrics:
                    print(
                        f"  {strategy:15s} | "
                        f"Latency: {metrics.get('latency_ms', 0):7.2f}ms | "
                        f"Index: {metrics.get('index_mb', 0):7.2f}MB | "
                        f"Recall: {metrics.get('recall', 0):6.3f}"
                    )
                else:
                    print(f"  {strategy:15s} | FAILED/ERROR")
            print()


def run_performance_tests():
    """Run comprehensive performance test suite."""
    tester = PerformanceTester()

    print("\n" + "=" * 60)
    print("RETRIEVAL STRATEGY PERFORMANCE TEST SUITE")
    print("=" * 60 + "\n")

    # Test 1: Scalability
    print("\nTest 1: Scalability with increasing dataset size")
    scalability_results = tester.test_scalability([100, 500, 1000, 5000])
    tester.print_summary(scalability_results, "Scalability Test")

    # Test 2: Document length variation
    print("\nTest 2: Performance with varying document lengths")
    length_results = tester.test_document_length_variation()
    tester.print_summary(length_results, "Document Length Test")

    # Test 3: Vocabulary size
    print("\nTest 3: Performance with varying vocabulary sizes")
    vocab_results = tester.test_vocabulary_size()
    tester.print_summary(vocab_results, "Vocabulary Size Test")

    # Test 4: Query complexity
    print("\nTest 4: Performance with varying query complexity")
    complexity_results = tester.test_query_complexity()
    tester.print_summary(complexity_results, "Query Complexity Test")

    print("\n" + "=" * 60)
    print("ALL PERFORMANCE TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_performance_tests()
