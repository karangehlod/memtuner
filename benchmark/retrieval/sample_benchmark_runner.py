"""Sample benchmark runner for demonstration and validation."""

import json
from typing import Any

# Import all strategy adapters to register them
from benchmark.retrieval.strategies.bm25_adapter import BM25Adapter
from benchmark.retrieval.strategies.tfidf_adapter import TFIDFAdapter
from benchmark.retrieval.strategies.boolean_adapter import BooleanAdapter
from benchmark.retrieval.strategies.dense_vector_adapter import DenseVectorAdapter
from benchmark.retrieval.strategies.learned_dense_adapter import LearnedDenseAdapter
from benchmark.retrieval.strategies.ann_adapter import ANNAdapter
from benchmark.retrieval.strategies.quantized_adapter import QuantizedAdapter
from benchmark.retrieval.strategies.hybrid_fusion_adapter import HybridFusionAdapter
from benchmark.retrieval.strategies.cascading_adapter import CascadingAdapter
from benchmark.retrieval.strategies.retrieval_rerank_adapter import RetrievalRerankAdapter
from benchmark.retrieval.strategies.chainsearch_adapter import ChainSearchAdapter

from benchmark.retrieval.benchmark_orchestrator import RetrievalBenchmarkOrchestrator
from benchmark.retrieval.strategies.base import RetrievalStrategyRegistry


def generate_sample_dataset(name: str, num_docs: int = 100, num_queries: int = 20) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate sample dataset for benchmarking."""
    docs = []
    queries = []

    if name == "qa_dataset":
        # Question-answering style dataset
        qa_pairs = [
            ("What is machine learning?", "Machine learning is a branch of artificial intelligence that enables systems to learn from data."),
            ("How do neural networks work?", "Neural networks are inspired by biological neurons and consist of interconnected layers of artificial neurons."),
            ("What is deep learning?", "Deep learning uses multiple layers of neural networks to extract progressively higher-level features."),
            ("Define natural language processing", "NLP is the field of AI that focuses on enabling computers to understand human language."),
            ("What are transformers?", "Transformers are deep learning models based on the attention mechanism for processing sequences."),
            ("Explain embeddings", "Embeddings are dense vector representations of words or documents in continuous space."),
            ("What is retrieval?", "Retrieval is the task of finding relevant documents for a given query."),
            ("How does ranking work?", "Ranking orders documents by relevance score to the query."),
            ("What is semantic search?", "Semantic search finds documents based on meaning rather than exact keywords."),
            ("Define dense retrieval", "Dense retrieval uses embedding vectors to represent and match documents."),
        ]

        for i in range(num_docs):
            q, a = qa_pairs[i % len(qa_pairs)]
            docs.append({"id": f"doc_{i}", "content": a})

        for i in range(num_queries):
            queries.append(qa_pairs[i % len(qa_pairs)][0])

    elif name == "news_dataset":
        # News articles style dataset
        articles = [
            "Tech company releases new AI model beating previous benchmarks",
            "Researchers develop faster algorithms for machine learning",
            "New dataset enables better natural language understanding",
            "Cloud providers optimize retrieval systems for speed",
            "Open source project gains traction in retrieval community",
            "Benchmark results show significant improvements in recall",
            "Industry adopts new standards for semantic search",
            "Competition drives innovation in dense retrieval",
            "Academic paper proposes novel ranking approach",
            "Startup launches production retrieval system",
        ]

        for i in range(num_docs):
            docs.append({"id": f"doc_{i}", "content": articles[i % len(articles)]})

        queries = [
            "AI model improvements",
            "machine learning algorithms",
            "language understanding",
            "retrieval systems",
            "ranking methods",
        ]

    elif name == "domain_dataset":
        # Domain-specific technical dataset
        content = [
            "BM25 is a probabilistic retrieval model based on term frequency and inverse document frequency",
            "TF-IDF scoring computes term importance in documents relative to a corpus",
            "Boolean retrieval uses AND/OR/NOT operators for exact matching",
            "Vector space model represents documents and queries as vectors",
            "Latent semantic indexing discovers hidden semantic structure",
            "Dense embeddings map text to continuous vector spaces",
            "Approximate nearest neighbor search accelerates retrieval in high dimensions",
            "Cross-encoders provide fine-grained semantic matching between query and document",
            "Reciprocal rank fusion combines multiple ranking signals",
            "Cascading retrieval uses multi-stage pipelines for efficiency",
        ]

        for i in range(num_docs):
            docs.append({"id": f"doc_{i}", "content": content[i % len(content)]})

        queries = [
            "probabilistic retrieval models",
            "term scoring methods",
            "vector representations",
            "neural ranking",
            "fusion strategies",
        ]

    else:  # generic_dataset
        content = [
            "Information retrieval is the task of finding relevant documents in large collections",
            "Queries contain keywords or phrases representing user information needs",
            "Relevance assessment determines how well documents match queries",
            "Metrics like recall and precision evaluate retrieval quality",
            "Efficiency matters for systems handling large-scale data",
            "Scalability requires optimization of algorithms and data structures",
            "User experience depends on fast response times and relevant results",
            "Indexing structures accelerate retrieval operations",
            "Query expansion improves recall by adding related terms",
            "Result diversification reduces redundancy in top results",
        ]

        for i in range(num_docs):
            docs.append({"id": f"doc_{i}", "content": content[i % len(content)]})

        queries = [
            "information retrieval",
            "document ranking",
            "search quality",
            "retrieval efficiency",
            "relevance metrics",
        ]

    return docs, queries


def run_sample_benchmark(output_file: str = "benchmark_results.json") -> dict[str, Any]:
    """Run sample benchmark on multiple datasets."""
    print("=" * 60)
    print("PHASE 4 - RETRIEVAL BENCHMARK - SAMPLE EXECUTION")
    print("=" * 60)

    orchestrator = RetrievalBenchmarkOrchestrator()

    # All 11 retrieval strategies
    all_strategies = [
        "bm25",
        "tfidf",
        "boolean",
        "dense_vector",
        "learned_dense",
        "ann",
        "quantized",
        "hybrid_fusion",
        "cascading",
        "retrieval_rerank",
        "chainsearch",
    ]

    # Sample datasets
    datasets = [
        ("qa_dataset", 50, 15),
        ("news_dataset", 75, 15),
        ("domain_dataset", 100, 20),
        ("generic_dataset", 100, 20),
    ]

    all_results = {}

    for dataset_name, num_docs, num_queries in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name} ({num_docs} docs, {num_queries} queries)")
        print(f"{'=' * 60}")

        docs, queries = generate_sample_dataset(dataset_name, num_docs, num_queries)

        # Run benchmark
        result = orchestrator.benchmark_all_strategies(
            docs,
            queries,
            dataset_name,
            strategy_names=all_strategies,
            top_k=10,
        )

        all_results[dataset_name] = result

        # Print leaderboard
        print(f"\nLeaderboard for {dataset_name}:")
        print("-" * 60)
        leaderboard = orchestrator.get_leaderboard(dataset_name)
        for entry in leaderboard[:5]:  # Top 5
            print(
                f"  {entry['rank']:2d}. {entry['strategy']:18s} | "
                f"Recall@10: {entry['recall_at_10']:.3f} | "
                f"Latency: {entry['latency_ms']:6.2f}ms | "
                f"Score: {entry['score']:.3f}"
            )

    # Export final results
    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}")

    summary = orchestrator.get_summary()
    print(f"\nTotal Datasets: {len(summary['datasets'])}")
    print(f"Total Strategies: {summary['total_strategies']}")

    print("\nPer-Dataset Summary:")
    for dataset, stats in summary["summary"].items():
        print(f"\n  {dataset}:")
        print(f"    Best Overall: {stats['best_overall']['strategy']} (score: {stats['best_overall']['score']:.3f})")
        print(f"    Best Recall: {stats['best_recall']['strategy']} ({stats['best_recall']['recall_at_10']:.3f})")
        print(f"    Best Speed: {stats['best_speed']['strategy']} ({stats['best_speed']['latency_ms']:.2f}ms)")
        print(f"    Avg Recall@10: {stats['avg_recall_at_10']:.3f}")
        print(f"    Avg Latency: {stats['avg_query_latency_ms']:.2f}ms")

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_sample_benchmark()
