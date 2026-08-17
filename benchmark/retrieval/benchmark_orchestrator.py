"""Orchestrator for retrieval benchmark execution."""

import time
from typing import Any
import logging

from benchmark.retrieval.strategies.base import RetrievalStrategyRegistry
from benchmark.retrieval.leaderboard_generator import LeaderboardGenerator


logger = logging.getLogger(__name__)


class RetrievalBenchmarkOrchestrator:
    """Orchestrates full retrieval benchmark across strategies and datasets."""

    def __init__(self):
        self.registry = RetrievalStrategyRegistry
        self.leaderboard_generator = LeaderboardGenerator()
        self.results: dict[str, dict[str, Any]] = {}

    def benchmark_strategy(
        self,
        strategy_name: str,
        documents: list[dict[str, Any]],
        queries: list[str],
        dataset_name: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Benchmark a single strategy on a dataset."""
        try:
            # Get adapter from registry
            adapter = self.registry.get(strategy_name)
            if not adapter:
                logger.error(f"Strategy {strategy_name} not registered")
                return {"status": "failed", "reason": "Strategy not registered"}

            # Initialize
            logger.info(f"Initializing {strategy_name}...")
            adapter.initialize(documents)

            # Run queries
            logger.info(f"Running {len(queries)} queries with {strategy_name}...")
            for query in queries:
                try:
                    adapter.search(query, top_k=top_k)
                except Exception as e:
                    logger.warning(f"Query failed for {strategy_name}: {e}")

            # Get metrics
            metrics = adapter.get_metrics()

            # Add to leaderboard
            self.leaderboard_generator.add_result(dataset_name, metrics)

            # Cleanup
            adapter.teardown()

            logger.info(
                f"✓ {strategy_name}: recall@10={metrics.recall_at_10:.3f}, "
                f"latency={metrics.query_latency_ms:.2f}ms"
            )

            return {
                "status": "success",
                "strategy": strategy_name,
                "recall_at_10": float(metrics.recall_at_10),
                "query_latency_ms": float(metrics.query_latency_ms),
                "index_size_mb": float(metrics.index_size_bytes / (1024 * 1024)),
            }

        except Exception as e:
            logger.error(f"Benchmark failed for {strategy_name}: {e}")
            return {"status": "failed", "reason": str(e)}

    def benchmark_all_strategies(
        self,
        documents: list[dict[str, Any]],
        queries: list[str],
        dataset_name: str,
        strategy_names: list[str] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Benchmark all strategies on a dataset."""
        if strategy_names is None:
            strategy_names = self.registry.list_all()

        logger.info(
            f"Starting benchmark on {dataset_name} "
            f"({len(documents)} docs, {len(queries)} queries)"
        )

        start_time = time.time()
        results = {}

        for strategy_name in strategy_names:
            result = self.benchmark_strategy(
                strategy_name,
                documents,
                queries,
                dataset_name,
                top_k=top_k,
            )
            results[strategy_name] = result

        elapsed = time.time() - start_time
        logger.info(
            f"Benchmark complete: {len(results)} strategies in {elapsed:.1f}s"
        )

        self.results[dataset_name] = {
            "strategies": results,
            "num_documents": len(documents),
            "num_queries": len(queries),
            "total_elapsed_seconds": elapsed,
        }

        return self.results[dataset_name]

    def get_leaderboard(
        self,
        dataset_name: str,
        by: str = "score",
    ) -> list[dict[str, Any]]:
        """Get leaderboard for dataset."""
        leaderboard = self.leaderboard_generator.generate_leaderboard(
            dataset_name,
            by=by,
        )

        return [
            {
                "rank": entry.rank,
                "strategy": entry.strategy_name,
                "recall_at_10": float(entry.recall_at_10),
                "precision_at_10": float(entry.precision_at_10),
                "ndcg": float(entry.ndcg),
                "latency_ms": float(entry.query_latency_ms),
                "index_size_mb": float(entry.index_size_bytes / (1024 * 1024)),
                "score": float(entry.score),
            }
            for entry in leaderboard
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark summary."""
        summary = self.leaderboard_generator.summary()
        return {
            "summary": summary,
            "datasets": list(self.results.keys()),
            "total_strategies": len(self.registry.list_all()),
        }

    def export_results(
        self,
        dataset_name: str,
        format: str = "json",
    ) -> str:
        """Export results in specified format."""
        if format == "json":
            return self.leaderboard_generator.to_json(dataset_name)
        elif format == "csv":
            return self.leaderboard_generator.to_csv(dataset_name)
        else:
            raise ValueError(f"Unknown format: {format}")
