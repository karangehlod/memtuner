"""Leaderboard generator for retrieval benchmark results."""

from dataclasses import dataclass, field, asdict
from typing import Any
import json
from benchmark.retrieval.strategies.base import RetrievalStrategyRegistry, RetrievalMetrics


@dataclass
class LeaderboardEntry:
    """Entry in a retrieval leaderboard."""
    rank: int
    strategy_name: str
    dataset_name: str
    recall_at_10: float
    recall_at_100: float
    mrr: float
    ndcg: float
    precision_at_10: float
    query_latency_ms: float
    index_build_time_sec: float
    index_size_bytes: float
    success_rate: float
    error_count: int
    num_queries: int
    num_documents: int
    elapsed_seconds: float
    score: float = field(default=0.0)  # Composite score


class LeaderboardGenerator:
    """Generate retrieval benchmarks leaderboards."""

    def __init__(self):
        self.results: dict[str, list[LeaderboardEntry]] = {}
        self.registry = RetrievalStrategyRegistry

    def add_result(
        self,
        dataset_name: str,
        metrics: RetrievalMetrics,
    ) -> None:
        """Add benchmark result to leaderboard."""
        if dataset_name not in self.results:
            self.results[dataset_name] = []

        # Compute composite score (weighted metrics)
        score = self._compute_score(metrics)

        entry = LeaderboardEntry(
            rank=0,  # Will be set during ranking
            strategy_name=metrics.strategy_name,
            dataset_name=dataset_name,
            recall_at_10=metrics.recall_at_10,
            recall_at_100=metrics.recall_at_100,
            mrr=metrics.mrr,
            ndcg=metrics.ndcg,
            precision_at_10=metrics.precision_at_10,
            query_latency_ms=metrics.query_latency_ms,
            index_build_time_sec=metrics.index_build_time_sec,
            index_size_bytes=metrics.index_size_bytes,
            success_rate=metrics.success_rate,
            error_count=metrics.error_count,
            num_queries=metrics.num_queries,
            num_documents=metrics.num_documents,
            elapsed_seconds=metrics.elapsed_seconds,
            score=score,
        )

        self.results[dataset_name].append(entry)

    def _compute_score(self, metrics: RetrievalMetrics) -> float:
        """Compute composite retrieval score."""
        # Balanced scoring: 50% accuracy, 30% efficiency, 20% reliability
        accuracy_score = (
            0.3 * metrics.recall_at_10 +
            0.2 * metrics.recall_at_100 +
            0.3 * metrics.ndcg +
            0.2 * metrics.precision_at_10
        )

        # Efficiency: normalize latency (lower is better)
        # Assume optimal is ~1ms, penalize anything slower
        latency_score = max(0.0, 1.0 - (metrics.query_latency_ms / 100.0))

        # Index efficiency: smaller index is better
        # Normalize by assuming ~1GB is large
        index_score = max(0.0, 1.0 - (metrics.index_size_bytes / (1e9)))

        efficiency_score = 0.6 * latency_score + 0.4 * index_score
        reliability_score = metrics.success_rate

        # Weighted composite
        return (
            0.5 * accuracy_score +
            0.3 * efficiency_score +
            0.2 * reliability_score
        )

    def generate_leaderboard(self, dataset_name: str, by: str = "score") -> list[LeaderboardEntry]:
        """Generate ranked leaderboard for dataset."""
        if dataset_name not in self.results:
            return []

        entries = self.results[dataset_name]

        # Sort by requested metric
        if by == "score":
            sorted_entries = sorted(entries, key=lambda x: x.score, reverse=True)
        elif by == "recall_at_10":
            sorted_entries = sorted(entries, key=lambda x: x.recall_at_10, reverse=True)
        elif by == "precision_at_10":
            sorted_entries = sorted(entries, key=lambda x: x.precision_at_10, reverse=True)
        elif by == "query_latency_ms":
            sorted_entries = sorted(entries, key=lambda x: x.query_latency_ms)
        elif by == "index_size_bytes":
            sorted_entries = sorted(entries, key=lambda x: x.index_size_bytes)
        else:
            sorted_entries = sorted(entries, key=lambda x: x.score, reverse=True)

        # Assign ranks
        for rank, entry in enumerate(sorted_entries, 1):
            entry.rank = rank

        return sorted_entries

    def generate_all_leaderboards(self) -> dict[str, list[LeaderboardEntry]]:
        """Generate leaderboards for all datasets."""
        leaderboards = {}
        for dataset_name in self.results.keys():
            leaderboards[dataset_name] = self.generate_leaderboard(dataset_name)
        return leaderboards

    def to_json(self, dataset_name: str, by: str = "score") -> str:
        """Export leaderboard as JSON."""
        leaderboard = self.generate_leaderboard(dataset_name, by=by)
        entries = [asdict(entry) for entry in leaderboard]
        return json.dumps(entries, indent=2)

    def to_csv(self, dataset_name: str, by: str = "score") -> str:
        """Export leaderboard as CSV."""
        leaderboard = self.generate_leaderboard(dataset_name, by=by)

        if not leaderboard:
            return ""

        # Header
        headers = [
            "rank",
            "strategy_name",
            "recall_at_10",
            "recall_at_100",
            "ndcg",
            "precision_at_10",
            "query_latency_ms",
            "index_size_mb",
            "success_rate",
            "score",
        ]

        lines = [",".join(headers)]

        # Rows
        for entry in leaderboard:
            row = [
                str(entry.rank),
                entry.strategy_name,
                f"{entry.recall_at_10:.4f}",
                f"{entry.recall_at_100:.4f}",
                f"{entry.ndcg:.4f}",
                f"{entry.precision_at_10:.4f}",
                f"{entry.query_latency_ms:.2f}",
                f"{entry.index_size_bytes / (1024 * 1024):.2f}",
                f"{entry.success_rate:.4f}",
                f"{entry.score:.4f}",
            ]
            lines.append(",".join(row))

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        summary_data = {}

        for dataset_name, entries in self.results.items():
            if not entries:
                continue

            top_by_score = sorted(entries, key=lambda x: x.score, reverse=True)[0]
            top_by_recall = sorted(entries, key=lambda x: x.recall_at_10, reverse=True)[0]
            top_by_speed = sorted(entries, key=lambda x: x.query_latency_ms)[0]

            summary_data[dataset_name] = {
                "num_strategies": len(entries),
                "num_queries": entries[0].num_queries,
                "num_documents": entries[0].num_documents,
                "best_overall": {
                    "strategy": top_by_score.strategy_name,
                    "score": float(top_by_score.score),
                },
                "best_recall": {
                    "strategy": top_by_recall.strategy_name,
                    "recall_at_10": float(top_by_recall.recall_at_10),
                },
                "best_speed": {
                    "strategy": top_by_speed.strategy_name,
                    "latency_ms": float(top_by_speed.query_latency_ms),
                },
                "avg_recall_at_10": float(
                    sum(e.recall_at_10 for e in entries) / len(entries)
                ),
                "avg_query_latency_ms": float(
                    sum(e.query_latency_ms for e in entries) / len(entries)
                ),
            }

        return summary_data
