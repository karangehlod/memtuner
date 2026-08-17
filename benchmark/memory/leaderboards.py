"""Memory adapter leaderboard generator (adapter-layer metrics).

NOTE: These leaderboards are built from MemoryAdapter.get_metrics() values,
which use SCORE-ESTIMATED relevance (threshold=0.5). They reflect operational
characteristics (latency, storage, error rate) reliably, but recall/MRR/NDCG
values should NOT be treated as gold-grounded benchmark results. For
gold-grounded results see benchmark_results/leaderboards.json written by
study_runner.py._write_leaderboards_json() which uses StudyAggregator output.

SCORE FORMULAS
--------------
accuracy_score = (recall@1 + recall@5 + recall@10 + mrr + ndcg) / 5.0
  All five metrics are on [0,1]. The equal weighting is intentional but
  note that recall@1 <= recall@5 <= recall@10 by monotonicity, so recall
  has more influence than a naive count suggests. See NOTE 19 in the
  math review. To change weighting, update LeaderboardGenerator.add_result().

efficiency_score = 0.7 * latency_score + 0.3 * storage_score
  latency_score  = 1 / (1 + (write_ms + query_ms) / 10)   [10ms reference]
  storage_score  = 1 / (1 + storage_mb / 100)              [100MB reference]
  Range: (0, 1], higher is better (approaches 1.0 at zero latency/storage).
  This SAME formula is used in both add_result() and efficiency_leaderboard()
  so balanced_score() and the efficiency ranking are consistent.

balanced_score = 0.6 * accuracy_score + 0.4 * efficiency_score
  Range: [0, 1]. The 60/40 split prioritises accuracy over speed.
  To adjust the tradeoff, change the weights in LeaderboardEntry.balanced_score().

WHERE WRITTEN
-------------
benchmark_runner.py  (legacy, only if run standalone)
The canonical output file is benchmark_results/leaderboards.json written by
study_runner.py._write_leaderboards_json() using gold-grounded StudyAggregator data.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from benchmark.memory.adapters import MemoryMetrics, MemoryRegistry


@dataclass
class LeaderboardEntry:
    """Single leaderboard entry.

    Fields marked [score-estimated] come from MemoryAdapter.get_metrics() and
    use a relevance threshold of 0.5 — not gold-label grounded.  Fields marked
    [measured] are wall-clock / OS resource readings and are reliable.
    Fields marked [computed] are derived from other fields in add_result().
    """

    memory_adapter: str          # str  — unique adapter name from MemoryRegistry
    dataset: str                 # str  — dataset name / benchmark label
    accuracy_score: float = 0.0  # float [0,1]  [computed] mean(recall@1,5,10, mrr, ndcg)
    efficiency_score: float = 0.0  # float (0,1]  [computed] 0.7*latency_score + 0.3*storage_score
    recall_at_1: float = 0.0    # float [0,1]  [score-estimated] hits@1 / total queries
    recall_at_5: float = 0.0    # float [0,1]  [score-estimated] hits@5 / total queries
    recall_at_10: float = 0.0   # float [0,1]  [score-estimated] hits@10 / total queries
    recall_at_100: float = 0.0  # float [0,1]  [score-estimated] hits@100 / total queries
    mrr: float = 0.0            # float [0,1]  [score-estimated] mean reciprocal rank
    ndcg: float = 0.0           # float [0,1]  [score-estimated] normalised DCG
    write_latency_ms: float = 0.0       # float  ms  [measured] mean latency per write call
    query_latency_ms: float = 0.0       # float  ms  [measured] mean latency per query call
    query_latency_p50_ms: float = 0.0   # float  ms  [measured] p50 (median) query latency
    query_latency_p95_ms: float = 0.0   # float  ms  [measured] p95 tail query latency
    index_build_ms: float = 0.0         # float  ms  [measured] index construction time
    storage_mb: float = 0.0             # float  MB  [measured] storage_bytes / (1024*1024)
    peak_rss_mb: float = 0.0            # float  MB  [measured] peak resident-set size
    cpu_percent: float = 0.0            # float  %   [measured] CPU utilisation during run
    success_rate: float = 1.0           # float [0,1] [measured] fraction of non-error queries
    num_memories: int = 0               # int   count [measured] total memories written
    num_queries: int = 0                # int   count [measured] total queries executed
    elapsed_seconds: float = 0.0        # float  s   [measured] total wall-clock run time

    def balanced_score(self) -> float:
        """Compute balanced accuracy/efficiency score."""
        return 0.6 * self.accuracy_score + 0.4 * self.efficiency_score


class LeaderboardGenerator:
    """Generates leaderboards from memory adapter benchmarks."""

    def __init__(self):
        self.entries: list[LeaderboardEntry] = []

    def add_result(self, metrics: MemoryMetrics, memory_adapter: str, dataset: str) -> None:
        """Add benchmark result to leaderboard.

        Computes accuracy_score and efficiency_score from raw MemoryMetrics and
        appends a new LeaderboardEntry.

        Accuracy formula (equal-weighted average of five retrieval metrics):
          accuracy_score = (recall@1 + recall@5 + recall@10 + mrr + ndcg) / 5.0
          All inputs are on [0,1]; result is on [0,1].
          NOTE: recall@1 <= recall@5 <= recall@10 by monotonicity, so recall
          terms collectively carry more weight than a naïve equal-weight count
          implies. See module docstring NOTE 19.

        Efficiency formula (same as efficiency_leaderboard() for consistency):
          total_latency  = write_latency_ms + query_latency_ms
          latency_score  = 1 / (1 + total_latency / 10)    # 10 ms reference
          storage_score  = 1 / (1 + storage_mb / 100)      # 100 MB reference
          efficiency_score = 0.7 * latency_score + 0.3 * storage_score
          Range: (0, 1], higher is better.
          Using the identical formula in both places ensures balanced_score()
          and the efficiency leaderboard ranking are always in agreement.
        """
        # Compute accuracy score (average of recall@1, recall@5, recall@10, MRR, NDCG)
        accuracy_score = (metrics.recall_at_1 + metrics.recall_at_5 + metrics.recall_at_10 + metrics.mrr + metrics.ndcg) / 5.0

        # Efficiency score: blended latency + storage signal so it matches what
        # efficiency_leaderboard() independently computes (0.7 latency + 0.3 storage).
        # Using the same formula keeps balanced_score() and efficiency_leaderboard()
        # in agreement — previously they used different signals, producing contradictory rankings.
        total_latency = metrics.write_latency_ms + metrics.query_latency_ms
        latency_score  = 1.0 / (1.0 + total_latency / 10.0)            # 10ms reference
        storage_mb     = metrics.storage_bytes / (1024 * 1024)
        storage_score  = 1.0 / (1.0 + storage_mb / 100.0)              # 100MB reference
        efficiency_score = 0.7 * latency_score + 0.3 * storage_score

        entry = LeaderboardEntry(
            memory_adapter=memory_adapter,
            dataset=dataset,
            accuracy_score=accuracy_score,
            efficiency_score=efficiency_score,
            recall_at_1=metrics.recall_at_1,
            recall_at_5=metrics.recall_at_5,
            recall_at_10=metrics.recall_at_10,
            recall_at_100=metrics.recall_at_100,
            mrr=metrics.mrr,
            ndcg=metrics.ndcg,
            write_latency_ms=metrics.write_latency_ms,
            query_latency_ms=metrics.query_latency_ms,
            query_latency_p50_ms=metrics.query_latency_p50_ms,
            query_latency_p95_ms=metrics.query_latency_p95_ms,
            index_build_ms=metrics.index_build_ms,
            storage_mb=metrics.storage_bytes / (1024 * 1024),
            peak_rss_mb=metrics.peak_rss_mb,
            cpu_percent=metrics.cpu_percent,
            success_rate=metrics.success_rate,
            num_memories=metrics.num_memories,
            num_queries=metrics.num_queries,
            elapsed_seconds=metrics.elapsed_seconds,
        )

        self.entries.append(entry)

    def accuracy_leaderboard(self) -> dict[str, Any]:
        """Generate accuracy leaderboard across all datasets.

        Ranks memory adapters by average accuracy (recall, MRR, NDCG).

        Returns a dict with the following top-level keys:
          "title"   (str)  — human-readable label, "Accuracy Leaderboard"
          "metric"  (str)  — description of the ranking metric
          "entries" (list) — one dict per adapter, sorted by accuracy_score desc:
            "rank"           (int)   — 1-based rank position
            "memory_adapter" (str)   — adapter identifier
            "accuracy_score" (float) — mean accuracy score across datasets, [0,1]
            "num_datasets"   (int)   — number of datasets this adapter was evaluated on
            "recall_at_1"    (float) — mean recall@1 across datasets, [0,1]
            "recall_at_5"    (float) — mean recall@5 across datasets, [0,1]
            "mrr"            (float) — mean MRR across datasets, [0,1]
            "ndcg"           (float) — mean nDCG across datasets, [0,1]

        Returns {} if no entries have been added.
        """
        if not self.entries:
            return {}

        # Group by memory adapter
        adapter_scores: dict[str, list[float]] = {}

        for entry in self.entries:
            adapter = entry.memory_adapter
            if adapter not in adapter_scores:
                adapter_scores[adapter] = []

            adapter_scores[adapter].append(entry.accuracy_score)

        # Compute average accuracy per adapter
        adapter_averages = {
            adapter: sum(scores) / len(scores)
            for adapter, scores in adapter_scores.items()
        }

        # Sort by accuracy
        ranked = sorted(
            adapter_averages.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build leaderboard
        leaderboard = {
            "title": "Accuracy Leaderboard",
            "metric": "Average Recall@1, Recall@5, Recall@10, MRR, NDCG",
            "entries": [
                {
                    "rank": idx + 1,
                    "memory_adapter": adapter,
                    "accuracy_score": score,
                    "num_datasets": len(adapter_scores[adapter]),
                    "recall_at_1": sum(e.recall_at_1 for e in self.entries if e.memory_adapter == adapter) / len(adapter_scores[adapter]),
                    "recall_at_5": sum(e.recall_at_5 for e in self.entries if e.memory_adapter == adapter) / len(adapter_scores[adapter]),
                    "mrr": sum(e.mrr for e in self.entries if e.memory_adapter == adapter) / len(adapter_scores[adapter]),
                    "ndcg": sum(e.ndcg for e in self.entries if e.memory_adapter == adapter) / len(adapter_scores[adapter]),
                }
                for idx, (adapter, score) in enumerate(ranked)
            ],
        }

        return leaderboard

    def efficiency_leaderboard(self) -> dict[str, Any]:
        """Generate efficiency leaderboard across all datasets.

        Ranks memory adapters by the same efficiency formula used in add_result():
          latency_score  = 1 / (1 + avg_total_latency_ms / 10)
          storage_score  = 1 / (1 + avg_storage_mb / 100)
          efficiency_score = 0.7 * latency_score + 0.3 * storage_score

        Returns a dict with the following top-level keys:
          "title"   (str)  — human-readable label, "Efficiency Leaderboard"
          "metric"  (str)  — description of the ranking metric
          "entries" (list) — one dict per adapter, sorted by efficiency_score desc:
            "rank"                 (int)   — 1-based rank position
            "memory_adapter"       (str)   — adapter identifier
            "efficiency_score"     (float) — blended efficiency score, (0,1]
            "avg_write_latency_ms" (float) — mean write latency across datasets, ms
            "avg_query_latency_ms" (float) — mean query latency across datasets, ms
            "p50_query_ms"         (float) — mean p50 query latency across datasets, ms
            "p95_query_ms"         (float) — mean p95 query latency across datasets, ms
            "avg_index_build_ms"   (float) — mean index build time across datasets, ms
            "avg_storage_mb"       (float) — mean storage across datasets, MB
            "avg_peak_rss_mb"      (float) — mean peak RSS across datasets, MB

        Returns {} if no entries have been added.
        """
        if not self.entries:
            return {}

        # Group by memory adapter
        adapter_latencies: dict[str, list[float]] = {}
        adapter_storage: dict[str, list[float]] = {}

        for entry in self.entries:
            adapter = entry.memory_adapter
            if adapter not in adapter_latencies:
                adapter_latencies[adapter] = []
                adapter_storage[adapter] = []

            total_latency = entry.write_latency_ms + entry.query_latency_ms
            adapter_latencies[adapter].append(total_latency)
            adapter_storage[adapter].append(entry.storage_mb)

        # Compute average efficiency per adapter
        adapter_scores = {}
        for adapter in adapter_latencies:
            avg_latency = sum(adapter_latencies[adapter]) / len(adapter_latencies[adapter])
            avg_storage = sum(adapter_storage[adapter]) / len(adapter_storage[adapter])

            # Efficiency score: inverse of latency + storage
            latency_score = 1.0 / (1.0 + avg_latency / 10.0)
            storage_score = 1.0 / (1.0 + avg_storage / 100.0)  # 100MB reference

            adapter_scores[adapter] = 0.7 * latency_score + 0.3 * storage_score

        # Sort by efficiency
        ranked = sorted(
            adapter_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build leaderboard
        leaderboard = {
            "title": "Efficiency Leaderboard",
            "metric": "Write + Query Latency (lower is better)",
            "entries": [
                {
                    "rank": idx + 1,
                    "memory_adapter": adapter,
                    "efficiency_score": score,
                    "avg_write_latency_ms": sum(e.write_latency_ms for e in self.entries if e.memory_adapter == adapter) / len(adapter_latencies[adapter]),
                    "avg_query_latency_ms": sum(e.query_latency_ms for e in self.entries if e.memory_adapter == adapter) / len(adapter_latencies[adapter]),
                    "p50_query_ms": sum(e.query_latency_p50_ms for e in self.entries if e.memory_adapter == adapter) / len(adapter_latencies[adapter]),
                    "p95_query_ms": sum(e.query_latency_p95_ms for e in self.entries if e.memory_adapter == adapter) / len(adapter_latencies[adapter]),
                    "avg_index_build_ms": sum(e.index_build_ms for e in self.entries if e.memory_adapter == adapter) / len(adapter_latencies[adapter]),
                    "avg_storage_mb": sum(adapter_storage[adapter]) / len(adapter_storage[adapter]),
                    "avg_peak_rss_mb": sum(e.peak_rss_mb for e in self.entries if e.memory_adapter == adapter) / len(adapter_latencies[adapter]),
                }
                for idx, (adapter, score) in enumerate(ranked)
            ],
        }

        return leaderboard

    def balanced_leaderboard(self) -> dict[str, Any]:
        """Generate balanced leaderboard (accuracy + efficiency tradeoff).

        Ranks memory adapters by LeaderboardEntry.balanced_score():
          balanced_score = 0.6 * accuracy_score + 0.4 * efficiency_score
          Range: [0, 1], higher is better.

        Returns a dict with the following top-level keys:
          "title"   (str)  — human-readable label, "Balanced Leaderboard"
          "metric"  (str)  — description of the ranking metric
          "entries" (list) — one dict per adapter, sorted by balanced_score desc:
            "rank"           (int)   — 1-based rank position
            "memory_adapter" (str)   — adapter identifier
            "balanced_score" (float) — 0.6*accuracy + 0.4*efficiency, [0,1]
            "profile"        (str)   — one of:
                "High-accuracy specialist"    (rank 1)
                "Ultra-low latency specialist" (last rank)
                "Balanced"                    (middle rank when len > 2)
                "Standard"                    (all other adapters)

        Returns {} if no entries have been added.
        """
        if not self.entries:
            return {}

        # Group by memory adapter
        adapter_balanced_scores: dict[str, list[float]] = {}

        for entry in self.entries:
            adapter = entry.memory_adapter
            if adapter not in adapter_balanced_scores:
                adapter_balanced_scores[adapter] = []

            balanced = entry.balanced_score()
            adapter_balanced_scores[adapter].append(balanced)

        # Compute average balanced score per adapter
        adapter_averages = {
            adapter: sum(scores) / len(scores)
            for adapter, scores in adapter_balanced_scores.items()
        }

        # Sort by balanced score
        ranked = sorted(
            adapter_averages.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Determine profiles
        profiles = {
            ranked[0][0]: "High-accuracy specialist",
            ranked[-1][0]: "Ultra-low latency specialist",
        }

        if len(ranked) > 2:
            mid_idx = len(ranked) // 2
            profiles[ranked[mid_idx][0]] = "Balanced"

        # Build leaderboard
        leaderboard = {
            "title": "Balanced Leaderboard",
            "metric": "60% Accuracy + 40% Efficiency",
            "entries": [
                {
                    "rank": idx + 1,
                    "memory_adapter": adapter,
                    "balanced_score": score,
                    "profile": profiles.get(adapter, "Standard"),
                }
                for idx, (adapter, score) in enumerate(ranked)
            ],
        }

        return leaderboard

    def cross_dataset_analysis(self) -> dict[str, Any]:
        """Generate cross-dataset analysis.

        Shows which memory adapters excel on which datasets.
        """
        if not self.entries:
            return {}

        # Build matrix: adapters × datasets
        analysis: dict[str, dict[str, float]] = {}

        for entry in self.entries:
            adapter = entry.memory_adapter
            dataset = entry.dataset

            if adapter not in analysis:
                analysis[adapter] = {}

            analysis[adapter][dataset] = entry.accuracy_score

        # Find specializations
        specializations: dict[str, dict[str, Any]] = {}

        for adapter, datasets in analysis.items():
            best_dataset = max(datasets.items(), key=lambda x: x[1])
            worst_dataset = min(datasets.items(), key=lambda x: x[1])

            specializations[adapter] = {
                "best_dataset": best_dataset[0],
                "best_score": best_dataset[1],
                "worst_dataset": worst_dataset[0],
                "worst_score": worst_dataset[1],
                "avg_score": sum(datasets.values()) / len(datasets),
            }

        return {
            "title": "Cross-Dataset Analysis",
            "specializations": specializations,
            "analysis_matrix": analysis,
        }

    def summary_report(self) -> str:
        """Generate human-readable summary report."""
        accuracy = self.accuracy_leaderboard()
        efficiency = self.efficiency_leaderboard()
        balanced = self.balanced_leaderboard()
        analysis = self.cross_dataset_analysis()

        report = []
        report.append("=" * 80)
        report.append("MEMORY ADAPTER BENCHMARKING LEADERBOARDS")
        report.append("=" * 80)
        report.append("")

        # Accuracy leaderboard
        report.append("📊 ACCURACY LEADERBOARD")
        report.append("-" * 80)
        if accuracy.get("entries"):
            for entry in accuracy["entries"]:
                report.append(
                    f"  {entry['rank']}. {entry['memory_adapter']:<20} "
                    f"Accuracy: {entry['accuracy_score']:.3f} "
                    f"({entry['num_datasets']} datasets)"
                )
        report.append("")

        # Efficiency leaderboard
        report.append("⚡ EFFICIENCY LEADERBOARD")
        report.append("-" * 80)
        if efficiency.get("entries"):
            for entry in efficiency["entries"]:
                report.append(
                    f"  {entry['rank']}. {entry['memory_adapter']:<20} "
                    f"Efficiency: {entry['efficiency_score']:.3f} "
                    f"Write: {entry.get('avg_write_latency_ms', 0):.2f}ms  "
                    f"Query P50: {entry.get('p50_query_ms', 0):.2f}ms  "
                    f"Storage: {entry['avg_storage_mb']:.1f}MB"
                )
        report.append("")

        # Balanced leaderboard
        report.append("⚖️  BALANCED LEADERBOARD")
        report.append("-" * 80)
        if balanced.get("entries"):
            for entry in balanced["entries"]:
                report.append(
                    f"  {entry['rank']}. {entry['memory_adapter']:<20} "
                    f"Score: {entry['balanced_score']:.3f} "
                    f"({entry['profile']})"
                )
        report.append("")

        # Specializations
        if analysis.get("specializations"):
            report.append("🎯 SPECIALIZATIONS")
            report.append("-" * 80)
            for adapter, spec in analysis["specializations"].items():
                report.append(f"  {adapter}:")
                report.append(f"    Best on:   {spec['best_dataset']} ({spec['best_score']:.3f})")
                report.append(f"    Worst on:  {spec['worst_dataset']} ({spec['worst_score']:.3f})")
                report.append(f"    Average:   {spec['avg_score']:.3f}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def to_json(self) -> str:
        """Export all leaderboards as JSON."""
        data = {
            "accuracy_leaderboard": self.accuracy_leaderboard(),
            "efficiency_leaderboard": self.efficiency_leaderboard(),
            "balanced_leaderboard": self.balanced_leaderboard(),
            "cross_dataset_analysis": self.cross_dataset_analysis(),
        }

        return json.dumps(data, indent=2)

    def to_html(self) -> str:
        """Export leaderboards as HTML."""
        accuracy = self.accuracy_leaderboard()
        efficiency = self.efficiency_leaderboard()
        balanced = self.balanced_leaderboard()

        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <title>Memory Adapter Leaderboards</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("    table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
        html.append("    th { background-color: #4CAF50; color: white; }")
        html.append("    tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        html.append("  <h1>Memory Adapter Leaderboards</h1>")

        # Accuracy table
        html.append("  <h2>📊 Accuracy Leaderboard</h2>")
        html.append("  <table>")
        html.append("    <tr><th>Rank</th><th>Memory Adapter</th><th>Accuracy Score</th>"
                    "<th>Recall@1</th><th>Recall@5</th><th>MRR</th><th>NDCG</th><th>Datasets</th></tr>")
        if accuracy.get("entries"):
            for entry in accuracy["entries"]:
                html.append(
                    f"    <tr><td>{entry['rank']}</td><td>{entry['memory_adapter']}</td>"
                    f"<td>{entry['accuracy_score']:.3f}</td>"
                    f"<td>{entry.get('recall_at_1', 0):.3f}</td>"
                    f"<td>{entry.get('recall_at_5', 0):.3f}</td>"
                    f"<td>{entry.get('mrr', 0):.3f}</td>"
                    f"<td>{entry.get('ndcg', 0):.3f}</td>"
                    f"<td>{entry['num_datasets']}</td></tr>"
                )
        html.append("  </table>")

        # Efficiency table
        html.append("  <h2>⚡ Efficiency Leaderboard</h2>")
        html.append("  <table>")
        html.append("    <tr><th>Rank</th><th>Memory Adapter</th><th>Efficiency</th>"
                    "<th>Write ms</th><th>Query P50 ms</th><th>Query P95 ms</th>"
                    "<th>Index Build ms</th><th>Storage MB</th><th>Peak RSS MB</th></tr>")
        if efficiency.get("entries"):
            for entry in efficiency["entries"]:
                html.append(
                    f"    <tr><td>{entry['rank']}</td><td>{entry['memory_adapter']}</td>"
                    f"<td>{entry['efficiency_score']:.3f}</td>"
                    f"<td>{entry.get('avg_write_latency_ms', 0):.3f}</td>"
                    f"<td>{entry.get('p50_query_ms', 0):.3f}</td>"
                    f"<td>{entry.get('p95_query_ms', 0):.3f}</td>"
                    f"<td>{entry.get('avg_index_build_ms', 0):.1f}</td>"
                    f"<td>{entry['avg_storage_mb']:.2f}</td>"
                    f"<td>{entry.get('avg_peak_rss_mb', 0):.1f}</td></tr>"
                )
        html.append("  </table>")

        # Balanced table
        html.append("  <h2>⚖️ Balanced Leaderboard</h2>")
        html.append("  <table>")
        html.append("    <tr><th>Rank</th><th>Memory Adapter</th><th>Score</th><th>Profile</th></tr>")
        if balanced.get("entries"):
            for entry in balanced["entries"]:
                html.append(
                    f"    <tr><td>{entry['rank']}</td><td>{entry['memory_adapter']}</td>"
                    f"<td>{entry['balanced_score']:.3f}</td><td>{entry['profile']}</td></tr>"
                )
        html.append("  </table>")

        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)
