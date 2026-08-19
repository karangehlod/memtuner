"""Cross-dataset results report generator.

Presents measured results across datasets: rankings by Recall@K, per-phase
breakdowns, structural dataset characteristics, and IR-theory context for
each (dataset, strategy) combination.

Interpretation of which configuration is appropriate belongs to the researcher.

Usage:
    agg = StudyAggregator(all_results)
    gen = NarrativeReportGenerator()
    text = gen.generate(gold_path_to_name_map, agg, output_path)
"""

from __future__ import annotations

import math
from pathlib import Path

from benchmark.gold.dataset_profiles import DATASET_PROFILES, DatasetProfile, get_profile


# Explanation library: maps (dataset_key, strategy) → WHY this works/fails.
# These are structural IR-theory explanations about dataset/strategy compatibility.
# They do NOT contain specific numbers (recall values, lambda, weights) —
# those come from the actual run results passed to generate().
_STRATEGY_EXPLANATIONS: dict[tuple[str, str], str] = {
    ("locomo", "bm25"): (
        "BM25 performs well on LoCoMo because conversation logs retain exact words "
        "from the original exchange — queries often match memory content verbatim."
    ),
    ("locomo", "hybrid"): (
        "Hybrid wins on LoCoMo because conversations mix exact quotes (favoring BM25) "
        "with paraphrases. RRF fusion captures both signals. Decay is critical — "
        "without it, recent low-relevance memories outrank old gold memories."
    ),
    ("locomo", "embeddings"): (
        "Pure embedding can underperform on LoCoMo because semantic similarity "
        "conflates related topics that are not the specific answer. Paraphrase "
        "retrieval helps but keyword precision matters in conversation logs."
    ),
    ("locomo", "recency"): (
        "Recency performs poorly on LoCoMo because gold memories span many months. "
        "The most recently injected memory is rarely the correct one — this confirms "
        "that retrieval quality (not recency bias) is what the system measures."
    ),
    ("longmemeval", "embeddings"): (
        "Semantic embeddings capture meaning drift on LongMemEval — facts that change "
        "over time produce memories with overlapping embeddings even without shared "
        "keywords, allowing the model to distinguish old from new."
    ),
    ("longmemeval", "bm25"): (
        "BM25 struggles on LongMemEval because updated facts share keywords with "
        "old facts. Both 'junior developer' and 'team lead' match 'current role?' "
        "equally via keyword overlap; semantic search distinguishes them by meaning."
    ),
    ("squad", "bm25"): (
        "SQuAD paragraphs are extractive and keyword-dense — the answer span appears "
        "verbatim in the passage. BM25 exploits this directly; semantic search "
        "adds false positives from thematically similar but wrong paragraphs."
    ),
    ("squad", "hybrid"): (
        "Hybrid performs well on SQuAD because the semantic component helps with "
        "unanswerable questions by detecting when no passage is sufficiently close."
    ),
    ("coqa", "hybrid"): (
        "CoQA requires resolving coreference across turns. BM25 matches the topic "
        "word; embeddings match the semantic context of the referent. Together "
        "they surface the turn containing the antecedent."
    ),
    ("hotpotqa", "hybrid"): (
        "HotpotQA requires two documents for a complete answer. BM25 finds "
        "documents with explicit keywords; semantic search finds the bridge document "
        "whose relevance is implicit. Hybrid fusion covers both hops."
    ),
    ("synthetic", "bm25"): (
        "Synthetic data uses templated content with predictable vocabulary. BM25 "
        "finds the correct memory easily. Synthetic results are most useful for "
        "verifying that decay and temporal constraints behave correctly."
    ),
}


def _get_explanation(ds_key: str, strategy: str) -> str:
    """Return an explanation for why this strategy performs as it does."""
    return _STRATEGY_EXPLANATIONS.get(
        (ds_key, strategy),
        f"{strategy} ranked highest on this dataset in the empirical sweep. "
        f"No structural IR-theory explanation is registered for this (dataset, strategy) pair."
    )


class NarrativeReportGenerator:
    """Generates the cross-dataset story from benchmark results."""

    def generate(
        self,
        per_dataset_summary: dict[str, dict],
        output_path: Path | None = None,
    ) -> str:
        """Generate the narrative report.

        Args:
            per_dataset_summary: Maps dataset name → study_summary dict
                                  (from StudyAggregator.study_summary()).
            output_path: Optional path to write the report text.

        Returns:
            The report as a plain-text string.
        """
        lines = self._header()
        lines += self._executive_summary(per_dataset_summary)
        lines += self._winner_matrix(per_dataset_summary)

        for ds_name, summary in sorted(per_dataset_summary.items()):
            lines += self._dataset_section(ds_name, summary)

        lines += self._cross_dataset_insights(per_dataset_summary)
        lines += self._bring_your_own_dataset()

        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
        return report

    # ── Section builders ──────────────────────────────────────────────────────

    def _header(self) -> list[str]:
        return [
            "=" * 72,
            "  AGENTIC MEMORY BENCHMARK — CROSS-DATASET RESULTS REPORT",
            "=" * 72,
            "",
            "  This report presents measured results across datasets.",
            "  Rankings are by avg Recall@K. Interpretation is left to the researcher.",
            "",
        ]

    def _executive_summary(self, summaries: dict[str, dict]) -> list[str]:
        lines = ["─" * 72, "  TOP-RANKED CONFIGURATION PER DATASET", "─" * 72, ""]
        for ds_name, summary in sorted(summaries.items()):
            profile = get_profile(ds_name)
            recs = summary.get("recommendations", {})
            best_strat = recs.get("best_retrieval_strategy", "—")
            best_embed = (recs.get("best_embedding_model") or "—").split("/")[-1]
            best_rerank = recs.get("best_reranker", "none")
            best_lambda = _best_lambda(summary)

            char = profile.character if profile else "unknown"
            lines += [
                f"  {ds_name.upper():<20s} ({char})",
                f"    → {best_strat} | embed={best_embed} | reranker={best_rerank} | λ={best_lambda}",
                "",
            ]
        return lines

    def _winner_matrix(self, summaries: dict[str, dict]) -> list[str]:
        strategies = ["bm25", "embeddings", "hybrid", "recency", "api_embeddings"]
        present_strats = set()
        for s in summaries.values():
            for row in s.get("retrieval_strategy_ranking", []):
                present_strats.add(row["retrieval_strategy"])
        strategies = [s for s in strategies if s in present_strats]

        lines = ["─" * 72, "  RECALL@K BY STRATEGY × DATASET  (★ = top-ranked per dataset)", "─" * 72, ""]
        header = f"  {'Dataset':<20}" + "".join(f"  {s[:12]:<14}" for s in strategies)
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))

        for ds_name, summary in sorted(summaries.items()):
            recs = summary.get("recommendations", {})
            winner = recs.get("best_retrieval_strategy", "")
            strat_ranks = {
                r["retrieval_strategy"]: r["avg_recall"]
                for r in summary.get("retrieval_strategy_ranking", [])
            }
            row = f"  {ds_name:<20}"
            for s in strategies:
                if s not in strat_ranks:
                    row += f"  {'—':<14}"
                elif s == winner:
                    row += f"  {'★ ' + f'{strat_ranks[s]:.3f}':<14}"
                else:
                    row += f"  {strat_ranks[s]:.3f}{'':<9}"
            lines.append(row)

        lines.append("")
        return lines

    def _dataset_section(self, ds_name: str, summary: dict) -> list[str]:
        profile = get_profile(ds_name)
        recs = summary.get("recommendations", {})
        best_strat = recs.get("best_retrieval_strategy") or "—"
        best_embed = recs.get("best_embedding_model") or "—"
        best_rerank = recs.get("best_reranker") or "none"

        strat_ranks = summary.get("retrieval_strategy_ranking", [])
        embed_ranks = summary.get("embedding_model_ranking", [])
        top_recall = strat_ranks[0]["avg_recall"] if strat_ranks else 0.0
        second_recall = strat_ranks[1]["avg_recall"] if len(strat_ranks) > 1 else 0.0
        margin = top_recall - second_recall

        lines = [
            "─" * 72,
            f"  DATASET: {(profile.display_name if profile else ds_name.upper())}",
            "─" * 72,
        ]

        if profile:
            lines += [
                f"  Character   : {profile.character}",
                f"  Query style : {profile.query_style}",
                f"  Time span   : {profile.time_span_days} days" if isinstance(profile.time_span_days, int) else f"  Time span   : {profile.time_span_days}",
                f"  Known challenge : {_wrap(profile.why_hard, 68, prefix='               ')}",
                "",
            ]

        lines += [
            f"  Top config     : {best_strat} | embed={(best_embed or '—').split('/')[-1]} | reranker={best_rerank} | λ={_best_lambda(summary)}",
            f"  Recall@K    : {top_recall:.1%}   (gap to 2nd place: {margin:.3f})",
            "",
            f"  Structural explanation (IR theory, not a verdict):",
            f"  {_wrap(_get_explanation(ds_name.lower(), best_strat), 68, '  ')}",
            "",
        ]

        if strat_ranks:
            lines.append("  Strategy ranking (Recall@K | Composite):")
            for i, r in enumerate(strat_ranks[:4], 1):
                marker = "★" if i == 1 else " "
                lines.append(
                    f"  {marker} {i}. {r['retrieval_strategy']:<18} "
                    f"recall={r['avg_recall']:.4f}  composite={r['avg_composite']:.4f}"
                )
            lines.append("")

        if embed_ranks:
            lines.append("  Embedding model ranking (Recall@K | Latency P50):")
            for i, r in enumerate(embed_ranks[:4], 1):
                marker = "★" if i == 1 else " "
                lines.append(
                    f"  {marker} {i}. {(r['embedding_model'] or '—').split('/')[-1]:<25} "
                    f"recall={r['avg_recall']:.4f}  lat={r['avg_latency_ms']:.0f}ms"
                )
            lines.append("")

        return lines

    def _cross_dataset_insights(self, summaries: dict[str, dict]) -> list[str]:
        """Synthesise patterns from actual results across all datasets."""
        lines = ["─" * 72, "  CROSS-DATASET MEASUREMENT SUMMARY", "─" * 72, ""]

        # ── Strategy win counts ───────────────────────────────────────────────
        win_counts: dict[str, int] = {}
        for s in summaries.values():
            winner = s.get("recommendations", {}).get("best_retrieval_strategy", "")
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1

        most_wins = max(win_counts.values()) if win_counts else 0
        top_strats = [k for k, v in win_counts.items() if v == most_wins]
        lines += [
            f"  Most frequent winner: {', '.join(top_strats)}",
            f"  (wins on {most_wins} of {len(summaries)} datasets in this run)",
            "",
        ]

        # ── Data-driven decision framework ────────────────────────────────────
        # Build from actual results: which datasets have which character, what won?
        lines += ["  TOP-RANKED CONFIGURATION PER DATASET (sorted by Recall@K):",""]
        for ds_name, summary in sorted(summaries.items()):
            profile = get_profile(ds_name)
            recs = summary.get("recommendations", {})
            winner = recs.get("best_retrieval_strategy", "—")
            best_embed = (recs.get("best_embedding_model") or "—").split("/")[-1]
            best_reranker = recs.get("best_reranker", "none")
            best_lam = _best_lambda(summary)
            best_bm25w = _best_bm25_weight(summary)

            strat_ranks = summary.get("retrieval_strategy_ranking", [])
            winner_recall = strat_ranks[0]["avg_recall"] if strat_ranks else 0.0

            char = profile.character if profile else ds_name
            config_parts = [winner]
            if best_embed and winner in ("embeddings", "hybrid", "api_embeddings"):
                config_parts.append(f"embed={best_embed}")
            if best_bm25w and winner == "hybrid":
                config_parts.append(f"bm25w={best_bm25w}")
            if best_reranker != "none":
                config_parts.append(f"reranker={(best_reranker or 'none').split('/')[-1]}")
            if best_lam != "—":
                config_parts.append(f"λ={best_lam}")

            lines += [
                f"  If your data is like {ds_name.upper()} ({char}):",
                f"    → Use: {' | '.join(config_parts)}",
                f"    → Recall achieved: {winner_recall:.1%}",
                "",
            ]

        # ── Decay lambda summary from actual results ──────────────────────────
        lines += ["  DECAY LAMBDA RESULTS FROM THIS RUN:", ""]

        # Collect all (policy, lambda, recall) from decay rankings
        decay_data: list[tuple[str, float, float]] = []
        for summary in summaries.values():
            for r in summary.get("decay_policy_ranking", []):
                policy = r.get("decay_policy", "")
                lam = r.get("lambda_value", 0.0)
                recall = r.get("avg_recall", 0.0)
                if policy and lam is not None and recall > 0:
                    decay_data.append((policy, lam, recall))

        if decay_data:
            # Group by policy, show best lambda per policy
            by_policy: dict[str, list] = {}
            for policy, lam, recall in decay_data:
                by_policy.setdefault(policy, []).append((lam, recall))

            for policy, entries in sorted(by_policy.items()):
                entries.sort(key=lambda x: -x[1])  # best recall first
                best_lam, best_recall = entries[0]
                lines.append(
                    f"    {policy:<14}  best λ={best_lam:.3f}  recall={best_recall:.4f}  "
                    f"(tested {len(entries)} λ values)"
                )
            lines.append("")
        else:
            lines += [
                "    (Run --mode full to include Phase 5 decay sweep data)",
                "",
            ]

        lines += [
            "  NOTE: The archival floor (0.65) ensures memories older than 90 days",
            "  retain at least 65% weight — prevents very old gold from collapsing.",
            "",
        ]
        return lines

    def _bring_your_own_dataset(self) -> list[str]:
        return [
            "─" * 72,
            "  ADD YOUR OWN DATASET",
            "─" * 72,
            "",
            "  1. Implement DatasetAdapter:",
            "       cp benchmark/gold/adapters/synthetic_adapter.py benchmark/gold/adapters/my_adapter.py",
            "       # Edit load() to parse your data format",
            "       # Ensure query.day >= injection_day(gold_memory) for all queries",
            "",
            "  2. Convert to gold format:",
            "       python scripts/prepare_datasets.py --convert --source data/my_data.json",
            "",
            "  3. Run the full sweep:",
            "       python study_runner.py --gold-dataset data/my_data_gold.json --mode full",
            "",
            "  4. The report tells you:",
            "       - Which embedding model fits your data character",
            "       - Whether BM25 or semantic search dominates",
            "       - The optimal decay lambda for your time horizon",
            "",
            "  See CONTRIBUTING.md for the full adapter guide.",
            "",
            "=" * 72,
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_lambda(summary: dict) -> str:
    """Extract best lambda from actual decay ranking results.
    Returns '—' if Phase 5 was not run (no decay data available).
    Never falls back to a hardcoded default.
    """
    decay_ranks = summary.get("decay_policy_ranking", [])
    if not decay_ranks:
        return "—"
    try:
        best = decay_ranks[0]
        lam = best.get("lambda_value")
        return f"{lam:.3f}" if lam is not None else "—"
    except (IndexError, KeyError):
        return "—"


def _best_bm25_weight(summary: dict) -> str:
    """Extract best BM25 weight from actual hybrid sweep results.
    Returns '—' if Phase 3 was not run.
    """
    bm25_ranks = summary.get("bm25_weight_ranking", [])
    if not bm25_ranks:
        return "—"
    try:
        best = bm25_ranks[0]
        w = best.get("bm25_weight")
        return f"{w:.1f}" if w is not None else "—"
    except (IndexError, KeyError):
        return "—"


def _wrap(text: str, width: int = 68, prefix: str = "  ") -> str:
    """Soft-wrap text to width, preserving sentence structure."""
    if len(text) <= width:
        return text
    words = text.split()
    lines, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return ("\n" + prefix).join(lines)
