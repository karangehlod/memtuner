"""Dataset personality profiles — explains WHY certain strategies win on each dataset.

Each profile captures the structural characteristics that determine which retrieval
strategy, embedding model, and decay policy will perform best. These profiles are
used by the narrative report generator to explain results to practitioners.

Key insight: no single strategy dominates all datasets. The right choice depends
entirely on your data's character — time horizon, query style, memory density,
and whether the text contains keywords or semantic paraphrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetProfile:
    """Structural characterisation of a benchmark dataset.

    These fields describe the *structure* of the dataset — facts that are
    independent of any benchmark run. They never include which strategy won
    or what recall numbers were achieved (those come from results).
    """

    display_name:    str
    character:       str            # one-line personality descriptor
    query_style:     str            # "conversational" | "direct_qa" | "reading_comprehension"
    memory_density:  str            # "sparse" | "medium" | "dense"
    time_span_days:  int | str      # 0 = no temporal dimension, int = typical span
    why_hard:        str            # the structural retrieval challenge (IR theory, not run data)
    strengths:       tuple[str, ...] = field(default_factory=tuple)  # challenge tags


# Registry: maps the normalised dataset name (lowercase, no suffix) → profile
DATASET_PROFILES: dict[str, DatasetProfile] = {

    "locomo": DatasetProfile(
        display_name    = "LoCoMo",
        character       = "Long-horizon multi-session episodic memory",
        query_style     = "conversational",
        memory_density  = "sparse",
        time_span_days  = 300,
        why_hard        = (
            "Gold memory may be 200+ days old. Recency bias kills performance — "
            "the most recently injected memory is NOT the correct one. "
            "Queries paraphrase conversation content; exact keywords rare."
        ),
        strengths       = ("temporal_reasoning", "multi_turn", "paraphrase_gap"),
    ),

    "longmemeval": DatasetProfile(
        display_name    = "LongMemEval",
        character       = "Temporal reasoning and knowledge updates",
        query_style     = "direct_qa",
        memory_density  = "medium",
        time_span_days  = 90,
        why_hard        = (
            "Facts change over time — an old memory and a new memory may contradict. "
            "The model must surface the NEWER fact, not the first occurrence. "
            "Keyword overlap often matches both old and new; semantic similarity "
            "helps distinguish by meaning drift."
        ),
        strengths       = ("knowledge_update", "temporal_facts", "contradiction_resolution"),
    ),

    "squad": DatasetProfile(
        display_name    = "SQuAD 2.0",
        character       = "Dense keyword-rich paragraphs with unanswerable questions",
        query_style     = "reading_comprehension",
        memory_density  = "dense",
        time_span_days  = 0,
        why_hard        = (
            "~33% of questions are unanswerable — the paragraph exists but the "
            "specific answer doesn't. A retriever that confidently returns noisy "
            "results hurts quality here. BM25 excels because paragraphs contain "
            "the exact query keywords; semantic search adds noise."
        ),
        strengths       = ("keyword_matching", "exact_span", "unanswerable_detection"),
    ),

    "coqa": DatasetProfile(
        display_name    = "CoQA",
        character       = "Conversational QA with coreference across turns",
        query_style     = "conversational",
        memory_density  = "medium",
        time_span_days  = 0,
        why_hard        = (
            "Pronouns and ellipsis: 'What did he say?' requires knowing who 'he' is "
            "from a prior turn. The story is the memory; individual turns are not "
            "self-contained. Reranking helps surface the turn that provides the "
            "antecedent when BM25 scores diverge."
        ),
        strengths       = ("coreference", "multi_turn_context", "abstractive_answers"),
    ),

    "synthetic": DatasetProfile(
        display_name    = "Synthetic",
        character       = "Controlled ground truth for ablation studies",
        query_style     = "varied",
        memory_density  = "configurable",
        time_span_days  = "configurable",
        why_hard        = (
            "Not hard — designed for isolation of single variables. "
            "Use to verify that decay λ changes actually move recall, "
            "that embedding quality matters when vocabulary is novel, "
            "or that the temporal constraint is enforced correctly."
        ),
        strengths       = ("ablation", "reproducible", "configurable_difficulty"),
    ),

    "hotpotqa": DatasetProfile(
        display_name    = "HotpotQA",
        character       = "Multi-hop reasoning across multiple documents",
        query_style     = "direct_qa",
        memory_density  = "dense",
        time_span_days  = 0,
        why_hard        = (
            "Answering requires chaining two facts from different documents. "
            "A single-hop retriever that finds document A misses document B, "
            "which is only relevant given the fact in A. Hybrid fusion helps "
            "bridge lexical and semantic gaps across the two hops."
        ),
        strengths       = ("multi_hop", "cross_document_reasoning", "bridge_entities"),
    ),
}


def get_profile(dataset_path_or_name: str) -> DatasetProfile | None:
    """Look up a dataset profile by path, filename, or normalised name.

    Accepts any of:
      - "data/locomo10.json"            → "locomo"
      - "longmemeval_oracle_gold.json"  → "longmemeval"
      - "locomo"                        → "locomo"
    """
    name = str(dataset_path_or_name).lower()
    # Strip path and extensions
    import re
    name = re.sub(r".*[/\\]", "", name)       # remove directory
    name = re.sub(r"(_gold|_oracle|_dev.*|_train.*)?\.json$", "", name)  # remove suffixes
    name = re.sub(r"\d+$", "", name)           # remove trailing numbers (locomo10 → locomo)
    name = name.rstrip("_-")

    # Direct match
    if name in DATASET_PROFILES:
        return DATASET_PROFILES[name]

    # Prefix match (e.g. "longmemeval_oracle" → "longmemeval")
    for key in DATASET_PROFILES:
        if name.startswith(key) or key.startswith(name):
            return DATASET_PROFILES[key]

    return None
