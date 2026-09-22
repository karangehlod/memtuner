#!/usr/bin/env python3
"""MemTuner Arena Runner — one protocol, every memory system.

Runs memory systems (MemTuner's native stack and external backends like Mem0)
through the identical benchmark protocol: ingest the dataset's timeline,
answer the holdout queries, score answers with the pinned LLM judge.

Unlike study_runner.py this runs NO tuning phases — sweeping BM25 weights
against an external system is meaningless. One cell per system × dataset ×
seed: ingest → query → judge.

Ground rules enforced here (docs/ARENA_PLAN.md):
- Judge required. Judged answer accuracy is the only cross-system metric;
  a run without a judge configured is refused (override: --allow-no-judge,
  which produces retrieval-only numbers that must not be published).
- Holdout on by default (--test-holdout-fraction 0.20).
- Horizon = each dataset's natural span (never padded, never truncated).
- Native baseline runs the FULL system (all long-term stores), because
  external systems ingest all memory types too.

Usage:
    python scripts/arena_runner.py \\
        --systems native mem0 \\
        --gold-dataset data/input/locomo10.json \\
        --seeds 42 123 456 \\
        --ollama-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from benchmark.workload.study_matrix import DecaySpec, StudyCell  # noqa: E402

KNOWN_EXTERNAL_BACKENDS = {"mem0", "graphiti", "zep"}  # letta: ARENA_PLAN task 1.5
# zep = Zep CLOUD (paid, dormant per ground rule 9); graphiti is the free slot.


def build_arena_cells(
    systems: list[str],
    seeds: list[int],
    test_holdout_fraction: float,
    baseline: dict | None = None,
) -> list[StudyCell]:
    """One flat cell per system × seed. Pure — unit-tested directly.

    Args:
        systems: "native" and/or external backend names (e.g. "mem0").
        seeds: One cell per seed; external nondeterminism (LLM ingest) makes
            per-seed spread a first-class arena number.
        test_holdout_fraction: Trailing-days holdout (0.20 for reported runs).
        baseline: Native-stack config (from configs/arena/baseline_rag.yaml):
            keys retrieval_strategy, embedding_model, embedding_backend,
            bm25_weight. None → documented placeholder (bm25l).
    """
    baseline = baseline or {}
    no_decay = DecaySpec(policy="none", lambda_value=0.0, pruning_threshold=0.15)
    cells: list[StudyCell] = []
    for system in systems:
        for seed in seeds:
            if system == "native":
                cells.append(StudyCell(
                    memory_type="all",  # full system — fairness vs external
                    retrieval_strategy=baseline.get("retrieval_strategy", "bm25l"),
                    decay=no_decay,
                    workload_profile="arena",
                    embedding_model=baseline.get("embedding_model", ""),
                    embedding_backend=baseline.get("embedding_backend", "none"),
                    bm25_weight=float(baseline.get("bm25_weight", 1.0)),
                    test_holdout_fraction=test_holdout_fraction,
                    seed=seed,
                    study_phase="arena",
                ))
            else:
                cells.append(StudyCell(
                    memory_type="all",
                    retrieval_strategy="external",  # sequential execution path
                    decay=no_decay,
                    workload_profile="arena",
                    embedding_model="",
                    embedding_backend="none",
                    bm25_weight=0.0,
                    test_holdout_fraction=test_holdout_fraction,
                    seed=seed,
                    study_phase="arena",
                    backend=system,
                ))
    return cells


def _load_baseline() -> tuple[dict | None, str]:
    """Load the frozen native baseline config, if Phase 0 has produced one."""
    path = project_root / "configs" / "arena" / "baseline_rag.yaml"
    if not path.exists():
        return None, (
            "[warn] configs/arena/baseline_rag.yaml not found — native baseline "
            "falls back to plain bm25l. Freeze the Phase-0 winner before any "
            "reported arena run (ARENA_PLAN task 1.6)."
        )
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("baseline", data), f"  Native baseline: {path.name}"


def _resolve_judge(args) -> tuple[str, str]:
    """Resolve (model, endpoint) from CLI > study_defaults.yaml > .env."""
    model = args.judge_model or ""
    if not model:
        try:
            import yaml
            with open(project_root / "configs" / "study_defaults.yaml", encoding="utf-8") as f:
                model = str((yaml.safe_load(f) or {}).get("judge", {}).get("model", "") or "")
        except OSError:
            model = ""
    endpoint = args.ollama_url or os.environ.get("BENCHMARK_JUDGE_BASE_URL", "")
    return model, endpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="MemTuner Arena Runner")
    parser.add_argument("--systems", nargs="+", required=True,
                        help=f"Systems to run: native, {', '.join(sorted(KNOWN_EXTERNAL_BACKENDS))}")
    parser.add_argument("--gold-dataset", nargs="+", required=True,
                        help="Gold dataset path(s)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="One arena cell per seed (external ingest is nondeterministic)")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--test-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--ollama-url", default="",
                        help="Judge endpoint (OpenAI-compatible)")
    parser.add_argument("--judge-model", default="",
                        help="Judge model override (default: pinned in study_defaults.yaml)")
    parser.add_argument("--allow-no-judge", action="store_true",
                        help="Run without a judge — retrieval-only numbers, NOT publishable")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    unknown = [s for s in args.systems if s != "native" and s not in KNOWN_EXTERNAL_BACKENDS]
    if unknown:
        parser.error(f"unknown system(s): {unknown}. Known: native, {sorted(KNOWN_EXTERNAL_BACKENDS)}")
    if "zep" in args.systems:
        print("  [notice] zep = Zep CLOUD, a paid managed service (needs ZEP_API_KEY; "
              "vendor-side billing applies). It is outside the zero-cost arena "
              "(ARENA_PLAN ground rule 9); results must disclose the asymmetry.")

    # ── Judge enforcement (ARENA_PLAN ground rule 6) ─────────────────────────
    judge_model, judge_endpoint = _resolve_judge(args)
    if judge_model and judge_endpoint:
        os.environ["BENCHMARK_JUDGE_MODEL"] = judge_model
        os.environ.setdefault("BENCHMARK_JUDGE_BASE_URL", judge_endpoint)
        os.environ.setdefault("BENCHMARK_LLM_BASE_URL", judge_endpoint)
        print(f"  LLM judge: {judge_model} @ {judge_endpoint}")
    elif args.allow_no_judge:
        print("  [warn] Running WITHOUT a judge — cross-system comparison is "
              "undefined for these results. Do not publish them.")
    else:
        print("  ERROR: no judge configured. Arena numbers are judged answer "
              "accuracy; set --ollama-url (and optionally --judge-model), or "
              "BENCHMARK_JUDGE_BASE_URL in .env. Override with --allow-no-judge.")
        return 2

    baseline, baseline_note = _load_baseline() if "native" in args.systems else (None, "")
    if baseline_note:
        print(baseline_note)

    from benchmark.workload.study_aggregator import StudyAggregator, StudyReporter
    from benchmark.workload.study_scheduler import StudyScheduler

    exit_code = 0
    for dataset in args.gold_dataset:
        gold_path = Path(dataset)
        print(f"\n{'─' * 60}\n  ARENA: {gold_path.name}\n{'─' * 60}")

        # Preflight + natural-span horizon (never padded — see loader fix).
        try:
            from benchmark.gold.oracle import GoldOracle
            ds = GoldOracle().load_dataset(gold_path, scenario_name="arena-preflight")
        except Exception as e:
            print(f"  [skip] cannot load {gold_path.name}: {e}")
            exit_code = 1
            continue
        horizon = max(
            max((q.day for q in ds.queries), default=0),
            max((d.day for d in ds.events), default=0),
        ) + 1

        cells = build_arena_cells(args.systems, args.seeds, args.test_holdout_fraction, baseline)
        run_id = uuid.uuid4().hex[:12]
        out_dir = Path(args.output_dir) / f"study_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        scheduler = StudyScheduler(max_workers=args.workers, output_dir=str(out_dir), run_id=run_id)
        results = scheduler.run(cells, str(gold_path), evaluation_horizon=horizon)

        failed = [r for r in results if not r.success]
        if failed:
            print(f"  [error] {len(failed)}/{len(results)} cells failed:")
            for r in failed[:5]:
                print(f"    {r.backend or 'native'} seed={r.seed}: "
                      f"{(r.error_message or '?').strip().splitlines()[-1]}")
            exit_code = 1

        agg = StudyAggregator(results, dataset_name=gold_path.stem)
        reporter = StudyReporter(out_dir)
        reporter.write_all(
            agg, run_id, skip_plots=args.no_plots, all_results=results,
            run_metadata={
                "arena": True,
                "systems": args.systems,
                "seeds": args.seeds,
                "judge_model": judge_model or "DISABLED",
                "test_holdout_fraction": args.test_holdout_fraction,
                "evaluation_horizon": horizon,
            },
        )

        # ── Per-system summary (judge first — it is the headline metric) ────
        print(f"\n  {'system':10} {'seeds':>5} {'judge':>8} {'judge±':>7} "
              f"{'recall@k*':>10} {'p50 ms':>8} {'fail':>5}")
        for system in args.systems:
            rows = [r for r in results
                    if (r.backend if r.backend != "native" else "native") == system]
            ok = [r for r in rows if r.success]
            judged = [r.llm_judge_score for r in ok if r.llm_judge_score is not None]
            recalls = [r.recall_at_k for r in ok]
            lat = [r.latency_p50_ms for r in ok]
            import statistics as st
            j_mean = f"{st.mean(judged):.4f}" if judged else "—"
            j_std = f"{st.stdev(judged):.4f}" if len(judged) > 1 else "—"
            r_mean = f"{st.mean(recalls):.4f}" if recalls else "—"
            l_mean = f"{st.mean(lat):.1f}" if lat else "—"
            print(f"  {system:10} {len(ok):>5} {j_mean:>8} {j_std:>7} "
                  f"{r_mean:>10} {l_mean:>8} {len(rows) - len(ok):>5}")
        print("  * recall@k for external systems is a provenance-mapped LOWER "
              "BOUND — compare systems on the judge column only.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
