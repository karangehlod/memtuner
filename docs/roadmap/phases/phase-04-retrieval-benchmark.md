# Phase 4: Retrieval Benchmark

## Purpose

Evaluate sparse, dense, hybrid, and multi-stage retrieval strategies under shared metrics and retrieval budgets.

## Research Question

How do major retrieval strategy families compare on recall, ranking quality, latency, cost, and downstream benchmark performance?

## Target Output

Retrieval benchmark leaderboard and benchmark report.

## Scope

- Compare sparse retrieval
- Compare dense retrieval
- Compare hybrid retrieval
- Compare multi-stage retrieval where appropriate
- Standardize retrieval metrics and retrieval settings

## Deliverables

- Retrieval strategy registry
- Retrieval benchmark configs
- Retrieval metrics specification
- Retrieval benchmark reports
- Retrieval leaderboard

## Workstreams

- Strategy onboarding
- Retrieval metrics and evaluation
- Benchmark execution
- Reporting

## Dependencies

- Phase 1 protocol
- Phase 2 dataset framework
- Phase 5 and Phase 6 may inform later refinements

## Acceptance Criteria

1. Sparse, dense, and hybrid retrieval baselines are benchmarked.
2. Metrics include retrieval quality and system-level efficiency.
3. Retrieval settings are documented and reproducible.
4. Benchmark outputs identify the best retrieval families under different constraints.
5. The report clearly distinguishes first-stage retrieval from reranking effects.

## Verification

1. Execute retrieval benchmark runs for all selected families.
2. Validate result schema consistency.
3. Confirm candidate counts, cutoff values, and budget parameters are preserved in metadata.
4. Review report interpretation for fairness and clarity.

## Out of Scope

- Learned retrieval policies requiring new training pipelines
- Frontend dashboard implementation

## Definition of Done

Phase 4 is complete when retrieval families are benchmarked under a clear, reproducible, and interpretable protocol.

## Completion Checklist

- [ ] Sparse baselines included
- [ ] Dense baselines included
- [ ] Hybrid baselines included
- [ ] Metrics specification finalized
- [ ] Retrieval leaderboard generated
- [ ] Retrieval report reviewed
- [ ] Phase accepted and status updated in master roadmap
