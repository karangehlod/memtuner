# Phase 6: Reranker Benchmark

## Purpose

Benchmark rerankers as second-stage ranking components under controlled candidate budgets and retrieval pipelines.

## Research Question

Which rerankers improve retrieval quality most effectively under fixed candidate budgets, latency targets, and operational constraints?

## Target Output

Reranker benchmark leaderboard and reranker comparison report.

## Scope

- Evaluate rerankers under common first-stage retrieval inputs
- Compare quality gains versus cost and latency
- Standardize candidate-budget reporting
- Identify practical reranking defaults

## Deliverables

- Reranker registry
- Reranker benchmark configs
- Candidate-budget protocol
- Reranker benchmark report
- Reranker leaderboard

## Workstreams

- Reranker onboarding
- Candidate-budget protocol design
- Experiment execution
- Reporting

## Dependencies

- Phase 1 protocol
- Phase 2 dataset framework
- Phase 4 retrieval benchmark

## Acceptance Criteria

1. Rerankers are compared under fixed and documented candidate budgets.
2. Reports distinguish first-stage recall from second-stage ranking gains.
3. Cost and latency are reported alongside quality metrics.
4. Inputs and reranker settings are reproducible.
5. The report identifies cases where rerankers materially change system ranking.

## Verification

1. Execute reranker benchmarks for selected rerankers.
2. Confirm candidate-budget metadata is stored in outputs.
3. Validate schema consistency and report interpretability.
4. Review recommendations for fairness and deployment realism.

## Out of Scope

- End-to-end learned retriever-reranker training
- UI-first visualization work

## Definition of Done

Phase 6 is complete when rerankers can be compared fairly and their value is quantified under practical constraints.

## Completion Checklist

- [ ] Reranker registry defined
- [ ] Candidate-budget protocol defined
- [ ] Selected rerankers benchmarked
- [ ] Cost and latency tracked
- [ ] Reranker leaderboard generated
- [ ] Report reviewed
- [ ] Phase accepted and status updated in master roadmap
