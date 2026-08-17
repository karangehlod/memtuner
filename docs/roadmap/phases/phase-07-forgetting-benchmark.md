# Phase 7: Forgetting Benchmark

## Purpose

Evaluate forgetting and retention policies for long-lived agent memory systems under bounded resource conditions.

## Research Question

How do forgetting policies influence answer quality, temporal correctness, hallucination risk, storage size, and system efficiency over time?

## Target Output

Forgetting benchmark leaderboard and policy comparison report.

## Scope

- Compare retention policies
- Compare decay and eviction policies
- Measure performance under constrained memory budgets
- Assess temporal correctness and hallucination behavior

## Deliverables

- Forgetting policy taxonomy
- Forgetting benchmark configs
- Forgetting metrics specification
- Forgetting benchmark report
- Forgetting leaderboard

## Workstreams

- Policy taxonomy and selection
- Metrics definition
- Experiment execution
- Reporting

## Dependencies

- Phase 1 protocol
- Phase 2 dataset framework
- Stable memory and retrieval surfaces from earlier phases

## Acceptance Criteria

1. Multiple forgetting policy families are benchmarked under controlled memory budgets.
2. Reports include accuracy, temporal correctness, storage size, and efficiency metrics.
3. Hallucination or stale-memory behavior is measured where task design allows.
4. Policy settings and budget constraints are documented and reproducible.
5. The benchmark identifies useful policy trade-offs rather than a single universal winner.

## Verification

1. Execute forgetting benchmarks under at least two budget settings.
2. Confirm policy metadata and budget metadata are persisted.
3. Validate metric outputs and benchmark report consistency.
4. Review whether conclusions are supported by the task design.

## Out of Scope

- Learned retention-policy training
- Human annotation pipelines unless explicitly added later

## Definition of Done

Phase 7 is complete when forgetting policy trade-offs are benchmarked in a reproducible and interpretable way.

## Completion Checklist

- [ ] Policy taxonomy defined
- [ ] Metrics specification finalized
- [ ] Budget settings defined
- [ ] Selected policies benchmarked
- [ ] Forgetting leaderboard generated
- [ ] Report reviewed
- [ ] Phase accepted and status updated in master roadmap
