# Phase 8: Large-Scale Benchmark

## Purpose

Measure how benchmarked systems behave as memory volume increases from small controlled settings to realistic large-scale regimes.

## Research Question

How do memory systems, retrieval pipelines, and forgetting policies scale with memory volume in terms of quality, latency, storage, and cost?

## Target Output

Scaling benchmark reports and scaling curves for selected benchmark systems.

## Scope

- Define scale tiers
- Benchmark selected systems at increasing memory volumes
- Measure latency, quality, storage, and cost scaling
- Identify breakdown points and scaling regimes

## Deliverables

- Scale tier specification
- Large-scale benchmark configs
- Scale benchmark reports
- Scaling curves and comparison figures

## Workstreams

- Scale tier design
- Experiment execution
- Scaling analysis
- Reporting

## Dependencies

- Stable systems from Phases 3 through 7
- Phase 1 protocol and Phase 2 dataset framework

## Acceptance Criteria

1. Large-scale experiments are executed across clearly defined memory-volume tiers.
2. Reports include scaling curves for latency, quality, storage, and cost.
3. At least one practically important scale target is reached and documented.
4. Failure modes and breakdown regimes are identified.
5. The benchmark supports defensible scaling claims in the paper.

## Verification

1. Execute selected systems at multiple increasing scale tiers.
2. Confirm scale metadata is preserved in artifacts.
3. Validate scaling report consistency and interpretability.
4. Review any extrapolations or scale limitations for rigor.

## Out of Scope

- Distributed systems research beyond benchmark needs
- Systems not stable enough for fair comparison

## Definition of Done

Phase 8 is complete when scaling behavior is measured with artifact-backed evidence and is suitable for paper claims.

## Completion Checklist

- [ ] Scale tiers defined
- [ ] Large-scale configs created
- [ ] Selected systems benchmarked across tiers
- [ ] Scaling curves generated
- [ ] Scale report reviewed
- [ ] Limitations documented
- [ ] Phase accepted and status updated in master roadmap
