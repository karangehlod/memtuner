# Phase 0: Foundation

## Purpose

Transform the repository from a general framework into a benchmark platform with a clear research-product architecture, planning surface, and documentation set.

## Research Question

What architecture, package boundaries, and benchmark-facing documentation are required to treat the project as a benchmark platform rather than a collection of components?

## Target Output

Stable Benchmark v1 architecture and benchmark-product planning package.

## Scope

- Establish benchmark-facing module taxonomy
- Create benchmark-product documentation set
- Refactor configuration model naming where necessary
- Define roadmap and completion governance
- Align repo structure with benchmark-product framing

## Deliverables

- Benchmark architecture map with benchmark-facing modules
- Canonical roadmap package
- Benchmark specification documentation set
- Benchmark module taxonomy and architecture alignment note
- Refined config model taxonomy:
  - ExperimentConfig
  - DatasetConfig
  - RetrievalConfig
  - EmbeddingConfig
  - DecayConfig
  - MemoryConfig
  - OutputConfig

## Workstreams

- Documentation architecture
- Package taxonomy alignment
- Configuration model alignment
- Planning and governance setup

## Dependencies

- Existing architecture rules in docs/architecture.md
- Existing benchmark package structure
- Existing remediation and phase docs as historical context

## Acceptance Criteria

1. A canonical benchmark-product roadmap exists under docs/roadmap.
2. The benchmark-product documentation set exists and is internally consistent.
3. Configuration taxonomy is documented and mapped to implementation surfaces.
4. The benchmark-facing architecture is described in a way that reviewers and contributors can understand quickly.
5. The phase completion policy is defined and referenced by the roadmap and phase docs.

## Verification

1. Review architecture and roadmap docs for consistency with docs/architecture.md.
2. Confirm the benchmark-facing module taxonomy covers protocol, datasets, retrieval, embeddings, rerankers, forgetting, statistics, visualization, reporting, and orchestration.
3. Confirm configuration model names and responsibilities are documented.

## Out of Scope

- Adding new benchmark methods
- Running benchmark experiments
- Building dashboards or leaderboards

## Definition of Done

Phase 0 is complete when the repo has a benchmark-product planning and architecture surface that can guide all later phases without ambiguity.

## Completion Checklist

- [x] Master roadmap exists
- [x] Benchmark specification doc exists
- [x] Evaluation protocol doc exists
- [x] Metric definitions doc exists
- [x] Experiment guidelines doc exists
- [x] Benchmark module taxonomy doc exists
- [x] Contribution guide update exists
- [x] Configuration taxonomy documented
- [x] Architecture alignment reviewed
- [ ] Phase accepted and status updated in master roadmap
