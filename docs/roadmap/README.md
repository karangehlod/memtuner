# AgentMemoryBench Roadmap

This directory is the canonical planning surface for AgentMemoryBench as a research product.

AgentMemoryBench is not only a software framework. It is a reproducible benchmark program intended to support benchmark releases, large-scale experiments, leaderboard generation, and a submission-quality ACL/EMNLP paper.

## Contents

- MASTER_ROADMAP.md
  The canonical product and research roadmap. This is the top-level source of truth for phase status, sequencing, dependencies, and completion policy.

- ../BENCHMARK_SPECIFICATION.md
  Defines AgentMemoryBench as a benchmark product, its scope, comparison units, and artifact contract.

- ../EVALUATION_PROTOCOL.md
  Defines the reproducibility, provenance, fairness, and artifact requirements for benchmark execution.

- ../METRIC_DEFINITIONS.md
  Defines the benchmark metrics and the interpretation rules used across reports and phases.

- ../EXPERIMENT_GUIDELINES.md
  Defines how benchmark experiments should be designed, reviewed, and archived.

- ../CONFIGURATION_TAXONOMY.md
  Maps the roadmap configuration taxonomy to the current implementation surfaces in the codebase.

- ../BENCHMARK_MODULE_TAXONOMY.md
  Maps benchmark-facing capability areas to the current package owners.

- ../ARTIFACT_SCHEMA.md
  Describes the current artifact surface emitted by the benchmark CLI, especially `benchmark analyze`.

- phases/
  One planning document per phase. Each phase plan defines the research question, deliverables, workstreams, dependencies, acceptance criteria, verification, and definition of done.

- stories/
  One backlog document per phase. Each story document breaks the phase into executable work items with explicit acceptance criteria and completion evidence.

## Planning Rules

- A phase can only be marked Completed when the acceptance criteria in its phase document are satisfied and the required artifacts exist.
- The master roadmap status table and the per-phase checklists must remain in sync.
- Existing documents such as docs/REMEDIATION_PLAN.md, docs/critical_plan.md, and docs/PHASE3_PLAN.md are historical context, not the canonical roadmap.

## Scope of the Current Roadmap

This roadmap covers all 15 phases required to evolve the project into a benchmark platform and publication vehicle:

- Foundation
- Benchmark protocol
- Dataset framework
- Memory architecture benchmark
- Retrieval benchmark
- Embedding benchmark
- Reranker benchmark
- Forgetting benchmark
- Large-scale benchmark
- Statistics
- Visualization
- Dashboard
- Benchmark suite
- Paper experiments
- Paper

## Recommended Usage

1. Read MASTER_ROADMAP.md for overall direction and phase status.
2. Use the relevant phase document to understand what "done" means.
3. Use the phase story document to execute or track work at story granularity.
4. Update both the master status table and the phase checklist when work advances.
