# AgentMemoryBench Master Roadmap

## Vision

AgentMemoryBench is a reproducible benchmark framework for evaluating long-term memory architectures, retrieval strategies, embedding models, rerankers, and forgetting policies for LLM agents.

The goal is to turn the current repository into a benchmark platform that can support benchmark releases, reproducible experiments, public leaderboards, and a submission-ready ACL/EMNLP benchmark paper.

## Research Framing

Each phase must answer a specific research or benchmark-design question and produce a tangible artifact. Tangible artifacts can include code, experiments, figures, benchmark reports, leaderboards, statistical analyses, documentation, or paper sections.

The benchmark should become influential because its methodology is trusted, not because it includes the largest number of methods.

## Final Deliverables

- Open-source benchmark framework
- Benchmark protocol specification
- Multiple dataset adapters
- Embedding benchmark
- Reranker benchmark
- Retrieval benchmark
- Forgetting benchmark
- Leaderboard
- Experiment dashboard
- Statistical analysis toolkit
- Reproducibility package
- Benchmark paper

## First-Paper Scope Guidance

The full roadmap spans 15 phases. For the first benchmark paper, the recommended priority is Phases 0 through 10 with careful baseline selection and rigorous protocol enforcement.

The following topics are important but should remain future work for the first paper unless they become necessary for a specific comparison:

- Graph memory
- Reflection memory
- Learned write policies
- Open-ended learned memory controllers

## Phase Overview

| Phase | Title | Primary Research Question | Main Artifact | Status |
|------|-------|---------------------------|---------------|--------|
| 0 | Foundation | What architecture and documentation are required to treat the repo as a benchmark platform? | Stable benchmark architecture, benchmark docs, and module taxonomy | Ready for Review |
| 1 | Benchmark Protocol | What protocol guarantees reproducible and fair benchmark execution? | Protocol spec plus reproducibility artifacts | In Progress (sample run verified under Python 3.14; Python 3.13 runtime still failing during memory-type query) |
| 2 | Dataset Framework | How can datasets become plug-and-play with standardized validation and statistics? | Dataset adapter framework and dataset reports | Not Started |
| 3 | Memory Architecture Benchmark | Which mature memory architectures can be compared fairly under a common interface? | Memory leaderboard | Not Started |
| 4 | Retrieval Benchmark | How do sparse, dense, hybrid, and multi-stage retrieval systems compare under common metrics? | Retrieval leaderboard | Not Started |
| 5 | Embedding Benchmark | Which embedding models offer the best accuracy-efficiency trade-offs? | Embedding leaderboard | Not Started |
| 6 | Reranker Benchmark | Which rerankers improve retrieval quality under controlled candidate budgets? | Reranker leaderboard | Not Started |
| 7 | Forgetting Benchmark | How do forgetting policies affect accuracy, latency, size, and hallucination risk? | Forgetting leaderboard | Not Started |
| 8 | Large-Scale Benchmark | How do systems scale from 10K to 5M memories? | Scaling curves and scale reports | Not Started |
| 9 | Statistics Module | What statistical procedures are needed to make benchmark claims defensible? | Statistical analysis toolkit | Not Started |
| 10 | Visualization | What figures are needed for benchmark interpretation and publication? | Publication-ready figures | Not Started |
| 11 | Dashboard | How can users interactively explore benchmark results and compare systems? | Interactive benchmark dashboard | Not Started |
| 12 | Benchmark Suite | Which datasets should be included in the standard benchmark suite? | Multi-dataset suite | Not Started |
| 13 | Paper Experiments | What experiment matrix is required for the benchmark paper? | Full experiment run corpus | Not Started |
| 14 | Paper | How should the benchmark be presented to maximize clarity, rigor, and adoption? | Submission-ready paper package | Not Started |

## Dependency Map

The roadmap is sequential at the phase level, but some workstreams can overlap.

- Phase 0 blocks all later phases because it establishes architecture, planning, and document contracts.
- Phase 1 blocks all later benchmark claims because protocol consistency is foundational.
- Phase 2 should be completed before broad multi-dataset experiments.
- Phase 3, Phase 4, Phase 5, Phase 6, and Phase 7 can proceed as separate benchmark workstreams once Phase 1 and Phase 2 are stable.
- Phase 8 depends on stable baselines from Phases 3 through 7.
- Phase 9 and Phase 10 should start as soon as result schemas stabilize, but they should be validated against outputs from Phases 4 through 8.
- Phase 11 depends on stable outputs from Phases 1 through 10.
- Phase 12 formalizes the benchmark suite once the dataset framework is stable.
- Phase 13 depends on stable experimental infrastructure, benchmark suites, and result schemas.
- Phase 14 depends on validated outputs from Phases 1 through 13.

## Required Artifact Types by Phase

- Code artifacts
  Benchmark modules, adapters, evaluators, experiment runners, dashboards, scheduling tools

- Experiment artifacts
  Run manifests, configs, environment snapshots, cached outputs, statistical summaries

- Reporting artifacts
  Leaderboards, benchmark reports, comparison tables, figures, dashboards

- Documentation artifacts
  Specifications, guidelines, methodology notes, reproducibility docs, paper content

## Phase Completion Policy

A phase may be marked Completed only if all of the following are true:

1. The phase research question has been answered with evidence.
2. The phase deliverables listed in the phase document exist.
3. The required code, experiment, and documentation artifacts exist in the repository or designated artifact storage.
4. All mandatory stories for the phase meet their acceptance criteria.
5. The verification section in the phase document has been executed and documented.
6. Any deferred work is explicitly moved to future work or a later phase.

## Status Meanings

- Not Started
  No phase work has begun beyond exploratory notes.

- In Progress
  Work has started, but mandatory acceptance criteria are not yet met.

- Blocked
  Work cannot proceed because an upstream dependency, infrastructure dependency, or critical decision is unresolved.

- Ready for Review
  Acceptance criteria appear satisfied, but completion evidence is still being reviewed.

- Completed
  Acceptance criteria are satisfied, artifacts exist, and the phase checklist is complete.

## Roadmap Governance

- MASTER_ROADMAP.md is the canonical roadmap.
- Each phase document is the authoritative definition of done for that phase.
- Each story document is the authoritative backlog for that phase.
- Status updates must keep the master roadmap and the phase checklist in sync.
- Historical plans remain useful context but do not override this roadmap.

## Immediate Priorities

1. Phase 0: formalize the benchmark-product architecture and roadmap package.
2. Phase 1: freeze the benchmark protocol and reproducibility metadata model.
3. Phase 2: standardize dataset adapters and dataset statistics.
4. Phases 4, 5, 6, and 7: build the core benchmark comparisons that most directly support the paper.

## Success Criteria by Phase

- Phase 0
  Stable architecture and benchmark documentation

- Phase 1
  Reproducible, versioned experiment protocol

- Phase 2
  Plug-and-play dataset framework with validation and statistics

- Phase 3
  Common memory architecture interface and fair baseline implementations

- Phase 4
  Retrieval benchmark with sparse, dense, and hybrid baselines

- Phase 5
  Embedding leaderboard with approximately 20 to 30 modern models

- Phase 6
  Reranker leaderboard with fair candidate-budget comparisons

- Phase 7
  Forgetting benchmark with multiple policy families and benchmark metrics

- Phase 8
  Scalability results up to at least 1M memories, with clear rationale for any larger target

- Phase 9
  Statistical significance and confidence intervals integrated into benchmark reports

- Phase 10
  Publication-ready figures and leaderboards

- Phase 11
  Interactive dashboard for exploration and comparison

- Phase 12
  Standardized multi-dataset benchmark suite with normalized adapters

- Phase 13
  Reproducible experiment matrix with resumable execution and cached results

- Phase 14
  Submission-ready benchmark paper with reproducibility appendix
