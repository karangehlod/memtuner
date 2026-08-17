# Benchmark Specification

## Purpose

This document defines AgentMemoryBench as a benchmark product.

AgentMemoryBench evaluates memory systems, retrieval strategies, embedding models, rerankers, and forgetting policies for agent workloads under a reproducible protocol. The benchmark is intended to support internal engineering decisions, public benchmark releases, and a submission-quality research paper.

## Benchmark Identity

AgentMemoryBench is:

- A reproducible benchmark framework
- A benchmark protocol and artifact model
- A collection of benchmark adapters, baselines, and reporting surfaces
- A research-product platform for controlled memory-system comparison

AgentMemoryBench is not:

- A production memory database
- A standalone vector database
- A general-purpose agent framework
- A substitute for task-specific product evaluation

## Benchmark Questions

The benchmark is designed to answer the following classes of questions:

1. Which memory architectures perform best under a shared benchmark contract?
2. Which retrieval strategies give the strongest quality-efficiency trade-offs?
3. Which embedding models and rerankers provide the most practical improvements?
4. How do forgetting policies affect long-horizon quality, temporal correctness, and cost?
5. How do these systems scale as memory volume grows?

## Benchmark Units of Comparison

The benchmark compares systems through explicit, versioned configurations.

A benchmark system may vary along these axes:

- Dataset
- Memory module selection
- Retrieval strategy
- Retrieval parameters
- Embedding model
- Reranker
- Decay policy
- Retention or pruning policy
- Answering or judge settings when enabled

The benchmark does not treat informal implementation differences as valid comparison units. If a difference matters, it must appear in the stored configuration and run metadata.

## Benchmark Scope

### In Scope

- Reproducible offline benchmark execution
- Dataset adapters with validation and statistics
- Retrieval, embedding, reranker, and forgetting benchmarks
- Quality, temporal, efficiency, and cost-aware analysis
- Deterministic or protocol-controlled experiment execution
- Benchmark reports, figures, and leaderboard outputs

### Out of Scope

- Real-time production serving claims
- Human subject evaluation unless explicitly added
- Untracked ad hoc experiments used as benchmark evidence
- Claims unsupported by archived artifacts

## Benchmark Inputs

A valid benchmark run requires the following input classes:

- Dataset input
  A dataset or pack resolved through the dataset framework

- Experiment configuration
  A versioned configuration that identifies the benchmark dimensions being tested

- Runtime environment
  Model endpoints, provider settings, and machine-specific runtime details

- Benchmark code revision
  The repository state or version used for execution

## Benchmark Outputs

Every benchmark run should produce protocol-compliant artifacts that make the run auditable and reproducible.

Required output categories:

- Configuration artifacts
- Run metadata artifacts
- Environment and provenance artifacts
- Metric outputs
- Reporting outputs
- Optional visualization outputs

## Core Benchmark Dimensions

### Datasets

Datasets must enter the benchmark through a stable adapter interface. Each dataset must support validation, fingerprinting, and standardized statistics.

### Memory Systems

Memory systems are compared through benchmark-compatible interfaces and documented assumptions.

### Retrieval

Retrieval strategies must be evaluated with explicit settings, candidate budgets, and result metadata.

### Embeddings and Rerankers

Embedding and reranker comparisons are valid only when model identity, revision, and runtime conditions are captured.

### Forgetting Policies

Retention, decay, and eviction strategies must be benchmarked under explicit budget assumptions.

## Benchmark Claims Standard

A claim may be included in a benchmark report or paper only if all of the following are true:

1. The claim maps to one or more benchmark artifacts.
2. The compared systems were run under the same benchmark protocol.
3. The configuration differences are explicit and auditable.
4. The report discloses important exclusions, caveats, and limitations.
5. The claim can be regenerated from stored artifacts.

## Benchmark Artifact Contract

Each benchmarked result must be attributable to:

- A dataset fingerprint
- A versioned configuration
- A seed or deterministic execution setup
- A code revision
- A recorded runtime environment
- A known output location

## Phase Alignment

This specification is the benchmark-product anchor for the roadmap.

- Phase 0 defines the benchmark-product planning and documentation surface.
- Phase 1 defines the execution protocol and reproducibility contract.
- Later phases add benchmark families, result layers, and publication artifacts.

## Definition of Readiness

AgentMemoryBench is ready to operate as a benchmark product when:

- The benchmark specification is stable
- The evaluation protocol is stable
- Metric definitions are documented
- Experiment execution guidelines are documented
- The roadmap and completion policy are active and maintained
