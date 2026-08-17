# Experiment Guidelines

## Purpose

This document defines how to design, run, review, and archive experiments for AgentMemoryBench.

The goal is to keep experiments reproducible, interpretable, and usable in benchmark reports and paper artifacts.

## Experiment Design Principles

A valid benchmark experiment should be:

- Explicit about what changes between systems
- Controlled on all other important dimensions
- Grounded in a stored configuration
- Traceable to a dataset identity and protocol version
- Suitable for later reproduction

## Recommended Experiment Structure

Each experiment should answer one main question.

Examples:

- Which retrieval strategy performs best on a target dataset under fixed settings?
- Which embedding model gives the best quality-latency trade-off?
- How does a forgetting policy behave under different storage budgets?
- How does quality change as memory scale increases?

Avoid bundling too many independent questions into one experiment run family.

## Controlled Variables

When comparing systems, hold constant all non-target dimensions unless the change is intentional and documented.

Typical controlled variables include:

- Dataset and split
- Protocol version
- Seed or deterministic mode
- Query budget or candidate budget
- Evaluation task definition
- Reporting schema

## Recorded Variables

Every experiment should record the variables that define the comparison, including:

- Retrieval strategy and settings
- Memory policy settings
- Model identifiers
- Runtime environment
- Dataset fingerprint
- Repository revision
- Output location

## Experiment Naming and Grouping

Experiment families should use stable, descriptive names so results can be grouped and compared without ambiguity.

A good experiment family name reflects:

- Dataset or suite
- Benchmark family
- Key varying dimension
- Important constraints or budget assumptions

## Reproducibility Rules

An experiment may be used for reporting only if:

- The effective config is stored
- The dataset identity is known
- The environment context is recorded
- The outputs can be mapped back to the intended question
- Important deviations or failures are disclosed

## Ablations

Ablations should isolate one explanatory factor at a time whenever possible.

Examples:

- Same retriever, different embedding model
- Same retrieval pipeline, reranker enabled versus disabled
- Same memory system, different decay lambda
- Same system, different storage budgets

Ablation reports should explain why the factor matters.

## Failure and Partial Results

Failed or incomplete experiments are useful for diagnosis, but they should be separated from benchmark evidence.

When an experiment fails:

- Preserve manifests and partial artifacts if possible
- Record the failure reason
- Distinguish between infrastructure failure and benchmark-result failure
- Do not merge partial evidence into final comparisons without disclosure

## Review Before Publication

Before including an experiment in a report, leaderboard, or paper, verify:

- The experiment question is still the right one
- The compared systems were run fairly
- The artifacts are complete enough to support the claim
- Known limitations are disclosed
- The metrics used match the question being asked

## Archival Guidance

Important experiment families should retain:

- Configs
- Metadata
- Reports
- Figure inputs or figure artifacts
- Manifest or index files
- Notes on exclusions and caveats

## Relationship to the Roadmap

These guidelines satisfy a Phase 0 documentation requirement and support later execution-heavy phases, especially benchmark families, scale experiments, and paper experiments.
