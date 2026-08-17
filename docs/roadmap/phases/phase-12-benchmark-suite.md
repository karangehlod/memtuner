# Phase 12: Benchmark Suite

## Purpose

Formalize the standard multi-dataset benchmark suite and define which datasets are considered part of the benchmark release.

## Research Question

Which datasets should comprise the standard benchmark suite, and how should they be normalized so cross-dataset comparisons remain meaningful?

## Target Output

Standard benchmark suite definition with normalized dataset inclusion rules.

## Scope

- Define benchmark suite inclusion rules
- Select standard datasets
- Normalize dataset-level reporting
- Define benchmark suite release criteria

## Deliverables

- Benchmark suite specification
- Dataset inclusion policy
- Cross-dataset normalization rules
- Suite-level benchmark report template

## Workstreams

- Inclusion-policy design
- Dataset selection
- Cross-dataset normalization
- Suite-level reporting

## Dependencies

- Phase 2 dataset framework
- Outputs from Phases 4 through 10

## Acceptance Criteria

1. The benchmark suite has explicit inclusion and exclusion rules.
2. Selected datasets are normalized through the adapter framework.
3. Suite-level reporting distinguishes within-dataset and cross-dataset claims.
4. The suite definition is stable enough to support benchmark releases.
5. The benchmark paper can clearly state which suite it evaluates.

## Verification

1. Produce a suite definition with selected datasets.
2. Confirm all included datasets pass framework validation.
3. Review cross-dataset normalization assumptions.
4. Generate a suite-level report example.

## Out of Scope

- Unlimited dataset expansion
- Dataset inclusion without adapter and validation support

## Definition of Done

Phase 12 is complete when the benchmark suite is explicitly defined and reproducible as a standard evaluation package.

## Completion Checklist

- [ ] Inclusion policy defined
- [ ] Standard datasets selected
- [ ] Normalization rules defined
- [ ] Suite report template defined
- [ ] Validation completed for included datasets
- [ ] Phase accepted and status updated in master roadmap
