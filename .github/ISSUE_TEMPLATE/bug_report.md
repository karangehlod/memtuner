---
name: Bug report
about: Something is broken or producing wrong results
title: '[BUG] '
labels: bug
assignees: ''
---

## Describe the bug
A clear description of what is wrong.

## Steps to reproduce
```bash
memtuner study --gold-dataset data/input/locomo10.json --mode quick
```

## Expected behaviour
What you expected to happen.

## Actual behaviour
What actually happened (paste error output, wrong numbers, etc.).

## Environment
- OS: [e.g. macOS 14, Ubuntu 22.04, Windows 11]
- GPU: [e.g. NVIDIA RTX 3090, Apple M2, CPU-only]
- Python version: [e.g. 3.11.9]
- MemTuner version: [run `memtuner --version`]
- Install method: `pip install -e .` / `pip install memtuner`

## `memtuner doctor` output
<details>
<summary>Paste output here</summary>

```
(paste here)
```
</details>

## Additional context
Any other context (logs from `data/output/study_*/`, dataset used, etc.).
