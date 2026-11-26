# Security Policy

## Supported versions

MemTuner is pre-1.0. Security fixes are made on the `main` branch and included in the next release.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report privately through GitHub: <https://github.com/karangehlod/memtuner/security/advisories/new>.
If that is not possible, email karan.gehlod@gmail.com with a description, reproduction steps, and the affected version or commit.

You can expect an acknowledgement within 7 days. Fix timelines depend on severity, and you will be credited in the release notes unless you prefer otherwise.

## Scope notes

- MemTuner reads API keys (for example for LLM judges and embedding providers) from environment variables or `.env`. Never commit real keys; use `.env.example` as the template.
- Datasets are downloaded from third-party sources and are not redistributed by this project.
