<!--
This is a frozen paper companion repo. Please read CONTRIBUTING.md before
opening a PR.

Accepted: reproducibility fixes (missing config entries, preprocessing
mismatches, broken paths) and small documentation fixes (typos, broken
links, schema clarifications).

Likely closed without merging: new features, new factorizer architectures,
figure-rendering code (the repo is metrics-only), refactors to "modernize"
the code, public-API renames, and scripts whose metrics don't map to a
paper figure or table.
-->

## What this fixes

<!-- One or two sentences. Link the issue it addresses, if any. -->

## Scope

- [ ] This is a reproducibility fix or a small documentation fix (per CONTRIBUTING.md).
- [ ] It does not add features, new factorizers, or figure-rendering code.
- [ ] It does not rename public API symbols or break downstream re-runs.

## Checks

- [ ] `uv run pytest -q` passes (the small CI-safe suite).
- [ ] `uv run ruff check src/ scripts/ tests/` is clean.
- [ ] `uv run ruff format --check src/ scripts/ tests/` is clean.
