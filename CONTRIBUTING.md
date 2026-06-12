# Contributing

Thanks for your interest. **This is a paper companion repo**, not a
general-purpose interpretability library. Setting expectations up front
saves everyone time.

## What we accept

- **Issues** about reproducibility: a script that doesn't run, a config
  that doesn't match the paper, a checkpoint download that 404s.
- **PRs that fix reproducibility bugs**: missing config entries, off-by-one
  preprocessing mismatches, broken paths.
- **Small documentation fixes**: typos, broken links, schema clarifications.

## What we'll likely close without merging

- New features, new factorizer architectures. (Note: this repo is
  metrics-only — it ships no figure-rendering code and imports no
  matplotlib. Every script computes and persists its metrics
  (CSV/JSON/`.pt`); the paper's figures are rendered separately from
  those committed metrics. PRs adding rendering code are out of scope.)
- Refactors to "modernize" the code (the structure is what it is; the paper
  is frozen).
- Renames of public API symbols (would break downstream re-runs).
- New scripts whose metrics don't map to a figure or table in the paper.
- Changes that break the small CI-safe test suite (identity correctness,
  schema invariants, layout guards).

Forks are welcome. If you build something on top, file an issue with a
pointer — we'll happily add a link from the README.

## Setting up locally

```bash
uv sync --extra dev          # installs the pinned ruff + pytest
uv run pytest -q             # a small, CI-safe suite, ~3s, no GPU
make lint                    # ruff check + format --check, same pinned ruff as CI
```

`make lint` and CI both lint through the locked dev environment
(`uv run --extra dev`), so the ruff version matches the one pinned in `uv.lock`
and in `.pre-commit-config.yaml` — local results and CI agree. (Avoid
`uv run --with ruff ...`, which pulls the latest ruff and can disagree with CI.)
CI runs the same pytest matrix on py3.11 and py3.12.

Optional pre-commit hooks (pinned ruff + whitespace / EOF / YAML / TOML checks):

```bash
uvx pre-commit install       # run the hooks automatically on every `git commit`
uvx pre-commit run --all-files
```

## Code style

- `ruff` is the single tool; config is in
  [`pyproject.toml`](pyproject.toml). Format with `ruff format` and lint
  with `ruff check`.
- Line length 120.
- Prefer existing patterns over introducing new abstractions.
- Comment shapes for non-obvious tensors (`# (batch, seq, d_model)`).

## Filing an issue

Please include:

1. The full command line and the script path.
2. Python + torch versions (`uv run python -c "import torch; print(torch.__version__)"`).
3. The exact error / stack trace.
4. Whether you're using the pretrained dictionaries from Hub or a
   freshly-trained one.

For checkpoint / dataset mismatch reports, please paste the manifest
(`manifest.json` from the extract or training run) — that captures the
revision pins that make the bug reproducible.
