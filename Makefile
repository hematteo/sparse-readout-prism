.DEFAULT_GOAL := help

.PHONY: help install test smoke lint clean dist-clean package

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## uv sync --extra dev
	uv sync --extra dev

test: ## uv run --extra dev pytest -q
	uv run --extra dev pytest -q

smoke: ## CPU-only synthetic smoke run
	uv run python scripts/train/train_readout_sae_from_config.py --config configs/smoke.yaml

lint: ## ruff check + format --check (same as CI)
	uv run --extra dev ruff check src/ scripts/ tests/
	uv run --extra dev ruff format --check src/ scripts/ tests/

clean: ## remove .pytest_cache .ruff_cache .mypy_cache __pycache__
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ ! -path './.venv/*' -prune -exec rm -rf {} +

dist-clean: clean ## clean + drop build artifacts (dist/, *.egg-info)
	rm -rf dist build *.egg-info src/*.egg-info

VERSION := $(shell grep -m1 '^version' pyproject.toml | cut -d'"' -f2)

package: ## build a release tarball of TRACKED files only (via git archive — no caches/secrets/venv)
	@command -v git >/dev/null 2>&1 || { echo "error: git is required for 'make package'"; exit 1; }
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "error: not a git repo — run 'git init && git add -A && git commit' first"; exit 1; }
	@test -z "$$(git status --porcelain)" || echo "warning: uncommitted changes are NOT in the archive (git archive packages HEAD only)"
	git archive --format=tar.gz --prefix=sparse-readout-prism-$(VERSION)/ \
		-o sparse-readout-prism-$(VERSION).tar.gz HEAD
	@echo "Done -> sparse-readout-prism-$(VERSION).tar.gz"
	@echo "git archive ships only tracked files, so .gitignore'd caches/venv/results/secrets are excluded by construction."
	@echo "Verify the scripts/ tree is present: tar tzf sparse-readout-prism-$(VERSION).tar.gz | grep 'scripts/data/' (expect hits)."
