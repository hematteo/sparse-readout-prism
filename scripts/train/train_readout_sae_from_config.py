#!/usr/bin/env python3
"""Thin model-agnostic CLI around sparse_readout_prism.runner.run_experiment.

Generic entrypoint for the config-driven trainer: load one complete YAML run
config, apply `--set a.b.c=value` overrides, call run_experiment. No
model-specific logic lives here; reusable code stays in src/sparse_readout_prism/.

run_experiment is itself idempotent/resumable (DONE marker + train_ckpt.pt),
so re-running the same sbatch after a preemption is safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from sparse_readout_prism.runner import run_experiment


def _coerce(v: str) -> Any:
    """JSON-ish scalar coercion for --set values (null/bool/int/float/str)."""
    low = v.lower()
    if low in {"null", "none", "~"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _validate_run_config(cfg: dict[str, Any], config_path: Path) -> None:
    """Refuse grid templates and incomplete run configs before training.

    Without this guard a config that lacks a ``factorizer`` / ``training`` block
    (e.g. a sweep grid template, or a stub) trains an all-defaults toy factorizer
    (topk, d_features=1024, k=32) on synthetic-fallback data — a meaningless run
    with no error. See ``configs/sweeps/README.md``.
    """
    grid_keys = [k for k in ("grid", "output_root", "fixed") if k in cfg]
    if grid_keys:
        raise SystemExit(
            f"{config_path} looks like a sweep grid template (top-level "
            f"{', '.join(grid_keys)}), not a run config — passing it to --config would "
            "silently train an all-defaults toy model. Expand it into per-cell run "
            "configs first (see configs/sweeps/README.md)."
        )
    missing = [block for block in ("factorizer", "training") if block not in cfg]
    if missing:
        raise SystemExit(
            f"{config_path} is missing required run-config block(s): {', '.join(missing)}. "
            "A complete run config needs at least factorizer + training blocks; refusing "
            "to fall back to an all-defaults toy model (see configs/sweeps/README.md)."
        )


def _apply_override(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise SystemExit(f"--set {dotted}: '{k}' is not a mapping")
    node[keys[-1]] = value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path, help="complete run YAML")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="override a config field (repeatable); value is type-coerced",
    )
    p.add_argument("--out-dir", type=Path, help="shortcut for --set run.output_dir=...")
    p.add_argument("--data-path", type=Path, help="shortcut for --set data.path=...")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="print the merged config and exit (no training)",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{args.config} did not parse to a mapping")

    if args.out_dir is not None:
        _apply_override(cfg, "run.output_dir", str(args.out_dir))
    if args.data_path is not None:
        _apply_override(cfg, "data.path", str(args.data_path))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects dotted.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        _apply_override(cfg, dotted.strip(), _coerce(raw.strip()))

    if args.print_config:
        yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
        return

    _validate_run_config(cfg, args.config)
    metrics = run_experiment(cfg)
    # Compact stdout summary for the slurm log.
    keys = [
        "selection_score",
        "val_row_explained_variance",
        "val_top1_match",
        "val_logit_kl_bits_mean",
        "dead_feature_rate",
        "rare_feature_rate",
        "train_seconds",
    ]
    print(
        "[result] " + "  ".join(f"{k}={metrics[k]}" for k in keys if k in metrics),
        flush=True,
    )


if __name__ == "__main__":
    main()
