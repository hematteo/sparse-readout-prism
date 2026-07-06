"""Shared helpers for the Result-1 run scripts.

``run_query_fidelity_bank.py`` and ``run_readout_baseline_comparisons.py`` emit
the same per-bank / per-method CSV summaries from the same in-memory row dicts;
the helpers here were byte-identical copies in each runner (row grouping, CSV
writing, bank loading, A/B token resolution + audit rows, per-row margin
statistics, margin-bin constants).

Their sibling ``_group_metrics`` functions are deliberately *not* shared: the
baseline variant carries an extra ``accepted_rate`` / CI column that the
``readout_baseline_comparisons`` figure consumes and the fidelity variant does
not. Likewise each runner keeps its own ``native_queries*`` (five frontier
queries in the fidelity runner vs. two in the baseline runner) and its own
``margin_from_rows`` (SAE-direct vs. method-object dispatch).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from sparse_readout_prism.research.registry import resolve_single_token_strict

EPS = 1e-6
MARGIN_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
MARGIN_BIN_LABELS = ["<0.5", "0.5-1", "1-2", ">=2"]


def group_rows(rows: list[dict], key: str | tuple[str, ...]) -> dict:
    """Group row dicts by a column name, or by a tuple of column names."""
    g: dict = {}
    for r in rows:
        g.setdefault(r[key] if not isinstance(key, tuple) else tuple(r[k] for k in key), []).append(r)
    return g


def run_provenance(args: Any = None) -> dict[str, Any]:
    """Provenance block for run manifests: command line, args, git hash, versions.

    ``args`` is an ``argparse.Namespace`` (or dict) of the parsed CLI args;
    merge the result into the script's ``manifest.json`` so every artifact
    records how it was produced.
    """
    import torch

    from sparse_readout_prism import __version__
    from sparse_readout_prism.utils import git_commit, to_jsonable

    if args is not None and not isinstance(args, dict):
        args = vars(args)
    return {
        "command": " ".join(sys.argv),
        "args": to_jsonable(args) if args is not None else None,
        "git_commit": git_commit(),
        "package_version": __version__,
        "torch_version": torch.__version__,
        "timestamp_unix": time.time(),
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    """Write row dicts to CSV; empty rows -> 0-byte file, atomic temp-replace."""
    from sparse_readout_prism.utils import write_csv

    write_csv(path, rows, write_empty=True, atomic=True)


def load_bank(path: Path, cap: int | None) -> list[dict]:
    """Load a JSONL query bank, capping by *base case* (not raw rows) so the
    per-base-case bootstrap in the group metrics stays honest."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if cap is not None and cap < len(rows):
        keep_bases = list(dict.fromkeys(r["base_case_id"] for r in rows))[:cap]
        rows = [r for r in rows if r["base_case_id"] in set(keep_bases)]
    return rows


def margin_row_stats(mr: dict) -> dict:
    """Per-row margin statistics for the Result-1 emit schema.

    Key order matters: callers ``row.update()`` this into their identity
    columns and the CSV writers derive column order from first-seen keys.
    """
    exact, sparse = mr["exact"], mr["sparse"]
    binlbl = next(MARGIN_BIN_LABELS[i] for i, (lo, hi) in enumerate(MARGIN_BINS) if lo <= abs(exact) < hi)
    return {
        "exact_margin": exact,
        "sparse_margin": sparse,
        "residual": exact - sparse,
        "abs_residual": abs(exact - sparse),
        "residual_direct": abs(exact - sparse) / max(abs(exact), EPS),
        "resid_term": mr["resid_term"],
        "sign_match": bool((exact > 0) == (sparse > 0)),
        "abs_exact": abs(exact),
        "margin_bin": binlbl,
        "tiny_margin": abs(exact) < 0.5,
    }


def resolve_ab_case(
    tok,
    r: dict,
    model_name: str,
    audit: list[dict],
    skipped: list[dict],
) -> tuple[list[int], list[int], str, str] | None:
    """Resolve a curated A/B or token-family bank row into token-id lists.

    Returns ``(a_ids, b_ids, family_label, query_label)``; appends the
    per-target audit rows either way, and on failure appends a ``skipped`` row
    and returns ``None``. This block was byte-identical in both Result-1
    runners — only the margin computation that follows it differs.
    """
    fam = r["family"]
    if r.get("target_a") is not None:  # single-token A/B
        ta, va, ra = resolve_single_token_strict(tok, r["target_a"])
        tb, vb, rb = resolve_single_token_strict(tok, r["target_b"])
        audit += [
            {
                "model": model_name,
                "case_id": r["case_id"],
                "target": r["target_a"],
                "side": "A",
                "token_id": ta,
                "variant": va,
                "reason": ra or "ok",
            },
            {
                "model": model_name,
                "case_id": r["case_id"],
                "target": r["target_b"],
                "side": "B",
                "token_id": tb,
                "variant": vb,
                "reason": rb or "ok",
            },
        ]
        if ta is None or tb is None:
            skipped.append({"case_id": r["case_id"], "reason": f"tok A={ra} B={rb}"})
            return None
        if ta == tb:
            skipped.append({"case_id": r["case_id"], "reason": "ab_collision"})
            return None
        return [ta], [tb], fam, "single_token"

    a_ids, b_ids = [], []
    for side, key, out in (("Afam", "target_a_family", a_ids), ("Bfam", "target_b_family", b_ids)):
        for s in r[key]:
            tid, v, rsn = resolve_single_token_strict(tok, s)
            audit.append(
                {
                    "model": model_name,
                    "case_id": r["case_id"],
                    "target": s,
                    "side": side,
                    "token_id": tid,
                    "variant": v,
                    "reason": rsn or "ok",
                }
            )
            if tid is not None:
                out.append(tid)
    if not a_ids or not b_ids:
        skipped.append({"case_id": r["case_id"], "reason": "family_empty"})
        return None
    return a_ids, b_ids, fam + "_family", "token_family"
