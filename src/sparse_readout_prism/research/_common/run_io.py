"""Shared row-grouping + CSV-writing helpers for the Result-1 run scripts.

``run_query_fidelity_bank.py`` and ``run_readout_baseline_comparisons.py`` emit
the same per-bank / per-method CSV summaries from the same in-memory row dicts;
these two helpers were byte-identical copies in each runner.

Their sibling ``_group_metrics`` functions are deliberately *not* shared: the
baseline variant carries an extra ``accepted_rate`` / CI column that the
``readout_baseline_comparisons`` figure consumes and the fidelity variant does
not.
"""

from __future__ import annotations

from pathlib import Path


def group_rows(rows: list[dict], key) -> dict:
    """Group row dicts by a column name, or by a tuple of column names."""
    g: dict = {}
    for r in rows:
        g.setdefault(r[key] if not isinstance(key, tuple) else tuple(r[k] for k in key), []).append(r)
    return g


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    """Write row dicts to CSV; empty rows -> 0-byte file, atomic temp-replace."""
    from sparse_readout_prism.utils import write_csv

    write_csv(path, rows, write_empty=True, atomic=True)
