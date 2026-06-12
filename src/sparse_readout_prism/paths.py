"""Path helpers for the Sparse Readout Prism workspace.

Honours ``SRP_SSD_ROOT`` (default: a ``local_snapshots/`` dir under the repo)
so the same code runs on the laptop or the cluster after ``export SRP_SSD_ROOT=...``.

Use:

    from sparse_readout_prism.paths import repo_root, ssd_root
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SSD_ROOT = PROJECT / "local_snapshots"


def repo_root() -> Path:
    """Repo root, robust to scripts moving up/down the tree.

    Walks up from this module to the first dir containing ``.git`` or
    ``pyproject.toml``. Use instead of ``Path(__file__).resolve().parents[N]``
    in experiment scripts so path resolution survives reorgs.
    """
    for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    raise RuntimeError(
        "could not find repo root from sparse_readout_prism/paths.py (no .git or pyproject.toml in any parent dir)"
    )


def ssd_root() -> Path:
    """Root of the canonical project SSD."""
    return Path(os.environ.get("SRP_SSD_ROOT", str(DEFAULT_SSD_ROOT)))
