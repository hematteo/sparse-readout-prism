"""Path helpers for the Sparse Readout Prism workspace.

Honours ``SRP_SSD_ROOT`` (default: a ``local_snapshots/`` dir under the repo)
so the same code runs on the laptop or the cluster after ``export SRP_SSD_ROOT=...``.

These helpers locate the *repo checkout* — they are for the ``scripts/`` entry
points, not for the installed package: under a wheel install (no ``.git``, no
``pyproject.toml`` in any parent) ``repo_root()`` raises, and ``ssd_root()``
then requires ``SRP_SSD_ROOT`` to be set.

Use:

    from sparse_readout_prism.paths import repo_root, ssd_root
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repo root, robust to scripts moving up/down the tree.

    Walks up from this module to the first dir containing ``.git``, or a
    ``pyproject.toml`` alongside ``src/sparse_readout_prism`` (the src check
    keeps a wheel installed into some *other* project's venv from silently
    resolving to that project's root). Use instead of
    ``Path(__file__).resolve().parents[N]`` in experiment scripts so path
    resolution survives reorgs.
    """
    for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (p / ".git").exists():
            return p
        if (p / "pyproject.toml").exists() and (p / "src" / "sparse_readout_prism").is_dir():
            return p
    raise RuntimeError(
        "could not find repo root from sparse_readout_prism/paths.py (no .git or pyproject.toml in any parent dir)"
    )


def ssd_root() -> Path:
    """Root of the canonical project SSD (``SRP_SSD_ROOT``, else ``<repo>/local_snapshots``)."""
    env = os.environ.get("SRP_SSD_ROOT")
    if env:
        return Path(env)
    return repo_root() / "local_snapshots"
