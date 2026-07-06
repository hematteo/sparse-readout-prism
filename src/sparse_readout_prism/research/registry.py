"""Shared run-script helpers: checkpoint-registry resolution and strict
single-token resolution.

Extracted verbatim from the byte-identical blocks that
``scripts/run/run_query_fidelity_bank.py`` and
``scripts/run/run_readout_baseline_comparisons.py`` previously each carried
inline, so the five-model runners share one definition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(raw: str, root: Path) -> str:
    """Expand env vars in a registry path and anchor it under ``root``.

    Paths that are absolute after expansion (including entries that embed
    ``${SRP_ARCHIVE_ROOT}`` themselves) are used as-is instead of being joined,
    so a root-prefixed entry does not end up with a doubled prefix.
    """
    expanded = os.path.expandvars(raw)
    if "${" in expanded:
        raise ValueError(
            f"registry path {raw!r} contains an unset environment variable — "
            "export it (e.g. SRP_ARCHIVE_ROOT) before resolving the registry"
        )
    p = Path(expanded)
    return str(p if p.is_absolute() else root / p)


def resolve_registry(path: Path) -> dict:
    """Load a checkpoint-registry YAML and resolve every artifact path.

    ``archive_root`` and all per-model paths go through ``os.path.expandvars``,
    so registries can anchor themselves at ``${SRP_ARCHIVE_ROOT}``; an unset
    variable raises instead of silently producing literal ``${...}`` paths
    (which the runners would then "skip gracefully" as missing checkpoints).
    """
    reg = yaml.safe_load(path.read_text())
    root_str = os.path.expandvars(str(reg["archive_root"]))
    if "${" in root_str:
        raise ValueError(
            f"registry {path} archive_root {reg['archive_root']!r} contains an unset "
            "environment variable — export SRP_ARCHIVE_ROOT before running"
        )
    root = Path(root_str)
    out: dict[str, Any] = {"archive_root": str(root), "models": {}}
    for name, spec in reg["models"].items():
        entry: dict[str, Any] = {
            "model_id": spec["model_id"],
            "revision": spec.get("revision"),
            "w_u_artifact": _resolve_path(spec["w_u_artifact"], root),
            "manifest": _resolve_path(spec["manifest"], root) if spec.get("manifest") else None,
            "operating_points": {},
        }
        for op_name, op in spec["operating_points"].items():
            entry["operating_points"][op_name] = {
                "id": op["id"],
                "checkpoint": _resolve_path(op["checkpoint"], root),
            }
        out["models"][name] = entry
    return out


def resolve_single_token_strict(tok, s: str):
    """Return (token_id, variant, reason). reason is '' on success.

    The *strict* resolver used by the Result-1 runners: unlike
    ``research.query_decompose.resolve_single_token`` (which returns the used
    text + decoded label for audit CSVs), this one additionally rejects special
    tokens and reports which spacing variant matched (``" +s"`` or ``"bare"``).
    """
    if s is None or s == "":
        return None, None, "empty"
    specials = set(getattr(tok, "all_special_ids", []) or [])
    for variant, text in ((" +s", " " + s), ("bare", s)):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            if ids[0] in specials:
                return None, variant, "special_token"
            return int(ids[0]), variant, ""
    ids = tok.encode(" " + s, add_special_tokens=False)
    if len(ids) == 0:
        return None, None, "empty_tokenization"
    return None, None, f"multi_token({len(ids)})"
