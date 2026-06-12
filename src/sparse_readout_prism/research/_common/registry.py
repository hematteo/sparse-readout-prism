"""Shared run-script helpers: HF model loading, checkpoint-registry
resolution, and single-token resolution.

Extracted verbatim from the byte-identical blocks that
``scripts/run/run_query_fidelity_bank.py`` and
``scripts/run/run_readout_baseline_comparisons.py`` previously each carried
inline, so the five-model runners share one definition.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sparse_readout_prism.utils import load_causal_lm


def load_model(model_id: str, revision, dtype):
    return load_causal_lm(model_id, revision=revision, dtype=dtype)


def resolve_registry(path: Path) -> dict:
    reg = yaml.safe_load(path.read_text())
    root = Path(reg["archive_root"])
    out = {"archive_root": str(root), "models": {}}
    for name, spec in reg["models"].items():
        entry = {
            "model_id": spec["model_id"],
            "revision": spec.get("revision"),
            "w_u_artifact": str(root / spec["w_u_artifact"]),
            "manifest": str(root / spec["manifest"]) if spec.get("manifest") else None,
            "operating_points": {},
        }
        for op_name, op in spec["operating_points"].items():
            entry["operating_points"][op_name] = {
                "id": op["id"],
                "checkpoint": str(root / op["checkpoint"]),
            }
        out["models"][name] = entry
    return out


def resolve_single_token(tok, s: str):
    """Return (token_id, variant, reason). reason is '' on success."""
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
