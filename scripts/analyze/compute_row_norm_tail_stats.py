#!/usr/bin/env python3
"""Emit the W_U row-norm tail statistics table (Appendix E.4, ``tab:app-k-row-norm-tail``).

The TopK recipe operates on row-centred, row-normalised W_U rows, so the
*row-norm distribution* is the set of per-row centred norms ``||W_U[v] - mu||``
that the factoriser divides out (``preprocess_rows`` in
``sparse_readout_prism.data``). A heavy upper tail amplifies small per-row errors
on the high-norm rows that most affect top1/KL after the softmax, which is the
second-order property the appendix reports for the reasoning-distilled rows.

This was previously hand-computed from the per-model extraction artifacts; this
script makes it reproducible. It reads one or more extraction ``.pt`` payloads
(``{W_U_orig, ...}`` written by ``scripts/data/extract_model_readout.py``) and
reports, per model, the centred-row-norm summary the appendix table uses:
min / p50 / max / mean / CoV (sigma/mu) / max-over-median.

No GPU, no SAE checkpoint, no model load required.

Examples
--------
Per-artifact (label=path):

    uv run python scripts/analyze/compute_row_norm_tail_stats.py \
        --artifact "Qwen3.5-9B (base)=data/qwen35-9b/qwen35_9b.pt" \
        --artifact "Ministral-3-8B (base)=data/ministral3-8b/ministral3_8b.pt" \
        --out-csv results/row_norm_tail/row_norm_tail.csv \
        --out-json results/row_norm_tail/row_norm_tail.json

From a fidelity registry (resolves each model's ``w_u_artifact``):

    uv run python scripts/analyze/compute_row_norm_tail_stats.py \
        --registry configs/registries/result1_query_fidelity_cluster.yaml \
        --models Qwen3.5-9B,Ministral-3-8B-Base,R1-Distill-Qwen-7B,R1-Distill-Llama-8B \
        --out-csv results/row_norm_tail/row_norm_tail.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparse_readout_prism.utils import write_csv, write_json

CENTER_CHOICES = ("masked_mean", "full_mean", "none")


def row_norm_tail_stats(
    W_U: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    center: str = "masked_mean",
) -> dict[str, float]:
    """Centred-row-norm summary statistics for one readout matrix.

    ``W_U`` is ``(vocab, d_model)``. ``token_mask`` (``(vocab,)`` bool) restricts
    to text rows the way training does (drops vision / special rows). ``center``
    selects the subtraction: ``masked_mean`` (mean over kept rows, matching
    ``preprocess_rows`` on the training pool), ``full_mean`` (mean over the full
    vocab), or ``none`` (raw row norms ``||W_U[v]||``).
    """
    if center not in CENTER_CHOICES:
        raise ValueError(f"center must be one of {CENTER_CHOICES}, got {center!r}")
    W = W_U.detach().float()  # (vocab, d_model)
    if token_mask is not None:
        mask = token_mask.bool().cpu()
        if mask.numel() != W.shape[0]:
            raise ValueError(f"token_mask length {mask.numel()} != vocab {W.shape[0]}")
        rows = W[mask]  # (kept, d_model)
    else:
        rows = W
    if center == "masked_mean":
        rows = rows - rows.mean(dim=0)
    elif center == "full_mean":
        rows = rows - W.mean(dim=0)
    norms = rows.norm(dim=1)  # (kept,)
    mean = float(norms.mean())
    std = float(norms.std(unbiased=True))
    median = float(norms.median())
    return {
        "n_rows": int(norms.numel()),
        "min": float(norms.min()),
        "p50": median,
        "max": float(norms.max()),
        "mean": mean,
        "cov": (std / mean) if mean > 0 else float("nan"),
        "max_over_med": (float(norms.max()) / median) if median > 0 else float("nan"),
    }


def _load_w_u(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    W_U = payload.get("W_U_orig", payload.get("W_U"))
    if W_U is None:
        raise KeyError(f"{path}: no 'W_U_orig'/'W_U' key (keys: {list(payload)})")
    return W_U, payload.get("token_mask")


def _parse_artifacts(raw: list[str]) -> list[tuple[str, Path]]:
    """Each entry is ``LABEL=PATH`` (or bare ``PATH``, label = file stem)."""
    out: list[tuple[str, Path]] = []
    for entry in raw:
        if "=" in entry:
            label, path = entry.split("=", 1)
        else:
            label, path = Path(entry).stem, entry
        out.append((label.strip(), Path(path.strip())))
    return out


def _resolve_registry_artifacts(registry: Path, models: list[str]) -> list[tuple[str, Path]]:
    from sparse_readout_prism.research.registry import resolve_registry

    reg = resolve_registry(registry)
    out: list[tuple[str, Path]] = []
    for name in models:
        if name not in reg["models"]:
            raise SystemExit(f"model {name!r} not in registry; have {list(reg['models'])}")
        out.append((name, Path(reg["models"][name]["w_u_artifact"])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Extraction .pt payload; repeatable. LABEL is the row label in the table.",
    )
    ap.add_argument("--registry", type=Path, default=None, help="Fidelity registry to resolve w_u_artifact paths from.")
    ap.add_argument("--models", type=str, default="", help="Comma-separated model names to pull from --registry.")
    ap.add_argument("--center", choices=CENTER_CHOICES, default="masked_mean")
    ap.add_argument("--no-token-mask", action="store_true", help="Ignore token_mask in the payload (use all rows).")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    artifacts = _parse_artifacts(args.artifact)
    if args.registry is not None:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            ap.error("--registry requires --models")
        artifacts += _resolve_registry_artifacts(args.registry, models)
    if not artifacts:
        ap.error("provide at least one --artifact or --registry/--models")

    rows: list[dict] = []
    for label, path in artifacts:
        if not path.exists():
            raise SystemExit(f"artifact missing: {path}")
        W_U, token_mask = _load_w_u(path)
        if args.no_token_mask:
            token_mask = None
        stats = row_norm_tail_stats(W_U, token_mask=token_mask, center=args.center)
        rows.append({"model": label, "artifact": str(path), "center": args.center, **stats})
        print(
            f"{label:28s} min={stats['min']:.3f} p50={stats['p50']:.3f} max={stats['max']:.3f} "
            f"mean={stats['mean']:.3f} CoV={stats['cov']:.3f} max/med={stats['max_over_med']:.2f} "
            f"(n={stats['n_rows']})"
        )

    if args.out_csv:
        write_csv(
            args.out_csv,
            rows,
            fieldnames=["model", "min", "p50", "max", "mean", "cov", "max_over_med", "n_rows", "center", "artifact"],
        )
        print(f"wrote {args.out_csv}")
    if args.out_json:
        write_json({"center": args.center, "rows": rows}, args.out_json)
        print(f"wrote {args.out_json}")
    if not args.out_csv and not args.out_json:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
