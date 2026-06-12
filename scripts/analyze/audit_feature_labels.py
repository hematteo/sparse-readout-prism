#!/usr/bin/env python3
"""Feature-label audit producer (Appendix L).

The paper's feature labels are reading aids: short summaries of the vocabulary
rows that load most strongly on each sparse readout feature. The audit
(``tab:app-qualitative-feature-label-audit``, ``tab:app-main-case-study-feature-audit``)
separates *coherent lexical* labels from *ambiguous* and *token-form* ones, and
checks that coherent labels point to the side the signed term claims.

Two halves, matching what is and isn't automatable:

* ``substrate`` -- the data substrate, fully automatable. For a list of feature
  ids it emits each feature's top associated unembedding rows (decoded tokens),
  i.e. the "Top associated rows" column of ``tab:app-main-case-study-feature-audit``.
  Reuses ``display_label_features`` (the same labeller the figure scripts use).

* ``aggregate`` -- the count table. The coherent/ambiguous/token-form
  classification is a human judgment, so this reads a checked-in annotations CSV
  (one row per displayed label) and tallies it into
  ``tab:app-qualitative-feature-label-audit``. A starter annotations file for the
  main ``bug`` panels (transcribed from ``tab:app-main-case-study-feature-audit``)
  ships at ``data/audit/feature_label_audit_annotations.csv``; the per-set totals the
  paper reports are recorded in ``data/audit/feature_label_audit.json`` for validation.
  Annotating the other figure-sets needs the displayed feature ids, which come
  from running the figure scripts (GPU + Hub checkpoints) and then classifying.

Examples
--------
Substrate for the main case-study features:

    uv run python scripts/analyze/audit_feature_labels.py substrate \
        --w-u data/qwen2b/qwen2b.pt --checkpoint <qwen2b 32x/k256 checkpoint.pt> \
        --model-id Qwen/Qwen3.5-2B --top-tokens 12 \
        --feature-ids 36,4095,18302,58330,13081,59571,21804,43419,5680,34693,23681 \
        --out-csv results/feature_label_audit/main_case_study_substrate.csv

Reproduce the count table from annotations:

    uv run python scripts/analyze/audit_feature_labels.py aggregate \
        --annotations data/audit/feature_label_audit_annotations.csv \
        --validate-against data/audit/feature_label_audit.json \
        --out-csv results/feature_label_audit/counts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import torch

from sparse_readout_prism.data import preprocess_rows
from sparse_readout_prism.utils import write_csv, write_json

CATEGORIES = ("coherent", "ambiguous", "token_form")


# --------------------------------------------------------------------------- #
# substrate: per-feature top associated rows
# --------------------------------------------------------------------------- #


def run_substrate(args) -> int:
    from transformers import AutoTokenizer

    from sparse_readout_prism.research.qwen_readout import display_label_features, load_sae

    feature_ids = [int(x) for x in str(args.feature_ids).replace(" ", "").split(",") if x != ""]
    if not feature_ids:
        raise SystemExit("--feature-ids must list at least one id")

    payload = torch.load(args.w_u, map_location="cpu", weights_only=True)
    W_U = payload.get("W_U_orig", payload.get("W_U"))
    if W_U is None:
        raise SystemExit(f"{args.w_u}: no W_U_orig/W_U")
    W_U = W_U.float()
    token_mask = payload.get("token_mask")
    # Centre by the same pool training used, but keep the FULL matrix so a row
    # index still equals a token id for the tokenizer decode in the labeller.
    if token_mask is not None:
        row_mean = W_U[token_mask.bool()].mean(dim=0)
    else:
        row_mean = W_U.mean(dim=0)

    _decoder, encoder_w, encoder_b, _cfg = load_sae(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    labels = display_label_features(
        W=W_U,
        row_mean=row_mean,
        feature_ids=feature_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=args.top_tokens,
        chunk_size=args.chunk_size,
    )
    rows = [
        {"feature_id": fid, "n_tokens": len(labels[fid]), "top_associated_rows": " ; ".join(labels[fid])}
        for fid in feature_ids
    ]
    for r in rows:
        print(f"f{r['feature_id']}: {r['top_associated_rows']}")
    if args.out_csv:
        write_csv(args.out_csv, rows, fieldnames=["feature_id", "n_tokens", "top_associated_rows"])
        print(f"wrote {args.out_csv}")
    return 0


# --------------------------------------------------------------------------- #
# aggregate: annotations -> count table
# --------------------------------------------------------------------------- #


def _read_annotations(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        cat = (r.get("category") or "").strip().lower()
        if cat not in CATEGORIES:
            raise ValueError(f"{path}:{i + 2}: category {cat!r} not in {CATEGORIES}")
        tc_raw = (r.get("target_consistent") or "").strip()
        out.append(
            {
                "feature_set": (r.get("feature_set") or "").strip(),
                "feature_id": (r.get("feature_id") or "").strip(),
                "category": cat,
                "target_consistent": tc_raw in ("1", "true", "True", "yes"),
            }
        )
    return out


def tally_annotations(annotations: list[dict]) -> list[dict]:
    """Per feature-set counts: labels / coherent / ambiguous / token-form /
    target-consistent (among coherent). Order is first-seen, then a Total row."""
    by_set: "OrderedDict[str, list[dict]]" = OrderedDict()
    for a in annotations:
        by_set.setdefault(a["feature_set"], []).append(a)
    out: list[dict] = []
    tot = {"labels": 0, "coherent": 0, "ambiguous": 0, "token_form": 0, "target_consistent": 0}
    for fset, items in by_set.items():
        coherent = [a for a in items if a["category"] == "coherent"]
        row = {
            "feature_set": fset,
            "labels": len(items),
            "coherent": len(coherent),
            "ambiguous": sum(a["category"] == "ambiguous" for a in items),
            "token_form": sum(a["category"] == "token_form" for a in items),
            "target_consistent": sum(a["target_consistent"] for a in coherent),
        }
        out.append(row)
        for key in tot:
            tot[key] += row[key]
    out.append({"feature_set": "Total", **tot})
    return out


def run_aggregate(args) -> int:
    annotations = _read_annotations(args.annotations)
    counts = tally_annotations(annotations)
    width = max(len(r["feature_set"]) for r in counts)
    print(f"{'feature_set':<{width}}  labels  coher  ambig  token  target-consistent")
    for r in counts:
        print(
            f"{r['feature_set']:<{width}}  {r['labels']:>6}  {r['coherent']:>5}  {r['ambiguous']:>5}  "
            f"{r['token_form']:>5}  {r['target_consistent']}/{r['coherent']}"
        )

    if args.validate_against:
        ref = json.loads(Path(args.validate_against).read_text())
        ref_by_set = {r["feature_set"]: r for r in ref.get("feature_sets", [])}
        mismatches = []
        for r in counts:
            if r["feature_set"] == "Total":
                continue
            ref_row = ref_by_set.get(r["feature_set"])
            if ref_row is None:
                continue
            for key in ("labels", "coherent", "ambiguous", "token_form"):
                if int(ref_row.get(key, -1)) != r[key]:
                    mismatches.append(f"{r['feature_set']}.{key}: got {r[key]}, reference {ref_row.get(key)}")
        if mismatches:
            print("VALIDATION MISMATCHES (annotations vs recorded paper counts):")
            for m in mismatches:
                print(f"  - {m}")
        else:
            print("validation: annotated sets match the recorded paper counts.")

    if args.out_csv:
        write_csv(
            args.out_csv,
            counts,
            fieldnames=["feature_set", "labels", "coherent", "ambiguous", "token_form", "target_consistent"],
        )
        print(f"wrote {args.out_csv}")
    if args.out_json:
        write_json({"counts": counts}, args.out_json)
        print(f"wrote {args.out_json}")
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("substrate", help="emit per-feature top associated rows")
    s.add_argument("--w-u", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--model-id", type=str, required=True, help="HF id for the tokenizer (token decode).")
    s.add_argument("--feature-ids", type=str, required=True, help="Comma-separated feature ids.")
    s.add_argument("--top-tokens", type=int, default=12)
    s.add_argument("--chunk-size", type=int, default=16384)
    s.add_argument("--out-csv", type=Path, default=None)

    a = sub.add_parser("aggregate", help="annotations CSV -> count table")
    a.add_argument("--annotations", type=Path, required=True)
    a.add_argument("--validate-against", type=Path, default=None, help="Recorded per-set counts JSON.")
    a.add_argument("--out-csv", type=Path, default=None)
    a.add_argument("--out-json", type=Path, default=None)

    args = ap.parse_args(argv)
    if args.mode == "substrate":
        return run_substrate(args)
    return run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
