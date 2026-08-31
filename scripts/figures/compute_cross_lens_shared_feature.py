#!/usr/bin/env python3
"""Metrics behind ``fig:cross-lens-butterfly``: two lenses, two surface tokens, one dominant readout feature.

Offline over two (or more) readout dumps from
``scripts/run/run_cross_lens_readouts.py`` produced with ``--decompose-top1`` on the
same bank. For one prompt id and one layer the script extracts, per lens: the
top-1 token the lens reports and its logit, the feature sum of that token's
decomposition, the dominant feature (largest |contribution|) with its share of
the feature sum, and the largest contribution from any other feature. It then
reports whether the dominant feature is shared across the lenses, each lens's
share carried by the shared feature, and the feature's top unembedding rows from
the dump's ``feature_top_tokens`` labels. No rendering; the figure is drawn in
the paper source from these numbers.

Paper run (factual-recall prompt ``fac_03``, "The capital of China is Beijing. The
capital of the UK is", layer 26, English- and Chinese-fitted 100-prompt Jacobian
lenses)::

  compute_cross_lens_shared_feature.py --dump EN=<dumps>/en_zh__jlens_en_n100.json \
      --dump ZH=<dumps>/en_zh__jlens_zh_n100.json --prompt-id fac_03 --layer 26 \
      --out-csv <out>/cross_lens_shared_feature.csv --out-json <out>/cross_lens_shared_feature.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_readout_prism.utils import write_csv, write_json

POS = "-1"


def parse_dump_args(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``LABEL=path`` arguments, preserving order."""
    out = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise ValueError(f"--dump expects LABEL=path, got {spec!r}")
        out.append((label, Path(path)))
    return out


def lens_top1_summary(record: dict, layer: str) -> dict:
    """Top-1 token, its logit, and the dominant-feature share of its decomposition at one layer."""
    e = record["layers"][layer][POS]
    if "top1_decomp" not in e:
        raise KeyError("dump lacks top1_decomp; rerun run_cross_lens_readouts.py with --decompose-top1")
    tok1, logit1 = e["top5"][0]
    d = e["top1_decomp"]
    feats = d["top_features"]
    dom = feats[0] if feats else {"id": None, "contribution": 0.0}
    others = [f["contribution"] for f in feats[1:]]
    fsum = d["feature_sum"]
    return {
        "top1_token": tok1,
        "top1_lens_logit": logit1,
        "top1_original_logit": d["original_logit"],
        "feature_sum": fsum,
        "residual": d["residual"],
        "dominant_feature": dom["id"],
        "dominant_contribution": dom["contribution"],
        "dominant_share_of_feature_sum": dom["contribution"] / fsum if fsum else None,
        "largest_other_contribution": max(others) if others else None,
        "contributions": {int(f["id"]): float(f["contribution"]) for f in feats},
    }


def shared_feature_metrics(records: dict[str, dict], layer: str, labels_by_lens: dict[str, dict]) -> dict:
    """Per-lens top-1 summaries plus the shared dominant feature and its share under each lens."""
    per_lens = {label: lens_top1_summary(rec, layer) for label, rec in records.items()}
    dominant = {s["dominant_feature"] for s in per_lens.values()}
    shared = next(iter(dominant)) if len(dominant) == 1 else None
    shared_share = {}
    if shared is not None:
        for label, s in per_lens.items():
            c = s["contributions"].get(shared, 0.0)
            shared_share[label] = c / s["feature_sum"] if s["feature_sum"] else None
    top_tokens = None
    if shared is not None:
        for labels in labels_by_lens.values():
            if str(shared) in labels:
                top_tokens = labels[str(shared)]
                break
    return {
        "layer": layer,
        "per_lens": per_lens,
        "shared_dominant_feature": shared,
        "shared_feature_share_of_feature_sum": shared_share,
        "shared_feature_top_tokens": top_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", action="append", required=True, help="LABEL=path, repeatable (e.g. EN=..., ZH=...)")
    p.add_argument("--prompt-id", default="fac_03", help="record id (paper: fac_03)")
    p.add_argument("--layer", default="26", help="layer key (paper: 26)")
    p.add_argument("--out-csv", default=None, help="one row per lens")
    p.add_argument("--out-json", default=None, help="full metrics including the shared-feature block")
    args = p.parse_args(argv)

    records, labels_by_lens = {}, {}
    for label, path in parse_dump_args(args.dump):
        dump = json.loads(path.read_text())
        matches = [r for r in dump["records"] if r["id"] == args.prompt_id]
        if not matches:
            raise KeyError(f"prompt id {args.prompt_id!r} not in {path}")
        records[label] = matches[0]
        labels_by_lens[label] = dump.get("feature_top_tokens", {})

    metrics = shared_feature_metrics(records, args.layer, labels_by_lens)
    metrics["prompt_id"] = args.prompt_id
    rows = []
    for label, s in metrics["per_lens"].items():
        row = {"lens": label, "prompt_id": args.prompt_id, "layer": args.layer}
        row.update({k: v for k, v in s.items() if k != "contributions"})
        row["shared_dominant_feature"] = metrics["shared_dominant_feature"]
        row["shared_feature_share_of_feature_sum"] = metrics["shared_feature_share_of_feature_sum"].get(label)
        rows.append(row)
        share = s["dominant_share_of_feature_sum"]
        print(
            f"{label:>6s}: top1={s['top1_token']!r} logit={s['top1_lens_logit']:.1f} "
            f"dominant=f{s['dominant_feature']} share={share:.3f} "
            f"largest_other={s['largest_other_contribution']:+.2f} feature_sum={s['feature_sum']:.2f}"
        )
    print(
        f"shared dominant feature: {metrics['shared_dominant_feature']} top rows: {metrics['shared_feature_top_tokens']}"
    )

    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"wrote {args.out_csv}")
    if args.out_json:
        write_json(metrics, args.out_json)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
