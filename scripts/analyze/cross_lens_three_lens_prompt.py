#!/usr/bin/env python3
"""One prompt's frozen states read by several fitted lenses (``tab:app-cross-lens-de-antonym-layers``).

Offline over readout dumps from ``scripts/run/run_cross_lens_readouts.py``, one per
lens, all produced from the same one-prompt bank
(``data/cross_lens/cross_lens_three_lens_prompts.json``). For the chosen prompt id
and layers the script tabulates, per lens: the top-1 token and its transported
logit (the table), the full top-5 reading, the lens logit, rank and decomposed
(fp32) logit of every probed target (e.g. ' groß', ' large', ' big', '大'), and
the dominant readout feature of each target's decomposition and of the lens's
own top-1 token.

Paper run (the German antonym prompt ``antonym_de_01``, answer ``groß``, read by the
English-, Chinese- and German-fitted 100-prompt Jacobian lenses; dumps produced
with ``--n-positions 1 --decompose-top1 --top-feats 10``)::

  cross_lens_three_lens_prompt.py --dump EN=<dumps>/three_lens__jlens_en.json \
      --dump ZH=<dumps>/three_lens__jlens_zh.json --dump DE=<dumps>/three_lens__jlens_de.json \
      --prompt-id antonym_de_01 --layers 24,26,29 \
      --out-csv <out>/three_lens_top1.csv --out-targets-csv <out>/three_lens_targets.csv \
      --out-features-csv <out>/three_lens_features.csv

Adding ``--dump LOGIT=<dumps>/three_lens__identity.json`` (the ``--lens identity``
run) includes the lens-free reading as a further column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_readout_prism.utils import write_csv

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


def select_record(dump: dict, prompt_id: str | None) -> dict:
    records = dump["records"]
    if prompt_id is None:
        if len(records) != 1:
            raise ValueError(f"dump has {len(records)} records; pass --prompt-id")
        return records[0]
    for r in records:
        if r["id"] == prompt_id:
            return r
    raise KeyError(f"prompt id {prompt_id!r} not in dump")


def three_lens_tables(
    records: dict[str, dict], layers: list[str], targets: list[str] | None
) -> tuple[list[dict], list[dict], list[dict]]:
    """Per layer x lens: top-1 rows, per-target rows, and dominant-feature rows.

    ``records`` maps a lens label to the prompt's record in that lens's dump.
    """
    top1_rows, target_rows, feature_rows = [], [], []
    for L in layers:
        for label, rec in records.items():
            e = rec["layers"][L][POS]
            tok1, logit1 = e["top5"][0]
            top1_rows.append(
                {
                    "layer": L,
                    "lens": label,
                    "top1_token": tok1,
                    "top1_lens_logit": logit1,
                    "top5_tokens": " | ".join(t for t, _ in e["top5"]),
                }
            )
            if e.get("top1_decomp"):
                d = e["top1_decomp"]
                f0 = d["top_features"][0] if d["top_features"] else {"id": None, "contribution": None}
                feature_rows.append(
                    {
                        "layer": L,
                        "lens": label,
                        "token": tok1,
                        "role": "top1",
                        "original_logit": d["original_logit"],
                        "feature_sum": d["feature_sum"],
                        "residual": d["residual"],
                        "dominant_feature": f0["id"],
                        "dominant_contribution": f0["contribution"],
                    }
                )
            probe = targets if targets is not None else list(e["targets"])
            for t in probe:
                d = e["targets"].get(t)
                if d is None:
                    continue
                f0 = d["top_features"][0] if d["top_features"] else {"id": None, "contribution": None}
                target_rows.append(
                    {
                        "layer": L,
                        "lens": label,
                        "target": t,
                        "lens_logit": d["lens_logit"],
                        "lens_rank": d["lens_rank"],
                        "original_logit": d["original_logit"],
                    }
                )
                feature_rows.append(
                    {
                        "layer": L,
                        "lens": label,
                        "token": t,
                        "role": "target",
                        "original_logit": d["original_logit"],
                        "feature_sum": d["feature_sum"],
                        "residual": d["residual"],
                        "dominant_feature": f0["id"],
                        "dominant_contribution": f0["contribution"],
                    }
                )
    return top1_rows, target_rows, feature_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dump", action="append", required=True, help="LABEL=path, repeatable (e.g. EN=..., ZH=..., DE=...)"
    )
    p.add_argument("--prompt-id", default=None, help="record id to tabulate (default: the only record)")
    p.add_argument("--layers", default="24,26,29", help="comma-separated layer keys (default: paper rows)")
    p.add_argument("--targets", default=None, help="comma-separated probed targets (default: all in the dump)")
    p.add_argument("--out-csv", default=None, help="top-1 token and transported logit per layer x lens")
    p.add_argument("--out-targets-csv", default=None, help="lens logit and rank per layer x lens x target")
    p.add_argument("--out-features-csv", default=None, help="dominant feature per layer x lens x token")
    args = p.parse_args(argv)

    dumps = parse_dump_args(args.dump)
    records = {label: select_record(json.loads(path.read_text()), args.prompt_id) for label, path in dumps}
    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    targets = [s for s in args.targets.split(",")] if args.targets else None
    top1_rows, target_rows, feature_rows = three_lens_tables(records, layers, targets)

    first = next(iter(records.values()))
    labels = list(records)
    print(
        f"prompt {first['id']} ({first['group']}), answer {first['answer']!r}, continuation {first['continuation']!r}"
    )
    print("layer  " + "".join(f"{lb:>24s}" for lb in labels))
    for L in layers:
        cells = [r for r in top1_rows if r["layer"] == L]
        print(f"{L:>5s}  " + "".join(f"{c['top1_token']!r} {c['top1_lens_logit']:.1f}".rjust(24) for c in cells))
    probe = sorted({r["target"] for r in target_rows}, key=lambda t: [r["target"] for r in target_rows].index(t))
    for L in layers:
        print(f"-- layer {L}: lens logit (rank) per target")
        for t in probe:
            cells = [r for r in target_rows if r["layer"] == L and r["target"] == t]
            print(f"  {t!r:14s}" + "".join(f"{c['lens_logit']:.2f} (r{c['lens_rank']})".rjust(24) for c in cells))
        for r in feature_rows:
            if r["layer"] == L and r["role"] == "top1":
                print(
                    f"  {r['lens']:>6s} top1={r['token']!r:12s} f{r['dominant_feature']} "
                    f"{r['dominant_contribution']:+.2f} (sum {r['feature_sum']:.1f}, resid {r['residual']:.2f})"
                )

    if args.out_csv:
        write_csv(args.out_csv, top1_rows)
        print(f"wrote {args.out_csv}")
    if args.out_targets_csv:
        write_csv(args.out_targets_csv, target_rows)
        print(f"wrote {args.out_targets_csv}")
    if args.out_features_csv:
        write_csv(args.out_features_csv, feature_rows)
        print(f"wrote {args.out_features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
