#!/usr/bin/env python3
"""Aggregate an English-Chinese cross-lens sweep: dominant-feature agreement, null floors, surface split.

Reads two readout dumps from ``scripts/run/run_cross_lens_readouts.py`` (the same
frozen Qwen3.5-9B states read through two lenses) and the EN-ZH prompt bank, and
computes the numbers behind ``tab:app-cross-lens-families`` and the EN-ZH rows of
``tab:app-cross-lens-extension`` / ``fig:cross-lens-extension``:

  - per-family and pooled cross-lens agreement: the dominant feature (largest
    |contribution| in the decomposition of the prompt's answer form ``form_a``)
    is the same feature id under both lenses on a majority of the mid-band
    layers {21, 24, 26, 29} at the final position, with Wilson 95% CIs;
  - within-lens cross-form agreement: ``form_a`` and ``form_b`` share a dominant
    feature under the same lens ("one feature carries both surface forms");
  - two null floors: (a) unrelated token, ``form_a`` vs each ``null_targets``
    entry within the slot-A lens; (b) shuffled pairing, prompt i's ``form_a``
    under lens A vs prompt j's ``form_a`` under lens B for three seeded draws of
    j with a different concept;
  - the lens-only surface split: the script (Latin / CJK) of each lens's top-1
    token by mid-band plurality vote (a 2-2 tie resolves to the alphabetically
    first label, CJK < LATIN < OTHER), and the count of prompts read Latin by
    lens A and CJK by lens B (the "Lens-only split" column);
  - control families (``ctrl_*``) reported separately, never pooled into the
    headline.

Majority rule. ``--agreement-rule half`` (default, the paper) passes a vote on at
least ``ceil(n/2)`` of the scored mid-band layers (2 of 4); ``strict`` needs more
than half (3 of 4). The rule applies to every vote above (headline, within-lens
cross-form, both null floors). The metric bodies live in
``research.cross_lens`` (``aggregate_pair``); this entry point supplies the
EN-ZH families and the script-based surface call.

Slot semantics: ``--lens-a`` is the English-fitted lens and ``--lens-b`` the
Chinese-fitted lens (output keys ``*_en`` / ``*_zh``). For the construction
comparison the two slots hold the English-fitted Jacobian lens and the
English-fitted ridge translator; the key names are unchanged. The summary JSON
carries a ``provenance`` block.

Paper runs (all with ``--prompts data/cross_lens/cross_lens_prompts_en_zh.json --seed 0``)::

  # main study: extension-matrix row 1 and tab:app-cross-lens-families
  aggregate_cross_lens_en_zh.py --lens-a <dumps>/en_zh__jlens_en_n100.json \
      --lens-b <dumps>/en_zh__jlens_zh_n100.json --out <out>/en_zh_jlens_n100_summary.json
  # ridge translators: row 3
  aggregate_cross_lens_en_zh.py --lens-a <dumps>/en_zh__ridge_en_n100.json \
      --lens-b <dumps>/en_zh__ridge_zh_n100.json --out <out>/en_zh_ridge_n100_summary.json
  # Jacobian vs ridge on the English corpus: row 4
  aggregate_cross_lens_en_zh.py --lens-a <dumps>/en_zh__jlens_en_n100.json \
      --lens-b <dumps>/en_zh__ridge_en_n100.json --out <out>/en_jlens_vs_ridge_summary.json
  # 300-prompt refits: row 5
  aggregate_cross_lens_en_zh.py --lens-a <dumps>/en_zh__jlens_en_n300.json \
      --lens-b <dumps>/en_zh__jlens_zh_n300.json --out <out>/en_zh_jlens_n300_summary.json
"""

from __future__ import annotations

import argparse
import unicodedata

from sparse_readout_prism.research.cross_lens import (
    AGREEMENT_RULES,
    PairSpec,
    aggregate_pair,
    load_bank_items,
    load_dump_records,
    print_summary,
    write_summary,
)
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import write_csv

CROSS_GROUPS = (
    "antonym_zh",
    "trans_zh2en",
    "trans_en2zh",
    "cloze_zh",
    "cloze_en",
    "fact_zh",
    "exemplar_zh",
)


def script_of(tok: str) -> str:
    for ch in tok:
        if ch.strip() == "" or not ch.isalnum():
            continue
        name = unicodedata.name(ch, "")
        if "CJK" in name:
            return "CJK"
        if "LATIN" in name:
            return "LATIN"
    return "OTHER"


SPEC = PairSpec(
    tag_b="zh",
    cross_groups=CROSS_GROUPS,
    surface_column="script",
    surface_call=lambda tok, _item: script_of(tok),
    split_labels=("LATIN", "CJK"),
    split_caption="EN=Latin & ZH=CJK",
    surface_caption="top-1 script",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lens-a", required=True, help="dump under the English-fitted lens (output keys *_en)")
    p.add_argument("--lens-b", required=True, help="dump under the Chinese-fitted lens (output keys *_zh)")
    p.add_argument("--prompts", required=True, help="EN-ZH prompt bank JSON")
    p.add_argument("--seed", type=int, default=0, help="seed for the shuffled-pairing null")
    p.add_argument(
        "--agreement-rule",
        choices=AGREEMENT_RULES,
        default="half",
        help="mid-band majority: half = at least ceil(n/2) layers (paper), strict = more than half",
    )
    p.add_argument("--out", required=True, help="summary JSON (per-group rates, floors, per-prompt rows)")
    p.add_argument("--rows-csv", default=None, help="optional per-prompt rows as CSV")
    args = p.parse_args(argv)

    summary = aggregate_pair(
        load_dump_records(args.lens_a),
        load_dump_records(args.lens_b),
        load_bank_items(args.prompts),
        SPEC,
        seed=args.seed,
        rule=args.agreement_rule,
    )
    print(f"[agg] {len(summary['rows'])} prompts present in both dumps")
    print_summary(summary, SPEC)
    summary["provenance"] = run_provenance(args)
    write_summary(summary, args.out)
    print(f"\n[agg] wrote {args.out}")
    if args.rows_csv:
        write_csv(args.rows_csv, summary["rows"])
        print(f"[agg] wrote {args.rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
