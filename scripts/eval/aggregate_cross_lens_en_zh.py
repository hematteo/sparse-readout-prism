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

Slot semantics: ``--lens-a`` is the English-fitted lens and ``--lens-b`` the
Chinese-fitted lens (output keys ``*_en`` / ``*_zh``). For the construction
comparison the two slots hold the English-fitted Jacobian lens and the
English-fitted ridge translator; the key names are unchanged.

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
import json
import math
import random
import unicodedata
from pathlib import Path

from sparse_readout_prism.utils import write_csv

MIDBAND = ["21", "24", "26", "29"]
POS = "-1"
CROSS_GROUPS = (
    "antonym_zh",
    "trans_zh2en",
    "trans_en2zh",
    "cloze_zh",
    "cloze_en",
    "fact_zh",
    "exemplar_zh",
)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def script_of(tok):
    for ch in tok:
        if ch.strip() == "" or not ch.isalnum():
            continue
        name = unicodedata.name(ch, "")
        if "CJK" in name:
            return "CJK"
        if "LATIN" in name:
            return "LATIN"
    return "OTHER"


def dom_feat(rec, layer, target):
    t = rec["layers"].get(layer, {}).get(POS, {}).get("targets", {}).get(target)
    if not t or not t.get("top_features"):
        return None
    return t["top_features"][0]["id"]


def majority_same(rec_a, rec_b, target_a, target_b):
    """Majority-of-midband agreement between dom(target_a in rec_a) and dom(target_b in rec_b)."""
    same = total = 0
    for L in MIDBAND:
        fa, fb = dom_feat(rec_a, L, target_a), dom_feat(rec_b, L, target_b)
        if fa is None or fb is None:
            continue
        total += 1
        same += int(fa == fb)
    if total == 0:
        return None
    return same >= (total + 1) // 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lens-a", required=True, help="dump under the English-fitted lens (output keys *_en)")
    p.add_argument("--lens-b", required=True, help="dump under the Chinese-fitted lens (output keys *_zh)")
    p.add_argument("--prompts", required=True, help="EN-ZH prompt bank JSON")
    p.add_argument("--seed", type=int, default=0, help="seed for the shuffled-pairing null")
    p.add_argument("--out", required=True, help="summary JSON (per-group rates, floors, per-prompt rows)")
    p.add_argument("--rows-csv", default=None, help="optional per-prompt rows as CSV")
    args = p.parse_args(argv)

    enr = {r["id"]: r for r in json.loads(Path(args.lens_a).read_text())["records"]}
    zhr = {r["id"]: r for r in json.loads(Path(args.lens_b).read_text())["records"]}
    bank = {v["id"]: v for v in json.loads(Path(args.prompts).read_text())["prompts"]}
    ids = sorted(set(enr) & set(zhr) & set(bank))
    print(f"[agg] {len(ids)} prompts present in both dumps")

    rows = []
    for rid in ids:
        v, e, z = bank[rid], enr[rid], zhr[rid]
        fa, fb = v["form_a"], v["form_b"]
        row = {"id": rid, "group": v["group"], "concept": v.get("concept", "")}
        # headline: same dominant feature for the SAME concept token, lens A vs lens B
        row["cross_lens_pass"] = majority_same(e, z, fa, fa)
        # one feature carries both surface forms, within each lens
        row["cross_form_en"] = majority_same(e, e, fa, fb)
        row["cross_form_zh"] = majority_same(z, z, fa, fb)
        # null floor (a): form_a vs unrelated token, within lens A
        nulls = [majority_same(e, e, fa, nt) for nt in v.get("null_targets", [])]
        nulls = [x for x in nulls if x is not None]
        row["null_hits"] = sum(nulls)
        row["null_total"] = len(nulls)
        # surface script of top-1 under each lens (mid-band vote)
        for tag, rec in (("en", e), ("zh", z)):
            scripts = []
            for L in MIDBAND:
                top5 = rec["layers"].get(L, {}).get(POS, {}).get("top5")
                if top5:
                    scripts.append(script_of(top5[0][0]))
            # Plurality vote; a 2-2 tie resolves to the alphabetically first label
            # (CJK < LATIN < OTHER), fixed so the vote is deterministic.
            row[f"top1_script_{tag}"] = max(sorted(set(scripts)), key=scripts.count) if scripts else "NA"
        rows.append(row)

    # null floor (b): shuffled pairing, form_a of prompt i under lens A vs form_a of j under lens B
    rng = random.Random(args.seed)
    cross_ids = [r["id"] for r in rows if r["group"] in CROSS_GROUPS]
    shuffle_hits = shuffle_total = 0
    for rid in cross_ids:
        others = [x for x in cross_ids if bank[x]["concept"] != bank[rid]["concept"]]
        for oid in rng.sample(others, min(3, len(others))):
            res = majority_same(enr[rid], zhr[oid], bank[rid]["form_a"], bank[oid]["form_a"])
            if res is not None:
                shuffle_total += 1
                shuffle_hits += int(res)

    def rate(sel):
        vals = [r for r in rows if sel(r) and r["cross_lens_pass"] is not None]
        k = sum(r["cross_lens_pass"] for r in vals)
        return k, len(vals), wilson(k, len(vals))

    summary: dict = {"per_group": {}, "rows": rows}
    print("\n=== CROSS-LENS DOMINANT-FEATURE AGREEMENT (majority of mid-band) ===")
    for grp in sorted({r["group"] for r in rows}):
        k, n, (pt, lo, hi) = rate(lambda r, g=grp: r["group"] == g)
        summary["per_group"][grp] = {"pass": k, "n": n, "rate": pt, "ci": [lo, hi]}
        print(f"  {grp:14s} {k:3d}/{n:<3d}  {pt:.2f}  [{lo:.2f}, {hi:.2f}]")
    k, n, (pt, lo, hi) = rate(lambda r: r["group"] in CROSS_GROUPS)
    summary["headline"] = {"pass": k, "n": n, "rate": pt, "ci": [lo, hi]}
    print(f"  {'ALL CROSS':14s} {k:3d}/{n:<3d}  {pt:.2f}  [{lo:.2f}, {hi:.2f}]")

    cf_en = [r["cross_form_en"] for r in rows if r["group"] in CROSS_GROUPS and r["cross_form_en"] is not None]
    cf_zh = [r["cross_form_zh"] for r in rows if r["group"] in CROSS_GROUPS and r["cross_form_zh"] is not None]
    nk = sum(r["null_hits"] for r in rows)
    nn = sum(r["null_total"] for r in rows)
    summary["cross_form_en"] = {"pass": sum(cf_en), "n": len(cf_en)}
    summary["cross_form_zh"] = {"pass": sum(cf_zh), "n": len(cf_zh)}
    summary["null_within_lens"] = {"pass": nk, "n": nn, "rate": wilson(nk, nn)[0]}
    summary["null_shuffle_cross_lens"] = {
        "pass": shuffle_hits,
        "n": shuffle_total,
        "rate": wilson(shuffle_hits, shuffle_total)[0],
    }
    print("\n=== ONE FEATURE CARRIES BOTH FORMS (within-lens) ===")
    print(f"  EN lens: {sum(cf_en)}/{len(cf_en)}   ZH lens: {sum(cf_zh)}/{len(cf_zh)}")
    print("\n=== NULL FLOORS ===")
    print(f"  (a) form vs unrelated token, within-lens: {nk}/{nn} ({wilson(nk, nn)[0]:.2f})")
    print(
        f"  (b) cross-lens shuffled prompts:          {shuffle_hits}/{shuffle_total} "
        f"({wilson(shuffle_hits, shuffle_total)[0]:.2f})"
    )

    print("\n=== LANGUAGE-FOLLOWS-LENS (top-1 script, mid-band vote) ===")
    summary["lens_only_split"] = {}
    for grp in sorted({r["group"] for r in rows}):
        sel = [r for r in rows if r["group"] == grp]
        flip = sum(1 for r in sel if r["top1_script_en"] == "LATIN" and r["top1_script_zh"] == "CJK")
        summary["lens_only_split"][grp] = {"pass": flip, "n": len(sel)}
        print(f"  {grp:14s} EN=Latin & ZH=CJK on {flip}/{len(sel)}")
    cross_sel = [r for r in rows if r["group"] in CROSS_GROUPS]
    flip = sum(1 for r in cross_sel if r["top1_script_en"] == "LATIN" and r["top1_script_zh"] == "CJK")
    summary["lens_only_split"]["all_cross"] = {"pass": flip, "n": len(cross_sel)}
    print(f"  {'ALL CROSS':14s} EN=Latin & ZH=CJK on {flip}/{len(cross_sel)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n[agg] wrote {out}")
    if args.rows_csv:
        write_csv(args.rows_csv, rows)
        print(f"[agg] wrote {args.rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
