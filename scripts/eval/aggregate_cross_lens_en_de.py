#!/usr/bin/env python3
"""Aggregate an English-German cross-lens sweep: same metrics as the EN-ZH aggregator, lexical language call.

Reads two readout dumps from ``scripts/run/run_cross_lens_readouts.py`` (the same
frozen Qwen3.5-9B states read through the English- and German-fitted lenses) and
the EN-DE prompt bank, and computes the EN-DE rows of
``tab:app-cross-lens-extension`` / ``fig:cross-lens-extension`` and the per-family
counts of the second-language-pair paragraph: cross-lens dominant-feature
agreement (majority of the mid-band layers {21, 24, 26, 29}, Wilson 95% CIs),
within-lens cross-form agreement, the unrelated-token and shuffled-pairing null
floors, and the token divergence rate (the two lenses' top-1 strings differ on a
majority of mid-band layers).

Both languages share the Latin script, so the surface-language call for a top-1
token is lexical rather than by script. In order: (1) membership in the bank
item's own gold forms (the form in the item's language); (2) an umlaut / eszett
cue means German; (3) membership in small embedded EN / DE common-word lists
(casefolded, diacritic-normalised); else OTHER. The per-prompt call is the
mid-band plurality vote (a 2-2 tie resolves to the alphabetically first label,
DE < EN < OTHER). The headline agreement metric never uses the language call;
it is a descriptive diagnostic.

Paper runs (``--prompts data/cross_lens/cross_lens_prompts_en_de.json --seed 0``)::

  # 100-prompt lenses: extension-matrix row 2
  aggregate_cross_lens_en_de.py --lens-a <dumps>/en_de__jlens_en_n100.json \
      --lens-b <dumps>/en_de__jlens_de_n100.json --out <out>/en_de_jlens_n100_summary.json
  # 300-prompt refits: row 6
  aggregate_cross_lens_en_de.py --lens-a <dumps>/en_de__jlens_en_n300.json \
      --lens-b <dumps>/en_de__jlens_de_n300.json --out <out>/en_de_jlens_n300_summary.json
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
    "antonym_de",
    "trans_de2en",
    "trans_en2de",
    "cloze_de",
    "cloze_en",
    "fact_de",
    "exemplar_de",
)

EN_WORDS = set(
    """the of and to in is was for on as with by at from it an be this that or
    are not his her their they them he she you we all one two more no yes big
    small slow bright light bad empty dry weak dark late cheap new quiet dog
    tree table chair window horse bird mountain rain cheese egg door flower
    knife train city forest sea fire newspaper kitchen sister pen station book
    snow coffee key head money voice war week year tooth island church school
    ship path car work child woman brother song game time clock street village
    meat fruit sugar eye mouth lake sky water day night house man word for is
    """.split()
)
DE_WORDS = set(
    """der die das und ist war für auf als mit von bei aus es ein eine sein
    nicht sie er wir ihr alle einen einem einer dem den groß klein langsam
    hell leicht schlecht leer trocken schwach dunkel spät billig neu leise
    hund baum tisch stuhl fenster pferd vogel berg regen käse ei tür blume
    messer zug stadt wald meer feuer zeitung küche schwester stift bahnhof
    buch schnee kaffee schlüssel kopf geld stimme krieg woche jahr zahn insel
    kirche schule schiff weg wagen arbeit kind frau bruder lied spiel zeit
    uhr straße dorf fleisch obst zucker auge mund see himmel wasser tag nacht
    haus mann wort löwe biene mond arzt lehrer mantel hemd saft
    """.split()
)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def norm(s):
    s = s.strip().casefold().replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def lang_of(tok, item):
    t = tok.strip()
    if not t:
        return "OTHER"
    n = norm(t)
    forms = {
        "de": item["form_a"] if item["lang_a"] == "de" else item["form_b"],
        "en": item["form_a"] if item["lang_a"] == "en" else item["form_b"],
    }
    if n == norm(forms["de"]):
        return "DE"
    if n == norm(forms["en"]):
        return "EN"
    if any(c in t for c in "äöüÄÖÜß"):
        return "DE"
    in_de, in_en = n in DE_WORDS, n in EN_WORDS
    if in_de and not in_en:
        return "DE"
    if in_en and not in_de:
        return "EN"
    return "OTHER"


def dom_feat(rec, layer, target):
    t = rec["layers"].get(layer, {}).get(POS, {}).get("targets", {}).get(target)
    if not t or not t.get("top_features"):
        return None
    return t["top_features"][0]["id"]


def majority_same(rec_a, rec_b, target_a, target_b):
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
    p.add_argument("--lens-b", required=True, help="dump under the German-fitted lens (output keys *_de)")
    p.add_argument("--prompts", required=True, help="EN-DE prompt bank JSON")
    p.add_argument("--seed", type=int, default=0, help="seed for the shuffled-pairing null")
    p.add_argument("--out", required=True, help="summary JSON (per-group rates, floors, per-prompt rows)")
    p.add_argument("--rows-csv", default=None, help="optional per-prompt rows as CSV")
    args = p.parse_args(argv)

    ar = {r["id"]: r for r in json.loads(Path(args.lens_a).read_text())["records"]}
    br = {r["id"]: r for r in json.loads(Path(args.lens_b).read_text())["records"]}
    bank_list = json.loads(Path(args.prompts).read_text())["prompts"]
    assert len({v["id"] for v in bank_list}) == len(bank_list), "duplicate prompt ids in bank"
    bank = {v["id"]: v for v in bank_list}
    ids = sorted(set(ar) & set(br) & set(bank))
    print(f"[agg] {len(ids)} prompts present in both dumps")

    rows = []
    for rid in ids:
        v, a, b = bank[rid], ar[rid], br[rid]
        fa, fb = v["form_a"], v["form_b"]
        row = {"id": rid, "group": v["group"], "concept": v.get("concept", "")}
        row["cross_lens_pass"] = majority_same(a, b, fa, fa)
        row["cross_form_en"] = majority_same(a, a, fa, fb)
        row["cross_form_de"] = majority_same(b, b, fa, fb)
        nulls = [majority_same(a, a, fa, nt) for nt in v.get("null_targets", [])]
        nulls = [x for x in nulls if x is not None]
        row["null_hits"] = sum(nulls)
        row["null_total"] = len(nulls)
        # top-1 language under each lens (mid-band vote), lexical call
        for tag, rec in (("en", a), ("de", b)):
            langs = []
            for L in MIDBAND:
                top5 = rec["layers"].get(L, {}).get(POS, {}).get("top5")
                if top5:
                    langs.append(lang_of(top5[0][0], v))
            # Plurality vote; a 2-2 tie resolves to the alphabetically first label
            # (DE < EN < OTHER), fixed so the vote is deterministic.
            row[f"top1_lang_{tag}"] = max(sorted(set(langs)), key=langs.count) if langs else "NA"
        # divergence: do the two lenses' top-1 token strings differ (mid-band vote)?
        diff = tot = 0
        for L in MIDBAND:
            ta = a["layers"].get(L, {}).get(POS, {}).get("top5")
            tb = b["layers"].get(L, {}).get(POS, {}).get("top5")
            if ta and tb:
                tot += 1
                diff += int(ta[0][0] != tb[0][0])
        row["token_diverges"] = (diff >= (tot + 1) // 2) if tot else None
        rows.append(row)

    rng = random.Random(args.seed)
    cross_ids = [r["id"] for r in rows if r["group"] in CROSS_GROUPS]
    shuffle_hits = shuffle_total = 0
    for rid in cross_ids:
        others = [x for x in cross_ids if bank[x]["concept"] != bank[rid]["concept"]]
        for oid in rng.sample(others, min(3, len(others))):
            res = majority_same(ar[rid], br[oid], bank[rid]["form_a"], bank[oid]["form_a"])
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
    cf_de = [r["cross_form_de"] for r in rows if r["group"] in CROSS_GROUPS and r["cross_form_de"] is not None]
    nk = sum(r["null_hits"] for r in rows)
    nn = sum(r["null_total"] for r in rows)
    summary["cross_form_en"] = {"pass": sum(cf_en), "n": len(cf_en)}
    summary["cross_form_de"] = {"pass": sum(cf_de), "n": len(cf_de)}
    summary["null_within_lens"] = {"pass": nk, "n": nn, "rate": wilson(nk, nn)[0]}
    summary["null_shuffle_cross_lens"] = {
        "pass": shuffle_hits,
        "n": shuffle_total,
        "rate": wilson(shuffle_hits, shuffle_total)[0],
    }
    div = [r["token_diverges"] for r in rows if r["group"] in CROSS_GROUPS and r["token_diverges"] is not None]
    summary["divergence_rate"] = {"diverging": sum(div), "n": len(div)}
    print("\n=== ONE FEATURE CARRIES BOTH FORMS (within-lens) ===")
    print(f"  EN lens: {sum(cf_en)}/{len(cf_en)}   DE lens: {sum(cf_de)}/{len(cf_de)}")
    print("\n=== NULL FLOORS ===")
    print(f"  (a) form vs unrelated token, within-lens: {nk}/{nn} ({wilson(nk, nn)[0]:.2f})")
    print(
        f"  (b) cross-lens shuffled prompts:          {shuffle_hits}/{shuffle_total} "
        f"({wilson(shuffle_hits, shuffle_total)[0]:.2f})"
    )
    print("\n=== TOKEN DIVERGENCE (descriptive) ===")
    print(f"  top-1 differs between lenses on {sum(div)}/{len(div)} cross prompts")
    print("\n=== LANGUAGE-FOLLOWS-LENS (top-1 lexical call, mid-band vote) ===")
    summary["lens_only_split"] = {}
    for grp in sorted({r["group"] for r in rows}):
        sel = [r for r in rows if r["group"] == grp]
        flip = sum(1 for r in sel if r["top1_lang_en"] == "EN" and r["top1_lang_de"] == "DE")
        summary["lens_only_split"][grp] = {"pass": flip, "n": len(sel)}
        print(f"  {grp:14s} EN-lens=EN & DE-lens=DE on {flip}/{len(sel)}")
    cross_sel = [r for r in rows if r["group"] in CROSS_GROUPS]
    flip = sum(1 for r in cross_sel if r["top1_lang_en"] == "EN" and r["top1_lang_de"] == "DE")
    summary["lens_only_split"]["all_cross"] = {"pass": flip, "n": len(cross_sel)}
    print(f"  {'ALL CROSS':14s} EN-lens=EN & DE-lens=DE on {flip}/{len(cross_sel)}")

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
