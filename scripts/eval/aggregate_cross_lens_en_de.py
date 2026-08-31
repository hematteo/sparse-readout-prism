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
(casefolded, diacritic-normalised through ``research.cross_lens.fold_text``);
else OTHER. The per-prompt call is the mid-band plurality vote (a 2-2 tie
resolves to the alphabetically first label, DE < EN < OTHER). The headline
agreement metric never uses the language call; it is a descriptive diagnostic.

Options that change the population, not the arithmetic:
  - ``--agreement-rule half`` (default, the paper) passes a vote on at least
    ``ceil(n/2)`` scored mid-band layers (2 of 4); ``strict`` needs more than
    half (3 of 4). Applies to every vote, divergence included.
  - ``--null-population all`` (default, the paper) pools the unrelated-token
    null over every bank item, controls included (204 comparisons on the
    shipped bank); ``cross`` drops the 12 controls (180 comparisons), which
    matches the EN-ZH population, whose controls carry no nulls.

The metric bodies live in ``research.cross_lens`` (``aggregate_pair``); this
entry point supplies the EN-DE families, the lexical call and the divergence
column. The summary JSON carries a ``provenance`` block.

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

from sparse_readout_prism.research.cross_lens import (
    AGREEMENT_RULES,
    NULL_POPULATIONS,
    PairSpec,
    aggregate_pair,
    fold_text,
    load_bank_items,
    load_dump_records,
    print_summary,
    write_summary,
)
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import write_csv

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


def lang_of(tok: str, item: dict) -> str:
    t = tok.strip()
    if not t:
        return "OTHER"
    n = fold_text(t)
    forms = {
        "de": item["form_a"] if item["lang_a"] == "de" else item["form_b"],
        "en": item["form_a"] if item["lang_a"] == "en" else item["form_b"],
    }
    if n == fold_text(forms["de"]):
        return "DE"
    if n == fold_text(forms["en"]):
        return "EN"
    if any(c in t for c in "äöüÄÖÜß"):
        return "DE"
    in_de, in_en = n in DE_WORDS, n in EN_WORDS
    if in_de and not in_en:
        return "DE"
    if in_en and not in_de:
        return "EN"
    return "OTHER"


SPEC = PairSpec(
    tag_b="de",
    cross_groups=CROSS_GROUPS,
    surface_column="lang",
    surface_call=lang_of,
    split_labels=("EN", "DE"),
    split_caption="EN-lens=EN & DE-lens=DE",
    surface_caption="top-1 lexical call",
    divergence=True,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lens-a", required=True, help="dump under the English-fitted lens (output keys *_en)")
    p.add_argument("--lens-b", required=True, help="dump under the German-fitted lens (output keys *_de)")
    p.add_argument("--prompts", required=True, help="EN-DE prompt bank JSON")
    p.add_argument("--seed", type=int, default=0, help="seed for the shuffled-pairing null")
    p.add_argument(
        "--agreement-rule",
        choices=AGREEMENT_RULES,
        default="half",
        help="mid-band majority: half = at least ceil(n/2) layers (paper), strict = more than half",
    )
    p.add_argument(
        "--null-population",
        choices=NULL_POPULATIONS,
        default="all",
        help="rows pooled into the unrelated-token floor: all = every bank item incl. controls (paper, /204), "
        "cross = cross families only (/180, the EN-ZH population)",
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
        null_population=args.null_population,
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
