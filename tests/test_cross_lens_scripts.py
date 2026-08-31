"""CPU-only checks for the cross-lens study scripts (no GPU, no model, no jlens).

Covers the offline aggregation and table-assembly paths on synthetic dumps:
  * scripts/eval/aggregate_cross_lens_en_zh.py    (wilson, script_of, mid-band vote, end-to-end)
  * scripts/eval/aggregate_cross_lens_en_de.py    (lexical language call, divergence)
  * scripts/data/build_cross_lens_de_bank.py      (normalisation, Levenshtein, bank rules)
  * scripts/analyze/cross_lens_three_lens_prompt.py
  * scripts/figures/compute_cross_lens_shared_feature.py
  * scripts/analyze/cross_lens_antonym_layers.py  (table subcommand)
  * scripts/run/fit_jlens.py                       (source-layer picker)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agg_zh = _load("aggregate_cross_lens_en_zh", "scripts/eval/aggregate_cross_lens_en_zh.py")
agg_de = _load("aggregate_cross_lens_en_de", "scripts/eval/aggregate_cross_lens_en_de.py")
bank_de = _load("build_cross_lens_de_bank", "scripts/data/build_cross_lens_de_bank.py")
three = _load("cross_lens_three_lens_prompt", "scripts/analyze/cross_lens_three_lens_prompt.py")
shared = _load("compute_cross_lens_shared_feature", "scripts/figures/compute_cross_lens_shared_feature.py")
antonym = _load("cross_lens_antonym_layers", "scripts/analyze/cross_lens_antonym_layers.py")
fit_jlens = _load("fit_jlens", "scripts/run/fit_jlens.py")

MIDBAND = ["21", "24", "26", "29"]


# --------------------------------------------------------------------------- #
# synthetic dump builders
# --------------------------------------------------------------------------- #


def _decomp(feats, lens_logit=None, rank=None):
    d = {
        "original_logit": sum(c for _, c in feats) + 0.5,
        "base": -0.1,
        "feature_sum": sum(c for _, c in feats),
        "residual": 0.6,
        "top_features": [{"id": i, "contribution": c} for i, c in feats],
    }
    if lens_logit is not None:
        d["lens_logit"] = lens_logit
        d["lens_rank"] = rank
    return d


def _record(pid, group, top1, dom_by_target, top1_feats=None):
    """One prompt record whose mid-band layers all carry the same readings."""
    layers = {}
    for L in ["2", *MIDBAND]:
        entry = {
            "top5": [[top1, 20.0], ["x", 1.0]],
            "targets": {t: _decomp([(fid, 5.0), (fid + 1000, 0.5)], 10.0, 0) for t, fid in dom_by_target.items()},
        }
        if top1_feats is not None:
            entry["top1_decomp"] = _decomp(top1_feats)
        layers[L] = {"-1": entry}
    return {
        "id": pid,
        "group": group,
        "continuation": top1,
        "intermediate": None,
        "answer": top1.strip(),
        "targets": {t: i for i, t in enumerate(dom_by_target)},
        "layers": layers,
    }


def _dump(records, labels=None):
    return {
        "model": "m",
        "lens": "l",
        "sae": "s",
        "positions": [-1],
        "layers": [2, 21, 24, 26, 29],
        "records": records,
        "feature_top_tokens": labels or {},
    }


# --------------------------------------------------------------------------- #
# EN-ZH aggregator
# --------------------------------------------------------------------------- #


def test_wilson_matches_reference_values():
    p, lo, hi = agg_zh.wilson(77, 80)
    assert abs(p - 0.9625) < 1e-9
    assert round(lo, 2) == 0.90 and round(hi, 2) == 0.99
    assert agg_zh.wilson(0, 0) == (0.0, 0.0, 1.0)


def test_script_of_distinguishes_scripts():
    assert agg_zh.script_of(" London") == "LATIN"
    assert agg_zh.script_of("伦敦") == "CJK"
    assert agg_zh.script_of(" 42") == "OTHER"


def test_majority_same_uses_midband_vote():
    a = _record("p", "g", "t", {"f": 7})
    b = _record("p", "g", "t", {"f": 7})
    assert agg_zh.majority_same(a, b, "f", "f") is True
    # flip two of four mid-band layers: 2/4 still counts as a majority (>= ceil(n/2))
    for L in ["21", "24"]:
        b["layers"][L]["-1"]["targets"]["f"]["top_features"][0]["id"] = 99
    assert agg_zh.majority_same(a, b, "f", "f") is True
    b["layers"]["26"]["-1"]["targets"]["f"]["top_features"][0]["id"] = 99
    assert agg_zh.majority_same(a, b, "f", "f") is False
    assert agg_zh.majority_same(a, b, "missing", "f") is None


def test_en_zh_aggregator_end_to_end(tmp_path):
    bank = {
        "prompts": [
            {
                "id": "a1",
                "group": "antonym_zh",
                "concept": "big",
                "form_a": "大",
                "form_b": " big",
                "null_targets": ["桌"],
            },
            {
                "id": "a2",
                "group": "antonym_zh",
                "concept": "slow",
                "form_a": "慢",
                "form_b": " slow",
                "null_targets": ["门"],
            },
            {"id": "d1", "group": "ctrl_digit", "concept": "7", "form_a": "七", "form_b": " seven", "null_targets": []},
        ]
    }
    en = _dump(
        [
            _record("a1", "antonym_zh", " big", {"大": 1, " big": 1, "桌": 50}),
            _record("a2", "antonym_zh", " slow", {"慢": 2, " slow": 3, "门": 60}),
            _record("d1", "ctrl_digit", " 7", {"七": 4, " seven": 4}),
        ]
    )
    zh = _dump(
        [
            _record("a1", "antonym_zh", "大", {"大": 1, " big": 1, "桌": 50}),
            _record("a2", "antonym_zh", "慢", {"慢": 9, " slow": 3, "门": 60}),
            _record("d1", "ctrl_digit", "七", {"七": 4, " seven": 4}),
        ]
    )
    (tmp_path / "en.json").write_text(json.dumps(en, ensure_ascii=False))
    (tmp_path / "zh.json").write_text(json.dumps(zh, ensure_ascii=False))
    (tmp_path / "bank.json").write_text(json.dumps(bank, ensure_ascii=False))
    out = tmp_path / "summary.json"
    rc = agg_zh.main(
        [
            "--lens-a",
            str(tmp_path / "en.json"),
            "--lens-b",
            str(tmp_path / "zh.json"),
            "--prompts",
            str(tmp_path / "bank.json"),
            "--out",
            str(out),
            "--rows-csv",
            str(tmp_path / "rows.csv"),
        ]
    )
    assert rc == 0
    s = json.loads(out.read_text())
    assert s["headline"] == {"pass": 1, "n": 2, "rate": 0.5, "ci": s["headline"]["ci"]}
    assert s["per_group"]["ctrl_digit"]["pass"] == 1
    assert s["cross_form_en"] == {"pass": 1, "n": 2}
    assert s["null_within_lens"]["pass"] == 0 and s["null_within_lens"]["n"] == 2
    assert s["lens_only_split"]["antonym_zh"] == {"pass": 2, "n": 2}
    assert s["lens_only_split"]["all_cross"] == {"pass": 2, "n": 2}
    assert (tmp_path / "rows.csv").read_text().startswith("id,group,concept,cross_lens_pass")


# --------------------------------------------------------------------------- #
# EN-DE aggregator and bank builder
# --------------------------------------------------------------------------- #


def test_lang_of_lexical_call():
    item = {"form_a": " groß", "form_b": " big", "lang_a": "de", "lang_b": "en"}
    assert agg_de.lang_of(" groß", item) == "DE"
    assert agg_de.lang_of("big", item) == "EN"
    assert agg_de.lang_of(" Küche", item) == "DE"  # umlaut cue
    assert agg_de.lang_of(" hund", item) == "DE"  # word list
    assert agg_de.lang_of(" window", item) == "EN"
    assert agg_de.lang_of(" xyzzy", item) == "OTHER"
    assert agg_de.lang_of("  ", item) == "OTHER"


def test_norm_and_levenshtein_cognate_rule():
    assert bank_de.norm("Käse") == "kase"
    assert bank_de.norm("groß") == "gross"
    assert bank_de.levenshtein("neu", "new") == 1
    assert bank_de.levenshtein("leicht", "light") == 2
    assert bank_de.levenshtein("hund", "dog") > 2


def test_build_bank_applies_rules_with_stub_tokenizer():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            # Everything is one token except any surface containing 'schw'.
            return [0, 1] if "schw" in s.lower() else [0]

    prompts, excluded = bank_de.build_bank(Tok())
    ids = {p["id"] for p in prompts}
    reasons = {e["id"]: e["reason"] for e in excluded}
    assert reasons["antonym_de_13"] == "cognate (lev=1)"  # neu / new
    assert reasons["antonym_de_09"].startswith("not single-token: de")  # schwach
    assert "antonym_de_01" in ids
    first = next(p for p in prompts if p["id"] == "antonym_de_01")
    assert first["form_a"] == "groß" and first["form_b"] == " big"  # quoted frame: no leading space
    assert first["targets"] == [first["form_a"], first["form_b"], *first["null_targets"]]
    assert all(p["lang_b"] != p["lang_a"] for p in prompts)


# --------------------------------------------------------------------------- #
# three-lens table, shared-feature metrics, antonym table
# --------------------------------------------------------------------------- #


def test_three_lens_tables(tmp_path):
    recs = {
        "EN": _record("q", "antonym_de", "large", {" groß": 6764, " large": 12474}, top1_feats=[(12474, 9.0)]),
        "DE": _record("q", "antonym_de", " groß", {" groß": 6764, " large": 12474}, top1_feats=[(6764, 8.0)]),
    }
    top1, targets, feats = three.three_lens_tables(recs, ["24", "29"], None)
    assert [r["top1_token"] for r in top1] == ["large", " groß", "large", " groß"]
    assert {r["target"] for r in targets} == {" groß", " large"}
    dom = {(r["lens"], r["token"]): r["dominant_feature"] for r in feats if r["layer"] == "24"}
    assert dom[("EN", "large")] == 12474 and dom[("DE", " groß")] == 6764
    for label, rec in recs.items():
        (tmp_path / f"{label}.json").write_text(json.dumps(_dump([rec]), ensure_ascii=False))
    rc = three.main(
        [
            "--dump",
            f"EN={tmp_path / 'EN.json'}",
            "--dump",
            f"DE={tmp_path / 'DE.json'}",
            "--layers",
            "24,29",
            "--out-csv",
            str(tmp_path / "t.csv"),
        ]
    )
    assert rc == 0 and (tmp_path / "t.csv").exists()


def test_shared_feature_metrics():
    recs = {
        "EN": _record("f", "fact_zh", " London", {"伦敦": 1}, top1_feats=[(23180, 30.0), (26030, 1.7), (5, 0.3)]),
        "ZH": _record("f", "fact_zh", "伦敦", {"伦敦": 1}, top1_feats=[(23180, 24.0), (5150, 1.5)]),
    }
    m = shared.shared_feature_metrics(recs, "26", {"EN": {"23180": [" London", "伦敦"]}})
    assert m["shared_dominant_feature"] == 23180
    assert abs(m["per_lens"]["EN"]["dominant_share_of_feature_sum"] - 30.0 / 32.0) < 1e-9
    assert m["per_lens"]["EN"]["largest_other_contribution"] == 1.7
    assert m["shared_feature_share_of_feature_sum"]["ZH"] == 24.0 / 25.5
    assert m["shared_feature_top_tokens"] == [" London", "伦敦"]
    recs["ZH"]["layers"]["26"]["-1"]["top1_decomp"]["top_features"][0]["id"] = 7
    assert shared.shared_feature_metrics(recs, "26", {})["shared_dominant_feature"] is None


def test_antonym_layer_table():
    def payload(tok, pct, fid):
        rows = {
            L: {
                "jlens_top10": [[tok, pct], ["x", 1.0]],
                "logitlens_top10": [["y", 50.0]],
                "targets": {"大": _decomp([(fid, 12.5)]), "big": _decomp([(fid, 5.0)])},
            }
            for L in ["24", "29", "final"]
        }
        return {"prompt": "p", "rows": rows}

    table, feats = antonym.antonym_layer_table(
        {"EN": payload("large", 39.7, 112), "ZH": payload("大的", 36.4, 112)}, ["24", "final"]
    )
    assert [(r["layer"], r["lens"], r["top1_token"], r["top1_softmax_pct"]) for r in table][:2] == [
        ("24", "EN", "large", 39.7),
        ("24", "ZH", "大的", 36.4),
    ]
    assert all(r["dominant_feature"] == 112 for r in feats)
    assert antonym.parse_dump_args(["EN=a.json", "ZH=b.json"]) == [("EN", Path("a.json")), ("ZH", Path("b.json"))]


def test_pick_source_layers_matches_paper_layer_set():
    assert fit_jlens.pick_source_layers(32, 12) == [2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 29]
    assert fit_jlens.pick_source_layers(12, 4) == [2, 4, 7, 9]
