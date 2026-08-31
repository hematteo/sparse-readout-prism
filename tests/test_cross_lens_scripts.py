"""CPU-only checks for the cross-lens study scripts and their shared toolkit (no GPU, no model, no jlens).

Covers, on synthetic dumps and tensors:
  * sparse_readout_prism.research.cross_lens    (both majority rules, aggregation core and its key order,
    decomposition pinned to the pre-0.2.1 inline formula, raw-decode labels, k / centering resolution,
    final-norm lookup, resume sidecars, text folding)
  * scripts/eval/aggregate_cross_lens_en_zh.py    (script_of, end-to-end, --agreement-rule)
  * scripts/eval/aggregate_cross_lens_en_de.py    (lexical call, divergence, --null-population)
  * scripts/data/build_cross_lens_de_bank.py      (Levenshtein, bank rules)
  * scripts/analyze/cross_lens_three_lens_prompt.py
  * scripts/figures/compute_cross_lens_shared_feature.py
  * scripts/analyze/cross_lens_antonym_layers.py  (table subcommand)
  * scripts/run/fit_jlens.py                       (source-layer picker, dim-batch fallback, shard sidecars)
  * scripts/run/fit_ridge_lens.py                  (--holdout guard, diagnostic prompt slice)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from conftest import load_script
from torch import nn

from sparse_readout_prism.research import cross_lens

agg_zh = load_script("scripts/eval/aggregate_cross_lens_en_zh.py")
agg_de = load_script("scripts/eval/aggregate_cross_lens_en_de.py")
bank_de = load_script("scripts/data/build_cross_lens_de_bank.py")
three = load_script("scripts/analyze/cross_lens_three_lens_prompt.py")
shared = load_script("scripts/figures/compute_cross_lens_shared_feature.py")
antonym = load_script("scripts/analyze/cross_lens_antonym_layers.py")
fit_jlens = load_script("scripts/run/fit_jlens.py")
fit_ridge = load_script("scripts/run/fit_ridge_lens.py")

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


def _set_dom(rec, layers, target, fid):
    for L in layers:
        rec["layers"][L]["-1"]["targets"][target]["top_features"][0]["id"] = fid


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


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload, ensure_ascii=False))
    return str(p)


# --------------------------------------------------------------------------- #
# votes and rules
# --------------------------------------------------------------------------- #


def test_wilson_matches_reference_values():
    p, lo, hi = cross_lens.wilson(77, 80)
    assert abs(p - 0.9625) < 1e-9
    assert round(lo, 2) == 0.90 and round(hi, 2) == 0.99
    assert cross_lens.wilson(0, 0) == (0.0, 0.0, 1.0)


def test_script_of_distinguishes_scripts():
    assert agg_zh.script_of(" London") == "LATIN"
    assert agg_zh.script_of("伦敦") == "CJK"
    assert agg_zh.script_of(" 42") == "OTHER"


def test_majority_same_half_and_strict_rules():
    a = _record("p", "g", "t", {"f": 7})
    b = _record("p", "g", "t", {"f": 7})
    assert cross_lens.majority_same(a, b, "f", "f", rule="half") is True
    assert cross_lens.majority_same(a, b, "f", "f", rule="strict") is True
    # 2 of 4 agreeing layers: half (>= ceil(n/2), the paper) passes, strict (> n/2) does not
    _set_dom(b, ["21", "24"], "f", 99)
    assert cross_lens.majority_same(a, b, "f", "f", rule="half") is True
    assert cross_lens.majority_same(a, b, "f", "f", rule="strict") is False
    # 3 of 4 agreeing: both pass
    _set_dom(b, ["24"], "f", 7)
    assert cross_lens.majority_same(a, b, "f", "f", rule="strict") is True
    # 1 of 4 agreeing: neither
    _set_dom(b, ["24", "26"], "f", 99)
    assert cross_lens.majority_same(a, b, "f", "f", rule="half") is False
    assert cross_lens.majority_same(a, b, "missing", "f", rule="half") is None
    with pytest.raises(ValueError, match="agreement rule"):
        cross_lens.majority_same(a, b, "f", "f", rule="most")


def test_token_diverges_rules():
    a = _record("p", "g", " dog", {"f": 1})
    b = _record("p", "g", " dog", {"f": 1})
    assert cross_lens.token_diverges(a, b, rule="half") is False
    for L in ["21", "24"]:
        b["layers"][L]["-1"]["top5"][0][0] = " Hund"
    assert cross_lens.token_diverges(a, b, rule="half") is True
    assert cross_lens.token_diverges(a, b, rule="strict") is False
    assert cross_lens.token_diverges({"layers": {}}, b, rule="half") is None


# --------------------------------------------------------------------------- #
# EN-ZH aggregator
# --------------------------------------------------------------------------- #


def _en_zh_case(tmp_path):
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
    return bank, en, zh


def test_en_zh_aggregator_end_to_end(tmp_path):
    bank, en, zh = _en_zh_case(tmp_path)
    out = tmp_path / "summary.json"
    rc = agg_zh.main(
        [
            "--lens-a",
            _write(tmp_path, "en.json", en),
            "--lens-b",
            _write(tmp_path, "zh.json", zh),
            "--prompts",
            _write(tmp_path, "bank.json", bank),
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
    # the dump format: key order of the summary and of each row, provenance last
    assert list(s) == [
        "per_group",
        "rows",
        "headline",
        "cross_form_en",
        "cross_form_zh",
        "null_within_lens",
        "null_shuffle_cross_lens",
        "lens_only_split",
        "provenance",
    ]
    assert list(s["rows"][0]) == [
        "id",
        "group",
        "concept",
        "cross_lens_pass",
        "cross_form_en",
        "cross_form_zh",
        "null_hits",
        "null_total",
        "top1_script_en",
        "top1_script_zh",
    ]
    assert s["provenance"]["args"]["agreement_rule"] == "half"
    assert (tmp_path / "rows.csv").read_text().startswith("id,group,concept,cross_lens_pass")


def test_en_zh_aggregator_strict_rule_flag(tmp_path):
    bank, en, zh = _en_zh_case(tmp_path)
    # a1 now agrees on exactly 2 of 4 mid-band layers: passes under half, fails under strict
    _set_dom(zh["records"][0], ["21", "24"], "大", 77)
    argv = [
        "--lens-a",
        _write(tmp_path, "en.json", en),
        "--lens-b",
        _write(tmp_path, "zh.json", zh),
        "--prompts",
        _write(tmp_path, "bank.json", bank),
    ]
    agg_zh.main([*argv, "--out", str(tmp_path / "half.json")])
    agg_zh.main([*argv, "--out", str(tmp_path / "strict.json"), "--agreement-rule", "strict"])
    half = json.loads((tmp_path / "half.json").read_text())
    strict = json.loads((tmp_path / "strict.json").read_text())
    assert half["headline"]["pass"] == 1 and strict["headline"]["pass"] == 0
    assert strict["provenance"]["args"]["agreement_rule"] == "strict"


def test_load_bank_items_rejects_duplicate_ids(tmp_path):
    p = _write(tmp_path, "dupe.json", {"prompts": [{"id": "x"}, {"id": "x"}]})
    with pytest.raises(ValueError, match="duplicate prompt ids"):
        cross_lens.load_bank_items(p)


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


def test_en_de_null_population_and_divergence(tmp_path):
    bank = {
        "prompts": [
            {
                "id": "t1",
                "group": "trans_de2en",
                "concept": "dog",
                "form_a": " dog",
                "form_b": " Hund",
                "lang_a": "en",
                "lang_b": "de",
                "null_targets": [" Salz", " salt"],
            },
            {
                "id": "c1",
                "group": "ctrl_digit",
                "concept": "seven",
                "form_a": "7",
                "form_b": "7",
                "lang_a": "de",
                "lang_b": "en",
                "null_targets": [" Schuh", " shoe"],
            },
        ]
    }
    # the control's first null shares the dominant feature with form_a: one null hit, on a control row
    a = _dump(
        [
            _record("t1", "trans_de2en", " dog", {" dog": 1, " Hund": 1, " Salz": 50, " salt": 51}),
            _record("c1", "ctrl_digit", "7", {"7": 4, " Schuh": 4, " shoe": 60}),
        ]
    )
    b = _dump(
        [
            _record("t1", "trans_de2en", " Hund", {" dog": 1, " Hund": 1, " Salz": 50, " salt": 51}),
            _record("c1", "ctrl_digit", "7", {"7": 4, " Schuh": 4, " shoe": 60}),
        ]
    )
    argv = [
        "--lens-a",
        _write(tmp_path, "a.json", a),
        "--lens-b",
        _write(tmp_path, "b.json", b),
        "--prompts",
        _write(tmp_path, "bank.json", bank),
    ]
    agg_de.main([*argv, "--out", str(tmp_path / "all.json")])
    agg_de.main([*argv, "--out", str(tmp_path / "cross.json"), "--null-population", "cross"])
    s_all = json.loads((tmp_path / "all.json").read_text())
    s_cross = json.loads((tmp_path / "cross.json").read_text())
    # default pools every bank item (controls included); cross drops them
    assert s_all["null_within_lens"] == {"pass": 1, "n": 4, "rate": 0.25}
    assert s_cross["null_within_lens"] == {"pass": 0, "n": 2, "rate": 0.0}
    # per-row null counts are unchanged by the population switch
    assert [(r["null_hits"], r["null_total"]) for r in s_all["rows"]] == [(1, 2), (0, 2)]
    assert s_all["rows"] == s_cross["rows"]
    assert s_all["headline"] == {"pass": 1, "n": 1, "rate": 1.0, "ci": s_all["headline"]["ci"]}
    assert s_all["divergence_rate"] == {"diverging": 1, "n": 1}
    assert s_all["lens_only_split"] == {
        "ctrl_digit": {"pass": 0, "n": 1},
        "trans_de2en": {"pass": 1, "n": 1},
        "all_cross": {"pass": 1, "n": 1},
    }
    assert list(s_all) == [
        "per_group",
        "rows",
        "headline",
        "cross_form_en",
        "cross_form_de",
        "null_within_lens",
        "null_shuffle_cross_lens",
        "divergence_rate",
        "lens_only_split",
        "provenance",
    ]
    assert list(s_all["rows"][0]) == [
        "id",
        "group",
        "concept",
        "cross_lens_pass",
        "cross_form_en",
        "cross_form_de",
        "null_hits",
        "null_total",
        "top1_lang_en",
        "top1_lang_de",
        "token_diverges",
    ]
    assert s_cross["provenance"]["args"]["null_population"] == "cross"


def test_fold_text_and_levenshtein_cognate_rule():
    assert cross_lens.fold_text("Käse") == "kase"
    assert cross_lens.fold_text("groß") == "gross"
    assert cross_lens.fold_text(" Straße ") == "strasse"
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
    assert first["null_targets"] == [" Salz", " salt"]  # space-prefixed, language-matched, round-robin start
    assert all(p["lang_b"] != p["lang_a"] for p in prompts)
    ctrl = next(p for p in prompts if p["group"] == "ctrl_digit")
    assert ctrl["form_a"] == ctrl["form_b"] and ctrl["targets"][:2] == [ctrl["form_a"]] * 2


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
        _write(tmp_path, f"{label}.json", _dump([rec]))
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
    manifest = json.loads((tmp_path / "t.manifest.json").read_text())
    assert manifest["lenses"] == ["EN", "DE"] and "provenance" in manifest


def test_shared_feature_metrics(tmp_path):
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
    for label, rec in recs.items():
        _write(tmp_path, f"{label}.json", _dump([rec]))
    rc = shared.main(
        [
            "--dump",
            f"EN={tmp_path / 'EN.json'}",
            "--dump",
            f"ZH={tmp_path / 'ZH.json'}",
            "--prompt-id",
            "f",
            "--out-json",
            str(tmp_path / "m.json"),
        ]
    )
    assert rc == 0
    written = json.loads((tmp_path / "m.json").read_text())
    assert written["shared_dominant_feature"] == 23180 and written["provenance"]["args"]["layer"] == "26"
    recs["ZH"]["layers"]["26"]["-1"]["top1_decomp"]["top_features"][0]["id"] = 7
    assert shared.shared_feature_metrics(recs, "26", {})["shared_dominant_feature"] is None


def test_antonym_layer_table(tmp_path):
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

    dumps = {"EN": payload("large", 39.7, 112), "ZH": payload("大的", 36.4, 112)}
    table, feats = antonym.antonym_layer_table(dumps, ["24", "final"])
    assert [(r["layer"], r["lens"], r["top1_token"], r["top1_softmax_pct"]) for r in table][:2] == [
        ("24", "EN", "large", 39.7),
        ("24", "ZH", "大的", 36.4),
    ]
    assert all(r["dominant_feature"] == 112 for r in feats)
    assert cross_lens.parse_dump_args(["EN=a.json", "ZH=b.json"]) == [("EN", Path("a.json")), ("ZH", Path("b.json"))]
    with pytest.raises(ValueError, match="LABEL=path"):
        cross_lens.parse_dump_args(["a.json"])
    # the table subcommand end to end: CSVs plus a sibling manifest (args carry the subcommand callable)
    for label, d in dumps.items():
        _write(tmp_path, f"{label}.json", d)
    rc = antonym.main(
        [
            "table",
            "--dump",
            f"EN={tmp_path / 'EN.json'}",
            "--dump",
            f"ZH={tmp_path / 'ZH.json'}",
            "--layers",
            "24,final",
            "--out-csv",
            str(tmp_path / "table.csv"),
            "--out-features-csv",
            str(tmp_path / "features.csv"),
        ]
    )
    assert rc == 0 and (tmp_path / "features.csv").exists()
    manifest = json.loads((tmp_path / "table.manifest.json").read_text())
    assert manifest["lenses"] == ["EN", "ZH"] and manifest["provenance"]["args"]["layers"] == "24,final"
    assert "fn" not in manifest["provenance"]["args"]


# --------------------------------------------------------------------------- #
# decomposition, labels, k and centering resolution, final norm
# --------------------------------------------------------------------------- #


def _dictionary(seed=0, V=12, d=6, D=9):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, d, generator=g)
    decoder = torch.randn(D, d, generator=g)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)
    encoder_w = torch.randn(D, d, generator=g)
    encoder_b = 0.1 * torch.randn(D, generator=g)
    h = torch.randn(d, generator=g)
    return W, decoder, encoder_w, encoder_b, h


def test_decompose_token_pins_pre_0_2_1_inline_formula():
    W, decoder, encoder_w, encoder_b, h = _dictionary()
    row_mean = W.mean(dim=0)
    k, top_feats, token_id = 3, 2, 5

    # the block both GPU scripts inlined before 0.2.1
    W_row = W[token_id]
    centered = W_row - row_mean
    norm = centered.norm().clamp_min(1e-8)
    acts = F.relu((centered / norm)[None, :] @ encoder_w.T + encoder_b)
    values, indices = torch.topk(acts, k=k, dim=-1)
    code = torch.zeros_like(acts).scatter_(-1, indices, values)[0]
    contributions = norm * code * (h @ decoder.T)
    active = torch.nonzero(contributions != 0).flatten()
    top = active[contributions[active].abs().argsort(descending=True)[:top_feats]]
    expected = {
        "original_logit": float(h @ W_row),
        "base": float(h @ row_mean),
        "feature_sum": float(contributions.sum()),
        "top_features": [{"id": int(f), "contribution": float(contributions[f])} for f in top],
    }
    expected["residual"] = expected["original_logit"] - expected["base"] - expected["feature_sum"]

    got = cross_lens.decompose_token(
        h,
        token_id,
        W=W,
        row_mean=row_mean,
        decoder=decoder,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        k=k,
        top_feats=top_feats,
    )
    assert list(got) == ["original_logit", "base", "feature_sum", "residual", "top_features"]
    for key in ("original_logit", "base", "feature_sum", "residual"):
        assert got[key] == pytest.approx(expected[key], abs=1e-6)
    assert [f["id"] for f in got["top_features"]] == [f["id"] for f in expected["top_features"]]
    assert [f["contribution"] for f in got["top_features"]] == pytest.approx(
        [f["contribution"] for f in expected["top_features"]], abs=1e-6
    )
    assert len(got["top_features"]) == top_feats
    assert got["original_logit"] == pytest.approx(got["base"] + got["feature_sum"] + got["residual"])


def test_feature_top_tokens_keeps_raw_decode_labels():
    W, _decoder, encoder_w, encoder_b, _h = _dictionary(seed=1)
    row_mean = W.mean(dim=0)

    class Tok:
        def decode(self, ids):
            return f"<{ids[0]}>"

    labels = cross_lens.feature_top_tokens(W, row_mean, [4, 1, 4], encoder_w, encoder_b, Tok(), top_tokens=3, chunk=5)
    assert list(labels) == [1, 4]  # sorted, de-duplicated feature ids
    _norms, x = cross_lens.center_normalize_rows(W, row_mean)
    for fid, got in labels.items():
        scores = F.relu(x @ encoder_w[fid] + encoder_b[fid])
        order = scores.argsort(descending=True)[:3]
        assert got == [f"<{int(i)}>" for i in order if scores[i] > 0]
    assert cross_lens.feature_top_tokens(W, row_mean, [], encoder_w, encoder_b, Tok()) == {}


def test_resolve_k_defaults_to_checkpoint_and_warns_on_mismatch():
    config = {"factorizer": {"k": 128}}
    assert cross_lens.resolve_k(None, config) == 128
    assert cross_lens.resolve_k(128, config) == 128
    with pytest.warns(UserWarning, match="differs from the checkpoint"):
        assert cross_lens.resolve_k(64, config) == 64
    with pytest.raises(ValueError, match="factorizer.k"):
        cross_lens.resolve_k(None, {})
    assert cross_lens.resolve_k(32, {}) == 32


class _Tok:
    def __init__(self, n, special=()):
        self._n, self.all_special_ids = n, list(special)

    def __len__(self):
        return self._n


def test_centering_trained_equals_live_when_mask_keeps_every_row():
    W, *_ = _dictionary(seed=2)
    live = cross_lens.centering_row_mean(W, "live", tok=_Tok(W.shape[0]), ckpt_row_mean=None)
    trained = cross_lens.centering_row_mean(W, "trained", tok=_Tok(W.shape[0]), ckpt_row_mean=None)
    assert torch.equal(live, W.mean(dim=0)) and torch.equal(trained, live)
    masked = cross_lens.centering_row_mean(W, "trained", tok=_Tok(W.shape[0], special=[3]), ckpt_row_mean=None)
    assert torch.equal(masked, W[[i for i in range(W.shape[0]) if i != 3]].mean(dim=0))
    stored = torch.full((W.shape[1],), 0.25)
    assert torch.equal(cross_lens.centering_row_mean(W, "trained", tok=_Tok(1), ckpt_row_mean=stored), stored)
    assert torch.equal(cross_lens.centering_row_mean(W, "live", tok=_Tok(1), ckpt_row_mean=stored), live)
    with pytest.raises(ValueError, match="centering mode"):
        cross_lens.centering_row_mean(W, "mean", tok=_Tok(1), ckpt_row_mean=None)


def test_find_final_norm_checks_known_paths():
    def wrap(path):
        root = nn.Module()
        node = root
        parts = path.split(".")
        for part in parts[:-1]:
            child = nn.Module()
            setattr(node, part, child)
            node = child
        setattr(node, parts[-1], nn.LayerNorm(4))
        return root

    for path in ("model.norm", "model.language_model.norm", "language_model.norm", "norm"):
        found = cross_lens.find_final_norm(wrap(path))
        assert isinstance(found, nn.LayerNorm), path
    with pytest.raises(RuntimeError, match="final norm"):
        cross_lens.find_final_norm(nn.Linear(2, 2))


# --------------------------------------------------------------------------- #
# lens fitters: layer picker, OOM fallback, resume sidecars, holdout guard
# --------------------------------------------------------------------------- #


def test_pick_source_layers_matches_paper_layer_set():
    assert fit_jlens.pick_source_layers(32, 12) == [2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 29]
    assert fit_jlens.pick_source_layers(12, 4) == [2, 4, 7, 9]


def test_dim_batch_fallback_halves_down_to_one(tmp_path):
    assert fit_jlens.dim_batch_schedule(8) == [8, 4, 2, 1]
    assert fit_jlens.dim_batch_schedule(3) == [3, 1]
    assert fit_jlens.dim_batch_schedule(1) == [1]
    with pytest.raises(ValueError):
        fit_jlens.dim_batch_schedule(0)

    attempts = []

    def fake_fit(model, prompts, *, source_layers, dim_batch, checkpoint_path, resume):
        attempts.append(dim_batch)
        if dim_batch > 2:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return SimpleNamespace(n_prompts=len(prompts), dim_batch=dim_batch, ckpt=checkpoint_path, resume=resume)

    lens = fit_jlens.fit_with_fallback(fake_fit, "model", ["p1", "p2"], [2, 5], 8, tmp_path / "shard0.ckpt.pt")
    assert attempts == [8, 4, 2] and lens.dim_batch == 2 and lens.resume is True

    attempts.clear()

    def always_oom(*a, **k):
        attempts.append(k["dim_batch"])
        raise torch.cuda.OutOfMemoryError("fake OOM")

    with pytest.raises(RuntimeError, match="all dim_batch fallbacks OOMed"):
        fit_jlens.fit_with_fallback(always_oom, "model", ["p1"], [2], 3, tmp_path / "shard1.ckpt.pt")
    assert attempts == [3, 1]


def test_check_resume_meta_refuses_a_different_fit(tmp_path):
    meta_path = tmp_path / "shard0.meta.json"
    lens = tmp_path / "shard0.lens.pt"
    expected = {"prompts_sha1": "abc", "n_prompts": 100, "start": 0}
    cross_lens.check_resume_meta(meta_path, expected, [lens])  # nothing on disk: fresh run
    lens.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="sidecar"):
        cross_lens.check_resume_meta(meta_path, expected, [lens])  # artifact without sidecar
    cross_lens.write_resume_meta(meta_path, expected)
    cross_lens.check_resume_meta(meta_path, expected, [lens])  # same fit: fine
    with pytest.raises(RuntimeError) as err:
        cross_lens.check_resume_meta(meta_path, {**expected, "n_prompts": 300, "start": 0}, [lens])
    assert "n_prompts: on disk 100, requested 300" in str(err.value) and "start" not in str(err.value).split("\n", 1)[1]


def test_cmd_fit_refuses_stale_shard_before_loading_model(tmp_path, monkeypatch):
    prompts_json = _write(tmp_path, "prompts.json", {"prompts": [f"prompt {i}" for i in range(8)]})
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    (out_dir / "shard0.lens.pt").write_bytes(b"stale")
    args = SimpleNamespace(
        model_id="m",
        prompts_json=prompts_json,
        n_prompts=4,
        start=0,
        shard=0,
        num_shards=2,
        n_layers=12,
        dim_batch=8,
        ckpt_dir=str(out_dir),
        out_dir=str(out_dir),
        device="cpu",
    )
    monkeypatch.setattr(fit_jlens, "import_jlens", lambda: SimpleNamespace(fit=None))

    def no_model(*a, **k):
        raise AssertionError("the model must not be loaded when the sidecar check fails or the shard is done")

    monkeypatch.setattr(fit_jlens, "load_lens_model", no_model)
    # sidecar from a 4-prompt fit; a re-run asking for 8 prompts is refused, naming the key
    cross_lens.write_resume_meta(
        out_dir / "shard0.meta.json", fit_jlens.shard_meta(args, fit_jlens.read_prompts(prompts_json, 4))
    )
    args.n_prompts = 8
    with pytest.raises(RuntimeError, match="n_prompts: on disk 4, requested 8"):
        fit_jlens.cmd_fit(args)
    # the matching request resumes (returns) without touching the model
    args.n_prompts = 4
    assert fit_jlens.cmd_fit(args) is None


def test_check_shard_metas_agree_names_the_key():
    base = {k: 1 for k in fit_jlens.SHARED_META_KEYS}
    base["num_shards"] = 2
    metas = [{**base, "shard": 0}, {**base, "shard": 1}]
    assert fit_jlens.check_shard_metas_agree(metas, num_shards=2) == {k: base[k] for k in fit_jlens.SHARED_META_KEYS}
    with pytest.raises(RuntimeError, match="prompts_sha1"):
        fit_jlens.check_shard_metas_agree([metas[0], {**metas[1], "prompts_sha1": "other"}], num_shards=2)
    with pytest.raises(RuntimeError, match="num_shards"):
        fit_jlens.check_shard_metas_agree(metas, num_shards=4)
    with pytest.raises(RuntimeError, match="shard:"):
        fit_jlens.check_shard_metas_agree([metas[0], metas[0]], num_shards=2)


class _FakeLens:
    def __init__(self, jacobians, n_prompts):
        self.jacobians, self.n_prompts = jacobians, n_prompts
        self.source_layers = sorted(jacobians)

    @classmethod
    def load(cls, path):
        return cls({2: torch.ones(2, 2)}, n_prompts=3)

    @classmethod
    def merge(cls, lenses):
        n = sum(lens.n_prompts for lens in lenses)
        return cls({2: sum(lens.jacobians[2] * lens.n_prompts for lens in lenses) / n}, n_prompts=n)

    def save(self, path):
        Path(path).write_bytes(b"lens")


def test_cmd_merge_requires_agreeing_sidecars_and_records_them(tmp_path, monkeypatch):
    monkeypatch.setattr(fit_jlens, "import_jlens", lambda: SimpleNamespace(JacobianLens=_FakeLens))
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    base = {
        "model_id": "m",
        "prompts_json": "p.json",
        "prompts_sha1": "abc",
        "n_prompts": 6,
        "start": 0,
        "num_shards": 2,
        "n_layers": 12,
    }
    for i in range(2):
        (shard_dir / f"shard{i}.lens.pt").write_bytes(b"x")
    args = SimpleNamespace(
        shard_dir=str(shard_dir), num_shards=2, out=str(tmp_path / "merged.pt"), fn=fit_jlens.cmd_merge
    )
    with pytest.raises(FileNotFoundError, match="sidecar"):
        fit_jlens.cmd_merge(args)
    cross_lens.write_resume_meta(shard_dir / "shard0.meta.json", {**base, "shard": 0})
    cross_lens.write_resume_meta(shard_dir / "shard1.meta.json", {**base, "shard": 1, "n_prompts": 9})
    with pytest.raises(RuntimeError, match="n_prompts"):
        fit_jlens.cmd_merge(args)
    cross_lens.write_resume_meta(shard_dir / "shard1.meta.json", {**base, "shard": 1})
    fit_jlens.cmd_merge(args)
    assert (tmp_path / "merged.pt").read_bytes() == b"lens"
    meta = json.loads((tmp_path / "merged.meta.json").read_text())
    assert meta["prompts_sha1"] == "abc" and meta["n_prompts"] == 6 and meta["shard_n_prompts"] == [3, 3]
    assert meta["merged_n_prompts"] == 6 and meta["layers"] == [2]
    assert meta["provenance"]["args"] == {
        "shard_dir": str(shard_dir),
        "num_shards": 2,
        "out": str(tmp_path / "merged.pt"),
    }


def test_prompt_list_sha1_is_content_addressed():
    assert cross_lens.prompt_list_sha1(["a", "b"]) == cross_lens.prompt_list_sha1(["a", "b"])
    assert cross_lens.prompt_list_sha1(["a", "b"]) != cross_lens.prompt_list_sha1(["a\nb"])


def test_ridge_holdout_guard_and_diagnostic_slice(tmp_path):
    argv = ["--prompts-json", "missing.json", "--layers-from", "missing.pt", "--ckpt-dir", str(tmp_path), "--out"]
    with pytest.raises(SystemExit, match="--holdout must be >= 1"):
        fit_ridge.main([*argv, str(tmp_path / "o.pt"), "--holdout", "0"])
    prompts = [f"p{i}" for i in range(100)]
    assert fit_ridge.diagnostic_prompts(prompts, 90) == prompts[95:100]  # the paper's slice, holdout 10
    assert fit_ridge.diagnostic_prompts(prompts, 98) == prompts[98:100]  # never reaches into the fit prompts
