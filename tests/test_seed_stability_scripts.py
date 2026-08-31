"""CPU tests for the cross-seed stability scripts on synthetic inputs.

Three tiny dictionaries, a 40-row readout and a stub tokenizer; no network, no
GPU. Two layers of protection for the shared pipeline in
``sparse_readout_prism.research.seed_stability``:

* invariants -- identical dictionaries reproduce each other exactly (same-side
  Jaccard 1, best-single Jaccard 1, held-out core recall 1);
* pins -- the exact numbers the scripts produced on these fixtures before the
  0.2.1 consolidation (recorded from the pre-refactor code), so a refactor of
  the shared helpers cannot move an output silently. The seed-variation
  dictionaries of the paper are not distributed, so these pins are the
  behaviour-preservation proof for the three scripts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import REPO_ROOT, load_script

from sparse_readout_prism.research.qwen_readout import encode_topk
from sparse_readout_prism.research.row_geometry import resolve_single_token_bare_first
from sparse_readout_prism.research.seed_stability import (
    Dictionary,
    center_rows,
    load_contrast_pairs,
    load_dictionary,
    load_readout,
    resolve_centering,
    summary_stats,
    unit_rows,
)

V, D_MODEL, D_FEAT, K = 40, 8, 64, 4


class _StubTokenizer:
    all_special_ids = [0]  # row 0 is a "special" token for token_mask_from_tokenizer

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        w = text.strip()
        return [self.index[w]] if w in self.index else [0, 1]

    def decode(self, ids: list[int]) -> str:
        return " " + self.vocab[int(ids[0])]


def _inputs(tmp_path: Path, identical: bool):
    g = torch.Generator().manual_seed(0)
    vocab = [f"w{i:02d}" for i in range(V)]
    W = torch.randn(V, D_MODEL, generator=g)
    h_LN = torch.randn(3, 5, D_MODEL, generator=g)
    dicts = []
    for s in range(3):
        gs = torch.Generator().manual_seed(100 if identical else 100 + s)
        dec = torch.randn(D_FEAT, D_MODEL, generator=gs)
        dec = dec / dec.norm(dim=1, keepdim=True)
        ckpt = {
            "model_state_dict": {"decoder": dec, "encoder.weight": dec.clone(), "encoder.bias": torch.zeros(D_FEAT)},
            "factorizer": {"architecture": "topk", "d_features": D_FEAT, "k": K},
        }
        path = tmp_path / f"ckpt_s{s}.pt"
        torch.save(ckpt, path)
        dicts.append(load_dictionary(path))
    recs = [{"target_a": vocab[2 * i], "target_b": vocab[2 * i + 1]} for i in range(12)]
    recs.append({"target_a": "notaword", "target_b": vocab[0]})  # multi-token target: skipped
    recs.append({"target_a": vocab[0], "target_b": vocab[1]})  # duplicate pair: skipped
    bank = tmp_path / "bank.jsonl"
    bank.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    token_mask = torch.ones(V, dtype=torch.bool)
    token_mask[0] = False
    torch.save({"W_U_orig": W, "h_LN": h_LN, "token_mask": token_mask}, tmp_path / "w_u.pt")
    return _StubTokenizer(vocab), W, h_LN, dicts, bank


def _live_mean(W: torch.Tensor, dicts: list) -> torch.Tensor:
    return resolve_centering(W, dicts, "live", None)


# --------------------------------------------------------------------------- #
# Pins recorded from the pre-0.2.1 scripts on the fixtures above (rng seed 0,
# top_m=3, top_r=4; cross-seed n_sample=16 n_hidden=8; loo side_n=6
# n_clusters=8 kmeans_iters=2 kmeans_seed=0 n_boot=50 bootstrap_seed=1).
# Set-derived statistics are rationals and are compared to 1e-12; float32
# cosine / correlation statistics to 1e-6.
# --------------------------------------------------------------------------- #

SET_TOL, F32_TOL = 1e-12, 1e-6

CROSS_SEED_PINS = {
    True: dict(  # identical dictionaries
        same_side={"mean": 1.0, "median": 1.0, "p10": 1.0, "n": 72},
        cross_side={"mean": 0.17782421647553226, "median": 0.17207792207792205, "p10": 0.0, "n": 36},
        cross_contrast_null={"mean": 0.1740578507606371, "median": 0.125, "p10": 0.059027777777777776, "n": 36},
        null_p90=0.35714285714285715,
        frac_contrasts_above_null_p90=1.0,
        matched_projection_corr={"s0-s1": 1.0, "s0-s2": 1.0, "s1-s2": 1.0},
        nn_cos_used_median={"s0-s1": 1.0, "s0-s2": 1.0, "s1-s2": 1.0},
    ),
    False: dict(  # independently seeded dictionaries
        same_side={"mean": 0.39943998902332234, "median": 0.39230769230769236, "p10": 0.25, "n": 72},
        cross_side={"mean": 0.15661895520964147, "median": 0.13942307692307693, "p10": 0.0, "n": 36},
        cross_contrast_null={"mean": 0.13614265182892632, "median": 0.125, "p10": 0.05277777777777778, "n": 36},
        null_p90=0.2857142857142857,
        frac_contrasts_above_null_p90=0.8055555555555556,
        matched_projection_corr={"s0-s1": 0.8081061840057373, "s0-s2": 0.7951256632804871, "s1-s2": 0.6831156015396118},
        nn_cos_used_median={"s0-s1": 0.7546381950378418, "s0-s2": 0.7305821180343628, "s1-s2": 0.7207158505916595},
    ),
}

LOO_PINS = {
    True: dict(
        n_loo_cells=72,
        core_median_size=9.0,
        srp_heldout_seed_recall=1.0,
        srp_ci95=[1.0, 1.0],
        knn_recall_same_cores=0.4006,
        knn_ci95=[0.3556, 0.4554],
        cluster_recall_same_cores=0.3197,
        cluster_ci95=[0.2873, 0.3823],
        paper_dict_recall_same_cores=1.0,
        ratio_srp_over_knn=2.496,
        ratio_ci95=[2.1961, 2.8125],
    ),
    False: dict(
        n_loo_cells=72,
        core_median_size=5.0,
        srp_heldout_seed_recall=0.697,
        srp_ci95=[0.6543, 0.7468],
        knn_recall_same_cores=0.4882,
        knn_ci95=[0.4338, 0.5503],
        cluster_recall_same_cores=0.4344,
        cluster_ci95=[0.3674, 0.506],
        paper_dict_recall_same_cores=0.9053,
        ratio_srp_over_knn=1.428,
        ratio_ci95=[1.2778, 1.6402],
    ),
}

# feature_group_matching: only the first seed pair's null-independent fields
# were reproducible before 0.2.1 (the null draws depended on PYTHONHASHSEED and
# their RNG consumption leaked into the query sampling of later pairs).
FGM_S0S1_PINS = {
    True: dict(
        best_jaccard={"mean": 1.0, "median": 1.0, "p90": 1.0, "p99": 1.0, "n": 10},
        frac_jaccard_ge_050=1.0,
        frac_jaccard_ge_025=1.0,
        recall_at_1={"mean": 1.0, "median": 1.0, "p90": 1.0, "p99": 1.0, "n": 10},
        recall_at_3={"mean": 1.0, "median": 1.0, "p90": 1.0, "p99": 1.0, "n": 10},
        matched_decoder_cosine_mean=0.9688253819942474,
        features=[47, 26, 0, 52, 39, 1, 36, 16, 45, 49],
        best_jaccards=[1.0] * 10,
        best_features=[61, 26, 0, 52, 39, 1, 36, 16, 45, 49],
        cosines=[0.6883, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    False: dict(
        best_jaccard={
            "mean": 0.6133333333333332,
            "median": 0.6,
            "p90": 0.6399999999999998,
            "p99": 0.9640000000000001,
            "n": 10,
        },
        frac_jaccard_ge_050=0.9,
        frac_jaccard_ge_025=1.0,
        recall_at_1={"mean": 0.75, "median": 0.75, "p90": 0.7749999999999999, "p99": 0.9775, "n": 10},
        recall_at_3={"mean": 1.0, "median": 1.0, "p90": 1.0, "p99": 1.0, "n": 10},
        matched_decoder_cosine_mean=0.5398867003619671,
        features=[47, 26, 0, 52, 39, 1, 36, 16, 45, 49],
        best_jaccards=[0.6, 0.6, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3333, 0.6],
        best_features=[58, 5, 61, 24, 54, 59, 46, 44, 51, 52],
        cosines=[0.6178, 0.6934, 0.8721, 0.5615, 0.5403, 0.4987, 0.665, 0.4312, 0.0773, 0.4416],
    ),
}

# feature_group_matching null-dependent values, recorded AFTER the 0.2.1 sorted-pool
# fix: (null99, frac_above_null_p99, direction_check.n_below_null_p99, sampled query
# features). Reproducible in every process now; before the fix they moved with the
# hash seed (the first pair's features excepted).
FGM_NULL_PINS = {
    True: {
        "s0->s1": (0.6, 1.0, 0, [47, 26, 0, 52, 39, 1, 36, 16, 45, 49]),
        "s0->s2": (0.6, 1.0, 0, [59, 31, 34, 45, 16, 27, 53, 40, 10, 25]),
        "s1->s2": (0.6, 1.0, 0, [35, 59, 61, 24, 26, 45, 44, 33, 0, 38]),
    },
    False: {
        "s0->s1": (0.6, 0.1, 9, [47, 26, 0, 52, 39, 1, 36, 16, 45, 49]),
        "s0->s2": (0.6, 0.0, 10, [59, 31, 34, 45, 16, 27, 53, 40, 10, 25]),
        "s1->s2": (0.6, 0.1, 9, [35, 59, 61, 25, 28, 44, 56, 34, 0, 39]),
    },
}

# loo_core_recovery with --no-knn-exclude-self (the target row admitted to its own kNN
# side set): kNN recall and the SRP/kNN ratio, recorded on the consolidated code.
LOO_INCLUDE_SELF_PINS = {True: (0.4377, 2.285), False: (0.6453, 1.08)}

FIRST_PAIRS = [("w18", "w19", 18, 19), ("w04", "w05", 4, 5), ("w14", "w15", 14, 15)]


def _assert_stats(got: dict, want: dict, tol: float) -> None:
    assert set(got) == set(want)
    assert got["n"] == want["n"]
    for key in want:
        if key != "n":
            assert got[key] == pytest.approx(want[key], abs=tol), key


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def test_dictionary_and_bank_helpers(tmp_path: Path) -> None:
    tok, W, h_LN, dicts, bank = _inputs(tmp_path, identical=False)
    d = dicts[0]
    assert isinstance(d, Dictionary)
    assert d.decoder.shape == (D_FEAT, D_MODEL) and d.encoder_w.shape == (D_FEAT, D_MODEL)
    assert d.encoder_b.shape == (D_FEAT,) and d.k == K and d.row_mean is None
    assert d[0] is d.decoder and d[3] == K  # positional layout of the former 4-tuple
    assert resolve_single_token_bare_first(tok, "w03") == 3 and resolve_single_token_bare_first(tok, "notaword") is None
    pairs = load_contrast_pairs(bank, tok, np.random.default_rng(0), max_contrasts=10)
    assert len(pairs) == 10 and len({(a, b) for a, b, _, _ in pairs}) == 10
    assert pairs == load_contrast_pairs(bank, tok, np.random.default_rng(0), max_contrasts=10)
    assert load_contrast_pairs(bank, tok, np.random.default_rng(0), 150)[:3] == FIRST_PAIRS
    codes = encode_topk(W[:5], d.encoder_w, d.encoder_b, K)
    assert codes.shape == (5, D_FEAT) and bool(((codes > 0).sum(dim=1) <= K).all())
    W2, h2, mask = load_readout(tmp_path / "w_u.pt")
    assert torch.equal(W2, W) and torch.equal(h2, h_LN) and mask is not None and int(mask.sum()) == V - 1


def test_load_dictionary_surfaces_training_row_mean(tmp_path: Path) -> None:
    _, _, _, dicts, _ = _inputs(tmp_path, identical=True)
    ckpt = torch.load(tmp_path / "ckpt_s0.pt", weights_only=True)
    ckpt["row_mean"] = torch.full((D_MODEL,), 0.25)
    torch.save(ckpt, tmp_path / "ckpt_rm.pt")
    d = load_dictionary(tmp_path / "ckpt_rm.pt")
    assert torch.equal(d.row_mean, torch.full((D_MODEL,), 0.25)) and d.k == K
    assert torch.equal(d.encoder_w, dicts[0].encoder_w)


def test_center_rows_matches_full_vocabulary_centering(tmp_path: Path) -> None:
    # pins the pre-0.2.1 center_rows(W) arithmetic exactly (W - W.mean(0), norms floored at 1e-8)
    _, W, _, dicts, _ = _inputs(tmp_path, identical=True)
    row_mean = _live_mean(W, dicts)
    assert torch.equal(row_mean, W.mean(0))
    W_c, rn, W_n = center_rows(W, row_mean)
    assert torch.equal(W_c, W - W.mean(0))
    assert torch.equal(rn, (W - W.mean(0)).norm(dim=1).clamp_min(1e-8))
    assert torch.equal(W_n, W_c / rn[:, None])


def test_resolve_centering_modes(tmp_path: Path) -> None:
    tok, W, _, dicts, _ = _inputs(tmp_path, identical=True)
    mask = torch.ones(V, dtype=torch.bool)
    mask[:5] = False
    assert torch.equal(resolve_centering(W, dicts, "live", mask), W.mean(0))  # live ignores the mask
    assert torch.equal(resolve_centering(W, dicts, "trained", mask), W[mask].mean(0))  # nothing stored
    assert torch.equal(resolve_centering(W, dicts, "trained", None, tok=tok), W[1:].mean(0))  # mask from tok
    assert torch.equal(resolve_centering(W, dicts, "trained", None), W.mean(0))  # no mask, no tok, no store
    stored = [d._replace(row_mean=torch.full((D_MODEL,), 0.5)) for d in dicts]
    assert torch.equal(resolve_centering(W, stored, "trained", mask), torch.full((D_MODEL,), 0.5))
    assert torch.equal(resolve_centering(W, stored, "live", mask), W.mean(0))
    mixed = [stored[0], stored[1]._replace(row_mean=torch.zeros(D_MODEL)), stored[2]]
    with pytest.raises(ValueError):
        resolve_centering(W, mixed, "trained", mask)


def test_topk_keeps_exactly_k_at_ties() -> None:
    # documented 0.2.1 change: the former >= rule kept both tied 2.0 codes at k=1
    codes = encode_topk(torch.tensor([[2.0, 2.0, 1.0]]), torch.eye(3), torch.zeros(3), 1)
    assert int((codes != 0).sum()) == 1 and float(codes.max()) == 2.0


def test_shared_stats_and_unit_helpers() -> None:
    assert list(summary_stats([1.0, 2.0, 4.0], (10,))) == ["mean", "median", "p10", "n"]
    assert list(summary_stats([1.0, 2.0, 4.0], (90, 99))) == ["mean", "median", "p90", "p99", "n"]
    s = summary_stats([1.0, 2.0, 4.0], (90,))
    assert s["mean"] == pytest.approx(7 / 3) and s["median"] == 2.0 and s["n"] == 3
    u = unit_rows(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    assert torch.allclose(u[0], torch.tensor([0.6, 0.8])) and torch.equal(u[1], torch.zeros(2))


# --------------------------------------------------------------------------- #
# cross_seed_stability.py
# --------------------------------------------------------------------------- #


def _run_cross_seed(tmp_path: Path, identical: bool) -> dict:
    tok, W, h_LN, dicts, bank = _inputs(tmp_path, identical=identical)
    mod = load_script("scripts/eval/cross_seed_stability.py")
    rng = np.random.default_rng(0)
    pairs = load_contrast_pairs(bank, tok, rng, 150)
    return mod.run(
        dicts,
        [0, 1, 2],
        W,
        h_LN,
        tok,
        pairs,
        rng,
        row_mean=_live_mean(W, dicts),
        top_m=3,
        top_r=4,
        n_sample=16,
        n_hidden=8,
        width_tag="test",
    )


@pytest.mark.parametrize("identical", [True, False])
def test_cross_seed_pins(tmp_path: Path, identical: bool) -> None:
    out = _run_cross_seed(tmp_path, identical)
    pin = CROSS_SEED_PINS[identical]
    assert out["n_contrasts"] == 12 and out["k"] == K and out["width"] == "test"
    for key in ("same_side", "cross_side", "cross_contrast_null"):
        _assert_stats(out[key], pin[key], SET_TOL)
    assert out["null_p90"] == pytest.approx(pin["null_p90"], abs=SET_TOL)
    assert out["frac_contrasts_above_null_p90"] == pytest.approx(pin["frac_contrasts_above_null_p90"], abs=SET_TOL)
    assert set(out["basis_nn_cosine"]) == set(out["matched_projection_corr"]) == {"s0-s1", "s0-s2", "s1-s2"}
    for key, r in pin["matched_projection_corr"].items():
        assert out["matched_projection_corr"][key] == pytest.approx(r, abs=F32_TOL)
    for key, med in pin["nn_cos_used_median"].items():
        assert out["basis_nn_cosine"][key]["nn_cos_used"]["median"] == pytest.approx(med, abs=F32_TOL)


def test_cross_seed_identical_dictionaries_reproduce(tmp_path: Path) -> None:
    out = _run_cross_seed(tmp_path, identical=True)
    assert out["same_side"]["mean"] == 1.0
    assert out["frac_contrasts_above_null_p90"] == 1.0
    assert out["same_side"]["n"] == 3 * 12 * 2  # seed pairs x contrasts x sides
    for v in out["basis_nn_cosine"].values():
        assert abs(v["nn_cos_used"]["median"] - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# feature_group_matching.py
# --------------------------------------------------------------------------- #


def _run_matching(tmp_path: Path, identical: bool) -> dict:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=identical)
    mod = load_script("scripts/eval/feature_group_matching.py")
    rng = np.random.default_rng(0)
    pairs = load_contrast_pairs(bank, tok, rng, 150)
    return mod.run(
        dicts,
        [0, 1, 2],
        W,
        tok,
        pairs,
        rng,
        torch.device("cpu"),
        row_mean=_live_mean(W, dicts),
        top_m=3,
        top_r=4,
        n_query=10,
        n_null=20,
        greedy_pool=20,
        greedy_k=3,
        min_group_size=4,
        width_tag="test",
        direction_check=True,
        direction_null=20,
        direction_seed=0,
    )


@pytest.mark.parametrize("identical", [True, False])
def test_feature_group_matching_first_pair_pins(tmp_path: Path, identical: bool) -> None:
    out = _run_matching(tmp_path, identical)
    assert set(out["pairs"]) == {"s0->s1", "s0->s2", "s1->s2"}
    res = out["pairs"]["s0->s1"]
    pin = FGM_S0S1_PINS[identical]
    assert res["n_query"] == 10 and len(res["per_query"]) == 10
    for key in ("best_jaccard", "recall_at_1", "recall_at_3"):
        _assert_stats(res[key], pin[key], SET_TOL)
    assert res["frac_jaccard_ge_050"] == pytest.approx(pin["frac_jaccard_ge_050"], abs=SET_TOL)
    assert res["frac_jaccard_ge_025"] == pytest.approx(pin["frac_jaccard_ge_025"], abs=SET_TOL)
    assert res["matched_decoder_cosine"]["mean"] == pytest.approx(pin["matched_decoder_cosine_mean"], abs=F32_TOL)
    assert [q["feature"] for q in res["per_query"]] == pin["features"]
    assert [q["best_jaccard"] for q in res["per_query"]] == pin["best_jaccards"]
    assert [q["best_feature"] for q in res["per_query"]] == pin["best_features"]
    assert [q["cosine"] for q in res["per_query"]] == pin["cosines"]


@pytest.mark.parametrize("identical", [True, False])
def test_feature_group_matching_null_pins(tmp_path: Path, identical: bool) -> None:
    out = _run_matching(tmp_path, identical)
    for key, (null99, frac_above, n_below, features) in FGM_NULL_PINS[identical].items():
        res = out["pairs"][key]
        assert res["null99"] == pytest.approx(null99, abs=SET_TOL), key
        assert res["frac_above_null_p99"] == pytest.approx(frac_above, abs=SET_TOL), key
        assert res["direction_check"]["n_below_null_p99"] == n_below, key
        assert [q["feature"] for q in res["per_query"]] == features, key


def test_feature_group_matching_tail_agrees_with_frac_above_null(tmp_path: Path) -> None:
    # 0.2.1 fix: the direction-check tail and frac_above_null_p99 are complementary counts
    out = _run_matching(tmp_path, identical=False)
    for res in out["pairs"].values():
        n_above = round(res["frac_above_null_p99"] * res["n_query"])
        assert res["direction_check"]["n_below_null_p99"] == res["n_query"] - n_above


def test_null_token_pool_is_sorted_and_hash_seed_independent() -> None:
    # 0.2.1 fix: the null pool is enumerated in sorted order, so the pseudo-group draws
    # (and the RNG state they consume) no longer depend on PYTHONHASHSEED
    script = REPO_ROOT / "scripts" / "eval" / "feature_group_matching.py"
    child = textwrap.dedent(
        f"""
        import importlib.util, json, sys
        import numpy as np
        spec = importlib.util.spec_from_file_location("fgm_child", {str(script)!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fgm_child"] = mod
        spec.loader.exec_module(mod)
        groups = [frozenset(g) for g in (("a", "b", "c"), ("b", "c", "d"), ("e", "f", "a"), ("x", "y", "z", "w"))]
        toks, p = mod.null_token_pool(groups)
        draw = np.random.default_rng(0).choice(toks, size=3, replace=False, p=p).tolist()
        print(json.dumps({{"toks": toks, "p": p.tolist(), "draw": draw}}))
        """
    )
    outs = []
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT / "src")}
        proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, env=env, check=True)
        outs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
    assert outs[0]["toks"] == ["a", "b", "c", "d", "e", "f", "w", "x", "y", "z"]
    assert outs[0]["p"] == pytest.approx(
        [2 / 13, 2 / 13, 2 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13]
    )


def test_feature_group_matching_identical_dictionaries(tmp_path: Path) -> None:
    out = _run_matching(tmp_path, identical=True)
    for res in out["pairs"].values():
        assert res["best_jaccard"]["median"] == 1.0
        assert res["frac_jaccard_ge_050"] == 1.0
        assert res["recall_at_3"]["median"] == 1.0
        # best match may be a tied duplicate group (source tie-break), so test the group, not the index
        assert all(q["best_match_tokens"] == q["query_tokens"] for q in res["per_query"])
        assert all(-1.0 <= q["cosine"] <= 1.0 for q in res["per_query"])
        assert res["direction_check"]["n_below_null_p99"] == 0


def test_feature_group_matching_direction_check_annotates_tail(tmp_path: Path) -> None:
    out = _run_matching(tmp_path, identical=False)
    for res in out["pairs"].values():
        check = res["direction_check"]
        fails = [q for q in res["per_query"] if q["best_jaccard"] <= res["null99"]]
        assert check["n_below_null_p99"] == len(fails)
        assert all("max_decoder_cosine" in q for q in fails)
        assert all("max_decoder_cosine" not in q for q in res["per_query"] if q["best_jaccard"] > res["null99"])
        assert 0 <= check["n_decoder_cosine_ge_0p5"] <= len(fails)
        assert 0.0 <= res["null_best_jaccard"]["p99"] <= 1.0


# --------------------------------------------------------------------------- #
# loo_core_recovery.py
# --------------------------------------------------------------------------- #


def _run_loo(tmp_path: Path, identical: bool, **overrides) -> dict:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=identical)
    mod = load_script("scripts/eval/loo_core_recovery.py")
    rng = np.random.default_rng(0)
    pairs = load_contrast_pairs(bank, tok, rng, 150)
    return mod.run(
        dicts,
        W,
        tok,
        pairs,
        torch.device("cpu"),
        row_mean=_live_mean(W, dicts),
        top_m=3,
        top_r=4,
        side_n=6,
        n_clusters=8,
        kmeans_iters=2,
        kmeans_seed=0,
        n_boot=50,
        bootstrap_seed=1,
        reference_dict=dicts[0],
        width_tag="test",
        **overrides,
    )


@pytest.mark.parametrize("identical", [True, False])
def test_loo_core_recovery_pins(tmp_path: Path, identical: bool) -> None:
    out = _run_loo(tmp_path, identical)
    pin = LOO_PINS[identical]
    assert out["width"] == "test"
    for key, want in pin.items():
        assert out[key] == want, key


@pytest.mark.parametrize("identical", [True, False])
def test_loo_no_knn_exclude_self(tmp_path: Path, identical: bool) -> None:
    out = _run_loo(tmp_path, identical, knn_exclude_self=False)
    knn, ratio = LOO_INCLUDE_SELF_PINS[identical]
    assert out["knn_recall_same_cores"] == knn and out["ratio_srp_over_knn"] == ratio
    # only the kNN side changes; the SRP and cluster sides never excluded the target
    for key in ("srp_heldout_seed_recall", "cluster_recall_same_cores", "core_median_size", "n_loo_cells"):
        assert out[key] == LOO_PINS[identical][key], key


def test_knn_side_sets_self_exclusion(tmp_path: Path) -> None:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=True)
    mod = load_script("scripts/eval/loo_core_recovery.py")
    pairs = load_contrast_pairs(bank, tok, np.random.default_rng(0), 150)
    _, _, W_n = center_rows(W, _live_mean(W, dicts))
    excl = mod.knn_side_sets(W_n, pairs, 6, tok, {}, exclude_self=True)
    incl = mod.knn_side_sets(W_n, pairs, 6, tok, {}, exclude_self=False)
    for (a, b, _ia, _ib), e, i in zip(pairs, excl, incl):
        assert a not in e[0] and b not in e[1]
        assert a in i[0] and b in i[1]  # the target's own row (cosine 1) takes a slot
        assert len(e[0]) == len(i[0]) == 6


def test_loo_core_recovery_identical_dictionaries(tmp_path: Path) -> None:
    out = _run_loo(tmp_path, identical=True)
    assert out["n_loo_cells"] == 12 * 2 * 3  # contrasts x sides x held-out seeds
    assert out["srp_heldout_seed_recall"] == 1.0
    assert out["srp_ci95"] == [1.0, 1.0]
    assert out["paper_dict_recall_same_cores"] == 1.0
    # the kNN control is neither trivially perfect nor empty on this case, so the
    # ratio is informative: it must be the SRP / kNN quotient of the same output
    knn = out["knn_recall_same_cores"]
    assert 0.0 < knn < 1.0 and 0.0 < out["cluster_recall_same_cores"] < 1.0
    assert out["ratio_srp_over_knn"] == round(out["srp_heldout_seed_recall"] / knn, 3)
