"""CPU smoke tests for the cross-seed stability scripts on synthetic inputs.

Three tiny dictionaries, a 40-row readout and a stub tokenizer; no network, no
GPU. Identical dictionaries must reproduce each other exactly (same-side
Jaccard 1, best-single Jaccard 1, held-out core recall 1), which pins the
shared pipeline in ``sparse_readout_prism.research.seed_stability``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.research.seed_stability import (
    load_contrast_pairs,
    load_dictionary,
    single_token_id,
    topk_codes,
)

ROOT = Path(__file__).resolve().parents[1]
V, D_MODEL, D_FEAT, K = 40, 8, 64, 4


class _StubTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(vocab)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        w = text.strip()
        return [self.index[w]] if w in self.index else [0, 1]

    def decode(self, ids: list[int]) -> str:
        return " " + self.vocab[int(ids[0])]


def _load_script(name: str):
    path = ROOT / "scripts" / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_seed_stability_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    return _StubTokenizer(vocab), W, h_LN, dicts, bank


def test_dictionary_and_bank_helpers(tmp_path: Path) -> None:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=False)
    dec, enc_w, enc_b, k = dicts[0]
    assert dec.shape == (D_FEAT, D_MODEL) and enc_w.shape == (D_FEAT, D_MODEL) and enc_b.shape == (D_FEAT,)
    assert k == K
    assert single_token_id(tok, "w03") == 3 and single_token_id(tok, "notaword") is None
    pairs = load_contrast_pairs(bank, tok, np.random.default_rng(0), max_contrasts=10)
    assert len(pairs) == 10 and len({(a, b) for a, b, _, _ in pairs}) == 10
    assert pairs == load_contrast_pairs(bank, tok, np.random.default_rng(0), max_contrasts=10)
    codes = topk_codes(W[:5], enc_w, enc_b, K)
    assert codes.shape == (5, D_FEAT) and bool(((codes > 0).sum(dim=1) <= K).all())


def test_cross_seed_identical_dictionaries_reproduce(tmp_path: Path) -> None:
    tok, W, h_LN, dicts, bank = _inputs(tmp_path, identical=True)
    mod = _load_script("cross_seed_stability")
    rng = np.random.default_rng(0)
    pairs = load_contrast_pairs(bank, tok, rng, 150)
    out = mod.run(
        dicts, [0, 1, 2], W, h_LN, tok, pairs, rng, top_m=3, top_r=4, n_sample=16, n_hidden=8, width_tag="test"
    )
    assert out["n_contrasts"] == 12 and out["k"] == K
    assert out["same_side"]["mean"] == 1.0
    assert out["frac_contrasts_above_null_p90"] == 1.0
    for v in out["basis_nn_cosine"].values():
        assert abs(v["nn_cos_used"]["median"] - 1.0) < 1e-5
    for r in out["matched_projection_corr"].values():
        assert abs(r - 1.0) < 1e-5


def test_cross_seed_different_seeds_in_range(tmp_path: Path) -> None:
    tok, W, h_LN, dicts, bank = _inputs(tmp_path, identical=False)
    mod = _load_script("cross_seed_stability")
    rng = np.random.default_rng(0)
    pairs = load_contrast_pairs(bank, tok, rng, 150)
    out = mod.run(
        dicts, [0, 1, 2], W, h_LN, tok, pairs, rng, top_m=3, top_r=4, n_sample=16, n_hidden=8, width_tag="test"
    )
    for key in ("same_side", "cross_side", "cross_contrast_null"):
        assert 0.0 <= out[key]["mean"] <= 1.0
    assert out["same_side"]["n"] == 3 * 12 * 2  # seed pairs x contrasts x sides
    assert set(out["basis_nn_cosine"]) == {"s0-s1", "s0-s2", "s1-s2"}
    assert set(out["matched_projection_corr"]) == {"s0-s1", "s0-s2", "s1-s2"}


def _run_matching(mod, tok, W, dicts, bank):
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


def test_feature_group_matching_identical_dictionaries(tmp_path: Path) -> None:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=True)
    out = _run_matching(_load_script("feature_group_matching"), tok, W, dicts, bank)
    assert set(out["pairs"]) == {"s0->s1", "s0->s2", "s1->s2"}
    for res in out["pairs"].values():
        assert res["n_query"] == 10 and len(res["per_query"]) == 10
        assert res["best_jaccard"]["median"] == 1.0
        assert res["frac_jaccard_ge_050"] == 1.0
        assert res["recall_at_3"]["median"] == 1.0
        # best match may be a tied duplicate group (source tie-break), so test the group, not the index
        assert all(q["best_match_tokens"] == q["query_tokens"] for q in res["per_query"])
        assert all(-1.0 <= q["cosine"] <= 1.0 for q in res["per_query"])
        assert res["direction_check"]["n_below_null_p99"] == 0


def test_feature_group_matching_direction_check_annotates_tail(tmp_path: Path) -> None:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=False)
    out = _run_matching(_load_script("feature_group_matching"), tok, W, dicts, bank)
    for res in out["pairs"].values():
        check = res["direction_check"]
        fails = [q for q in res["per_query"] if q["best_jaccard"] <= res["null99"]]
        assert check["n_below_null_p99"] == len(fails)
        assert all("max_decoder_cosine" in q for q in fails)
        assert all("max_decoder_cosine" not in q for q in res["per_query"] if q["best_jaccard"] > res["null99"])
        assert 0 <= check["n_decoder_cosine_ge_0p5"] <= len(fails)
        assert 0.0 <= res["null_best_jaccard"]["p99"] <= 1.0


def test_loo_core_recovery_identical_dictionaries(tmp_path: Path) -> None:
    tok, W, _, dicts, bank = _inputs(tmp_path, identical=True)
    mod = _load_script("loo_core_recovery")
    rng = np.random.default_rng(0)
    pairs = load_contrast_pairs(bank, tok, rng, 150)
    out = mod.run(
        dicts,
        W,
        tok,
        pairs,
        torch.device("cpu"),
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
    )
    assert out["n_loo_cells"] == 12 * 2 * 3  # contrasts x sides x held-out seeds
    assert out["srp_heldout_seed_recall"] == 1.0
    assert out["srp_ci95"] == [1.0, 1.0]
    assert out["paper_dict_recall_same_cores"] == 1.0
    assert 0.0 <= out["knn_recall_same_cores"] <= 1.0
    assert 0.0 <= out["cluster_recall_same_cores"] <= 1.0
    assert out["ratio_srp_over_knn"] >= 1.0
