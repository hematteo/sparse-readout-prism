"""CPU smoke tests for the CoarseWSD-20 sense analyses on a synthetic bundle.

* scripts/analyze/analyze_wsd_sense_groups.py      (tab:app-sense-alignment)
* scripts/analyze/analyze_wsd_classifier_framing.py (classifier-framing paragraph)

The bundle mimics ``representations.pt`` from ``run_wsd_feature_alignment.py``:
two words, two or three senses, 16 dictionary positions, with a strong and a
weaker planted discriminative position per sense. No model, no network, well under a second.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


sg = _load("analyze_wsd_sense_groups", "scripts/analyze/analyze_wsd_sense_groups.py")
cf = _load("analyze_wsd_classifier_framing", "scripts/analyze/analyze_wsd_classifier_framing.py")

WIDTH = 16
D_HIDDEN = 8
# word -> {sense: (planted contribution position, n_train, n_test)}
DESIGN = {
    "bank": {"0": (3, 30, 15), "1": (7, 10, 5)},
    "seal": {"0": (1, 20, 10), "1": (9, 12, 6), "2": (13, 8, 4)},
}


def make_bundle(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    metadata, contribution, projection, hidden = [], [], [], []
    for word, senses in DESIGN.items():
        beta = rng.uniform(0.5, 2.0, size=WIDTH).astype(np.float32)
        for sense, (pos, n_train, n_test) in senses.items():
            for split, n in (("train", n_train), ("test", n_test)):
                for i in range(n):
                    p = rng.normal(0.0, 0.3, size=WIDTH).astype(np.float32)
                    p[pos] += 3.0
                    p[pos + 1] += 1.5  # weaker second cue so group sizes 1 and 2 are both recoverable
                    h = rng.normal(0.0, 0.3, size=D_HIDDEN).astype(np.float32)
                    h[int(sense) % D_HIDDEN] += 3.0
                    metadata.append(
                        {
                            "item_id": f"coarse:{word}:{split}:{sense}:{i}",
                            "kind": "context",
                            "dataset": "coarsewsd20",
                            "split": split,
                            "target": word,
                            "sense": sense,
                            "token_id": 100 + len(word),
                            "row_relative_error": 0.2,
                        }
                    )
                    projection.append(p)
                    contribution.append(beta * p)
                    hidden.append(h)
    contribution = torch.tensor(np.stack(contribution))
    n = contribution.shape[0]
    return {
        "metadata": metadata,
        "hidden": torch.tensor(np.stack(hidden)),
        "projection": torch.tensor(np.stack(projection)),
        "contribution": contribution,
        "beta": torch.stack([torch.ones(WIDTH)] * n),
        "feature_ids": torch.stack([torch.arange(WIDTH, dtype=torch.int32)] * n),
        "exact_logit": contribution.sum(dim=1) + 5.0,
        "reconstructed_logit": contribution.sum(dim=1) + 5.0,
        "target_logprob": torch.zeros(n),
        "target_rank": torch.ones(n, dtype=torch.int32),
        "run": {"dataset": "coarsewsd20", "model_id": "synthetic", "k": WIDTH, "seed": seed},
    }


def _word_arrays(bundle: dict, word: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    idx = [i for i, r in enumerate(bundle["metadata"]) if r["target"] == word and r["split"] == split]
    x = bundle["contribution"].numpy()[idx]
    y = np.array([bundle["metadata"][i]["sense"] for i in idx])
    return x, y


def _standardize(train_c: np.ndarray) -> np.ndarray:
    """Train-statistic standardization, as the script applies before selection."""
    return (train_c - train_c.mean(axis=0)) / (train_c.std(axis=0) + 1e-6)


def test_select_anchors_recovers_planted_positions_and_is_disjoint():
    bundle = make_bundle()
    projection = bundle["projection"].numpy()
    for word, senses in DESIGN.items():
        idx = [i for i, r in enumerate(bundle["metadata"]) if r["target"] == word and r["split"] == "train"]
        train_y = np.array([bundle["metadata"][i]["sense"] for i in idx])
        names = sorted(senses)
        # On unscaled projections the strong cue is the single best feature under both selectors.
        for selector in ("mean_diff", "top1_freq"):
            anchors = sg.select_anchors(projection[idx], names, train_y, selector, 1)
            assert {s: anchors[s][0] for s in names} == {s: senses[s][0] for s in names}, (word, selector)
        # On train-standardized contributions (the script's default input) a group of two is the planted pair;
        # standardization equalizes the two cues' gaps, so only the set is determined.
        train_c, _ = _word_arrays(bundle, word, "train")
        groups = sg.select_anchors(_standardize(train_c), names, train_y, "mean_diff", 2)
        assert {s: set(groups[s]) for s in names} == {s: {senses[s][0], senses[s][0] + 1} for s in names}, word
        groups = sg.select_anchors(_standardize(train_c), names, train_y, "mean_diff", 4)
        flat = [p for s in names for p in groups[s]]
        assert all(len(groups[s]) == 4 for s in names)
        assert len(flat) == len(set(flat)), "groups must not share positions"


def test_group_prediction_and_balanced_accuracy():
    bundle = make_bundle()
    train_c, train_y = _word_arrays(bundle, "bank", "train")
    test_c, test_y = _word_arrays(bundle, "bank", "test")
    anchors = sg.select_anchors(train_c, ["0", "1"], train_y, "mean_diff", 2)
    acc, pred = sg.word_accuracy(test_c, anchors, ["0", "1"], test_y)
    assert acc == 1.0 and sg.balanced_accuracy(pred, test_y) == 1.0
    # Balanced accuracy is the mean per-sense recall: all-majority on a 15/5 split is 0.5.
    assert sg.balanced_accuracy(np.full_like(test_y, "0"), test_y) == 0.5


def test_sense_groups_end_to_end_sweep(tmp_path, monkeypatch):
    bundle_path = tmp_path / "representations.pt"
    torch.save(make_bundle(), bundle_path)
    out = tmp_path / "out" / "synthetic__srp.json"
    argv = [
        "analyze_wsd_sense_groups.py",
        "--bundle",
        str(bundle_path),
        "--out",
        str(out),
        "--group-sizes",
        "1,2",
        "--n-boot",
        "20",
        "--n-null",
        "10",
        "--seed",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert sg.main() == 0
    for g in (1, 2):
        path = out.with_name(f"synthetic__srp_g{g}.json")
        assert path.exists(), path
        summary = json.loads(path.read_text())
        assert summary["group_size"] == g and summary["basis"] == "srp" and summary["standardized"] is True
        assert summary["n_words"] == 2 and set(summary["per_word"]) == set(DESIGN)
        assert summary["word_mean_balanced"] > 0.95
        assert summary["word_mean_null_balanced"] < 0.8
        assert summary["word_mean_majority_balanced"] == (1 / 2 + 1 / 3) / 2
        assert summary["words_beating_majority_balanced"] == 2
        assert summary["words_beating_null_p95_balanced"] == 2
        assert summary["word_mean_full_account_balanced"] > 0.95
        assert summary["word_mean_hidden_balanced"] > 0.95
        assert summary["gate_fraction_overall"] == 1.0
        for word, senses in DESIGN.items():
            pw = summary["per_word"][word]
            assert pw["n_senses"] == len(senses)
            assert all(len(pw["anchors"][s]) == g for s in senses)
            assert pw["chance_balanced_accuracy"] == 1 / len(senses)
    # Single-size runs write --out verbatim and are reproducible under the seed.
    single = tmp_path / "single.json"
    monkeypatch.setattr(
        sys, "argv", argv[:5] + ["--out", str(single), "--group-size", "1", "--n-boot", "20", "--n-null", "10"]
    )
    assert sg.main() == 0
    a = json.loads(single.read_text())
    b = json.loads(out.with_name("synthetic__srp_g1.json").read_text())
    assert a["per_word"] == b["per_word"]


def test_classifier_framing_self_test_and_coarse_path(tmp_path):
    assert cf.self_test() == 0
    bundle = make_bundle()
    cf.validate_feature_alignment(bundle)
    gate, gate_summary = cf.score_gate(bundle)
    assert gate.all() and gate_summary["pass_fraction"] == 1.0
    metrics = cf.analyze_coarse(
        bundle, ks=[2, 4], primary_k=2, primary_encoding="weighted", n_boot=5, n_null_seeds=2, seed=0
    )
    assert set(metrics["primary"]["methods"]) == set(cf.METHODS)
    srp = metrics["primary"]["methods"]["srp"]
    assert srp["n_words"] == 2 and srp["accuracy"] > 0.95 and srp["balanced_accuracy"] > 0.95
    # Projections and contributions differ only by a per-word rescaling of the coordinates,
    # which top-|value| selection is not invariant to, so both are reported separately.
    assert "projection_only" in metrics["results"]["weighted"]["4"]
    assert set(metrics["primary"]["comparisons"]) == {f"srp_minus_{m}" for m in cf.METHODS[1:]}
    assert set(metrics["primary"]["null_seed_sensitivity"]) == {"shuffled_srp", "random_srp_features"}
