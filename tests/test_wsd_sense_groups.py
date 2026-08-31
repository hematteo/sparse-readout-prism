"""CoarseWSD-20 sense analyses and the bundle writer, pinned on synthetic fixtures.

* scripts/run/run_wsd_feature_alignment.py          (bundle writer, full-vector references)
* scripts/analyze/analyze_wsd_sense_groups.py       (tab:app-sense-alignment)
* scripts/analyze/analyze_wsd_classifier_framing.py (classifier-framing paragraph)
* sparse_readout_prism.research.wsd                 (their shared helpers)

``PINS`` holds outputs of the pre-0.2.1 scripts (per-script helper copies,
right truncation, glob-picked HF snapshot) on the fixtures below: hashes of
canonical JSON or array bytes plus a few readable values. The tests assert the
refactored code reproduces them exactly. The bundle mimics
``representations.pt``: two words, two or three senses, 16 dictionary
positions, a strong and a weaker planted discriminative position per sense.
No model, no network, well under a second per test.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import load_script
from sparse_readout_prism.data import centering_mean
from sparse_readout_prism.factorizers import build_factorizer
from sparse_readout_prism.research import wsd
from sparse_readout_prism.research.qwen_readout import load_sae
from sparse_readout_prism.utils import spearman

rw = load_script("scripts/run/run_wsd_feature_alignment.py")
sg = load_script("scripts/analyze/analyze_wsd_sense_groups.py")
cf = load_script("scripts/analyze/analyze_wsd_classifier_framing.py")

PINS = {
    "auc": {"big_0": 0.5645077524610911, "big_3": 0.5645737167712954, "one_class": None, "small_0": 0.4735317279695978},
    "cb": {
        "ci": [-0.14836898540991997, 0.3938881306083598],
        "first3": [0.10616698864212924, 0.04648800212459197, 0.05718697644108653],
        "n": 50,
    },
    "cf": {
        "cmp_projection": {
            "accuracy_difference": 0.0,
            "accuracy_difference_ci": [0.0, 0.0],
            "accuracy_positive_words": 0,
            "balanced_accuracy_difference": 0.0,
            "balanced_accuracy_difference_ci": [0.0, 0.0],
            "balanced_accuracy_positive_words": 0,
            "macro_f1_difference": 0.0,
            "macro_f1_difference_ci": [0.0, 0.0],
            "macro_f1_positive_words": 0,
            "n_words": 2,
        },
        "gate": {
            "median_rho_plus_0p5": 0.0,
            "n_items": 120,
            "n_pass": 120,
            "pass_fraction": 1.0,
            "sign_match_fraction": 1.0,
        },
        "hash": "7e49e4804b5837ce",
        "null_shuffled": {
            "macro_f1": {
                "fraction_null_at_least_observed": 1.0,
                "max": 1.0,
                "mean": 1.0,
                "min": 1.0,
                "n_seeds": 2,
                "observed_srp": 1.0,
                "q05_q95": [1.0, 1.0],
                "std": 0.0,
            }
        },
        "primary_methods": {
            "projection_only": {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "macro_f1_ci": [1.0, 1.0],
                "n_test": 40,
                "n_train": 80,
                "n_words": 2,
            },
            "random_srp_features": {
                "accuracy": 0.5,
                "balanced_accuracy": 0.5805555555555555,
                "macro_f1": 0.48367027970608534,
                "macro_f1_ci": [0.32539682539682535, 0.48367027970608534],
                "n_test": 40,
                "n_train": 80,
                "n_words": 2,
            },
            "shuffled_srp": {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "macro_f1_ci": [1.0, 1.0],
                "n_test": 40,
                "n_train": 80,
                "n_words": 2,
            },
            "srp": {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "macro_f1_ci": [1.0, 1.0],
                "n_test": 40,
                "n_train": 80,
                "n_words": 2,
            },
        },
        "results_signed_4_srp_agg": {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "n_test": 40,
            "n_train": 80,
            "n_words": 2,
        },
    },
    "cf_cli": {
        "analysis": {
            "encodings": ["signed", "weighted"],
            "ks": [2, 4],
            "methods": ["srp", "projection_only", "shuffled_srp", "random_srp_features"],
            "n_boot": 7,
            "n_null_seeds": 3,
            "primary_encoding": "signed",
            "primary_k": 4,
            "seed": 5,
        },
        "hash": "7da70bb3210641aa",
        "primary_srp": {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "macro_f1_ci": [1.0, 1.0],
            "n_test": 40,
            "n_train": 80,
            "n_words": 2,
        },
    },
    "cf_random_enc_hash": "d8dfe216739112f6",
    "cf_shuffled_hash": {"0": "aaff27a124e088d4", "3": "c164033316962177"},
    "ptc": {
        "audit": {
            "n_targets_requested": 3,
            "n_targets_single_token": 2,
            "single_token_fraction": 0.6666666666666666,
            "skipped_targets": {"multi": "multi_token"},
            "targets": {
                "bank": {
                    "continuation": " bank",
                    "n_positive_codes": 6,
                    "row_cosine": 0.3883303105831146,
                    "row_relative_error": 3.34517765045166,
                    "token_id": 3,
                },
                "seal": {
                    "continuation": " seal",
                    "n_positive_codes": 6,
                    "row_cosine": -0.21251048147678375,
                    "row_relative_error": 2.8938305377960205,
                    "token_id": 4,
                },
            },
        },
        "bank_beta": "3770c9b09cff729c",
        "bank_fids": [21, 2, 24, 17, 8, 5],
        "k": 6,
        "seal_beta": "a378b719424cc9bb",
        "seal_fids": [8, 25, 31, 26, 28, 21],
    },
    "rw_method_hashes": {
        "hidden": "710352578f8a4b30",
        "shuffled_srp": "ab38110ffb3cb5d8",
        "srp": "f203f93dc79b5f48",
        "support_projection": "9e00ed52bd0f0d38",
        "token_only": "4f22f486a7f821af",
    },
    "rw_metrics": {
        "bank_hidden_per_word": {
            "accuracy": 1.0,
            "ari": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "n_senses": 2,
            "n_test": 20,
            "n_train": 40,
            "nmi": 1.0,
            "pairwise_auc": 1.0,
            "row_relative_error": 0.2,
        },
        "cmp_hidden": {
            "accuracy_difference": 0.0,
            "accuracy_difference_ci": [0.0, 0.0],
            "ari_difference": 0.0,
            "ari_difference_ci": [0.0, 0.0],
            "balanced_accuracy_difference": 0.0,
            "balanced_accuracy_difference_ci": [0.0, 0.0],
            "macro_f1_difference": 0.0,
            "macro_f1_difference_ci": [0.0, 0.0],
            "n_words": 2,
            "nmi_difference": 0.0,
            "nmi_difference_ci": [0.0, 0.0],
            "pairwise_auc_difference": 0.0,
            "pairwise_auc_difference_ci": [0.0, 0.0],
        },
        "hash": "8a1eab9e0f070776",
        "srp_aggregate": {
            "accuracy": 1.0,
            "accuracy_ci": [1.0, 1.0],
            "ari": 1.0,
            "ari_ci": [1.0, 1.0],
            "balanced_accuracy": 1.0,
            "balanced_accuracy_ci": [1.0, 1.0],
            "macro_f1": 1.0,
            "macro_f1_ci": [1.0, 1.0],
            "n_words": 2,
            "nmi": 1.0,
            "nmi_ci": [1.0, 1.0],
            "pairwise_auc": 1.0,
            "pairwise_auc_ci": [1.0, 1.0],
            "row_gate": {
                "accuracy": 1.0,
                "ari": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "nmi": 1.0,
                "pairwise_auc": 1.0,
            },
            "row_gate_fraction": 1.0,
            "row_gate_words": 2,
        },
    },
    "rw_predictions_hash": "370a835c74b3674d",
    "rw_scoring_summary": {
        "mean_absolute_logit_residual": 0.0,
        "median_absolute_logit_residual": 0.0,
        "median_target_rank": 1.0,
        "n_scored": 120,
        "n_targets": 2,
        "row_gate_fraction_items": 1.0,
        "target_top10_fraction": 1.0,
        "target_top1_fraction": 1.0,
    },
    "rw_shuffled_seed3": "fefe77440592b1af",
    "score": {
        "beta": "4aa2fd96e66a070b",
        "contribution": "7b2444e5e923ea31",
        "exact_logit": "7b694ac7b5c01f09",
        "feature_ids": "e1fe1b77ec0dadd3",
        "hidden": "ae1639028022b00a",
        "metadata": "aad43d9a08a44c58",
        "n": 4,
        "projection": "6d0a986328961978",
        "reconstructed_logit": "f3fe5890e11445ad",
        "target_logprob": "b0ec08f04887fc35",
        "target_rank": "b0f18af3aeb57306",
    },
    "score_summary": {
        "mean_absolute_logit_residual": 1.5911362171173096,
        "median_absolute_logit_residual": 1.7451845407485962,
        "median_target_rank": 11.0,
        "n_scored": 4,
        "n_targets": 2,
        "row_gate_fraction_items": 0.0,
        "target_top10_fraction": 0.25,
        "target_top1_fraction": 0.0,
    },
    "sg_g1": {
        "bank_anchors": {"0": [3], "1": [7]},
        "bank_balanced_ci": [1.0, 1.0],
        "hash": "67da5f98b744d457",
        "seal_anchors": {"0": [1], "1": [9], "2": [13]},
        "seal_null_balanced_mean": 0.30666666666666664,
        "seal_null_balanced_p95": 0.48305555555555546,
        "word_mean_balanced": 1.0,
        "word_mean_balanced_ci": [1.0, 1.0],
        "word_mean_full_account_balanced": 1.0,
        "word_mean_hidden_balanced": 1.0,
        "word_mean_majority_balanced": 0.41666666666666663,
        "word_mean_null_balanced": 0.3616666666666667,
    },
    "sg_g2": {
        "bank_anchors": {"0": [3, 4], "1": [7, 8]},
        "bank_balanced_ci": [1.0, 1.0],
        "hash": "c3b3927e799785e3",
        "seal_anchors": {"0": [1, 2], "1": [9, 10], "2": [13, 14]},
        "seal_null_balanced_mean": 0.33999999999999997,
        "seal_null_balanced_p95": 0.5816666666666666,
        "word_mean_balanced": 1.0,
        "word_mean_balanced_ci": [1.0, 1.0],
        "word_mean_full_account_balanced": 1.0,
        "word_mean_hidden_balanced": 1.0,
        "word_mean_majority_balanced": 0.41666666666666663,
        "word_mean_null_balanced": 0.3933333333333333,
    },
    "sg_g4": {
        "bank_anchors": {"0": [3, 4, 13, 1], "1": [7, 8, 10, 15]},
        "bank_balanced_ci": [1.0, 1.0],
        "hash": "f98c9b8b349b06bf",
        "seal_anchors": {"0": [1, 2, 12, 6], "1": [9, 10, 11, 0], "2": [13, 14, 7, 5]},
        "seal_null_balanced_mean": 0.3827777777777778,
        "seal_null_balanced_p95": 0.6624999999999998,
        "word_mean_balanced": 0.95,
        "word_mean_balanced_ci": [0.9, 1.0],
        "word_mean_full_account_balanced": 1.0,
        "word_mean_hidden_balanced": 1.0,
        "word_mean_majority_balanced": 0.41666666666666663,
        "word_mean_null_balanced": 0.3697222222222223,
    },
    "sg_raw_top1": {"bank_anchors": {"0": [4, 3], "1": [7]}, "hash": "4ac6081f915fc861", "word_mean_balanced": 1.0},
    "spearman": {"const": None, "ok": 0.7999999999999999, "short": None, "tiny": None},
    "stable_seed": {"bank_0": 3184672968, "random_bank_0": 3187410737, "seal_3": 252631022},
}

WIDTH = 16
D_HIDDEN = 8
# word -> {sense: (planted contribution position, n_train, n_test)}
DESIGN = {
    "bank": {"0": (3, 30, 15), "1": (7, 10, 5)},
    "seal": {"0": (1, 20, 10), "1": (9, 12, 6), "2": (13, 8, 4)},
}
VOCAB = [
    "<pad>",
    "<eos>",
    "<unk>",
    " bank",
    " seal",
    "the",
    "river",
    "money",
    "is",
    "of",
    "a",
    "word",
    "missing",
    ":",
    "x",
    "y",
]
PROMPTS = [
    "the river bank is a word :",
    "money of a bank is the word :",
    "the seal of the x y :",
    "a seal is a word missing :",
    "the missing word is :",
]


def make_bundle(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    metadata, contribution, projection, hidden, betas = [], [], [], [], []
    for word_index, (word, senses) in enumerate(DESIGN.items()):
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
                            "token_id": 100 + word_index,
                            "row_relative_error": 0.2,
                        }
                    )
                    projection.append(p)
                    contribution.append(beta * p)
                    betas.append(beta)
                    hidden.append(h)
    contribution = torch.tensor(np.stack(contribution))
    n = contribution.shape[0]
    return {
        "metadata": metadata,
        "hidden": torch.tensor(np.stack(hidden)),
        "projection": torch.tensor(np.stack(projection)),
        "contribution": contribution,
        "beta": torch.tensor(np.stack(betas)),
        "feature_ids": torch.stack([torch.arange(WIDTH, dtype=torch.int32)] * n),
        "exact_logit": contribution.sum(dim=1) + 5.0,
        "reconstructed_logit": contribution.sum(dim=1) + 5.0,
        "target_logprob": torch.zeros(n),
        "target_rank": torch.ones(n, dtype=torch.int32),
        "run": {
            "dataset": "coarsewsd20",
            "model_id": "synthetic",
            "k": WIDTH,
            "seed": seed,
        },
    }


class _Batch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    """Whitespace tokenizer over a fixed vocabulary. ``' word'`` forms are single
    tokens when present; unknown continuations tokenize to two ids."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    eos_token_id = 1

    def __init__(self, vocab: list[str], special_ids: tuple[int, ...] = (0, 1)):
        self.vocab = list(vocab)
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.padding_side = "right"
        self.truncation_side = "right"
        self.all_special_ids = list(special_ids)

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.index:
            return [self.index[text]]
        return [2, 2]

    def _ids(self, text: str) -> list[int]:
        return [self.index.get(w, 2) for w in text.split()]

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=True,
        truncation=False,
        max_length=None,
    ):
        seqs = [self._ids(t) for t in texts]
        if not padding:
            return _Batch(input_ids=seqs, attention_mask=[[1] * len(s) for s in seqs])
        if truncation and max_length is not None:
            seqs = [s[-max_length:] if self.truncation_side == "left" else s[:max_length] for s in seqs]
        width = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [self.pad_token_id] * (width - len(s))
            if self.padding_side == "left":
                ids.append(pad + s)
                mask.append([0] * len(pad) + [1] * len(s))
            else:
                ids.append(s + pad)
                mask.append([1] * len(s) + [0] * len(pad))
        return _Batch(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(mask, dtype=torch.long),
        )


class FakeLM(torch.nn.Module):
    """Causal bag-of-tokens LM: the state at position t is the running mean of the
    real-token embeddings up to t, so context (and truncation) changes the state."""

    def __init__(self, vocab: int, d_model: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(vocab, d_model, generator=gen))
        self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, d_model, generator=gen))

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        emb = self.table[input_ids]  # (B, T, d)
        m = attention_mask[..., None].to(emb.dtype)
        h = (emb * m).cumsum(dim=1) / m.cumsum(dim=1).clamp_min(1.0)
        return SimpleNamespace(logits=self.lm_head(h))


def make_checkpoint(
    path: Path,
    d_model: int = 8,
    d_features: int = 32,
    k: int = 6,
    *,
    runner_layout: bool = False,
):
    torch.manual_seed(0)
    cfg = {"architecture": "topk", "d_features": d_features, "k": k}
    model = build_factorizer({"factorizer": cfg}, d_model=d_model)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.randn(d_features, d_model))
        model.encoder.bias.copy_(0.1 * torch.randn(d_features))
        model.decoder.copy_(torch.randn(d_features, d_model))
        model.normalize_decoder_()
    ckpt = {"model_state_dict": model.state_dict()}
    if runner_layout:
        ckpt["factorizer"] = cfg  # what runner.py writes: no "config" key
    else:
        ckpt["config"] = {"factorizer": cfg}
    torch.save(ckpt, path)


def _strip(obj, drop: set[str]):
    if isinstance(obj, dict):
        return {k: _strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_strip(v, drop) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    return obj


def _canon(obj, drop=()) -> str:
    return hashlib.sha256(json.dumps(_strip(obj, set(drop)), sort_keys=True).encode()).hexdigest()[:16]


def _array_hash(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()[:16]


def _nan_none(value):
    return None if (isinstance(value, float) and math.isnan(value)) else value


def _scoring_setup(tmp_path: Path, *, runner_layout: bool = False):
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(path, runner_layout=runner_layout)
    tok = FakeTokenizer(VOCAB)
    lm = FakeLM(len(VOCAB), D_HIDDEN, seed=0)
    w_u = lm.lm_head.weight.detach().float()
    w_dec_unit, w_enc, b_enc, sae_config, ckpt_row_mean = load_sae(path)
    k = rw.checkpoint_k(path, sae_config)
    row_mean = rw.scoring_row_mean(w_u, tok, "live", ckpt_row_mean)
    info, audit = rw.prepare_target_codes(["bank", "seal", "multi"], tok, w_u, row_mean, w_enc, b_enc, w_dec_unit, k)
    return tok, lm, w_u, row_mean, w_dec_unit, info, audit, k


def _items(prompts):
    targets = ["bank", "bank", "seal", "seal", "multi"]
    return [
        {
            "item_id": f"i{j}",
            "kind": "context",
            "dataset": "coarsewsd20",
            "split": "train" if j % 2 == 0 else "test",
            "target": targets[j],
            "sense": str(j % 2),
            "sense_name": str(j % 2),
            "prefix_words": 1,
            "prompt": p,
        }
        for j, p in enumerate(prompts)
    ]


# --------------------------------------------------------------------------- #
# sense groups
# --------------------------------------------------------------------------- #


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


def test_sense_groups_cli_reproduces_pinned_outputs(tmp_path):
    bundle_path = tmp_path / "representations.pt"
    torch.save(make_bundle(), bundle_path)
    out = tmp_path / "out" / "synthetic__srp.json"
    base = ["--bundle", str(bundle_path), "--out", str(out)]
    assert (
        sg.main(
            base
            + [
                "--group-sizes",
                "1,2,4",
                "--n-boot",
                "20",
                "--n-null",
                "10",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    for g in (1, 2, 4):
        path = out.with_name(f"synthetic__srp_g{g}.json")
        summary = json.loads(path.read_text())
        pin = PINS[f"sg_g{g}"]
        assert _canon(summary, drop=("bundle", "provenance")) == pin["hash"], g
        assert summary["word_mean_balanced"] == pin["word_mean_balanced"]
        assert summary["word_mean_null_balanced"] == pin["word_mean_null_balanced"]
        assert summary["word_mean_majority_balanced"] == pin["word_mean_majority_balanced"] == (1 / 2 + 1 / 3) / 2
        assert summary["word_mean_full_account_balanced"] == pin["word_mean_full_account_balanced"]
        assert summary["word_mean_hidden_balanced"] == pin["word_mean_hidden_balanced"]
        assert summary["word_mean_balanced_ci"] == pin["word_mean_balanced_ci"]
        assert summary["per_word"]["bank"]["anchors"] == pin["bank_anchors"]
        assert summary["per_word"]["seal"]["anchors"] == pin["seal_anchors"]
        assert summary["per_word"]["bank"]["balanced_accuracy_ci"] == pin["bank_balanced_ci"]
        assert summary["per_word"]["seal"]["null_balanced_p95"] == pin["seal_null_balanced_p95"]
        assert summary["group_size"] == g and summary["basis"] == "srp" and summary["standardized"] is True
        assert summary["words_beating_majority_balanced"] == 2 and summary["words_beating_null_p95_balanced"] == 2
        assert summary["gate_fraction_overall"] == 1.0
        assert "command" in summary["provenance"] and summary["provenance"]["args"]["seed"] == 0
        for word, senses in DESIGN.items():
            pw = summary["per_word"][word]
            assert pw["n_senses"] == len(senses) and all(len(pw["anchors"][s]) == g for s in senses)
    # The other selector, unscaled contributions and a non-zero seed.
    raw = tmp_path / "raw_top1.json"
    args = [
        "--group-size",
        "2",
        "--selector",
        "top1_freq",
        "--raw-scale",
        "--n-boot",
        "20",
        "--n-null",
        "10",
    ]
    assert sg.main(base[:2] + ["--out", str(raw)] + args + ["--seed", "3"]) == 0
    summary = json.loads(raw.read_text())
    assert _canon(summary, drop=("bundle", "provenance")) == PINS["sg_raw_top1"]["hash"]
    assert summary["per_word"]["bank"]["anchors"] == PINS["sg_raw_top1"]["bank_anchors"]
    # Single-size runs write --out verbatim and match the sweep's file for that size.
    single = tmp_path / "single.json"
    assert (
        sg.main(
            base[:2]
            + [
                "--out",
                str(single),
                "--group-size",
                "1",
                "--n-boot",
                "20",
                "--n-null",
                "10",
            ]
        )
        == 0
    )
    a = json.loads(single.read_text())
    b = json.loads(out.with_name("synthetic__srp_g1.json").read_text())
    assert a["per_word"] == b["per_word"]


def test_geometry_controls_route_centering_through_centering_mean(tmp_path):
    bundle = make_bundle()
    bundle_path = tmp_path / "representations.pt"
    torch.save(bundle, bundle_path)
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(128, D_HIDDEN, generator=gen)
    mask_all = torch.ones(128, dtype=torch.bool)
    # The controls' row mean is data.centering_mean: live is the full-vocabulary mean ...
    live = sg.geometry_row_mean(W, "live", token_mask=None, checkpoint=None, model_id="x", revision=None)
    assert torch.equal(live, centering_mean(W, mode="live")) and torch.equal(live, W.mean(dim=0))
    # ... and trained with an all-True token_mask (no checkpoint) is the same vector.
    trained = sg.geometry_row_mean(W, "trained", token_mask=mask_all, checkpoint=None, model_id="x", revision=None)
    assert torch.equal(trained, live)
    partial = mask_all.clone()
    partial[:8] = False
    assert not torch.equal(
        sg.geometry_row_mean(
            W,
            "trained",
            token_mask=partial,
            checkpoint=None,
            model_id="x",
            revision=None,
        ),
        live,
    )
    # geometry_contributions takes that mean rather than recomputing it.
    tids = [100, 101]
    hidden = {
        tid: bundle["hidden"][[i for i, r in enumerate(bundle["metadata"]) if r["token_id"] == tid]] for tid in tids
    }
    geo = sg.geometry_contributions("knn128", W, live, tids, hidden, 4, 4, 1e-3, 0)
    assert set(geo) == set(tids) and geo[100]["contribution"].shape == (
        len(hidden[100]),
        4,
    )
    np.testing.assert_allclose(geo[100]["exact"], (hidden[100] @ W[100]).numpy(), rtol=1e-6)
    # End to end through --w-u: trained (payload token_mask all True) == live, provenance aside.
    outputs = {}
    for mode in ("live", "trained"):
        torch.save({"W_U_orig": W, "token_mask": mask_all}, tmp_path / f"wu_{mode}.pt")
        out = tmp_path / f"knn_{mode}.json"
        argv = [
            "--bundle",
            str(bundle_path),
            "--out",
            str(out),
            "--basis",
            "knn128",
            "--neighbor-k",
            "4",
        ]
        argv += [
            "--w-u",
            str(tmp_path / f"wu_{mode}.pt"),
            "--centering",
            mode,
            "--n-boot",
            "5",
            "--n-null",
            "3",
        ]
        assert sg.main(argv) == 0
        outputs[mode] = json.loads(out.read_text())
        assert outputs[mode]["basis"] == "knn128" and outputs[mode]["n_words"] == 2
    assert _canon(outputs["live"], drop=("provenance",)) == _canon(outputs["trained"], drop=("provenance",))
    assert outputs["trained"]["provenance"]["args"]["centering"] == "trained"


def test_load_wu_reads_the_resolved_local_snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    from safetensors.torch import save_file

    snap = tmp_path / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    head = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    save_file(
        {"model.embed_tokens.weight": torch.zeros(4, 3)},
        str(snap / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"lm_head.weight": head.to(torch.bfloat16)},
        str(snap / "model-00002-of-00002.safetensors"),
    )
    seen = {}

    def fake_snapshot_download(model_id, **kwargs):
        seen.update(model_id=model_id, **kwargs)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    W = sg.load_wu("org/model", revision="abc123")
    assert torch.equal(W, head) and W.dtype == torch.float32
    assert seen == {
        "model_id": "org/model",
        "revision": "abc123",
        "local_files_only": True,
    }
    # Tied-embedding fallback when no lm_head.weight is stored.
    (snap / "model-00002-of-00002.safetensors").unlink()
    assert torch.equal(sg.load_wu("org/model"), torch.zeros(4, 3)) and seen["revision"] is None


# --------------------------------------------------------------------------- #
# classifier framing
# --------------------------------------------------------------------------- #


def test_classifier_framing_reproduces_pinned_outputs(tmp_path):
    assert cf.self_test() == 0
    bundle = make_bundle()
    gate, gate_summary = cf.score_gate(bundle)
    assert gate.all() and gate_summary == PINS["cf"]["gate"]
    metrics = cf.analyze_coarse(
        bundle,
        ks=[2, 4],
        primary_k=2,
        primary_encoding="weighted",
        n_boot=5,
        n_null_seeds=2,
        seed=0,
    )
    assert _canon(metrics) == PINS["cf"]["hash"]
    assert metrics["primary"]["methods"] == PINS["cf"]["primary_methods"]
    assert metrics["primary"]["comparisons"]["srp_minus_projection_only"] == PINS["cf"]["cmp_projection"]
    assert metrics["primary"]["null_seed_sensitivity"]["shuffled_srp"] == PINS["cf"]["null_shuffled"]
    assert metrics["results"]["signed"]["4"]["srp"]["aggregate"] == PINS["cf"]["results_signed_4_srp_agg"]
    assert set(metrics["primary"]["methods"]) == set(cf.METHODS)
    assert set(metrics["primary"]["comparisons"]) == {f"srp_minus_{m}" for m in cf.METHODS[1:]}
    assert (
        _array_hash(cf.encode_top_features(bundle, "random_srp_features", 3, "signed", 2)) == PINS["cf_random_enc_hash"]
    )
    # CLI: the file carries the analysis block and provenance, on top of the pinned metrics.
    bundle_path = tmp_path / "representations.pt"
    torch.save(bundle, bundle_path)
    out = tmp_path / "cf" / "out.json"
    argv = [
        "--bundle",
        str(bundle_path),
        "--out",
        str(out),
        "--ks",
        "2,4",
        "--primary-k",
        "4",
    ]
    argv += [
        "--primary-encoding",
        "signed",
        "--n-boot",
        "7",
        "--n-null-seeds",
        "3",
        "--seed",
        "5",
    ]
    assert cf.main(argv) == 0
    written = json.loads(out.read_text())
    assert _canon(written, drop=("bundle", "provenance")) == PINS["cf_cli"]["hash"]
    assert _strip(written["analysis"], {"bundle"}) == PINS["cf_cli"]["analysis"]
    assert written["primary"]["methods"]["srp"] == PINS["cf_cli"]["primary_srp"]
    assert written["provenance"]["args"]["primary_k"] == 4 and "command" in written["provenance"]


def test_shuffled_srp_null_is_the_pre_merge_construction():
    bundle = make_bundle()
    projection = bundle["projection"].numpy()
    beta = bundle["beta"].numpy()
    for seed in (0, 3):
        shuffled = wsd.shuffled_srp(projection, beta, bundle["metadata"], seed)
        assert _array_hash(shuffled) == PINS["cf_shuffled_hash"][str(seed)]
        assert _array_hash(cf.source_matrix(bundle, "shuffled_srp", seed)) == PINS["cf_shuffled_hash"][str(seed)]
        assert not np.array_equal(shuffled, bundle["contribution"].numpy())
    matrices = rw.method_matrices(bundle, 0)
    assert {name: _array_hash(m) for name, m in matrices.items()} == PINS["rw_method_hashes"]
    assert _array_hash(rw.method_matrices(bundle, 3)["shuffled_srp"]) == PINS["rw_shuffled_seed3"]
    # One seed form for both scripts: the classifier framing's modular sha1 prefix ...
    assert {
        "bank_0": wsd.stable_seed("bank", 0),
        "seal_3": wsd.stable_seed("seal", 3),
        "random_bank_0": wsd.stable_seed("random:bank", 0),
    } == PINS["stable_seed"]
    # ... which coincides with the run script's former ``seed + int(sha1[:8], 16)`` at seed 0.
    for target in ("bank", "seal", "apple"):
        prefix = int(hashlib.sha1(target.encode()).hexdigest()[:8], 16)
        assert wsd.stable_seed(target, 0) == prefix
        assert wsd.stable_seed(target, 3) == (prefix + 3) % 2**32


# --------------------------------------------------------------------------- #
# run script: analysis, statistics, scoring path
# --------------------------------------------------------------------------- #


def test_run_analysis_reproduces_pinned_outputs(tmp_path):
    bundle = make_bundle()
    metrics = rw.analyze_coarsewsd(bundle, tmp_path, 20, 0)
    assert _canon(metrics) == PINS["rw_metrics"]["hash"]
    assert metrics["methods"]["srp"]["aggregate"] == PINS["rw_metrics"]["srp_aggregate"]
    assert metrics["comparisons"]["srp_minus_hidden"] == PINS["rw_metrics"]["cmp_hidden"]
    assert metrics["methods"]["hidden"]["per_word"]["bank"] == PINS["rw_metrics"]["bank_hidden_per_word"]
    digest = hashlib.sha256((tmp_path / "predictions.jsonl").read_bytes()).hexdigest()[:16]
    assert digest == PINS["rw_predictions_hash"]
    assert rw.summarize_scoring(bundle) == PINS["rw_scoring_summary"]
    assert list(rw.METRIC_NAMES) == [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "pairwise_auc",
        "ari",
        "nmi",
    ]


def test_statistics_helpers_reproduce_pinned_values():
    rng = np.random.default_rng(0)
    big = rng.normal(size=(250, 8)).astype(np.float32)
    labels = rng.integers(0, 3, 250)
    big[labels == 1] += 0.5
    big = wsd.l2_normalize(big)
    assert rw.sampled_pair_auc(big, labels, 0) == PINS["auc"]["big_0"]  # sampled branch (31125 > 20000 pairs)
    assert rw.sampled_pair_auc(big, labels, 3) == PINS["auc"]["big_3"]
    assert rw.sampled_pair_auc(big[:40], labels[:40], 0) == PINS["auc"]["small_0"]  # exhaustive branch
    assert _nan_none(rw.sampled_pair_auc(big[:10], np.zeros(10, int), 0)) == PINS["auc"]["one_class"]
    rows = [{"g": i % 7, "x": float(i), "y": float((i * 37) % 11)} for i in range(40)]
    values = wsd.cluster_bootstrap(
        rows,
        "g",
        lambda s: wsd.safe_spearman([r["x"] for r in s], [r["y"] for r in s]),
        50,
        0,
    )
    assert len(values) == PINS["cb"]["n"] and values[:3] == PINS["cb"]["first3"]
    assert wsd.percentile_ci(values) == PINS["cb"]["ci"]
    assert all(math.isnan(v) for v in wsd.percentile_ci([])) and all(
        math.isnan(v) for v in wsd.percentile_ci(np.array([]))
    )
    assert wsd.percentile_ci(np.asarray(values)) == wsd.percentile_ci(values)
    assert _nan_none(wsd.safe_spearman([1, 1, 1], [1, 2, 3])) == PINS["spearman"]["const"]
    assert _nan_none(wsd.safe_spearman([1, 2], [1, 2])) == PINS["spearman"]["short"]
    assert wsd.safe_spearman([1, 2, 3, 4], [1, 3, 2, 4]) == PINS["spearman"]["ok"]
    # The documented difference from utils.spearman: a range under 1e-8 is degenerate here, correlated there.
    tiny = [0.0, 1e-9, 2e-9]
    assert _nan_none(wsd.safe_spearman(tiny, [1, 2, 3])) == PINS["spearman"]["tiny"] is None
    assert spearman(tiny, [1, 2, 3]) == pytest.approx(1.0)
    gen = np.random.default_rng(1)
    vals = gen.normal(size=9)
    assert (
        wsd.bootstrap_mean(vals, np.random.default_rng(4), 3)
        == [float(np.random.default_rng(4).choice(vals, size=9, replace=True).mean()) for _ in range(1)]
        + wsd.bootstrap_mean(vals, np.random.default_rng(4), 3)[1:]
    )


def test_prepare_target_codes_matches_the_pre_refactor_path(tmp_path):
    tok, lm, w_u, row_mean, w_dec_unit, info, audit, k = _scoring_setup(tmp_path)
    assert k == PINS["ptc"]["k"] == 6
    assert audit == PINS["ptc"]["audit"]
    for target in ("bank", "seal"):
        assert info[target]["feature_ids"].tolist() == PINS["ptc"][f"{target}_fids"]
        assert _array_hash(info[target]["beta"].numpy()) == PINS["ptc"][f"{target}_beta"]
    # Runner-layout checkpoints (top-level ``factorizer``, no ``config``) yield the same k via the mmap read.
    (tmp_path / "runner").mkdir()
    _tok, _lm, _w, _rm, _wd, info_runner, audit_runner, k_runner = _scoring_setup(
        tmp_path / "runner", runner_layout=True
    )
    assert k_runner == 6 and audit_runner == audit
    assert torch.equal(info_runner["bank"]["beta"], info["bank"]["beta"])
    assert rw.checkpoint_k(tmp_path / "checkpoint.pt", {}) == 6


def test_score_items_matches_the_pre_refactor_path(tmp_path):
    tok, lm, w_u, row_mean, w_dec_unit, info, _audit, _k = _scoring_setup(tmp_path)
    rw.configure_tokenizer(tok)
    assert (tok.padding_side, tok.truncation_side) == ("left", "left")
    bundle = rw.score_items(_items(PROMPTS), lm, tok, lm.lm_head, row_mean, w_dec_unit, info, 2, 16, "cpu")
    assert len(bundle["metadata"]) == PINS["score"]["n"] == 4  # the multi-token target is dropped
    assert _canon(bundle["metadata"]) == PINS["score"]["metadata"]
    for key in wsd.BUNDLE_TENSOR_KEYS:
        assert _array_hash(bundle[key].numpy()) == PINS["score"][key], key
    assert bundle["truncation"] == {
        "max_length": 16,
        "truncation_side": "left",
        "n_truncated_prompts": 0,
        "max_prompt_tokens": max(len(p.split()) for p in PROMPTS[:4]),
    }
    summary = rw.summarize_scoring(bundle)
    assert summary.pop("truncation") == bundle["truncation"]
    assert summary == PINS["score_summary"]
    bundle["run"] = {"dataset": "coarsewsd20"}
    wsd.validate_bundle(bundle)


def test_left_truncation_keeps_the_cloze_cue(tmp_path):
    tok, lm, w_u, row_mean, w_dec_unit, info, _audit, _k = _scoring_setup(tmp_path)
    rw.configure_tokenizer(tok)
    long_prompt = "x y x y x y x y the river bank is the missing word :"
    tail = " ".join(long_prompt.split()[-6:])
    head = " ".join(long_prompt.split()[:6])

    def hidden_of(prompt, max_length):
        return rw.score_items(
            _items([prompt] * 5)[:1],
            lm,
            tok,
            lm.lm_head,
            row_mean,
            w_dec_unit,
            info,
            1,
            max_length,
            "cpu",
        )

    truncated = hidden_of(long_prompt, 6)
    assert truncated["truncation"]["n_truncated_prompts"] == 1 and truncated["truncation"]["max_prompt_tokens"] == len(
        long_prompt.split()
    )
    assert torch.equal(truncated["hidden"], hidden_of(tail, 64)["hidden"])  # the cue survives ...
    assert not torch.equal(truncated["hidden"], hidden_of(head, 64)["hidden"])  # ... unlike the old right truncation
    tok.truncation_side = "right"
    assert torch.equal(hidden_of(long_prompt, 6)["hidden"], hidden_of(head, 64)["hidden"])


def test_centering_trained_equals_live_when_the_mask_keeps_all_rows():
    lm = FakeLM(len(VOCAB), D_HIDDEN, seed=0)
    w_u = lm.lm_head.weight.detach().float()
    live = rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "live", None)
    assert torch.equal(live, w_u.mean(dim=0))
    assert torch.equal(
        rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB, special_ids=()), "trained", None),
        live,
    )
    masked = rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "trained", None)  # rows 0 and 1 are special ids
    assert torch.equal(masked, w_u[2:].mean(dim=0)) and not torch.equal(masked, live)
    stored = torch.full((D_HIDDEN,), 0.25)
    assert torch.equal(rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "trained", stored), stored)
    assert torch.equal(rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "live", stored), live)


def test_analyze_only_writes_analysis_config_and_keeps_run_config(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    torch.save(make_bundle(), out_dir / "representations.pt")
    sentinel = '{"scoring_run": true}\n'
    (out_dir / "run_config.json").write_text(sentinel)
    assert (
        rw.main(
            [
                "--analyze-only",
                "--out-dir",
                str(out_dir),
                "--n-boot",
                "5",
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert (out_dir / "run_config.json").read_text() == sentinel
    analysis = json.loads((out_dir / "analysis_config.json").read_text())
    assert analysis["provenance"]["args"]["analyze_only"] is True and "command" in analysis["provenance"]
    assert analysis["coarsewsd_template"] == rw.COARSEWSD_TEMPLATE
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["provenance"] == analysis["provenance"]
    assert metrics["methods"]["srp"]["aggregate"]["accuracy"] == 1.0 and metrics["run"]["dataset"] == "coarsewsd20"
    assert (out_dir / "predictions.jsonl").exists()
    with pytest.raises(SystemExit):
        rw.main(["--analyze-only"])


# --------------------------------------------------------------------------- #
# shared bundle helpers and removed paths
# --------------------------------------------------------------------------- #


def test_bundle_loading_and_validation(tmp_path):
    bundle = make_bundle()
    path = tmp_path / "representations.pt"
    torch.save(bundle, path)
    loaded = wsd.load_bundle(path)
    assert loaded["metadata"] == bundle["metadata"] and torch.equal(loaded["contribution"], bundle["contribution"])
    broken = dict(bundle)
    del broken["beta"]
    with pytest.raises(KeyError):
        wsd.validate_bundle(broken)
    broken = dict(bundle, hidden=bundle["hidden"][:-1])
    with pytest.raises(ValueError, match="rows"):
        wsd.validate_bundle(broken)
    with pytest.raises(ValueError, match="dataset"):
        wsd.validate_bundle(dict(bundle, run={"dataset": "ambistory"}))
    drifted = bundle["feature_ids"].clone()
    drifted[1, 0] = 99
    with pytest.raises(ValueError, match="feature coordinates"):
        wsd.validate_bundle(dict(bundle, feature_ids=drifted))


def test_word_splits_apply_the_shared_rule():
    bundle = make_bundle()
    splits = wsd.word_splits(bundle["metadata"])
    assert [s.word for s in splits] == ["bank", "seal"]
    bank = splits[0]
    assert bank.senses == ["0", "1"] and len(bank.train_idx) == 40 and len(bank.test_idx) == 20
    assert set(bank.train_y) == {"0", "1"} and bank.train_idx.dtype == np.int64
    metadata = [
        {"target": "one", "split": "train", "sense": "0"},
        {"target": "one", "split": "test", "sense": "0"},
        {"target": "two", "split": "train", "sense": "0"},
        {"target": "two", "split": "train", "sense": "1"},
        {"target": "two", "split": "test", "sense": "9"},
        {"target": "three", "split": "train", "sense": "0"},
        {"target": "three", "split": "train", "sense": "1"},
        {"target": "three", "split": "test", "sense": "1"},
    ]
    assert [s.word for s in wsd.word_splits(metadata)] == ["three"]  # one sense; unseen test sense
    keep = np.array([True] * 7 + [False])
    assert wsd.word_splits(metadata, keep=keep) == []  # the gate emptied three's test split


def test_removed_paths_are_gone():
    with pytest.raises(SystemExit):
        rw.parse_args(["--dataset", "ambistory"])
    args = rw.parse_args([])
    assert args.dataset == "coarsewsd20" and args.centering == "live" and args.splits == "train,test"
    assert not hasattr(args, "anchors") and not hasattr(args, "position_mode")
    for name in (
        "load_ambistory",
        "analyze_ambistory",
        "make_story",
        "mask_exact_target",
    ):
        assert not hasattr(rw, name), name
    for name in ("ambistory_rows", "analyze_ambistory", "summarize_ambistory_rows"):
        assert not hasattr(cf, name), name
    assert rw.self_test() == 0
