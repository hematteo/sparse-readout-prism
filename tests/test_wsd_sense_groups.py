"""CoarseWSD-20 sense analyses and the bundle writer, pinned on synthetic fixtures.

* scripts/run/run_wsd_feature_alignment.py          (bundle writer, full-vector references)
* scripts/analyze/analyze_wsd_sense_groups.py       (tab:app-sense-alignment)
* scripts/analyze/analyze_wsd_classifier_framing.py (classifier-framing paragraph)
* sparse_readout_prism.research.wsd                 (their shared helpers)

``tests/fixtures/wsd_pins.json`` holds the outputs of these scripts on the
fixtures below, captured after the 0.2.1 merge of the per-script helper copies
was shown to reproduce the pre-merge outputs exactly on the capture machine.
Floats are compared with ``rel=1e-5, atol=1e-6`` (BLAS/platform noise sits
near 1e-7 relative; bundle tensors stored in float16 get two fp16 ulps);
counts, ids, orderings, flags and strings are compared exactly. Where the
refactor touched arithmetic (the target codes, the shuffle null) a verbatim
copy of the pre-0.2.1 computation also runs in-process and must agree.
Regenerate the fixture with ``uv run --no-sync python tests/test_wsd_sense_groups.py``.

The synthetic bundle mimics ``representations.pt``: two words, two or three
senses, 16 dictionary positions, a strong and a weaker planted discriminative
position per sense. No model, no network, no Hugging Face cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import load_script

from sparse_readout_prism.data import centering_mean
from sparse_readout_prism.factorizers import build_factorizer
from sparse_readout_prism.research import wsd
from sparse_readout_prism.research.qwen_readout import load_sae
from sparse_readout_prism.utils import spearman, to_jsonable

rw = load_script("scripts/run/run_wsd_feature_alignment.py")
sg = load_script("scripts/analyze/analyze_wsd_sense_groups.py")
cf = load_script("scripts/analyze/analyze_wsd_classifier_framing.py")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wsd_pins.json"
PINS: dict = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {}
REL, ABS = 1e-5, 1e-6  # float tolerance against the fixture
HALF_REL, HALF_ABS = 2e-3, 1e-5  # bundle tensors stored as float16: two fp16 ulps
HALF_KEYS = ("hidden", "projection", "contribution", "beta")

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
TARGETS = ["bank", "bank", "seal", "seal", "multi"]
PROMPTS = [
    "the river bank is a word :",
    "money of a bank is the word :",
    "the seal of the x y :",
    "a seal is a word missing :",
    "the missing word is :",
]


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #


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
        "run": {"dataset": "coarsewsd20", "model_id": "synthetic", "k": WIDTH, "seed": seed},
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

    def __call__(self, texts, return_tensors=None, padding=True, truncation=False, max_length=None):
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
            input_ids=torch.tensor(ids, dtype=torch.long), attention_mask=torch.tensor(mask, dtype=torch.long)
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


def make_checkpoint(path: Path, d_model: int = 8, d_features: int = 32, k: int = 6, *, runner_layout: bool = False):
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


# --------------------------------------------------------------------------- #
# fixture comparison
# --------------------------------------------------------------------------- #


def _assert_close(actual, expected, path: str = "$") -> None:
    """Recursive comparison against a fixture value: floats within REL/ABS, everything else exact."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: {type(actual).__name__} is not a dict"
        assert set(actual) == set(expected), f"{path}: key mismatch {sorted(set(actual) ^ set(expected))}"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: {type(actual).__name__} is not a list"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert isinstance(actual, float), f"{path}: {actual!r} is not a float"
        assert math.isclose(actual, expected, rel_tol=REL, abs_tol=ABS), f"{path}: {actual!r} != {expected!r}"
    else:  # bool, int, str, None: exact, same type
        assert type(actual) is type(expected) and actual == expected, f"{path}: {actual!r} != {expected!r}"


def _pinned(section: str):
    assert section in PINS, f"{section!r} missing from {FIXTURE}; regenerate with `python tests/{Path(__file__).name}`"
    return PINS[section]


def _check(section: str, actual) -> dict:
    """Normalize the actual (numpy -> python, NaN -> None), compare with the fixture section, return it."""
    actual = to_jsonable(actual)
    _assert_close(actual, _pinned(section), section)
    return actual


def _array_pin(array) -> dict:
    """Platform-robust array pin: full values when small, shape + scale statistics + a sample when large."""
    a = np.asarray(array, dtype=np.float64)
    flat = a.ravel()
    if a.size <= 256:
        return {"shape": list(a.shape), "values": flat.tolist()}
    return {
        "shape": list(a.shape),
        "abs_sum": float(np.abs(flat).sum()),
        "sq_sum": float((flat * flat).sum()),
        "head": flat[:32].tolist(),
        "col_means": a.reshape(a.shape[0], -1).mean(axis=0).tolist(),
    }


def _strip(obj, drop: tuple[str, ...] = ("bundle", "provenance")):
    if isinstance(obj, dict):
        return {k: _strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_strip(v, drop) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# pinned computations (shared between the tests and the fixture regeneration)
# --------------------------------------------------------------------------- #


def _sense_groups_actual(tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    bundle_path = tmp / "representations.pt"
    torch.save(make_bundle(), bundle_path)
    out = tmp / "out" / "synthetic__srp.json"
    base = ["--bundle", str(bundle_path), "--out", str(out)]
    assert sg.main(base + ["--group-sizes", "1,2,4", "--n-boot", "20", "--n-null", "10", "--seed", "0"]) == 0
    actual = {f"g{g}": _strip(json.loads(out.with_name(f"synthetic__srp_g{g}.json").read_text())) for g in (1, 2, 4)}
    raw = tmp / "raw_top1.json"
    flags = ["--group-size", "2", "--selector", "top1_freq", "--raw-scale", "--n-boot", "20", "--n-null", "10"]
    assert sg.main(base[:2] + ["--out", str(raw)] + flags + ["--seed", "3"]) == 0
    actual["raw_top1"] = _strip(json.loads(raw.read_text()))
    return actual


def _classifier_actual(tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    bundle = make_bundle()
    gate, gate_summary = cf.score_gate(bundle)
    metrics = cf.analyze_coarse(
        bundle, ks=[2, 4], primary_k=2, primary_encoding="weighted", n_boot=5, n_null_seeds=2, seed=0
    )
    bundle_path = tmp / "representations.pt"
    torch.save(bundle, bundle_path)
    out = tmp / "cf" / "out.json"
    argv = ["--bundle", str(bundle_path), "--out", str(out), "--ks", "2,4", "--primary-k", "4"]
    argv += ["--primary-encoding", "signed", "--n-boot", "7", "--n-null-seeds", "3", "--seed", "5"]
    assert cf.main(argv) == 0
    encoded = cf.encode_top_features(bundle, "random_srp_features", 3, "signed", 2)
    positions = {}
    for word in DESIGN:
        rows = [i for i, r in enumerate(bundle["metadata"]) if r["target"] == word]
        positions[word] = sorted(np.flatnonzero(np.abs(encoded[rows]).sum(axis=0) > 0).tolist())
    return {
        "gate_all": bool(gate.all()),
        "gate": gate_summary,
        "metrics": metrics,
        "cli": _strip(json.loads(out.read_text())),
        "random_positions": positions,
        "random_encoded": _array_pin(encoded),
    }


def _shuffle_actual() -> dict:
    bundle = make_bundle()
    projection = bundle["projection"].numpy()
    beta = bundle["beta"].numpy()
    actual: dict = {"permutations": {}, "shuffled": {}}
    for seed in (0, 3):
        for target in DESIGN:
            perm = np.random.default_rng(wsd.stable_seed(target, seed)).permutation(WIDTH)
            actual["permutations"][f"{target}:{seed}"] = perm.tolist()
        actual["shuffled"][str(seed)] = _array_pin(wsd.shuffled_srp(projection, beta, bundle["metadata"], seed))
    actual["method_matrices"] = {name: _array_pin(m) for name, m in rw.method_matrices(bundle, 0).items()}
    actual["method_matrices_seed3_shuffled"] = _array_pin(rw.method_matrices(bundle, 3)["shuffled_srp"])
    actual["stable_seed"] = {
        "bank_0": wsd.stable_seed("bank", 0),
        "seal_3": wsd.stable_seed("seal", 3),
        "random_bank_0": wsd.stable_seed("random:bank", 0),
    }
    return actual


def _run_analysis_actual(tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    bundle = make_bundle()
    metrics = rw.analyze_coarsewsd(bundle, tmp, 20, 0)
    predictions = [json.loads(line) for line in (tmp / "predictions.jsonl").read_text().splitlines()]
    return {"metrics": metrics, "predictions": predictions, "scoring_summary": rw.summarize_scoring(bundle)}


def _statistics_actual() -> dict:
    rng = np.random.default_rng(0)
    big = rng.normal(size=(250, 8)).astype(np.float32)
    labels = rng.integers(0, 3, 250)
    big[labels == 1] += 0.5
    big = wsd.l2_normalize(big)
    rows = [{"g": i % 7, "x": float(i), "y": float((i * 37) % 11)} for i in range(40)]
    values = wsd.cluster_bootstrap(
        rows, "g", lambda s: wsd.safe_spearman([r["x"] for r in s], [r["y"] for r in s]), 50, 0
    )
    sample = np.random.default_rng(1).normal(size=9)
    return {
        "auc": {
            "sampled_seed0": rw.sampled_pair_auc(big, labels, 0),  # sampled branch: 31125 > 20000 pairs
            "sampled_seed3": rw.sampled_pair_auc(big, labels, 3),
            "exhaustive": rw.sampled_pair_auc(big[:40], labels[:40], 0),
            "one_class": rw.sampled_pair_auc(big[:10], np.zeros(10, int), 0),
        },
        "cluster_bootstrap": {"n": len(values), "values": values, "ci": wsd.percentile_ci(values)},
        "spearman": {
            "constant": wsd.safe_spearman([1, 1, 1], [1, 2, 3]),
            "short": wsd.safe_spearman([1, 2], [1, 2]),
            "tiny_range": wsd.safe_spearman([0.0, 1e-9, 2e-9], [1, 2, 3]),
            "ok": wsd.safe_spearman([1, 2, 3, 4], [1, 3, 2, 4]),
        },
        "bootstrap_mean": wsd.bootstrap_mean(sample, np.random.default_rng(4), 3),
    }


def _scoring_setup(tmp_path: Path, *, runner_layout: bool = False) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.pt"
    make_checkpoint(checkpoint, runner_layout=runner_layout)
    tok = FakeTokenizer(VOCAB)
    lm = FakeLM(len(VOCAB), D_HIDDEN, seed=0)
    w_u = lm.lm_head.weight.detach().float()
    w_dec_unit, w_enc, b_enc, sae_config, ckpt_row_mean = load_sae(checkpoint)
    k = rw.checkpoint_k(checkpoint, sae_config)
    row_mean = rw.scoring_row_mean(w_u, tok, "live", ckpt_row_mean)
    info, audit = rw.prepare_target_codes(["bank", "seal", "multi"], tok, w_u, row_mean, w_enc, b_enc, w_dec_unit, k)
    return SimpleNamespace(
        checkpoint=checkpoint,
        tok=tok,
        lm=lm,
        w_u=w_u,
        row_mean=row_mean,
        w_dec_unit=w_dec_unit,
        w_enc=w_enc,
        b_enc=b_enc,
        info=info,
        audit=audit,
        k=k,
    )


def _target_codes_actual(setup: SimpleNamespace) -> dict:
    return {
        "k": setup.k,
        "audit": setup.audit,
        "feature_ids": {t: setup.info[t]["feature_ids"].tolist() for t in ("bank", "seal")},
        "beta": {t: setup.info[t]["beta"].tolist() for t in ("bank", "seal")},
    }


def _items(prompts: list[str]) -> list[dict]:
    return [
        {
            "item_id": f"i{j}",
            "kind": "context",
            "dataset": "coarsewsd20",
            "split": "train" if j % 2 == 0 else "test",
            "target": target,
            "sense": str(j % 2),
            "sense_name": str(j % 2),
            "prefix_words": 1,
            "prompt": prompt,
        }
        for j, (prompt, target) in enumerate(zip(prompts, TARGETS))
    ]


def _score(setup: SimpleNamespace, prompts: list[str], max_length: int = 16, batch_size: int = 2) -> dict:
    return rw.score_items(
        _items(prompts),
        setup.lm,
        setup.tok,
        setup.lm.lm_head,
        setup.row_mean,
        setup.w_dec_unit,
        setup.info,
        batch_size,
        max_length,
        "cpu",
    )


def _scoring_actual(bundle: dict) -> tuple[dict, dict]:
    pinned = {
        "metadata": bundle["metadata"],
        "vectors": {
            key: bundle[key].tolist()
            for key in ("feature_ids", "exact_logit", "reconstructed_logit", "target_logprob", "target_rank")
        },
        "truncation": bundle["truncation"],
        "summary": rw.summarize_scoring(bundle),
    }
    half = {key: bundle[key].float().tolist() for key in HALF_KEYS}
    return pinned, half


# --------------------------------------------------------------------------- #
# verbatim pre-0.2.1 references, run in-process
# --------------------------------------------------------------------------- #


def _legacy_target_codes(token_ids, w_u, row_mean, encoder_w, encoder_b, w_dec_unit, k) -> dict:
    """Verbatim pre-0.2.1 arithmetic of prepare_target_codes: 1-D centring and
    normalisation, ReLU on the product against the contiguous encoder transpose,
    a single torch.topk."""
    w_enc = encoder_w.T.contiguous()
    out = {}
    for token_id in token_ids:
        row = w_u[token_id] - row_mean
        row_norm = row.norm().clamp_min(1e-8)
        acts = torch.relu((row / row_norm) @ w_enc + encoder_b)
        values, indices = torch.topk(acts, k=min(k, acts.numel()))
        beta = row_norm * values
        recon = beta @ w_dec_unit[indices]
        out[token_id] = {
            "feature_ids": indices,
            "beta": beta,
            "row_relative_error": float((row - recon).norm() / row_norm),
            "row_cosine": float(F.cosine_similarity(row[None], recon[None]).item()),
        }
    return out


def _legacy_shuffled_srp(projection, beta, metadata, seed, *, modular: bool) -> np.ndarray:
    """Verbatim pre-0.2.1 null: the run script seeded ``seed + int(sha1[:8], 16)``,
    the classifier framing ``(int(sha1[:8], 16) + seed) % 2**32``."""
    shuffled = np.empty_like(projection)
    for target in sorted({row["target"] for row in metadata}):
        idx = [i for i, row in enumerate(metadata) if row["target"] == target]
        digest = int(hashlib.sha1(target.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng((digest + seed) % (2**32) if modular else seed + digest)
        permutation = rng.permutation(beta.shape[1])
        shuffled[idx] = projection[idx] * beta[idx][:, permutation]
    return shuffled


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
    actual = _check("sense_groups", _sense_groups_actual(tmp_path))
    for g in (1, 2, 4):
        summary = actual[f"g{g}"]
        assert summary["group_size"] == g and summary["basis"] == "srp" and summary["standardized"] is True
        assert summary["n_words"] == 2 and set(summary["per_word"]) == set(DESIGN)
        assert summary["word_mean_balanced"] > 0.9 > 0.8 > summary["word_mean_null_balanced"]
        assert summary["word_mean_majority_balanced"] == pytest.approx((1 / 2 + 1 / 3) / 2)
        assert summary["words_beating_majority_balanced"] == 2 and summary["words_beating_null_p95_balanced"] == 2
        assert summary["word_mean_full_account_balanced"] > 0.95 and summary["word_mean_hidden_balanced"] > 0.95
        assert summary["gate_fraction_overall"] == 1.0
        for word, senses in DESIGN.items():
            pw = summary["per_word"][word]
            assert pw["n_senses"] == len(senses) and all(len(pw["anchors"][s]) == g for s in senses)
            assert pw["chance_balanced_accuracy"] == pytest.approx(1 / len(senses))
    # The planted positions are the g=1 anchors.
    assert actual["g1"]["per_word"]["bank"]["anchors"] == {"0": [3], "1": [7]}
    assert actual["g1"]["per_word"]["seal"]["anchors"] == {"0": [1], "1": [9], "2": [13]}
    assert actual["raw_top1"]["selector"] == "top1_freq" and actual["raw_top1"]["standardized"] is False
    written = json.loads((tmp_path / "out" / "synthetic__srp_g1.json").read_text())
    assert "command" in written["provenance"] and written["provenance"]["args"]["seed"] == 0
    # Single-size runs write --out verbatim and match the sweep's file for that size.
    single = tmp_path / "single.json"
    argv = ["--bundle", str(tmp_path / "representations.pt"), "--out", str(single)]
    assert sg.main(argv + ["--group-size", "1", "--n-boot", "20", "--n-null", "10"]) == 0
    _assert_close(json.loads(single.read_text())["per_word"], written["per_word"], "single_vs_sweep")


def test_geometry_controls_route_centering_through_centering_mean(tmp_path):
    bundle = make_bundle()
    bundle_path = tmp_path / "representations.pt"
    torch.save(bundle, bundle_path)
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(128, D_HIDDEN, generator=gen)
    mask_all = torch.ones(128, dtype=torch.bool)
    # The controls' row mean is data.centering_mean: live is the full-vocabulary mean ...
    live = sg.geometry_row_mean(W, "live", token_mask=None, checkpoint=None, model_id="x", revision=None)
    assert torch.equal(live, centering_mean(W, mode="live"))
    assert torch.allclose(live, W.mean(dim=0), rtol=1e-6, atol=1e-7)
    # ... and trained with an all-True token_mask (no checkpoint) is the same vector.
    trained = sg.geometry_row_mean(W, "trained", token_mask=mask_all, checkpoint=None, model_id="x", revision=None)
    assert torch.allclose(trained, live, rtol=1e-6, atol=1e-7)
    partial = mask_all.clone()
    partial[:8] = False
    masked = sg.geometry_row_mean(W, "trained", token_mask=partial, checkpoint=None, model_id="x", revision=None)
    assert not torch.allclose(masked, live, rtol=1e-6, atol=1e-7)
    # geometry_contributions takes that mean rather than recomputing it.
    tids = [100, 101]
    hidden = {t: bundle["hidden"][[i for i, r in enumerate(bundle["metadata"]) if r["token_id"] == t]] for t in tids}
    geo = sg.geometry_contributions("knn128", W, live, tids, hidden, 4, 4, 1e-3, 0)
    assert set(geo) == set(tids) and geo[100]["contribution"].shape == (len(hidden[100]), 4)
    np.testing.assert_allclose(geo[100]["exact"], (hidden[100] @ W[100]).numpy(), rtol=REL, atol=ABS)
    # End to end through --w-u: trained (payload token_mask all True) == live, provenance aside.
    outputs = {}
    for mode in ("live", "trained"):
        torch.save({"W_U_orig": W, "token_mask": mask_all}, tmp_path / f"wu_{mode}.pt")
        out = tmp_path / f"knn_{mode}.json"
        argv = ["--bundle", str(bundle_path), "--out", str(out), "--basis", "knn128", "--neighbor-k", "4"]
        argv += ["--w-u", str(tmp_path / f"wu_{mode}.pt"), "--centering", mode, "--n-boot", "5", "--n-null", "3"]
        assert sg.main(argv) == 0
        outputs[mode] = json.loads(out.read_text())
        assert outputs[mode]["basis"] == "knn128" and outputs[mode]["n_words"] == 2
    _assert_close(_strip(outputs["trained"]), _strip(outputs["live"]), "trained_vs_live")
    assert outputs["trained"]["provenance"]["args"]["centering"] == "trained"


def test_load_wu_reads_the_resolved_local_snapshot(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import huggingface_hub

    snap = tmp_path / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    head = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    save_file({"model.embed_tokens.weight": torch.zeros(4, 3)}, str(snap / "model-00001-of-00002.safetensors"))
    save_file({"lm_head.weight": head.to(torch.bfloat16)}, str(snap / "model-00002-of-00002.safetensors"))
    seen = {}

    def fake_snapshot_download(model_id, **kwargs):
        seen.update(model_id=model_id, **kwargs)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    W = sg.load_wu("org/model", revision="abc123")
    assert torch.equal(W, head) and W.dtype == torch.float32
    assert seen == {"model_id": "org/model", "revision": "abc123", "local_files_only": True}
    # Tied-embedding fallback when no lm_head.weight is stored.
    (snap / "model-00002-of-00002.safetensors").unlink()
    assert torch.equal(sg.load_wu("org/model"), torch.zeros(4, 3)) and seen["revision"] is None


# --------------------------------------------------------------------------- #
# classifier framing
# --------------------------------------------------------------------------- #


def test_classifier_framing_reproduces_pinned_outputs(tmp_path):
    assert cf.self_test() == 0
    actual = _check("classifier_framing", _classifier_actual(tmp_path))
    assert actual["gate_all"] is True and actual["gate"]["pass_fraction"] == 1.0
    metrics = actual["metrics"]
    assert set(metrics["primary"]["methods"]) == set(cf.METHODS)
    assert set(metrics["primary"]["comparisons"]) == {f"srp_minus_{m}" for m in cf.METHODS[1:]}
    assert set(metrics["primary"]["null_seed_sensitivity"]) == {"shuffled_srp", "random_srp_features"}
    srp = metrics["primary"]["methods"]["srp"]
    assert srp["n_words"] == 2 and srp["accuracy"] > 0.95 and srp["balanced_accuracy"] > 0.95
    # Projections and contributions differ only by a per-word rescaling of the coordinates,
    # which top-|value| selection is not invariant to, so both are reported separately.
    assert "projection_only" in metrics["results"]["weighted"]["4"]
    # The random source draws k positions per word.
    assert all(len(positions) == 3 for positions in actual["random_positions"].values())
    assert actual["cli"]["analysis"]["primary_k"] == 4 and actual["cli"]["analysis"]["primary_encoding"] == "signed"
    written = json.loads((tmp_path / "cf" / "out.json").read_text())
    assert written["provenance"]["args"]["primary_k"] == 4 and "command" in written["provenance"]


def test_shuffled_srp_null_is_the_pre_merge_construction():
    actual = _check("shuffle_null", _shuffle_actual())
    bundle = make_bundle()
    projection = bundle["projection"].numpy()
    beta = bundle["beta"].numpy()
    for seed in (0, 3):
        new = wsd.shuffled_srp(projection, beta, bundle["metadata"], seed)
        # Both pre-merge constructions (run-script and classifier-framing seed forms), in-process.
        for modular in (True, False):
            legacy = _legacy_shuffled_srp(projection, beta, bundle["metadata"], seed, modular=modular)
            np.testing.assert_allclose(new, legacy, rtol=REL, atol=ABS)
        np.testing.assert_allclose(cf.source_matrix(bundle, "shuffled_srp", seed), new, rtol=REL, atol=ABS)
        assert not np.allclose(new, bundle["contribution"].numpy())
    shuffled0 = wsd.shuffled_srp(projection, beta, bundle["metadata"], 0)
    np.testing.assert_allclose(
        rw.method_matrices(bundle, 0)["shuffled_srp"], wsd.l2_normalize(shuffled0), rtol=REL, atol=ABS
    )
    # One seed form for both scripts: the modular sha1 prefix, which coincides with
    # the run script's former ``seed + int(sha1[:8], 16)`` at seed 0.
    for target in ("bank", "seal", "apple"):
        prefix = int(hashlib.sha1(target.encode()).hexdigest()[:8], 16)
        assert wsd.stable_seed(target, 0) == prefix
        assert wsd.stable_seed(target, 3) == (prefix + 3) % 2**32
    assert actual["stable_seed"]["bank_0"] == wsd.stable_seed("bank", 0)


# --------------------------------------------------------------------------- #
# run script: analysis, statistics, scoring path
# --------------------------------------------------------------------------- #


def test_run_analysis_reproduces_pinned_outputs(tmp_path):
    actual = _check("run_analysis", _run_analysis_actual(tmp_path))
    assert set(actual["metrics"]["methods"]) == {"srp", "support_projection", "hidden", "shuffled_srp", "token_only"}
    assert set(actual["metrics"]["comparisons"]) == {f"srp_minus_{b}" for b in rw.COMPARISON_BASELINES}
    assert actual["metrics"]["methods"]["srp"]["aggregate"]["accuracy"] == 1.0
    assert len(actual["predictions"]) == 40 and all(row["correct"] == 1 for row in actual["predictions"])
    assert list(rw.METRIC_NAMES) == ["accuracy", "balanced_accuracy", "macro_f1", "pairwise_auc", "ari", "nmi"]


def test_statistics_helpers_reproduce_pinned_values():
    actual = _check("statistics", _statistics_actual())
    assert actual["auc"]["one_class"] is None  # single-class labels: no AUC
    assert actual["cluster_bootstrap"]["n"] == 50
    assert actual["spearman"]["constant"] is None and actual["spearman"]["short"] is None
    assert actual["spearman"]["ok"] == pytest.approx(0.8)
    # The documented difference from utils.spearman: a range under 1e-8 is degenerate here, correlated there.
    tiny = [0.0, 1e-9, 2e-9]
    assert actual["spearman"]["tiny_range"] is None and math.isnan(wsd.safe_spearman(tiny, [1, 2, 3]))
    assert spearman(tiny, [1, 2, 3]) == pytest.approx(1.0)
    assert all(math.isnan(v) for v in wsd.percentile_ci([]))
    assert all(math.isnan(v) for v in wsd.percentile_ci(np.array([])))
    # bootstrap_mean is the shared form of the inline per-word bootstrap it replaced.
    sample = np.random.default_rng(1).normal(size=9)
    rng = np.random.default_rng(4)
    expected = [float(rng.choice(sample, size=len(sample), replace=True).mean()) for _ in range(3)]
    np.testing.assert_allclose(wsd.bootstrap_mean(sample, np.random.default_rng(4), 3), expected, rtol=REL, atol=ABS)


def test_prepare_target_codes_matches_the_pre_refactor_path(tmp_path):
    setup = _scoring_setup(tmp_path)
    actual = _check("target_codes", _target_codes_actual(setup))
    assert actual["k"] == 6 and actual["audit"]["n_targets_single_token"] == 2
    assert actual["audit"]["skipped_targets"] == {"multi": "multi_token"}
    # The verbatim pre-0.2.1 computation, in-process: same feature order, same codes.
    legacy = _legacy_target_codes(
        [setup.info[t]["token_id"] for t in ("bank", "seal")],
        setup.w_u,
        setup.row_mean,
        setup.w_enc,
        setup.b_enc,
        setup.w_dec_unit,
        setup.k,
    )
    for target in ("bank", "seal"):
        info = setup.info[target]
        ref = legacy[info["token_id"]]
        assert info["feature_ids"].tolist() == ref["feature_ids"].tolist()
        np.testing.assert_allclose(info["beta"].numpy(), ref["beta"].numpy(), rtol=REL, atol=ABS)
        assert math.isclose(info["row_relative_error"], ref["row_relative_error"], rel_tol=REL, abs_tol=ABS)
        assert math.isclose(info["row_cosine"], ref["row_cosine"], rel_tol=REL, abs_tol=ABS)
        assert info["n_positive_codes"] == 6
    # Runner-layout checkpoints (top-level ``factorizer``, no ``config``) yield the same k and codes.
    runner = _scoring_setup(tmp_path / "runner", runner_layout=True)
    assert runner.k == 6
    _assert_close(to_jsonable(runner.audit), to_jsonable(setup.audit), "runner_vs_config_audit")
    torch.testing.assert_close(runner.info["bank"]["beta"], setup.info["bank"]["beta"])
    assert rw.checkpoint_k(setup.checkpoint, {}) == 6


def test_score_items_matches_the_pre_refactor_path(tmp_path):
    setup = _scoring_setup(tmp_path)
    rw.configure_tokenizer(setup.tok)
    assert (setup.tok.padding_side, setup.tok.truncation_side) == ("left", "left")
    bundle = _score(setup, PROMPTS)
    assert len(bundle["metadata"]) == 4  # the multi-token target is dropped
    pinned, half = _scoring_actual(bundle)
    _check("scoring", pinned)
    for key, expected in _pinned("scoring_half").items():
        np.testing.assert_allclose(half[key], np.asarray(expected), rtol=HALF_REL, atol=HALF_ABS, err_msg=key)
    assert bundle["truncation"] == {
        "max_length": 16,
        "truncation_side": "left",
        "n_truncated_prompts": 0,
        "max_prompt_tokens": max(len(p.split()) for p in PROMPTS[:4]),
    }
    summary = rw.summarize_scoring(bundle)
    assert summary.pop("truncation") == bundle["truncation"]
    bundle["run"] = {"dataset": "coarsewsd20"}
    wsd.validate_bundle(bundle)


def test_left_truncation_keeps_the_cloze_cue(tmp_path):
    setup = _scoring_setup(tmp_path)
    rw.configure_tokenizer(setup.tok)
    long_prompt = "x y x y x y x y the river bank is the missing word :"
    tail = " ".join(long_prompt.split()[-6:])
    head = " ".join(long_prompt.split()[:6])

    def scored(prompt, max_length):
        return _score(setup, [prompt], max_length=max_length, batch_size=1)

    truncated = scored(long_prompt, 6)
    assert truncated["truncation"]["n_truncated_prompts"] == 1
    assert truncated["truncation"]["max_prompt_tokens"] == len(long_prompt.split())
    assert torch.equal(truncated["hidden"], scored(tail, 64)["hidden"])  # the cue survives ...
    assert not torch.equal(truncated["hidden"], scored(head, 64)["hidden"])  # ... unlike under right truncation
    setup.tok.truncation_side = "right"
    assert torch.equal(scored(long_prompt, 6)["hidden"], scored(head, 64)["hidden"])


def test_centering_trained_equals_live_when_the_mask_keeps_all_rows():
    lm = FakeLM(len(VOCAB), D_HIDDEN, seed=0)
    w_u = lm.lm_head.weight.detach().float()
    live = rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "live", None)
    assert torch.allclose(live, w_u.mean(dim=0), rtol=1e-6, atol=1e-7)
    trained_all = rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB, special_ids=()), "trained", None)
    assert torch.allclose(trained_all, live, rtol=1e-6, atol=1e-7)
    masked = rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "trained", None)  # rows 0 and 1 are special ids
    assert torch.allclose(masked, w_u[2:].mean(dim=0), rtol=1e-6, atol=1e-7)
    assert not torch.allclose(masked, live, rtol=1e-6, atol=1e-7)
    stored = torch.full((D_HIDDEN,), 0.25)
    assert torch.equal(rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "trained", stored), stored)
    assert torch.allclose(rw.scoring_row_mean(w_u, FakeTokenizer(VOCAB), "live", stored), live)


def test_analyze_only_writes_analysis_config_and_keeps_run_config(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    torch.save(make_bundle(), out_dir / "representations.pt")
    sentinel = '{"scoring_run": true}\n'
    (out_dir / "run_config.json").write_text(sentinel)
    assert rw.main(["--analyze-only", "--out-dir", str(out_dir), "--n-boot", "5", "--seed", "0"]) == 0
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
    for name in ("load_ambistory", "analyze_ambistory", "make_story", "mask_exact_target"):
        assert not hasattr(rw, name), name
    for name in ("ambistory_rows", "analyze_ambistory", "summarize_ambistory_rows"):
        assert not hasattr(cf, name), name
    assert rw.self_test() == 0


# --------------------------------------------------------------------------- #
# fixture regeneration: uv run --no-sync python tests/test_wsd_sense_groups.py
# --------------------------------------------------------------------------- #


def regenerate_pins() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        setup = _scoring_setup(root / "scoring")
        rw.configure_tokenizer(setup.tok)
        scoring, scoring_half = _scoring_actual(_score(setup, PROMPTS))
        pins = {
            "sense_groups": _sense_groups_actual(root / "sense_groups"),
            "classifier_framing": _classifier_actual(root / "classifier_framing"),
            "shuffle_null": _shuffle_actual(),
            "run_analysis": _run_analysis_actual(root / "run_analysis"),
            "statistics": _statistics_actual(),
            "target_codes": _target_codes_actual(setup),
            "scoring": scoring,
            "scoring_half": scoring_half,
        }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(to_jsonable(pins), indent=1, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(regenerate_pins())
