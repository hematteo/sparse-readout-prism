"""CPU-only tests for the error-tail and nearest-row analyses:

* scripts/eval/analyze_error_tails.py            (tab:app-error-tails, tab:app-error-tail-margins)
* scripts/analyze/nearest_rows_baseline.py       (tab:app-nearest-rows)

Synthetic inputs only: hand-built baseline_query_rows.csv files
and a random W_U with a fake tokenizer; no model, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


tails = _load("analyze_error_tails", "scripts/eval/analyze_error_tails.py")
nrb = _load("nearest_rows_baseline", "scripts/analyze/nearest_rows_baseline.py")


# --------------------------------------------------------------------------- #
# error tails
# --------------------------------------------------------------------------- #


def _fake_run_dir(root: Path, tag: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for method, scale in (("sparse_rp", 0.2), ("nearest_row_ridge_top128", 1.0)):
        for i in range(60):
            exact = float(rng.normal(0, 3))
            rows.append(
                {
                    "model": tag,
                    "method": method,
                    "bank": "curated_ab",
                    "family": "fam_a" if i % 2 else "fam_b",
                    "base_case_id": f"base_{i // 3}",
                    "case_id": f"case_{i}",
                    "query": "single_token",
                    "exact_margin": exact,
                    "sparse_margin": exact + float(rng.normal(0, scale)),
                }
            )
    d = root / f"run_{tag}"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "baseline_query_rows.csv", index=False)
    pd.DataFrame([{"case_id": "case_0", "token_id": 1, "reason": "ok", "side": "A", "target": " x"}]).to_csv(
        d / "tokenizer_audit.csv", index=False
    )


def test_error_tails_row_metrics_and_pooled_tables(tmp_path: Path, monkeypatch) -> None:
    for tag, seed in (("m1", 0), ("m2", 1)):
        _fake_run_dir(tmp_path / "in", tag, seed)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_error_tails.py",
            "--input-root",
            str(tmp_path / "in"),
            "--input-prefix",
            "run_",
            "--model-tags",
            "m1,m2",
            "--primary-model-tag",
            "m1",
            "--out-dir",
            str(out),
            "--n-boot",
            "25",
            "--seed",
            "3",
        ],
    )
    assert tails.main() == 0

    rows, _ = tails.load_rows(tmp_path / "in", "run_", ["m1", "m2"])
    exact = rows.exact_margin.to_numpy()
    recon = rows.sparse_margin.to_numpy()
    # Definitions: floored rho, unfloored rho_0 == the runner's residual_direct convention.
    assert np.allclose(rows.rho, np.abs(exact - recon) / (np.abs(exact) + 0.5))
    assert np.allclose(rows.rho_0, np.abs(exact - recon) / np.maximum(np.abs(exact), 1e-6))
    assert set(rows.margin_bin) <= set(tails.MARGIN_LABELS)
    assert set(rows.margin_bin_fine) <= set(tails.MARGIN_LABELS_FINE)
    assert {"absolute_error", "rho", "rho_0", "sign_match", "sign_match_strict", "accepted", "covered"} <= set(
        rows.columns
    )

    pooled = pd.read_csv(out / "summary_pooled_by_method.csv").set_index("method")
    assert pooled.loc["sparse_rp", "n_rows"] == 120 and pooled.loc["sparse_rp", "n_models"] == 2
    srp = rows[rows.method == "sparse_rp"]
    assert np.isclose(pooled.loc["sparse_rp", "absolute_error_mean"], srp.absolute_error.mean())
    assert np.isclose(pooled.loc["sparse_rp", "coverage"], (srp.sign_match & (srp.rho_0 < 0.5)).mean())
    # The tighter method has the better tail on every pooled statistic.
    assert pooled.loc["sparse_rp", "absolute_error_p95"] < pooled.loc["nearest_row_ridge_top128", "absolute_error_p95"]

    bins = pd.read_csv(out / "summary_pooled_by_method_margin_bin.csv")
    srp_bins = bins[bins.method == "sparse_rp"]
    assert np.isclose(srp_bins.share.sum(), 1.0)
    assert list(srp_bins.margin_bin) == [b for b in tails.MARGIN_LABELS if b in set(srp_bins.margin_bin)]

    by_method = pd.read_csv(out / "summary_by_method.csv")
    assert {"rho_p95", "absolute_error_max", "accepted_rate_ci_lo"} <= set(by_method.columns)
    paired = pd.read_csv(out / "paired_method_differences.csv")
    assert paired.n_pairs.iloc[0] == 60
    manifest = json.loads((out / "summary.json").read_text())
    assert manifest["rows_total"] == 240 and "command" in manifest


# --------------------------------------------------------------------------- #
# nearest rows
# --------------------------------------------------------------------------- #


class _FakeTokenizer:
    """Vocabulary of single-character tokens; ' x' is one token, 'x' is one token."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.index[text]] if text in self.index else [0, 0]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)


def test_nearest_rows_excludes_contrast_tokens_and_ranks_by_abs_cosine() -> None:
    vocab = [" bug", " insect", " error", "bug", "Bug", " bugs", "昆虫", "z1", "z2", "z3", "z4", "z5"]
    tok = _FakeTokenizer(vocab)
    gen = torch.Generator().manual_seed(0)
    d = 64  # wide enough that random rows have small |cosine| with q, planted rows large
    W = torch.randn(len(vocab), d, generator=gen)
    # Plant structure: 'bug'/'Bug' near ' bug', '昆虫' near ' insect' (the other side of q).
    W[3] = W[0] + 0.1 * torch.randn(d, generator=gen)
    W[4] = W[0] + 0.1 * torch.randn(d, generator=gen)
    W[6] = W[1] + 0.1 * torch.randn(d, generator=gen)
    row_mean = W.mean(dim=0)
    res, report, ids = nrb.nearest_rows(W, row_mean, tok, [("bug", "insect")], top_n=4)
    assert ids == {"bug": 0, "insect": 1}
    assert report["insect"]["without_leading_space"]["single_token"] is False
    top = res["bug_minus_insect"]["top_centered"]
    listed = {r["token_id"] for r in top}
    assert not listed & {0, 1}, "contrast tokens must be excluded"
    assert [r["rank"] for r in top] == [1, 2, 3, 4]
    abs_cos = [abs(r["cosine"]) for r in top]
    assert abs_cos == sorted(abs_cos, reverse=True)
    assert {3, 4, 6} <= listed  # the planted rows lead the |cosine| ranking
    ins = next(r for r in top if r["token_id"] == 6)
    assert ins["cosine"] < 0  # insect-side rows carry a negative signed cosine
    assert all(r["cosine"] > 0 for r in top if r["token_id"] in (3, 4))
    table = nrb.table_rows(res)
    assert len(table) == 4 and table[0]["contrast"] == "bug_minus_insect"
    assert all(t["centered_cosine"].startswith(("+", "-")) for t in table)
    assert any(t["centered_gloss"] == "insect" for t in table) == any(t["centered_token"] == "昆虫" for t in table)
