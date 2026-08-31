"""CPU-only tests for the error-tail and nearest-row analyses:

* scripts/eval/analyze_error_tails.py            (tab:app-error-tails, tab:app-error-tail-margins)
* scripts/analyze/nearest_rows_baseline.py       (tab:app-nearest-rows)

Synthetic inputs only: hand-built baseline_query_rows.csv files
and a random W_U with a fake tokenizer; no model, no network. The bootstrap
tests pin the vectorised gathers to the 0.2.0 per-cluster ``np.concatenate``
loops, element for element.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conftest import load_script

tails = load_script("scripts/eval/analyze_error_tails.py")
nrb = load_script("scripts/analyze/nearest_rows_baseline.py")


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


def _bootstrap_ci_v020(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    """The 0.2.0 loop, kept verbatim as the reference for the vectorised gather."""
    clusters = {str(k): idx.to_numpy() for k, idx in df.groupby("base_case_id", sort=True).groups.items()}
    keys = sorted(clusters)
    if len(keys) < 2 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    rho = df["rho"].to_numpy(float)
    sign = df["sign_match_strict"].to_numpy(float)
    accepted = df["accepted"].to_numpy(float)
    stats = np.empty((n_boot, 5), dtype=float)
    for b in range(n_boot):
        sampled = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([clusters[keys[i]] for i in sampled])
        rb = rho[idx]
        stats[b] = (rb.mean(), np.median(rb), np.percentile(rb, 95), sign[idx].mean(), accepted[idx].mean())
    names = ("rho_mean", "rho_median", "rho_p95", "sign_agreement", "accepted_rate")
    out: dict[str, float] = {}
    for j, name in enumerate(names):
        out[f"{name}_ci_lo"] = float(np.percentile(stats[:, j], 2.5))
        out[f"{name}_ci_hi"] = float(np.percentile(stats[:, j], 97.5))
    return out


def _paired_differences_v020(qdf: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """The 0.2.0 pandas ``iloc`` loop, kept verbatim as the reference for the vectorised gather."""
    keys = ["bank", "base_case_id", "case_id", "query"]
    cols = keys + ["rho", "sign_match_strict", "accepted"]
    a = qdf[qdf.method == tails.ANCHOR][cols].copy()
    results: list[dict] = []
    for method in sorted(set(qdf.method) - {tails.ANCHOR}):
        b = qdf[qdf.method == method][cols].copy()
        m = a.merge(b, on=keys, suffixes=("_srp", "_base"), validate="one_to_one")
        if m.empty:
            continue
        clusters = {str(k): idx.to_numpy() for k, idx in m.groupby("base_case_id", sort=True).groups.items()}
        ckeys = sorted(clusters)
        rng = np.random.default_rng(seed + int(hashlib.sha1(method.encode()).hexdigest()[:7], 16))

        def diffs(frame: pd.DataFrame) -> np.ndarray:
            return np.array(
                [
                    frame.rho_srp.mean() - frame.rho_base.mean(),
                    np.percentile(frame.rho_srp, 95) - np.percentile(frame.rho_base, 95),
                    frame.sign_match_strict_srp.mean() - frame.sign_match_strict_base.mean(),
                    frame.accepted_srp.mean() - frame.accepted_base.mean(),
                ]
            )

        point = diffs(m)
        boots = np.empty((n_boot, 4), float)
        for i in range(n_boot):
            sample = rng.integers(0, len(ckeys), size=len(ckeys))
            idx = np.concatenate([clusters[ckeys[j]] for j in sample])
            boots[i] = diffs(m.iloc[idx])
        rec: dict[str, float | str | int] = {"baseline": method, "n_pairs": len(m)}
        for j, name in enumerate(("mean_rho", "p95_rho", "sign_agreement", "accepted_rate")):
            rec[f"delta_srp_minus_baseline_{name}"] = float(point[j])
            rec[f"{name}_ci_lo"] = float(np.percentile(boots[:, j], 2.5))
            rec[f"{name}_ci_hi"] = float(np.percentile(boots[:, j], 97.5))
        results.append(rec)
    return pd.DataFrame(results)


def test_resample_index_matches_concatenate() -> None:
    rng = np.random.default_rng(7)
    # Ragged clusters in scrambled row positions, keys whose sorted order differs from insertion order.
    clusters = {"k3": np.array([5, 1, 9]), "k1": np.array([0, 7]), "k2": np.array([2, 3, 4, 8, 6])}
    order, starts, lengths = tails._flat_clusters(clusters)
    keys = sorted(clusters)
    assert list(lengths) == [2, 5, 3] and list(starts) == [0, 2, 7]
    for _ in range(50):
        sampled = rng.integers(0, len(keys), size=len(keys))
        expected = np.concatenate([clusters[keys[i]] for i in sampled])
        assert np.array_equal(order[tails._resample_index(sampled, starts, lengths)], expected)
    assert tails._resample_index(np.array([], dtype=int), starts, lengths).size == 0


def test_bootstraps_are_bit_identical_to_the_v020_loops(tmp_path: Path) -> None:
    for tag, seed in (("m1", 11), ("m2", 12)):
        _fake_run_dir(tmp_path / "in", tag, seed)
    rows, _ = tails.load_rows(tmp_path / "in", "run_", ["m1", "m2"])
    qdf = rows[rows.model_tag == "m1"].copy()
    part = qdf[qdf.method == "sparse_rp"].reset_index(drop=True)
    assert tails.bootstrap_ci(part, 200, 5) == _bootstrap_ci_v020(part, 200, 5)
    # A second group with a different seed, as summarize() draws them.
    part2 = rows[rows.method == "sparse_rp"].reset_index(drop=True)
    assert tails.bootstrap_ci(part2, 100, 20260711) == _bootstrap_ci_v020(part2, 100, 20260711)
    new = tails.paired_differences(qdf, 200, 5)
    old = _paired_differences_v020(qdf, 200, 5)
    assert list(new.columns) == list(old.columns)
    assert new.equals(old), (new, old)


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


def _planted_case():
    vocab = [" bug", " insect", " error", "bug", "Bug", " bugs", "昆虫", "z1", "z2", "z3", "z4", "z5"]
    tok = _FakeTokenizer(vocab)
    gen = torch.Generator().manual_seed(0)
    d = 64  # wide enough that random rows have small |cosine| with q, planted rows large
    W = torch.randn(len(vocab), d, generator=gen)
    # Plant structure: 'bug'/'Bug' near ' bug', '昆虫' near ' insect' (the other side of q).
    W[3] = W[0] + 0.1 * torch.randn(d, generator=gen)
    W[4] = W[0] + 0.1 * torch.randn(d, generator=gen)
    W[6] = W[1] + 0.1 * torch.randn(d, generator=gen)
    return W, tok


def test_nearest_rows_excludes_contrast_tokens_and_ranks_by_abs_cosine() -> None:
    W, tok = _planted_case()
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


def _v020_csv_bytes(contrast_results: dict) -> tuple[str, str]:
    """The 0.2.0 writers (csv.writer / csv.DictWriter over table[0].keys()), kept as the byte reference."""
    long = io.StringIO(newline="")
    w = csv.writer(long)
    w.writerow(["contrast", "variant", "rank", "token_id", "token", "cosine"])
    for name, c in contrast_results.items():
        for variant in ("top_centered", "top_raw"):
            for row in c[variant]:
                w.writerow([name, variant, row["rank"], row["token_id"], repr(row["token"]), row["cosine"]])
    table = nrb.table_rows(contrast_results)
    tab = io.StringIO(newline="")
    dw = csv.DictWriter(tab, fieldnames=list(table[0].keys()))
    dw.writeheader()
    dw.writerows(table)
    return long.getvalue(), tab.getvalue()


def test_write_tables_matches_v020_bytes_and_survives_empty(tmp_path: Path) -> None:
    W, tok = _planted_case()
    res, _report, _ids = nrb.nearest_rows(W, W.mean(dim=0), tok, [("bug", "insect"), ("bug", "error")], top_n=3)
    nrb.write_tables(tmp_path, res)
    long_ref, table_ref = _v020_csv_bytes(res)
    assert (tmp_path / "nearest_rows.csv").read_bytes().decode("utf-8") == long_ref
    assert (tmp_path / "nearest_rows_table.csv").read_bytes().decode("utf-8") == table_ref
    assert long_ref.splitlines()[0] == ",".join(nrb.LONG_FIELDS)
    assert table_ref.splitlines()[0] == ",".join(nrb.TABLE_FIELDS)
    assert len(long_ref.splitlines()) == 1 + 2 * 2 * 3
    # --top-n 0 style empty result: the 0.2.0 table writer crashed on table[0]; now both files are 0 bytes.
    empty = tmp_path / "empty"
    empty.mkdir()
    nrb.write_tables(empty, {})
    assert (empty / "nearest_rows.csv").stat().st_size == 0
    assert (empty / "nearest_rows_table.csv").stat().st_size == 0
