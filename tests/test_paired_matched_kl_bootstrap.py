"""Smoke test for scripts/eval/paired_matched_kl_bootstrap.py on a synthetic
candidate_constrained_rows.csv: the matched scales are the log-nearest median
KLs, the paired mean difference equals the direct per-candidate mean, both
clusterings (held-out term and prompt) are reported, the run is seeded, and the
output carries a ``provenance`` block next to the ``comparisons`` records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import load_script

boot = load_script("scripts/eval/paired_matched_kl_bootstrap.py")

TERMS = ["damn", "crap", "bullshit"]
PROMPTS = ["Oh", "Holy", "This is", "You are such a", "That really", "I am so", "You really"]
# Per-method KL per scale: SRP reaches KL 0.05 at scale 2, the baseline at scale 4.
KL = {
    "feature_suppression": {1.0: 0.02, 2.0: 0.05, 4.0: 0.2},
    "mean_row_direction": {1.0: 0.004, 2.0: 0.01, 4.0: 0.05, 8.0: 0.2},
}


def _write_rows(path: Path) -> None:
    fields = [
        "id",
        "prompt",
        "bad",
        "good",
        "split",
        "method",
        "scale",
        "bad_prob_reduction",
        "flip",
        "full_vocab_kl_bits",
    ]
    rows = []
    for ti, term in enumerate(TERMS):
        for pi, prompt in enumerate(PROMPTS):
            cid = f"c{ti}_{pi}"
            for method, kls in KL.items():
                for scale, kl in kls.items():
                    # Deterministic outcome: SRP suppresses 0.1 more than the baseline.
                    base = 0.05 * (ti + 1) + 0.01 * pi
                    dp = base + (0.1 if method == "feature_suppression" else 0.0)
                    rows.append(
                        {
                            "id": cid,
                            "prompt": prompt,
                            "bad": term,
                            "good": "ref",
                            "split": "heldout",
                            "method": method,
                            "scale": scale,
                            "bad_prob_reduction": dp,
                            "flip": int(method == "feature_suppression"),
                            "full_vocab_kl_bits": kl,
                        }
                    )
    # A discovery-split row that must be ignored.
    rows.append({**rows[0], "id": "disc", "split": "discovery", "bad_prob_reduction": 99.0})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_paired_bootstrap_on_synthetic_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidate_constrained_rows.csv"
    _write_rows(csv_path)
    out = tmp_path / "paired.json"
    rc = boot.main(
        ["--input", f"toy={csv_path}", "--out", str(out), "--n-boot", "50", "--seed", "0", "--target-kl", "0.05", "0.2"]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert set(payload) == {"comparisons", "provenance"}
    prov = payload["provenance"]
    assert {"command", "args", "git_commit", "package_version", "torch_version", "timestamp_unix"} <= set(prov)
    assert prov["args"]["n_boot"] == 50 and prov["args"]["seed"] == 0
    results = payload["comparisons"]
    # Only mean_row_direction is present -> 2 targets x 1 baseline x 2 outcomes.
    assert len(results) == 4
    assert {r["model"] for r in results} == {"toy"}
    assert list(results[0]) == [
        "model",
        "target_kl",
        "baseline",
        "outcome",
        "n",
        "srp_scale",
        "srp_kl",
        "base_scale",
        "base_kl",
        "mean_diff",
        "ci_term",
        "ci_prompt",
    ]
    by_key = {(r["target_kl"], r["outcome"]): r for r in results}

    r = by_key[(0.05, "dp")]
    assert r["baseline"] == "mean_row_direction"
    assert r["srp_scale"] == 2.0 and r["base_scale"] == 4.0
    assert r["srp_kl"] == 0.05 and r["base_kl"] == 0.05
    assert r["n"] == len(TERMS) * len(PROMPTS)
    assert abs(r["mean_diff"] - 0.1) < 1e-9
    # Constant per-candidate difference -> both cluster CIs collapse onto it.
    assert all(abs(v - 0.1) < 1e-9 for v in r["ci_term"])
    assert all(abs(v - 0.1) < 1e-9 for v in r["ci_prompt"])
    assert by_key[(0.05, "flip")]["mean_diff"] == 1.0
    assert by_key[(0.2, "dp")]["srp_scale"] == 4.0 and by_key[(0.2, "dp")]["base_scale"] == 8.0

    # Seeded: a second run reproduces the comparison records exactly (provenance carries a timestamp).
    out2 = tmp_path / "paired2.json"
    boot.main(["--input", f"toy={csv_path}", "--out", str(out2), "--n-boot", "50", "--target-kl", "0.05", "0.2"])
    assert json.loads(out2.read_text())["comparisons"] == results


def test_scale_for_kl_respects_two_x_window() -> None:
    rows = [
        {"method": "m", "scale": "1", "full_vocab_kl_bits": "0.5"},
        {"method": "m", "scale": "2", "full_vocab_kl_bits": "1.0"},
    ]
    assert boot.scale_for_kl(rows, "m", 0.02) == (None, None)
    scale, med = boot.scale_for_kl(rows, "m", 0.6)
    assert scale == 1.0 and med == 0.5


def test_cluster_ci_seeded_and_bracketing() -> None:
    import numpy as np

    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    lo, hi = boot.cluster_ci(d, clusters, n_boot=200, seed=3)
    assert boot.cluster_ci(d, clusters, n_boot=200, seed=3) == (lo, hi)
    assert 0.0 <= lo <= d.mean() <= hi <= 5.0
