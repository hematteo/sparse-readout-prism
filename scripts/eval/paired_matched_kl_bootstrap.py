#!/usr/bin/env python3
"""Paired cluster bootstrap for the matched-KL edit frontier.

Backs the paired differences quoted with ``fig:lexical-matched-kl-frontier``
(Appendix J, "Matched-KL Frontier and Cross-Model Outcome"): the readout SAE
direction edit minus each label-free category-direction baseline in held-out
suppression, compared at matched distributional cost.

The marginal CIs in ``candidate_constrained_summary.csv`` treat each method's
mean independently, but every method is evaluated on the SAME held-out
candidates, so the test for "SRP beats a baseline at matched KL" is a paired
one: per-candidate differences, resampled by cluster.

Input: one ``candidate_constrained_rows.csv`` per model, as written by
``scripts/run/run_qwen_profanity_suppression_eval.py``; only ``split == heldout``
rows are used. For each target KL: pick, per method, the swept scale whose
median full-vocabulary KL (bits) is log-nearest to the target (within a factor
of 2); align rows by case id; form ``d_i = y_SRP,i - y_base,i``; cluster-bootstrap
the mean of ``d``. Clusters: held-out term (``bad``; 12 clusters,
primary/conservative) and prompt (20; secondary). Outcomes:
``bad_prob_reduction`` (primary) and ``flip`` (secondary). Comparisons with fewer
than 20 aligned candidates are skipped.

Output: a JSON list with one record per (model, target KL, baseline, outcome):
the matched scales and their median KLs, ``mean_diff``, ``ci_term`` and
``ci_prompt`` (95% percentile intervals).

Paper run (10,000 resamples, seed 0, target KL 0.02 / 0.05 / 0.10 / 0.20)::

    python scripts/eval/paired_matched_kl_bootstrap.py \\
        --input qwen2b=<qwen2b run>/candidate_constrained_rows.csv \\
        --input qwen0p8b=<qwen0p8b run>/candidate_constrained_rows.csv \\
        --input qwen9b=<qwen9b run>/candidate_constrained_rows.csv \\
        --input r1qwen7b=<r1qwen7b run>/candidate_constrained_rows.csv \\
        --out paired_matched_kl_results.json --n-boot 10000 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BASELINES = ["mean_row_direction", "pca_group_rank4", "pca_group_direction"]
SRP = "feature_suppression"
DEFAULT_TARGET_KLS = [0.02, 0.05, 0.1, 0.2]
MIN_PAIRED_N = 20


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r["split"] == "heldout"]


def scale_for_kl(rows: list[dict], method: str, target: float):
    """Swept scale whose median KL is log-nearest to target (within 2x)."""
    kls: dict[float, list[float]] = {}
    for r in rows:
        if r["method"] == method:
            kls.setdefault(float(r["scale"]), []).append(float(r["full_vocab_kl_bits"]))
    best, best_d, best_med = None, None, None
    for s, v in kls.items():
        med = float(np.median(v))
        if med <= 0:
            continue
        d = abs(np.log(med) - np.log(target))
        if best_d is None or d < best_d:
            best, best_d, best_med = s, d, med
    if best is None or best_d > np.log(2.0):
        return None, None
    return best, best_med


def paired_diff(rows, method_a, scale_a, method_b, scale_b, ykey):
    a = {r["id"]: float(r[ykey]) for r in rows if r["method"] == method_a and float(r["scale"]) == scale_a}
    b = {r["id"]: float(r[ykey]) for r in rows if r["method"] == method_b and float(r["scale"]) == scale_b}
    meta = {r["id"]: (r["bad"], r["prompt"]) for r in rows if r["method"] == method_a and float(r["scale"]) == scale_a}
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    term = np.array([meta[i][0] for i in ids])
    prompt = np.array([meta[i][1] for i in ids])
    return d, term, prompt


def cluster_ci(d: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int):
    keys = np.unique(clusters)
    idx_by = {k: np.where(clusters == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sel = np.concatenate([idx_by[keys[p]] for p in pick])
        means.append(d[sel].mean())
    means = np.array(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparisons(
    model: str,
    rows: list[dict],
    *,
    target_kls: list[float],
    n_boot: int,
    seed: int,
    verbose: bool = True,
) -> list[dict]:
    out_rows: list[dict] = []
    for target in target_kls:
        s_srp, kl_srp = scale_for_kl(rows, SRP, target)
        if s_srp is None:
            continue
        for base in BASELINES:
            s_b, kl_b = scale_for_kl(rows, base, target)
            if s_b is None:
                continue
            for ykey, tag in [("bad_prob_reduction", "dp"), ("flip", "flip")]:
                d, term, prompt = paired_diff(rows, SRP, s_srp, base, s_b, ykey)
                if len(d) < MIN_PAIRED_N:
                    continue
                lo_t, hi_t = cluster_ci(d, term, n_boot, seed)
                lo_p, hi_p = cluster_ci(d, prompt, n_boot, seed)
                sig = "*" if (lo_t > 0 or hi_t < 0) else " "
                out_rows.append(
                    dict(
                        model=model,
                        target_kl=target,
                        baseline=base,
                        outcome=tag,
                        n=len(d),
                        srp_scale=s_srp,
                        srp_kl=round(kl_srp, 4),
                        base_scale=s_b,
                        base_kl=round(kl_b, 4),
                        mean_diff=round(float(d.mean()), 4),
                        ci_term=[round(lo_t, 4), round(hi_t, 4)],
                        ci_prompt=[round(lo_p, 4), round(hi_p, 4)],
                    )
                )
                if verbose and tag == "dp":
                    print(
                        f"  KL~{target:<5} vs {base:<22} d={d.mean():+.4f} "
                        f"CI_term[{lo_t:+.4f},{hi_t:+.4f}]{sig} CI_prompt[{lo_p:+.4f},{hi_p:+.4f}] "
                        f"(SRP@{s_srp} kl={kl_srp:.3f} vs @{s_b} kl={kl_b:.3f}, n={len(d)})"
                    )
    return out_rows


def parse_input(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--input expects MODEL=path/to/candidate_constrained_rows.csv, got {spec!r}")
    model, path = spec.split("=", 1)
    model = model.strip()
    if not model:
        raise argparse.ArgumentTypeError(f"--input has an empty model label: {spec!r}")
    return model, Path(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        type=parse_input,
        action="append",
        required=True,
        metavar="MODEL=CSV",
        help="Model label and its candidate_constrained_rows.csv; repeat once per model.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output JSON (list of paired comparisons).")
    ap.add_argument("--n-boot", type=int, default=10_000, help="Cluster-bootstrap resamples (paper: 10000).")
    ap.add_argument("--seed", type=int, default=0, help="numpy default_rng seed, reused for every CI (paper: 0).")
    ap.add_argument(
        "--target-kl",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_KLS,
        help="Matched-KL targets in bits (paper: 0.02 0.05 0.1 0.2).",
    )
    args = ap.parse_args(argv)

    out_rows: list[dict] = []
    for model, path in args.input:
        rows = load_rows(path)
        print(f"\n=== {model}: {len(rows)} held-out rows ===")
        out_rows.extend(
            paired_comparisons(model, rows, target_kls=list(args.target_kl), n_boot=args.n_boot, seed=args.seed)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_rows, indent=2))
    print(f"\nwrote {args.out} ({len(out_rows)} comparisons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
