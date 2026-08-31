#!/usr/bin/env python3
"""Mean and tail statistics of selected-score reconstruction error, per method and model.

Consumes completed baseline-comparison runs (the ``baseline_query_rows.csv``
written by ``scripts/run/run_readout_baseline_comparisons.py``, one directory
per model, laid out as ``<input-root>/<input-prefix><model_tag>/``) and
recomputes every per-row statistic from the raw margins rather than trusting
the runner's summary columns:

    absolute_error   = |m_exact - m_recon|
    rho              = absolute_error / (|m_exact| + 0.5)      floored relative error rho_0.5
    rho_0            = absolute_error / max(|m_exact|, 1e-6)   unfloored rho_0 (the runner's residual_direct)
    sign_match       = (m_exact > 0) == (m_recon > 0)          the runner's convention
    sign_match_strict= sign_match with an exact-zero m_exact counted as disagreement
    accepted         = sign_match_strict and rho < 0.5
    margin_bin       = |m_exact| in {<0.5, 0.5-1, 1-2, >=2}    the runner's four bins
    margin_bin_fine  = |m_exact| in {<0.5, 0.5-1, 1-2, 2-5, >=5}

``margin_bin`` and ``sign_match`` are recomputed with definitions identical to
the runner's input columns of the same name; ``rho_0`` equals ``residual_direct``.

Outputs (CSV/JSON only):

    summary_by_method.csv          primary-model rows, per method: mean/median/p90/p95/p99/max of
                                   rho and absolute_error, sign agreement, rho<0.5, accepted rate,
                                   cluster-bootstrap 95% CIs (clusters = base_case_id)
    summary_by_model.csv           SRP rows per model, same columns and CIs
    summary_by_margin_bin.csv      SRP rows per (model, margin_bin_fine), no CIs
    summary_by_family.csv          SRP rows per (model, family), no CIs
    paired_method_differences.csv  primary model, SRP minus each baseline on the same cases
                                   (mean rho, p95 rho, sign agreement, accepted), paired cluster bootstrap
    tail_rows.csv                  primary-model SRP rows in the top 1% by rho, joined to the tokenizer audit
    summary_pooled_by_method.csv   all models pooled, per method: mean/p95/max absolute error, median
                                   rho_0, sign agreement, coverage (sign_match and rho_0<0.5)
    summary_by_model_method.csv    the same six statistics per (model, method)
    summary_pooled_by_method_margin_bin.csv
                                   the same statistics plus bin share, per (method, margin_bin), pooled
    summary.json                   provenance, input hashes, row counts, tail threshold

Paper artifacts. ``tab:app-error-tails`` is ``summary_pooled_by_method.csv``
(8,009 contrasts per method over the six softcap-free readouts) and
``tab:app-error-tail-margins`` is the ``sparse_rp`` block of
``summary_pooled_by_method_margin_bin.csv``; the per-model ranges quoted in the
same appendix come from ``summary_by_model_method.csv``. The floored-rho
summaries with bootstrap intervals are the distribution-wide companion to
``tab:app-direct-geometry-grid`` on the primary model.

Paper run (inputs are the six direct-geometry runs of the baseline script, one
directory per model tag qwen0p8b, qwen2b, qwen9b, ministral8b, r1qwen7b,
r1llama8b; ``--input-prefix`` names any common directory prefix, e.g.
``run_`` for ``run_qwen2b/``):

    uv run python scripts/eval/analyze_error_tails.py \\
        --input-root results/direct_geometry_runs \\
        --out-dir results/error_tails --n-boot 10000 --seed 20260711
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_readout_prism.research.run_io import run_provenance

MODEL_TAGS = (
    "qwen0p8b",
    "qwen2b",
    "qwen9b",
    "ministral8b",
    "r1qwen7b",
    "r1llama8b",
)
MARGIN_BINS_FINE = [-np.inf, 0.5, 1.0, 2.0, 5.0, np.inf]
MARGIN_LABELS_FINE = ["<0.5", "0.5-1", "1-2", "2-5", ">=5"]
MARGIN_BINS = [-np.inf, 0.5, 1.0, 2.0, np.inf]
MARGIN_LABELS = ["<0.5", "0.5-1", "1-2", ">=2"]
ANCHOR = "sparse_rp"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_row_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute the per-row error statistics from ``exact_margin`` / ``sparse_margin``."""
    required = {
        "model_tag",
        "method",
        "base_case_id",
        "case_id",
        "family",
        "exact_margin",
        "sparse_margin",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"raw rows missing columns: {sorted(missing)}")
    exact = pd.to_numeric(rows["exact_margin"], errors="raise").to_numpy(float)
    recon = pd.to_numeric(rows["sparse_margin"], errors="raise").to_numpy(float)
    err = np.abs(exact - recon)
    rows["absolute_error"] = err
    rows["rho"] = err / (np.abs(exact) + 0.5)
    rows["rho_0"] = err / np.maximum(np.abs(exact), 1e-6)
    rows["exact_zero"] = np.isclose(exact, 0.0, atol=1e-12)
    rows["near_zero"] = np.abs(exact) < 0.5
    rows["sign_match"] = (exact > 0) == (recon > 0)
    rows["sign_match_strict"] = (~rows["exact_zero"]) & (np.signbit(exact) == np.signbit(recon))
    rows["accepted"] = rows["sign_match_strict"] & (rows["rho"] < 0.5)
    rows["covered"] = rows["sign_match"] & (rows["rho_0"] < 0.5)
    rows["margin_bin"] = pd.cut(np.abs(exact), bins=MARGIN_BINS, labels=MARGIN_LABELS, right=False).astype(str)
    rows["margin_bin_fine"] = pd.cut(
        np.abs(exact), bins=MARGIN_BINS_FINE, labels=MARGIN_LABELS_FINE, right=False
    ).astype(str)
    return rows


def load_rows(input_root: Path, prefix: str, model_tags: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    inputs: list[dict] = []
    for tag in model_tags:
        p = input_root / f"{prefix}{tag}" / "baseline_query_rows.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["model_tag"] = tag
        frames.append(df)
        inputs.append({"path": str(p.resolve()), "sha256": sha256(p), "rows": len(df)})
    if not frames:
        raise FileNotFoundError(f"no {prefix}*/baseline_query_rows.csv under {input_root}")
    rows = pd.concat(frames, ignore_index=True)
    return add_row_metrics(rows), inputs


def point_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    rho = df["rho"].to_numpy(float)
    ae = df["absolute_error"].to_numpy(float)
    out: dict[str, float | int] = {"n_rows": int(len(df))}
    for prefix, arr in (("rho", rho), ("absolute_error", ae)):
        out[f"{prefix}_mean"] = float(np.mean(arr))
        out[f"{prefix}_median"] = float(np.median(arr))
        for q in (90, 95, 99):
            out[f"{prefix}_p{q}"] = float(np.percentile(arr, q))
        out[f"{prefix}_max"] = float(np.max(arr))
    out["sign_agreement"] = float(df["sign_match_strict"].mean())
    out["rho_lt_0p5"] = float((df["rho"] < 0.5).mean())
    out["accepted_rate"] = float(df["accepted"].mean())
    out["exact_zero_n"] = int(df["exact_zero"].sum())
    out["near_zero_n"] = int(df["near_zero"].sum())
    return out


def bootstrap_ci(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
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
        stats[b] = (
            rb.mean(),
            np.median(rb),
            np.percentile(rb, 95),
            sign[idx].mean(),
            accepted[idx].mean(),
        )
    names = ("rho_mean", "rho_median", "rho_p95", "sign_agreement", "accepted_rate")
    out: dict[str, float] = {}
    for j, name in enumerate(names):
        out[f"{name}_ci_lo"] = float(np.percentile(stats[:, j], 2.5))
        out[f"{name}_ci_hi"] = float(np.percentile(stats[:, j], 97.5))
    return out


def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    bootstrap: bool,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict] = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, part in df.groupby(grouper, sort=True, observed=True):
        key = (key,) if not isinstance(key, tuple) else key
        rec = dict(zip(group_cols, key))
        rec.update(point_metrics(part))
        if bootstrap:
            group_seed = seed + int(hashlib.sha1(str(key).encode()).hexdigest()[:7], 16)
            rec.update(bootstrap_ci(part.reset_index(drop=True), n_boot, group_seed))
        records.append(rec)
    return pd.DataFrame(records)


def paired_differences(qdf: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    keys = ["bank", "base_case_id", "case_id", "query"]
    cols = keys + ["rho", "sign_match_strict", "accepted"]
    a = qdf[qdf.method == ANCHOR][cols].copy()
    results: list[dict] = []
    for method in sorted(set(qdf.method) - {ANCHOR}):
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


def attach_token_audit(tail: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
    if not audit_path.exists() or tail.empty:
        return tail
    audit = pd.read_csv(audit_path)
    if "case_id" not in audit.columns:
        return tail
    keep = [c for c in ("token_id", "reason", "side", "target") if c in audit.columns]
    grouped = audit.groupby("case_id", sort=False)[keep].agg(lambda s: ";".join(dict.fromkeys(map(str, s.dropna()))))
    grouped = grouped.add_prefix("token_audit_").reset_index()
    return tail.merge(grouped, on="case_id", how="left")


# --------------------------------------------------------------------------- #
# Paper-table summaries: pooled over models, unfloored rho_0, runner conventions
# --------------------------------------------------------------------------- #


def coverage_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """Mean/p95/max absolute error, median rho_0, sign agreement and coverage,
    the six statistics of the error-tail tables (unfloored rho_0, the runner's
    sign convention)."""
    ae = df["absolute_error"].to_numpy(float)
    return {
        "n_rows": int(len(df)),
        "n_models": int(df["model_tag"].nunique()),
        "absolute_error_mean": float(np.mean(ae)),
        "absolute_error_p95": float(np.percentile(ae, 95)),
        "absolute_error_max": float(np.max(ae)),
        "rho_0_median": float(df["rho_0"].median()),
        "sign_agreement": float(df["sign_match"].mean()),
        "coverage": float(df["covered"].mean()),
    }


def summarize_coverage(df: pd.DataFrame, group_cols: list[str], *, share_within: str | None = None) -> pd.DataFrame:
    """``coverage_metrics`` per group. ``share_within`` names a column whose
    groups are the denominator of a ``share`` column (bin share within a method)."""
    records: list[dict] = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    totals = df.groupby(share_within, sort=True).size() if share_within else None
    for key, part in df.groupby(grouper, sort=True, observed=True):
        key = (key,) if not isinstance(key, tuple) else key
        rec = dict(zip(group_cols, key))
        if totals is not None:
            rec["share"] = float(len(part) / totals[rec[share_within]])
        rec.update(coverage_metrics(part))
        records.append(rec)
    out = pd.DataFrame(records)
    if "margin_bin" in group_cols and not out.empty:
        order = {lbl: i for i, lbl in enumerate(MARGIN_LABELS)}
        out = out.sort_values(
            [c for c in group_cols if c != "margin_bin"] + ["margin_bin"],
            key=lambda s: s.map(order) if s.name == "margin_bin" else s,
        )
        out = out.reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input-root", type=Path, required=True, help="directory holding <prefix><tag>/baseline_query_rows.csv"
    )
    ap.add_argument("--input-prefix", type=str, default="", help="common prefix of the per-model run directories")
    ap.add_argument(
        "--model-tags",
        type=str,
        default=",".join(MODEL_TAGS),
        help="comma-separated model tags; missing directories are skipped",
    )
    ap.add_argument(
        "--primary-model-tag",
        type=str,
        default="qwen2b",
        help="model for the per-method / paired / tail analyses (the paper's Qwen3.5-2B)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_tags = [t.strip() for t in args.model_tags.split(",") if t.strip()]
    rows, inputs = load_rows(args.input_root, args.input_prefix, model_tags)
    qdf = rows[rows.model_tag == args.primary_model_tag].copy()
    if qdf.empty:
        raise ValueError(f"rows for the primary model tag {args.primary_model_tag!r} are required")

    by_method = summarize(qdf, ["method"], bootstrap=True, n_boot=args.n_boot, seed=args.seed)
    srp_rows = rows[rows.method == ANCHOR]
    by_model = summarize(srp_rows, ["model_tag"], bootstrap=True, n_boot=args.n_boot, seed=args.seed)
    by_margin = summarize(srp_rows, ["model_tag", "margin_bin_fine"], bootstrap=False, n_boot=0, seed=args.seed)
    by_family = summarize(srp_rows, ["model_tag", "family"], bootstrap=False, n_boot=0, seed=args.seed)
    paired = paired_differences(qdf, args.n_boot, args.seed)

    srp_q = qdf[qdf.method == ANCHOR].copy()
    threshold = float(srp_q.rho.quantile(0.99))
    tail = srp_q[srp_q.rho >= threshold].copy().sort_values("rho", ascending=False)
    tail["tail_definition"] = f"{args.primary_model_tag}_srp_top1pct_rho_ge_{threshold:.9g}"
    tail = attach_token_audit(
        tail, args.input_root / f"{args.input_prefix}{args.primary_model_tag}" / "tokenizer_audit.csv"
    )

    pooled_by_method = summarize_coverage(rows, ["method"])
    by_model_method = summarize_coverage(rows, ["model_tag", "method"])
    pooled_by_method_bin = summarize_coverage(rows, ["method", "margin_bin"], share_within="method")

    by_method.to_csv(args.out_dir / "summary_by_method.csv", index=False)
    by_model.to_csv(args.out_dir / "summary_by_model.csv", index=False)
    by_margin.to_csv(args.out_dir / "summary_by_margin_bin.csv", index=False)
    by_family.to_csv(args.out_dir / "summary_by_family.csv", index=False)
    paired.to_csv(args.out_dir / "paired_method_differences.csv", index=False)
    tail.to_csv(args.out_dir / "tail_rows.csv", index=False)
    pooled_by_method.to_csv(args.out_dir / "summary_pooled_by_method.csv", index=False)
    by_model_method.to_csv(args.out_dir / "summary_by_model_method.csv", index=False)
    pooled_by_method_bin.to_csv(args.out_dir / "summary_pooled_by_method_margin_bin.csv", index=False)

    manifest = {
        **run_provenance(args),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "rho_definition": "abs(exact-reconstructed)/(abs(exact)+0.5)",
        "rho_0_definition": "abs(exact-reconstructed)/max(abs(exact),1e-6)",
        "coverage_definition": "sign_match and rho_0 < 0.5",
        "inputs": inputs,
        "model_tags": sorted(rows.model_tag.unique()),
        "rows_total": len(rows),
        "primary_model_tag": args.primary_model_tag,
        "primary_model_rows": len(qdf),
        "tail_threshold": threshold,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
