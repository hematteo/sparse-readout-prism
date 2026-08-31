#!/usr/bin/env python3
"""Predicted versus realized readout-side changes (causal validation of feature contributions).

Backs ``tab:causal-validation-summary`` (one row per model) and the protocol
appendix ``app:causal-validation``, whose quoted self-test numbers come from
``--self-test``.

For a selected contrast q = w_A - w_B (two rows of the unembedding matrix W_U)
at decoded state h, the decomposition assigns feature i the contribution

    c_i = beta_i(q) * (h . d_i),    beta_i(q) = rn_A z_A,i - rn_B z_B,i

where z are the TopK codes of the centered, per-row normalized rows, rn their
centered norms, and d_i the unit-norm decoder direction. The test ablates the
decoded state along d_i,

    h' = h - (h . d_i) d_i,

and measures the realized margin change with the dense LM head,

    m(h') - m(h) = -(h . d_i) (q . d_i)

(exact, because the readout is linear in h; no new forward pass, so the claim
is scoped to the readout side). The prediction uses the sparse code beta_i(q)
while the measurement uses the true projection q . d_i, so sparse-code
misattribution, decoder non-orthogonality and the row residual all break the
agreement: slope ~ 1 with high r^2 is falsifiable, not an identity.

Per contrast the top ``--top-features`` features by |c_i| are tested plus
``--random-per-case`` random features outside that set as controls. The stored
``delta_real`` is the ablated mass +(h . d_i)(q . d_i), i.e. minus the realized
margin change, so it is directly comparable to c_i.

Slope orientation: ``fit_r2_slope(x=c_pred, y=delta_real)`` fits the realized
change on the predicted contribution through the origin (y = s x), so the slope
is realized-on-predicted. Slope > 1 means the realized change exceeds the
prediction; slope < 1 means it falls short.

``summary.json`` subsets: ``gated_predicted`` (covered contrasts, rho below
``--rho-gate`` with sign agreement; the table's r^2, CI, slope and n),
``ungated_predicted`` (all contrasts) and ``gated_random_control`` (the table's
"Random" column). CIs are 95% cluster bootstraps over base cases.

Inputs: W_U is read from the live ``lm_head`` of ``--model-id``;
``--checkpoint`` is a TopK dictionary in the trainer schema (the released
dictionaries at hematteo/sparse-readout-prism, laid out as
``<model>/<operating_point>/checkpoint.pt``); ``--bank-dir`` holds the JSONL
query banks (``data/query_banks``). The ``model_native`` bank is the C4 slice
(``qwen_gemma_result1_model_native_prompts_c4.jsonl``, regenerated locally by
``scripts/data/build_query_banks.py --native-source c4``); its rows carry no A/B
targets, so it contributes no contrasts and a missing file is logged and
skipped. The ~260 contrasts per model come from ``curated_ab`` (240 targeted
rows) and ``case_candidates`` (36), minus targets that do not tokenize to a
single token.

Outputs: ``causal_rows.csv`` (one row per prediction-realization pair),
``summary.json`` and ``manifest.json`` (run provenance).

Paper run: six models, each at the fidelity operating point (32x width,
k=256; Hub path ``<model>/k256_32x/checkpoint.pt``), every other flag at its
default::

    uv run python scripts/eval/run_causal_contribution_validation.py \\
        --model-id <MODEL_ID> --checkpoint <downloaded checkpoint.pt> \\
        --bank-dir data/query_banks --out-dir results/causal_<TAG> \\
        --device cuda --dtype bfloat16 --seed 0

    MODEL_ID                                  Hub checkpoint                              TAG
    Qwen/Qwen3.5-0.8B                         qwen3.5-0.8b/k256_32x/checkpoint.pt         qwen0p8b
    Qwen/Qwen3.5-2B                           qwen3.5-2b/k256_32x/checkpoint.pt           qwen2b
    Qwen/Qwen3.5-9B                           qwen3.5-9b/k256_32x/checkpoint.pt           qwen9b
    mistralai/Ministral-3-8B-Base-2512        ministral-3-8b/k256_32x/checkpoint.pt       ministral8b
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   r1-distill-qwen-7b/k256_32x/checkpoint.pt   r1qwen7b
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B  r1-distill-llama-8b/k256_32x/checkpoint.pt  r1llama8b

Defaults the paper run relied on: ``--banks curated_ab,case_candidates,model_native
--max-native 300 --top-features 10 --random-per-case 10 --rho-gate 0.5
--n-boot 2000 --max-len 64``.

``--self-test`` runs the synthetic residual-free check of the math core on CPU
(no model, no checkpoint) and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.data import center_normalize_rows
from sparse_readout_prism.factorizers import TopKSAE, load_factorizer
from sparse_readout_prism.research.qwen_readout import encode_topk
from sparse_readout_prism.research.run_io import load_bank, run_provenance
from sparse_readout_prism.utils import find_lm_head, load_causal_lm, resolve_device, set_seed, write_csv, write_json

# bank name -> (file under --bank-dir, default cap on base cases)
BANK_FILES = {
    "curated_ab": ("qwen_gemma_result1_curated_ab.jsonl", 320),
    "case_candidates": ("qwen_gemma_result1_case_candidates.jsonl", 40),
    "model_native": ("qwen_gemma_result1_model_native_prompts_c4.jsonl", 300),
}


def log(msg: str) -> None:
    print(f"[causal {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Banks and tokens
# --------------------------------------------------------------------------- #


def load_banks(bank_dir: Path, banks: list[str], caps: dict[str, int]) -> list[dict]:
    """Load the requested banks, capping each by base case; a missing file is skipped."""
    rows: list[dict] = []
    for b in banks:
        fname, default_cap = BANK_FILES[b]
        path = bank_dir / fname
        if not path.exists():
            log(f"bank missing, skipping: {path}")
            continue
        kept = load_bank(path, caps.get(b, default_cap))
        for r in kept:
            r["bank"] = b
        rows.extend(kept)
        log(f"bank {b}: {len(kept)} records ({len({r['base_case_id'] for r in kept})} base cases)")
    return rows


def single_token_id(tok, term: str):
    """Token id of ``term`` if it (bare first, then space-prefixed) is a single token, else None."""
    for variant in (term, " " + term):
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


# --------------------------------------------------------------------------- #
# Core math (unit-testable without a model)
# --------------------------------------------------------------------------- #


def causal_pairs_for_contrast(
    h: torch.Tensor,  # (d,)
    q: torch.Tensor,  # (d,) = w_A - w_B (raw)
    beta: torch.Tensor,  # (D,) raw-space code difference (rn_A*z_A - rn_B*z_B)
    W_dec: torch.Tensor,  # (D, d) unit rows
    top_k: int,
    rng: np.random.Generator,
    n_random: int,
):
    """Return (feature_id, c_pred, delta_real, is_random) tuples."""
    proj = W_dec @ h  # (D,) h . d_i
    c = beta * proj  # predicted signed contribution per feature
    qd = W_dec @ q  # (D,) q . d_i
    realized = proj * qd  # = -(delta m) under ablation; sign matches c's claim
    top = torch.topk(c.abs(), k=min(top_k, c.numel())).indices
    active = set(top.tolist())
    out = [(int(i), float(c[i]), float(realized[i]), 0) for i in top]
    # random control: features NOT in the predicted-top set (mostly beta=0)
    pool = rng.choice(c.numel(), size=min(4 * n_random, c.numel()), replace=False)
    ctrl = [int(i) for i in pool if int(i) not in active][:n_random]
    out += [(int(i), float(c[i]), float(realized[i]), 1) for i in ctrl]
    return out


def fit_r2_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS y = s*x through the origin and the r2 of that fit.

    Called as ``fit_r2_slope(x=c_pred, y=delta_real)``: the slope is realized
    change on predicted contribution, so slope > 1 means the realized change
    exceeds the prediction.
    """
    if len(x) < 3 or float(np.dot(x, x)) == 0.0:
        return float("nan"), float("nan")
    s = float(np.dot(x, y) / np.dot(x, x))
    resid = y - s * x
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / denom if denom > 0 else float("nan")
    return r2, s


def cluster_bootstrap_r2(rows: list[dict], n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of r2 under a cluster bootstrap over ``row['cluster']``."""
    by_cluster: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_cluster.setdefault(r["cluster"], []).append((r["c_pred"], r["delta_real"]))
    keys = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(len(keys), size=len(keys), replace=True)
        xs, ys = [], []
        for si in sample:
            for cx, cy in by_cluster[keys[si]]:
                xs.append(cx)
                ys.append(cy)
        r2, _ = fit_r2_slope(np.array(xs), np.array(ys))
        if np.isfinite(r2):
            stats.append(r2)
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def self_test_metrics(seed: int = 0) -> dict:
    """Synthetic residual-free check of the math core (no model).

    A random unit-row dictionary, a random decoded state, and a contrast built
    exactly in the dictionary span from a known 3-sparse beta. Returns the
    fitted statistics, the pass flags and the synthetic inputs so callers can
    run further identity checks.
    """
    torch.manual_seed(seed)
    d, D = 64, 512
    W_dec = torch.randn(D, d)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    h = torch.randn(d)
    rng = np.random.default_rng(seed)
    # Construct q exactly in the dictionary span with known sparse beta:
    beta = torch.zeros(D)
    beta[[3, 40, 100]] = torch.tensor([2.0, -1.5, 0.7])
    q = beta @ W_dec  # residual-free contrast
    pairs = causal_pairs_for_contrast(h, q, beta, W_dec, top_k=3, rng=rng, n_random=8)
    pred = np.array([p[1] for p in pairs if p[3] == 0])
    real = np.array([p[2] for p in pairs if p[3] == 0])
    r2, slope = fit_r2_slope(pred, real)
    # With orthonormal-ish random dirs and residual-free q, agreement is high
    # but NOT exact (off-diagonal d_i . d_j != 0) -- assert strong, not perfect.
    ok1 = r2 > 0.9 and 0.7 < slope < 1.3
    ctrl_pred = np.array([abs(p[1]) for p in pairs if p[3] == 1])
    ctrl_max_abs_pred = float(ctrl_pred.max(initial=0.0))
    ok2 = ctrl_max_abs_pred < 1e-6  # random features carry ~0 prediction
    # Identity check: realized values equal the algebraic form
    i0 = pairs[0][0]
    lhs = pairs[0][2]
    rhs = float((W_dec[i0] @ h) * (W_dec[i0] @ q))
    identity_abs_err = abs(lhs - rhs)
    ok3 = identity_abs_err < 1e-4
    return {
        "r2": r2,
        "slope": slope,
        "ctrl_max_abs_pred": ctrl_max_abs_pred,
        "identity_abs_err": identity_abs_err,
        "checks": [
            ("r2/slope on span-constructed q", ok1),
            ("random controls predict ~0", ok2),
            ("realized identity", ok3),
        ],
        "pairs": pairs,
        "h": h,
        "q": q,
        "beta": beta,
        "W_dec": W_dec,
    }


def self_test() -> int:
    m = self_test_metrics()
    for name, ok in m["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  (r2={m['r2']:.3f} slope={m['slope']:.3f})")
    return 0 if all(ok for _, ok in m["checks"]) else 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", type=str, help="HF model id; W_U is read from its live lm_head")
    ap.add_argument("--checkpoint", type=Path, help="TopK dictionary checkpoint.pt (trainer schema)")
    ap.add_argument("--bank-dir", type=Path, help="directory holding the JSONL query banks")
    ap.add_argument("--banks", type=str, default="curated_ab,case_candidates,model_native")
    ap.add_argument("--max-native", type=int, default=300, help="base-case cap for the model_native bank")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--top-features", type=int, default=10)
    ap.add_argument("--random-per-case", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--rho-gate", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap contrasts")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not (args.model_id and args.checkpoint and args.bank_dir and args.out_dir):
        ap.error("--model-id, --checkpoint, --bank-dir, --out-dir required (or --self-test)")
    bad = [b for b in args.banks.split(",") if b.strip() and b.strip() not in BANK_FILES]
    if bad:
        ap.error(f"unknown bank(s) {bad}; have {list(BANK_FILES)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    log(f"loading model {args.model_id}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model, tok = load_causal_lm(args.model_id, dtype=dtype, device_map=None)
    model.to(device)
    lm_head = find_lm_head(model)
    W_U = lm_head.weight.detach().float().to(device)  # (V, d)
    V, d = W_U.shape
    row_mean = W_U.mean(dim=0)  # (d,) centering mean of the live readout
    log(f"W_U ({V}, {d}) from live lm_head")

    log(f"loading dictionary {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sae = load_factorizer(ckpt, device=device, freeze=True)
    if not isinstance(sae, TopKSAE):
        raise ValueError(f"expected a TopK-family dictionary, got architecture {sae.architecture!r}")
    k = int(sae.k)
    W_dec = sae.decoder.detach().float()  # (D, d)
    enc_w = sae.encoder.weight.detach().float()  # (D, d)
    enc_b = sae.encoder.bias.detach().float()  # (D,)
    dec_norm = W_dec.norm(dim=1)
    W_dec_unit = W_dec / dec_norm[:, None].clamp_min(1e-8)  # (D, d) unit rows
    ckpt_row_mean = ckpt.get("row_mean")
    if ckpt_row_mean is not None:
        gap = float((ckpt_row_mean.float().to(device) - row_mean).abs().max())
        log(f"checkpoint row_mean vs live W_U mean: max abs gap {gap:.3e} (the live mean is used)")
    log(f"dictionary D={W_dec.shape[0]} k={k} (decoder norms {dec_norm.min():.3f}-{dec_norm.max():.3f})")

    banks = [b.strip() for b in args.banks.split(",") if b.strip()]
    caps = {"model_native": args.max_native}
    records = load_banks(args.bank_dir, banks, caps)
    if args.limit:
        records = records[: args.limit]

    # hidden-state capture: lm_head input at final position
    cache: dict[str, torch.Tensor] = {}
    grabbed: list[torch.Tensor] = []

    def pre_hook(_m, inputs):
        grabbed.append((inputs[0] if isinstance(inputs, tuple) else inputs).detach())

    hook = lm_head.register_forward_pre_hook(pre_hook)

    @torch.no_grad()
    def hidden_for(prompt: str) -> torch.Tensor:
        if prompt not in cache:
            grabbed.clear()
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            model(**enc)
            cache[prompt] = grabbed[-1][0, -1, :].float()  # (d,)
            if len(cache) > 4096:
                cache.pop(next(iter(cache)))
        return cache[prompt]

    @torch.no_grad()
    def raw_beta(a_id: int, b_id: int) -> torch.Tensor:
        rn, x = center_normalize_rows(W_U[[a_id, b_id]], row_mean)  # (2,), (2, d)
        codes = encode_topk(x, enc_w, enc_b, k)  # (2, D)
        return rn[0] * codes[0] - rn[1] * codes[1]  # raw-space beta_i(q)

    out_rows: list[dict] = []
    n_done = n_skip = 0
    for rec in records:
        a = rec.get("target_a")
        b = rec.get("target_b")
        prompt = rec.get("prompt")
        if not (a and b and prompt):
            n_skip += 1
            continue
        a_id, b_id = single_token_id(tok, a), single_token_id(tok, b)
        if a_id is None or b_id is None or a_id == b_id:
            n_skip += 1
            continue
        h = hidden_for(prompt)
        q = (W_U[a_id] - W_U[b_id]).float()
        beta = raw_beta(a_id, b_id)
        # coverage quantities (margin reconstruction; offset cancels for a difference)
        m_exact = float(h @ q)
        m_recon = float((beta * (W_dec_unit @ h)).sum())
        rho = abs(m_exact - m_recon) / max(abs(m_exact), 1e-6)
        gated = int(rho < args.rho_gate and (m_exact > 0) == (m_recon > 0))
        cluster = rec.get("base_case_id") or rec.get("case_id") or hashlib.sha1(prompt.encode()).hexdigest()[:10]
        pairs = causal_pairs_for_contrast(h, q, beta, W_dec_unit, args.top_features, rng, args.random_per_case)
        for fid, c_pred, delta_real, is_rand in pairs:
            out_rows.append(
                dict(
                    case_id=rec.get("case_id", ""),
                    cluster=cluster,
                    bank=rec.get("bank", ""),
                    target_a=a,
                    target_b=b,
                    feature_id=fid,
                    c_pred=c_pred,
                    delta_real=delta_real,
                    is_random=is_rand,
                    m_exact=m_exact,
                    rho=rho,
                    gated=gated,
                )
            )
        n_done += 1
        if n_done % 100 == 0:
            log(f"contrasts {n_done}/{len(records)} (skipped {n_skip})")
    hook.remove()
    log(f"done: {n_done} contrasts, {n_skip} skipped, {len(out_rows)} feature pairs")

    write_csv(args.out_dir / "causal_rows.csv", out_rows, write_empty=True)

    summary = {
        "model_id": args.model_id,
        "n_contrasts": n_done,
        "k": k,
        "top_features": args.top_features,
        "seed": args.seed,
    }
    for subset, sel in [
        ("gated_predicted", lambda r: r["gated"] and not r["is_random"]),
        ("ungated_predicted", lambda r: not r["is_random"]),
        ("gated_random_control", lambda r: r["gated"] and r["is_random"]),
    ]:
        rows = [r for r in out_rows if sel(r)]
        x = np.array([r["c_pred"] for r in rows])
        y = np.array([r["delta_real"] for r in rows])
        r2, slope = fit_r2_slope(x, y)
        lo, hi = cluster_bootstrap_r2(rows, args.n_boot, args.seed)
        summary[subset] = dict(n=len(rows), r2=r2, slope=slope, r2_ci=[lo, hi])
        log(f"{subset}: n={len(rows)} r2={r2:.4f} slope={slope:.3f} CI[{lo:.3f},{hi:.3f}]")
    write_json(summary, args.out_dir / "summary.json")
    write_json(run_provenance(args), args.out_dir / "manifest.json")
    log(f"wrote {args.out_dir}/causal_rows.csv, summary.json and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
