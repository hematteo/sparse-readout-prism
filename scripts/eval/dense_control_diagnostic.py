#!/usr/bin/env python3
"""Dense and negative-control row-reconstruction diagnostic (Appendix K).

Reproduces the "Dense and Negative-Control Results Omitted from Section 5"
numbers (recorded literals in ``assets/appendix/appendix_k_sweep_tables.json`` under
``omp_diagnostic``): on the centred+normalised W_U rows the trained encoder is
compared against coefficient refits on its own support and against greedy
matching pursuit over the full decoder dictionary, plus a dense rank-k
reference. All methods are scored by the same row-centred explained variance
(``row_centered_ev`` in ``sparse_readout_prism.evaluate``):

    rowEV = 1 - ||X - X_hat||_F^2 / ||X||_F^2      (X = rows_normalized)

Methods (``--methods``):
  encoder       trained TopK SAE reconstruction (the deployed object)        [encoder]
  ls_support    signed least squares refit on the encoder's active support   [refit]
  nnls_support  non-negative least squares refit on the encoder's support    [refit]
  omp_signed    greedy OMP over the full dictionary, k atoms, signed refit    [omp]
  omp_nonneg    greedy OMP, positive-correlation selection, NNLS refit        [omp]
  dense_rank    best rank-k reconstruction (truncated SVD of the row matrix)  [dense]

These are diagnostics over a *fixed* dictionary, not the deployed encoder: they
bound how much of the encoder's headroom is selection vs coefficients, and show
the sparse dictionary is not merely a dense low-rank basis. They are NOT a new
readout-side decomposition claim.

Greedy OMP selects atoms by residual correlation (signed: max |corr|; nonneg:
max positive corr) with unit-norm matching-pursuit residual updates, then refits
coefficients over the selected support by (N)NLS. The paper used a seeded
20,000-row subsample with 1,000-sample row bootstraps; defaults here are smaller
so the script runs without a GPU. OMP over a 131k-atom dictionary wants a GPU.

Example
-------
    uv run python scripts/eval/dense_control_diagnostic.py \
        --w-u data/qwen35-9b/qwen35_9b.pt \
        --checkpoint results/qwen9b_no_pca_width_k_sweep/converged/topk_d131072_k256_rowseeded_hybrid_lamramp_s0/checkpoint.pt \
        --setting "Qwen-9B 32x, k=256" --n-rows 20000 --bootstrap 1000 \
        --out-csv results/dense_control/qwen9b_32x_k256.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.data import preprocess_rows
from sparse_readout_prism.factorizers import load_factorizer
from sparse_readout_prism.utils import set_seed, write_csv, write_json

ALL_METHODS = ("encoder", "ls_support", "nnls_support", "omp_signed", "omp_nonneg", "dense_rank")
METHOD_CLASS = {
    "encoder": "encoder",
    "ls_support": "refit",
    "nnls_support": "refit",
    "omp_signed": "omp",
    "omp_nonneg": "omp",
    "dense_rank": "dense",
}
METHOD_LABEL = {
    "encoder": "Trained encoder",
    "ls_support": "Signed LS on support",
    "nnls_support": "NNLS on support",
    "omp_signed": "Signed OMP",
    "omp_nonneg": "Nonneg OMP",
    "dense_rank": "Dense rank-k",
}


def row_explained_variance(X: torch.Tensor, X_hat: torch.Tensor) -> float:
    """Row-centred explained variance, matching evaluate.row_centered_ev."""
    resid = X - X_hat
    return 1.0 - float(resid.pow(2).sum() / X.pow(2).sum().clamp_min(1e-12))


def _per_row_sq_error(X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
    """Per-row squared reconstruction error ``||x - x_hat||^2`` (for bootstrap)."""
    return (X - X_hat).pow(2).sum(dim=1)


@torch.no_grad()
def _encoder_recon(model, X: torch.Tensor, k: int, device, batch: int = 8192) -> torch.Tensor:
    out = []
    for s in range(0, X.shape[0], batch):
        xb = X[s : s + batch].to(device)
        out.append(model(xb, k=k).reconstruction.float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _encoder_support(model, X: torch.Tensor, k: int, device, batch: int = 8192) -> torch.Tensor:
    """Top-k encoder atom indices per row, ``(N, k)``."""
    import torch.nn.functional as F

    out = []
    for s in range(0, X.shape[0], batch):
        xb = X[s : s + batch].to(device)
        acts = F.relu(model.encoder(xb))  # (b, d_features)
        _, idx = acts.topk(k, dim=1)
        out.append(idx.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _refit_recon(
    X: torch.Tensor,
    decoder: torch.Tensor,
    support: torch.Tensor,
    *,
    nonneg: bool,
    device,
    row_chunk: int = 256,
) -> torch.Tensor:
    """Reconstruct each row from ``decoder[support_row]`` by (N)NLS / signed LS."""
    N, k = support.shape
    recon = torch.empty_like(X)
    nnls = None
    if nonneg:
        from scipy.optimize import nnls as _nnls

        nnls = _nnls
    for s in range(0, N, row_chunk):
        idx = support[s : s + row_chunk].to(device)  # (b, k)
        atoms = decoder[idx]  # (b, k, d_model)
        xb = X[s : s + row_chunk].to(device)  # (b, d_model)
        if nonneg:
            atoms_cpu = atoms.cpu().numpy()
            xb_cpu = xb.cpu().numpy()
            rec = np.empty_like(xb_cpu)
            for j in range(xb_cpu.shape[0]):
                A = atoms_cpu[j].T  # (d_model, k)
                c, _ = nnls(A, xb_cpu[j])
                rec[j] = A @ c
            recon[s : s + row_chunk] = torch.from_numpy(rec)
        else:
            A = atoms.transpose(1, 2)  # (b, d_model, k)
            sol = torch.linalg.lstsq(A, xb.unsqueeze(-1)).solution  # (b, k, 1)
            rec = torch.bmm(sol.transpose(1, 2), atoms).squeeze(1)  # (b, d_model)
            recon[s : s + row_chunk] = rec.float().cpu()
    return recon


@torch.no_grad()
def _omp_select(
    X: torch.Tensor,
    decoder: torch.Tensor,
    k: int,
    *,
    nonneg: bool,
    device,
    row_chunk: int = 256,
) -> torch.Tensor:
    """Greedy matching-pursuit atom selection over the full dictionary, ``(N, k)``.

    Atoms are unit-norm, so the projection coefficient on a chosen atom is the
    residual correlation; the residual update ``r -= (r·d) d`` is the standard MP
    step. Selection: signed picks max |corr|, nonneg picks max positive corr.
    Coefficients are refit afterwards by :func:`_refit_recon`.
    """
    N = X.shape[0]
    D = decoder.to(device)  # (M, d_model)
    support = torch.empty((N, k), dtype=torch.long)
    for s in range(0, N, row_chunk):
        r = X[s : s + row_chunk].to(device).clone()  # (b, d_model)
        b = r.shape[0]
        chosen = torch.full((b, k), -1, dtype=torch.long, device=device)
        for step in range(k):
            corr = r @ D.T  # (b, M)
            score = corr if nonneg else corr.abs()
            if step > 0:
                score.scatter_(1, chosen[:, :step], -float("inf"))
            best = score.argmax(dim=1)  # (b,)
            chosen[:, step] = best
            coeff = corr.gather(1, best[:, None]).squeeze(1)  # (b,)
            r = r - coeff[:, None] * D[best]  # MP residual update
        support[s : s + row_chunk] = chosen.cpu()
    return support


@torch.no_grad()
def _dense_rank_recon(X: torch.Tensor, k: int, device, oversample: int = 16) -> torch.Tensor:
    """Best rank-k reconstruction via truncated SVD of the row matrix."""
    Xd = X.to(device)
    q = min(k + oversample, min(Xd.shape) - 1)
    _, _, V = torch.svd_lowrank(Xd, q=q, niter=4)
    Vk = V[:, :k]  # (d_model, k)
    return ((Xd @ Vk) @ Vk.T).float().cpu()


def _bootstrap_rowev(
    per_row_err: torch.Tensor,
    row_energy: torch.Tensor,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if n_boot <= 0:
        return float("nan"), float("nan")
    err = per_row_err.numpy()
    energy = row_energy.numpy()
    rng = np.random.default_rng(seed)
    n = err.shape[0]
    evs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        evs[b] = 1.0 - err[pick].sum() / max(float(energy[pick].sum()), 1e-12)
    return float(np.percentile(evs, 2.5)), float(np.percentile(evs, 97.5))


def run_diagnostic(
    *,
    rows_normalized: torch.Tensor,
    model,
    k: int,
    methods: list[str],
    setting: str,
    n_boot: int,
    seed: int,
    device,
) -> list[dict]:
    X = rows_normalized  # (N, d_model), already centred + normalised
    decoder = model.decoder.detach().float()
    row_energy = X.pow(2).sum(dim=1)
    encoder_support = None
    results: list[dict] = []
    for method in methods:
        t0 = time.time()
        if method == "encoder":
            recon = _encoder_recon(model, X, k, device)
        elif method in ("ls_support", "nnls_support"):
            if encoder_support is None:
                encoder_support = _encoder_support(model, X, k, device)
            recon = _refit_recon(X, decoder, encoder_support, nonneg=(method == "nnls_support"), device=device)
        elif method in ("omp_signed", "omp_nonneg"):
            nonneg = method == "omp_nonneg"
            support = _omp_select(X, decoder, k, nonneg=nonneg, device=device)
            recon = _refit_recon(X, decoder, support, nonneg=nonneg, device=device)
        elif method == "dense_rank":
            recon = _dense_rank_recon(X, k, device)
        else:
            raise ValueError(f"unknown method: {method}")
        ev = row_explained_variance(X, recon)
        ci_lo, ci_hi = _bootstrap_rowev(_per_row_sq_error(X, recon), row_energy, n_boot=n_boot, seed=seed)
        results.append(
            {
                "setting": setting,
                "method": METHOD_LABEL[method],
                "method_key": method,
                "method_class": METHOD_CLASS[method],
                "rowEV": round(ev, 4),
                "rowEV_ci_lo": round(ci_lo, 4) if ci_lo == ci_lo else None,
                "rowEV_ci_hi": round(ci_hi, 4) if ci_hi == ci_hi else None,
                "n_rows": int(X.shape[0]),
                "k": int(k),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(f"  {METHOD_LABEL[method]:22s} rowEV={ev:.4f}  ({results[-1]['seconds']}s)")
    return results


def _subsample_rows(rows_normalized: torch.Tensor, n_rows: int | None, seed: int) -> torch.Tensor:
    if n_rows is None or n_rows >= rows_normalized.shape[0]:
        return rows_normalized
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(rows_normalized.shape[0], generator=gen)[:n_rows]
    return rows_normalized[idx].contiguous()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w-u", type=Path, required=True, help="Extraction .pt payload ({W_U_orig, ...}).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Trained SAE checkpoint.pt.")
    ap.add_argument("--setting", type=str, required=True, help='Row label, e.g. "Qwen-9B 32x, k=256".')
    ap.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    ap.add_argument("--k", type=int, default=None, help="Active budget; default reads it from the checkpoint.")
    ap.add_argument("--n-rows", type=int, default=4000, help="Seeded row subsample (paper used 20000).")
    ap.add_argument("--bootstrap", type=int, default=200, help="Row bootstrap resamples for the rowEV CI.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        ap.error(f"unknown method(s) {bad}; have {list(ALL_METHODS)}")

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.w_u, map_location="cpu", weights_only=True)
    W_U = payload.get("W_U_orig", payload.get("W_U"))
    if W_U is None:
        raise SystemExit(f"{args.w_u}: no W_U_orig/W_U")
    token_mask = payload.get("token_mask")
    W_U = W_U.float()
    if token_mask is not None:
        W_U = W_U[token_mask.bool()]
    _, _, rows_normalized = preprocess_rows(W_U)
    rows_normalized = _subsample_rows(rows_normalized, args.n_rows, args.seed + 23)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_factorizer(ckpt, d_model=rows_normalized.shape[1], freeze=True).to(device)
    cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer", {})
    k = int(args.k or ckpt.get("evaluation", {}).get("k") or cfg.get("k"))

    print(f"[dense-control] {args.setting}: {rows_normalized.shape[0]} rows, k={k}, device={device}")
    results = run_diagnostic(
        rows_normalized=rows_normalized,
        model=model,
        k=k,
        methods=methods,
        setting=args.setting,
        n_boot=args.bootstrap,
        seed=args.seed,
        device=device,
    )

    if args.out_csv:
        write_csv(args.out_csv, results)
        print(f"wrote {args.out_csv}")
    if args.out_json:
        write_json({"setting": args.setting, "results": results}, args.out_json)
        print(f"wrote {args.out_json}")
    if not args.out_csv and not args.out_json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
