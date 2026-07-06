#!/usr/bin/env python3
"""Task-fidelity evaluation: per-task margin / query fidelity for the paper's fidelity gate.

Scores a trained SAE checkpoint's RECONSTRUCTED W_U against task-level
decisions (contrastive margins + linear readout queries), not row
reconstruction. This is the paper-facing fidelity evaluation: row-level EV
alone is insufficient to support a headline claim without it.

It does NOT invent task data. It consumes a benchmark bank produced separately
(its schema is documented just below). Bank schema (a torch .pt dict):

  h:            (N, d_model) float  -- benchmark hidden states, SAME space as
                                       W_U (logit = h @ W_U.T); final-norm.
  margin_items: list[dict], each:
      h_idx:    int                 -- row into h
      A_tokens: 1-D long            -- target token family (vocab ids)
      B_tokens: 1-D long            -- contrast token family
      A_weights/B_weights: 1-D float (optional; default uniform mean)
      battery:  str                 -- benchmark id (stratification key)
      pair_id:  str                 -- base contrast pair (split clustering)
  query_items:  list[dict], each:
      h_idx:    int
      query_type: str               -- availability|authority|tool|winner_vs_rest|target_vs_rest
      signed:   bool
      battery:  str
      query_id: str
      # linear-score queries (correlation/sign/residual):
      tokens:   1-D long            -- support of q = sum_t alpha_t W_U[t]
      weights:  1-D float           -- alpha_t
      # rank-preservation queries (target-vs-rest); EITHER form may be present:
      target_token:  int
      candidate_set: 1-D long       -- the "rest" pool incl. target

m_exact(h)  = h . (mean(W_U[A]) - mean(W_U[B]))     [family averages]
m_approx(h) = h . (mean(W_U_recon[A]) - mean(W_U_recon[B]))

Writes metrics_task_fidelity.json into the cell dir; idempotent (skips if complete,
--force to recompute). W_U reconstruction uses the standard decode-and-
denormalize transform, so the checkpoint/config schema is identical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from sparse_readout_prism.data import center_normalize_rows, resolve_row_mean
from sparse_readout_prism.factorizers import load_factorizer
from sparse_readout_prism.utils import resolve_device

EPS = 1e-6


def atomic_write_json(data, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@torch.no_grad()
def reconstruct_w_u(model, rows_normalized, row_mean, row_norms, k, batch=4096):
    """(vocab, d_model) reconstructed W_U — decode normalized rows, restore raw coords."""
    chunks = []
    for s in range(0, rows_normalized.shape[0], batch):
        chunks.append(model(rows_normalized[s : s + batch], k=k).reconstruction)
    recon_x = torch.cat(chunks, dim=0)
    return row_mean[None, :] + row_norms[:, None] * recon_x


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    a = a.double()
    b = b.double()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / d)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    ra = a.double().argsort().argsort().double()
    rb = b.double().argsort().argsort().double()
    return _pearson(ra, rb)


def _resid_direct(exact: torch.Tensor, approx: torch.Tensor) -> torch.Tensor:
    return (exact - approx).abs() / exact.abs().clamp_min(EPS)


def _family_vec(W: torch.Tensor, toks: torch.Tensor, wts: torch.Tensor | None) -> torch.Tensor:
    rows = W[toks]  # (|fam|, d_model)
    if wts is None:
        return rows.mean(dim=0)
    w = wts.to(rows.dtype)
    return (w[:, None] * rows).sum(dim=0) / w.sum().clamp_min(EPS)


def _strat(values: torch.Tensor, keys: list[str]) -> dict:
    out = {}
    uniq = sorted(set(keys))
    for kname in uniq:
        sel = torch.tensor([j for j, kk in enumerate(keys) if kk == kname])
        if sel.numel():
            v = values[sel]
            out[kname] = {"n": int(sel.numel()), "median_resid_direct": float(v.median())}
    return out


@torch.no_grad()
def margin_metrics(h, W_orig, W_rec, items: list[dict], device) -> dict:
    if not items:
        return {"n": 0}
    me, ma, bat, mag_keys = [], [], [], []
    for it in items:
        hv = h[int(it["h_idx"])]
        A = it["A_tokens"].to(device)
        B = it["B_tokens"].to(device)
        aw = it.get("A_weights")
        bw = it.get("B_weights")
        aw = aw.to(device) if aw is not None else None
        bw = bw.to(device) if bw is not None else None
        de = _family_vec(W_orig, A, aw) - _family_vec(W_orig, B, bw)
        da = _family_vec(W_rec, A, aw) - _family_vec(W_rec, B, bw)
        me.append(hv @ de)
        ma.append(hv @ da)
        bat.append(str(it.get("battery", "all")))
    me = torch.stack(me)
    ma = torch.stack(ma)
    rd = _resid_direct(me, ma)
    # magnitude strata by |m_exact| terciles
    q1, q2 = torch.quantile(me.abs().double(), torch.tensor([1 / 3, 2 / 3], device=device).double())
    for v in me.abs():
        mag_keys.append("low" if v < q1 else ("mid" if v < q2 else "high"))
    A_ = me.double()
    A_c = A_ - A_.mean()
    slope = float((A_c @ (ma.double() - ma.double().mean())) / A_c.pow(2).sum().clamp_min(1e-12))
    intercept = float(ma.double().mean() - slope * A_.mean())
    per_bat = _strat(rd, bat)
    return {
        "n": len(items),
        "pearson": _pearson(me, ma),
        "spearman": _spearman(me, ma),
        "sign_agreement": float((me.sign() == ma.sign()).double().mean()),
        "median_resid_direct": float(rd.median()),
        "p90_resid_direct": float(torch.quantile(rd.double(), 0.90)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "by_battery": per_bat,
        "worst_battery_median_resid_direct": (
            max(v["median_resid_direct"] for v in per_bat.values()) if per_bat else None
        ),
        "by_margin_magnitude": _strat(rd, mag_keys),
    }


@torch.no_grad()
def query_metrics(h, W_orig, W_rec, items: list[dict], device) -> dict:
    if not items:
        return {"n": 0}
    lin_e, lin_a, lin_signed, lin_bat, lin_type = [], [], [], [], []
    rank_top1, rank_spear, rank_bat, rank_type = [], [], [], []
    for it in items:
        hv = h[int(it["h_idx"])]
        qtype = str(it.get("query_type", "linear"))
        if "candidate_set" in it and it.get("candidate_set") is not None:
            cset = it["candidate_set"].to(device)
            tgt = int(it["target_token"])
            se = hv @ W_orig[cset].T  # (|cset|,)
            sa = hv @ W_rec[cset].T
            tpos = (cset == tgt).nonzero(as_tuple=True)[0]
            if tpos.numel():
                ti = int(tpos[0])
                rank_top1.append(float((se.argmax() == ti) and (sa.argmax() == ti)))
                rank_spear.append(_spearman(se, sa))
                rank_bat.append(str(it.get("battery", "all")))
                rank_type.append(qtype)
        if "tokens" in it and it.get("tokens") is not None:
            toks = it["tokens"].to(device)
            w = it["weights"].to(device).to(W_orig.dtype)
            qe = (w[:, None] * W_orig[toks]).sum(dim=0)
            qa = (w[:, None] * W_rec[toks]).sum(dim=0)
            lin_e.append(hv @ qe)
            lin_a.append(hv @ qa)
            lin_signed.append(bool(it.get("signed", True)))
            lin_bat.append(str(it.get("battery", "all")))
            lin_type.append(qtype)
    out: dict = {"n": len(items)}
    if lin_e:
        e = torch.stack(lin_e)
        a = torch.stack(lin_a)
        rd = _resid_direct(e, a)
        sgn = torch.tensor(lin_signed)
        out["linear"] = {
            "n": int(e.numel()),
            "pearson": _pearson(e, a),
            "spearman": _spearman(e, a),
            "sign_agreement_signed": (float((e[sgn].sign() == a[sgn].sign()).double().mean()) if sgn.any() else None),
            "median_resid_direct": float(rd.median()),
            "p90_resid_direct": float(torch.quantile(rd.double(), 0.90)),
            "by_battery": _strat(rd, lin_bat),
            "by_query_type": _strat(rd, lin_type),
        }
    if rank_top1:
        out["rank_preservation"] = {
            "n": len(rank_top1),
            "target_top1_preserved": float(torch.tensor(rank_top1).mean()),
            "mean_candidate_spearman": float(torch.tensor([x for x in rank_spear if x == x]).mean()),
        }
    return out


def run(args, device) -> None:
    cell = Path(args.cell_dir)
    ckpt_p = Path(args.checkpoint) if args.checkpoint else cell / "checkpoint.pt"
    cfg_p = cell / "config.yaml"
    if not ckpt_p.exists() or not cfg_p.exists():
        raise SystemExit(f"missing {ckpt_p} or {cfg_p}")
    out_p = cell / args.out_name
    if out_p.exists() and not args.force and json.loads(out_p.read_text()).get("complete"):
        print(f"[skip] {out_p} already complete")
        return

    cfg = yaml.safe_load(cfg_p.read_text())
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=True)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)

    raw = torch.load(args.data_path, map_location="cpu", weights_only=True)
    W_U = raw.get("W_U_orig", raw.get("W_U")).float().to(device)
    # Centering matched to training: checkpoint row_mean, else token_mask-kept
    # mean, else full-vocab mean — NOT an ad-hoc W_U.mean(dim=0), which silently
    # diverges from training on masked (multimodal) vocabularies.
    row_mean = resolve_row_mean(W_U, token_mask=raw.get("token_mask"), ckpt=ckpt).to(device)
    row_norms, rows_normalized = center_normalize_rows(W_U, row_mean)

    model = load_factorizer(ckpt, factorizer_config=cfg.get("factorizer", {}), d_model=W_U.shape[1], device=device)
    k = int(args.k or cfg.get("factorizer", {}).get("k", 128))

    t0 = time.time()
    W_rec = reconstruct_w_u(model, rows_normalized, row_mean, row_norms, k)
    h = bank["h"].float().to(device)
    if h.shape[1] != W_U.shape[1]:
        raise SystemExit(f"bank h d_model {h.shape[1]} != W_U {W_U.shape[1]}")

    margins = margin_metrics(h, W_U, W_rec, bank.get("margin_items", []), device)
    queries = query_metrics(h, W_U, W_rec, bank.get("query_items", []), device)
    out = {
        "cell": cell.name,
        "checkpoint": str(ckpt_p),
        "k": k,
        "bank": str(args.bank),
        "n_h": int(h.shape[0]),
        "margins": margins,
        "queries": queries,
        "complete": True,
    }
    atomic_write_json(out, out_p)
    mp = margins.get("pearson")
    print(
        f"[task_fidelity] {out_p} margin_pearson={mp if mp is None else round(mp, 4)} "
        f"margin_sign_agree={margins.get('sign_agreement')} "
        f"({time.time() - t0:.1f}s)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", required=True)
    p.add_argument("--data-path", required=True, help="W_U source .pt (W_U_orig/W_U)")
    p.add_argument("--bank", required=True, help="benchmark margin/query bank .pt")
    p.add_argument("--checkpoint", default=None, help="specific ckpt (default cell/checkpoint.pt)")
    p.add_argument("--k", type=int, default=0, help="eval k (default factorizer.k)")
    p.add_argument("--out-name", default="metrics_task_fidelity.json")
    p.add_argument("--device", default="auto", help="cuda|mps|cpu|auto")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args, resolve_device(args.device))


if __name__ == "__main__":
    main()
