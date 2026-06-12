#!/usr/bin/env python3
"""Result-1 five-model query-fidelity runner.

Tests whether the Sparse Readout Prism preserves selected readout-query scores
and signs (exact h^T(W_U[A]-W_U[B]) vs sparse h^T(W_hat[A]-W_hat[B])) for the
five paper models, across three query banks and two operating points.

Design anchors:
  * W_U is the exact training matrix from the model's extraction artifact
    (data/<m>/<m>.pt key W_U_orig) -- identical to what each SAE was trained on.
  * h is the EXACT readout input captured via a forward_pre_hook on lm_head,
    so h @ W_U.T == lm_head(h) (pre-softcap for Gemma; the prism decomposes the
    pre-softcap readout, rowEV is cap-independent).
  * Decomposition uses src/sparse_readout_prism.decompose (the canonical API);
    margins are differences of per-row decompositions, sharing the feature basis.
  * Per (model, operating_point) cell is resumable: a cell is skipped if its
    done.json exists; rows are written atomically.

Outputs include cell-level metrics, per-query rows, and paper-facing mirrors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.decompose import decompose_token_logit
from sparse_readout_prism.evaluate import (
    reconstruct_normalized_rows,
)
from sparse_readout_prism.factorizers import load_factorizer
from sparse_readout_prism.paths import repo_root
from sparse_readout_prism.research._common.cell_metrics import coverage_stats
from sparse_readout_prism.research._common.run_io import (
    group_rows as _by,
    write_rows_csv as _write_csv,
)
from sparse_readout_prism.research._common.registry import (
    load_model,
    resolve_registry,
    resolve_single_token,
)
from sparse_readout_prism.utils import (
    atomic_write_text as _atomic_write,
    find_lm_head,
    pearson as _pearson,
    spearman as _spearman,
    write_jsonl as _write_jsonl,
)

REPO = repo_root()

LN2 = float(np.log(2.0))

EPS = 1e-6
MARGIN_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
MARGIN_BIN_LABELS = ["<0.5", "0.5-1", "1-2", ">=2"]


def log(msg: str) -> None:
    print(f"[qfid {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Decomposition helpers
# --------------------------------------------------------------------------- #


def preprocess_row(W_row: torch.Tensor, row_mean: torch.Tensor):
    centered = W_row - row_mean
    rn = centered.norm().clamp_min(1e-8)
    return rn, centered / rn


def decompose_row(h, W_row, row_mean, model, k):
    rn, rnorm = preprocess_row(W_row, row_mean)
    return decompose_token_logit(h, W_row, row_mean, rn, rnorm, model, k)


def margin_from_rows(h, W, rmean, model, k, a_ids, b_ids, mean_row=None):
    """Decompose A and B (single id, id-list-mean, or mean_row pseudo-target).
    Returns dict with exact/sparse/residual margins and per-feature margin."""

    @torch.no_grad()
    def agg(ids, pseudo):
        if pseudo is not None:
            d = decompose_row(h, pseudo, rmean, model, k)
            return (
                d.original_logit,
                d.reconstructed_logit,
                d.residual_term,
                d.feature_contributions,
            )
        accs = None
        for tid in ids:
            d = decompose_row(h, W[tid], rmean, model, k)
            cur = (
                d.original_logit,
                d.reconstructed_logit,
                d.residual_term,
                d.feature_contributions,
            )
            accs = cur if accs is None else tuple(a + b for a, b in zip(accs, cur))
        n = float(len(ids))
        return tuple(a / n for a in accs)

    oA, sA, rA, fA = agg(a_ids, mean_row if a_ids is None else None)
    oB, sB, rB, fB = agg(b_ids, mean_row if b_ids is None else None)
    exact = float((oA - oB).detach())
    sparse = float((sA - sB).detach())
    feat_margin = (fA - fB).detach().cpu()
    return {
        "exact": exact,
        "sparse": sparse,
        "residual": exact - sparse,
        "resid_term": float((rA - rB).detach()),
        "feat_margin": feat_margin,
    }


# --------------------------------------------------------------------------- #
# Model-native frontier query construction
# --------------------------------------------------------------------------- #


def native_queries(logits, row_norms, row_mean, case_id, n_vocab):
    """Return list of (query_name, a_id, b_id, b_is_mean). top1 vs competitors."""
    topv, topi = torch.topk(logits, k=min(6, n_vocab))
    top1 = int(topi[0].item())
    out = [
        ("top1_top2", top1, int(topi[1].item()), False),
        ("top1_top5", top1, int(topi[min(4, len(topi) - 1)].item()), False),
        ("top1_vocabmean", top1, None, True),
    ]
    rng = np.random.default_rng(int(hashlib.md5(case_id.encode()).hexdigest()[:8], 16))
    ranked = torch.argsort(logits, descending=True)
    excl = set(int(x) for x in ranked[:20].tolist())
    # row-norm matched distractor among a sampled candidate pool
    pool = rng.choice(n_vocab, size=min(4096, n_vocab), replace=False)
    pool = [int(p) for p in pool if int(p) not in excl]
    if pool:
        tn = row_norms[top1]
        best = min(pool, key=lambda j: abs(float(row_norms[j] - tn)))
        out.append(("top1_rownorm_distractor", top1, best, False))
    lo, hi = 20, min(200, n_vocab)
    if hi > lo:
        samp = int(ranked[int(rng.integers(lo, hi))].item())
        out.append(("top1_sampled_competitor", top1, samp, False))
    return out


# --------------------------------------------------------------------------- #
# Headline corpus metrics (C4 model-native): KL bits/token, top1, frontier@0.5
# --------------------------------------------------------------------------- #


def _apply_softcap(logits: torch.Tensor, cap):
    if cap is None:
        return logits
    return float(cap) * torch.tanh(logits / float(cap))


@torch.no_grad()
def headline_position_row(
    h_dev: torch.Tensor,
    W_dev: torch.Tensor,
    reconW_dev: torch.Tensor,
    softcap,
    denom_floor: float,
) -> dict:
    """One C4 position: KL bits/token, top1 agreement, frontier_pass@0.5.

    Frontier A/B and margins use the linear pre-softcap readout; KL/top1 use
    the softcapped distribution when the model declares a tanh softcap (top1
    argmax is softcap-invariant; KL is not). Mirrors evaluate.py's
    val_logit_kl_bits_mean / val_top1_match (logits = hidden @ recon_W.T)."""
    le = W_dev @ h_dev  # (vocab,) exact linear logits
    ls = reconW_dev @ h_dev  # (vocab,) sparse linear logits
    a = int(torch.argmax(le).item())  # exact top1
    le_a = le.clone()
    le_a[a] = float("-inf")
    b = int(torch.argmax(le_a).item())  # exact top2
    m_exact = float((le[a] - le[b]).item())
    m_sparse = float((ls[a] - ls[b]).item())
    sign_match = (m_exact > 0) == (m_sparse > 0)
    rel = abs(m_exact - m_sparse) / max(abs(m_exact), denom_floor)
    frontier_pass = bool(sign_match and rel <= 0.5)

    le_c = _apply_softcap(le, softcap)
    ls_c = _apply_softcap(ls, softcap)
    logp = torch.log_softmax(le_c.double(), dim=0)
    logq = torch.log_softmax(ls_c.double(), dim=0)
    kl_bits = float(((logp.exp() * (logp - logq)).sum().item()) / LN2)
    top1_match = bool(int(torch.argmax(le_c).item()) == int(torch.argmax(ls_c).item()))
    return {
        "kl_bits": kl_bits,
        "top1_match": top1_match,
        "exact_top1_id": a,
        "exact_top2_id": b,
        "m_exact": m_exact,
        "m_sparse": m_sparse,
        "frontier_sign_match": bool(sign_match),
        "frontier_rel": float(rel),
        "frontier_pass": frontier_pass,
    }


def _corpus_metrics(rows: list[dict]) -> dict:
    """Headline aggregate with 95% CIs bootstrapped by base_case_id (C4 chunk)."""
    if not rows:
        return {}
    kl = np.array([r["kl_bits"] for r in rows], float)
    t1 = np.array([r["top1_match"] for r in rows], bool)
    fp = np.array([r["frontier_pass"] for r in rows], bool)
    by_base: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_base.setdefault(r["base_case_id"], []).append(i)
    bases = sorted(by_base)
    rng = np.random.default_rng(0)
    bk, bt, bf = [], [], []
    for _ in range(400):
        pick = rng.choice(bases, size=len(bases), replace=True)
        idx = [i for b in pick for i in by_base[b]]
        bk.append(kl[idx].mean())
        bt.append(t1[idx].mean())
        bf.append(fp[idx].mean())
    return {
        "n_base_cases": len(bases),
        "n_positions": len(rows),
        "kl_bits_mean": float(kl.mean()),
        "kl_bits_ci_lo": float(np.percentile(bk, 2.5)),
        "kl_bits_ci_hi": float(np.percentile(bk, 97.5)),
        "kl_bits_p90": float(np.percentile(kl, 90)),
        "top1_agreement": float(t1.mean()),
        "top1_agreement_ci_lo": float(np.percentile(bt, 2.5)),
        "top1_agreement_ci_hi": float(np.percentile(bt, 97.5)),
        "frontier_pass_at_0.5": float(fp.mean()),
        "frontier_pass_ci_lo": float(np.percentile(bf, 2.5)),
        "frontier_pass_ci_hi": float(np.percentile(bf, 97.5)),
    }


# --------------------------------------------------------------------------- #
# Per-cell evaluation
# --------------------------------------------------------------------------- #


def run_cell(model_name, op_name, m_entry, op, banks, args, cell_dir: Path):
    done = cell_dir / "done.json"
    if done.exists():
        log(f"[resume] cell {model_name}/{op_name} already done -> skip")
        return json.loads((cell_dir / "summary_cell.json").read_text())
    cell_dir.mkdir(parents=True, exist_ok=True)

    art = torch.load(m_entry["w_u_artifact"], map_location="cpu", weights_only=True)
    W = art.get("W_U_orig", art.get("W_U")).float()
    token_mask = art.get("token_mask")
    n_vocab, d_model = W.shape
    manifest = {}
    if m_entry.get("manifest") and Path(m_entry["manifest"]).exists():
        manifest = json.loads(Path(m_entry["manifest"]).read_text())
    softcap = None
    pr = manifest.get("post_readout_transform")
    if isinstance(pr, dict) and pr.get("type") == "tanh_softcap":
        softcap = float(pr["cap"])

    ckpt = torch.load(op["checkpoint"], map_location="cpu", weights_only=True)
    # readout-prism checkpoint schema: {model_state_dict, config{factorizer,...}}.
    # Workspace schema also stores factorizer/row_mean at top level; support both.
    cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer")
    if cfg is None:
        raise KeyError(f"no factorizer config in {op['checkpoint']}")
    sae = load_factorizer(ckpt, factorizer_config=cfg, d_model=d_model, freeze=True)
    k = int(ckpt.get("evaluation", {}).get("k") or cfg.get("k"))
    # row_mean: stored in workspace ckpts; for readout-prism ckpts reproduce the
    # training preprocessing exactly. config.data.row_preprocessing is
    # center_normalize with max_rows=null, and the extraction artifact carries
    # no token_mask, so row_mean is the full-vocab mean of W_U_orig (identical
    # to src/sparse_readout_prism.data.preprocess_rows over all rows).
    if ckpt.get("row_mean") is not None:
        row_mean = ckpt["row_mean"].float()
    elif token_mask is not None:
        row_mean = W[token_mask.bool()].mean(dim=0)
    else:
        row_mean = W.mean(dim=0)
    row_norms_all = (W - row_mean).norm(dim=1)
    ckpt_metrics = ckpt.get("metrics") or {}
    if not ckpt_metrics:
        for mf in ("metrics_v3.json", "metrics.json"):
            mp = Path(op["checkpoint"]).parent / mf
            if mp.exists():
                try:
                    ckpt_metrics = json.loads(mp.read_text())
                except Exception:  # noqa: BLE001
                    pass
                break

    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32
    model, tok = load_model(m_entry["model_id"], m_entry["revision"], dtype)
    dev = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm_head = find_lm_head(model)
    cap_box: dict = {}
    h_box: dict = {}

    def pre_hook(_m, inp):
        h_box["h"] = (inp[0] if isinstance(inp, tuple) else inp).detach()

    hh = lm_head.register_forward_pre_hook(pre_hook)

    # Identity guard: artifact W_U vs live lm_head. bf16 model weights round to
    # ~2^-8 relative; a gross mismatch means the artifact/model are not the same
    # readout and every metric below would be meaningless -> abort the cell.
    with torch.no_grad():
        wl = lm_head.weight.detach().float().cpu()[:n_vocab]
        wu_rel = float((W - wl).abs().max() / wl.abs().max().clamp_min(1e-9))
        cap_box["wu_vs_lmhead_rel"] = wu_rel
    # GPU placement for the curated_ab / case_candidates banks: sae.to(dev)
    # was previously only called inside the model_native branch, leaving the
    # earlier banks running d_features=131k matmuls on CPU.
    sae.to(dev)
    W = W.to(dev)
    row_mean = row_mean.to(dev)
    row_norms_all = row_norms_all.to(dev)
    if wu_rel > args.wu_tol:
        hh.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"W_U vs live lm_head mismatch {wu_rel:.4g} > --wu-tol {args.wu_tol} "
            f"for {model_name}/{op_name}: artifact and model readout differ"
        )

    @torch.no_grad()
    def hidden_for(prompt: str):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(dev)
        # noqa: F821 is a pyflakes false positive here -- `model` is a valid closure
        # variable (bound above); the `del model` below runs only after every
        # hidden_for() call, so the cell is never empty when this executes.
        model(**enc)  # noqa: F821
        am = enc["attention_mask"][0].bool()
        pos = int(am.nonzero()[-1].item())
        h = h_box["h"][0, pos].float().contiguous()  # keep on GPU
        return h

    rows: list[dict] = []
    audit: list[dict] = []
    skipped: list[dict] = []

    def emit(base_id, case_id, bank, family, query, mr, expected_side):
        exact, sparse = mr["exact"], mr["sparse"]
        rd = abs(exact - sparse) / max(abs(exact), EPS)
        cov = coverage_stats(mr["feat_margin"], sparse)
        sign_match = (exact > 0) == (sparse > 0)
        binlbl = next(MARGIN_BIN_LABELS[i] for i, (lo, hi) in enumerate(MARGIN_BINS) if lo <= abs(exact) < hi)
        row = {
            "model": model_name,
            "operating_point": op_name,
            "op_id": op["id"],
            "bank": bank,
            "family": family,
            "base_case_id": base_id,
            "case_id": case_id,
            "query": query,
            "k": k,
            "exact_margin": exact,
            "sparse_margin": sparse,
            "residual": exact - sparse,
            "abs_residual": abs(exact - sparse),
            "residual_direct": rd,
            "resid_term": mr["resid_term"],
            "sign_match": bool(sign_match),
            "abs_exact": abs(exact),
            "margin_bin": binlbl,
            "tiny_margin": abs(exact) < 0.5,
            "expected_side": expected_side,
            "softcap": softcap,
        }
        row.update(cov)
        rows.append(row)

    # ---- Bank: curated A/B and case candidates (string / family targets) ----
    for bank_name, bank_rows in banks.items():
        if bank_name == "model_native":
            continue
        for r in bank_rows:
            cap = args.max_cases if bank_name == "case_candidates" else args.max_curated
            if sum(1 for x in rows if x["bank"] == bank_name) >= cap * 4:
                # *4: each base case emits at most a few queries; safety cap
                pass
            fam = r["family"]
            try:
                h = hidden_for(r["prompt"])
            except Exception as e:  # noqa: BLE001
                skipped.append({"case_id": r["case_id"], "reason": f"forward:{e}"})
                continue
            if r.get("target_a") is not None:  # single-token A/B
                ta, va, ra = resolve_single_token(tok, r["target_a"])
                tb, vb, rb = resolve_single_token(tok, r["target_b"])
                audit += [
                    {
                        "model": model_name,
                        "case_id": r["case_id"],
                        "target": r["target_a"],
                        "side": "A",
                        "token_id": ta,
                        "variant": va,
                        "reason": ra or "ok",
                    },
                    {
                        "model": model_name,
                        "case_id": r["case_id"],
                        "target": r["target_b"],
                        "side": "B",
                        "token_id": tb,
                        "variant": vb,
                        "reason": rb or "ok",
                    },
                ]
                if ta is None or tb is None:
                    skipped.append({"case_id": r["case_id"], "reason": f"tok A={ra} B={rb}"})
                    continue
                if ta == tb:
                    skipped.append({"case_id": r["case_id"], "reason": "ab_collision"})
                    continue
                mr = margin_from_rows(h, W, row_mean, sae, k, [ta], [tb])
                emit(
                    r["base_case_id"],
                    r["case_id"],
                    bank_name,
                    fam,
                    "single_token",
                    mr,
                    r.get("expected_side"),
                )
            else:  # token-family row
                a_ids, b_ids = [], []
                for s in r["target_a_family"]:
                    tid, v, rsn = resolve_single_token(tok, s)
                    audit.append(
                        {
                            "model": model_name,
                            "case_id": r["case_id"],
                            "target": s,
                            "side": "Afam",
                            "token_id": tid,
                            "variant": v,
                            "reason": rsn or "ok",
                        }
                    )
                    if tid is not None:
                        a_ids.append(tid)
                for s in r["target_b_family"]:
                    tid, v, rsn = resolve_single_token(tok, s)
                    audit.append(
                        {
                            "model": model_name,
                            "case_id": r["case_id"],
                            "target": s,
                            "side": "Bfam",
                            "token_id": tid,
                            "variant": v,
                            "reason": rsn or "ok",
                        }
                    )
                    if tid is not None:
                        b_ids.append(tid)
                if not a_ids or not b_ids:
                    skipped.append({"case_id": r["case_id"], "reason": "family_empty"})
                    continue
                mr = margin_from_rows(h, W, row_mean, sae, k, a_ids, b_ids)
                emit(
                    r["base_case_id"],
                    r["case_id"],
                    bank_name,
                    fam + "_family",
                    "token_family",
                    mr,
                    r.get("expected_side"),
                )

    # ---- Bank: model-native frontier ----
    mn_rows = banks.get("model_native", [])
    corpus_rows: list[dict] = []
    if mn_rows and args.headline_corpus_summary:
        # Headline (C4): build the canonical full sparse unembedding once
        # (recon_W == evaluate.py: row_mean + row_norm * decode(encode(.))),
        # then KL bits/token, top1 agreement and frontier_pass@0.5 per position.
        # Headline-only: the per-query margin decompose is skipped here.
        rn = row_norms_all.clamp_min(1e-8)
        rows_norm_all = (W - row_mean) / rn[:, None]
        sae.to(dev)
        recon_x, _, _ = reconstruct_normalized_rows(sae, rows_norm_all, k=k)
        # recon_x is returned on CPU; build reconW on CPU too (row_mean/rn may be
        # on GPU) and move to dev below. Mixing devices here crashes on CUDA.
        reconW = row_mean.cpu()[None, :] + rn.cpu()[:, None] * recon_x  # cpu float32
        W_dev = W.to(dev, torch.float32)
        reconW_dev = reconW.to(dev, torch.float32)
        for r in mn_rows:
            try:
                h = hidden_for(r["prompt"])
            except Exception as e:  # noqa: BLE001
                skipped.append({"case_id": r["case_id"], "reason": f"forward:{e}"})
                continue
            hp = headline_position_row(
                h.to(dev, torch.float32),
                W_dev,
                reconW_dev,
                softcap,
                args.frontier_denom_floor,
            )
            hp.update(
                {
                    "model": model_name,
                    "operating_point": op_name,
                    "op_id": op["id"],
                    "k": k,
                    "bank": "model_native",
                    "family": r.get("family", "c4_model_native"),
                    "base_case_id": r["base_case_id"],
                    "case_id": r["case_id"],
                    "softcap": softcap,
                }
            )
            corpus_rows.append(hp)
        del W_dev, reconW_dev, reconW, recon_x, rows_norm_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        for r in mn_rows:
            try:
                h = hidden_for(r["prompt"])
            except Exception as e:  # noqa: BLE001
                skipped.append({"case_id": r["case_id"], "reason": f"forward:{e}"})
                continue
            logits = h @ W.T
            for qname, a_id, b_id, b_mean in native_queries(logits, row_norms_all, row_mean, r["case_id"], n_vocab):
                mr = margin_from_rows(
                    h,
                    W,
                    row_mean,
                    sae,
                    k,
                    [a_id],
                    None if b_mean else [b_id],
                    mean_row=row_mean if b_mean else None,
                )
                emit(
                    r["base_case_id"],
                    f"{r['case_id']}__{qname}",
                    "model_native",
                    f"native_{qname}",
                    qname,
                    mr,
                    None,
                )

    hh.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _write_jsonl(cell_dir / "rows.jsonl", rows)
    _write_jsonl(cell_dir / "tokenizer_audit.jsonl", audit)
    _write_jsonl(cell_dir / "skipped.jsonl", skipped)
    headline = {}
    if args.headline_corpus_summary:
        _write_jsonl(cell_dir / "corpus_rows.jsonl", corpus_rows)
        headline = _corpus_metrics(corpus_rows)
        if headline:
            log(
                f"cell {model_name}/{op_name} headline: "
                f"n={headline['n_positions']} "
                f"KL={headline['kl_bits_mean']:.4f} bits "
                f"top1={headline['top1_agreement']:.3f} "
                f"frontier@0.5={headline['frontier_pass_at_0.5']:.3f}"
            )
    cell_summary = {
        "model": model_name,
        "operating_point": op_name,
        "op_id": op["id"],
        "checkpoint": op["checkpoint"],
        "k": k,
        "n_vocab": int(n_vocab),
        "d_model": int(d_model),
        "softcap": softcap,
        "wu_vs_lmhead_rel": cap_box.get("wu_vs_lmhead_rel"),
        "ckpt_metrics": {kk: ckpt_metrics.get(kk) for kk in ("rowEV", "top1", "KL") if isinstance(ckpt_metrics, dict)},
        "n_rows": len(rows),
        "n_skipped": len(skipped),
        "n_audit": len(audit),
        "n_corpus_positions": len(corpus_rows),
        "headline": headline,
    }
    _atomic_write(cell_dir / "summary_cell.json", json.dumps(cell_summary, indent=2))
    _atomic_write(done, json.dumps({"done_at": time.time()}))
    log(f"cell {model_name}/{op_name}: {len(rows)} rows, {len(skipped)} skipped")
    return cell_summary


# --------------------------------------------------------------------------- #
# Aggregation + metrics
# --------------------------------------------------------------------------- #


def _group_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ex = [r["exact_margin"] for r in rows]
    sp = [r["sparse_margin"] for r in rows]
    rd = np.array([r["residual_direct"] for r in rows], float)
    ar = np.array([r["abs_residual"] for r in rows], float)
    sign = np.array([r["sign_match"] for r in rows], bool)
    bases = sorted({r["base_case_id"] for r in rows})
    # bootstrap by base_case_id
    rng = np.random.default_rng(0)
    by_base: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_base.setdefault(r["base_case_id"], []).append(i)
    boot_sign, boot_med = [], []
    for _ in range(400):
        pick = rng.choice(bases, size=len(bases), replace=True)
        idx = [i for b in pick for i in by_base[b]]
        boot_sign.append(sign[idx].mean())
        boot_med.append(np.median(rd[idx]))
    return {
        "n_base_cases": len(bases),
        "n_rows": len(rows),
        "pearson": _pearson(ex, sp),
        "spearman": _spearman(ex, sp),
        "sign_agreement": float(sign.mean()),
        "sign_agreement_ci_lo": float(np.percentile(boot_sign, 2.5)),
        "sign_agreement_ci_hi": float(np.percentile(boot_sign, 97.5)),
        "median_abs_residual": float(np.median(ar)),
        "p90_abs_residual": float(np.percentile(ar, 90)),
        "median_residual_direct": float(np.median(rd)),
        "median_rd_ci_lo": float(np.percentile(boot_med, 2.5)),
        "median_rd_ci_hi": float(np.percentile(boot_med, 97.5)),
        "p90_residual_direct": float(np.percentile(rd, 90)),
        "pass_rd_lt_0.25": float((rd < 0.25).mean()),
        "pass_rd_lt_0.50": float((rd < 0.50).mean()),
        "pass_rd_lt_1.00": float((rd < 1.00).mean()),
        "sign_flips": int((~sign).sum()),
        "tiny_margin_rows": int(sum(r["tiny_margin"] for r in rows)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_bank(path: Path, cap: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if cap is not None:
        # cap by base case to keep bootstrap honest
        seen, out = set(), []
        for r in rows:
            seen.add(r["base_case_id"])
            out.append(r)
            if len(seen) >= cap and r is rows[-1]:
                break
        if cap < len(rows):
            keep_bases = list(dict.fromkeys(r["base_case_id"] for r in rows))[:cap]
            out = [r for r in rows if r["base_case_id"] in set(keep_bases)]
        return out
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--paper-dir", type=Path, default=REPO / "paper/figures")
    ap.add_argument("--models", default="", help="comma list; '' = all")
    ap.add_argument("--operating-points", default="fidelity,strict_budget")
    ap.add_argument(
        "--banks",
        default="",
        help="comma list of banks to run (model_native,curated_ab,case_candidates); '' = all three",
    )
    ap.add_argument(
        "--frontier-denom-floor",
        type=float,
        default=0.5,
        help="denominator floor for frontier_pass: |m_exact-m_sparse|/max(|m_exact|,floor) <= 0.5",
    )
    ap.add_argument(
        "--headline-corpus-summary",
        action="store_true",
        help="compute per-position KL bits/token, top1 agreement and "
        "frontier_pass@0.5 for the model_native (C4) bank and emit corpus_*.csv "
        "(headline-only: skips the per-query margin decompose for model_native)",
    )
    ap.add_argument(
        "--wu-tol",
        type=float,
        default=0.05,
        help="abort a cell if max relative diff between artifact W_U and the "
        "live lm_head weight exceeds this (guards model/artifact mismatch)",
    )
    ap.add_argument("--max-native", type=int, default=500)
    ap.add_argument("--max-curated", type=int, default=320)
    ap.add_argument("--max-cases", type=int, default=40)
    ap.add_argument("--model-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny caps (20/20/10) for Stage-0 validation",
    )
    ap.add_argument(
        "--cells-only",
        action="store_true",
        help="compute per-cell outputs only; skip cross-cell aggregation and "
        "figures (used by array tasks; a dependent job aggregates).",
    )
    args = ap.parse_args()

    if args.dry_run:
        args.max_native, args.max_curated, args.max_cases = 20, 20, 10

    t0 = time.time()
    reg = resolve_registry(args.registry)
    want_models = [m for m in (args.models.split(",") if args.models else reg["models"]) if m]
    want_ops = [o for o in args.operating_points.split(",") if o]

    bank_files = {
        "model_native": (
            "qwen_gemma_result1_model_native_prompts.jsonl",
            args.max_native,
        ),
        "curated_ab": ("qwen_gemma_result1_curated_ab.jsonl", args.max_curated),
        "case_candidates": (
            "qwen_gemma_result1_case_candidates.jsonl",
            args.max_cases,
        ),
    }
    want_banks = [b for b in args.banks.split(",") if b] or list(bank_files)
    bad = [b for b in want_banks if b not in bank_files]
    if bad:
        ap.error(f"unknown bank(s) {bad}; choose from {list(bank_files)}")
    banks = {}
    for bname in want_banks:
        fname, cap = bank_files[bname]
        path = args.bank_dir / fname
        if not path.exists():
            log(f"bank file missing, skipping {bname}: {path}")
            continue
        banks[bname] = load_bank(path, cap)
    if not banks:
        log("no bank files loaded; nothing to do")
        return 1
    log("banks: " + " ".join(f"{b}={len(r)}" for b, r in banks.items()))

    # C4 provenance manifest for the model-native bank, embedded if present.
    bank_manifest = {}
    bmp = args.bank_dir / "qwen_gemma_result1_model_native_prompts_c4_manifest.json"
    if "model_native" in banks and bmp.exists():
        try:
            bank_manifest = json.loads(bmp.read_text())
        except Exception:  # noqa: BLE001
            pass

    args.out_root.mkdir(parents=True, exist_ok=True)
    cells_root = args.out_root / "cells"
    cell_summaries = []
    for mname in want_models:
        if mname not in reg["models"]:
            log(f"skip unknown model {mname}")
            continue
        me = reg["models"][mname]
        for op_name in want_ops:
            if op_name not in me["operating_points"]:
                log(f"skip {mname}: no operating point {op_name}")
                continue
            op = me["operating_points"][op_name]
            if not Path(op["checkpoint"]).exists():
                log(f"skip {mname}/{op_name}: checkpoint missing {op['checkpoint']}")
                continue
            cd = cells_root / f"{mname}__{op_name}"
            try:
                cell_summaries.append(run_cell(mname, op_name, me, op, banks, args, cd))
            except Exception as e:  # noqa: BLE001
                import traceback

                log(f"CELL FAILED {mname}/{op_name}: {e}\n{traceback.format_exc()}")

    if args.cells_only:
        log(f"cells-only: {len(cell_summaries)} cell(s) done, skipping aggregate")
        print(json.dumps({"cells_only": True, "cells": len(cell_summaries)}, indent=2))
        return 0

    # ---- aggregate all completed cells ----
    all_rows, all_audit, all_skip, all_corpus = [], [], [], []
    for cd in sorted(cells_root.glob("*")):
        rf = cd / "rows.jsonl"
        if rf.exists():
            all_rows += [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
        af = cd / "tokenizer_audit.jsonl"
        if af.exists():
            all_audit += [json.loads(l) for l in af.read_text().splitlines() if l.strip()]
        sf = cd / "skipped.jsonl"
        if sf.exists():
            all_skip += [json.loads(l) for l in sf.read_text().splitlines() if l.strip()]
        cf = cd / "corpus_rows.jsonl"
        if cf.exists():
            all_corpus += [json.loads(l) for l in cf.read_text().splitlines() if l.strip()]

    # ---- headline corpus artifacts (C4 model-native): the three metrics ----
    if args.headline_corpus_summary:
        if not all_corpus:
            log("headline mode but no corpus rows produced; aborting")
            return 1
        _write_csv(args.out_root / "corpus_position_rows.csv", all_corpus)
        _write_csv(
            args.out_root / "corpus_frontier_margin_rows.csv",
            [
                {
                    kk: r[kk]
                    for kk in (
                        "model",
                        "operating_point",
                        "base_case_id",
                        "case_id",
                        "m_exact",
                        "m_sparse",
                        "frontier_sign_match",
                        "frontier_rel",
                        "frontier_pass",
                    )
                }
                for r in all_corpus
            ],
        )
        overall = {
            "model": "ALL",
            "operating_point": "ALL",
            **_corpus_metrics(all_corpus),
        }
        _write_csv(args.out_root / "corpus_headline_summary.csv", [overall])

        def _corpus_grouped(name, key):
            out = []
            for gkey, gr in sorted(_by(all_corpus, key).items(), key=lambda x: str(x[0])):
                m = {**dict(zip(key, gkey))} if isinstance(key, tuple) else {key: gkey}
                out.append({**m, **_corpus_metrics(gr)})
            _write_csv(args.out_root / name, out)
            return out

        _corpus_grouped("corpus_headline_by_model.csv", ("model", "operating_point"))
        _corpus_grouped("corpus_headline_by_operating_point.csv", "operating_point")

        git_hash = None
        try:
            import subprocess

            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
        except Exception:  # noqa: BLE001
            pass
        corpus_manifest = {
            "command": " ".join(sys.argv),
            "args": {k: str(v) for k, v in vars(args).items()},
            "registry": str(args.registry),
            "archive_root": reg["archive_root"],
            "git_hash": git_hash,
            "c4_bank_manifest": bank_manifest,
            "frontier_denom_floor": args.frontier_denom_floor,
            "wu_tol": args.wu_tol,
            "headline_overall": overall,
            "cells": cell_summaries,
            "n_positions": len(all_corpus),
            "runtime_seconds": time.time() - t0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        _atomic_write(
            args.out_root / "manifest.json",
            json.dumps(corpus_manifest, indent=2, default=str),
        )
        log(
            f"HEADLINE n={overall['n_positions']} "
            f"KL={overall['kl_bits_mean']:.4f} bits "
            f"top1={overall['top1_agreement']:.3f} "
            f"frontier@0.5={overall['frontier_pass_at_0.5']:.3f} "
            f"-> {args.out_root}"
        )
        print(json.dumps(overall, indent=2))
        if not all_rows:
            return 0

    if not all_rows:
        log("no margin rows produced; aborting margin aggregation")
        return 0 if args.headline_corpus_summary else 1

    _write_csv(args.out_root / "query_fidelity_rows.csv", all_rows)
    _write_csv(args.out_root / "tokenizer_audit.csv", all_audit)
    _write_csv(args.out_root / "skipped_rows.csv", all_skip)

    feat_rows, cov_rows = [], []
    for r in all_rows:
        cov_rows.append(
            {
                kk: r[kk]
                for kk in (
                    "model",
                    "operating_point",
                    "bank",
                    "family",
                    "case_id",
                    "top5_signed_cov",
                    "top10_signed_cov",
                    "top5_abs_cov",
                    "top10_abs_cov",
                    "n_feat_80pct_abs",
                    "largest_pos_feat",
                    "largest_pos_contrib",
                    "largest_neg_feat",
                    "largest_neg_contrib",
                )
            }
        )
        feat_rows.append(
            {
                kk: r[kk]
                for kk in (
                    "model",
                    "operating_point",
                    "case_id",
                    "query",
                    "largest_pos_feat",
                    "largest_pos_contrib",
                    "largest_neg_feat",
                    "largest_neg_contrib",
                )
            }
        )
    _write_csv(args.out_root / "query_feature_coverage.csv", cov_rows)
    _write_csv(args.out_root / "query_feature_contributions.csv", feat_rows)

    def grouped_csv(name, key):
        out = []
        for gkey, gr in sorted(_by(all_rows, key).items(), key=lambda x: str(x[0])):
            m = _group_metrics(gr)
            if isinstance(key, tuple):
                m = {**dict(zip(key, gkey)), **m}
            else:
                m = {key: gkey, **m}
            out.append(m)
        _write_csv(args.out_root / name, out)
        return out

    summary_overall = _group_metrics(all_rows)
    _write_csv(args.out_root / "query_fidelity_summary.csv", [summary_overall])
    grouped_csv("query_fidelity_by_model.csv", "model")
    grouped_csv("query_fidelity_by_operating_point.csv", "operating_point")
    grouped_csv("query_fidelity_by_bank.csv", "bank")
    by_family = grouped_csv("query_fidelity_by_family.csv", "family")
    grouped_csv("query_fidelity_by_margin_bin.csv", "margin_bin")
    grouped_csv("query_fidelity_appendix_family.csv", ("model", "operating_point", "family"))

    sign_flips = [r for r in all_rows if not r["sign_match"]]
    faithful_wrong = [
        r for r in all_rows if r["sign_match"] and r.get("expected_side") == "A" and r["exact_margin"] < 0
    ]
    _write_csv(args.out_root / "sign_flip_examples.csv", sign_flips[:200])
    _write_csv(args.out_root / "faithful_model_wrong_examples.csv", faithful_wrong[:200])

    paper_dir = args.paper_dir
    paper_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        paper_dir / "result1_query_fidelity_five_model_summary.csv",
        grouped_csv("query_fidelity_by_model.csv", "model"),
    )

    manifest = {
        "command": " ".join(sys.argv),
        "args": {k: str(v) for k, v in vars(args).items()},
        "registry": str(args.registry),
        "archive_root": reg["archive_root"],
        "cells": cell_summaries,
        "n_rows": len(all_rows),
        "n_skipped": len(all_skip),
        "runtime_seconds": time.time() - t0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_write(args.out_root / "manifest.json", json.dumps(manifest, indent=2, default=str))

    lines = [
        "# Result 1 Query-Fidelity Summary",
        "",
        f"rows={len(all_rows)} skipped={len(all_skip)} runtime={manifest['runtime_seconds']:.0f}s",
        "",
        "## Overall",
        f"- sign agreement: {summary_overall['sign_agreement']:.3f} "
        f"[{summary_overall['sign_agreement_ci_lo']:.3f}, "
        f"{summary_overall['sign_agreement_ci_hi']:.3f}]",
        f"- Pearson: {summary_overall['pearson']:.3f}  Spearman: {summary_overall['spearman']:.3f}",
        f"- median residual/direct: "
        f"{summary_overall['median_residual_direct']:.3f}  "
        f"p90: {summary_overall['p90_residual_direct']:.3f}",
        f"- pass<0.5: {summary_overall['pass_rd_lt_0.50']:.3f}",
        "",
        "## By family",
    ]
    for f in by_family:
        lines.append(
            f"- {f['family']}: sign {f['sign_agreement']:.3f}, "
            f"med rd {f['median_residual_direct']:.3f}, "
            f"n {f['n_rows']}"
        )
    _atomic_write(args.out_root / "summary.md", "\n".join(lines))

    log(f"DONE rows={len(all_rows)} -> {args.out_root}")
    print(
        json.dumps(
            {
                "out_root": str(args.out_root),
                "n_rows": len(all_rows),
                "sign_agreement": summary_overall["sign_agreement"],
                "median_rd": summary_overall["median_residual_direct"],
                "seconds": manifest["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
