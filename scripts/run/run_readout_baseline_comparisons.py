#!/usr/bin/env python3
"""Baseline and reference comparisons for Sparse Readout Prism query fidelity.

Implements the baseline / reference-comparison experiment. For one model and
one operating point at a time, evaluates a panel of baseline methods on the
same Result-1 query banks (model-native, curated A/B, case candidates) using
the same accounting metrics (rho, sign agreement, accepted rate, top5_cov,
n80) and the same single-token tokenization audit as
`run_query_fidelity_bank.py`.

Methods evaluated:
    sparse_rp                       - main method (anchor)
    shuffled_row_code               - sparsity-preserving null
    random_support_same_magnitudes  - support-randomization null
    pca_64, pca_256, pca_1024       - dense PCA references
    nearest_row_ridge_top128        - lexical / token-prototype reference

Output schema:
    cells/<bank>__<method>/{rows.jsonl, summary_cell.json, done.json}
    baseline_query_rows.csv
    baseline_summary.csv
    baseline_by_method.csv
    baseline_by_method_family.csv
    baseline_by_method_margin_bin.csv
    baseline_feature_compactness.csv
    manifest.json

Designed for one-shot resumable execution on a single A40: per-cell `done.json`
sentinel means restart after preemption only re-runs the unfinished
(bank, method) cells. Hidden states are recomputed per cell so each cell is
self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from sparse_readout_prism.data import resolve_row_mean
from sparse_readout_prism.decompose import decompose_token_logit
from sparse_readout_prism.factorizers import TopKSAE, load_factorizer
from sparse_readout_prism.research.cell_metrics import coverage_stats
from sparse_readout_prism.research.run_io import (
    group_rows as _by,
    load_bank,
    margin_row_stats,
    resolve_ab_case,
    run_provenance,
    write_rows_csv as _write_csv,
)
from sparse_readout_prism.research.registry import resolve_registry
from sparse_readout_prism.utils import (
    atomic_write_text as _atomic_write,
    find_lm_head,
    load_causal_lm,
    pearson as _pearson,
    spearman as _spearman,
    write_jsonl as _write_jsonl,
)

# Repo root from this file's location: scripts/run/THIS_FILE.py
#   parents[0]=run/ [1]=scripts/ [2]=repo root
REPO = Path(__file__).resolve().parents[2]


# =========================================================================== #
# Baseline / reference methods (inlined from the former
# research/run/readout_baselines.py — single consumer, so kept self-contained).
#
# Each method exposes a uniform `decompose_row` API so the panel below can plug a
# different reconstruction scheme into the same per-row accounting pipeline. All
# methods share the preprocessing:  centered = W_U[t] - row_mean;  row_norm =
# ||centered||;  s_approx = base + sum_i contributions[i]  with  base = h @ row_mean.
# =========================================================================== #


# --------------------------------------------------------------------------- #
# Decomposition record
# --------------------------------------------------------------------------- #


@dataclass
class BaselineDecomposition:
    """Mirror of `sparse_readout_prism.decompose.Decomposition` plus a method tag.

    `feature_contributions` is a 1-D tensor of signed contributions whose
    length depends on the method:
      * sparse_rp / shuffled_row_code / random_support: d_features (mostly zero)
      * pca_<n>:           n_components
      * nearest_row_ridge_top<k>: k
    """

    original_logit: torch.Tensor
    base_term: torch.Tensor
    feature_contributions: torch.Tensor
    feature_sum: torch.Tensor
    residual_term: torch.Tensor
    reconstructed_logit: torch.Tensor
    identity_error: torch.Tensor
    active_feature_indices: torch.Tensor
    method: str


# --------------------------------------------------------------------------- #
# Method base
# --------------------------------------------------------------------------- #


class BaselineMethod:
    """Subclass and implement `decompose_row`."""

    name: str = "base"

    def __init__(
        self,
        *,
        W: torch.Tensor,  # (V, d_model) cpu fp32
        row_mean: torch.Tensor,  # (d_model,)
        device: torch.device,
    ) -> None:
        # Caller may pass W on CPU and row_mean on either device; align both
        # explicitly so the centered subtraction does not crash with a
        # cross-device error. We keep `self.W` on CPU and only materialise the
        # centered/normalised rows on `device` (~V*d_model*4 bytes).
        self.W = W
        self.device = device
        self.row_mean = row_mean.to(device)
        centered = W.to(device) - self.row_mean
        self.row_norm_all = centered.norm(dim=1).clamp_min(1e-8)
        self.row_normalized_all = centered / self.row_norm_all[:, None]

    # ---- subclasses fill these in -----------------------------------------

    def decompose_row(
        self,
        h: torch.Tensor,  # (d_model,) on self.device
        row_idx: int,  # row index into W
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> BaselineDecomposition:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# A. Sparse Readout Prism anchor
# --------------------------------------------------------------------------- #


class SparseRPMethod(BaselineMethod):
    name = "sparse_rp"

    def __init__(self, *, sae: TopKSAE, k: int, **kw) -> None:
        super().__init__(**kw)
        self.sae = sae.to(self.device).eval()
        for p in self.sae.parameters():
            p.requires_grad_(False)
        self.k = int(k)

    @torch.no_grad()
    def decompose_row(
        self,
        h: torch.Tensor,
        row_idx: int,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> BaselineDecomposition:
        W_row = self.W[row_idx].to(self.device)
        d = decompose_token_logit(
            h,
            W_row,
            self.row_mean,
            self.row_norm_all[row_idx],
            self.row_normalized_all[row_idx],
            self.sae,
            self.k,
        )
        return BaselineDecomposition(
            original_logit=d.original_logit,
            base_term=d.base_term,
            feature_contributions=d.feature_contributions,
            feature_sum=d.feature_sum,
            residual_term=d.residual_term,
            reconstructed_logit=d.reconstructed_logit,
            identity_error=d.identity_error,
            active_feature_indices=d.active_feature_indices,
            method=self.name,
        )


# --------------------------------------------------------------------------- #
# B. Shuffled row-code control
# --------------------------------------------------------------------------- #


class ShuffledRowCodeMethod(BaselineMethod):
    """Use the learned SAE decoder, but assign each row the sparse code that
    the encoder produced for a different (permuted) row.

    Sparsity, code distribution and decoder geometry are preserved; only the
    row -> code-row assignment is randomized.
    """

    name = "shuffled_row_code"

    def __init__(self, *, sae: TopKSAE, k: int, seed: int, **kw) -> None:
        super().__init__(**kw)
        self.sae = sae.to(self.device).eval()
        for p in self.sae.parameters():
            p.requires_grad_(False)
        self.k = int(k)
        V = self.W.shape[0]

        # Precompute per-row top-k (indices, values) once.
        # Storing the full (V, d_features) dense code would be ~40 GB for
        # Qwen2B 32x; (V, k) sparse representation is ~470 MB.
        idxs = torch.empty((V, self.k), dtype=torch.long, device=self.device)
        vals = torch.empty((V, self.k), dtype=torch.float32, device=self.device)
        chunk = 4096
        with torch.no_grad():
            for s in range(0, V, chunk):
                xb = self.row_normalized_all[s : s + chunk]
                code = self.sae.encode(xb, k=self.k)
                v, i = torch.topk(code, k=self.k, dim=-1)
                idxs[s : s + chunk] = i
                vals[s : s + chunk] = v
        self.code_indices_per_row = idxs  # (V, k)
        self.code_values_per_row = vals  # (V, k)

        # Permutation seeded for reproducibility.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(V, generator=gen).to(self.device)
        # Self-permutation could leak the true code; we don't enforce a
        # derangement (cycle expectation 1/V is fine at this scale) but log it.
        n_fixed = int((perm == torch.arange(V, device=self.device)).sum().item())
        self.permutation = perm
        self.n_fixed_points = n_fixed

    @torch.no_grad()
    def decompose_row(
        self,
        h: torch.Tensor,
        row_idx: int,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> BaselineDecomposition:
        # Use the code of the permuted row, scaled by THIS row's row_norm.
        src = int(self.permutation[row_idx].item())
        idx = self.code_indices_per_row[src]  # (k,)
        val = self.code_values_per_row[src]  # (k,)
        decoder_rows = self.sae.decoder[idx]  # (k, d_model)
        decoded = (val[:, None] * decoder_rows).sum(0)  # (d_model,)
        W_row = self.W[row_idx].to(self.device)
        row_norm = self.row_norm_all[row_idx]
        reconstructed_row = self.row_mean + row_norm * decoded
        residual_row = W_row - reconstructed_row

        d_features = self.sae.d_features
        feature_contributions = torch.zeros(d_features, device=self.device)
        h_dec = h @ decoder_rows.T  # (k,)
        contribs_k = row_norm * val * h_dec  # (k,)
        feature_contributions.scatter_add_(0, idx, contribs_k)

        base_term = h @ self.row_mean
        feature_sum = contribs_k.sum()
        residual_term = h @ residual_row
        original_logit = h @ W_row
        reconstructed_logit = base_term + feature_sum
        identity_error = original_logit - (base_term + feature_sum + residual_term)
        return BaselineDecomposition(
            original_logit=original_logit,
            base_term=base_term,
            feature_contributions=feature_contributions,
            feature_sum=feature_sum,
            residual_term=residual_term,
            reconstructed_logit=reconstructed_logit,
            identity_error=identity_error,
            active_feature_indices=idx.detach().cpu(),
            method=self.name,
        )


# --------------------------------------------------------------------------- #
# C. Random active-support control (same magnitudes)
# --------------------------------------------------------------------------- #


class RandomSupportSameMagnitudesMethod(BaselineMethod):
    """Per row, keep the same sorted code magnitudes and signs the SAE
    produced, but place them at random feature ids (sampled without
    replacement from {0..d_features-1}).

    Row_norm and the row_mean base term are preserved.
    """

    name = "random_support_same_magnitudes"

    def __init__(self, *, sae: TopKSAE, k: int, seed: int, **kw) -> None:
        super().__init__(**kw)
        self.sae = sae.to(self.device).eval()
        for p in self.sae.parameters():
            p.requires_grad_(False)
        self.k = int(k)
        self.seed = int(seed)
        self.d_features = self.sae.d_features

        # Cache per-row sorted-by-|val| code magnitudes (with sign).
        V = self.W.shape[0]
        vals_sorted = torch.empty((V, self.k), dtype=torch.float32, device=self.device)
        chunk = 4096
        with torch.no_grad():
            for s in range(0, V, chunk):
                xb = self.row_normalized_all[s : s + chunk]
                code = self.sae.encode(xb, k=self.k)
                # top-k magnitudes (ReLU codes are non-negative, so signed == abs)
                v_signed, _ = torch.topk(code, k=self.k, dim=-1)
                vals_sorted[s : s + chunk] = v_signed
        self.code_vals_sorted = vals_sorted  # (V, k) -- magnitudes from SAE

    @torch.no_grad()
    def decompose_row(
        self,
        h: torch.Tensor,
        row_idx: int,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> BaselineDecomposition:
        # Seed a per-row generator: (global_seed, row_idx) -> reproducible
        # random support for that row.
        gen = torch.Generator(device="cpu").manual_seed((self.seed * 0x9E3779B9 + int(row_idx)) & 0x7FFFFFFF)
        rand_ids = torch.randperm(self.d_features, generator=gen)[: self.k].to(self.device)
        val = self.code_vals_sorted[row_idx]  # (k,) signed magnitudes
        decoder_rows = self.sae.decoder[rand_ids]  # (k, d_model)
        decoded = (val[:, None] * decoder_rows).sum(0)  # (d_model,)
        W_row = self.W[row_idx].to(self.device)
        row_norm = self.row_norm_all[row_idx]
        reconstructed_row = self.row_mean + row_norm * decoded
        residual_row = W_row - reconstructed_row

        feature_contributions = torch.zeros(self.d_features, device=self.device)
        h_dec = h @ decoder_rows.T
        contribs_k = row_norm * val * h_dec
        feature_contributions.scatter_add_(0, rand_ids, contribs_k)

        base_term = h @ self.row_mean
        feature_sum = contribs_k.sum()
        residual_term = h @ residual_row
        original_logit = h @ W_row
        reconstructed_logit = base_term + feature_sum
        identity_error = original_logit - (base_term + feature_sum + residual_term)
        return BaselineDecomposition(
            original_logit=original_logit,
            base_term=base_term,
            feature_contributions=feature_contributions,
            feature_sum=feature_sum,
            residual_term=residual_term,
            reconstructed_logit=reconstructed_logit,
            identity_error=identity_error,
            active_feature_indices=rand_ids.detach().cpu(),
            method=self.name,
        )


# --------------------------------------------------------------------------- #
# D. Dense PCA reference
# --------------------------------------------------------------------------- #


class DensePCAMethod(BaselineMethod):
    """PCA basis fit on the same `row_normalized` rows the SAE was trained on.

    Reconstruction = row_mean + row_norm * (rows_normalized @ V_pca.T) @ V_pca.
    Feature contributions report per-component signed scores so the runner can
    audit compactness on this method as it would on a sparse method, though
    we do not promote `top5_cov` / `n80` for PCA in the paper table.
    """

    name = "pca"  # actual instance name is set per n_components

    def __init__(self, *, n_components: int, **kw) -> None:
        super().__init__(**kw)
        self.n_components = int(n_components)
        self.name = f"pca_{int(n_components)}"

        # SVD of (V, d_model) row-normalized matrix. Since d_model ~= 2048 for
        # Qwen2B and V ~= 152k, the right singular vectors V_pca are (n_comp,
        # d_model) -- we always have n_components <= d_model.
        with torch.no_grad():
            X = self.row_normalized_all  # (V, d_model)
            # svd_lowrank: q must be slightly above n_components for accuracy.
            q = min(self.n_components + 16, min(X.shape) - 1)
            U, S, Vt = torch.svd_lowrank(X, q=q, niter=4)
            self.V_pca = Vt[:, : self.n_components].T.contiguous()  # (n_comp, d_model)

    @torch.no_grad()
    def decompose_row(
        self,
        h: torch.Tensor,
        row_idx: int,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> BaselineDecomposition:
        x = self.row_normalized_all[row_idx]  # (d_model,)
        coeffs = self.V_pca @ x  # (n_comp,)
        decoded = coeffs @ self.V_pca  # (d_model,)
        W_row = self.W[row_idx].to(self.device)
        row_norm = self.row_norm_all[row_idx]
        reconstructed_row = self.row_mean + row_norm * decoded
        residual_row = W_row - reconstructed_row

        h_pca = self.V_pca @ h  # (n_comp,)
        feature_contributions = row_norm * coeffs * h_pca  # (n_comp,)
        base_term = h @ self.row_mean
        feature_sum = feature_contributions.sum()
        residual_term = h @ residual_row
        original_logit = h @ W_row
        reconstructed_logit = base_term + feature_sum
        identity_error = original_logit - (base_term + feature_sum + residual_term)
        active = torch.arange(self.n_components)  # all components active
        return BaselineDecomposition(
            original_logit=original_logit,
            base_term=base_term,
            feature_contributions=feature_contributions,
            feature_sum=feature_sum,
            residual_term=residual_term,
            reconstructed_logit=reconstructed_logit,
            identity_error=identity_error,
            active_feature_indices=active,
            method=self.name,
        )


# --------------------------------------------------------------------------- #
# E. Nearest-row ridge reference
# --------------------------------------------------------------------------- #


class NearestRowRidgeMethod(BaselineMethod):
    """For each target row, pick the top-K nearest centered-and-normalized
    vocabulary rows by cosine (excluding self and any contrast ids supplied
    by the caller), then fit a signed ridge regression to reconstruct the
    target row from those neighbours.

    `top_k` controls how many neighbours we fit over. The ridge regulariser
    `lam` is tiny by default; we keep it nonzero to ensure invertibility
    when neighbours are highly collinear.
    """

    name = "nearest_row_ridge"

    def __init__(self, *, top_k: int = 128, lam: float = 1e-3, **kw) -> None:
        super().__init__(**kw)
        self.top_k = int(top_k)
        self.lam = float(lam)
        self.name = f"nearest_row_ridge_top{self.top_k}"

    @torch.no_grad()
    def decompose_row(
        self,
        h: torch.Tensor,
        row_idx: int,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> BaselineDecomposition:
        x = self.row_normalized_all[row_idx]  # (d_model,)
        # Cosine sim: rows are already unit-norm centered.
        sims = self.row_normalized_all @ x  # (V,)
        # Exclude self + any contrast ids.
        excl = {int(row_idx)}
        if exclude_ids is not None:
            excl.update(int(i) for i in exclude_ids)
        if excl:
            mask = torch.zeros_like(sims, dtype=torch.bool)
            mask[list(excl)] = True
            sims = sims.masked_fill(mask, -float("inf"))
        nbr_ids = torch.topk(sims, k=self.top_k).indices  # (top_k,)
        X = self.row_normalized_all[nbr_ids]  # (top_k, d_model)
        # Ridge: beta = (X X^T + lam I)^-1 X x   -- closed form, top_k x top_k solve.
        XXT = X @ X.T  # (top_k, top_k)
        Xx = X @ x  # (top_k,)
        A = XXT + self.lam * torch.eye(self.top_k, device=self.device)
        beta = torch.linalg.solve(A, Xx)  # (top_k,)
        decoded = beta @ X  # (d_model,)
        W_row = self.W[row_idx].to(self.device)
        row_norm = self.row_norm_all[row_idx]
        reconstructed_row = self.row_mean + row_norm * decoded
        residual_row = W_row - reconstructed_row

        # Per-neighbour contribution to h. The neighbour basis is normalized;
        # contribution_j = beta_j * row_norm * (h . X_j).
        h_nbr = X @ h  # (top_k,)
        feature_contributions = row_norm * beta * h_nbr  # (top_k,)
        base_term = h @ self.row_mean
        feature_sum = feature_contributions.sum()
        residual_term = h @ residual_row
        original_logit = h @ W_row
        reconstructed_logit = base_term + feature_sum
        identity_error = original_logit - (base_term + feature_sum + residual_term)
        return BaselineDecomposition(
            original_logit=original_logit,
            base_term=base_term,
            feature_contributions=feature_contributions,
            feature_sum=feature_sum,
            residual_term=residual_term,
            reconstructed_logit=reconstructed_logit,
            identity_error=identity_error,
            active_feature_indices=nbr_ids.detach().cpu(),
            method=self.name,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_method(
    spec: str,
    *,
    W: torch.Tensor,
    row_mean: torch.Tensor,
    device: torch.device,
    sae: Optional[TopKSAE] = None,
    k: Optional[int] = None,
    seed: int = 0,
) -> BaselineMethod:
    """spec examples:
    'sparse_rp'
    'shuffled_row_code'
    'random_support_same_magnitudes'
    'pca_64', 'pca_256', 'pca_1024'
    'nearest_row_ridge_top128'
    """
    common = dict(W=W, row_mean=row_mean, device=device)
    if spec == "sparse_rp":
        if sae is None or k is None:
            raise ValueError("sparse_rp requires sae and k")
        return SparseRPMethod(sae=sae, k=k, **common)
    if spec == "shuffled_row_code":
        if sae is None or k is None:
            raise ValueError("shuffled_row_code requires sae and k")
        return ShuffledRowCodeMethod(sae=sae, k=k, seed=seed, **common)
    if spec == "random_support_same_magnitudes":
        if sae is None or k is None:
            raise ValueError("random_support_same_magnitudes requires sae and k")
        return RandomSupportSameMagnitudesMethod(sae=sae, k=k, seed=seed, **common)
    if spec.startswith("pca_"):
        n = int(spec.split("_", 1)[1])
        return DensePCAMethod(n_components=n, **common)
    if spec.startswith("nearest_row_ridge_top"):
        top = int(spec.split("nearest_row_ridge_top", 1)[1])
        return NearestRowRidgeMethod(top_k=top, **common)
    raise ValueError(f"unknown method spec: {spec}")


# --------------------------------------------------------------------------- #
# Margin (mirror of `margin_from_rows` in run_query_fidelity_bank.py)
# --------------------------------------------------------------------------- #


def margin_from_rows(
    h: torch.Tensor,
    method: BaselineMethod,
    a_ids: Optional[list[int]],
    b_ids: Optional[list[int]],
    mean_row: Optional[torch.Tensor] = None,
    exclude_contrast: bool = True,
) -> dict:
    """Decompose A and B with `method` and return margin + per-feature contribs.

    `mean_row` is used for the pseudo-target model-native `top1_vocabmean`
    query (A=top1, B=row_mean). In that case `b_ids` is None and the B
    decompose is replaced by `(h @ row_mean, h @ row_mean, 0, zeros)` so
    the base cancels exactly (the prism-side `s_approx_B = base + sum_i 0`).
    """

    @torch.no_grad()
    def _agg(ids: Optional[list[int]], pseudo: Optional[torch.Tensor], excl):
        if pseudo is not None:
            # Pseudo-target row equal to row_mean: original = base, sparse =
            # base + 0, residual = 0, feat = 0.
            base = h @ method.row_mean
            feat = torch.zeros(1, device=h.device)
            # Use a length-matched zero vector so coverage stats are well-defined.
            return base, base, torch.zeros((), device=h.device), feat * 0.0
        accs = None
        for tid in ids:
            other = set(ids) - {int(tid)}
            if excl:
                other.update(int(x) for x in (excl or []))
            d = method.decompose_row(h, int(tid), exclude_ids=other)
            cur = (
                d.original_logit,
                d.reconstructed_logit,
                d.residual_term,
                d.feature_contributions,
            )
            if accs is None:
                accs = cur
            else:
                accs = tuple(a + b for a, b in zip(accs, cur))
        n = float(len(ids))
        return tuple(a / n for a in accs)

    excl_a = list(b_ids) if (exclude_contrast and b_ids) else []
    excl_b = list(a_ids) if (exclude_contrast and a_ids) else []
    oA, sA, rA, fA = _agg(a_ids, mean_row if a_ids is None else None, excl_a)
    oB, sB, rB, fB = _agg(b_ids, mean_row if b_ids is None else None, excl_b)
    exact = float((oA - oB).detach())
    sparse = float((sA - sB).detach())
    # Pad the shorter of fA / fB with zeros so subtraction is well-defined
    # (pseudo-target B has length-1 zero tensor; A may have longer features).
    if fA.shape != fB.shape:
        if fA.numel() == 1 and float(fA.abs().sum().item()) == 0.0:
            fA = torch.zeros_like(fB)
        elif fB.numel() == 1 and float(fB.abs().sum().item()) == 0.0:
            fB = torch.zeros_like(fA)
        else:
            raise RuntimeError(f"feature_contribution shape mismatch: A={tuple(fA.shape)} B={tuple(fB.shape)}")
    feat_margin = (fA - fB).detach().cpu()
    return {
        "exact": exact,
        "sparse": sparse,
        "residual": exact - sparse,
        "resid_term": float((rA - rB).detach()),
        "feat_margin": feat_margin,
    }


# Method panel for Stages 1+2 (Qwen2B). All seven share the same row budget
# and run in the order listed; sparse_rp first so its precomputed h cache
# warms the model forward path for the rest.
DEFAULT_METHODS = (
    "sparse_rp",
    "shuffled_row_code",
    "random_support_same_magnitudes",
    "pca_64",
    "pca_256",
    "pca_1024",
    "nearest_row_ridge_top128",
)


def log(msg: str) -> None:
    print(f"[rbase {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _group_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ex = [r["exact_margin"] for r in rows]
    sp = [r["sparse_margin"] for r in rows]
    rd = np.array([r["residual_direct"] for r in rows], float)
    ar = np.array([r["abs_residual"] for r in rows], float)
    sign = np.array([r["sign_match"] for r in rows], bool)
    bases = sorted({r["base_case_id"] for r in rows})
    rng = np.random.default_rng(0)
    by_base: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_base.setdefault(r["base_case_id"], []).append(i)
    boot_sign, boot_med, boot_acc = [], [], []
    accepted = sign & (rd < 0.5)
    for _ in range(400):
        pick = rng.choice(bases, size=len(bases), replace=True)
        idx = [i for b in pick for i in by_base[b]]
        boot_sign.append(sign[idx].mean())
        boot_med.append(np.median(rd[idx]))
        boot_acc.append(accepted[idx].mean())
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
        "accepted_rate": float(accepted.mean()),
        "accepted_ci_lo": float(np.percentile(boot_acc, 2.5)),
        "accepted_ci_hi": float(np.percentile(boot_acc, 97.5)),
        "sign_flips": int((~sign).sum()),
        "tiny_margin_rows": int(sum(r["tiny_margin"] for r in rows)),
    }


# --------------------------------------------------------------------------- #
# Model-native frontier query construction
# --------------------------------------------------------------------------- #


def native_queries_top1(logits: torch.Tensor):
    """Yield (qname, a_id, b_id, b_is_mean) tuples for model-native rows.

    Deliberately REDUCED relative to ``run_query_fidelity_bank.native_queries``
    (five queries): the baseline comparison only needs the two cheap anchors,
    * top1_vs_top2: A = argmax logit, B = second-argmax
    * top1_vs_vocabmean: A = argmax, B = row_mean (pseudo-target)
    hence the distinct name.
    """
    vals, idx = torch.topk(logits, k=2)
    a_id = int(idx[0].item())
    b_id = int(idx[1].item())
    yield ("top1_vs_top2", a_id, b_id, False)
    yield ("top1_vs_vocabmean", a_id, -1, True)


# --------------------------------------------------------------------------- #
# Per-cell evaluation
# --------------------------------------------------------------------------- #


def run_cell(
    model_name: str,
    op_name: str,
    method_spec: str,
    bank_name: str,
    bank_rows: list[dict],
    *,
    m_entry: dict,
    op: dict,
    args,
    cell_dir: Path,
    cache: dict,
):
    """Run one (model, op, method, bank) cell. `cache` carries the loaded
    artefacts (W, row_mean, sae, k, model, tok, h_cache) so we don't reload
    them for every method.
    """
    done = cell_dir / "done.json"
    if done.exists():
        log(f"[resume] cell {bank_name}/{method_spec} already done")
        return json.loads((cell_dir / "summary_cell.json").read_text())
    cell_dir.mkdir(parents=True, exist_ok=True)

    W = cache["W"]
    row_mean = cache["row_mean"]
    sae = cache["sae"]
    k = cache["k"]
    tok = cache["tok"]
    hidden_for = cache["hidden_for"]
    softcap = cache["softcap"]
    device = cache["device"]

    # Build method (per-cell so PCA/shuffle precomputation is timed honestly
    # against the cell budget). Methods are cheap to re-build except for the
    # SAE all-row encode in shuffled / random-support, which we cache across
    # cells via the `cache["methods"]` dict.
    methods_cache = cache.setdefault("methods", {})
    if method_spec not in methods_cache:
        t0 = time.time()
        methods_cache[method_spec] = build_method(
            method_spec,
            W=W,
            row_mean=row_mean,
            device=device,
            sae=sae,
            k=k,
            seed=args.seed,
        )
        log(f"built method {method_spec} in {time.time() - t0:.1f}s")
    method = methods_cache[method_spec]

    rows: list[dict] = []
    audit: list[dict] = []
    skipped: list[dict] = []

    def emit(base_id, case_id, family, query, mr, expected_side):
        row = {
            "model": model_name,
            "operating_point": op_name,
            "op_id": op["id"],
            "method": method_spec,
            "bank": bank_name,
            "family": family,
            "base_case_id": base_id,
            "case_id": case_id,
            "query": query,
            "k": k,
        }
        row.update(margin_row_stats(mr))
        row["expected_side"] = expected_side
        row["softcap"] = softcap
        row.update(coverage_stats(mr["feat_margin"], mr["sparse"]))
        rows.append(row)

    # Process bank rows.
    if bank_name in ("curated_ab", "case_candidates"):
        for r in bank_rows:
            try:
                h = hidden_for(r["prompt"]).to(device)
            except Exception as e:  # noqa: BLE001
                skipped.append({"case_id": r["case_id"], "reason": f"forward:{e}"})
                continue
            resolved = resolve_ab_case(tok, r, model_name, audit, skipped)
            if resolved is None:
                continue
            a_ids, b_ids, fam_label, query_label = resolved
            mr = margin_from_rows(h, method, a_ids, b_ids)
            emit(
                r["base_case_id"],
                r["case_id"],
                fam_label,
                query_label,
                mr,
                r.get("expected_side"),
            )
    elif bank_name == "model_native":
        for r in bank_rows:
            try:
                h = hidden_for(r["prompt"]).to(device)
            except Exception as e:  # noqa: BLE001
                skipped.append({"case_id": r["case_id"], "reason": f"forward:{e}"})
                continue
            logits = h.cpu() @ W.T
            for qname, a_id, b_id, b_mean in native_queries_top1(logits):
                mr = margin_from_rows(
                    h,
                    method,
                    [a_id],
                    None if b_mean else [b_id],
                    mean_row=row_mean if b_mean else None,
                )
                emit(
                    r["base_case_id"],
                    f"{r['case_id']}__{qname}",
                    f"native_{qname}",
                    qname,
                    mr,
                    None,
                )
    else:
        raise ValueError(f"unknown bank: {bank_name}")

    _write_jsonl(cell_dir / "rows.jsonl", rows)
    _write_jsonl(cell_dir / "tokenizer_audit.jsonl", audit)
    _write_jsonl(cell_dir / "skipped.jsonl", skipped)
    cell_summary = {
        "model": model_name,
        "operating_point": op_name,
        "op_id": op["id"],
        "method": method_spec,
        "bank": bank_name,
        "n_rows": len(rows),
        "n_skipped": len(skipped),
        "n_audit": len(audit),
        "k": k,
        "softcap": softcap,
    }
    _atomic_write(cell_dir / "summary_cell.json", json.dumps(cell_summary, indent=2))
    _atomic_write(done, json.dumps({"done_at": time.time()}))
    log(f"cell {bank_name}/{method_spec}: {len(rows)} rows, {len(skipped)} skipped")
    return cell_summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--model", type=str, default="Qwen3.5-2B")
    ap.add_argument("--operating-point", type=str, default="fidelity")
    ap.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="comma-separated method specs",
    )
    ap.add_argument("--banks", type=str, default="curated_ab,case_candidates,model_native")
    ap.add_argument("--max-native", type=int, default=500)
    ap.add_argument("--max-curated", type=int, default=320)
    ap.add_argument("--max-cases", type=int, default=40)
    ap.add_argument("--model-dtype", type=str, default="bfloat16")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--wu-tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    reg = resolve_registry(args.registry)
    if args.model not in reg["models"]:
        ap.error(f"unknown model {args.model}; have {list(reg['models'])}")
    m_entry = reg["models"][args.model]
    if args.operating_point not in m_entry["operating_points"]:
        ap.error(f"unknown op {args.operating_point}; have {list(m_entry['operating_points'])}")
    op = m_entry["operating_points"][args.operating_point]
    if not Path(op["checkpoint"]).exists():
        ap.error(f"checkpoint missing: {op['checkpoint']}")

    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    log(f"methods: {methods}")

    bank_files = {
        "model_native": (
            "qwen_gemma_result1_model_native_prompts.jsonl",
            args.max_native,
        ),
        "curated_ab": (
            "qwen_gemma_result1_curated_ab.jsonl",
            args.max_curated,
        ),
        "case_candidates": (
            "qwen_gemma_result1_case_candidates.jsonl",
            args.max_cases,
        ),
    }
    want_banks = [b.strip() for b in args.banks.split(",") if b.strip()]
    bad = [b for b in want_banks if b not in bank_files]
    if bad:
        ap.error(f"unknown bank(s) {bad}; have {list(bank_files)}")
    banks = {}
    for bname in want_banks:
        fname, cap = bank_files[bname]
        if args.dry_run:
            cap = min(cap or 0, 20)
        path = args.bank_dir / fname
        if not path.exists():
            log(f"bank file missing, skipping {bname}: {path}")
            continue
        banks[bname] = load_bank(path, cap)
    if not banks:
        log("no banks loaded; nothing to do")
        return 1
    log("banks: " + " ".join(f"{b}={len(r)}" for b, r in banks.items()))

    args.out_root.mkdir(parents=True, exist_ok=True)
    cells_root = args.out_root / "cells"

    # ---- load model / W / SAE ONCE for all cells ----------------------------
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
    cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer")
    if cfg is None:
        raise KeyError(f"no factorizer config in {op['checkpoint']}")
    sae = load_factorizer(ckpt, factorizer_config=cfg, d_model=d_model, freeze=True)
    k = int(ckpt.get("evaluation", {}).get("k") or cfg.get("k"))
    # row_mean matched to training: checkpoint's stored value, else the
    # token_mask-kept mean, else the full-vocab mean (one shared definition).
    row_mean = resolve_row_mean(W, token_mask=token_mask, ckpt=ckpt)
    row_norms_all = (W - row_mean).norm(dim=1)

    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32
    model, tok = load_causal_lm(m_entry["model_id"], revision=m_entry["revision"], dtype=dtype)
    dev = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm_head = find_lm_head(model)
    h_box: dict = {}

    def pre_hook(_m, inp):
        h_box["h"] = (inp[0] if isinstance(inp, tuple) else inp).detach()

    hh = lm_head.register_forward_pre_hook(pre_hook)

    with torch.no_grad():
        wl = lm_head.weight.detach().float().cpu()[:n_vocab]
        wu_rel = float((W - wl).abs().max() / wl.abs().max().clamp_min(1e-9))
    if wu_rel > args.wu_tol:
        hh.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"W_U vs live lm_head mismatch {wu_rel:.4g} > --wu-tol {args.wu_tol} "
            f"for {args.model}/{args.operating_point}"
        )
    log(f"wu_vs_lmhead_rel={wu_rel:.3e}")

    # Cache hidden states per prompt across method cells -- the model is the
    # most expensive resource we hold and prompts are deterministic.
    h_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def hidden_for(prompt: str) -> torch.Tensor:
        if prompt in h_cache:
            return h_cache[prompt]
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(dev)
        # noqa: F821 is a pyflakes false positive here -- `model` is a valid closure
        # variable (bound above); the `del model` below runs only after every
        # hidden_for() call, so the cell is never empty when this executes.
        model(**enc)  # noqa: F821
        am = enc["attention_mask"][0].bool()
        pos = int(am.nonzero()[-1].item())
        h = h_box["h"][0, pos].float().cpu().contiguous()
        h_cache[prompt] = h
        return h

    cache = {
        "W": W,
        "row_mean": row_mean.to(dev) if torch.cuda.is_available() else row_mean,
        "row_norms_all": row_norms_all,
        "sae": sae,
        "k": k,
        "tok": tok,
        "n_vocab": n_vocab,
        "softcap": softcap,
        "device": dev,
        "hidden_for": hidden_for,
    }

    # ---- loop bank x method -------------------------------------------------
    cell_summaries: list[dict] = []
    for bank_name, bank_rows in banks.items():
        for method_spec in methods:
            cd = cells_root / f"{bank_name}__{method_spec}"
            try:
                cs = run_cell(
                    args.model,
                    args.operating_point,
                    method_spec,
                    bank_name,
                    bank_rows,
                    m_entry=m_entry,
                    op=op,
                    args=args,
                    cell_dir=cd,
                    cache=cache,
                )
                cell_summaries.append(cs)
            except Exception as e:  # noqa: BLE001
                import traceback

                log(f"CELL FAILED {bank_name}/{method_spec}: {e}\n{traceback.format_exc()}")

    hh.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- aggregate ----------------------------------------------------------
    all_rows, all_audit, all_skip = [], [], []
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

    if not all_rows:
        log("no rows produced")
        return 1

    _write_csv(args.out_root / "baseline_query_rows.csv", all_rows)
    _write_csv(args.out_root / "tokenizer_audit.csv", all_audit)
    _write_csv(args.out_root / "skipped_rows.csv", all_skip)

    overall = {
        "model": args.model,
        "operating_point": args.operating_point,
        "method": "ALL",
        **_group_metrics(all_rows),
    }
    _write_csv(args.out_root / "baseline_summary.csv", [overall])

    def grouped(name, key):
        out = []
        for gkey, gr in sorted(_by(all_rows, key).items(), key=lambda x: str(x[0])):
            m = {**dict(zip(key, gkey))} if isinstance(key, tuple) else {key: gkey}
            out.append({**m, **_group_metrics(gr)})
        _write_csv(args.out_root / name, out)
        return out

    grouped("baseline_by_method.csv", "method")
    grouped("baseline_by_method_family.csv", ("method", "family"))
    grouped("baseline_by_method_margin_bin.csv", ("method", "margin_bin"))
    grouped("baseline_by_bank_method.csv", ("bank", "method"))

    # Feature compactness: per-method aggregates of top5_abs_cov and n80
    # (means and medians over rows).
    compact = []
    for method_spec in sorted({r["method"] for r in all_rows}):
        mr = [r for r in all_rows if r["method"] == method_spec]
        compact.append(
            {
                "method": method_spec,
                "n_rows": len(mr),
                "top5_abs_cov_mean": float(np.mean([r["top5_abs_cov"] for r in mr])),
                "top5_abs_cov_median": float(np.median([r["top5_abs_cov"] for r in mr])),
                "top10_abs_cov_mean": float(np.mean([r["top10_abs_cov"] for r in mr])),
                "top10_abs_cov_median": float(np.median([r["top10_abs_cov"] for r in mr])),
                "n80_mean": float(np.mean([r["n_feat_80pct_abs"] for r in mr])),
                "n80_median": float(np.median([r["n_feat_80pct_abs"] for r in mr])),
            }
        )
    _write_csv(args.out_root / "baseline_feature_compactness.csv", compact)

    _atomic_write(
        args.out_root / "manifest.json",
        json.dumps(
            {
                **run_provenance(args),
                "registry": str(args.registry),
                "archive_root": reg["archive_root"],
                "model": args.model,
                "operating_point": args.operating_point,
                "methods": methods,
                "banks": list(banks.keys()),
                "wu_vs_lmhead_rel": wu_rel,
                "n_rows": len(all_rows),
                "n_skipped": len(all_skip),
                "runtime_seconds": time.time() - t0,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cell_summaries": cell_summaries,
            },
            indent=2,
            default=str,
        ),
    )

    log(f"DONE n_rows={len(all_rows)} methods={len(methods)} -> {args.out_root}")
    print(json.dumps(overall, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
