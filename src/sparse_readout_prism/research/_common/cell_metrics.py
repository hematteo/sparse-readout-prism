"""Per-cell feature-coverage metrics shared by the five-model run scripts.

Extracted from the byte-identical ``coverage_stats`` that
``scripts/run/run_query_fidelity_bank.py`` and
``scripts/run/run_readout_baseline_comparisons.py`` each carried inline. The
small-margin denominator floor (formerly the module-level ``EPS`` in both
runners) is lifted to the ``eps`` keyword so the helper is self-contained; the
default matches the runners' ``EPS = 1e-6`` exactly, so call sites are
behaviour-preserving.
"""

from __future__ import annotations

import torch


def coverage_stats(feat_margin: torch.Tensor, sparse_margin: float, *, eps: float = 1e-6) -> dict:
    fm = feat_margin
    order = torch.argsort(fm.abs(), descending=True)
    total_abs = fm.abs().sum().item()
    out: dict = {}
    for kk in (5, 10):
        idx = order[:kk]
        out[f"top{kk}_signed_cov"] = fm[idx].sum().item() / sparse_margin if abs(sparse_margin) > eps else 0.0
        out[f"top{kk}_abs_cov"] = fm[idx].abs().sum().item() / total_abs if total_abs > eps else 0.0
    if total_abs > eps:
        sorted_abs = fm.abs()[order]
        csum = torch.cumsum(sorted_abs, 0) / total_abs
        out["n_feat_80pct_abs"] = int((csum < 0.80).sum().item()) + 1
    else:
        out["n_feat_80pct_abs"] = 0
    pos_i = int(torch.argmax(fm).item())
    neg_i = int(torch.argmin(fm).item())
    out["largest_pos_feat"] = pos_i
    out["largest_pos_contrib"] = float(fm[pos_i].item())
    out["largest_neg_feat"] = neg_i
    out["largest_neg_contrib"] = float(fm[neg_i].item())
    return out
