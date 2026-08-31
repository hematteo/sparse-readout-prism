"""Correctness tests for scripts/eval/run_causal_contribution_validation.py (loaded by path).

Pins the math core behind ``tab:causal-validation-summary`` on the script's own
synthetic residual-free case (a contrast built exactly in the span of a random
unit-row dictionary from a known 3-sparse beta):

* predicted contributions track the realized changes (r2 ~ 1, slope ~ 1),
* random-feature controls carry zero predicted contribution, so they have no
  explanatory power (the through-origin fit is undefined for them),
* the additive identity holds: the contributions sum to the exact margin, each
  stored realized change equals (h . d_i)(q . d_i), and ablating h along d_i
  moves the margin read off a dense two-row LM head by exactly minus that amount
  (the self-test's third check, measured from the head rather than the formula),
* ``--centering`` selects the live full-vocabulary mean, the checkpoint's
  ``row_mean``, or the tokenizer text-token mean, through ``data.centering_mean``.

CPU only, no model, well under a second.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from conftest import load_script

from sparse_readout_prism.research import row_geometry

causal = load_script("scripts/eval/run_causal_contribution_validation.py")


def test_self_test_passes_and_recovers_unit_agreement() -> None:
    m = causal.self_test_metrics()
    assert all(ok for _, ok in m["checks"]), m["checks"]
    assert len(m["checks"]) == 3
    # Residual-free contrast: the fit through the origin recovers the prediction.
    assert abs(m["r2"] - 1.0) < 1e-2
    assert abs(m["slope"] - 1.0) < 0.1
    assert causal.self_test() == 0


def test_random_controls_have_no_explanatory_power() -> None:
    m = causal.self_test_metrics()
    ctrl = [p for p in m["pairs"] if p[3] == 1]
    assert len(ctrl) == 8
    # Controls lie outside the code support, so their predicted contribution is 0 ...
    assert m["ctrl_max_abs_pred"] < 1e-6
    x = np.array([p[1] for p in ctrl])
    y = np.array([p[2] for p in ctrl])
    # ... and a through-origin fit of realized on predicted is undefined (no r2 to claim).
    r2, slope = causal.fit_r2_slope(x, y)
    assert math.isnan(r2) and math.isnan(slope)


def test_additive_identity_and_ablation_sign() -> None:
    m = causal.self_test_metrics()
    h, q, beta, W_dec = m["h"], m["q"], m["beta"], m["W_dec"]
    # Sum of all per-feature contributions equals the exact margin (residual-free q).
    m_exact = float(h @ q)
    m_recon = float((beta * (W_dec @ h)).sum())
    assert abs(m_exact - m_recon) < 1e-4 * max(1.0, abs(m_exact))
    for fid, c_pred, delta_real, is_random in m["pairs"]:
        d_i = W_dec[fid]
        # Stored realized change is the algebraic form (h . d_i)(q . d_i).
        assert abs(delta_real - float((d_i @ h) * (d_i @ q))) < 1e-4
        # Ablating h along the unit direction moves the dense margin by exactly -delta_real.
        h_abl = h - (h @ d_i) * d_i
        assert abs(float(h_abl @ q - h @ q) + delta_real) < 1e-4
        if not is_random:
            # Predicted contribution is beta_i times the projection, by construction.
            assert abs(c_pred - float(beta[fid] * (d_i @ h))) < 1e-5


def test_third_check_measures_the_dense_head_not_the_formula() -> None:
    m = causal.self_test_metrics()
    W_head, h, q = m["W_head"], m["h"], m["q"]
    # The synthetic head is a genuine two-row readout whose row difference is q.
    assert W_head.shape == (2, h.numel())
    assert torch.allclose(W_head[0] - W_head[1], q, atol=1e-6)
    assert m["dense_head_abs_err"] < 1e-4
    m_base = causal.dense_head_margin(W_head, h)
    assert abs(m_base - float(h @ q)) < 1e-4
    for fid, _c_pred, delta_real, _is_random in m["pairs"]:
        d_i = m["W_dec"][fid]
        realized = causal.dense_head_margin(W_head, h - (h @ d_i) * d_i) - m_base
        assert abs(realized + delta_real) < 1e-4
    # A wrong stored value is caught: perturbing delta_real breaks the dense-head comparison.
    fid, _c, delta_real, _r = m["pairs"][0]
    d_i = m["W_dec"][fid]
    wrong = delta_real + 0.5
    assert abs((causal.dense_head_margin(W_head, h - (h @ d_i) * d_i) - m_base) + wrong) > 0.4


def test_fit_r2_slope_orientation() -> None:
    # y = 2x exactly: slope is realized-on-predicted, so it must read 2 (not 0.5).
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r2, slope = causal.fit_r2_slope(x, 2.0 * x)
    assert abs(slope - 2.0) < 1e-12 and abs(r2 - 1.0) < 1e-12
    assert all(math.isnan(v) for v in causal.fit_r2_slope(np.zeros(4), x))


def test_cluster_bootstrap_ci_brackets_point_estimate() -> None:
    m = causal.self_test_metrics()
    rows = [{"cluster": f"c{i % 3}", "c_pred": p[1], "delta_real": p[2]} for i, p in enumerate(m["pairs"]) if p[3] == 0]
    rows = rows * 3  # three copies so every resample keeps >= 3 points per fit
    lo, hi = causal.cluster_bootstrap_r2(rows, n_boot=50, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo > 0.9
    # Same seed, same interval.
    assert (lo, hi) == causal.cluster_bootstrap_r2(rows, n_boot=50, seed=0)


class _FakeTokenizer:
    """Vocabulary of ``n`` ids with the last rows of W_U past the tokenizer and id 0 special."""

    def __init__(self, n: int):
        self.n = n
        self.all_special_ids = [0]

    def __len__(self) -> int:
        return self.n


def test_select_row_mean_modes() -> None:
    gen = torch.Generator().manual_seed(0)
    W_U = torch.randn(10, 4, generator=gen)
    tok = _FakeTokenizer(8)  # rows 8, 9 are padded embedding rows; row 0 is special
    live, src = causal.select_row_mean("live", W_U, ckpt={}, tok=tok)
    assert torch.equal(live, W_U.mean(dim=0)) and "live" in src
    stored = torch.randn(4, generator=gen)
    trained_ckpt, src = causal.select_row_mean("trained", W_U, ckpt={"row_mean": stored}, tok=tok)
    assert torch.equal(trained_ckpt, stored) and "checkpoint" in src
    trained_mask, src = causal.select_row_mean("trained", W_U, ckpt={}, tok=tok)
    assert torch.equal(trained_mask, W_U[1:8].mean(dim=0)) and "text-token" in src
    assert not torch.equal(trained_mask, live)


def test_token_resolver_is_the_shared_bare_first_one() -> None:
    assert causal.resolve_single_token_bare_first is row_geometry.resolve_single_token_bare_first
    assert not hasattr(causal, "single_token_id")
