"""Correctness tests for scripts/eval/run_causal_contribution_validation.py (loaded by path).

Pins the math core behind ``tab:causal-validation-summary`` on the script's own
synthetic residual-free case (a contrast built exactly in the span of a random
unit-row dictionary from a known 3-sparse beta):

* predicted contributions track the realized changes (r2 ~ 1, slope ~ 1),
* random-feature controls carry zero predicted contribution, so they have no
  explanatory power (the through-origin fit is undefined for them),
* the additive identity holds: the contributions sum to the exact margin, each
  stored realized change equals (h . d_i)(q . d_i), and ablating h along d_i
  moves the dense margin by exactly minus that amount.

CPU only, no model, well under a second.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


causal = _load("scripts/eval/run_causal_contribution_validation.py", "causal_contribution_validation")


def test_self_test_passes_and_recovers_unit_agreement() -> None:
    m = causal.self_test_metrics()
    assert all(ok for _, ok in m["checks"]), m["checks"]
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
