"""Correctness tests for evaluate.py — the replacement/reconstruction diagnostics
and the operating-point selection score (both untested before).

selection_score and _duplicate_decoder_frac are pure → exact known-answer.
evaluate_model is exercised on the synthetic dataset; the load-bearing assertion
is that the additive decomposition identity holds (identity_abs_error ~ 0)
regardless of reconstruction quality.
"""

from __future__ import annotations

import math

import torch

from sparse_readout_prism.data import load_prism_dataset
from sparse_readout_prism.evaluate import (
    _duplicate_decoder_frac,
    contribution_coverages,
    evaluate_model,
    selection_score,
)
from sparse_readout_prism.factorizers import TopKSAE, build_factorizer
from sparse_readout_prism.utils import load_yaml

_PASSING = {
    "dead_feature_rate": 0.1,
    "val_logit_kl_bits_mean": 1.0,
    "identity_abs_error_max": 1e-9,
    "row_centered_ev": 0.5,
    "row_centered_cosine": 0.6,
    "top8_abs_contrib_coverage_mean": 0.7,
    "top8_positive_contrib_coverage_mean": 0.4,
    "val_top1_match": 0.8,
    "val_top5_overlap": 0.9,
    "top1_logit_residual_frac_mean": 0.1,
}


def test_selection_score_weighted_sum_when_gates_pass() -> None:
    expected = (
        1.5 * 0.5 + 0.5 * 0.6 + 0.7 * 0.7 + 0.4 * 0.4 + 0.5 * 0.8 + 0.3 * 0.9 - 0.8 * 0.1 - 0.03 * 1.0 - 0.2 * 0.1
    )
    assert abs(selection_score(dict(_PASSING)) - expected) < 1e-9


def test_selection_score_gates() -> None:
    assert selection_score({"x": float("nan")}) == -1e9  # non-finite
    dead = selection_score({**_PASSING, "dead_feature_rate": 0.99})
    assert -1e6 - 1 < dead <= -1e6 + 1  # dead-feature gate
    assert selection_score({**_PASSING, "val_logit_kl_bits_mean": 25.0}) <= -1e5  # KL gate
    assert selection_score({**_PASSING, "identity_abs_error_max": 1e-2}) <= -1e5  # identity gate


def test_duplicate_decoder_frac_counts_near_duplicates() -> None:
    # rows 0 and 1 are identical (cosine 1); rows 2,3 are orthogonal to all.
    dec = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert abs(_duplicate_decoder_frac(dec, threshold=0.9) - 0.5) < 1e-9
    # all-orthonormal decoder has no duplicates
    assert _duplicate_decoder_frac(torch.eye(5), threshold=0.9) == 0.0


def test_contribution_coverages_bounded_and_full_when_topk_covers_all() -> None:
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, d_features=16, k=4)
    h, rows, rn = torch.randn(3, 8), torch.randn(3, 8), torch.rand(3)
    pos, ab = contribution_coverages(sae, h, rows, rn, k=4, contribution_top_k=8)
    assert 0.0 <= pos <= 1.0 and 0.0 <= ab <= 1.0
    # contribution_top_k >= d_features -> the top set is everything -> coverage 1.
    pos2, ab2 = contribution_coverages(sae, h, rows, rn, k=4, contribution_top_k=16)
    assert abs(ab2 - 1.0) < 1e-6 and abs(pos2 - 1.0) < 1e-6


def test_evaluate_model_identity_holds_and_metrics_finite() -> None:
    config = load_yaml("configs/smoke.yaml")
    config["data"]["path"] = "/path/that/does/not/exist.pt"
    config["data"]["max_rows"] = 256
    config["data"]["max_hidden"] = 32
    config["data"]["val_hidden"] = 16
    dataset = load_prism_dataset(config, seed=0)
    sae = build_factorizer({"factorizer": config["factorizer"]}, d_model=dataset.W_U.shape[1])

    m = evaluate_model(sae, dataset, config, torch.device("cpu"))

    assert all(math.isfinite(float(v)) for v in m.values() if isinstance(v, (int, float)))
    # The additive decomposition identity must hold to float precision regardless
    # of how well the (barely-initialised) SAE reconstructs.
    assert m["identity_abs_error_max"] < 1e-3
    assert -1.0 <= m["row_centered_ev"] <= 1.0
    assert 0.0 <= m["val_top1_match"] <= 1.0
    assert m["val_logit_kl_bits_mean"] >= -1e-6
    assert 0.0 <= m["dead_feature_rate"] <= 1.0
    assert 0.0 <= m["top8_abs_contrib_coverage_mean"] <= 1.0
    assert math.isfinite(float(m["selection_score"]))
