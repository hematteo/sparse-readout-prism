from __future__ import annotations

import torch

from sparse_readout_prism.decompose import batch_decomposition_identity_error, decompose_token_logit
from sparse_readout_prism.factorizers import TopKSAE


def test_single_decomposition_identity() -> None:
    torch.manual_seed(0)
    d_model = 16
    d_features = 32
    model = TopKSAE(d_model=d_model, d_features=d_features, k=4)
    W = torch.randn(10, d_model)
    row_mean = W.mean(dim=0)
    centered = W - row_mean
    row_norms = centered.norm(dim=1).clamp_min(1e-8)
    rows = centered / row_norms[:, None]
    h = torch.randn(d_model)
    token = 3

    dec = decompose_token_logit(h, W[token], row_mean, row_norms[token], rows[token], model, k=4)
    assert dec.identity_error.abs().item() < 1e-5
    rebuilt = dec.base_term + dec.feature_sum + dec.residual_term
    assert torch.allclose(rebuilt, dec.original_logit, atol=1e-5)


def test_batch_decomposition_identity() -> None:
    torch.manual_seed(1)
    d_model = 12
    model = TopKSAE(d_model=d_model, d_features=24, k=5)
    W = torch.randn(7, d_model)
    row_mean = W.mean(dim=0)
    centered = W - row_mean
    row_norms = centered.norm(dim=1).clamp_min(1e-8)
    rows = centered / row_norms[:, None]
    h = torch.randn(7, d_model)
    errors = batch_decomposition_identity_error(h, W, row_mean, row_norms, rows, model, k=5)
    assert errors.max().item() < 1e-5
