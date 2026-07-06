from __future__ import annotations

from dataclasses import dataclass

import torch

from sparse_readout_prism.factorizers import SAEBase


@dataclass
class Decomposition:
    original_logit: torch.Tensor
    base_term: torch.Tensor
    feature_contributions: torch.Tensor
    feature_sum: torch.Tensor
    residual_term: torch.Tensor
    reconstructed_logit: torch.Tensor
    identity_error: torch.Tensor
    active_feature_indices: torch.Tensor


def decompose_token_logit(
    h: torch.Tensor,
    W_row: torch.Tensor,
    row_mean: torch.Tensor,
    row_norm: torch.Tensor,
    row_normalized: torch.Tensor,
    model: SAEBase,
    k: int,
) -> Decomposition:
    """Decompose one selected token logit into base, sparse features, and residual.

    The exact identity (checked by ``identity_error``) is
    ``original_logit == base_term + feature_sum + residual_term``;
    ``reconstructed_logit = base_term + feature_sum`` is the prism
    *approximation*, i.e. it deliberately excludes the residual.

    ``k`` only has an effect for TopK-family factorizers; the threshold-based
    architectures (jumprelu / gated / l1_relu) accept and ignore it.
    """
    if h.ndim != 1:
        raise ValueError("h must be a single hidden vector")
    if W_row.ndim != 1:
        raise ValueError("W_row must be a single unembedding row")

    code = model.encode(row_normalized[None, :], k=k)[0]
    decoded = model.decode(code[None, :])[0]
    reconstructed_row = row_mean + row_norm * decoded
    residual_row = W_row - reconstructed_row

    feature_hidden_scores = h @ model.decoder.T
    feature_contributions = row_norm * code * feature_hidden_scores
    base_term = h @ row_mean
    feature_sum = feature_contributions.sum()
    residual_term = h @ residual_row
    original_logit = h @ W_row
    reconstructed_logit = base_term + feature_sum
    identity_error = original_logit - (base_term + feature_sum + residual_term)
    active = torch.nonzero(code.detach().cpu() != 0, as_tuple=False).flatten()

    return Decomposition(
        original_logit=original_logit,
        base_term=base_term,
        feature_contributions=feature_contributions,
        feature_sum=feature_sum,
        residual_term=residual_term,
        reconstructed_logit=reconstructed_logit,
        identity_error=identity_error,
        active_feature_indices=active,
    )


def batch_decomposition_identity_error(
    h: torch.Tensor,
    W_rows: torch.Tensor,
    row_mean: torch.Tensor,
    row_norms: torch.Tensor,
    rows_normalized: torch.Tensor,
    model: SAEBase,
    k: int,
) -> torch.Tensor:
    """Return absolute identity errors for matching batches of h and W rows."""
    code = model.encode(rows_normalized, k=k)
    decoded = model.decode(code)
    reconstructed = row_mean[None, :] + row_norms[:, None] * decoded
    residual = W_rows - reconstructed
    base = (h * row_mean[None, :]).sum(dim=1)
    feature_scores = h @ model.decoder.T
    feature_sum = (row_norms[:, None] * code * feature_scores).sum(dim=1)
    residual_term = (h * residual).sum(dim=1)
    original = (h * W_rows).sum(dim=1)
    return (original - (base + feature_sum + residual_term)).abs()
