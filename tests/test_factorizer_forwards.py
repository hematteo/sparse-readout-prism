"""Forward/encode correctness for each factorizer architecture + reconstruction_loss.

test_factorizers.py covers *loading*; this pins the actual sparse-coding math
(known-answer, CPU-only). The paper compares TopK / Matryoshka / JumpReLU /
Gated, so a silent bug in any encode path would corrupt every downstream metric.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sparse_readout_prism.factorizers import (
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    L1ReLUSAE,
    TopKSAE,
    reconstruction_loss,
)


def test_topk_encode_matches_relu_topk_scatter() -> None:
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, d_features=32, k=4)
    x = torch.randn(5, 8)
    code = sae.encode(x, k=4)

    acts = F.relu(sae.encoder(x))
    vals, idx = torch.topk(acts, k=4, dim=-1)
    manual = torch.zeros_like(acts)
    manual.scatter_(1, idx, vals)

    assert torch.allclose(code, manual)
    assert (code != 0).sum(dim=1).max().item() <= 4  # at most k actives per row
    # decode is a plain linear map through the decoder
    assert torch.allclose(sae.decode(code), code @ sae.decoder)


def test_topk_full_pass_when_k_ge_d_features() -> None:
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, d_features=16, k=8)
    x = torch.randn(4, 8)
    code = sae.encode(x, k=64)  # k >= d_features -> no sparsity, raw relu acts
    assert torch.allclose(code, F.relu(sae.encoder(x)))


def test_batch_topk_train_selects_global_bk() -> None:
    torch.manual_seed(0)
    sae = BatchTopKSAE(d_model=8, d_features=32, k=3)
    sae.train()
    x = torch.randn(4, 8)
    code = sae.encode(x, k=3)

    n_pos = int((F.relu(sae.encoder(x)) > 0).sum())
    assert (code != 0).sum().item() == min(4 * 3, n_pos)  # B*k kept across the whole plane
    assert int(sae.threshold_initialized.item()) == 1  # EMA threshold initialised


def test_batch_topk_eval_before_threshold_fit_raises() -> None:
    # threshold buffer defaults to 0.0; silently passing every positive
    # activation would be a dense code masquerading as sparse.
    sae = BatchTopKSAE(d_model=8, d_features=32, k=3)
    sae.eval()
    with pytest.raises(RuntimeError, match="threshold"):
        sae.encode(torch.randn(4, 8), k=3)


def test_batch_topk_eval_applies_threshold() -> None:
    torch.manual_seed(0)
    sae = BatchTopKSAE(d_model=8, d_features=32, k=3)
    x = torch.randn(6, 8)
    sae.train()
    sae.encode(x, k=3)  # sets activation_threshold
    sae.eval()
    code = sae.encode(x, k=3)
    acts = F.relu(sae.encoder(x))
    expected = torch.where(acts > sae.activation_threshold, acts, torch.zeros_like(acts))
    assert torch.allclose(code, expected)


def test_jumprelu_gates_strictly_above_threshold() -> None:
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_model=8, d_features=16)
    x = torch.randn(5, 8)
    code = sae.encode(x)

    z = F.relu(sae.encoder(x))
    theta = sae.threshold().unsqueeze(0)
    expected = z * (z > theta).to(z.dtype)
    assert torch.allclose(code, expected)
    # forward exposes an L0 estimate equal to the gate mass
    batch = sae(x)
    assert torch.allclose(batch.aux_loss, (z > theta).to(z.dtype).sum(dim=-1).mean())


def test_gated_encode_matches_two_branch_definition() -> None:
    torch.manual_seed(0)
    sae = GatedSAE(d_model=8, d_features=16)
    with torch.no_grad():  # non-trivial gate/mag params
        sae.gate_bias.normal_()
        sae.log_r.normal_(0.0, 0.1)
        sae.encoder.bias.normal_()
    x = torch.randn(5, 8)
    code = sae.encode(x)

    z = x @ sae.encoder.weight.T
    gate = (z + sae.gate_bias > 0).to(z.dtype)
    mag = F.relu(sae.log_r.exp() * z + sae.encoder.bias)
    assert torch.allclose(code, gate * mag)


def test_l1relu_encode_is_relu_and_aux_is_decoder_weighted_l1() -> None:
    torch.manual_seed(0)
    sae = L1ReLUSAE(d_model=8, d_features=16)
    x = torch.randn(5, 8)
    code = sae.encode(x)
    assert torch.allclose(code, F.relu(sae.encoder(x)))

    batch = sae(x)
    expected_l1 = (code * sae.decoder.norm(dim=1).detach()).sum(dim=-1).mean()
    assert torch.allclose(batch.aux_loss, expected_l1)


def test_reconstruction_loss_topk_is_plain_mse() -> None:
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, d_features=32, k=4)
    x = torch.randn(6, 8)
    rec, code, aux, info = reconstruction_loss(sae, x, "topk", k=4)
    assert torch.allclose(rec, F.mse_loss(sae(x, k=4).reconstruction, x))
    assert (code != 0).sum(dim=1).max().item() <= 4


def test_reconstruction_loss_matryoshka_is_weighted_average_over_k_values() -> None:
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, d_features=32, k=8)
    x = torch.randn(6, 8)
    k_values, weights = [2, 4, 8], [3.0, 2.0, 1.0]
    rec, code, aux, _ = reconstruction_loss(sae, x, "matryoshka_topk", k=8, k_values=k_values, weights=weights)
    manual = sum(w * F.mse_loss(sae(x, k=kk).reconstruction, x) for kk, w in zip(k_values, weights))
    manual = manual / sum(weights)
    assert torch.allclose(rec, manual)
    assert torch.allclose(aux, torch.zeros(()))
    assert (code != 0).sum(dim=1).max().item() <= 8  # last code is the k=8 prefix


def test_reconstruction_loss_gated_adds_aux_recon_mse() -> None:
    torch.manual_seed(0)
    sae = GatedSAE(d_model=8, d_features=16)
    x = torch.randn(6, 8)
    _, _, _, info = reconstruction_loss(sae, x, "gated", k=16)
    assert "gated_aux_recon_mse" in info
    assert torch.isfinite(info["gated_aux_recon_mse"].detach()).all()
