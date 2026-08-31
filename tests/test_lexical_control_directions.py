"""Unit checks for the label-free category-direction controls of the lexical-edit
stress test (scripts/run/run_qwen_profanity_suppression_eval.py, loaded by path):

- ``category_direction_units`` returns unit-norm directions oriented toward the
  discovery side (non-negative dot with the mean-row direction) and is
  deterministic across calls (torch.linalg.svd, not a randomized PCA);
- the ``mean_row_direction`` / ``pca_group_direction`` / ``pca_group_rank4``
  methods of ``apply_method`` produce finite logits with a nonzero
  ``intervention_norm`` on a tiny synthetic W_U, and lower the discovery tokens'
  logits relative to the vocabulary average (the suppression sign convention).
"""

from __future__ import annotations

import pytest
import torch
from conftest import load_script

prof = load_script("scripts/run/run_qwen_profanity_suppression_eval.py")

NEW_METHODS = ("mean_row_direction", "pca_group_direction", "pca_group_rank4")


def _synthetic(vocab: int = 40, d_model: int = 12, d_features: int = 16):
    gen = torch.Generator().manual_seed(0)
    W_U = torch.randn(vocab, d_model, generator=gen)
    row_mean = W_U.mean(dim=0)
    decoder = torch.randn(d_features, d_model, generator=gen)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)
    logits = torch.randn(vocab, generator=gen)
    discovery_ids = [1, 4, 7, 9, 11]
    return W_U, row_mean, decoder, logits, discovery_ids


def test_category_direction_units_are_unit_norm_and_oriented() -> None:
    W_U, row_mean, _decoder, _logits, discovery_ids = _synthetic()
    mean_dir, pca_dir, pca_dirs = prof.category_direction_units(W_U, discovery_ids, row_mean)

    assert mean_dir.shape == (W_U.shape[1],)
    assert pca_dir.shape == (W_U.shape[1],)
    assert pca_dirs.shape == (min(4, len(discovery_ids)), W_U.shape[1])
    assert abs(mean_dir.norm().item() - 1.0) < 1e-5
    assert abs(pca_dir.norm().item() - 1.0) < 1e-5
    assert torch.allclose(pca_dirs.norm(dim=1), torch.ones(pca_dirs.shape[0]), atol=1e-5)

    # Oriented toward the discovery side: non-negative alignment with the mean row.
    assert torch.dot(mean_dir, pca_dir).item() >= 0.0
    for j in range(pca_dirs.shape[0]):
        assert torch.dot(mean_dir, pca_dirs[j]).item() >= 0.0
    # pca_dir is the first row of the rank-<=4 set.
    assert torch.allclose(pca_dir, pca_dirs[0], atol=1e-6)
    # Mean direction is the normalized centered mean of the discovery rows.
    m = (W_U[discovery_ids] - row_mean).mean(dim=0)
    assert torch.allclose(mean_dir, m / m.norm(), atol=1e-6)


def test_category_direction_units_deterministic() -> None:
    W_U, row_mean, _decoder, _logits, discovery_ids = _synthetic()
    first = prof.category_direction_units(W_U, discovery_ids, row_mean)
    second = prof.category_direction_units(W_U, discovery_ids, row_mean)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.mark.parametrize("method", NEW_METHODS)
def test_new_methods_produce_finite_logits_and_nonzero_norm(method: str) -> None:
    W_U, row_mean, decoder, logits, discovery_ids = _synthetic()
    mean_dir, pca_dir, pca_dirs = prof.category_direction_units(W_U, discovery_ids, row_mean)
    scale = 2.0
    edited, meta = prof.apply_method(
        logits=logits,
        W_U=W_U,
        decoder=decoder,
        method=method,
        feature_ids=[0, 1, 2],
        random_ids=[3, 4, 5],
        scale=scale,
        token_ids=[],
        audited_ids=[],
        mean_dir=mean_dir,
        pca_dir=pca_dir,
        pca_dirs=pca_dirs,
    )
    assert edited.shape == logits.shape
    assert torch.isfinite(edited).all()
    assert meta["intervention_norm"] > 0.0
    assert meta["token_bias_strength"] == 0.0
    assert not torch.allclose(edited, logits)
    if method != "pca_group_rank4":
        # A single unit direction scaled by ``scale``.
        assert abs(meta["intervention_norm"] - scale) < 1e-5
    # Suppression sign convention: the discovery tokens' logits drop relative to
    # the vocabulary-average change.
    delta = edited - logits
    assert delta[discovery_ids].mean().item() < delta.mean().item()


@pytest.mark.parametrize("method", NEW_METHODS)
def test_new_methods_require_their_direction(method: str) -> None:
    W_U, _row_mean, decoder, logits, _discovery_ids = _synthetic()
    with pytest.raises(ValueError):
        prof.apply_method(
            logits=logits,
            W_U=W_U,
            decoder=decoder,
            method=method,
            feature_ids=[0, 1, 2],
            random_ids=[3, 4, 5],
            scale=1.0,
            token_ids=[],
            audited_ids=[],
        )


def test_grid_constants_match_paper_protocol() -> None:
    assert len(prof.PROMPTS) == 20
    assert len(prof.PAIRS) == 17
    heldout = [bad for bad, _good in prof.PAIRS if bad not in prof.DISCOVERY_TERMS]
    assert len(heldout) == 12
    assert prof.SCALES == [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    assert all(bad in prof.PAIR_TO_FEATURE_SET for bad, _good in prof.PAIRS)
