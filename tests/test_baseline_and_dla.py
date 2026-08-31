"""Correctness tests for two standalone producers loaded by path:

- scripts/run/run_readout_baseline_comparisons.py — the null/baseline methods
  behind the paper's "Sparse RP beats every null" claim. Key property: every
  method preserves the EXACT additive identity (so the comparison is fair) while
  the nulls change only the support. Also covers the 0.2.1 consolidation: the
  shared ``_finish`` tail, the memoised k-means fit, and ``margin_from_rows``
  aligning per-row supports before subtracting.
- scripts/figures/compute_prism_dla.py — the feature-resolved DLA accounting.
  build_rows with top_features=0 skips labelling (no tokenizer needed), so we can
  assert the component identity (margin == sum of component direct contributions)
  and the per-component reconstruction identity on synthetic inputs.
"""

from __future__ import annotations

import pytest
import torch
from conftest import load_script

from sparse_readout_prism.factorizers import TopKSAE

baseline = load_script("scripts/run/run_readout_baseline_comparisons.py", "baseline_comparisons")
dla = load_script("scripts/figures/compute_prism_dla.py", "compute_prism_dla")

CPU = torch.device("cpu")

# spec -> (expected number of active atoms or None, expected feature_space_size)
METHOD_PANEL = {
    "sparse_rp": (None, 32),  # d_features
    "shuffled_row_code": (4, 32),  # nulls keep the SAE's k-sparsity, only the support/assignment changes
    "random_support_same_magnitudes": (4, 32),
    "pca_4": (4, 4),  # every component active; the basis is shared by all rows
    "nearest_row_ridge_top4": (4, 24),  # the four nearest rows, indices into the vocabulary
    "knn_basis_top4": (4, 24),
    "row_cluster_d16_k4": (4, 16),  # top-4 of 16 centroids
    "row_cluster_hard_d16": (1, 16),  # the row's own centroid
}


def _make_sae_inputs(vocab: int = 24, d_model: int = 8):
    torch.manual_seed(0)
    W = torch.randn(vocab, d_model)
    row_mean = W.mean(dim=0)
    sae = TopKSAE(d_model=d_model, d_features=32, k=4)
    return W, row_mean, sae


def _build(spec: str, W, row_mean, sae, **kw):
    torch.manual_seed(1)  # pca_*: torch.svd_lowrank draws its test matrix from the global RNG
    return baseline.build_method(spec, W=W, row_mean=row_mean, device=CPU, sae=sae, k=4, seed=0, **kw)


def test_baseline_methods_preserve_additive_identity() -> None:
    W, row_mean, sae = _make_sae_inputs()
    h = torch.randn(8)
    for spec, (n_active, space) in METHOD_PANEL.items():
        method = _build(spec, W, row_mean, sae)
        dec = method.decompose_row(h, row_idx=3, exclude_ids=[9])
        # original_logit == base + feature_sum + residual to float precision.
        assert dec.identity_error.abs().item() < 1e-4, spec
        assert torch.isfinite(dec.reconstructed_logit), spec
        assert dec.feature_space_size == space, spec
        if n_active is not None:
            assert dec.active_feature_indices.numel() == n_active, spec


def test_finish_tail_identity_for_every_method() -> None:
    """The shared ``_finish`` reproduces the accounting each method used to inline:
    exact base / original terms, feature_sum as the sum of the per-atom
    contributions, and contributions that index the declared feature space."""
    W, row_mean, sae = _make_sae_inputs()
    h = torch.randn(8)
    for spec, (_, space) in METHOD_PANEL.items():
        method = _build(spec, W, row_mean, sae)
        for row in (3, 17):
            dec = method.decompose_row(h, row_idx=row, exclude_ids=[9])
            assert torch.equal(dec.base_term, h @ row_mean), spec
            assert torch.equal(dec.original_logit, h @ W[row]), spec
            assert torch.equal(dec.reconstructed_logit, dec.base_term + dec.feature_sum), spec
            assert torch.equal(
                dec.identity_error, dec.original_logit - (dec.base_term + dec.feature_sum + dec.residual_term)
            ), spec
            active = dec.active_feature_indices
            assert active.device.type == "cpu" and active.numel() == active.unique().numel(), spec
            assert int(active.max()) < space, spec
            fc = dec.feature_contributions
            if fc.numel() == space:  # full-size vector: nonzeros live on the support
                assert torch.allclose(fc.sum(), dec.feature_sum, atol=1e-5), spec
                if spec != "pca_4":
                    off = torch.ones(space, dtype=torch.bool)
                    off[active] = False
                    assert torch.equal(fc[off], torch.zeros(int(off.sum()))), spec
            else:  # per-row support: one entry per active atom, in support order
                assert fc.numel() == active.numel(), spec
                assert torch.equal(dec.feature_sum, fc.sum()), spec


def test_row_cluster_fit_is_memoised_by_clusters_and_seed() -> None:
    W, row_mean, sae = _make_sae_inputs()
    cache: dict = {}
    soft = _build("row_cluster_d16_k4", W, row_mean, sae, kmeans_cache=cache)
    hard = _build("row_cluster_hard_d16", W, row_mean, sae, kmeans_cache=cache)
    assert hard.C is soft.C and list(cache) == [(16, 0)]
    other_seed = baseline.build_method(
        "row_cluster_hard_d16", W=W, row_mean=row_mean, device=CPU, seed=1, kmeans_cache=cache
    )
    assert other_seed.C is not soft.C and set(cache) == {(16, 0), (16, 1)}
    fresh = _build("row_cluster_d16_k4", W, row_mean, sae)  # no cache: refits, bit-exact on CPU
    assert torch.equal(fresh.C, soft.C)
    h = torch.randn(8)
    assert torch.equal(fresh.decompose_row(h, 3).feature_contributions, soft.decompose_row(h, 3).feature_contributions)


def test_margin_from_rows_aligns_per_row_supports() -> None:
    W, row_mean, sae = _make_sae_inputs()
    h = torch.randn(8)
    a, b = 3, 9
    knn = _build("knn_basis_top4", W, row_mean, sae)
    dA = knn.decompose_row(h, a, exclude_ids=[b])
    dB = knn.decompose_row(h, b, exclude_ids=[a])
    mr = baseline.margin_from_rows(h, knn, [a], [b])
    fm = mr["feat_margin"]
    # one entry per vocabulary row, A's neighbours positive-side, B's negated, by row index
    assert fm.numel() == W.shape[0] == dA.feature_space_size
    expected = torch.zeros(W.shape[0])
    expected.scatter_add_(0, dA.active_feature_indices, dA.feature_contributions)
    expected.scatter_add_(0, dB.active_feature_indices, -dB.feature_contributions)
    assert torch.allclose(fm, expected, atol=1e-6)
    support = set(dA.active_feature_indices.tolist()) | set(dB.active_feature_indices.tolist())
    assert len(support) > 4  # the two rows have different neighbourhoods on this fixture
    assert set(torch.nonzero(fm).flatten().tolist()) <= support
    # the margin scalars never depended on the alignment: sparse == s_A - s_B, and the aligned
    # vector still sums to it (the row_mean base cancels)
    assert mr["sparse"] == pytest.approx(float(dA.reconstructed_logit - dB.reconstructed_logit), abs=1e-5)
    assert float(fm.sum()) == pytest.approx(mr["sparse"], abs=1e-5)
    # ridge and cluster methods scatter the same way; shared-basis methods pass through untouched
    ridge_fm = baseline.margin_from_rows(h, _build("nearest_row_ridge_top4", W, row_mean, sae), [a], [b])["feat_margin"]
    assert ridge_fm.numel() == W.shape[0]
    hard_fm = baseline.margin_from_rows(h, _build("row_cluster_hard_d16", W, row_mean, sae), [a], [b])["feat_margin"]
    assert hard_fm.numel() == 16 and int((hard_fm != 0).sum()) <= 2
    assert baseline.margin_from_rows(h, _build("pca_4", W, row_mean, sae), [a], [b])["feat_margin"].numel() == 4
    assert baseline.margin_from_rows(h, _build("sparse_rp", W, row_mean, sae), [a], [b])["feat_margin"].numel() == 32
    # the pseudo-target (B = row_mean) path pads B with zeros of the aligned length
    fm_mean = baseline.margin_from_rows(h, knn, [a], None, mean_row=row_mean)["feat_margin"]
    assert fm_mean.numel() == W.shape[0]
    assert torch.allclose(
        fm_mean, torch.zeros(W.shape[0]).scatter_add(0, dA.active_feature_indices, dA.feature_contributions), atol=1e-6
    )


def test_build_method_rejects_bad_specs() -> None:
    W, row_mean, _ = _make_sae_inputs()
    with pytest.raises(ValueError):
        baseline.build_method("not_a_method", W=W, row_mean=row_mean, device=CPU)
    with pytest.raises(ValueError):  # sparse_rp needs sae + k
        baseline.build_method("sparse_rp", W=W, row_mean=row_mean, device=CPU)


def test_prism_dla_component_and_reconstruction_identities() -> None:
    torch.manual_seed(0)
    d_model, vocab, d_features, k = 8, 30, 24, 4
    W = torch.randn(vocab, d_model)
    row_mean = W.mean(dim=0)
    decoder = torch.randn(d_features, d_model)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)
    encoder_w = torch.randn(d_features, d_model) * 0.5
    encoder_b = torch.zeros(d_features)

    # Two residual-stream components that sum to the final hidden state, so the
    # component direct contributions must sum to the exact margin.
    h_final = torch.randn(d_model)
    h1 = torch.randn(d_model)
    components = [
        dla.Component(name="c1", label="c1", layer=0, kind="block_delta", vector=h1, norm_linearized=h1),
        dla.Component(
            name="c2", label="c2", layer=1, kind="block_delta", vector=h_final - h1, norm_linearized=h_final - h1
        ),
    ]
    logits = h_final @ W.T  # margin = logits[t] - logits[c] = h_final @ (W[t]-W[c])
    target_id, contrast_id = 5, 9

    comp_rows, feat_rows, agg_rows, _labels, summary = dla.build_rows(
        components=components,
        W=W,
        decoder=decoder,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=None,  # unused: top_features=0 short-circuits labelling
        target_id=target_id,
        contrast_id=contrast_id,
        k=k,
        logits=logits,
        label_top_tokens=4,
        label_chunk_size=64,
        top_features=0,
        contrast_mode="token_token",
    )

    # exact_margin == sum of per-component direct contributions
    assert summary["component_identity_abs_error"] < 1e-4
    # each component's direct contribution == feature_sum + W_U residual term
    assert comp_rows, "expected per-component rows"
    for row in comp_rows:
        assert row["component_reconstruction_abs_error"] < 1e-4
