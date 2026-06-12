"""Correctness tests for two standalone producers loaded by path:

- scripts/run/run_readout_baseline_comparisons.py — the null/baseline methods
  behind the paper's "Sparse RP beats every null" claim. Key property: every
  method preserves the EXACT additive identity (so the comparison is fair) while
  the nulls change only the support.
- scripts/figures/compute_prism_dla.py — the feature-resolved DLA accounting.
  build_rows with top_features=0 skips labelling (no tokenizer needed), so we can
  assert the component identity (margin == sum of component direct contributions)
  and the per-component reconstruction identity on synthetic inputs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass(KW_ONLY) resolution needs the module registered
    spec.loader.exec_module(mod)
    return mod


baseline = _load("scripts/run/run_readout_baseline_comparisons.py", "baseline_comparisons")
dla = _load("scripts/figures/compute_prism_dla.py", "compute_prism_dla")

from sparse_readout_prism.factorizers import TopKSAE  # noqa: E402


def _make_sae_inputs(vocab: int = 24, d_model: int = 8):
    torch.manual_seed(0)
    W = torch.randn(vocab, d_model)
    row_mean = W.mean(dim=0)
    sae = TopKSAE(d_model=d_model, d_features=32, k=4)
    return W, row_mean, sae


def test_baseline_methods_preserve_additive_identity() -> None:
    W, row_mean, sae = _make_sae_inputs()
    device = torch.device("cpu")
    h = torch.randn(8)
    for spec in ("sparse_rp", "shuffled_row_code", "random_support_same_magnitudes"):
        method = baseline.build_method(spec, W=W, row_mean=row_mean, device=device, sae=sae, k=4, seed=0)
        dec = method.decompose_row(h, row_idx=3)
        # original_logit == base + feature_sum + residual to float precision.
        assert dec.identity_error.abs().item() < 1e-4, spec
        if spec != "sparse_rp":
            # nulls keep the SAE's k-sparsity, only the support/assignment changes.
            assert dec.active_feature_indices.numel() == 4, spec


def test_build_method_rejects_bad_specs() -> None:
    W, row_mean, _ = _make_sae_inputs()
    device = torch.device("cpu")
    with pytest.raises(ValueError):
        baseline.build_method("not_a_method", W=W, row_mean=row_mean, device=device)
    with pytest.raises(ValueError):  # sparse_rp needs sae + k
        baseline.build_method("sparse_rp", W=W, row_mean=row_mean, device=device)


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
