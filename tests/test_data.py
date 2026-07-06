"""data.py loading branches: real payloads, token_mask, padding drop, row stats.

The synthetic-fallback branch is covered by the smoke tests; this file covers
the branches a real extraction exercises — payload loading, the token_mask
row filter (including the `apply_token_mask: false` off switch), padding-row
dropping, the small-corpus val fallback flag, and the row-statistics helpers
the evaluation scripts share. Pure tensors, CPU, deterministic.
"""

from __future__ import annotations

import torch

from sparse_readout_prism.data import (
    center_normalize_rows,
    load_prism_dataset,
    preprocess_rows,
    resolve_row_mean,
    sample_rows,
)

VOCAB, D = 32, 8
KEPT = 20  # token_mask keeps ids [0, KEPT)


def _payload(tmp_path, *, with_mask: bool = True, h: torch.Tensor | None = None):
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(VOCAB, D, generator=gen)
    if h is None:
        h = torch.randn(4, 6, D, generator=gen)
    payload = {"W_U_orig": W, "h_LN": h}
    if with_mask:
        mask = torch.zeros(VOCAB, dtype=torch.bool)
        mask[:KEPT] = True
        payload["token_mask"] = mask
    p = tmp_path / "extract.pt"
    torch.save(payload, p)
    return p, payload


def _config(path, **data_overrides):
    data = {"path": str(path), "max_hidden": 16, "val_hidden": 4, "data_seed": 0}
    data.update(data_overrides)
    return {"data": data}


def test_real_payload_token_mask_filters_rows(tmp_path) -> None:
    p, _ = _payload(tmp_path)
    ds = load_prism_dataset(_config(p), seed=0)
    assert not ds.used_fallback and ds.source_path == str(p)
    assert ds.vocab_size == KEPT
    assert set(ds.row_token_ids.tolist()) <= set(range(KEPT))


def test_apply_token_mask_false_disables_the_mask(tmp_path) -> None:
    # Regression: the flag used to be a silent no-op (the mask was still
    # passed to choose_row_subset).
    p, _ = _payload(tmp_path)
    ds = load_prism_dataset(_config(p, apply_token_mask=False), seed=0)
    assert ds.vocab_size == VOCAB
    assert max(ds.row_token_ids.tolist()) >= KEPT


def test_padding_rows_are_dropped(tmp_path) -> None:
    gen = torch.Generator().manual_seed(1)
    h = torch.randn(4, 6, D, generator=gen)
    h[0, :3] = 0.0  # 3 padded positions -> 21 usable rows
    p, _ = _payload(tmp_path, h=h)
    ds = load_prism_dataset(_config(p, max_hidden=17, val_hidden=4), seed=0)
    for split in (ds.hidden_train, ds.hidden_val):
        assert (split.norm(dim=1) > 1e-6).all()
    assert ds.hidden_train.shape[0] + ds.hidden_val.shape[0] == 21
    assert ds.val_overlaps_train is False


def test_small_corpus_val_fallback_is_flagged(tmp_path) -> None:
    p, _ = _payload(tmp_path)  # 24 usable hidden rows
    ds = load_prism_dataset(_config(p, max_hidden=30, val_hidden=8), seed=0)
    assert ds.val_overlaps_train is True
    assert ds.hidden_val.shape[0] > 0


def test_sample_rows_shuffles_even_when_keeping_all() -> None:
    x = torch.arange(20, dtype=torch.float32)[:, None]
    out = sample_rows(x, 50, seed=3)
    assert out.shape == x.shape
    assert not torch.equal(out, x)  # shuffled, not a pass-through prefix
    assert torch.equal(out.flatten().sort().values, x.flatten())


def test_center_normalize_rows_matches_preprocess_rows() -> None:
    W = torch.randn(16, D, generator=torch.Generator().manual_seed(2))
    row_mean, row_norms, rows_normalized = preprocess_rows(W)
    norms, x = center_normalize_rows(W, row_mean)
    assert torch.allclose(norms, row_norms)
    assert torch.allclose(x, rows_normalized)


def test_resolve_row_mean_preference_order() -> None:
    gen = torch.Generator().manual_seed(4)
    W = torch.randn(VOCAB, D, generator=gen)
    mask = torch.zeros(VOCAB, dtype=torch.bool)
    mask[:KEPT] = True
    stored = torch.randn(D, generator=gen)

    assert torch.allclose(resolve_row_mean(W, token_mask=mask, ckpt={"row_mean": stored}), stored)
    assert torch.allclose(resolve_row_mean(W, token_mask=mask), W[:KEPT].mean(dim=0))
    assert torch.allclose(resolve_row_mean(W), W.mean(dim=0))
    # ckpt without a stored row_mean falls through to the mask branch.
    assert torch.allclose(resolve_row_mean(W, token_mask=mask, ckpt={}), W[:KEPT].mean(dim=0))
