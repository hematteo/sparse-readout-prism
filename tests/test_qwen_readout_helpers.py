"""Unit tests for the research.qwen_readout toolkit.

Pins three contracts:
  * encode_topk (raw-tensor path) is bit-identical to TopKSAE.encode — the
    library/script selection semantics cannot drift;
  * load_sae round-trips a runner-schema checkpoint, surfaces the stored
    training row_mean, and rejects non-TopK architectures;
  * collect_readout_states_batched reads the LAST REAL position even when the
    tokenizer defaults to left padding (Qwen tokenizer configs often do).

Stub model/tokenizer only; no downloads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sparse_readout_prism.factorizers import GatedSAE, TopKSAE, build_factorizer
from sparse_readout_prism.research.qwen_readout import (
    collect_readout_states_batched,
    encode_topk,
    load_sae,
    readable_feature_label,
    token_summary,
)


def test_encode_topk_matches_topksae_encode() -> None:
    torch.manual_seed(0)
    sae = TopKSAE(d_model=16, d_features=64, k=6)
    x = torch.randn(8, 16)
    raw = encode_topk(x, sae.encoder.weight.detach(), sae.encoder.bias.detach(), k=6)
    assert torch.equal(raw, sae.encode(x, k=6))


def _runner_checkpoint(tmp_path, architecture: str):
    cfg = {"factorizer": {"architecture": architecture, "d_features": 32, "k": 4}}
    model = build_factorizer(cfg, d_model=8)
    row_mean = torch.randn(8)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "factorizer": cfg["factorizer"],
            "row_mean": row_mean,
        },
        path,
    )
    return path, model, row_mean


def test_load_sae_roundtrip_surfaces_training_row_mean(tmp_path) -> None:
    path, model, row_mean = _runner_checkpoint(tmp_path, "topk")
    decoder, encoder_w, encoder_b, _cfg, ckpt_row_mean = load_sae(path)
    assert ckpt_row_mean is not None
    assert torch.allclose(ckpt_row_mean, row_mean)
    assert torch.allclose(encoder_w, model.encoder.weight.detach())
    # Decoder rows come back unit-norm (training invariant re-asserted on load).
    assert torch.allclose(decoder.norm(dim=1), torch.ones(decoder.shape[0]), atol=1e-5)


def test_load_sae_legacy_checkpoint_without_row_mean(tmp_path) -> None:
    cfg = {"factorizer": {"architecture": "topk", "d_features": 32, "k": 4}}
    model = build_factorizer(cfg, d_model=8)
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "factorizer": cfg["factorizer"]}, path)
    *_rest, ckpt_row_mean = load_sae(path)
    assert ckpt_row_mean is None


def test_load_sae_rejects_non_topk(tmp_path) -> None:
    model = GatedSAE(d_model=8, d_features=32)
    path = tmp_path / "gated.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "factorizer": {"architecture": "gated", "d_features": 32}},
        path,
    )
    with pytest.raises(ValueError, match="TopK"):
        load_sae(path)


def test_feature_label_helpers() -> None:
    assert readable_feature_label(["cat", "dog", "bird", "fish"], 7) == "f7 cat/dog/bird"
    assert readable_feature_label([], 7) == "f7"
    assert token_summary({"feature_top_tokens": "cat; dog"}) == "cat, dog"
    assert token_summary({"feature_top_tokens": "", "contrast_token_hint": "cat"}) == "cat row"


class _Batch(dict):
    def to(self, device):
        return self


class _PadTok:
    """Whitespace-int tokenizer with configurable padding side."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, padding_side: str) -> None:
        self.padding_side = padding_side

    def __call__(self, prompts, return_tensors="pt", padding=True):
        seqs = [[int(part) for part in p.split()] for p in prompts]
        width = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [0] * (width - len(s))
            if self.padding_side == "left":
                ids.append(pad + s)
                mask.append([0] * len(pad) + [1] * len(s))
            else:
                ids.append(s + pad)
                mask.append([1] * len(s) + [0] * len(pad))
        return _Batch(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(mask, dtype=torch.long),
        )


class _TokenValueLM(torch.nn.Module):
    """h[b, t, :] = token id (so the state at a position IS its token id)."""

    def __init__(self, d_model: int = 4, vocab: int = 32) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None):
        h = input_ids[..., None].float().expand(*input_ids.shape, 4).contiguous()
        return SimpleNamespace(logits=self.lm_head(h))


def test_collect_batched_reads_last_real_position_under_left_padding_default() -> None:
    torch.manual_seed(0)
    model = _TokenValueLM()
    tok = _PadTok(padding_side="left")  # the trap: Qwen configs often default left
    states, logits = collect_readout_states_batched(
        model, tok, ["1 2 3", "7"], device=torch.device("cpu"), batch_size=2
    )
    # Last REAL token per prompt, not a pad position.
    assert torch.equal(states[0], torch.full((4,), 3.0))
    assert torch.equal(states[1], torch.full((4,), 7.0))
    assert len(logits) == 2
    # The tokenizer's own padding_side is restored afterwards.
    assert tok.padding_side == "left"
