"""Tests for factorizers.load_factorizer — the single checkpoint reader matched
to runner.py's writer. Covers the checkpoint-schema variants that the eval/run
scripts previously each handled with their own copy of the build+load logic.
"""

from __future__ import annotations

import pytest
import torch

from sparse_readout_prism.factorizers import build_factorizer, load_factorizer

CFG = {"architecture": "topk", "d_features": 64, "k": 8}
D_MODEL = 16


def _make() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_factorizer({"factorizer": CFG}, d_model=D_MODEL)


def test_load_from_path_infers_d_model_and_config(tmp_path):
    """Path checkpoint with config embedded under ['config']['factorizer']."""
    m = _make()
    p = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": m.state_dict(), "config": {"factorizer": CFG}}, p)

    loaded = load_factorizer(p)  # no d_model / config passed -> both inferred

    assert (loaded.d_model, loaded.d_features, loaded.k) == (D_MODEL, 64, 8)
    assert torch.equal(loaded.decoder, m.decoder)
    assert torch.equal(loaded.encoder.weight, m.encoder.weight)
    assert not loaded.training  # eval_mode default


def test_load_from_dict_state_dict_key_and_freeze():
    """Dict checkpoint using the alternate 'state_dict' key + top-level 'factorizer'."""
    m = _make()
    ckpt = {"state_dict": m.state_dict(), "factorizer": CFG}

    loaded = load_factorizer(ckpt, d_model=D_MODEL, freeze=True)

    assert torch.equal(loaded.decoder, m.decoder)
    assert all(not p.requires_grad for p in loaded.parameters())


def test_external_config_used_when_absent_from_checkpoint():
    m = _make()
    ckpt = {"model_state_dict": m.state_dict()}  # no factorizer config inside

    loaded = load_factorizer(ckpt, factorizer_config=CFG, d_model=D_MODEL)

    assert torch.equal(loaded.decoder, m.decoder)


def test_missing_config_raises():
    m = _make()
    with pytest.raises(KeyError):
        load_factorizer({"model_state_dict": m.state_dict()}, d_model=D_MODEL)
