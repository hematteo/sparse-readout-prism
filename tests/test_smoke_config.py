from __future__ import annotations

from sparse_readout_prism.data import load_prism_dataset, row_sampling_weights
from sparse_readout_prism.utils import load_yaml


def test_synthetic_fallback_loads() -> None:
    config = load_yaml("configs/smoke.yaml")
    config["data"]["path"] = "/path/that/does/not/exist.pt"
    config["data"]["max_rows"] = 128
    config["data"]["max_hidden"] = 16
    config["data"]["val_hidden"] = 8
    dataset = load_prism_dataset(config, seed=123)
    assert dataset.W_U.shape[0] == 128
    assert dataset.rows_normalized.shape == dataset.W_U.shape
    assert dataset.hidden_train.shape[0] == 16
    assert dataset.hidden_val.shape[0] == 8


def test_challenger_sampling_modes_are_valid() -> None:
    config = load_yaml("configs/smoke.yaml")
    config["data"]["path"] = "/path/that/does/not/exist.pt"
    config["data"]["max_rows"] = 128
    config["data"]["max_hidden"] = 16
    config["data"]["val_hidden"] = 8
    dataset = load_prism_dataset(config, seed=456)

    for mode in ["frequency_pow_0.5", "frequency_pow_0.75", "hybrid_80freq_20uniform"]:
        weights = row_sampling_weights(dataset, mode)
        assert weights.shape == (dataset.vocab_size,)
        assert weights.isfinite().all()
        assert weights.min().item() > 0
        assert abs(weights.sum().item() - 1.0) < 1e-6
