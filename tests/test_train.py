"""train.py: L0-controller engagement and the train_ckpt.pt resume path.

Neither was previously executed by any test (the runner smoke only trains a
matryoshka_topk cell with no controller and no interruption). Tiny synthetic
dataset, CPU, seconds.
"""

from __future__ import annotations

import torch

from sparse_readout_prism.data import load_prism_dataset
from sparse_readout_prism.factorizers import build_factorizer
from sparse_readout_prism.train import train_factorizer
from sparse_readout_prism.utils import set_seed

_DEVICE = torch.device("cpu")


def _dataset():
    return load_prism_dataset({"data": {"max_hidden": 64, "val_hidden": 16, "max_rows": 256}}, seed=7)


def test_l0_controller_engages_for_jumprelu() -> None:
    set_seed(0)
    dataset = _dataset()
    config = {
        "factorizer": {"architecture": "jumprelu", "d_features": 64},
        "training": {
            "steps": 30,
            "batch_size": 32,
            "log_every": 10,
            "checkpoint_every": 0,
            "l0_controller": {
                "enabled": True,
                "target_l0": 8,
                "warmup_steps": 5,
                "control_period": 2,
                "coeff_init": 1e-3,
                "coeff_min": 1e-8,
                "coeff_max": 0.01,
            },
        },
    }
    model = build_factorizer(config, d_model=dataset.d_model)
    history = train_factorizer(model, dataset, config, _DEVICE, output_dir=None)
    assert history["l0_controller_enabled"] is True
    assert history["l0_controller_target"] == 8.0
    coeff = history["l0_controller_final_coeff"]
    assert isinstance(coeff, float) and 1e-8 <= coeff <= 0.01
    assert history["steps_run"] == 30


def test_l0_controller_stays_off_for_topk() -> None:
    set_seed(0)
    dataset = _dataset()
    config = {
        "factorizer": {"architecture": "topk", "d_features": 64, "k": 8},
        # Controller requested but topk is k-controlled: must stay disabled.
        "training": {"steps": 5, "batch_size": 32, "l0_controller": {"enabled": True, "target_l0": 8}},
    }
    model = build_factorizer(config, d_model=dataset.d_model)
    history = train_factorizer(model, dataset, config, _DEVICE, output_dir=None)
    assert history["l0_controller_enabled"] is False
    assert history["l0_controller_final_coeff"] is None


def test_train_ckpt_resume_continues_from_saved_step(tmp_path) -> None:
    dataset = _dataset()
    factorizer_cfg = {"architecture": "topk", "d_features": 32, "k": 8}

    set_seed(0)
    cfg_first = {"factorizer": factorizer_cfg, "training": {"steps": 3, "batch_size": 16, "checkpoint_every": 3}}
    model = build_factorizer(cfg_first, d_model=dataset.d_model)
    h1 = train_factorizer(model, dataset, cfg_first, _DEVICE, output_dir=tmp_path)
    assert h1["steps_run"] == 3
    assert (tmp_path / "train_ckpt.pt").exists()

    # Same output_dir, longer horizon: resumes at step 4 instead of restarting.
    set_seed(0)
    cfg_more = {"factorizer": factorizer_cfg, "training": {"steps": 5, "batch_size": 16, "checkpoint_every": 5}}
    model2 = build_factorizer(cfg_more, d_model=dataset.d_model)
    h2 = train_factorizer(model2, dataset, cfg_more, _DEVICE, output_dir=tmp_path)
    assert h2["steps_run"] == 2  # steps 4..5 only
    steps_logged = [row["step"] for row in h2["losses"]]
    assert 3 in steps_logged and 5 in steps_logged  # history carried over

    # Horizon already reached: no training at all.
    model3 = build_factorizer(cfg_more, d_model=dataset.d_model)
    h3 = train_factorizer(model3, dataset, cfg_more, _DEVICE, output_dir=tmp_path)
    assert h3["steps_run"] == 0
