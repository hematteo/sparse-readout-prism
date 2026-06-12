from __future__ import annotations

import json
import math
from pathlib import Path

from sparse_readout_prism.runner import run_experiment
from sparse_readout_prism.utils import load_yaml


def test_runner_smoke_produces_finite_score_and_artifacts(tmp_path: Path) -> None:
    config = load_yaml("configs/smoke.yaml")
    # Force the synthetic fallback so the test is hermetic and CPU-only.
    config["data"]["path"] = "/path/that/does/not/exist.pt"
    config["run"]["device"] = "cpu"
    config["run"]["output_dir"] = str(tmp_path)
    config.setdefault("artifacts", {})["save_checkpoint"] = True
    config["artifacts"]["report_path"] = str(tmp_path / "smoke.md")

    metrics = run_experiment(config)

    assert math.isfinite(float(metrics["selection_score"]))
    assert (tmp_path / "checkpoint.pt").exists()
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    assert math.isfinite(float(json.loads(metrics_path.read_text())["selection_score"]))
