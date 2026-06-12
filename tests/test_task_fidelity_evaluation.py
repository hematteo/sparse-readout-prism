"""Correctness tests for the Experiment 2 scorer (scripts/eval/task_fidelity_evaluation.py).

The scorer is the load-bearing Experiment 2 (Part A) code and shipped without a
committed test. Part 1 pins the decision-fidelity math with known-answer inputs
(it bypasses the factorizer and sets W_rec directly, so the asserted values are
exact). Part 2 exercises the full checkpoint -> reconstruct -> score -> JSON
path with a minimal real factorizer.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "task_fidelity_evaluation", ROOT / "scripts" / "eval" / "task_fidelity_evaluation.py"
)
s7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s7)

from sparse_readout_prism.factorizers import build_factorizer  # noqa: E402

DEV = torch.device("cpu")


def _margin_items(n: int, vocab: int, gen: torch.Generator) -> list[dict]:
    items = []
    for i in range(n):
        a = torch.randint(0, vocab, (2,), generator=gen)
        b = torch.randint(0, vocab, (2,), generator=gen)
        items.append(
            {
                "h_idx": i,
                "A_tokens": a,
                "B_tokens": b,
                "battery": "synthA" if i % 2 else "synthB",
                "pair_id": f"p{i}",
            }
        )
    return items


def test_perfect_reconstruction_is_ideal():
    """W_rec == W_orig => pearson 1, sign-agreement 1, ~zero residual."""
    gen = torch.Generator().manual_seed(0)
    vocab, d = 50, 16
    W = torch.randn(vocab, d, generator=gen)
    h = torch.randn(12, d, generator=gen)
    items = _margin_items(12, vocab, gen)

    m = s7.margin_metrics(h, W, W.clone(), items, DEV)

    assert m["n"] == 12
    assert m["pearson"] == 1.0 or abs(m["pearson"] - 1.0) < 1e-9
    assert abs(m["spearman"] - 1.0) < 1e-9
    assert m["sign_agreement"] == 1.0
    assert m["median_resid_direct"] < 1e-5
    assert m["p90_resid_direct"] < 1e-5
    # strata are populated and consistent
    assert set(m["by_battery"]) == {"synthA", "synthB"}
    assert set(m["by_margin_magnitude"]) <= {"low", "mid", "high"}


def test_uniform_scaling_preserves_correlation_but_scales_residual():
    """W_rec == 0.5*W_orig => pearson still 1, residual exactly 0.5, slope 0.5."""
    gen = torch.Generator().manual_seed(1)
    vocab, d = 50, 16
    W = torch.randn(vocab, d, generator=gen)
    h = torch.randn(20, d, generator=gen)
    items = _margin_items(20, vocab, gen)

    m = s7.margin_metrics(h, W, 0.5 * W, items, DEV)

    assert abs(m["pearson"] - 1.0) < 1e-9
    assert m["sign_agreement"] == 1.0
    assert abs(m["median_resid_direct"] - 0.5) < 1e-6
    assert abs(m["calibration_slope"] - 0.5) < 1e-6


def test_sign_flip_is_detected():
    """W_rec == -W_orig => anticorrelated, zero sign-agreement, residual 2.0."""
    gen = torch.Generator().manual_seed(2)
    vocab, d = 50, 16
    W = torch.randn(vocab, d, generator=gen)
    h = torch.randn(16, d, generator=gen)
    items = _margin_items(16, vocab, gen)

    m = s7.margin_metrics(h, W, -W, items, DEV)

    assert abs(m["pearson"] + 1.0) < 1e-9
    assert m["sign_agreement"] == 0.0
    assert abs(m["median_resid_direct"] - 2.0) < 1e-6


def test_family_weights_are_applied():
    """_family_vec with weights == manual weighted mean; default == plain mean."""
    W = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    toks = torch.tensor([0, 1, 2])
    w = torch.tensor([1.0, 0.0, 1.0])
    got = s7._family_vec(W, toks, w)
    assert torch.allclose(got, torch.tensor([0.5, 2.5]))
    assert torch.allclose(s7._family_vec(W, toks, None), W.mean(0))


def test_query_metrics_linear_and_rank_under_identity():
    """Linear + rank-preservation queries are ideal when W_rec == W_orig."""
    gen = torch.Generator().manual_seed(3)
    vocab, d = 40, 12
    W = torch.randn(vocab, d, generator=gen)
    h = torch.randn(8, d, generator=gen)
    # Make token 5 the unambiguous argmax over candidate_set [5,6,7,8,9] for
    # h[0], so target_top1_preserved is well-defined: it must be 1.0 under
    # identity reconstruction and drop when the target row is degraded.
    basis = torch.zeros(d)
    basis[0] = 1.0
    W[5] = 10.0 * basis
    W[6:10] = 0.01 * torch.randn(4, d, generator=gen)
    h[0] = basis
    q_items = []
    for i in range(8):
        q_items.append(
            {
                "h_idx": i,
                "query_type": "authority",
                "signed": True,
                "battery": "q",
                "query_id": f"lin{i}",
                "tokens": torch.tensor([i % vocab, (i + 1) % vocab]),
                "weights": torch.tensor([1.0, -1.0]),
            }
        )
    q_items.append(
        {
            "h_idx": 0,
            "query_type": "target_vs_rest",
            "battery": "q",
            "query_id": "rank0",
            "target_token": 5,
            "candidate_set": torch.tensor([5, 6, 7, 8, 9]),
        }
    )
    out = s7.query_metrics(h, W, W.clone(), q_items, DEV)
    assert abs(out["linear"]["pearson"] - 1.0) < 1e-9
    assert out["linear"]["median_resid_direct"] < 1e-5
    assert out["linear"]["sign_agreement_signed"] == 1.0
    assert out["rank_preservation"]["target_top1_preserved"] == 1.0
    assert abs(out["rank_preservation"]["mean_candidate_spearman"] - 1.0) < 1e-9

    # Degrading the target row must drop top1 preservation: the metric reacts.
    W_bad = W.clone()
    W_bad[5] = torch.zeros(d)
    bad = s7.query_metrics(h, W, W_bad, q_items, DEV)
    assert bad["rank_preservation"]["target_top1_preserved"] == 0.0


def test_pure_helpers():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(s7._pearson(a, 2 * a + 1) - 1.0) < 1e-9
    assert abs(s7._pearson(a, -a) + 1.0) < 1e-9
    rd = s7._resid_direct(torch.tensor([2.0, -4.0]), torch.tensor([1.0, -2.0]))
    assert torch.allclose(rd, torch.tensor([0.5, 0.5]))


def test_full_cli_roundtrip(tmp_path: Path):
    """checkpoint.pt + config.yaml + data + bank -> metrics_task_fidelity.json, idempotent."""
    torch.manual_seed(7)
    d_model, vocab, d_features, k = 16, 40, 64, 8

    cell = tmp_path / "cell"
    cell.mkdir()
    model = build_factorizer(
        {"factorizer": {"architecture": "topk", "d_features": d_features, "k": k}},
        d_model=d_model,
    )
    torch.save({"model_state_dict": model.state_dict()}, cell / "checkpoint.pt")
    (cell / "config.yaml").write_text(
        yaml.safe_dump({"factorizer": {"architecture": "topk", "d_features": d_features, "k": k}})
    )

    data_p = tmp_path / "data.pt"
    torch.save({"W_U_orig": torch.randn(vocab, d_model)}, data_p)

    gen = torch.Generator().manual_seed(8)
    bank = {
        "h": torch.randn(10, d_model, generator=gen),
        "margin_items": _margin_items(10, vocab, gen),
        "query_items": [
            {
                "h_idx": 0,
                "query_type": "authority",
                "signed": True,
                "battery": "b",
                "query_id": "q0",
                "tokens": torch.tensor([1, 2]),
                "weights": torch.tensor([1.0, -1.0]),
            }
        ],
    }
    bank_p = tmp_path / "bank.pt"
    torch.save(bank, bank_p)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval" / "task_fidelity_evaluation.py"),
        "--cell-dir",
        str(cell),
        "--data-path",
        str(data_p),
        "--bank",
        str(bank_p),
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    out = json.loads((cell / "metrics_task_fidelity.json").read_text())
    assert out["complete"] is True
    assert out["n_h"] == 10
    assert out["k"] == k
    mar = out["margins"]
    assert mar["n"] == 10
    assert -1.0 <= mar["pearson"] <= 1.0
    assert 0.0 <= mar["sign_agreement"] <= 1.0
    assert mar["median_resid_direct"] >= 0.0
    assert mar["p90_resid_direct"] >= mar["median_resid_direct"]
    assert out["queries"]["linear"]["n"] == 1

    # idempotent: rerun without --force skips and leaves the file complete
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "skip" in (r2.stdout + r2.stderr).lower()
    assert json.loads((cell / "metrics_task_fidelity.json").read_text())["complete"] is True
