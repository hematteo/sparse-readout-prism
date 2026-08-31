"""CI-safe tests for the Appendix producer scripts added to close reproduction gaps:

* scripts/analyze/compute_row_norm_tail_stats.py  (tab:app-k-row-norm-tail)
* scripts/eval/dense_control_diagnostic.py         (Appendix F.1 omp_diagnostic)
* scripts/analyze/audit_feature_labels.py          (Appendix L feature-label audit)

No GPU, no model load, no network: row-norm and dense-control run on synthetic
W_U + an untrained TopK SAE; the audit aggregator runs on the shipped CSV/JSON.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rnt = _load("compute_row_norm_tail_stats", "scripts/analyze/compute_row_norm_tail_stats.py")
dcd = _load("dense_control_diagnostic", "scripts/eval/dense_control_diagnostic.py")
fla = _load("audit_feature_labels", "scripts/analyze/audit_feature_labels.py")

from sparse_readout_prism.data import preprocess_rows  # noqa: E402
from sparse_readout_prism.factorizers import TopKSAE  # noqa: E402


# --------------------------------------------------------------------------- #
# row-norm tail
# --------------------------------------------------------------------------- #


def test_row_norm_tail_stats_shape_and_ordering():
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(500, 32, generator=gen)
    # Inject a heavy upper tail on a few rows.
    W[:5] *= 6.0
    stats = rnt.row_norm_tail_stats(W, center="masked_mean")
    assert set(stats) == {"n_rows", "min", "p50", "max", "mean", "cov", "max_over_med"}
    assert stats["n_rows"] == 500
    assert stats["min"] <= stats["p50"] <= stats["max"]
    assert stats["cov"] > 0.0
    assert stats["max_over_med"] >= 1.0  # heavy tail -> max well above median


def test_row_norm_tail_token_mask_filters_rows():
    gen = torch.Generator().manual_seed(1)
    W = torch.randn(100, 16, generator=gen)
    mask = torch.zeros(100, dtype=torch.bool)
    mask[:40] = True
    stats = rnt.row_norm_tail_stats(W, token_mask=mask, center="masked_mean")
    assert stats["n_rows"] == 40


def test_row_norm_center_none_matches_raw_norm():
    W = torch.tensor([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])  # raw norms 5, 0, 10
    stats = rnt.row_norm_tail_stats(W, center="none")
    assert abs(stats["max"] - 10.0) < 1e-5
    assert abs(stats["min"] - 0.0) < 1e-5


# --------------------------------------------------------------------------- #
# dense / negative-control diagnostic
# --------------------------------------------------------------------------- #


def test_row_explained_variance_known_values():
    X = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert abs(dcd.row_explained_variance(X, X.clone()) - 1.0) < 1e-6
    assert abs(dcd.row_explained_variance(X, torch.zeros_like(X)) - 0.0) < 1e-6


def test_dense_control_diagnostic_orderings():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(256, 24, generator=gen)
    _, _, X = preprocess_rows(W)
    d_features, k = 64, 8
    model = TopKSAE(d_model=24, d_features=d_features, k=k).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    res = dcd.run_diagnostic(
        rows_normalized=X,
        model=model,
        k=k,
        methods=list(dcd.ALL_METHODS),
        setting="synthetic",
        n_boot=0,
        seed=0,
        device=torch.device("cpu"),
    )
    ev = {r["method_key"]: r["rowEV"] for r in res}
    assert set(ev) == set(dcd.ALL_METHODS)
    for v in ev.values():
        assert v <= 1.0 + 1e-6
    # LS refit on the encoder's own support is the optimal-coefficient version of
    # what the encoder does on that support, so it cannot be worse.
    assert ev["ls_support"] >= ev["encoder"] - 1e-4
    # Non-negativity is a constraint on the same support -> no better than signed LS.
    assert ev["nnls_support"] <= ev["ls_support"] + 1e-4
    # method_class tags match the recorded omp_diagnostic schema.
    classes = {r["method_key"]: r["method_class"] for r in res}
    assert classes["encoder"] == "encoder"
    assert classes["dense_rank"] == "dense"
    assert classes["omp_signed"] == "omp"


# --------------------------------------------------------------------------- #
# feature-label audit aggregator + shipped data self-consistency
# --------------------------------------------------------------------------- #


def test_feature_label_audit_aggregate_matches_shipped_counts():
    ann_path = ROOT / "data" / "audit" / "feature_label_audit_annotations.csv"
    ref_path = ROOT / "data" / "audit" / "feature_label_audit.json"
    annotations = fla._read_annotations(ann_path)
    counts = fla.tally_annotations(annotations)
    by_set = {r["feature_set"]: r for r in counts}

    # Shipped annotations cover the main bug panels; reproduce that paper row.
    main = by_set["Main bug panels"]
    assert (main["labels"], main["coherent"], main["ambiguous"], main["token_form"]) == (11, 8, 0, 3)
    assert main["target_consistent"] == 8

    # Annotated set agrees with the recorded paper count for that set.
    ref = json.loads(ref_path.read_text())
    ref_by_set = {r["feature_set"]: r for r in ref["feature_sets"]}
    for key in ("labels", "coherent", "ambiguous", "token_form"):
        assert ref_by_set["Main bug panels"][key] == main[key]


def test_recorded_audit_totals_sum():
    ref = json.loads((ROOT / "data" / "audit" / "feature_label_audit.json").read_text())
    total = ref["total"]
    for key in ("labels", "coherent", "ambiguous", "token_form", "target_consistent"):
        assert sum(s[key] for s in ref["feature_sets"]) == total[key]
    # Audit categories partition the labels.
    assert total["coherent"] + total["ambiguous"] + total["token_form"] == total["labels"]
