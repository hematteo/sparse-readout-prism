import torch

from sparse_readout_prism.research.cell_metrics import coverage_stats


def test_coverage_stats_values():
    fm = torch.tensor([3.0, -1.0, 0.5, -2.0])
    out = coverage_stats(fm, sparse_margin=float(fm.sum()))
    assert set(out) == {
        "top5_signed_cov",
        "top10_signed_cov",
        "top5_abs_cov",
        "top10_abs_cov",
        "n_feat_80pct_abs",
        "largest_pos_feat",
        "largest_pos_contrib",
        "largest_neg_feat",
        "largest_neg_contrib",
    }
    assert out["largest_pos_feat"] == 0 and out["largest_pos_contrib"] == 3.0
    assert out["largest_neg_feat"] == 3 and out["largest_neg_contrib"] == -2.0
    # |3|+|1|+|0.5|+|2| = 6.5; cumulative <0.8 after 3/6.5 and 5/6.5 -> 2, +1
    assert out["n_feat_80pct_abs"] == 3
    assert 0.0 < out["top5_abs_cov"] <= 1.0


def test_coverage_stats_zero_margin_guarded():
    out = coverage_stats(torch.zeros(4), sparse_margin=0.0)
    assert out["top5_signed_cov"] == 0.0
    assert out["top5_abs_cov"] == 0.0
    assert out["n_feat_80pct_abs"] == 0
