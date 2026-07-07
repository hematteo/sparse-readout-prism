"""Unit tests for research.query_decompose — the query machinery behind the
benchmark-derived suites and the taxonomy figure metrics.

Synthetic tensors only; logits are constructed as h @ W.T so the decomposition
identity must hold to float roundoff. No model loads, no network.
"""

from __future__ import annotations

import torch

from sparse_readout_prism.research.query_decompose import (
    decompose_query,
    resolve_single_token,
    token_rank_and_prob,
)


class _StubTok:
    _TABLE = {
        " cat": [5],
        "cat": [6],
        " dog": [7, 8],
        "dog": [9, 10],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._TABLE.get(text, [1, 2, 3]))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " tok" + "_".join(str(i) for i in ids)

    def __len__(self) -> int:
        return 64


def test_resolve_single_token_prefers_space_variant_and_reports_used_text() -> None:
    tid, used, label, reason = resolve_single_token(_StubTok(), "cat")
    assert (tid, used, reason) == (5, " cat", "ok")
    assert label  # decoded display label recorded for the audit CSV


def test_resolve_single_token_multi_token_failure() -> None:
    tid, used, label, reason = resolve_single_token(_StubTok(), "dog")
    assert tid is None
    assert used == " dog"
    assert reason.startswith("multi_token(")


def test_token_rank_and_prob() -> None:
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
    rank, prob = token_rank_and_prob(logits, 1)
    assert rank == 1
    rank3, prob3 = token_rank_and_prob(logits, 3)
    assert rank3 == 2
    assert abs(torch.softmax(logits, dim=0)[3].item() - prob3) < 1e-9


def test_decompose_query_identity_and_residual_accounting() -> None:
    gen = torch.Generator().manual_seed(0)
    vocab, d_model, d_features, k = 32, 16, 48, 8
    W = torch.randn(vocab, d_model, generator=gen)
    h = torch.randn(d_model, generator=gen)
    logits = W @ h  # identity precondition: logits are exactly h . W[t]
    row_mean = W.mean(dim=0)
    encoder_w = torch.randn(d_features, d_model, generator=gen) * 0.2
    encoder_b = torch.zeros(d_features)
    decoder = torch.randn(d_features, d_model, generator=gen)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)

    weights = {3: 1.0, 11: -1.0}
    summary, feature_rows = decompose_query(
        h=h,
        logits=logits,
        W=W,
        row_mean=row_mean,
        decoder=decoder,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        k=k,
        weights=weights,
        mean_subtract=0.0,
    )
    # Exact score matches the direct margin, and the identity check agrees.
    direct = float(h @ (W[3] - W[11]))
    assert abs(summary["exact_score"] - direct) < 1e-4
    assert summary["identity_abs_error"] < 1e-4
    # approx + residual reconstructs exact by construction.
    assert abs(summary["approx_score"] + summary["residual"] - summary["exact_score"]) < 1e-5
    # Feature rows are consistent with the reported sparse sum.
    assert abs(sum(r["contribution"] for r in feature_rows) - summary["sparse_feature_sum"]) < 1e-5
    assert summary["n_active_features"] == len(feature_rows)
    assert summary["weight_sum"] == 0.0


def test_decompose_query_mean_subtract_shifts_base_not_identity() -> None:
    gen = torch.Generator().manual_seed(1)
    vocab, d_model, d_features = 16, 8, 24
    W = torch.randn(vocab, d_model, generator=gen)
    h = torch.randn(d_model, generator=gen)
    logits = W @ h
    row_mean = W.mean(dim=0)
    encoder_w = torch.randn(d_features, d_model, generator=gen) * 0.2
    encoder_b = torch.zeros(d_features)
    decoder = torch.randn(d_features, d_model, generator=gen)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)

    summary, _rows = decompose_query(
        h=h,
        logits=logits,
        W=W,
        row_mean=row_mean,
        decoder=decoder,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        k=4,
        weights={2: 1.0},
        mean_subtract=1.0,
    )
    direct = float(h @ (W[2] - row_mean))
    assert abs(summary["exact_score"] - direct) < 1e-4
    assert summary["identity_abs_error"] < 1e-4
