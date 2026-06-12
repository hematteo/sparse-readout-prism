"""Shared readout-query decomposition toolkit: specs, loaders, and decompose.

Extracted from ``scripts/figures/compute_general_readout_queries.py`` so the
benchmark-derived query suite (a run script) and the task-group example figure no
longer import experiment/support primitives from a figure module (a run->figures
/ figures->figures dependency inversion).

The block here is behaviour-preserving: ``QuerySpec``, ``write_csv``,
``load_qwen_model``, ``resolve_single_token``, ``token_rank_and_prob``,
``build_query_weights`` (with its ``merge_weight`` helper), ``decompose_query``,
and ``attach_feature_labels`` are moved verbatim from the figure module.

Note: this module keeps its OWN ``load_qwen_model``. It is NOT the same as
``research.qwen_readout.load_qwen_model`` (that one omits
``local_files_only`` and only tries ``AutoModelForImageTextToText``); this one
takes ``local_files_only`` and falls back from ``AutoModelForImageTextToText`` to
``AutoModelForCausalLM``. They are kept separate intentionally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.token_display import clean_token
from sparse_readout_prism.research.qwen_readout import (
    display_label_features,
    encode_topk,
    readable_feature_label,
)

__all__ = [
    "QuerySpec",
    "attach_feature_labels",
    "build_query_weights",
    "decompose_query",
    "load_qwen_model",
    "resolve_single_token",
    "token_rank_and_prob",
    "write_csv",
]


@dataclass(frozen=True)
class QuerySpec:
    case_id: str
    role: str
    query_kind: str
    title: str
    prompt: str
    note: str
    target: str | None = None
    target_a: str | None = None
    target_b: str | None = None
    target_family: tuple[str, ...] = ()
    contrast_family: tuple[str, ...] = ()
    top_competitors: int = 10
    competitor_rank: int = 5


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    # delegate to shared helper; write_empty=True preserves the 0-byte-file behaviour
    from sparse_readout_prism.utils import write_csv as _write_csv

    return _write_csv(path, rows, write_empty=True)


def load_qwen_model(model_id: str, dtype: torch.dtype, *, local_files_only: bool):
    from transformers import AutoTokenizer

    model_classes = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except Exception:  # noqa: BLE001
        pass
    try:
        from transformers import AutoModelForCausalLM

        model_classes.append(AutoModelForCausalLM)
    except Exception:  # noqa: BLE001
        pass
    if not model_classes:
        raise RuntimeError("no compatible transformers auto model class is available")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    last_error: Exception | None = None
    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"could not load {model_id}") from last_error


def resolve_single_token(tokenizer, raw: str) -> tuple[int | None, str, str, str]:
    """Resolve a paper token string to one tokenizer row.

    Returns token id, used text, decoded label, and reason. The reason is
    "ok" on success.
    """
    candidates = [raw]
    if not raw.startswith(" "):
        candidates.insert(0, " " + raw)
    seen: set[str] = set()
    for text in candidates:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            tid = int(ids[0])
            return tid, text, clean_token(tokenizer, tid), "ok"
    ids = tokenizer.encode(candidates[0], add_special_tokens=False)
    labels = [clean_token(tokenizer, int(tid)) for tid in ids]
    return None, candidates[0], ";".join(labels), f"multi_token({len(ids)})"


def token_rank_and_prob(logits: torch.Tensor, token_id: int) -> tuple[int, float]:
    rank = int((logits > logits[token_id]).sum().item()) + 1
    prob = float(torch.softmax(logits.float(), dim=0)[token_id].item())
    return rank, prob


def merge_weight(weights: dict[int, float], token_id: int, value: float) -> None:
    weights[int(token_id)] = float(weights.get(int(token_id), 0.0) + value)


def build_query_weights(
    *,
    spec: QuerySpec,
    logits: torch.Tensor,
    tokenizer,
    token_audit_rows: list[dict[str, object]],
) -> tuple[dict[int, float], float, dict[str, object]]:
    """Return token weights and vocabulary-mean subtraction weight."""
    mean_subtract = 0.0
    metadata: dict[str, object] = {}
    weights: dict[int, float] = {}

    def resolve(raw: str, role: str) -> int | None:
        tid, used, label, reason = resolve_single_token(tokenizer, raw)
        token_audit_rows.append(
            {
                "case_id": spec.case_id,
                "query_kind": spec.query_kind,
                "role": role,
                "raw": raw,
                "used_text": used,
                "token_id": "" if tid is None else tid,
                "token_label": label,
                "reason": reason,
            }
        )
        return tid

    if spec.query_kind == "selected_token":
        assert spec.target is not None
        tid = resolve(spec.target, "target")
        if tid is None:
            raise ValueError(f"{spec.case_id}: target did not resolve")
        weights[tid] = 1.0
        metadata.update({"target_token_id": tid, "target_label": clean_token(tokenizer, tid)})
        return weights, mean_subtract, metadata

    if spec.query_kind == "target_vs_vocab_mean":
        assert spec.target is not None
        tid = resolve(spec.target, "target")
        if tid is None:
            raise ValueError(f"{spec.case_id}: target did not resolve")
        weights[tid] = 1.0
        mean_subtract = 1.0
        metadata.update({"target_token_id": tid, "target_label": clean_token(tokenizer, tid)})
        return weights, mean_subtract, metadata

    if spec.query_kind == "pairwise_margin":
        assert spec.target_a is not None and spec.target_b is not None
        aid = resolve(spec.target_a, "A")
        bid = resolve(spec.target_b, "B")
        if aid is None or bid is None:
            raise ValueError(f"{spec.case_id}: pair token did not resolve")
        weights[aid] = 1.0
        merge_weight(weights, bid, -1.0)
        metadata.update(
            {
                "target_token_id": aid,
                "target_label": clean_token(tokenizer, aid),
                "contrast_token_ids": str(bid),
                "contrast_labels": clean_token(tokenizer, bid),
            }
        )
        return weights, mean_subtract, metadata

    if spec.query_kind == "token_family_margin":
        a_ids = [tid for raw in spec.target_family if (tid := resolve(raw, "target_family")) is not None]
        b_ids = [tid for raw in spec.contrast_family if (tid := resolve(raw, "contrast_family")) is not None]
        if not a_ids or not b_ids:
            raise ValueError(f"{spec.case_id}: empty resolved token family")
        for tid in a_ids:
            merge_weight(weights, tid, 1.0 / len(a_ids))
        for tid in b_ids:
            merge_weight(weights, tid, -1.0 / len(b_ids))
        metadata.update(
            {
                "target_token_id": "",
                "target_label": "; ".join(clean_token(tokenizer, tid) for tid in a_ids),
                "contrast_token_ids": ";".join(str(tid) for tid in b_ids),
                "contrast_labels": "; ".join(clean_token(tokenizer, tid) for tid in b_ids),
            }
        )
        return weights, mean_subtract, metadata

    if spec.query_kind == "winner_vs_topk":
        top_count = max(2, spec.top_competitors + 1)
        top_values, top_ids = torch.topk(logits, k=min(top_count, logits.numel()))
        winner = int(top_ids[0].item())
        competitor_ids = [int(tid.item()) for tid in top_ids[1 : spec.top_competitors + 1]]
        competitor_logits = top_values[1 : spec.top_competitors + 1].float()
        competitor_weights = torch.softmax(competitor_logits, dim=0)
        weights[winner] = 1.0
        for tid, weight in zip(competitor_ids, competitor_weights.tolist()):
            merge_weight(weights, tid, -float(weight))
        metadata.update(
            {
                "target_token_id": winner,
                "target_label": clean_token(tokenizer, winner),
                "contrast_token_ids": ";".join(str(tid) for tid in competitor_ids),
                "contrast_labels": "; ".join(clean_token(tokenizer, tid) for tid in competitor_ids),
                "competitor_weight_entropy": float(
                    -(competitor_weights * torch.log(competitor_weights.clamp_min(1e-12))).sum().item()
                ),
            }
        )
        return weights, mean_subtract, metadata

    if spec.query_kind == "winner_vs_rank":
        rank = max(2, int(spec.competitor_rank))
        top_values, top_ids = torch.topk(logits, k=min(rank, logits.numel()))
        winner = int(top_ids[0].item())
        competitor = int(top_ids[-1].item())
        weights[winner] = 1.0
        merge_weight(weights, competitor, -1.0)
        metadata.update(
            {
                "target_token_id": winner,
                "target_label": clean_token(tokenizer, winner),
                "contrast_token_ids": str(competitor),
                "contrast_labels": clean_token(tokenizer, competitor),
                "competitor_rank": rank,
                "competitor_logit": float(top_values[-1].item()),
            }
        )
        return weights, mean_subtract, metadata

    raise ValueError(f"unknown query kind {spec.query_kind}")


@torch.no_grad()
def decompose_query(
    *,
    h: torch.Tensor,
    logits: torch.Tensor,
    W: torch.Tensor,
    row_mean: torch.Tensor,
    decoder: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    k: int,
    weights: dict[int, float],
    mean_subtract: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    token_ids = sorted(weights)
    rows = W[token_ids].float()
    centered = rows - row_mean
    norms = centered.norm(dim=1).clamp_min(1e-8)
    x = centered / norms[:, None]
    z = encode_topk(x, encoder_w, encoder_b, k=k)
    alpha = torch.tensor([weights[tid] for tid in token_ids], dtype=torch.float32)
    coeff = (alpha[:, None] * norms[:, None] * z).sum(dim=0)
    active = torch.nonzero(coeff != 0, as_tuple=False).flatten()
    feature_scores = h @ decoder[active].T
    contributions = coeff[active] * feature_scores

    h_mean = float((h @ row_mean).item())
    exact = float(sum(weights[tid] * float(logits[tid].item()) for tid in token_ids) - mean_subtract * h_mean)
    base = float((sum(weights.values()) - mean_subtract) * h_mean)
    feature_sum = float(contributions.sum().item())
    approx = base + feature_sum
    residual = exact - approx
    query_row = sum(weights[tid] * W[tid].float() for tid in token_ids) - mean_subtract * row_mean
    identity_error = abs(float((h @ query_row).item()) - exact)

    feature_rows: list[dict[str, object]] = []
    for rank, (fid, value, score, coef) in enumerate(
        zip(active.tolist(), contributions.tolist(), feature_scores.tolist(), coeff[active].tolist()),
        start=1,
    ):
        feature_rows.append(
            {
                "feature_id": int(fid),
                "feature_rank_active": rank,
                "contribution": float(value),
                "abs_contribution": abs(float(value)),
                "feature_hidden_score": float(score),
                "query_feature_coeff": float(coef),
            }
        )
    summary = {
        "exact_score": exact,
        "base_term": base,
        "sparse_feature_sum": feature_sum,
        "approx_score": approx,
        "residual": residual,
        "residual_abs_over_direct": abs(residual) / max(abs(exact), 1e-8),
        "sign_preserved": int(np.sign(exact) == np.sign(approx)),
        "n_active_features": len(active),
        "mean_subtract": mean_subtract,
        "weight_sum": float(sum(weights.values())),
        "identity_abs_error": identity_error,
    }
    return summary, feature_rows


def attach_feature_labels(
    *,
    W: torch.Tensor,
    row_mean: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer,
    display_rows: list[dict[str, object]],
    top_tokens: int,
    chunk_size: int,
) -> None:
    feature_ids = sorted({int(row["feature_id"]) for row in display_rows})
    labels = display_label_features(
        W=W,
        row_mean=row_mean,
        feature_ids=feature_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=top_tokens,
        chunk_size=chunk_size,
    )
    for row in display_rows:
        fid = int(row["feature_id"])
        row["feature_label"] = readable_feature_label(labels.get(fid, []), fid)
        row["feature_top_tokens"] = "; ".join(labels.get(fid, []))
