"""Mine same-token, different-context Sparse Readout Prism examples.

This script avoids A-vs-B token contrasts.  It keeps the target token fixed
and asks whether different contexts use different sparse readout features to
support the same centered token score:

    h^T (W_U[target] - mean_vocab_row)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import torch

from sparse_readout_prism.research.data.mine_polysemy_pairs import (
    candidate_pairs,
    domain_candidate_pairs,
    paper_relevant_candidate_pairs,
    pos_candidate_pairs,
)
from sparse_readout_prism.research._common.qwen_readout import (
    clean_token,
    collect_readout_states_batched,
    encode_topk,
    find_lm_head,
    display_label_features,
    load_qwen_model,
    load_sae,
    parse_single_token,
    readable_feature_label,
    token_summary,
)

DEFAULT_OUT_DIR = Path("results/qwen2b_32x_polysemy_features_20260519")
DEFAULT_PAPER_DIR = Path("paper/figures/qwen2b_32x_polysemy_features")

GENERIC_LABEL_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "form",
    "forms",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_label_piece(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def split_feature_tokens(row: dict[str, object]) -> list[str]:
    return [part.strip() for part in str(row.get("feature_top_tokens", "")).split(";") if part.strip()]


def is_uninteresting_label(row: dict[str, object]) -> bool:
    tokens = split_feature_tokens(row)[:4]
    if not tokens:
        return True
    target = normalize_label_piece(row.get("target_label", ""))
    pieces = [normalize_label_piece(token) for token in tokens]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return True
    generic = sum(1 for piece in pieces if piece in GENERIC_LABEL_TOKENS or len(piece) == 1)
    if generic >= max(2, len(pieces) - 1):
        return True
    if target:
        lexical = sum(1 for piece in pieces if target in piece or piece in target)
        if lexical == len(pieces):
            return True
    return False


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    from sparse_readout_prism.utils import write_csv as _write_csv

    # delegate; mkdir_parents=False preserves the original no-mkdir behaviour
    return _write_csv(path, rows, mkdir_parents=False)


def contribution_cosine(rows_a: list[dict[str, object]], rows_b: list[dict[str, object]]) -> float:
    by_a = {int(row["feature_id"]): float(row["contribution"]) for row in rows_a}
    by_b = {int(row["feature_id"]): float(row["contribution"]) for row in rows_b}
    fids = sorted(set(by_a) | set(by_b))
    if not fids:
        return float("nan")
    a = torch.tensor([by_a.get(fid, 0.0) for fid in fids], dtype=torch.float32)
    b = torch.tensor([by_b.get(fid, 0.0) for fid in fids], dtype=torch.float32)
    denom = float(a.norm() * b.norm())
    if denom <= 1e-8:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def top_feature_jaccard(rows_a: list[dict[str, object]], rows_b: list[dict[str, object]], top_n: int) -> float:
    top_a = {
        int(row["feature_id"])
        for row in sorted(rows_a, key=lambda item: -float(item["contribution"]))[:top_n]
        if float(row["contribution"]) > 0
    }
    top_b = {
        int(row["feature_id"])
        for row in sorted(rows_b, key=lambda item: -float(item["contribution"]))[:top_n]
        if float(row["contribution"]) > 0
    }
    union = top_a | top_b
    if not union:
        return float("nan")
    return len(top_a & top_b) / len(union)


@torch.no_grad()
def compute_rows(
    *,
    pairs: list[dict[str, str]],
    model,
    tokenizer,
    checkpoint: Path,
    k: int,
    model_id: str,
    device: torch.device,
    batch_size: int,
    rank_top_n: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    lm_head, lm_head_path = find_lm_head(model)
    W = lm_head.weight.detach().cpu()
    vocab, d_model = W.shape
    decoder, encoder_w, encoder_b, sae_config = load_sae(checkpoint)
    if decoder.shape[1] != d_model:
        raise ValueError(f"SAE d_model {decoder.shape[1]} does not match W_U d_model {d_model}")
    row_mean = W.float().mean(dim=0)

    contexts: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for pair in pairs:
        try:
            target_id = parse_single_token(tokenizer, pair["target"])
        except ValueError as exc:
            skipped.append({**pair, "skip_reason": str(exc)})
            continue
        for side in ("left", "right"):
            contexts.append(
                {
                    "pair_id": pair["pair_id"],
                    "side": side,
                    "target": pair["target"],
                    "target_id": target_id,
                    "target_label": clean_token(tokenizer, target_id),
                    "context_label": pair[f"{side}_label"],
                    "prompt": pair[f"{side}_prompt"],
                }
            )

    states, logits_list = collect_readout_states_batched(
        model,
        tokenizer,
        [str(row["prompt"]) for row in contexts],
        device,
        batch_size=batch_size,
    )

    context_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    z_cache: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for context_idx, (context, h, logits) in enumerate(zip(contexts, states, logits_list)):
        target_id = int(context["target_id"])
        if target_id not in z_cache:
            centered = W[target_id].float() - row_mean
            norm = centered.norm().clamp_min(1e-8)
            x = centered / norm
            z = encode_topk(x[None, :], encoder_w, encoder_b, k=k)[0]
            coeff = norm * z
            active = torch.nonzero(coeff != 0, as_tuple=False).flatten()
            z_cache[target_id] = (coeff, active, float(norm))
        coeff, active, row_norm = z_cache[target_id]
        feature_scores = h @ decoder[active].T
        contrib = coeff[active] * feature_scores
        sparse_sum = float(contrib.sum())
        exact_centered = float(h @ (W[target_id].float() - row_mean))
        raw_logit = float(logits[target_id])
        mean_logit = float(logits[:vocab].mean())
        target_rank = int((logits[:vocab] > logits[target_id]).sum().item() + 1)
        top_token_id = int(torch.argmax(logits[:vocab]).item())
        residual = exact_centered - sparse_sum
        row = {
            "context_id": f"{context['pair_id']}::{context['side']}",
            "pair_id": context["pair_id"],
            "side": context["side"],
            "context_label": context["context_label"],
            "model_id": model_id,
            "checkpoint": str(checkpoint),
            "k": k,
            "prompt": context["prompt"],
            "target": context["target"],
            "target_id": target_id,
            "target_label": context["target_label"],
            "raw_logit": raw_logit,
            "mean_vocab_logit": mean_logit,
            "exact_centered_logit": exact_centered,
            "sparse_feature_sum": sparse_sum,
            "residual": residual,
            "residual_abs_over_direct": abs(residual) / max(abs(exact_centered), 1e-8),
            "target_rank": target_rank,
            "top_token_id": top_token_id,
            "top_token_label": clean_token(tokenizer, top_token_id),
            "target_row_norm_after_centering": row_norm,
            "lm_head_path": lm_head_path,
            "vocab": vocab,
            "d_model": d_model,
            "context_index": context_idx,
        }
        rows_for_context: list[dict[str, object]] = []
        for fid, value, score, coef in zip(
            active.tolist(),
            contrib.tolist(),
            feature_scores.tolist(),
            coeff[active].tolist(),
        ):
            rows_for_context.append(
                {
                    **row,
                    "feature_id": int(fid),
                    "contribution": float(value),
                    "contribution_abs": abs(float(value)),
                    "feature_hidden_score": float(score),
                    "row_feature_coeff": float(coef),
                    "target_feature_activation": float(z_cache[target_id][0][int(fid)] / max(row_norm, 1e-8)),
                    "supports": "target" if float(value) >= 0 else "anti-target",
                    "contrast_token_hint": context["target_label"],
                }
            )
        rows_for_context.sort(key=lambda item: -float(item["contribution_abs"]))
        positive_top = [
            item
            for item in sorted(rows_for_context, key=lambda item: -float(item["contribution"]))
            if float(item["contribution"]) > 0
        ]
        row["top_positive_feature_sum"] = sum(float(item["contribution"]) for item in positive_top[:rank_top_n])
        row["top_positive_feature_ids"] = ",".join(str(int(item["feature_id"])) for item in positive_top[:rank_top_n])
        context_rows.append(row)
        feature_rows.extend(rows_for_context)

    features_by_context: dict[str, list[dict[str, object]]] = {}
    for row in feature_rows:
        features_by_context.setdefault(str(row["context_id"]), []).append(row)
    contexts_by_pair: dict[str, list[dict[str, object]]] = {}
    for row in context_rows:
        contexts_by_pair.setdefault(str(row["pair_id"]), []).append(row)

    pair_rows: list[dict[str, object]] = []
    for pair_id, rows in contexts_by_pair.items():
        if len(rows) != 2:
            continue
        left = next(row for row in rows if row["side"] == "left")
        right = next(row for row in rows if row["side"] == "right")
        left_features = features_by_context[str(left["context_id"])]
        right_features = features_by_context[str(right["context_id"])]
        cosine = contribution_cosine(left_features, right_features)
        jaccard = top_feature_jaccard(left_features, right_features, top_n=rank_top_n)
        max_resid = max(
            float(left["residual_abs_over_direct"]),
            float(right["residual_abs_over_direct"]),
        )
        min_centered = min(float(left["exact_centered_logit"]), float(right["exact_centered_logit"]))
        max_rank = max(int(left["target_rank"]), int(right["target_rank"]))
        divergence = (1.0 - cosine if math.isfinite(cosine) else 0.0) + (
            1.0 - jaccard if math.isfinite(jaccard) else 0.0
        )
        quality_penalty = max_resid + 0.02 * max(0, max_rank - 100) + 0.25 * max(0.0, 1.0 - min_centered)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "target": left["target"],
                "target_label": left["target_label"],
                "left_context_label": left["context_label"],
                "right_context_label": right["context_label"],
                "left_prompt": left["prompt"],
                "right_prompt": right["prompt"],
                "left_exact_centered_logit": left["exact_centered_logit"],
                "right_exact_centered_logit": right["exact_centered_logit"],
                "left_sparse_feature_sum": left["sparse_feature_sum"],
                "right_sparse_feature_sum": right["sparse_feature_sum"],
                "left_residual_abs_over_direct": left["residual_abs_over_direct"],
                "right_residual_abs_over_direct": right["residual_abs_over_direct"],
                "max_residual_abs_over_direct": max_resid,
                "left_target_rank": left["target_rank"],
                "right_target_rank": right["target_rank"],
                "max_target_rank": max_rank,
                "contribution_cosine": cosine,
                "top_positive_jaccard": jaccard,
                "profile_divergence_score": divergence,
                "selection_score": divergence - quality_penalty,
            }
        )

    meta = {
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "k": k,
        "sae_config": sae_config,
        "lm_head_path": lm_head_path,
        "vocab": int(vocab),
        "d_model": int(d_model),
        "skipped": skipped,
        "score_definition": "h dot (W_U[target] - mean_vocab_row)",
    }
    return (
        context_rows,
        feature_rows,
        pair_rows,
        W,
        row_mean,
        encoder_w,
        encoder_b,
        meta,
    )


def select_pairs(
    pair_rows: list[dict[str, object]],
    n_pairs: int,
    force_pair_ids: list[str] | None = None,
) -> list[dict[str, object]]:
    if force_pair_ids:
        by_id = {str(row["pair_id"]): row for row in pair_rows}
        missing = [pair_id for pair_id in force_pair_ids if pair_id not in by_id]
        if missing:
            raise ValueError(f"unknown forced pair ids: {missing}")
        selected = [by_id[pair_id] for pair_id in force_pair_ids]
        selected_ids = {str(row["pair_id"]) for row in selected}
        for row in pair_rows:
            row["selected_for_display"] = str(row["pair_id"]) in selected_ids
        return selected

    usable = [
        row
        for row in pair_rows
        if float(row["left_exact_centered_logit"]) > 0.0
        and float(row["right_exact_centered_logit"]) > 0.0
        and int(row["max_target_rank"]) <= 500
        and float(row["max_residual_abs_over_direct"]) <= 0.75
    ]
    if len(usable) < n_pairs:
        usable = [
            row
            for row in pair_rows
            if float(row["left_exact_centered_logit"]) > 0.0
            and float(row["right_exact_centered_logit"]) > 0.0
            and float(row["max_residual_abs_over_direct"]) <= 1.0
        ]
    usable.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    selected = usable[:n_pairs]
    selected_ids = {str(row["pair_id"]) for row in selected}
    for row in pair_rows:
        row["selected_for_display"] = str(row["pair_id"]) in selected_ids
    return selected


def label_display_rows(
    *,
    selected_pairs: list[dict[str, object]],
    context_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    W: torch.Tensor,
    row_mean: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer,
    top_features: int,
    label_top_tokens: int,
    label_chunk_size: int,
) -> list[dict[str, object]]:
    selected_pair_ids = {str(row["pair_id"]) for row in selected_pairs}
    selected_contexts = [row for row in context_rows if str(row["pair_id"]) in selected_pair_ids]
    features_by_context: dict[str, list[dict[str, object]]] = {}
    for row in feature_rows:
        if str(row["pair_id"]) in selected_pair_ids:
            features_by_context.setdefault(str(row["context_id"]), []).append(row)

    display_rows: list[dict[str, object]] = []
    for context in selected_contexts:
        rows = sorted(
            features_by_context.get(str(context["context_id"]), []),
            key=lambda row: -float(row["contribution"]),
        )
        positives = [row for row in rows if float(row["contribution"]) > 0]
        chosen = positives[:top_features] if len(positives) >= top_features else rows[:top_features]
        display_rows.extend(sorted(chosen, key=lambda row: float(row["contribution"])))

    label_ids = sorted({int(row["feature_id"]) for row in display_rows})
    labels = display_label_features(
        W=W,
        row_mean=row_mean,
        feature_ids=label_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=label_top_tokens,
        chunk_size=label_chunk_size,
    )
    for row in display_rows:
        fid = int(row["feature_id"])
        row["feature_label"] = readable_feature_label(labels.get(fid, []), fid)
        row["feature_top_tokens"] = "; ".join(labels.get(fid, []))
        row["feature_token_summary"] = token_summary(row)
    return display_rows


def label_delta_rows(
    *,
    selected_pairs: list[dict[str, object]],
    context_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    W: torch.Tensor,
    row_mean: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer,
    top_delta_features: int,
    label_top_tokens: int,
    label_chunk_size: int,
) -> list[dict[str, object]]:
    selected_pair_ids = {str(row["pair_id"]) for row in selected_pairs}
    contexts_by_pair: dict[str, dict[str, dict[str, object]]] = {}
    for row in context_rows:
        if str(row["pair_id"]) in selected_pair_ids:
            contexts_by_pair.setdefault(str(row["pair_id"]), {})[str(row["side"])] = row
    features_by_pair_side: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for row in feature_rows:
        if str(row["pair_id"]) in selected_pair_ids:
            features_by_pair_side.setdefault((str(row["pair_id"]), str(row["side"])), {})[int(row["feature_id"])] = row

    candidate_rows: list[dict[str, object]] = []
    candidate_pool_per_side = max(top_delta_features * 4, top_delta_features + 6)
    for pair in selected_pairs:
        pair_id = str(pair["pair_id"])
        left_context = contexts_by_pair[pair_id]["left"]
        right_context = contexts_by_pair[pair_id]["right"]
        left_features = features_by_pair_side[(pair_id, "left")]
        right_features = features_by_pair_side[(pair_id, "right")]
        deltas: list[dict[str, object]] = []
        for fid in sorted(set(left_features) | set(right_features)):
            left_value = float(left_features.get(fid, {}).get("contribution", 0.0))
            right_value = float(right_features.get(fid, {}).get("contribution", 0.0))
            delta = left_value - right_value
            context = left_context if delta >= 0 else right_context
            deltas.append(
                {
                    "pair_id": pair_id,
                    "target": pair["target"],
                    "target_label": pair["target_label"],
                    "left_context_label": pair["left_context_label"],
                    "right_context_label": pair["right_context_label"],
                    "left_context_id": left_context["context_id"],
                    "right_context_id": right_context["context_id"],
                    "left_prompt": pair["left_prompt"],
                    "right_prompt": pair["right_prompt"],
                    "feature_id": fid,
                    "left_contribution": left_value,
                    "right_contribution": right_value,
                    "contribution_delta_left_minus_right": delta,
                    "contribution_delta_abs": abs(delta),
                    "larger_context_label": context["context_label"],
                    "larger_context_side": context["side"],
                    "contrast_token_hint": pair["target_label"],
                }
            )
        left_deltas = [
            row
            for row in sorted(
                deltas,
                key=lambda item: -float(item["contribution_delta_left_minus_right"]),
            )
            if float(row["contribution_delta_left_minus_right"]) > 0
        ]
        right_deltas = [
            row
            for row in sorted(
                deltas,
                key=lambda item: float(item["contribution_delta_left_minus_right"]),
            )
            if float(row["contribution_delta_left_minus_right"]) < 0
        ]
        candidate_rows.extend(left_deltas[:candidate_pool_per_side])
        candidate_rows.extend(right_deltas[:candidate_pool_per_side])

    label_ids = sorted({int(row["feature_id"]) for row in candidate_rows})
    labels = display_label_features(
        W=W,
        row_mean=row_mean,
        feature_ids=label_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=label_top_tokens,
        chunk_size=label_chunk_size,
    )
    for row in candidate_rows:
        fid = int(row["feature_id"])
        row["feature_label"] = readable_feature_label(labels.get(fid, []), fid)
        row["feature_top_tokens"] = "; ".join(labels.get(fid, []))
        row["feature_token_summary"] = token_summary(row)
        row["filtered_as_uninteresting"] = is_uninteresting_label(row)

    display_rows: list[dict[str, object]] = []
    by_pair_side: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in candidate_rows:
        side = "left" if float(row["contribution_delta_left_minus_right"]) > 0 else "right"
        by_pair_side.setdefault((str(row["pair_id"]), side), []).append(row)
    for pair in selected_pairs:
        for side in ("left", "right"):
            rows = by_pair_side.get((str(pair["pair_id"]), side), [])
            rows.sort(key=lambda row: -float(row["contribution_delta_abs"]))
            filtered = [row for row in rows if not bool(row["filtered_as_uninteresting"])]
            chosen = (
                filtered[:top_delta_features]
                if len(filtered) >= max(2, top_delta_features // 2)
                else rows[:top_delta_features]
            )
            display_rows.extend(chosen[:top_delta_features])
    return display_rows


def write_report(
    path: Path,
    *,
    selected_pairs: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    context_rows: list[dict[str, object]],
    display_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
) -> None:
    contexts_by_pair: dict[str, dict[str, dict[str, object]]] = {}
    for row in context_rows:
        contexts_by_pair.setdefault(str(row["pair_id"]), {})[str(row["side"])] = row
    display_by_context: dict[str, list[dict[str, object]]] = {}
    for row in display_rows:
        display_by_context.setdefault(str(row["context_id"]), []).append(row)
    delta_by_pair: dict[str, list[dict[str, object]]] = {}
    for row in delta_rows:
        delta_by_pair.setdefault(str(row["pair_id"]), []).append(row)

    lines = [
        "# Qwen2B 32x Same-Token Polysemy Feature Mine",
        "",
        "This analysis keeps the target token fixed and decomposes the centered token score `h^T (W_U[token] - mean_vocab_row)`.",
        "Positive terms are sparse readout features that support that same token above the vocabulary mean in the given context.",
        "This is exploratory evidence about readout decomposition, not a causal intervention.",
        "",
        "## Selected Pairs",
        "",
        "| target | contexts | centered scores | sparse sums | max resid/direct | max rank | contrib cosine | top-feature Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_pairs:
        lines.append(
            f"| `{row['target_label']}` | {row['left_context_label']} / {row['right_context_label']} | "
            f"{float(row['left_exact_centered_logit']):+.2f} / {float(row['right_exact_centered_logit']):+.2f} | "
            f"{float(row['left_sparse_feature_sum']):+.2f} / {float(row['right_sparse_feature_sum']):+.2f} | "
            f"{float(row['max_residual_abs_over_direct']):.2f} | {int(row['max_target_rank'])} | "
            f"{float(row['contribution_cosine']):+.2f} | {float(row['top_positive_jaccard']):.2f} |"
        )
    for pair in selected_pairs:
        lines += [
            "",
            f"## `{pair['target_label']}`: {pair['left_context_label']} vs {pair['right_context_label']}",
            "",
            f"Left prompt: `{pair['left_prompt']}`",
            "",
            f"Right prompt: `{pair['right_prompt']}`",
            "",
        ]
        for side in ("left", "right"):
            context = contexts_by_pair[str(pair["pair_id"])][side]
            lines += [
                f"### {context['context_label']}",
                "",
                "| feature | token summary | contribution |",
                "|---|---|---:|",
            ]
            for feat in sorted(
                display_by_context.get(str(context["context_id"]), []),
                key=lambda item: -float(item["contribution"]),
            ):
                lines.append(
                    f"| {feat['feature_label']} | {feat.get('feature_token_summary') or token_summary(feat)} | "
                    f"{float(feat['contribution']):+.3f} |"
                )
            lines.append("")
        lines += [
            "### Differential context terms",
            "",
            f"Positive values are larger in `{pair['left_context_label']}`; negative values are larger in `{pair['right_context_label']}`.",
            "",
            "| feature | token summary | left contribution | right contribution | left - right | filtered? |",
            "|---|---|---:|---:|---:|---|",
        ]
        for feat in sorted(
            delta_by_pair.get(str(pair["pair_id"]), []),
            key=lambda item: -float(item["contribution_delta_abs"]),
        ):
            lines.append(
                f"| {feat['feature_label']} | {feat.get('feature_token_summary') or token_summary(feat)} | "
                f"{float(feat['left_contribution']):+.3f} | {float(feat['right_contribution']):+.3f} | "
                f"{float(feat['contribution_delta_left_minus_right']):+.3f} | {feat['filtered_as_uninteresting']} |"
            )
        lines.append("")
    lines += [
        "",
        "## Full Pair Ranking",
        "",
        "| pair | target | contexts | max rank | max resid/direct | contrib cosine | Jaccard | selection score | selected |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(pair_rows, key=lambda item: float(item["selection_score"]), reverse=True):
        lines.append(
            f"| {row['pair_id']} | `{row['target_label']}` | {row['left_context_label']} / {row['right_context_label']} | "
            f"{int(row['max_target_rank'])} | {float(row['max_residual_abs_over_direct']):.2f} | "
            f"{float(row['contribution_cosine']):+.2f} | {float(row['top_positive_jaccard']):.2f} | "
            f"{float(row['selection_score']):+.2f} | {row['selected_for_display']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-set",
        choices=["polysemy", "pos", "paper_relevant", "domain"],
        default="polysemy",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=6)
    parser.add_argument("--top-delta-features", type=int, default=4)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=4096)
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument(
        "--force-pair-ids",
        default="",
        help="Comma-separated pair ids to display instead of automatic ranking.",
    )
    parser.add_argument("--rank-top-n", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    model, tokenizer = load_qwen_model(args.model_id, dtype)
    model.to(device)

    candidate_sets = {
        "domain": domain_candidate_pairs,
        "paper_relevant": paper_relevant_candidate_pairs,
        "polysemy": candidate_pairs,
        "pos": pos_candidate_pairs,
    }
    context_rows, feature_rows, pair_rows, W, row_mean, encoder_w, encoder_b, meta = compute_rows(
        pairs=candidate_sets[args.candidate_set](),
        model=model,
        tokenizer=tokenizer,
        checkpoint=args.checkpoint,
        k=args.k,
        model_id=args.model_id,
        device=device,
        batch_size=args.batch_size,
        rank_top_n=args.rank_top_n,
    )
    force_pair_ids = [part.strip() for part in args.force_pair_ids.split(",") if part.strip()]
    selected_pairs = select_pairs(pair_rows, n_pairs=args.n_pairs, force_pair_ids=force_pair_ids or None)
    display_rows = label_display_rows(
        selected_pairs=selected_pairs,
        context_rows=context_rows,
        feature_rows=feature_rows,
        W=W,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_features=args.top_features,
        label_top_tokens=args.label_top_tokens,
        label_chunk_size=args.label_chunk_size,
    )
    delta_rows = label_delta_rows(
        selected_pairs=selected_pairs,
        context_rows=context_rows,
        feature_rows=feature_rows,
        W=W,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_delta_features=args.top_delta_features,
        label_top_tokens=args.label_top_tokens,
        label_chunk_size=args.label_chunk_size,
    )

    write_csv(args.out_dir / "qwen2b_32x_polysemy_contexts.csv", context_rows)
    write_csv(args.out_dir / "qwen2b_32x_polysemy_features.csv", feature_rows)
    write_csv(args.out_dir / "qwen2b_32x_polysemy_pairs.csv", pair_rows)
    write_csv(args.out_dir / "qwen2b_32x_polysemy_display_features.csv", display_rows)
    write_csv(args.out_dir / "qwen2b_32x_polysemy_delta_features.csv", delta_rows)
    write_csv(args.paper_dir / "qwen2b_32x_polysemy_contexts.csv", context_rows)
    write_csv(args.paper_dir / "qwen2b_32x_polysemy_pairs.csv", pair_rows)
    write_csv(args.paper_dir / "qwen2b_32x_polysemy_display_features.csv", display_rows)
    write_csv(args.paper_dir / "qwen2b_32x_polysemy_delta_features.csv", delta_rows)
    cache = {
        "meta": meta,
        "context_rows": context_rows,
        "feature_rows": feature_rows,
        "pair_rows": pair_rows,
        "selected_pairs": selected_pairs,
        "display_rows": display_rows,
        "delta_rows": delta_rows,
    }
    torch.save(cache, args.out_dir / "qwen2b_32x_polysemy_cache.pt")
    torch.save(cache, args.paper_dir / "qwen2b_32x_polysemy_cache.pt")
    write_report(
        args.out_dir / "qwen2b_32x_polysemy_features.md",
        selected_pairs=selected_pairs,
        pair_rows=pair_rows,
        context_rows=context_rows,
        display_rows=display_rows,
        delta_rows=delta_rows,
    )
    write_report(
        args.paper_dir / "qwen2b_32x_polysemy_features.md",
        selected_pairs=selected_pairs,
        pair_rows=pair_rows,
        context_rows=context_rows,
        display_rows=display_rows,
        delta_rows=delta_rows,
    )
    manifest = {
        "description": "Qwen2B 32x same-token context/polysemy Sparse Readout Prism feature mine.",
        **meta,
        "out_dir": str(args.out_dir),
        "paper_dir": str(args.paper_dir),
        "n_contexts": len(context_rows),
        "n_pairs": len(pair_rows),
        "n_selected_pairs": len(selected_pairs),
        "candidate_set": args.candidate_set,
        "force_pair_ids": force_pair_ids,
        "command": " ".join(sys.argv),
        "outputs": [
            "qwen2b_32x_polysemy_contexts.csv",
            "qwen2b_32x_polysemy_features.csv",
            "qwen2b_32x_polysemy_pairs.csv",
            "qwen2b_32x_polysemy_display_features.csv",
            "qwen2b_32x_polysemy_delta_features.csv",
            "qwen2b_32x_polysemy_cache.pt",
            "qwen2b_32x_polysemy_features.md",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "paper_dir": str(args.paper_dir),
                "pairs": len(pair_rows),
                "selected": [row["pair_id"] for row in selected_pairs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
