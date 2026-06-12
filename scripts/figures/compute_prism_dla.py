"""Compute and persist Qwen3.5-2B 32x Prism-DLA component metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sparse_readout_prism.research._common.qwen_readout import (
    clean_token,
    encode_topk,
    find_lm_head,
    display_label_features,
    load_qwen_model,
    load_sae,
    parse_single_token,
    readable_feature_label,
)
from sparse_readout_prism.utils import write_csv

DEFAULT_OUT_DIR = Path("results/qwen2b_32x_prism_dla_verify_assume")
DEFAULT_PAPER_DIR = Path("paper/figures/qwen2b_32x_prism_dla_verify_assume")
DEFAULT_PROMPT = "The source has not been checked yet. The appropriate next action is to"


@dataclass(frozen=True)
class Component:
    name: str
    label: str
    layer: int
    kind: str
    vector: torch.Tensor
    norm_linearized: torch.Tensor


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype {name!r}")


def text_model_from_qwen(model):
    return model.model.language_model


def as_hidden(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def collect_components(
    *,
    model,
    tokenizer,
    prompt: str,
    position: int,
    device: torch.device,
) -> tuple[
    list[Component],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    text_model = text_model_from_qwen(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("prompt tokenized to no tokens")
    pos = input_ids.shape[1] - 1 if position < 0 else int(position)
    if pos < 0 or pos >= input_ids.shape[1]:
        raise ValueError(f"position {position} outside token range 0..{input_ids.shape[1] - 1}")

    layer_inputs: dict[int, torch.Tensor] = {}
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx, layer in enumerate(text_model.layers):

        def layer_hook(_module, inputs, output, *, idx=layer_idx):
            layer_inputs[idx] = inputs[0][0, pos].detach().float().cpu().contiguous()
            layer_outputs[idx] = as_hidden(output)[0, pos].detach().float().cpu().contiguous()

        hooks.append(layer.register_forward_hook(layer_hook))

    try:
        out = model(input_ids=input_ids)
    finally:
        for hook in hooks:
            hook.remove()

    n_layers = len(text_model.layers)
    missing = set(range(n_layers)).difference(layer_inputs)
    missing |= set(range(n_layers)).difference(layer_outputs)
    if missing:
        raise RuntimeError(f"failed to collect Qwen residual components for layers {sorted(missing)}")

    initial_resid = layer_inputs[0]
    final_resid = layer_outputs[n_layers - 1]
    norm = text_model.norm
    gamma = (1.0 + norm.weight.detach().float().cpu()).contiguous()
    eps = float(norm.variance_epsilon if hasattr(norm, "variance_epsilon") else norm.eps)
    frozen_rms = torch.sqrt(final_resid.pow(2).mean() + eps)

    raw_components: list[tuple[str, str, int, str, torch.Tensor]] = [("embed", "embed", -1, "embed", initial_resid)]
    previous = initial_resid
    for layer in range(n_layers):
        current = layer_outputs[layer]
        raw_components.append((f"block_{layer:02d}", f"L{layer}", layer, "block_delta", current - previous))
        previous = current

    components = [
        Component(
            name=name,
            label=label,
            layer=layer,
            kind=kind,
            vector=vector,
            norm_linearized=vector / frozen_rms * gamma,
        )
        for name, label, layer, kind, vector in raw_components
    ]
    reconstructed_resid = sum((component.vector for component in components), start=torch.zeros_like(final_resid))
    reconstructed_norm = sum(
        (component.norm_linearized for component in components),
        start=torch.zeros_like(final_resid),
    )
    final_norm_exact = norm(final_resid.to(device)).detach().float().cpu().contiguous()
    logits = out.logits[0, pos].detach().float().cpu().contiguous()
    return (
        components,
        input_ids[0].detach().cpu(),
        final_resid,
        final_norm_exact,
        reconstructed_norm,
        reconstructed_resid,
        logits,
    )


def build_rows(
    *,
    components: list[Component],
    W: torch.Tensor,
    decoder: torch.Tensor,
    row_mean: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer,
    target_id: int,
    contrast_id: int,
    k: int,
    logits: torch.Tensor,
    label_top_tokens: int,
    label_chunk_size: int,
    top_features: int,
    contrast_mode: str = "token_token",
    final_norm_exact: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, str], dict[str, float]]:
    if contrast_mode == "vocab_mean":
        w_diff = W[target_id].float() - row_mean
        centered = (W[target_id].float() - row_mean)[None, :]
        norms = centered.norm(dim=1).clamp_min(1e-8)
        x = centered / norms[:, None]
        z = encode_topk(x, encoder_w, encoder_b, k=k)
        coeff = norms[0] * z[0]
        active = torch.nonzero(coeff != 0, as_tuple=False).flatten()
        decoder_t = decoder[active].T.contiguous()
        w_hat_diff = coeff[active] @ decoder[active]
        w_residual_diff = w_diff - w_hat_diff
    else:
        rows = W[[target_id, contrast_id]].float()
        centered = rows - row_mean
        norms = centered.norm(dim=1).clamp_min(1e-8)
        x = centered / norms[:, None]
        z = encode_topk(x, encoder_w, encoder_b, k=k)
        coeff = norms[0] * z[0] - norms[1] * z[1]
        active = torch.nonzero(coeff != 0, as_tuple=False).flatten()
        decoder_t = decoder[active].T.contiguous()
        w_diff = rows[0] - rows[1]
        w_hat_diff = coeff[active] @ decoder[active]
        w_residual_diff = w_diff - w_hat_diff

    component_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    by_feature_abs: dict[int, float] = {}
    by_component_feature: dict[tuple[str, int], float] = {}
    for component in components:
        h = component.norm_linearized
        direct = float(h @ w_diff)
        feature_scores = h @ decoder_t
        contrib = coeff[active] * feature_scores
        feature_sum = float(contrib.sum())
        residual = float(h @ w_residual_diff)
        component_rows.append(
            {
                "component": component.name,
                "component_label": component.label,
                "layer": component.layer,
                "component_kind": component.kind,
                "direct_contribution": direct,
                "feature_sum": feature_sum,
                "wu_residual_term": residual,
                "reconstructed_contribution": feature_sum + residual,
                "component_reconstruction_abs_error": abs(direct - feature_sum - residual),
            }
        )
        for fid, value, score, coef in zip(
            active.tolist(), contrib.tolist(), feature_scores.tolist(), coeff[active].tolist()
        ):
            fid = int(fid)
            value = float(value)
            by_feature_abs[fid] = by_feature_abs.get(fid, 0.0) + abs(value)
            by_component_feature[(component.name, fid)] = by_component_feature.get((component.name, fid), 0.0) + value
            feature_rows.append(
                {
                    "component": component.name,
                    "component_label": component.label,
                    "layer": component.layer,
                    "component_kind": component.kind,
                    "feature_id": fid,
                    "signed_contribution": value,
                    "contribution_abs": abs(value),
                    "component_dot_direction": float(score),
                    "margin_feature_coeff": float(coef),
                }
            )

    selected_feature_ids = [
        fid for fid, _score in sorted(by_feature_abs.items(), key=lambda item: item[1], reverse=True)[:top_features]
    ]
    aggregated_rows: list[dict[str, Any]] = []
    for component in components:
        for fid in selected_feature_ids:
            aggregated_rows.append(
                {
                    "component": component.name,
                    "component_label": component.label,
                    "layer": component.layer,
                    "component_kind": component.kind,
                    "feature_id": fid,
                    "signed_contribution": by_component_feature.get((component.name, fid), 0.0),
                }
            )

    labels = display_label_features(
        W=W,
        row_mean=row_mean,
        feature_ids=selected_feature_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=label_top_tokens,
        chunk_size=label_chunk_size,
    )
    label_by_id = {fid: readable_feature_label(labels.get(fid, []), fid, max_len=30) for fid in selected_feature_ids}
    for row in feature_rows + aggregated_rows:
        row["feature_label"] = label_by_id.get(int(row["feature_id"]), f"f{int(row['feature_id'])}")

    if contrast_mode == "vocab_mean":
        if final_norm_exact is None:
            raise ValueError("final_norm_exact is required for contrast_mode='vocab_mean'")
        exact_margin = float(final_norm_exact @ w_diff)
    else:
        exact_margin = float(logits[target_id] - logits[contrast_id])
    summary = {
        "exact_margin": exact_margin,
        "component_direct_sum": sum(float(row["direct_contribution"]) for row in component_rows),
        "component_feature_sum": sum(float(row["feature_sum"]) for row in component_rows),
        "component_wu_residual_sum": sum(float(row["wu_residual_term"]) for row in component_rows),
        "target_logit": float(logits[target_id]),
        "contrast_logit": float(logits[contrast_id]),
    }
    summary["component_identity_abs_error"] = abs(summary["exact_margin"] - summary["component_direct_sum"])
    summary["sparse_margin_residual"] = summary["exact_margin"] - summary["component_feature_sum"]
    summary["sparse_residual_abs_over_direct"] = abs(summary["sparse_margin_residual"]) / max(
        abs(summary["exact_margin"]), 1e-8
    )
    return component_rows, feature_rows, aggregated_rows, label_by_id, summary


def write_summary(
    path: Path,
    *,
    prompt: str,
    target_label: str,
    contrast_label: str,
    summary: dict[str, float],
    component_rows: list[dict[str, Any]],
    label_by_id: dict[int, str],
    aggregated_rows: list[dict[str, Any]],
) -> None:
    top_components = sorted(component_rows, key=lambda row: abs(float(row["direct_contribution"])), reverse=True)[:10]
    totals = {fid: 0.0 for fid in label_by_id}
    for row in aggregated_rows:
        fid = int(row["feature_id"])
        if fid in totals:
            totals[fid] += float(row["signed_contribution"])
    lines = [
        "# Qwen3.5-2B 32x Prism-DLA",
        "",
        f"Prompt: `{prompt}`",
        "",
        f"Contrast: `{target_label} - {contrast_label}`",
        "",
        f"Exact margin: `{summary['exact_margin']:+.4f}`",
        f"Component DLA sum: `{summary['component_direct_sum']:+.4f}`",
        f"Component identity error: `{summary['component_identity_abs_error']:.2e}`",
        f"Feature sum: `{summary['component_feature_sum']:+.4f}`",
        f"W_U residual sum: `{summary['component_wu_residual_sum']:+.4f}`",
        f"Sparse residual/direct: `{summary['sparse_residual_abs_over_direct']:.3f}`",
        "",
        "## Largest Components",
        "",
        "| component | direct | features | W_U residual |",
        "|---|---:|---:|---:|",
    ]
    for row in top_components:
        lines.append(
            f"| {row['component_label']} | {float(row['direct_contribution']):+.3f} | "
            f"{float(row['feature_sum']):+.3f} | {float(row['wu_residual_term']):+.3f} |"
        )
    lines += ["", "## Top Features", "", "| feature | signed total |", "|---|---:|"]
    for fid, value in sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:12]:
        lines.append(f"| {label_by_id.get(fid, f'f{fid}')} | {value:+.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    folder: Path,
    *,
    prompt: str,
    model_id: str,
    checkpoint: Path,
    k: int,
    target_label: str,
    contrast_label: str,
    target_id: int,
    contrast_id: int,
    component_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    aggregated_rows: list[dict[str, Any]],
    label_by_id: dict[int, str],
    summary: dict[str, float],
    token_ids: torch.Tensor,
    final_resid: torch.Tensor,
    final_norm_exact: torch.Tensor,
    reconstructed_norm: torch.Tensor,
    reconstructed_resid: torch.Tensor,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "qwen2b_32x_prism_dla_component_summary.csv", component_rows)
    write_csv(folder / "qwen2b_32x_prism_dla_feature_contributions.csv", feature_rows)
    write_csv(folder / "qwen2b_32x_prism_dla_feature_by_component.csv", aggregated_rows)
    write_csv(
        folder / "qwen2b_32x_prism_dla_identity_checks.csv",
        [
            {
                **summary,
                "target_id": target_id,
                "contrast_id": contrast_id,
                "target_label": target_label,
                "contrast_label": contrast_label,
                "residual_stream_reconstruction_abs_error": float((final_resid - reconstructed_resid).abs().max()),
                "frozen_norm_reconstruction_abs_error": float((final_norm_exact - reconstructed_norm).abs().max()),
            }
        ],
    )
    write_summary(
        folder / "qwen2b_32x_prism_dla_summary.md",
        prompt=prompt,
        target_label=target_label,
        contrast_label=contrast_label,
        summary=summary,
        component_rows=component_rows,
        label_by_id=label_by_id,
        aggregated_rows=aggregated_rows,
    )
    torch.save(
        {
            "prompt": prompt,
            "model_id": model_id,
            "checkpoint": str(checkpoint),
            "k": k,
            "target_id": target_id,
            "contrast_id": contrast_id,
            "summary": summary,
            "token_ids": token_ids,
            "final_resid": final_resid,
            "final_norm_exact": final_norm_exact,
            "reconstructed_norm": reconstructed_norm,
            "reconstructed_resid": reconstructed_resid,
        },
        folder / "qwen2b_32x_prism_dla_cache.pt",
    )
    manifest = {
        "description": "Qwen3.5-2B 32x feature-resolved Prism-DLA example.",
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "k": k,
        "prompt": prompt,
        "target": target_label,
        "contrast": contrast_label,
        "summary": summary,
        "outputs": [
            "qwen2b_32x_prism_dla_component_summary.csv",
            "qwen2b_32x_prism_dla_feature_contributions.csv",
            "qwen2b_32x_prism_dla_feature_by_component.csv",
            "qwen2b_32x_prism_dla_identity_checks.csv",
            "qwen2b_32x_prism_dla_cache.pt",
        ],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target", default=" verify")
    parser.add_argument("--contrast-target", default=" assume")
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=8192)
    parser.add_argument("--contrast-mode", choices=["token_token", "vocab_mean"], default="token_token")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    model, tokenizer = load_qwen_model(args.model_id, dtype)
    model.to(device)
    lm_head, lm_head_path = find_lm_head(model)
    W = lm_head.weight.detach().float().cpu().contiguous()
    decoder, encoder_w, encoder_b, sae_config = load_sae(args.checkpoint)
    if decoder.shape[1] != W.shape[1]:
        raise ValueError(f"SAE d_model {decoder.shape[1]} does not match lm_head d_model {W.shape[1]}")
    target_id = parse_single_token(tokenizer, args.target)
    contrast_id = parse_single_token(tokenizer, args.contrast_target)
    target_label = clean_token(tokenizer, target_id)
    contrast_label = clean_token(tokenizer, contrast_id)
    row_mean = W.mean(dim=0)

    components, token_ids, final_resid, final_norm_exact, reconstructed_norm, reconstructed_resid, logits = (
        collect_components(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            position=args.position,
            device=device,
        )
    )
    component_rows, feature_rows, aggregated_rows, label_by_id, summary = build_rows(
        components=components,
        W=W,
        decoder=decoder,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        target_id=target_id,
        contrast_id=contrast_id,
        k=args.k,
        logits=logits,
        label_top_tokens=args.label_top_tokens,
        label_chunk_size=args.label_chunk_size,
        top_features=args.top_features,
        contrast_mode=args.contrast_mode,
        final_norm_exact=final_norm_exact,
    )
    summary.update(
        {
            "lm_head_path": lm_head_path,
            "vocab": float(W.shape[0]),
            "d_model": float(W.shape[1]),
            "residual_stream_reconstruction_abs_error": float((final_resid - reconstructed_resid).abs().max()),
            "frozen_norm_reconstruction_abs_error": float((final_norm_exact - reconstructed_norm).abs().max()),
        }
    )
    for folder in (args.out_dir, args.paper_dir):
        if folder is None:
            continue
        save_outputs(
            folder,
            prompt=args.prompt,
            model_id=args.model_id,
            checkpoint=args.checkpoint,
            k=args.k,
            target_label=target_label,
            contrast_label=contrast_label,
            target_id=target_id,
            contrast_id=contrast_id,
            component_rows=component_rows,
            feature_rows=feature_rows,
            aggregated_rows=aggregated_rows,
            label_by_id=label_by_id,
            summary=summary,
            token_ids=token_ids,
            final_resid=final_resid,
            final_norm_exact=final_norm_exact,
            reconstructed_norm=reconstructed_norm,
            reconstructed_resid=reconstructed_resid,
        )
    print(json.dumps({"summary": summary, "out_dir": str(args.out_dir), "paper_dir": str(args.paper_dir)}, indent=2))
    _ = sae_config
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
