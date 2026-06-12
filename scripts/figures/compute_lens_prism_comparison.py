"""Compute and persist focused logit-lens vs Sparse Readout Prism comparison metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sparse_readout_prism.research.qwen_readout import (
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


DEFAULT_OUT_DIR = Path("results/qwen2b_lens_prism_comparison_verify_assume")
DEFAULT_PAPER_DIR = Path("paper/figures/qwen2b_lens_prism_comparison_verify_assume")
DEFAULT_PROMPT = "The source has not been checked yet. The appropriate next action is to"

FEATURE_LABEL_OVERRIDES: dict[int, str] = {
    25948: "verify family",
    47092: "check family",
    16851: "verify/check family",
    40205: "Ver surface",
    28177: "consider family",
    62519: "assume family",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    # delegate to shared helper (empty rows -> no file, matching original)
    from sparse_readout_prism.utils import write_csv as _write_csv

    return _write_csv(path, rows)


def decode_top_tokens(tokenizer, logits: torch.Tensor, *, top_n: int) -> list[dict[str, object]]:
    values, ids = torch.topk(logits.float(), k=top_n)
    rows: list[dict[str, object]] = []
    for rank, (token_id, logit) in enumerate(zip(ids.tolist(), values.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": clean_token(tokenizer, int(token_id)),
                "logit": float(logit),
            }
        )
    return rows


@torch.no_grad()
def compute_case(
    *,
    model,
    tokenizer,
    checkpoint: Path,
    prompt: str,
    target_a: str,
    target_b: str,
    k: int,
    top_features: int,
    top_tokens: int,
    label_top_tokens: int,
    label_chunk_size: int,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    lm_head, lm_head_path = find_lm_head(model)
    W = lm_head.weight.detach().cpu()
    vocab, d_model = W.shape
    decoder, encoder_w, encoder_b, sae_config = load_sae(checkpoint)
    decoder = decoder.float()
    row_mean = W.float().mean(dim=0)

    target_a_id = parse_single_token(tokenizer, target_a)
    target_b_id = parse_single_token(tokenizer, target_b)

    h_list, logits_list = collect_readout_states_batched(model, tokenizer, [prompt], device, batch_size=1)
    h = h_list[0].float()
    logits = logits_list[0].float()

    top_rows = decode_top_tokens(tokenizer, logits, top_n=top_tokens)
    target_a_logit = float(logits[target_a_id])
    target_b_logit = float(logits[target_b_id])
    exact_margin = target_a_logit - target_b_logit

    rows = W[[target_a_id, target_b_id]].float()
    centered = rows - row_mean
    norms = centered.norm(dim=1).clamp_min(1e-8)
    x = centered / norms[:, None]
    z = encode_topk(x, encoder_w, encoder_b, k=k)
    coeff = norms[0] * z[0] - norms[1] * z[1]
    active = torch.nonzero(coeff != 0, as_tuple=False).flatten()
    feature_scores = h @ decoder[active].T
    contrib = coeff[active] * feature_scores
    sparse_feature_sum = float(contrib.sum())
    residual = exact_margin - sparse_feature_sum

    feature_rows: list[dict[str, object]] = []
    for fid, value, score, coef in zip(
        active.tolist(), contrib.tolist(), feature_scores.tolist(), coeff[active].tolist()
    ):
        hint = (
            clean_token(tokenizer, target_a_id)
            if float(z[0, fid]) >= float(z[1, fid])
            else clean_token(tokenizer, target_b_id)
        )
        feature_rows.append(
            {
                "feature_id": int(fid),
                "contribution": float(value),
                "contribution_abs": abs(float(value)),
                "feature_hidden_score": float(score),
                "row_contrast_coeff": float(coef),
                "target_a_feature_activation": float(z[0, fid]),
                "target_b_feature_activation": float(z[1, fid]),
                "contrast_token_hint": hint,
                "supports": "A" if float(value) >= 0 else "B",
            }
        )
    feature_rows.sort(key=lambda row: -float(row["contribution_abs"]))
    display_rows = feature_rows[:top_features]
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

    summary = {
        "model_id": getattr(model.config, "_name_or_path", ""),
        "checkpoint": str(checkpoint),
        "k": k,
        "prompt": prompt,
        "target_a": target_a,
        "target_b": target_b,
        "target_a_id": int(target_a_id),
        "target_b_id": int(target_b_id),
        "target_a_label": clean_token(tokenizer, target_a_id),
        "target_b_label": clean_token(tokenizer, target_b_id),
        "target_a_logit": target_a_logit,
        "target_b_logit": target_b_logit,
        "exact_margin": exact_margin,
        "sparse_feature_sum": sparse_feature_sum,
        "residual": residual,
        "residual_abs_over_direct": abs(residual) / max(abs(exact_margin), 1e-8),
        "top_logit_lens_tokens": " | ".join(f"{row['token']}:{float(row['logit']):.3f}" for row in top_rows),
        "lm_head_path": lm_head_path,
        "vocab": int(vocab),
        "d_model": int(d_model),
    }
    meta = {
        "sae_config": sae_config,
        "lm_head_path": lm_head_path,
        "vocab": int(vocab),
        "d_model": int(d_model),
    }
    return summary, display_rows, top_rows, meta


def write_report(
    path: Path,
    *,
    summary: dict[str, object],
    feature_rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Qwen2B Logit-Lens vs Sparse Readout Prism Comparison",
        "",
        f"Prompt: `{summary['prompt']}`",
        "",
        f"Contrast: `{summary['target_a_label']} - {summary['target_b_label']}`",
        "",
        "## Logit Lens",
        "",
        "| rank | token | logit |",
        "|---:|---|---:|",
    ]
    for row in top_rows:
        lines.append(f"| {row['rank']} | `{row['token']}` | {float(row['logit']):+.4f} |")
    lines += [
        "",
        "## Sparse Readout Prism",
        "",
        f"Exact margin: `{float(summary['exact_margin']):+.4f}`",
        f"Sparse feature sum: `{float(summary['sparse_feature_sum']):+.4f}`",
        f"Residual: `{float(summary['residual']):+.4f}`",
        f"Residual/direct: `{float(summary['residual_abs_over_direct']):.4f}`",
        "",
        "| feature | token summary | contribution | supports |",
        "|---|---|---:|---|",
    ]
    for row in sorted(feature_rows, key=lambda item: -float(item["contribution_abs"])):
        lines.append(
            f"| {row['feature_label']} | {row['feature_token_summary']} | "
            f"{float(row['contribution']):+.4f} | {row['supports']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target-a", default=" verify")
    parser.add_argument("--target-b", default=" assume")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=6)
    parser.add_argument("--top-tokens", type=int, default=10)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    model, tokenizer = load_qwen_model(args.model_id, dtype)
    model.to(device)

    summary, feature_rows, top_rows, meta = compute_case(
        model=model,
        tokenizer=tokenizer,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        target_a=args.target_a,
        target_b=args.target_b,
        k=args.k,
        top_features=args.top_features,
        top_tokens=args.top_tokens,
        label_top_tokens=args.label_top_tokens,
        label_chunk_size=args.label_chunk_size,
        device=device,
    )
    summary["model_id"] = args.model_id
    write_csv(args.out_dir / "lens_prism_comparison_summary.csv", [summary])
    write_csv(args.out_dir / "lens_prism_comparison_top_tokens.csv", top_rows)
    write_csv(args.out_dir / "lens_prism_comparison_features.csv", feature_rows)
    write_csv(args.paper_dir / "lens_prism_comparison_summary.csv", [summary])
    write_csv(args.paper_dir / "lens_prism_comparison_top_tokens.csv", top_rows)
    write_csv(args.paper_dir / "lens_prism_comparison_features.csv", feature_rows)
    torch.save(
        {"summary": summary, "top_rows": top_rows, "feature_rows": feature_rows, "meta": meta},
        args.out_dir / "lens_prism_comparison_cache.pt",
    )
    write_report(
        args.out_dir / "lens_prism_comparison.md",
        summary=summary,
        feature_rows=feature_rows,
        top_rows=top_rows,
    )
    write_report(
        args.paper_dir / "lens_prism_comparison.md",
        summary=summary,
        feature_rows=feature_rows,
        top_rows=top_rows,
    )
    manifest = {
        "description": "Focused logit-lens top-token and Sparse Readout Prism comparison for one Qwen2B prompt.",
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "paper_dir": str(args.paper_dir),
        "prompt": args.prompt,
        "target_a": args.target_a,
        "target_b": args.target_b,
        "k": args.k,
        **meta,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "paper_dir": str(args.paper_dir),
                "exact_margin": summary["exact_margin"],
                "sparse_feature_sum": summary["sparse_feature_sum"],
                "residual": summary["residual"],
                "residual_abs_over_direct": summary["residual_abs_over_direct"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
