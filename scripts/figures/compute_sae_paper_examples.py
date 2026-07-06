"""Compute paper-example metrics with the selected Qwen3.5-2B readout SAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparse_readout_prism.utils import resolve_device, write_csv
from sparse_readout_prism.token_display import clean_token
from sparse_readout_prism.research.qwen_readout import (
    collect_readout_state,
    decompose_row_contrast,
    display_label_features,
    fallback_feature_label,
    find_lm_head_with_path,
    load_qwen_model,
    load_sae,
    parse_single_token,
    readable_feature_label,
    token_summary,
)
from sparse_readout_prism.research.qwen_example_prompts import EXAMPLE_SETS

DEFAULT_OUT_DIR = Path("results/qwen2b_sae_paper_examples")
DEFAULT_PAPER_DIR = Path("paper/figures")


@torch.no_grad()
def build_rows(
    *,
    examples: list[dict[str, str]],
    model,
    tokenizer,
    checkpoint: Path,
    k: int,
    model_id: str,
    device: torch.device,
    top_features: int,
    label_top_tokens: int,
    label_chunk_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict]:
    lm_head, lm_head_path = find_lm_head_with_path(model)
    W = lm_head.weight.detach().cpu()
    vocab, d_model = W.shape
    decoder, encoder_w, encoder_b, sae_config = load_sae(checkpoint)
    if decoder.shape[1] != d_model:
        raise ValueError(f"SAE d_model {decoder.shape[1]} does not match W_U d_model {d_model}")
    row_mean = W.float().mean(dim=0)

    raw_case_rows: list[dict[str, object]] = []
    raw_feature_rows: list[dict[str, object]] = []
    for panel_idx, ex in enumerate(examples):
        target_a = parse_single_token(tokenizer, ex["target_a"])
        target_b = parse_single_token(tokenizer, ex["target_b"])
        h, logits = collect_readout_state(model, tokenizer, ex["prompt"], device)
        exact = float(logits[target_a] - logits[target_b])
        dec = decompose_row_contrast(
            W=W,
            target_a=target_a,
            target_b=target_b,
            row_mean=row_mean,
            encoder_w=encoder_w,
            encoder_b=encoder_b,
            decoder=decoder,
            h=h,
            k=k,
            exact=exact,
        )
        z = dec.z
        coeff = dec.coeff
        active = dec.active
        feature_scores = dec.feature_scores
        contrib = dec.contrib
        feature_sum = dec.feature_sum
        residual = dec.residual
        case_row = {
            "panel": chr(ord("A") + panel_idx),
            "case_id": ex["case_id"],
            "model_id": model_id,
            "checkpoint": str(checkpoint),
            "k": k,
            "title": ex["title"],
            "claim": ex["claim"],
            "prompt": ex["prompt"],
            "target_a": ex["target_a"],
            "target_b": ex["target_b"],
            "target_a_id": target_a,
            "target_b_id": target_b,
            "target_a_label": clean_token(tokenizer, target_a),
            "target_b_label": clean_token(tokenizer, target_b),
            "exact_margin": exact,
            "sparse_feature_sum": feature_sum,
            "residual": residual,
            "residual_abs_over_direct": abs(residual) / max(abs(exact), 1e-8),
            "lm_head_path": lm_head_path,
            "vocab": vocab,
            "d_model": d_model,
        }
        raw_case_rows.append(case_row)
        rows_for_case = []
        for fid, value, score, coef in zip(
            active.tolist(),
            contrib.tolist(),
            feature_scores.tolist(),
            coeff[active].tolist(),
        ):
            hint = (
                clean_token(tokenizer, target_a)
                if float(z[0, fid]) >= float(z[1, fid])
                else clean_token(tokenizer, target_b)
            )
            rows_for_case.append(
                {
                    **case_row,
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
        rows_for_case.sort(key=lambda row: -float(row["contribution_abs"]))
        raw_feature_rows.extend(rows_for_case)

    display_rows: list[dict[str, object]] = []
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in raw_feature_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    for rows in by_case.values():
        selected = sorted(rows, key=lambda row: -float(row["contribution_abs"]))[:top_features]
        display_rows.extend(sorted(selected, key=lambda row: float(row["contribution"])))

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
    for row in raw_feature_rows:
        fid = int(row["feature_id"])
        row["feature_label"] = f"f{fid}"
        row["feature_top_tokens"] = ""
    for row in display_rows:
        fid = int(row["feature_id"])
        row["feature_label"] = (
            readable_feature_label(labels.get(fid, []), fid) if labels.get(fid) else fallback_feature_label(row, fid)
        )
        row["feature_top_tokens"] = "; ".join(labels.get(fid, []))
        row["feature_token_summary"] = token_summary(row)
    meta = {
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "k": k,
        "sae_config": sae_config,
        "lm_head_path": lm_head_path,
        "vocab": int(vocab),
        "d_model": int(d_model),
    }
    return raw_case_rows, raw_feature_rows, display_rows, meta


def write_markdown(
    path: Path,
    case_rows: list[dict[str, object]],
    display_rows: list[dict[str, object]],
) -> None:
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in display_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    lines = [
        "# Qwen3.5-2B Sparse Readout Prism Paper Examples",
        "",
        "Positive terms support the first token in the contrast; negative terms support the competitor. These are readout decompositions, not causal interventions.",
        "",
    ]
    for case in case_rows:
        lines += [
            f"## {case['panel']}. {case['title']}: `{case['target_a_label']} - {case['target_b_label']}`",
            "",
            f"Prompt: `{case['prompt']}`",
            "",
            f"Exact margin: `{float(case['exact_margin']):+.3f}`",
            f"Sparse feature sum: `{float(case['sparse_feature_sum']):+.3f}`",
            f"Residual: `{float(case['residual']):+.3f}`",
            f"Residual/direct: `{float(case['residual_abs_over_direct']):.3f}`",
            "",
            "| feature | token summary | contribution | supports |",
            "|---|---|---:|---|",
        ]
        for row in sorted(
            by_case.get(str(case["case_id"]), []),
            key=lambda item: -float(item["contribution_abs"]),
        ):
            lines.append(
                f"| {row['feature_label']} | {row.get('feature_token_summary') or token_summary(row)} | "
                f"{float(row['contribution']):+.3f} | {row['supports']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-set", choices=sorted(EXAMPLE_SETS), default="section43_polysemy_margin")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--paper-dir",
        type=lambda s: Path(s) if s else None,
        default=DEFAULT_PAPER_DIR,
        help="mirror paper-facing outputs here; pass '' to disable",
    )
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=6)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = args.paper_dir if args.paper_dir is not None else None
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = resolve_device(args.device)
    examples = EXAMPLE_SETS[args.example_set]
    model, tokenizer = load_qwen_model(args.model_id, dtype)
    model.to(device)

    case_rows, feature_rows, display_rows, meta = build_rows(
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        checkpoint=args.checkpoint,
        k=args.k,
        model_id=args.model_id,
        device=device,
        top_features=args.top_features,
        label_top_tokens=args.label_top_tokens,
        label_chunk_size=args.label_chunk_size,
    )
    write_csv(args.out_dir / "qwen2b_selected_paper_examples_cases.csv", case_rows)
    write_csv(
        args.out_dir / "qwen2b_selected_paper_examples_features.csv",
        feature_rows,
    )
    write_csv(
        args.out_dir / "qwen2b_selected_paper_examples_display_features.csv",
        display_rows,
    )
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        write_csv(paper_dir / "qwen2b_selected_paper_examples_cases.csv", case_rows)
        write_csv(
            paper_dir / "qwen2b_selected_paper_examples_features.csv",
            feature_rows,
        )
        write_csv(
            paper_dir / "qwen2b_selected_paper_examples_display_features.csv",
            display_rows,
        )
    cache = {
        "meta": meta,
        "case_rows": case_rows,
        "feature_rows": feature_rows,
        "display_rows": display_rows,
    }
    torch.save(cache, args.out_dir / "qwen2b_selected_paper_examples_cache.pt")
    if paper_dir is not None:
        torch.save(cache, paper_dir / "qwen2b_selected_paper_examples_cache.pt")
    write_markdown(
        args.out_dir / "qwen2b_selected_paper_examples.md",
        case_rows,
        display_rows,
    )
    manifest = {
        "description": "Qwen3.5-2B Sparse Readout Prism selected paper examples.",
        **meta,
        "out_dir": str(args.out_dir),
        "paper_dir": str(paper_dir) if paper_dir is not None else None,
        "example_set": args.example_set,
        "outputs": [
            "qwen2b_selected_paper_examples_cases.csv",
            "qwen2b_selected_paper_examples_features.csv",
            "qwen2b_selected_paper_examples_display_features.csv",
            "qwen2b_selected_paper_examples_cache.pt",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "paper_dir": str(paper_dir),
                "cases": len(case_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
