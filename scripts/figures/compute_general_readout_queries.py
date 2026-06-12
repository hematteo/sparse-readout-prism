"""Compute Qwen3.5-2B metrics for the paper's readout-query taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparse_readout_prism.research._common.qwen_readout import (
    clean_token,
    collect_readout_state,
    find_lm_head,
    load_sae,
)
from sparse_readout_prism.research._common.query_decompose import (
    QuerySpec,
    attach_feature_labels,
    build_query_weights,
    decompose_query,
    load_qwen_model,
    token_rank_and_prob,
    write_csv,
)

DEFAULT_OUT_DIR = Path("results/qwen2b_general_readout_queries")
DEFAULT_PAPER_DIR = Path("paper/figures/qwen2b_general_readout_queries")


QUERY_SPECS: tuple[QuerySpec, ...] = (
    QuerySpec(
        case_id="selected_bug_raw",
        role="Selected token",
        query_kind="selected_token",
        title="Selected token score",
        prompt="The programmer reproduced the crash and filed a",
        target="bug",
        note="Raw selected-token score h^T W_U[bug].",
    ),
    QuerySpec(
        case_id="availability_bug_mean",
        role="Availability",
        query_kind="target_vs_vocab_mean",
        title="Token vs vocabulary mean",
        prompt="The programmer reproduced the crash and filed a",
        target="bug",
        note="Centered selected-token availability h^T(W_U[bug]-mean(W_U)).",
    ),
    QuerySpec(
        case_id="pairwise_bug_insect",
        role="Named arbitration",
        query_kind="pairwise_margin",
        title="Software sense over insect",
        prompt="The programmer reproduced the crash and filed a",
        target_a="bug",
        target_b="insect",
        note="Pairwise token margin h^T(W_U[bug]-W_U[insect]).",
    ),
    QuerySpec(
        case_id="family_unknown_color",
        role="Token-family arbitration",
        query_kind="token_family_margin",
        title="Abstention family",
        prompt=("The passage says Alice owns a key. It does not say her favorite color. Alice's favorite color is"),
        target_family=("unknown", "Unknown", "unclear", "unavailable", "not"),
        contrast_family=("red", "blue"),
        note="Mean abstention-family rows versus mean forced-color rows.",
    ),
    QuerySpec(
        case_id="selection_bug_topk",
        role="Selection pressure",
        query_kind="winner_vs_topk",
        title="Winner vs local competitors",
        prompt="The programmer reproduced the crash and filed a",
        note="Exact local winner versus a softmax-weighted top-k competitor row average.",
        top_competitors=10,
    ),
)

BUG_TOP5_QUERY_SPECS: tuple[QuerySpec, ...] = (
    *QUERY_SPECS[:4],
    QuerySpec(
        case_id="selection_bug_top5",
        role="Selection pressure",
        query_kind="winner_vs_rank",
        title="Winner vs rank-5 competitor",
        prompt="The programmer reproduced the crash and filed a",
        note="Exact local winner versus the rank-5 competitor row.",
        competitor_rank=5,
    ),
)

JURY_TOP5_QUERY_SPECS: tuple[QuerySpec, ...] = (
    QuerySpec(
        case_id="selected_guilty_raw",
        role="Selected token",
        query_kind="selected_token",
        title="Selected token score",
        prompt="After reviewing the evidence, the jury found the defendant",
        target="guilty",
        note="Raw selected-token score h^T W_U[guilty].",
    ),
    QuerySpec(
        case_id="availability_guilty_mean",
        role="Availability",
        query_kind="target_vs_vocab_mean",
        title="Token vs vocabulary mean",
        prompt="After reviewing the evidence, the jury found the defendant",
        target="guilty",
        note="Centered selected-token availability h^T(W_U[guilty]-mean(W_U)).",
    ),
    QuerySpec(
        case_id="pairwise_guilty_not",
        role="Named arbitration",
        query_kind="pairwise_margin",
        title="Guilty over not",
        prompt="After reviewing the evidence, the jury found the defendant",
        target_a="guilty",
        target_b="not",
        note="Pairwise token margin h^T(W_U[guilty]-W_U[not]).",
    ),
    QUERY_SPECS[3],
    QuerySpec(
        case_id="selection_guilty_top5",
        role="Selection pressure",
        query_kind="winner_vs_rank",
        title="Winner vs rank-5 competitor",
        prompt="After reviewing the evidence, the jury found the defendant",
        note="Exact local winner versus the rank-5 competitor row.",
        competitor_rank=5,
    ),
)

CASE_SETS: dict[str, tuple[QuerySpec, ...]] = {
    "default": QUERY_SPECS,
    "bug_top5": BUG_TOP5_QUERY_SPECS,
    "jury_top5": JURY_TOP5_QUERY_SPECS,
}


def select_display_rows(feature_rows: list[dict[str, object]], *, top_features: int) -> list[dict[str, object]]:
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in feature_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    display: list[dict[str, object]] = []
    for rows in by_case.values():
        selected = sorted(rows, key=lambda row: -float(row["abs_contribution"]))[:top_features]
        display.extend(sorted(selected, key=lambda row: float(row["contribution"])))
    return display


@torch.no_grad()
def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    query_specs = CASE_SETS[args.case_set]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    model, tokenizer = load_qwen_model(args.model_id, dtype, local_files_only=args.local_files_only)
    model.to(device)
    lm_head, lm_head_path = find_lm_head(model)
    W = lm_head.weight.detach().cpu().float().contiguous()
    row_mean = W.mean(dim=0).contiguous()
    decoder, encoder_w, encoder_b, sae_config = load_sae(args.checkpoint)
    if decoder.shape[1] != W.shape[1]:
        raise ValueError(f"SAE d_model {decoder.shape[1]} does not match W_U d_model {W.shape[1]}")

    case_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    token_audit_rows: list[dict[str, object]] = []

    for panel_idx, spec in enumerate(query_specs):
        h, _model_logits = collect_readout_state(model, tokenizer, spec.prompt, device)
        logits = h @ W.T
        top_values, top_ids = torch.topk(logits, k=5)
        weights, mean_subtract, metadata = build_query_weights(
            spec=spec,
            logits=logits,
            tokenizer=tokenizer,
            token_audit_rows=token_audit_rows,
        )
        summary, rows = decompose_query(
            h=h,
            logits=logits,
            W=W,
            row_mean=row_mean,
            decoder=decoder,
            encoder_w=encoder_w,
            encoder_b=encoder_b,
            k=args.k,
            weights=weights,
            mean_subtract=mean_subtract,
        )
        target_id = metadata.get("target_token_id", "")
        target_rank = ""
        target_probability = ""
        if isinstance(target_id, int):
            target_rank, target_probability = token_rank_and_prob(logits, target_id)
        case_row = {
            "panel": chr(ord("A") + panel_idx),
            "case_id": spec.case_id,
            "role": spec.role,
            "query_kind": spec.query_kind,
            "title": spec.title,
            "prompt": spec.prompt,
            "note": spec.note,
            "model_id": args.model_id,
            "checkpoint": str(args.checkpoint),
            "k": args.k,
            "lm_head_path": lm_head_path,
            "vocab": int(W.shape[0]),
            "d_model": int(W.shape[1]),
            "target_rank": target_rank,
            "target_probability": target_probability,
            "top5_token_ids": ";".join(str(int(tid)) for tid in top_ids.tolist()),
            "top5_tokens": "; ".join(clean_token(tokenizer, int(tid)) for tid in top_ids.tolist()),
            "top5_logits": ";".join(f"{float(v):.6g}" for v in top_values.tolist()),
            "weights": json.dumps({str(k): v for k, v in sorted(weights.items())}),
            **metadata,
            **summary,
        }
        case_rows.append(case_row)
        for row in rows:
            feature_rows.append({**case_row, **row})

    display_rows = select_display_rows(feature_rows, top_features=args.top_features)
    attach_feature_labels(
        W=W,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        display_rows=display_rows,
        top_tokens=args.label_top_tokens,
        chunk_size=args.label_chunk_size,
    )

    meta = {
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "k": args.k,
        "case_set": args.case_set,
        "query_case_ids": [spec.case_id for spec in query_specs],
        "sae_config": sae_config,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "qwen2b_general_readout_query_cases.csv", case_rows)
    write_csv(args.out_dir / "qwen2b_general_readout_query_features.csv", feature_rows)
    write_csv(args.out_dir / "qwen2b_general_readout_query_display_features.csv", display_rows)
    write_csv(args.out_dir / "qwen2b_general_readout_query_token_audit.csv", token_audit_rows)
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    torch.save(
        {
            "meta": meta,
            "case_rows": case_rows,
            "feature_rows": feature_rows,
            "display_rows": display_rows,
            "token_audit_rows": token_audit_rows,
        },
        args.out_dir / "qwen2b_general_readout_queries_cache.pt",
    )
    if args.paper_dir is not None:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.paper_dir / "qwen2b_general_readout_query_cases.csv", case_rows)
        write_csv(args.paper_dir / "qwen2b_general_readout_query_features.csv", feature_rows)
        write_csv(args.paper_dir / "qwen2b_general_readout_query_display_features.csv", display_rows)
        write_csv(args.paper_dir / "qwen2b_general_readout_query_token_audit.csv", token_audit_rows)
        (args.paper_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        torch.save(
            {
                "meta": meta,
                "case_rows": case_rows,
                "feature_rows": feature_rows,
                "display_rows": display_rows,
                "token_audit_rows": token_audit_rows,
            },
            args.paper_dir / "qwen2b_general_readout_queries_cache.pt",
        )
    return case_rows, feature_rows, display_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--case-set", choices=sorted(CASE_SETS), default="default")
    parser.add_argument("--top-features", type=int, default=8)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "paper_dir": None if args.paper_dir is None else str(args.paper_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
