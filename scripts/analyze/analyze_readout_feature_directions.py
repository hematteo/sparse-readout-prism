"""Pre-token readout-direction ranking on paper-selected Qwen SAEs.

This is the prompt-context version of the cached-state probe. For each prompt,
it captures the final readout state immediately before the Qwen LM head and
ranks learned readout SAE decoder directions by:

    score_j = h @ decoder[j]

No target token row is used to select the directions. The unembedding rows are
used only after ranking, to attach token-summary labels to the selected
directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

from sparse_readout_prism.research.qwen_readout import (
    clean_token,
    collect_readout_state,
    find_lm_head,
    display_label_features,
    load_qwen_model,
    load_sae,
    readable_feature_label,
)
from sparse_readout_prism.research.qwen_example_prompts import EXAMPLE_SETS

DEFAULT_OUT_DIR = Path("results/qwen_pre_token_readout_directions")

SELECTED_SAES = {
    "qwen2b_k256": {
        "k": 256,
        "description": "Qwen3.5-2B 32x d65536 k256, fidelity/domain SAE",
    },
}

INTERESTING_CONTEXTS = [
    {
        "case_id": "shell_beach",
        "title": "shell: beach",
        "prompt": "Walking along the beach, she picked up a spiral",
        "paper_context": "same-token/domain shell pair",
    },
    {
        "case_id": "shell_terminal",
        "title": "shell: terminal",
        "prompt": "The command was typed into the Unix",
        "paper_context": "same-token/domain shell pair",
    },
    {
        "case_id": "java_coffee",
        "title": "Java: coffee",
        "prompt": "The cafe served a strong cup of",
        "paper_context": "same-token/domain Java pair",
    },
    {
        "case_id": "java_programming",
        "title": "Java: programming",
        "prompt": "The backend service was written in",
        "paper_context": "same-token/domain Java pair",
    },
    {
        "case_id": "court_legal",
        "title": "court: legal",
        "prompt": "The judge listened as the lawyer presented evidence in the",
        "paper_context": "same-token/domain court pair",
    },
    {
        "case_id": "court_tennis",
        "title": "court: tennis",
        "prompt": "During the match, the player served the ball across the",
        "paper_context": "same-token/domain court pair",
    },
    {
        "case_id": "port_harbor",
        "title": "port: harbor",
        "prompt": "After crossing the ocean, the ship entered the busy",
        "paper_context": "same-token/domain port pair",
    },
    {
        "case_id": "port_network",
        "title": "port: network",
        "prompt": "The web server was configured to listen on a network",
        "paper_context": "same-token/domain port pair",
    },
    {
        "case_id": "cell_prison",
        "title": "cell: prison",
        "prompt": "After the trial, the prisoner was locked in a small",
        "paper_context": "same-token/domain cell pair",
    },
    {
        "case_id": "cell_biology",
        "title": "cell: biology",
        "prompt": "Under the microscope, the biologist examined a single living",
        "paper_context": "same-token/domain cell pair",
    },
    {
        "case_id": "draft_document",
        "title": "draft: document",
        "prompt": "Before sending the final email, the writer revised a rough",
        "paper_context": "same-token/domain draft pair",
    },
    {
        "case_id": "draft_beer",
        "title": "draft: beer",
        "prompt": "At the pub, the bartender poured a cold",
        "paper_context": "same-token/domain draft pair",
    },
]


def paper_contexts(example_set: str) -> list[dict[str, str]]:
    rows = []
    for ex in EXAMPLE_SETS[example_set]:
        rows.append(
            {
                "case_id": ex["case_id"],
                "title": ex["title"],
                "prompt": ex["prompt"],
                "paper_context": str(ex.get("claim", "")),
                "target_a": str(ex.get("target_a", "")),
                "target_b": str(ex.get("target_b", "")),
            }
        )
    return rows


CONTEXT_SETS = {
    "interesting_domains": INTERESTING_CONTEXTS,
    **{key: paper_contexts(key) for key in EXAMPLE_SETS},
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    # delegate to shared helper (empty rows -> no file, matching original)
    from sparse_readout_prism.utils import write_csv as _write_csv

    return _write_csv(path, rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def score_metric(scores: torch.Tensor, rank_by: str) -> torch.Tensor:
    if rank_by == "abs_dot":
        return scores.abs()
    if rank_by == "positive_dot":
        return scores
    if rank_by == "negative_dot":
        return -scores
    raise ValueError(f"unknown rank_by: {rank_by}")


def shorten(text: str, max_len: int) -> str:
    # delegate to shared helper (single-dot ellipsis preserves original behaviour)
    from sparse_readout_prism.token_display import shorten as _shorten

    return _shorten(text, max_len, ellipsis=".")


def compact_prompt(prompt: str, max_len: int = 74) -> str:
    return shorten(prompt, max_len=max_len)


def top_logit_summary(logits: torch.Tensor, tokenizer, *, top_n: int = 5) -> str:
    values, ids = torch.topk(logits, k=min(top_n, logits.numel()))
    parts = []
    for token_id, value in zip(ids.tolist(), values.tolist(), strict=False):
        token = clean_token(tokenizer, int(token_id))
        if not token:
            token = f"<id:{int(token_id)}>"
        parts.append(f"{token}:{float(value):+.2f}")
    return "; ".join(parts)


def context_pair_key(context: dict[str, str]) -> tuple[str, str]:
    title = str(context["title"])
    if ":" not in title:
        return title.strip(), title.strip()
    left, right = title.split(":", 1)
    return left.strip(), right.strip()


@torch.no_grad()
def collect_direction_rows(
    *,
    contexts: list[dict[str, str]],
    model,
    tokenizer,
    decoder: torch.Tensor,
    device: torch.device,
    rank_by: str,
    top_directions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    direction_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    feature_ids: list[int] = []
    for context_id, context in enumerate(contexts):
        h, logits = collect_readout_state(model, tokenizer, context["prompt"], device)
        scores = h @ decoder.T
        metric = score_metric(scores, rank_by)
        values, ids = torch.topk(metric, k=min(top_directions, metric.numel()))
        context_row = {
            "context_id": context_id,
            "case_id": context["case_id"],
            "title": context["title"],
            "prompt": context["prompt"],
            "paper_context": context.get("paper_context", ""),
            "target_a": context.get("target_a", ""),
            "target_b": context.get("target_b", ""),
            "hidden_norm": float(h.norm()),
            "top_logits": top_logit_summary(logits, tokenizer),
            "max_rank_metric": float(values[0]),
        }
        context_rows.append(context_row)
        for rank, (fid_t, value_t) in enumerate(zip(ids.tolist(), values.tolist(), strict=False), start=1):
            fid = int(fid_t)
            score = float(scores[fid])
            direction_rows.append(
                {
                    **context_row,
                    "feature_rank": rank,
                    "feature_id": fid,
                    "rank_metric": float(value_t),
                    "h_dot_direction": score,
                    "abs_h_dot_direction": abs(score),
                    "sign": "positive" if score >= 0 else "negative",
                }
            )
            feature_ids.append(fid)
    return context_rows, direction_rows, sorted(set(feature_ids))


@torch.no_grad()
def collect_pair_delta_rows(
    *,
    contexts: list[dict[str, str]],
    model,
    tokenizer,
    decoder: torch.Tensor,
    device: torch.device,
    rank_by: str,
    top_directions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    state_rows: list[dict[str, object]] = []
    hidden_states: list[torch.Tensor] = []
    for context_id, context in enumerate(contexts):
        h, logits = collect_readout_state(model, tokenizer, context["prompt"], device)
        pair_key, side = context_pair_key(context)
        state_rows.append(
            {
                "context_id": context_id,
                "case_id": context["case_id"],
                "title": context["title"],
                "pair_key": pair_key,
                "side": side,
                "prompt": context["prompt"],
                "paper_context": context.get("paper_context", ""),
                "target_a": context.get("target_a", ""),
                "target_b": context.get("target_b", ""),
                "hidden_norm": float(h.norm()),
                "top_logits": top_logit_summary(logits, tokenizer),
            }
        )
        hidden_states.append(h)

    by_pair: dict[str, list[tuple[dict[str, object], torch.Tensor]]] = {}
    for row, h in zip(state_rows, hidden_states, strict=False):
        by_pair.setdefault(str(row["pair_key"]), []).append((row, h))

    pair_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    feature_ids: list[int] = []
    for pair_id, (pair_key, items) in enumerate(by_pair.items()):
        if len(items) != 2:
            continue
        (left, h_left), (right, h_right) = items
        delta_h = h_left - h_right
        scores = delta_h @ decoder.T
        metric = score_metric(scores, rank_by)
        values, ids = torch.topk(metric, k=min(top_directions, metric.numel()))
        pair_row = {
            "context_id": pair_id,
            "case_id": f"{left['case_id']}__minus__{right['case_id']}",
            "title": f"{pair_key}: {left['side']} - {right['side']}",
            "prompt": f"{left['prompt']}  ||  {right['prompt']}",
            "paper_context": left.get("paper_context", ""),
            "target_a": left.get("target_a", ""),
            "target_b": right.get("target_b", ""),
            "left_case_id": left["case_id"],
            "right_case_id": right["case_id"],
            "left_title": left["title"],
            "right_title": right["title"],
            "left_side": left["side"],
            "right_side": right["side"],
            "left_prompt": left["prompt"],
            "right_prompt": right["prompt"],
            "hidden_norm": float(delta_h.norm()),
            "top_logits": f"{left['side']}: {left['top_logits']} || {right['side']}: {right['top_logits']}",
            "max_rank_metric": float(values[0]),
        }
        pair_rows.append(pair_row)
        for rank, (fid_t, value_t) in enumerate(zip(ids.tolist(), values.tolist(), strict=False), start=1):
            fid = int(fid_t)
            score = float(scores[fid])
            direction_rows.append(
                {
                    **pair_row,
                    "feature_rank": rank,
                    "feature_id": fid,
                    "rank_metric": float(value_t),
                    "h_dot_direction": score,
                    "abs_h_dot_direction": abs(score),
                    "sign": "positive" if score >= 0 else "negative",
                    "larger_in": left["side"] if score >= 0 else right["side"],
                }
            )
            feature_ids.append(fid)
    return pair_rows, direction_rows, sorted(set(feature_ids))


def attach_feature_labels(
    *,
    direction_rows: list[dict[str, object]],
    feature_ids: list[int],
    W: torch.Tensor,
    row_mean: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer,
    label_top_tokens: int,
    label_chunk_size: int,
) -> None:
    labels = display_label_features(
        W=W,
        row_mean=row_mean,
        feature_ids=feature_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=label_top_tokens,
        chunk_size=label_chunk_size,
    )
    for row in direction_rows:
        fid = int(row["feature_id"])
        label_tokens = labels.get(fid, [])
        row["feature_label"] = readable_feature_label(label_tokens, fid, max_len=34)
        row["feature_top_tokens"] = "; ".join(label_tokens)


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    context_rows: list[dict[str, object]],
    direction_rows: list[dict[str, object]],
) -> None:
    by_context = {int(row["context_id"]): row for row in context_rows}
    lines = [
        "# Qwen Pre-Token Readout Directions",
        "",
        "Ranks are computed from hidden states and SAE decoder directions before choosing a token:",
        "",
        "```text",
        "score_j = h @ decoder[j]",
        "pair_delta score_j = (h_left - h_right) @ decoder[j]",
        "```",
        "",
        f"- selected SAE: `{args.sae_id}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- model: `{args.model_id}`",
        f"- context set: `{args.context_set}`",
        f"- analysis mode: `{args.analysis_mode}`",
        f"- rank_by: `{args.rank_by}`",
        "",
        "## Contexts",
        "",
        "| context | prompt | top directions | ordinary top logits |",
        "|---|---|---|---|",
    ]
    for context_id, context in by_context.items():
        top = [row for row in direction_rows if int(row["context_id"]) == context_id and int(row["feature_rank"]) <= 3]
        top_text = "; ".join(f"{row['feature_label']} {float(row['h_dot_direction']):+.2f}" for row in top)
        if "left_prompt" in context and "right_prompt" in context:
            prompt = f"{compact_prompt(str(context['left_prompt']), 42)} / {compact_prompt(str(context['right_prompt']), 42)}"
        else:
            prompt = compact_prompt(str(context["prompt"]), 84)
        lines.append(f"| {context['title']} | {prompt} | {top_text} | {context['top_logits']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `qwen_pre_token_readout_contexts.csv`",
            "- `qwen_pre_token_readout_directions.csv`",
            "- `qwen_pre_token_readout_cache.pt`",
            "- `manifest.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-id", choices=sorted(SELECTED_SAES), default="qwen2b_k256")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--context-set", choices=sorted(CONTEXT_SETS), default="interesting_domains")
    parser.add_argument("--analysis-mode", choices=["contexts", "pair_deltas"], default="contexts")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--rank-by", choices=["abs_dot", "positive_dot", "negative_dot"], default="abs_dot")
    parser.add_argument("--top-directions", type=int, default=10)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--label-top-tokens", type=int, default=5)
    parser.add_argument("--label-chunk-size", type=int, default=4096)
    args = parser.parse_args()
    selected = SELECTED_SAES[args.sae_id]
    if args.k is None:
        args.k = int(selected["k"])
    args.command_line = ["uv", "run", "python", sys.argv[0], *sys.argv[1:]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    model, tokenizer = load_qwen_model(args.model_id, dtype)
    model.to(device)
    lm_head, lm_head_path = find_lm_head(model)
    W = lm_head.weight.detach().float().cpu().contiguous()
    row_mean = W.mean(dim=0)
    decoder, encoder_w, encoder_b, sae_config = load_sae(args.checkpoint)
    if decoder.shape[1] != W.shape[1]:
        raise ValueError(f"SAE d_model {decoder.shape[1]} does not match LM head d_model {W.shape[1]}")

    contexts = CONTEXT_SETS[args.context_set]
    if args.case_ids is not None:
        selected_case_ids = [cid.strip() for cid in args.case_ids.split(",") if cid.strip()]
        case_id_order = {cid: i for i, cid in enumerate(selected_case_ids)}
        contexts = [ctx for ctx in contexts if ctx["case_id"] in case_id_order]
        contexts = sorted(contexts, key=lambda ctx: case_id_order[ctx["case_id"]])
        if not contexts:
            raise ValueError(f"no contexts in set '{args.context_set}' match --case-ids {args.case_ids}")
    if args.analysis_mode == "pair_deltas":
        context_rows, direction_rows, feature_ids = collect_pair_delta_rows(
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            decoder=decoder,
            device=device,
            rank_by=args.rank_by,
            top_directions=args.top_directions,
        )
    else:
        context_rows, direction_rows, feature_ids = collect_direction_rows(
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            decoder=decoder,
            device=device,
            rank_by=args.rank_by,
            top_directions=args.top_directions,
        )
    attach_feature_labels(
        direction_rows=direction_rows,
        feature_ids=feature_ids,
        W=W,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        label_top_tokens=args.label_top_tokens,
        label_chunk_size=args.label_chunk_size,
    )

    write_csv(args.out_dir / "qwen_pre_token_readout_contexts.csv", context_rows)
    write_csv(args.out_dir / "qwen_pre_token_readout_directions.csv", direction_rows)
    cache = {
        "contexts": context_rows,
        "directions": direction_rows,
        "feature_ids": feature_ids,
        "meta": {
            "sae_id": args.sae_id,
            "checkpoint": str(args.checkpoint),
            "k": args.k,
            "model_id": args.model_id,
            "context_set": args.context_set,
            "analysis_mode": args.analysis_mode,
            "rank_by": args.rank_by,
            "lm_head_path": lm_head_path,
            "d_model": int(W.shape[1]),
            "vocab": int(W.shape[0]),
            "sae_config": sae_config,
            "command_line": args.command_line,
        },
    }
    torch.save(cache, args.out_dir / "qwen_pre_token_readout_cache.pt")
    write_json(
        args.out_dir / "manifest.json",
        {
            "description": "Pre-token readout direction ranking on paper-selected Qwen readout SAE.",
            "sae_id": args.sae_id,
            "sae_description": selected["description"],
            "checkpoint": str(args.checkpoint),
            "k": args.k,
            "model_id": args.model_id,
            "context_set": args.context_set,
            "analysis_mode": args.analysis_mode,
            "rank_by": args.rank_by,
            "lm_head_path": lm_head_path,
            "d_model": int(W.shape[1]),
            "vocab": int(W.shape[0]),
            "command_line": args.command_line,
            "outputs": [
                "qwen_pre_token_readout_contexts.csv",
                "qwen_pre_token_readout_directions.csv",
                "qwen_pre_token_readout_cache.pt",
                "summary.md",
            ],
        },
    )
    write_summary(
        args.out_dir / "summary.md",
        args=args,
        context_rows=context_rows,
        direction_rows=direction_rows,
    )
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
