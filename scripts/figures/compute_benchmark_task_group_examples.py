"""Compute benchmark-derived task-group target examples for the paper."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sparse_readout_prism.research.qwen_readout import find_lm_head_with_path, load_qwen_model, load_sae
from sparse_readout_prism.research.query_decompose import (
    attach_feature_labels,
    write_csv,
)


DEFAULT_RUN_DIR = Path("results/benchmark_derived_query_suite_task_group_v2_qwen_20260523")
DEFAULT_OUT_DIR = Path("results/benchmark_task_group_examples_qwen2b_20260523")
DEFAULT_PAPER_DIR = Path("paper/figures/qwen2b_benchmark_task_group_targets")


SELECTED_CASES = (
    {
        "case_id": "squad2_ans_56ddde6b9a695914005b9628",
        "panel_title": "SQuAD2: France - distractors",
        "target_summary": "France vs Norman/French/Latin",
    },
    {
        "case_id": "hotpot_5a7759fc5542993569682d60",
        "panel_title": "HotpotQA: Canary - distractors",
        "target_summary": "Canary vs Santa/Cruz/Tenerife",
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run scripts/run/run_benchmark_derived_query_suite.py first "
            f"and point --run-dir at its --out-dir (the two defaults are different run names)"
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_selected_rows(
    run_dir: Path, *, top_features: int, model_slug: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case_rows_all = read_csv(run_dir / "case_rows.csv")
    feature_rows_all = read_csv(run_dir / "top_feature_rows.csv")
    selected_by_id = {case["case_id"]: case for case in SELECTED_CASES}
    case_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for panel_idx, selected in enumerate(SELECTED_CASES):
        matching_cases = [
            row for row in case_rows_all if row["model_slug"] == model_slug and row["case_id"] == selected["case_id"]
        ]
        if len(matching_cases) != 1:
            raise ValueError(f"expected one {model_slug} row for {selected['case_id']}, found {len(matching_cases)}")
        case = {
            **matching_cases[0],
            "panel": chr(ord("A") + panel_idx),
            "panel_title": selected["panel_title"],
            "target_summary": selected["target_summary"],
        }
        case_rows.append(case)
        rows = [
            row for row in feature_rows_all if row["model_slug"] == model_slug and row["case_id"] == selected["case_id"]
        ]
        rows = sorted(rows, key=lambda row: -float(row["abs_contribution"]))[:top_features]
        rows = sorted(rows, key=lambda row: -float(row["contribution"]))
        for display_rank, row in enumerate(rows, start=1):
            feature_rows.append(
                {
                    **case,
                    **row,
                    "display_rank": display_rank,
                    "display_target": selected_by_id[selected["case_id"]]["target_summary"],
                }
            )
    return case_rows, feature_rows


def attach_labels(
    *,
    feature_rows: list[dict[str, object]],
    model_id: str,
    checkpoint: Path,
    dtype: torch.dtype,
    top_tokens: int,
    chunk_size: int,
    local_files_only: bool = False,
) -> None:
    model, tokenizer = load_qwen_model(model_id, dtype, local_files_only=local_files_only)
    lm_head, _lm_head_path = find_lm_head_with_path(model)
    W = lm_head.weight.detach().cpu().float().contiguous()
    row_mean = W.mean(dim=0).contiguous()
    _decoder, encoder_w, encoder_b, _sae_config = load_sae(checkpoint)
    attach_feature_labels(
        W=W,
        row_mean=row_mean,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        display_rows=feature_rows,
        top_tokens=top_tokens,
        chunk_size=chunk_size,
    )
    del model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--paper-dir",
        type=lambda s: Path(s) if s else None,
        default=DEFAULT_PAPER_DIR,
        help="mirror paper-facing outputs here; pass '' to disable",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--model-slug", default="qwen2b", help="model_slug to select in the run CSVs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--top-features", type=int, default=6)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=8192)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    case_rows, feature_rows = load_selected_rows(
        args.run_dir, top_features=args.top_features, model_slug=args.model_slug
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    attach_labels(
        feature_rows=feature_rows,
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        dtype=dtype,
        top_tokens=args.label_top_tokens,
        chunk_size=args.label_chunk_size,
        local_files_only=args.local_files_only,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = args.paper_dir if args.paper_dir else None
    write_csv(args.out_dir / "benchmark_task_group_examples_cases.csv", case_rows)
    write_csv(args.out_dir / "benchmark_task_group_examples_features.csv", feature_rows)
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        write_csv(paper_dir / "benchmark_task_group_examples_cases.csv", case_rows)
        write_csv(paper_dir / "benchmark_task_group_examples_features.csv", feature_rows)
    manifest = {
        "run_dir": str(args.run_dir),
        "out_dir": str(args.out_dir),
        "paper_dir": str(paper_dir) if paper_dir else None,
        "selected_cases": SELECTED_CASES,
        "outputs": [
            "benchmark_task_group_examples_cases.csv",
            "benchmark_task_group_examples_features.csv",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "paper_dir": str(paper_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
