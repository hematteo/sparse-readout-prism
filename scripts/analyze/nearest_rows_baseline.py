#!/usr/bin/env python3
"""Nearest unembedding rows of selected contrast directions (``tab:app-nearest-rows``).

The zero-fitting reading of a selected readout direction q = w_a - w_b ranks
the vocabulary rows of W_U by |cosine(w_t, q)| and lists the top tokens, with
the two contrast tokens themselves excluded. Rows are ranked both row-centered
(w_t - mu, the SRP preprocessing; q is unchanged by centering) and raw, and the
signed cosine is reported so the supported side of the contrast is visible
(positive supports the first token).

Paper run: Qwen3.5-2B at HF revision 15852e8c16360a2fea060d615a32b45270f8a8fc
(tied embeddings, lm_head.weight == embed_tokens.weight, (248320, 2048)), the
contrasts " bug" - " insect" and " bug" - " error" as leading-space single
tokens, top 12 rows by |cosine|:

    uv run python scripts/analyze/nearest_rows_baseline.py \\
        --model-id Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \\
        --contrast bug,insect --contrast bug,error --top-n 12 \\
        --out-dir results/nearest_rows_qwen2b

The centering mean is the full-vocabulary mean of W_U
(``data.resolve_row_mean`` with no checkpoint), which is what the table used;
``--checkpoint`` centers with a dictionary's stored training mean instead.
``--w-u`` reads W_U from an extraction artifact (``W_U_orig`` written by
``scripts/data/extract_model_readout.py``) instead of loading the model; the
tokenizer is still loaded from ``--model-id``. No random numbers are used.

Outputs: ``nearest_rows.json`` (token report, per-contrast centered and raw
rankings, provenance), ``nearest_rows.csv`` (long form: contrast, variant,
rank, token_id, token, cosine) and ``nearest_rows_table.csv`` (the table
layout: per rank the centered and raw token with cosines rounded to two
decimals and glosses for the non-Latin tokens that occur in the paper's lists).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sparse_readout_prism.data import resolve_row_mean
from sparse_readout_prism.research.qwen_readout import load_qwen_model
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import find_lm_head_with_path

# Glosses for the non-Latin tokens that enter the paper's top-12 lists.
GLOSS = {
    "昆虫": "insect",
    "故障": "malfunction",
    "误差": "(measurement) error",
    "错误": "error",
    "漏洞": "vulnerability",
    " насеком": "insect- (ru)",
    " насекомых": "insects (ru)",
    " ошибки": "errors (ru)",
}


def single_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def token_report(tokenizer, words: list[str]) -> dict[str, dict]:
    """Single-token check for each contrast word with and without a leading space."""
    report: dict[str, dict] = {}
    for w in words:
        with_sp = single_token_id(tokenizer, " " + w)
        without_sp = single_token_id(tokenizer, w)
        report[w] = {
            "with_leading_space": {"id": with_sp, "single_token": with_sp is not None},
            "without_leading_space": {"id": without_sp, "single_token": without_sp is not None},
        }
    return report


def resolve_contrast_ids(tokenizer, words: list[str], *, leading_space: bool) -> dict[str, int]:
    ids: dict[str, int] = {}
    for w in words:
        text = (" " + w) if leading_space else w
        tid = single_token_id(tokenizer, text)
        if tid is None:
            raise ValueError(f"{text!r} is not a single token")
        if tokenizer.decode([tid]) != text:
            raise ValueError(f"{text!r} does not round-trip: id {tid} decodes to {tokenizer.decode([tid])!r}")
        ids[w] = tid
    return ids


@torch.no_grad()
def top_rows(tokenizer, q: torch.Tensor, unit_rows: torch.Tensor, exclude: set[int], n: int) -> list[dict]:
    """Top-``n`` rows of ``unit_rows`` by |cosine| with ``q``, excluded ids ranked last."""
    q_unit = q / q.norm()
    cos = unit_rows @ q_unit  # (V,)
    score = cos.abs().clone()
    for i in exclude:
        score[i] = -1.0
    idx = torch.topk(score, n).indices
    return [
        {
            "rank": r + 1,
            "token_id": int(i),
            "token": tokenizer.decode([int(i)]),
            "cosine": round(float(cos[i]), 4),
        }
        for r, i in enumerate(idx.tolist())
    ]


@torch.no_grad()
def nearest_rows(
    W_U: torch.Tensor,
    row_mean: torch.Tensor,
    tokenizer,
    contrasts: list[tuple[str, str]],
    *,
    top_n: int,
    leading_space: bool = True,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, int]]:
    """Centered and raw |cosine| rankings for each ``(a, b)`` contrast direction w_a - w_b."""
    words = list(dict.fromkeys(w for pair in contrasts for w in pair))
    report = token_report(tokenizer, words)
    ids = resolve_contrast_ids(tokenizer, words, leading_space=leading_space)
    W_c = W_U - row_mean  # (V, d) centered
    W_c_unit = W_c / W_c.norm(dim=1, keepdim=True).clamp_min(1e-8)
    W_raw_unit = W_U / W_U.norm(dim=1, keepdim=True).clamp_min(1e-8)
    prefix = " " if leading_space else ""
    out: dict[str, dict] = {}
    for a, b in contrasts:
        name = f"{a}_minus_{b}"
        q = W_U[ids[a]] - W_U[ids[b]]  # row_mean cancels: centered q == raw q
        exclude = {ids[a], ids[b]}
        out[name] = {
            "definition": f"w_[{prefix}{a}] - w_[{prefix}{b}]  (ids {ids[a]}, {ids[b]})",
            "q_norm": round(float(q.norm()), 4),
            "cos_q_centered_rows": {
                f"{prefix}{a}": round(float(W_c_unit[ids[a]] @ (q / q.norm())), 4),
                f"{prefix}{b}": round(float(W_c_unit[ids[b]] @ (q / q.norm())), 4),
            },
            "top_centered": top_rows(tokenizer, q, W_c_unit, exclude, top_n),
            "top_raw": top_rows(tokenizer, q, W_raw_unit, exclude, top_n),
        }
    return out, report, ids


def table_rows(contrast_results: dict[str, dict]) -> list[dict]:
    """Side-by-side centered / raw listing per rank, cosines to two decimals, glosses attached."""
    rows: list[dict] = []
    for name, c in contrast_results.items():
        for cen, raw in zip(c["top_centered"], c["top_raw"]):
            rows.append(
                {
                    "contrast": name,
                    "rank": cen["rank"],
                    "centered_cosine": f"{cen['cosine']:+.2f}",
                    "centered_token": cen["token"],
                    "centered_gloss": GLOSS.get(cen["token"], ""),
                    "raw_cosine": f"{raw['cosine']:+.2f}",
                    "raw_token": raw["token"],
                    "raw_gloss": GLOSS.get(raw["token"], ""),
                }
            )
    return rows


def parse_contrast(spec: str) -> tuple[str, str]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"contrast must be 'a,b', got {spec!r}")
    return parts[0], parts[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    ap.add_argument(
        "--revision", default=None, help="HF weight revision to pin (paper: 15852e8c16360a2fea060d615a32b45270f8a8fc)"
    )
    ap.add_argument("--w-u", type=Path, default=None, help="extraction artifact with W_U_orig; skips the model load")
    ap.add_argument(
        "--checkpoint", type=Path, default=None, help="dictionary checkpoint whose stored row_mean centers the rows"
    )
    ap.add_argument(
        "--contrast",
        type=parse_contrast,
        action="append",
        default=None,
        help="'a,b' for the direction w_a - w_b (repeatable; default bug,insect and bug,error)",
    )
    ap.add_argument(
        "--no-leading-space", action="store_true", help="resolve the contrast words without a leading space"
    )
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    contrasts = args.contrast or [("bug", "insect"), ("bug", "error")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.w_u is not None:
        from transformers import AutoTokenizer

        art = torch.load(args.w_u, map_location="cpu", weights_only=True)
        W_U = art.get("W_U_orig", art.get("W_U")).float()
        w_u_source = f"artifact W_U_orig: {args.w_u}"
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    else:
        model, tokenizer = load_qwen_model(args.model_id, torch.float32, revision=args.revision)
        lm_head, lm_head_path = find_lm_head_with_path(model)
        W_U = lm_head.weight.detach().float().cpu()
        w_u_source = f"{lm_head_path}.weight (tie_word_embeddings={getattr(model.config, 'tie_word_embeddings', None)})"
        del model
    V, d = W_U.shape
    print(f"W_U: {V} x {d} from {w_u_source}", flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
    row_mean = resolve_row_mean(W_U, ckpt=ckpt)

    contrast_results, report, ids = nearest_rows(
        W_U, row_mean, tokenizer, contrasts, top_n=args.top_n, leading_space=not args.no_leading_space
    )
    prefix = "" if args.no_leading_space else " "
    results = {
        "model_id": args.model_id,
        "revision": args.revision,
        "w_u_source": w_u_source,
        "vocab": V,
        "d_model": d,
        "tokenization": "single tokens " + ", ".join(repr(prefix + w) for w in ids),
        "token_report": report,
        "centering": (
            "row_mean from checkpoint " + str(args.checkpoint)
            if ckpt is not None and ckpt.get("row_mean") is not None
            else "mu = W_U.mean(dim=0), full-vocabulary mean (preprocess_rows centering)"
        ),
        "ranking": f"top-{args.top_n} by |cosine|; signed cosine reported; contrast tokens excluded",
        "contrasts": contrast_results,
        "provenance": run_provenance(args),
    }
    (args.out_dir / "nearest_rows.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    with (args.out_dir / "nearest_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "variant", "rank", "token_id", "token", "cosine"])
        for name, c in contrast_results.items():
            for variant in ("top_centered", "top_raw"):
                for row in c[variant]:
                    w.writerow([name, variant, row["rank"], row["token_id"], repr(row["token"]), row["cosine"]])

    table = table_rows(contrast_results)
    with (args.out_dir / "nearest_rows_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    for name, c in contrast_results.items():
        print(f"\n=== {name}  ({c['definition']}) ===")
        for variant in ("top_centered", "top_raw"):
            print(f"  -- {variant} --")
            for row in c[variant]:
                print(f"  {row['rank']:>2}  {row['cosine']:+.4f}  {row['token']!r}")
    print(f"\nWrote {args.out_dir / 'nearest_rows.json'}, nearest_rows.csv and nearest_rows_table.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
