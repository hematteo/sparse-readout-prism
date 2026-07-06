"""All-layer Qwen3.5-2B logit-lens table with current Sparse Readout Prism SAE."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sparse_readout_prism.research.qwen_readout import (
    clean_token,
    find_lm_head_with_path,
    display_label_features,
    load_sae,
    readable_feature_label,
)
from sparse_readout_prism.utils import write_csv


DEFAULT_PROMPT = "It was the best of times, it was the worst of times"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_OUT_DIR = Path("results/qwen2b_32x_dickens_best_times_all_layer")
DEFAULT_PAPER_DIR = Path("paper/figures/qwen2b_32x_dickens_best_times_all_layer")
DEFAULT_LABEL_GLOBS = (
    "paper/figures/qwen2b_32x_*/**/*.csv",
    "paper/figures/qwen2b_general_readout_queries/*.csv",
)
PUNCT_CHARS = set(".,;:!?-_\"'()[]{}")
CODE_LABEL_WORDS = {
    "ampl",
    "aspnet",
    "bytearray",
    "cgi",
    "ental",
    "eventsystems",
    "format",
    "fwd",
    "href",
    "is",
    "mpl",
    "nbsp",
    "nscoder",
    "ontal",
    "posts",
    "purecomponent",
    "setitem",
    "sheet",
    "subject",
    "suspendlayout",
    "trip",
    "webelementx",
}


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for raw, replacement in replacements.items():
        out = out.replace(raw, replacement)
    return out


def latex_tt(text: str) -> str:
    return rf"\texttt{{{latex_escape(text)}}}"


def short_text(text: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def ascii_text(text: str) -> str:
    text = str(text).replace("\n", "\\n").replace("\t", "\\t")
    text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def display_token(tokenizer, token_id: int, *, max_len: int = 14) -> str:
    text = ascii_text(clean_token(tokenizer, int(token_id)))
    if not text:
        text = ascii_text(tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False))
    text = text.strip() or "space"
    return short_text(text, max_len)


def load_label_cache(globs: tuple[str, ...]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for pattern in globs:
        for path in sorted(Path().glob(pattern)):
            try:
                with path.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or "feature_id" not in reader.fieldnames:
                        continue
                    for row in reader:
                        raw = row.get("feature_id")
                        if raw in (None, ""):
                            continue
                        try:
                            fid = int(float(raw))
                        except ValueError:
                            continue
                        label = (
                            row.get("feature_label")
                            or row.get("feature_token_summary")
                            or row.get("feature_top_tokens")
                            or ""
                        )
                        label = short_text(ascii_text(label), 32)
                        if label and fid not in labels:
                            labels[fid] = label
            except OSError:
                continue
    return labels


def split_label_tokens(text: str) -> list[str]:
    parts: list[str] = []
    for part in re.split(r"[;|]", text or ""):
        part = ascii_text(part)
        if part:
            parts.append(part)
    return parts


def token_summary(tokens: list[str], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        token = ascii_text(token)
        if token and token not in seen:
            seen.add(token)
            out.append(token)
        if len(out) >= limit:
            break
    return out


def contains_code_marker(tokens: list[str]) -> bool:
    for token in tokens:
        stripped = token.strip()
        low = stripped.lower().strip("._-/()[]{}")
        if low in CODE_LABEL_WORDS:
            return True
        if stripped.startswith("_") and len(stripped) > 2:
            return True
        if stripped.startswith(".") and stripped.lower() not in {".it", ".the", ".first"}:
            return True
    return False


def compact_counter(counter: Counter[str], n: int = 4) -> str:
    return ", ".join(f"{key} ({value})" for key, value in counter.most_common(n))


def normalized_label_token(token: str) -> str:
    return token.lower().strip("._-/()[]{}\"'")


def feature_description_from_tokens(tokens: list[str], predictions: list[tuple[str, int]]) -> str:
    toks = token_summary(tokens)
    raw_low = [token.lower() for token in toks]
    low = [normalized_label_token(token) for token in toks]
    raw_head_low = raw_low[:4]
    head_low = low[:4]
    joined = " / ".join(toks[:4]) if toks else ""

    if any(token in {"was", "were", "is", "are"} for token in raw_head_low):
        return f"copula/auxiliary readout ({joined})"
    if "best" in head_low:
        return f"best/superlative lexical readout ({joined})"
    if any(token in {"times", "-times", "_times", "time"} for token in head_low):
        return f"time/times lexical readout ({joined})"
    comparison_words = {
        "darkest",
        "first",
        "greatest",
        "least",
        "lowest",
        "naj",
        "najbardziej",
        "paling",
        "quickest",
        "same",
        "worst",
    }
    if any(token in comparison_words for token in head_low):
        return f"comparison/superlative lexical readout ({joined})"
    if any(token in {"damn", "darn", "fucking", "freaking"} for token in head_low):
        return f"profanity or expletive lexical readout ({joined})"
    if sum(token in {"marta", "magdalena", "manuela", "gabriela"} for token in low[:8]) >= 2:
        return f"proper-name readout ({joined})"
    if any(token in {"ten", "_ten"} for token in head_low):
        return f"numeric/number-word readout ({joined})"
    if toks and sum(bool(re.fullmatch(r"[0-9]+", token)) for token in toks[:8]) >= min(4, len(toks)):
        return f"digit/number readout ({joined})"
    function_words = {"a", "an", "and", "but", "for", "i", "in", "it", "of", "that", "the", "to", "with"}
    function_hits = [token for token in low[:8] if token in function_words]
    if len(set(function_hits)) >= 3:
        return f"mixed function-word readout ({joined})"
    if head_low[:1] == ["it"] or sum(token == "it" for token in low[:6]) >= 2:
        return f"pronoun readout for it/It ({joined})"
    if head_low[:1] == ["the"] or sum(token == "the" for token in low[:6]) >= 2:
        return f"determiner readout for the ({joined})"
    if head_low[:1] and head_low[0] in {"a", "an"}:
        return f"article/determiner readout ({joined})"
    if head_low[:1] and head_low[0] in {"of", "ofs"}:
        return f"preposition/function-word readout for of ({joined})"
    if toks and all(set(token) <= PUNCT_CHARS for token in toks):
        return f"punctuation readout for {joined}"
    if contains_code_marker(toks):
        return f"code or markup-shaped lexical readout ({joined})"
    if toks and any(re.search(r"[A-Z][a-z]{2,}", token) for token in toks):
        return f"proper-name or capitalized-token readout ({joined})"
    if toks:
        main_pred = predictions[0][0] if predictions else ""
        if main_pred and main_pred != "space":
            return f"fragment/readout feature associated here with `{main_pred}`; top rows {joined}"
        return f"lexical fragment readout ({joined})"
    if predictions:
        return f"feature associated here with `{predictions[0][0]}` predictions"
    return "uninterpreted sparse readout feature"


def description_confidence(row: dict[str, Any]) -> str:
    n_cells = int(row["n_cells"])
    desc = str(row["description"]).lower()
    clear_markers = (
        "best/superlative",
        "article/determiner",
        "comparison/superlative",
        "copula",
        "determiner",
        "digit/number",
        "mixed function-word",
        "numeric/number-word",
        "ordinal/comparison",
        "preposition",
        "pronoun",
        "profanity",
        "proper-name readout",
        "time/times",
    )
    clear = any(marker in desc for marker in clear_markers)
    if n_cells >= 5 and clear:
        return "medium"
    if n_cells >= 3 and clear:
        return "low-medium"
    if n_cells >= 5:
        return "low-medium"
    return "low"


def build_feature_description_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fid[int(row["feature_id"])].append(row)

    out_rows: list[dict[str, Any]] = []
    for fid, feature_rows in sorted(by_fid.items(), key=lambda item: (-len(item[1]), item[0])):
        label = str(feature_rows[0].get("feature_label") or f"f{fid}")
        tokens = split_label_tokens(str(feature_rows[0].get("feature_top_tokens") or ""))
        token_labels = token_summary(tokens)
        pred_counts: Counter[str] = Counter(str(row["prediction_token"]) for row in feature_rows)
        input_counts: Counter[str] = Counter(str(row["input_token"]) for row in feature_rows)
        layers = [int(row["layer"]) for row in feature_rows]
        contribs = [float(row["contribution_to_centered_score"]) for row in feature_rows]
        rank_counts: Counter[str] = Counter(str(row["feature_rank"]) for row in feature_rows)
        main_pred = pred_counts.most_common(1)[0][0]
        main_inputs = ", ".join(key for key, _ in input_counts.most_common(2))
        row = {
            "feature_id": int(fid),
            "n_cells": len(feature_rows),
            "layer_range": f"{min(layers)}-{max(layers)}",
            "label_token_count": len(token_labels),
            "top_tokens": "; ".join(token_labels),
            "feature_label": label,
            "description": feature_description_from_tokens(token_labels, pred_counts.most_common(5)),
            "table_context": f"In this table it mainly supports `{main_pred}` cells around input {main_inputs}.",
            "main_predictions": compact_counter(pred_counts),
            "main_input_tokens": compact_counter(input_counts),
            "mean_contribution": f"{statistics.mean(contribs):.3f}",
            "max_contribution": f"{max(contribs):.3f}",
            "ranks": compact_counter(rank_counts, n=3),
        }
        row["confidence"] = description_confidence(row)
        out_rows.append(row)
    return out_rows


def compact_token_list(tokens: str, n: int = 4) -> str:
    return "; ".join(split_label_tokens(tokens)[:n])


def write_feature_key_table(
    f,
    rows: list[dict[str, Any]],
    *,
    confidence: str,
    label: str,
    caption: str,
) -> None:
    selected = [row for row in rows if row["confidence"] == confidence]
    if not selected:
        return
    f.write("\\begin{table}[p]\n")
    f.write("\\caption{" + caption + "}\n")
    f.write("\\label{" + label + "}\n")
    f.write("\\centering\n")
    f.write("\\scriptsize\n")
    f.write("\\setlength{\\tabcolsep}{2pt}\n")
    f.write("\\begin{tabular}{@{}lrrp{0.17\\linewidth}p{0.25\\linewidth}p{0.36\\linewidth}@{}}\n")
    f.write("\\toprule\n")
    f.write("Feature & Cells & Layers & Main cells & Top token rows & Summary \\\\\n")
    f.write("\\midrule\n")
    for row in selected:
        feature = latex_tt(f"f{row['feature_id']}")
        cells = str(row["n_cells"])
        layers = latex_escape(str(row["layer_range"]))
        main = latex_tt(str(row["main_predictions"]))
        top_tokens = latex_tt(compact_token_list(str(row["top_tokens"])))
        description = latex_escape(str(row["description"]))
        f.write(f"{feature} & {cells} & {layers} & {main} & {top_tokens} & {description} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n\n")


def write_feature_key_tex(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    tex_path = out_dir / "qwen2b_32x_dickens_feature_key.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% Generated by scripts/figures/compute_all_layer_literature_prompt.py.\n")
        f.write("% The descriptions are token-label summaries, not validated concepts or causal mechanisms.\n\n")
        write_feature_key_table(
            f,
            rows,
            confidence="medium",
            label="tab:app-dickens-feature-key-medium",
            caption=(
                "\\textbf{Medium-confidence feature key for the Dickens dense display.} "
                "Rows list generated token-label summaries for recurring sparse readout "
                "features in Figure~\\ref{fig:app-dickens-all-layer-proto-lens}. "
                "The top-token column shows the first four of the twelve decoded token rows "
                "used for the audit."
            ),
        )
        write_feature_key_table(
            f,
            rows,
            confidence="low-medium",
            label="tab:app-dickens-feature-key-lowmedium",
            caption=(
                "\\textbf{Low-medium-confidence feature key for the Dickens dense display.} "
                "These rows are repeated or lexically suggestive enough to be useful as a "
                "reading aid, but remain heuristic token-label summaries."
            ),
        )


def write_feature_descriptions(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    description_rows = build_feature_description_rows(rows)
    csv_path = out_dir / "qwen2b_32x_dickens_feature_descriptions.csv"
    md_path = out_dir / "qwen2b_32x_dickens_feature_descriptions.md"
    write_csv(csv_path, description_rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3.5-2B Dickens Table Feature Descriptions\n\n")
        f.write(
            "These descriptions are token-label summaries of sparse readout features "
            "appearing in `qwen2b_32x_dickens_all_layer_lens`. They are based on top "
            "unembedding rows and the cells where the feature appears in this table; "
            "they are not causal or validated semantic-concept claims.\n\n"
        )
        f.write(
            "Evidence note: confidence is only medium for repeated, transparent lexical "
            "features, low-medium for repeated but less transparent features, and low "
            "for one-off or fragmentary features.\n\n"
        )
        f.write(
            "| feature | label tokens | confidence | cells | layers | concise description | "
            "table context | top tokens |\n"
        )
        f.write("|---:|---:|---|---:|---:|---|---|---|\n")
        for row in description_rows:
            desc = str(row["description"]).replace("|", "/")
            context = str(row["table_context"]).replace("|", "/")
            tokens = str(row["top_tokens"]).replace("|", "/")
            f.write(
                f"| f{row['feature_id']} | {row['label_token_count']} | {row['confidence']} | "
                f"{row['n_cells']} | {row['layer_range']} | {desc} | {context} | `{tokens}` |\n"
            )
    write_feature_key_tex(out_dir, description_rows)


def load_model(model_id: str, dtype: torch.dtype, *, local_files_only: bool):
    from sparse_readout_prism.utils import load_causal_lm

    return load_causal_lm(model_id, dtype=dtype, device_map=None, local_files_only=local_files_only)


def final_norm_for_layer(model, hidden: tuple[torch.Tensor, ...], layer_index: int) -> torch.Tensor:
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    h = hidden[layer_index]
    if layer_index == n_layers:
        return h
    norm = getattr(getattr(model, "model", None), "norm", None)
    if norm is None:
        raise SystemExit(
            f"{type(model).__name__} has no .model.norm submodule — this all-layer script "
            "assumes that layout; adapt final_norm_for_layer for other architectures"
        )
    return norm(h)


@torch.no_grad()
def collect_cells(
    *,
    model,
    tokenizer,
    model_id: str,
    prompt: str,
    checkpoint: Path,
    k: int,
    layers: list[int],
    positions: list[int],
    device: torch.device,
    label_cache: dict[int, str],
    display_features_per_cell: int,
    label_top_tokens: int,
    display_label_tokens: int,
    label_chunk_size: int,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    list[list[str]],
    np.ndarray,
    list[list[str]],
    list[str],
    dict[str, Any],
]:
    lm_head, lm_head_path = find_lm_head_with_path(model)
    W = lm_head.weight.detach().float().cpu().contiguous()
    vocab, d_model = W.shape
    decoder, encoder_w, encoder_b, sae_config = load_sae(checkpoint)
    decoder = decoder.float().contiguous()
    encoder_w = encoder_w.float().contiguous()
    encoder_b = encoder_b.float().contiguous()
    if decoder.shape[1] != d_model:
        raise ValueError(f"SAE d_model {decoder.shape[1]} does not match W_U d_model {d_model}")

    row_mean = W.mean(dim=0)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("prompt tokenized to no tokens")
    max_position = max(positions)
    if max_position >= len(prompt_ids):
        raise ValueError(f"position {max_position} outside prompt token range 0..{len(prompt_ids) - 1}")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    hidden = out.hidden_states

    logit_values = np.zeros((len(layers), len(positions)), dtype=np.float32)
    feature_values = np.zeros_like(logit_values)
    logit_labels: list[list[str]] = []
    feature_id_grid: list[list[list[int]]] = []
    rows: list[dict[str, Any]] = []

    vocab_size = min(len(tokenizer), W.shape[0])
    W_vocab = W[:vocab_size]
    decoder_t = decoder.T.contiguous()

    for row_idx, layer in enumerate(layers):
        h_layer = final_norm_for_layer(model, hidden, layer)[0].detach().float().cpu().contiguous()
        logit_row: list[str] = []
        feature_row_ids: list[list[int]] = []
        for col_idx, pos in enumerate(positions):
            h = h_layer[pos]
            logits = h @ W_vocab.T
            probs = torch.softmax(logits.float(), dim=-1)
            top_prob, top_id_t = probs.max(dim=-1)
            top_id = int(top_id_t.item())
            raw_logit = float(logits[top_id].item())
            mean_logit = float(h.dot(row_mean).item())
            exact_centered = raw_logit - mean_logit

            centered = W[top_id].float() - row_mean
            row_norm = centered.norm().clamp_min(1e-8)
            x = centered / row_norm
            acts = F.relu(x @ encoder_w.T + encoder_b)
            kk = min(int(k), acts.shape[-1])
            values, indices = torch.topk(acts, k=kk)
            code = torch.zeros_like(acts)
            code.scatter_(0, indices, values)
            h_dot_d = h @ decoder_t
            contrib = row_norm * code * h_dot_d
            positive = torch.clamp(contrib, min=0.0)
            total_positive = float(positive.sum().clamp_min(1e-12).item())
            active = torch.nonzero(positive > 0, as_tuple=False).flatten()
            if active.numel() == 0:
                active = torch.nonzero(code > 0, as_tuple=False).flatten()
                score = contrib[active].abs() if active.numel() else contrib.abs()
            else:
                score = positive[active]
            if active.numel() > 0:
                selected_values, order = score.topk(min(display_features_per_cell, active.numel()))
                selected = active[order]
            else:
                selected_values, selected = contrib.abs().topk(display_features_per_cell)
            selected_positive = float(torch.clamp(contrib[selected], min=0.0).sum().item())
            feature_values[row_idx, col_idx] = selected_positive / max(total_positive, 1e-12)
            sparse_sum = float(contrib.sum().item())
            residual = exact_centered - sparse_sum

            logit_values[row_idx, col_idx] = float(top_prob.item())
            top_label = display_token(tokenizer, top_id, max_len=14)
            logit_labels_row_label = top_label
            logit_row.append(logit_labels_row_label)

            cell_feature_ids: list[int] = []
            for rank, fid_t in enumerate(selected.tolist(), start=1):
                fid = int(fid_t)
                cached = label_cache.get(fid, "")
                label = cached if cached else f"f{fid}"
                label = short_text(label, 22)
                if not label.startswith("f"):
                    label = f"f{fid} {label}"
                cell_feature_ids.append(fid)
                rows.append(
                    {
                        "layer": int(layer),
                        "position": int(pos),
                        "input_token": display_token(tokenizer, prompt_ids[pos], max_len=18),
                        "target_next_token": (
                            display_token(tokenizer, prompt_ids[pos + 1], max_len=18)
                            if pos + 1 < len(prompt_ids)
                            else ""
                        ),
                        "prediction_token_id": top_id,
                        "prediction_token": top_label,
                        "prediction_probability": float(top_prob.item()),
                        "raw_logit": raw_logit,
                        "mean_vocab_logit": mean_logit,
                        "exact_centered_score": exact_centered,
                        "sparse_feature_sum": sparse_sum,
                        "residual": residual,
                        "residual_abs_over_direct": abs(residual) / max(abs(exact_centered), 1e-8),
                        "feature_rank": rank,
                        "feature_id": fid,
                        "feature_label": label,
                        "feature_top_tokens": "",
                        "target_feature_activation": float(code[fid].item()),
                        "feature_hidden_score": float(h_dot_d[fid].item()),
                        "contribution_to_centered_score": float(contrib[fid].item()),
                        "top_positive_contribution_share": float(feature_values[row_idx, col_idx]),
                        "total_positive_contribution": total_positive,
                    }
                )
            feature_row_ids.append(cell_feature_ids)
        logit_labels.append(logit_row)
        feature_id_grid.append(feature_row_ids)

    input_labels = [display_token(tokenizer, prompt_ids[pos], max_len=18) for pos in positions]
    label_ids = sorted({int(row["feature_id"]) for row in rows})
    computed_labels = display_label_features(
        W=W,
        row_mean=row_mean,
        feature_ids=label_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=label_top_tokens,
        chunk_size=label_chunk_size,
    )
    label_by_id: dict[int, str] = {}
    top_tokens_by_id: dict[int, str] = {}
    for fid in label_ids:
        top_tokens = computed_labels.get(fid, [])
        top_tokens_by_id[fid] = "; ".join(ascii_text(token) for token in top_tokens)
        if top_tokens:
            label_by_id[fid] = readable_feature_label(top_tokens[:display_label_tokens], fid, max_len=32)
        else:
            label_by_id[fid] = label_cache.get(fid, f"f{fid}")
        label_by_id[fid] = short_text(ascii_text(label_by_id[fid]), 32) or f"f{fid}"
    for row in rows:
        fid = int(row["feature_id"])
        row["feature_label"] = label_by_id[fid]
        row["feature_top_tokens"] = top_tokens_by_id.get(fid, "")
    feature_labels = [
        ["\n".join(short_text(label_by_id.get(fid, f"f{fid}"), 22) for fid in cell) for cell in row]
        for row in feature_id_grid
    ]
    meta = {
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "k": int(k),
        "sae_config": sae_config,
        "lm_head_path": lm_head_path,
        "vocab": int(vocab),
        "d_model": int(d_model),
        "prompt_token_ids": prompt_ids,
        "feature_label_method": "Top Qwen unembedding rows by current-SAE encoder preactivation.",
    }
    return rows, logit_values, logit_labels, feature_values, feature_labels, input_labels, meta


def parse_layers(raw: str, n_layers: int) -> list[int]:
    if raw == "all":
        return list(range(n_layers + 1))
    layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    for layer in layers:
        if layer < 0 or layer > n_layers:
            raise ValueError(f"layer {layer} outside range 0..{n_layers}")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--num-cols", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--paper-dir",
        type=lambda s: Path(s) if s else None,
        default=DEFAULT_PAPER_DIR,
        help="mirror paper-facing outputs here; pass '' to disable",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--display-features-per-cell", type=int, default=2)
    parser.add_argument("--label-top-tokens", type=int, default=12)
    parser.add_argument("--display-label-tokens", type=int, default=2)
    parser.add_argument("--label-chunk-size", type=int, default=4096)
    parser.add_argument("--label-glob", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    model, tokenizer = load_model(args.model_id, dtype, local_files_only=args.local_files_only)
    model.to(device)
    n_layers = int(getattr(model.config, "num_hidden_layers"))
    layers = parse_layers(args.layers, n_layers)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    positions = list(range(min(args.num_cols, len(prompt_ids))))
    label_globs = tuple(args.label_glob) if args.label_glob else DEFAULT_LABEL_GLOBS
    label_cache = load_label_cache(label_globs)
    rows, logit_values, _logit_labels, feature_values, _feature_labels, input_labels, meta = collect_cells(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model_id,
        prompt=args.prompt,
        checkpoint=args.checkpoint,
        k=args.k,
        layers=layers,
        positions=positions,
        device=device,
        label_cache=label_cache,
        display_features_per_cell=args.display_features_per_cell,
        label_top_tokens=args.label_top_tokens,
        display_label_tokens=args.display_label_tokens,
        label_chunk_size=args.label_chunk_size,
    )
    out_dirs = [args.out_dir]
    if args.paper_dir is not None:
        out_dirs.append(args.paper_dir)
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "qwen2b_32x_dickens_all_layer_cells.csv", rows)
        manifest = {
            "description": "Qwen3.5-2B logit-lens trajectory with current paper Sparse Readout Prism SAE feature terms.",
            "model_id": args.model_id,
            "checkpoint": str(args.checkpoint),
            "k": int(args.k),
            "prompt": args.prompt,
            "layers": layers,
            "positions": positions,
            "input_tokens": input_labels,
            "feature_display": "Top positive current-SAE readout features contributing to the centered logit-lens top-token score.",
            "color_top_panel": "Per-panel min-max normalized logit-lens top-token probability.",
            "color_feature_panel": "Per-panel min-max normalized share of positive sparse feature contribution shown in the cell.",
            "feature_description_file": "qwen2b_32x_dickens_feature_descriptions.csv",
            "feature_key_tex_file": "qwen2b_32x_dickens_feature_key.tex",
            "label_top_tokens": int(args.label_top_tokens),
            "display_label_tokens": int(args.display_label_tokens),
            "meta": meta,
        }
        (out_dir / "qwen2b_32x_dickens_all_layer_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "manifest": manifest,
                "cells": rows,
                "logit_values": logit_values,
                "feature_values": feature_values,
            },
            out_dir / "qwen2b_32x_dickens_all_layer_cache.pt",
        )
        write_feature_descriptions(out_dir, rows)
    print(f"Wrote metrics to {args.out_dir / 'qwen2b_32x_dickens_all_layer_cells.csv'}")


if __name__ == "__main__":
    main()
