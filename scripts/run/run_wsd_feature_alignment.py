#!/usr/bin/env python3
"""Frozen sense-labelled readout representations for Sparse Readout Prism.

Scores an ambiguous target word's selected logit at the final position of a
cloze prompt, decomposes it through a trained readout dictionary (top-k codes
of the centred, unit-normalised unembedding row; contribution_i = beta_i *
p_i(h)), and freezes the per-context representation bundle
(``representations.pt``: decoded state, projections, contributions, row
coefficients, feature ids, exact and reconstructed logits, target rank) that
the two sense analyses consume:

* ``scripts/analyze/analyze_wsd_sense_groups.py`` -- per-sense feature groups,
  permutation null, majority and no-selection references
  (``tab:app-sense-alignment``, appendix ``app:sense-labelled-evaluation``).
* ``scripts/analyze/analyze_wsd_classifier_framing.py`` -- truncated-display
  centroid classifier over signed contributions, unweighted projections,
  row-coefficient shuffles and random features (the "classifier framing"
  paragraph of the same appendix).

Its own outputs back that appendix's data and coverage statements:
``audit.json`` (contexts kept and single-token coverage per vocabulary: 20/20
words on the Qwen3.5 vocabularies, 13/20 on DeepSeek-R1-Distill-Llama-8B),
``scoring_summary.json`` (contexts scored, median target rank, row-gate
coverage) and ``metrics.json`` (full-vector centroid references computed on
the same bundle, with paired per-word bootstrap differences).

Datasets
--------
* CoarseWSD-20 (Loureiro, Rezaee, Pilehvar and Camacho-Collados, 2021,
  Computational Linguistics 47(2), doi:10.1162/coli_a_00405) -- the dataset
  used in the paper. ``--data-root`` is a checkout of
  https://github.com/danlou/bert-disambiguation; the loader looks for
  ``<root>/data/CoarseWSD-20/<word>/{train,test}.{data,gold}.txt`` plus
  ``class_map.txt`` (``<root>`` or ``<root>/CoarseWSD-20`` also work). The
  shipped train/test split is kept. The target occurrence is replaced by
  ``[BLANK]`` and the sentence is wrapped in ``COARSEWSD_TEMPLATE``; the
  target's leading-space token is scored at the prompt's final position, so
  it is a genuine next-token readout even when the disambiguating evidence
  originally followed the target.
* AmbiStory (SemEval-2026 Task 5): graded sense plausibility against
  gloss-prompt anchors. Served by the same scoring path; not used in the paper.

Paper runs (one per model, ``--seed 0``; the checkpoints are the 32x / k=256
selected dictionaries for Qwen3.5-2B, Qwen3.5-9B and
DeepSeek-R1-Distill-Llama-8B; ``--batch-size 8`` for the 2B model and ``4``
for the other two):

    python scripts/run/run_wsd_feature_alignment.py \
        --dataset coarsewsd20 --data-root <bert-disambiguation checkout> \
        --model-id Qwen/Qwen3.5-2B --checkpoint <qwen2b 32x/k256 checkpoint.pt> \
        --out-dir results/wsd_core/coarsewsd20_qwen2b \
        --splits train,test --batch-size 8 --n-boot 5000 --seed 0

``--analyze-only --out-dir <dir> --n-boot 5000 --seed 0`` recomputes
``metrics.json`` from a frozen bundle without model inference.

Gold labels are never used to fit the dictionary or to select features.
``--position-mode occurrence`` (score the sentence prefix so the target
occurrence itself is the next token) and ``--anchors`` (a frozen sense-anchor
token set, shipped as ``data/wsd/wsd_sense_anchors.json``, whose codes are
embedded in the bundle for an offline feature-typing analysis not included in
this repository) are variants the paper does not report; the paper's runs use
the cloze position and pass no anchors.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import find_lm_head, load_causal_lm, set_seed


AMBISTORY_STORY_TEMPLATE = """Read the story and recover the missing word.

{story}

The missing word is:"""

AMBISTORY_GLOSS_TEMPLATE = """Read the meaning and name the matching word.

Meaning: {gloss}

The matching word is:"""

COARSEWSD_TEMPLATE = """Read the sentence and recover the missing word.

{sentence}

The missing word is:"""


def log(message: str) -> None:
    print(f"[wsd {time.strftime('%H:%M:%S')}] {message}", flush=True)


def stable_id(*parts: str, n: int = 16) -> str:
    text = "\x1f".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mask_exact_target(text: str, target: str) -> str | None:
    """Replace the first case-insensitive whole target occurrence."""
    pattern = re.compile(rf"(?<!\w){re.escape(target)}(?!\w)", re.IGNORECASE)
    masked, count = pattern.subn("[BLANK]", text, count=1)
    return masked if count == 1 else None


def make_story(precontext: str, sentence: str, ending: str, target: str) -> str | None:
    masked = mask_exact_target(sentence.strip(), target)
    if masked is None:
        return None
    pieces = [precontext.strip(), masked]
    if ending.strip():
        pieces.append(ending.strip())
    return "\n".join(p for p in pieces if p)


def load_ambistory(root: Path, splits: list[str], limit: int | None) -> tuple[list[dict], dict]:
    records: list[dict] = []
    anchors: dict[tuple[str, str], dict] = {}
    skipped = defaultdict(int)
    split_counts: dict[str, int] = {}

    for split in splits:
        path = root / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text())
        rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
        grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            key = (
                str(row.get("homonym", "")).strip(),
                str(row.get("precontext", "")).strip(),
                str(row.get("sentence", "")).strip(),
                str(row.get("ending", "")).strip(),
            )
            grouped[key].append(row)

        made = 0
        for (target, precontext, sentence, ending), sense_rows in sorted(grouped.items()):
            if len(sense_rows) != 2:
                skipped["not_two_senses"] += 1
                continue
            story = make_story(precontext, sentence, ending, target)
            if story is None:
                skipped["target_not_found"] += 1
                continue
            ordered = sorted(sense_rows, key=lambda r: str(r.get("judged_meaning", "")))
            glosses = [str(r.get("judged_meaning", "")).strip() for r in ordered]
            if not all(glosses) or glosses[0] == glosses[1]:
                skipped["bad_gloss_pair"] += 1
                continue
            averages = [float(r["average"]) for r in ordered]
            stdevs = [float(r.get("stdev", 0.0)) for r in ordered]
            setup_id = stable_id(target, precontext, sentence)
            context_id = stable_id(target, precontext, sentence, ending)
            anchor_keys = []
            for gloss in glosses:
                akey = stable_id(target, gloss)
                anchor_keys.append(akey)
                anchors[(target, gloss)] = {
                    "item_id": f"anchor:{akey}",
                    "kind": "anchor",
                    "dataset": "ambistory",
                    "split": split,
                    "target": target,
                    "gloss": gloss,
                    "anchor_key": akey,
                    "prompt": AMBISTORY_GLOSS_TEMPLATE.format(gloss=gloss),
                }
            records.append(
                {
                    "item_id": f"story:{split}:{context_id}",
                    "kind": "story",
                    "dataset": "ambistory",
                    "split": split,
                    "target": target,
                    "setup_id": setup_id,
                    "context_id": context_id,
                    "has_ending": bool(ending.strip()),
                    "glosses": glosses,
                    "anchor_keys": anchor_keys,
                    "ratings": averages,
                    "rating_stdevs": stdevs,
                    "human_margin": averages[0] - averages[1],
                    "prompt": AMBISTORY_STORY_TEMPLATE.format(story=story),
                }
            )
            made += 1
            if limit is not None and len(records) >= limit:
                break
        split_counts[split] = made
        if limit is not None and len(records) >= limit:
            break

    # Only score anchors actually referenced by retained story records.
    used = {key for row in records for key in row["anchor_keys"]}
    anchor_rows = [row for row in anchors.values() if row["anchor_key"] in used]
    items = anchor_rows + records
    audit = {
        "dataset": "ambistory",
        "splits": splits,
        "n_story_contexts": len(records),
        "n_anchors": len(anchor_rows),
        "n_setups": len({r["setup_id"] for r in records}),
        "n_targets": len({r["target"] for r in records}),
        "split_story_counts": split_counts,
        "skipped": dict(skipped),
    }
    return items, audit


def parse_class_map(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    try:
        value = ast.literal_eval(text)
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
    except (ValueError, SyntaxError):
        pass
    mapping: dict[str, Any] = {}
    for line in text.splitlines():
        fields = line.strip().split("\t")
        if len(fields) >= 2:
            mapping[fields[0]] = fields[1]
    return mapping


def locate_coarse_root(root: Path) -> Path:
    candidates = [root, root / "data" / "CoarseWSD-20", root / "CoarseWSD-20"]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*/train.data.txt")):
            return candidate
    raise FileNotFoundError(f"could not find CoarseWSD-20 below {root}")


def resolve_target_position(tokens: list[str], raw_pos: int, word: str) -> int | None:
    candidates = [raw_pos, raw_pos - 1]
    for pos in candidates:
        if 0 <= pos < len(tokens) and tokens[pos].lower() == word.lower():
            return pos
    exact = [i for i, token in enumerate(tokens) if token.lower() == word.lower()]
    return exact[0] if len(exact) == 1 else None


def load_coarsewsd(
    root: Path,
    splits: list[str],
    limit: int | None,
    limit_per_word: int | None,
    position_mode: str = "cloze",
) -> tuple[list[dict], dict]:
    data_root = locate_coarse_root(root)
    records: list[dict] = []
    skipped = defaultdict(int)
    per_word_split: dict[str, dict[str, int]] = defaultdict(dict)

    for word_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        word = word_dir.name
        cmap_path = word_dir / "class_map.txt"
        class_map = parse_class_map(cmap_path) if cmap_path.exists() else {}
        for split in splits:
            data_path = word_dir / f"{split}.data.txt"
            gold_path = word_dir / f"{split}.gold.txt"
            if not data_path.exists() or not gold_path.exists():
                continue
            lines = data_path.read_text().splitlines()
            labels = gold_path.read_text().splitlines()
            if len(lines) != len(labels):
                raise ValueError(f"length mismatch: {data_path} and {gold_path}")
            candidates: list[dict] = []
            for local_idx, (line, label) in enumerate(zip(lines, labels)):
                fields = line.strip().split()
                if len(fields) < 3:
                    skipped["short_line"] += 1
                    continue
                try:
                    raw_pos = int(fields[0])
                except ValueError:
                    skipped["bad_position"] += 1
                    continue
                tokens = fields[1:]
                pos = resolve_target_position(tokens, raw_pos, word)
                if pos is None:
                    skipped["target_position_unresolved"] += 1
                    continue
                if position_mode == "occurrence":
                    # Score the real occurrence: the prompt is the sentence
                    # prefix, so the target's leading-space token is the
                    # genuine next token at the scored position.
                    if pos < 1:
                        skipped["no_left_context"] += 1
                        continue
                    prompt = " ".join(tokens[:pos])
                else:
                    tokens[pos] = "[BLANK]"
                    sentence = " ".join(tokens)
                    prompt = COARSEWSD_TEMPLATE.format(sentence=sentence)
                sense = label.strip()
                candidates.append(
                    {
                        "item_id": f"coarse:{word}:{split}:{local_idx}",
                        "kind": "context",
                        "dataset": "coarsewsd20",
                        "split": split,
                        "target": word,
                        "sense": sense,
                        "sense_name": class_map.get(sense, sense),
                        "prefix_words": pos,
                        "prompt": prompt,
                    }
                )
            # A prefix cap can contain only the majority sense because the
            # released files are sense-grouped. For pilots, round-robin across
            # senses while keeping the original order within each sense.
            if limit_per_word is not None and len(candidates) > limit_per_word:
                by_sense: dict[str, list[dict]] = defaultdict(list)
                for candidate in candidates:
                    by_sense[str(candidate["sense"])].append(candidate)
                balanced: list[dict] = []
                for offset in range(max(len(v) for v in by_sense.values())):
                    for sense in sorted(by_sense):
                        if offset < len(by_sense[sense]):
                            balanced.append(by_sense[sense][offset])
                            if len(balanced) >= limit_per_word:
                                break
                    if len(balanced) >= limit_per_word:
                        break
                candidates = balanced
            if limit is not None:
                candidates = candidates[: max(0, limit - len(records))]
            records.extend(candidates)
            made = len(candidates)
            per_word_split[word][split] = made
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break

    audit = {
        "dataset": "coarsewsd20",
        "data_root": str(data_root),
        "splits": splits,
        "n_contexts": len(records),
        "n_targets": len({r["target"] for r in records}),
        "per_word_split": dict(per_word_split),
        "skipped": dict(skipped),
    }
    return records, audit


def load_sae(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "W_dec" in ckpt:
        w_dec = ckpt["W_dec"].float()
        w_enc = ckpt["W_enc"].float()
        b_enc = ckpt.get("b_enc")
        k = int(ckpt.get("k") or ckpt.get("config", {}).get("k") or 256)
    else:
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer", {})
        if state is None:
            raise KeyError(f"unrecognized SAE checkpoint: {path}")
        w_dec = state["decoder"].float()
        w_enc = state["encoder.weight"].float().T.contiguous()
        b_enc = state.get("encoder.bias")
        k = int(cfg.get("k", 256))
    if w_enc.shape == w_dec.shape:
        w_enc = w_enc.T.contiguous()
    if w_enc.shape != (w_dec.shape[1], w_dec.shape[0]):
        raise ValueError(f"encoder {tuple(w_enc.shape)} incompatible with decoder {tuple(w_dec.shape)}")
    if b_enc is None:
        b_enc = torch.zeros(w_dec.shape[0])
    return w_dec, w_enc, b_enc.float(), k


def load_model(model_id: str, device: str, dtype: str):
    """Frozen causal LM + tokenizer; left padding keeps the scored position last in the batch."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model, tokenizer = load_causal_lm(model_id, dtype=torch_dtype, device_map=None)
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def resolve_single_token(tokenizer, target: str) -> tuple[int, str] | None:
    # The prompt ends in ':', so the natural continuation includes a leading
    # space. Requiring this exact continuation avoids silently scoring a token
    # form that generation would not use at the prompt boundary.
    candidate = " " + target
    ids = tokenizer.encode(candidate, add_special_tokens=False)
    return (int(ids[0]), candidate) if len(ids) == 1 else None


def prepare_target_codes(
    targets: list[str],
    tokenizer,
    w_u: torch.Tensor,
    row_mean: torch.Tensor,
    w_enc: torch.Tensor,
    b_enc: torch.Tensor,
    w_dec_unit: torch.Tensor,
    k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for target in sorted(set(targets)):
        resolved = resolve_single_token(tokenizer, target)
        if resolved is None:
            skipped[target] = "multi_token"
            continue
        token_id, continuation = resolved
        row = w_u[token_id] - row_mean
        row_norm = row.norm().clamp_min(1e-8)
        acts = torch.relu((row / row_norm) @ w_enc + b_enc)
        values, indices = torch.topk(acts, k=min(k, acts.numel()))
        beta = row_norm * values
        recon = beta @ w_dec_unit[indices]
        row_error = float((row - recon).norm() / row_norm)
        row_cos = float(torch.nn.functional.cosine_similarity(row[None], recon[None]).item())
        info[target] = {
            "token_id": token_id,
            "continuation": continuation,
            "feature_ids": indices.detach(),
            "beta": beta.detach(),
            "row_relative_error": row_error,
            "row_cosine": row_cos,
            "n_positive_codes": int((values > 0).sum().item()),
        }
    audit = {
        "n_targets_requested": len(set(targets)),
        "n_targets_single_token": len(info),
        "single_token_fraction": len(info) / max(len(set(targets)), 1),
        "skipped_targets": skipped,
        "targets": {
            target: {key: value for key, value in target_info.items() if key not in {"feature_ids", "beta"}}
            for target, target_info in info.items()
        },
    }
    return info, audit


def score_items(
    items: list[dict],
    model,
    tokenizer,
    lm_head,
    w_u: torch.Tensor,
    row_mean: torch.Tensor,
    w_dec_unit: torch.Tensor,
    target_info: dict[str, dict[str, Any]],
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    kept = [item for item in items if item["target"] in target_info]
    metadata: list[dict] = []
    hidden_rows: list[torch.Tensor] = []
    projection_rows: list[torch.Tensor] = []
    contribution_rows: list[torch.Tensor] = []
    beta_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    exact_logits: list[float] = []
    reconstructed_logits: list[float] = []
    target_logprobs: list[float] = []
    target_ranks: list[int] = []
    grabbed: list[torch.Tensor] = []

    def pre_hook(_module, inputs):
        grabbed.append(inputs[0].detach())

    hook = lm_head.register_forward_pre_hook(pre_hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(kept), batch_size):
                batch = kept[start : start + batch_size]
                enc = tokenizer(
                    [row["prompt"] for row in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                grabbed.clear()
                output = model(**enc, use_cache=False)
                if not grabbed:
                    raise RuntimeError("LM-head hook did not capture a hidden state")
                hidden = grabbed[-1][:, -1, :].float()
                logits = output.logits[:, -1, :].float()
                for offset, row in enumerate(batch):
                    tinfo = target_info[row["target"]]
                    feature_ids = tinfo["feature_ids"]
                    beta = tinfo["beta"]
                    projection = w_dec_unit[feature_ids] @ hidden[offset]
                    contribution = beta * projection
                    token_id = int(tinfo["token_id"])
                    exact = float(logits[offset, token_id])
                    reconstructed = float(hidden[offset] @ row_mean + contribution.sum())
                    logprob = float(logits[offset, token_id] - torch.logsumexp(logits[offset], dim=0))
                    rank = int((logits[offset] > logits[offset, token_id]).sum().item() + 1)
                    meta = {key: value for key, value in row.items() if key != "prompt"}
                    meta.update(
                        {
                            "prompt_sha1": stable_id(row["prompt"], n=20),
                            "token_id": token_id,
                            "continuation": tinfo["continuation"],
                            "row_relative_error": tinfo["row_relative_error"],
                            "row_cosine": tinfo["row_cosine"],
                        }
                    )
                    metadata.append(meta)
                    hidden_rows.append(hidden[offset].half().cpu())
                    projection_rows.append(projection.half().cpu())
                    contribution_rows.append(contribution.half().cpu())
                    beta_rows.append(beta.half().cpu())
                    feature_rows.append(feature_ids.int().cpu())
                    exact_logits.append(exact)
                    reconstructed_logits.append(reconstructed)
                    target_logprobs.append(logprob)
                    target_ranks.append(rank)
                del output, logits, hidden
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(kept):
                    log(f"scored {min(start + batch_size, len(kept))}/{len(kept)}")
    finally:
        hook.remove()

    if not metadata:
        raise RuntimeError("no tokenizable items were scored")
    return {
        "metadata": metadata,
        "hidden": torch.stack(hidden_rows),
        "projection": torch.stack(projection_rows),
        "contribution": torch.stack(contribution_rows),
        "beta": torch.stack(beta_rows),
        "feature_ids": torch.stack(feature_rows),
        "exact_logit": torch.tensor(exact_logits, dtype=torch.float32),
        "reconstructed_logit": torch.tensor(reconstructed_logits, dtype=torch.float32),
        "target_logprob": torch.tensor(target_logprobs, dtype=torch.float32),
        "target_rank": torch.tensor(target_ranks, dtype=torch.int32),
    }


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def method_matrices(bundle: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    hidden = bundle["hidden"].float().numpy()
    projection = bundle["projection"].float().numpy()
    contribution = bundle["contribution"].float().numpy()
    beta = bundle["beta"].float().numpy()
    metadata = bundle["metadata"]
    shuffled = np.empty_like(contribution)
    for target in sorted({row["target"] for row in metadata}):
        idx = [i for i, row in enumerate(metadata) if row["target"] == target]
        local_seed = seed + int(stable_id(target, n=8), 16)
        rng = np.random.default_rng(local_seed)
        permutation = rng.permutation(beta.shape[1])
        shuffled[idx] = projection[idx] * beta[idx][:, permutation]
    return {
        "srp": l2_normalize(contribution),
        "support_projection": l2_normalize(projection),
        "hidden": l2_normalize(hidden),
        "shuffled_srp": l2_normalize(shuffled),
        "token_only": l2_normalize(beta),
    }


def safe_spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) < 1e-8 or np.ptp(y) < 1e-8:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    cluster_key: str,
    stat_fn,
    n_boot: int,
    seed: int,
) -> list[float]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row[cluster_key])].append(row)
    keys = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sample: list[dict[str, Any]] = []
        for selected in rng.choice(keys, size=len(keys), replace=True):
            sample.extend(by_cluster[str(selected)])
        value = float(stat_fn(sample))
        if np.isfinite(value):
            values.append(value)
    return values


def ci(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyze_ambistory(bundle: dict[str, Any], out_dir: Path, n_boot: int, seed: int) -> dict:
    metadata = bundle["metadata"]
    matrices = method_matrices(bundle, seed)
    anchor_index = {(row["target"], row["anchor_key"]): i for i, row in enumerate(metadata) if row["kind"] == "anchor"}
    story_indices = [i for i, row in enumerate(metadata) if row["kind"] == "story"]
    predictions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"dataset": "ambistory", "methods": {}, "comparisons": {}}
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    shifts_by_method: dict[str, list[dict[str, Any]]] = {}

    for method, matrix in matrices.items():
        rows: list[dict[str, Any]] = []
        for i in story_indices:
            meta = metadata[i]
            anchor_ids = [anchor_index.get((meta["target"], key)) for key in meta["anchor_keys"]]
            if any(index is None for index in anchor_ids):
                continue
            similarities = [float(matrix[i] @ matrix[int(index)]) for index in anchor_ids]
            pred_margin = similarities[0] - similarities[1]
            human_margin = float(meta["human_margin"])
            row = {
                "method": method,
                "item_id": meta["item_id"],
                "setup_id": meta["setup_id"],
                "context_id": meta["context_id"],
                "split": meta["split"],
                "target": meta["target"],
                "has_ending": bool(meta["has_ending"]),
                "human_margin": human_margin,
                "pred_margin": pred_margin,
                "similarity_0": similarities[0],
                "similarity_1": similarities[1],
                "correct": int(human_margin != 0 and np.sign(pred_margin) == np.sign(human_margin)),
                "eligible_gap1": int(abs(human_margin) >= 1.0),
                "row_relative_error": float(meta["row_relative_error"]),
            }
            rows.append(row)
            if method == "srp":
                predictions.append(row)

        def rho_fn(sample):
            return safe_spearman([r["human_margin"] for r in sample], [r["pred_margin"] for r in sample])

        eligible = [r for r in rows if r["eligible_gap1"]]
        non_ties = [r for r in rows if r["human_margin"] != 0]
        rho = rho_fn(rows)
        rho_boot = cluster_bootstrap(rows, "setup_id", rho_fn, n_boot, seed)
        accuracy_gap1 = float(np.mean([r["correct"] for r in eligible])) if eligible else float("nan")
        acc_boot = (
            cluster_bootstrap(
                eligible,
                "setup_id",
                lambda sample: np.mean([r["correct"] for r in sample]),
                n_boot,
                seed + 1,
            )
            if eligible
            else []
        )

        # Controlled context change: representation distance should grow with
        # the change in human sense-preference margin within the same setup.
        shift_rows: list[dict[str, Any]] = []
        by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_item = {r["item_id"]: r for r in rows}
        for i in story_indices:
            meta = metadata[i]
            if meta["item_id"] in by_item:
                by_setup[meta["setup_id"]].append({"i": i, **by_item[meta["item_id"]]})
        reversal_total = reversal_correct = 0
        for setup_id, setup_rows in by_setup.items():
            for left, right in itertools.combinations(setup_rows, 2):
                distance = float(1.0 - matrix[left["i"]] @ matrix[right["i"]])
                human_shift = abs(left["human_margin"] - right["human_margin"])
                shift_rows.append({"setup_id": setup_id, "distance": distance, "human_shift": human_shift})
                if left["human_margin"] * right["human_margin"] < 0:
                    reversal_total += 1
                    if left["pred_margin"] * right["pred_margin"] < 0:
                        reversal_correct += 1
        shift_rho = safe_spearman([r["human_shift"] for r in shift_rows], [r["distance"] for r in shift_rows])
        shift_boot = (
            cluster_bootstrap(
                shift_rows,
                "setup_id",
                lambda sample: safe_spearman([r["human_shift"] for r in sample], [r["distance"] for r in sample]),
                n_boot,
                seed + 2,
            )
            if shift_rows
            else []
        )
        metrics["methods"][method] = {
            "n_contexts": len(rows),
            "n_setups": len({r["setup_id"] for r in rows}),
            "preference_spearman": rho,
            "preference_spearman_ci": ci(rho_boot),
            "preferred_sense_accuracy_gap1": accuracy_gap1,
            "preferred_sense_accuracy_gap1_ci": ci(acc_boot),
            "n_gap1": len(eligible),
            "preferred_sense_accuracy_non_tie": (
                float(np.mean([r["correct"] for r in non_ties])) if non_ties else float("nan")
            ),
            "profile_shift_spearman": shift_rho,
            "profile_shift_spearman_ci": ci(shift_boot),
            "n_shift_pairs": len(shift_rows),
            "sense_reversal_consistency": (reversal_correct / reversal_total if reversal_total else float("nan")),
            "n_reversal_pairs": reversal_total,
        }
        rows_by_method[method] = rows
        shifts_by_method[method] = shift_rows

    # Confidence intervals for the scientific comparisons must be paired: all
    # methods score the same stories, so resample setups once and take the
    # within-resample difference. Independent per-method intervals do not test
    # whether SRP itself differs from a control.
    for comparison_index, baseline in enumerate(("shuffled_srp", "support_projection", "hidden")):
        srp_rows = rows_by_method["srp"]
        baseline_rows = rows_by_method[baseline]
        if [row["item_id"] for row in srp_rows] != [row["item_id"] for row in baseline_rows]:
            raise ValueError(f"unaligned AmbiStory rows for srp and {baseline}")
        paired_rows = [
            {
                "setup_id": left["setup_id"],
                "human_margin": left["human_margin"],
                "eligible_gap1": left["eligible_gap1"],
                "srp_pred_margin": left["pred_margin"],
                "baseline_pred_margin": right["pred_margin"],
                "srp_correct": left["correct"],
                "baseline_correct": right["correct"],
            }
            for left, right in zip(srp_rows, baseline_rows)
        ]

        def preference_delta(sample):
            human = [row["human_margin"] for row in sample]
            return safe_spearman(human, [row["srp_pred_margin"] for row in sample]) - safe_spearman(
                human, [row["baseline_pred_margin"] for row in sample]
            )

        eligible_paired = [row for row in paired_rows if row["eligible_gap1"]]

        def accuracy_delta(sample):
            return np.mean([row["srp_correct"] for row in sample]) - np.mean(
                [row["baseline_correct"] for row in sample]
            )

        srp_shifts = shifts_by_method["srp"]
        baseline_shifts = shifts_by_method[baseline]
        if [row["setup_id"] for row in srp_shifts] != [row["setup_id"] for row in baseline_shifts] or not np.allclose(
            [row["human_shift"] for row in srp_shifts],
            [row["human_shift"] for row in baseline_shifts],
        ):
            raise ValueError(f"unaligned AmbiStory shift rows for srp and {baseline}")
        paired_shifts = [
            {
                "setup_id": left["setup_id"],
                "human_shift": left["human_shift"],
                "srp_distance": left["distance"],
                "baseline_distance": right["distance"],
            }
            for left, right in zip(srp_shifts, baseline_shifts)
        ]

        def shift_delta(sample):
            human = [row["human_shift"] for row in sample]
            return safe_spearman(human, [row["srp_distance"] for row in sample]) - safe_spearman(
                human, [row["baseline_distance"] for row in sample]
            )

        comparison_seed = seed + 100 + comparison_index * 10
        metrics["comparisons"][f"srp_minus_{baseline}"] = {
            "preference_spearman_difference": float(preference_delta(paired_rows)),
            "preference_spearman_difference_ci": ci(
                cluster_bootstrap(paired_rows, "setup_id", preference_delta, n_boot, comparison_seed)
            ),
            "preferred_sense_accuracy_gap1_difference": float(accuracy_delta(eligible_paired)),
            "preferred_sense_accuracy_gap1_difference_ci": ci(
                cluster_bootstrap(
                    eligible_paired,
                    "setup_id",
                    accuracy_delta,
                    n_boot,
                    comparison_seed + 1,
                )
            ),
            "profile_shift_spearman_difference": float(shift_delta(paired_shifts)),
            "profile_shift_spearman_difference_ci": ci(
                cluster_bootstrap(
                    paired_shifts,
                    "setup_id",
                    shift_delta,
                    n_boot,
                    comparison_seed + 2,
                )
            ),
        }
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    return metrics


def sampled_pair_auc(matrix: np.ndarray, labels: np.ndarray, seed: int, max_pairs: int = 20000) -> float:
    from sklearn.metrics import roc_auc_score

    n = len(labels)
    if n < 2:
        return float("nan")
    total = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if total <= max_pairs:
        pairs = list(itertools.combinations(range(n), 2))
    else:
        chosen: set[tuple[int, int]] = set()
        while len(chosen) < max_pairs:
            left = int(rng.integers(0, n))
            right = int(rng.integers(0, n - 1))
            if right >= left:
                right += 1
            chosen.add((min(left, right), max(left, right)))
        pairs = list(chosen)
    gold = np.array([int(labels[i] == labels[j]) for i, j in pairs])
    if len(np.unique(gold)) < 2:
        return float("nan")
    scores = np.array([float(matrix[i] @ matrix[j]) for i, j in pairs])
    return float(roc_auc_score(gold, scores))


def fit_train_clusters(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """Fit unsupervised clusters on train only and assign held-out examples."""
    from sklearn.cluster import KMeans

    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if len(train_matrix) < n_clusters:
        raise ValueError(f"need at least {n_clusters} train rows, received {len(train_matrix)}")
    if np.max(np.ptp(train_matrix, axis=0)) < 1e-8:
        return np.zeros(len(test_matrix), dtype=int)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    model.fit(train_matrix)
    return model.predict(test_matrix)


def analyze_coarsewsd(bundle: dict[str, Any], out_dir: Path, n_boot: int, seed: int) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        adjusted_rand_score,
        balanced_accuracy_score,
        f1_score,
        normalized_mutual_info_score,
    )

    metadata = bundle["metadata"]
    matrices = method_matrices(bundle, seed)
    words = sorted({row["target"] for row in metadata})
    metrics: dict[str, Any] = {"dataset": "coarsewsd20", "methods": {}, "comparisons": {}}
    srp_predictions: list[dict[str, Any]] = []
    per_word_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for method, matrix in matrices.items():
        per_word: dict[str, dict[str, float]] = {}
        for word in words:
            train_idx = [i for i, row in enumerate(metadata) if row["target"] == word and row["split"] == "train"]
            test_idx = [i for i, row in enumerate(metadata) if row["target"] == word and row["split"] == "test"]
            if not train_idx or not test_idx:
                continue
            train_y = np.array([metadata[i]["sense"] for i in train_idx])
            test_y = np.array([metadata[i]["sense"] for i in test_idx])
            senses = sorted(set(train_y))
            if len(senses) < 2 or not set(test_y).issubset(set(senses)):
                continue
            centroids = []
            for sense in senses:
                centroid = matrix[np.array(train_idx)[train_y == sense]].mean(axis=0)
                centroid /= max(np.linalg.norm(centroid), 1e-12)
                centroids.append(centroid)
            centroid_matrix = np.stack(centroids)
            scores = matrix[test_idx] @ centroid_matrix.T
            predictions = np.array([senses[i] for i in scores.argmax(axis=1)])
            n_clusters = len(senses)
            train_matrix = matrix[train_idx]
            test_matrix = matrix[test_idx]
            cluster_pred = fit_train_clusters(
                train_matrix,
                test_matrix,
                n_clusters=n_clusters,
                seed=seed,
            )
            word_metrics = {
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_senses": len(senses),
                "accuracy": float(accuracy_score(test_y, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
                "macro_f1": float(f1_score(test_y, predictions, average="macro")),
                "pairwise_auc": sampled_pair_auc(matrix[test_idx], test_y, seed),
                "ari": float(adjusted_rand_score(test_y, cluster_pred)),
                "nmi": float(normalized_mutual_info_score(test_y, cluster_pred)),
                "row_relative_error": float(metadata[test_idx[0]]["row_relative_error"]),
            }
            per_word[word] = word_metrics
            if method == "srp":
                for local, idx in enumerate(test_idx):
                    srp_predictions.append(
                        {
                            "item_id": metadata[idx]["item_id"],
                            "target": word,
                            "gold_sense": str(test_y[local]),
                            "predicted_sense": str(predictions[local]),
                            "correct": int(test_y[local] == predictions[local]),
                            "target_rank": int(bundle["target_rank"][idx]),
                            "target_logprob": float(bundle["target_logprob"][idx]),
                        }
                    )
        metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "pairwise_auc", "ari", "nmi"]
        aggregate: dict[str, Any] = {"n_words": len(per_word)}
        rng = np.random.default_rng(seed)
        word_names = sorted(per_word)
        for metric_name in metric_names:
            values = np.array([per_word[word][metric_name] for word in word_names], dtype=float)
            values = values[np.isfinite(values)]
            aggregate[metric_name] = float(values.mean()) if len(values) else float("nan")
            boot = []
            if len(values):
                for _ in range(n_boot):
                    boot.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            aggregate[f"{metric_name}_ci"] = ci(boot)
        gated_words = [w for w in word_names if per_word[w]["row_relative_error"] < 0.5]
        aggregate["row_gate_words"] = len(gated_words)
        aggregate["row_gate_fraction"] = len(gated_words) / max(len(word_names), 1)
        aggregate["row_gate"] = {
            metric_name: (
                float(np.mean([per_word[w][metric_name] for w in gated_words])) if gated_words else float("nan")
            )
            for metric_name in metric_names
        }
        metrics["methods"][method] = {"aggregate": aggregate, "per_word": per_word}
        per_word_by_method[method] = per_word

    # CoarseWSD words are the independent units. Bootstrap paired per-word
    # differences so the intervals answer whether SRP beats each control.
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "pairwise_auc", "ari", "nmi"]
    for comparison_index, baseline in enumerate(("shuffled_srp", "support_projection", "hidden")):
        common_words = sorted(set(per_word_by_method["srp"]) & set(per_word_by_method[baseline]))
        comparison: dict[str, Any] = {"n_words": len(common_words)}
        rng = np.random.default_rng(seed + 200 + comparison_index)
        for metric_name in metric_names:
            differences = np.array(
                [
                    per_word_by_method["srp"][word][metric_name] - per_word_by_method[baseline][word][metric_name]
                    for word in common_words
                ],
                dtype=float,
            )
            differences = differences[np.isfinite(differences)]
            comparison[f"{metric_name}_difference"] = float(differences.mean()) if len(differences) else float("nan")
            boot = (
                [float(rng.choice(differences, size=len(differences), replace=True).mean()) for _ in range(n_boot)]
                if len(differences)
                else []
            )
            comparison[f"{metric_name}_difference_ci"] = ci(boot)
        metrics["comparisons"][f"srp_minus_{baseline}"] = comparison
    write_jsonl(out_dir / "predictions.jsonl", srp_predictions)
    return metrics


def summarize_scoring(bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    exact = bundle["exact_logit"].numpy()
    recon = bundle["reconstructed_logit"].numpy()
    ranks = bundle["target_rank"].numpy()
    return {
        "n_scored": len(metadata),
        "n_targets": len({row["target"] for row in metadata}),
        "median_target_rank": float(np.median(ranks)),
        "target_top1_fraction": float(np.mean(ranks == 1)),
        "target_top10_fraction": float(np.mean(ranks <= 10)),
        "median_absolute_logit_residual": float(np.median(np.abs(exact - recon))),
        "mean_absolute_logit_residual": float(np.mean(np.abs(exact - recon))),
        "row_gate_fraction_items": float(np.mean([row["row_relative_error"] < 0.5 for row in metadata])),
    }


def self_test() -> int:
    assert mask_exact_target("They followed the track.", "track") == "They followed the [BLANK]."
    assert mask_exact_target("a tracker", "track") is None
    assert resolve_target_position(["the", "bank", "closed"], 1, "bank") == 1
    assert resolve_target_position(["the", "bank", "closed"], 2, "bank") == 1
    matrix = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]))
    auc = sampled_pair_auc(matrix, np.array([0, 0, 1, 1]), seed=0)
    assert auc > 0.99
    train = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [9.9, 10.0]])
    test = np.array([[0.05, 0.0], [9.95, 10.0]])
    cluster_pred = fit_train_clusters(train, test, n_clusters=2, seed=0)
    assert cluster_pred[0] != cluster_pred[1]
    assert safe_spearman([1, 2, 3], [2, 4, 6]) > 0.99
    print("self-test passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ambistory", "coarsewsd20"), help="the paper uses coarsewsd20")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="coarsewsd20: a bert-disambiguation checkout (or its data/CoarseWSD-20 dir); ambistory: dir of <split>.json",
    )
    parser.add_argument("--model-id", help="Hugging Face model id")
    parser.add_argument("--checkpoint", type=Path, help="trained readout dictionary checkpoint (.pt)")
    parser.add_argument("--out-dir", type=Path, help="output dir: representations.pt, audit/scoring/metrics/run_config")
    parser.add_argument(
        "--splits", default=None, help="comma-separated; defaults dev for AmbiStory, train,test for CoarseWSD"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-word", type=int, default=None)
    parser.add_argument(
        "--position-mode",
        choices=("cloze", "occurrence"),
        default="cloze",
        help="cloze: mask target + cue; occurrence: score the sentence "
        "prefix so the target occurrence is the genuine next token "
        "(coarsewsd20 only)",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=None,
        help="frozen sense-anchor JSON; anchor rows and sparse codes are "
        "embedded in the bundle for offline weights-only feature typing",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    required = (args.dataset, args.data_root, args.model_id, args.checkpoint, args.out_dir)
    if not args.analyze_only and not all(required):
        raise SystemExit("dataset, data-root, model-id, checkpoint, and out-dir are required")
    if args.analyze_only and args.out_dir is None:
        raise SystemExit("--out-dir is required with --analyze-only")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    representation_path = args.out_dir / "representations.pt"
    if args.analyze_only:
        bundle = torch.load(representation_path, map_location="cpu", weights_only=False)
    else:
        set_seed(args.seed)
        if args.splits:
            splits = [part.strip() for part in args.splits.split(",") if part.strip()]
        else:
            splits = ["dev"] if args.dataset == "ambistory" else ["train", "test"]
        if args.dataset == "ambistory":
            if args.position_mode != "cloze":
                raise SystemExit("--position-mode occurrence supports coarsewsd20 only")
            items, data_audit = load_ambistory(args.data_root, splits, args.limit)
        else:
            items, data_audit = load_coarsewsd(
                args.data_root, splits, args.limit, args.limit_per_word, args.position_mode
            )
        log(f"loaded {len(items)} items ({data_audit})")
        log(f"loading {args.model_id}")
        model, tokenizer = load_model(args.model_id, args.device, args.dtype)
        if args.position_mode == "occurrence":
            # Long prefixes must keep their tail (the context nearest the
            # scored position), so truncate from the left.
            tokenizer.truncation_side = "left"
        lm_head = find_lm_head(model)
        w_u = lm_head.weight.detach().float().to(args.device)
        row_mean = w_u.mean(dim=0)
        w_dec, w_enc, b_enc, k = load_sae(args.checkpoint)
        w_dec = w_dec.to(args.device)
        w_dec_unit = w_dec / w_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        w_enc = w_enc.to(args.device)
        b_enc = b_enc.to(args.device)
        if w_dec.shape[1] != w_u.shape[1]:
            raise ValueError(f"dictionary/model width mismatch: {w_dec.shape[1]} vs {w_u.shape[1]}")
        target_info, token_audit = prepare_target_codes(
            [row["target"] for row in items],
            tokenizer,
            w_u,
            row_mean,
            w_enc,
            b_enc,
            w_dec_unit,
            k,
        )
        anchor_bundle = None
        if args.anchors is not None:
            anchors_cfg = json.loads(args.anchors.read_text())
            anchor_words = sorted(
                {
                    anchor
                    for word, senses in anchors_cfg.items()
                    if not word.startswith("_")
                    for anchor_list in senses.values()
                    for anchor in anchor_list
                }
            )
            anchor_info, anchor_audit = prepare_target_codes(
                anchor_words,
                tokenizer,
                w_u,
                row_mean,
                w_enc,
                b_enc,
                w_dec_unit,
                k,
            )
            anchor_bundle = {
                "config": anchors_cfg,
                "codes": {
                    word: {
                        "token_id": info["token_id"],
                        "feature_ids": info["feature_ids"].cpu(),
                        "beta": info["beta"].cpu(),
                    }
                    for word, info in anchor_info.items()
                },
                "rows": {word: w_u[info["token_id"]].cpu().half() for word, info in anchor_info.items()},
                "audit": anchor_audit,
            }
            log(
                "anchor coverage "
                f"{anchor_audit['n_targets_single_token']}/"
                f"{anchor_audit['n_targets_requested']} single-token"
            )
        del w_enc, b_enc, w_dec
        audit = {**data_audit, "tokenization_and_rows": token_audit, "k": k}
        write_json(args.out_dir / "audit.json", audit)
        log(f"single-token coverage {token_audit['n_targets_single_token']}/{token_audit['n_targets_requested']}")
        bundle = score_items(
            items,
            model,
            tokenizer,
            lm_head,
            w_u,
            row_mean,
            w_dec_unit,
            target_info,
            args.batch_size,
            args.max_length,
            args.device,
        )
        bundle["run"] = {
            "dataset": args.dataset,
            "model_id": args.model_id,
            "checkpoint": str(args.checkpoint),
            "splits": splits,
            "seed": args.seed,
            "k": k,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "position_mode": args.position_mode,
        }
        if anchor_bundle is not None:
            bundle["anchors"] = anchor_bundle
        torch.save(bundle, representation_path)
        write_json(args.out_dir / "scoring_summary.json", summarize_scoring(bundle))

    metrics = (
        analyze_ambistory(bundle, args.out_dir, args.n_boot, args.seed)
        if bundle["run"]["dataset"] == "ambistory"
        else analyze_coarsewsd(bundle, args.out_dir, args.n_boot, args.seed)
    )
    metrics["run"] = bundle["run"]
    metrics["scoring"] = summarize_scoring(bundle)
    write_json(args.out_dir / "metrics.json", metrics)
    run_config = {
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "story_template": AMBISTORY_STORY_TEMPLATE,
        "gloss_template": AMBISTORY_GLOSS_TEMPLATE,
        "coarsewsd_template": COARSEWSD_TEMPLATE,
        "provenance": run_provenance(args),
    }
    write_json(args.out_dir / "run_config.json", run_config)
    log(f"wrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
