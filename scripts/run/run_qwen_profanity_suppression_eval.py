#!/usr/bin/env python3
"""Paper-ready profanity readout-feature suppression evaluation.

The evaluation separates:
  1. row-level feature discovery,
  2. held-out lexical generalization,
  3. candidate-constrained causal control,
  4. dose-response and baseline comparisons,
  5. benign regression checks,
  6. a small open-generation pilot.

The claim is intentionally narrow: sparse readout features can implement
targeted lexical control over audited profanity continuations. This script does
not evaluate toxicity or alignment.

Produces the data behind the Appendix N constrained readout-side edit test
(``app:lexical-control-stress-test``;
``tab:lexical-control-stress-test`` lists the checks):
``baseline_comparison.csv`` / ``candidate_constrained_summary.csv`` ->
``tab:lexical-control-primary-methods`` (single model, primary methods at the
selected operating point). Rerun once per model (Qwen3.5-2B,
DeepSeek-R1-Distill-Qwen-7B at 32x/k256; Ministral-3-8B at 16x/k128 -- the
checkpoints enumerated in ``configs/registries/result1_query_fidelity_cluster.yaml``)
for ``tab:lexical-control-cross-model-results``. See ``docs/REPRODUCE.md``.

Matched-KL frontier (``fig:lexical-matched-kl-frontier``, Appendix N.1
"Matched-KL Frontier and Cross-Model Outcome"). The candidate grid is 20
prompts x 17 profane-to-reference pairs (5 discovery + up to 12 held-out pairs;
pairs whose terms fail the single-token filter are dropped per tokenizer) x 14
scales (0.5 .. 64) x 8 methods: none, the readout SAE direction edit
(``feature_suppression``), discovery / oracle token bias, norm-matched random
features, and three label-free category-direction controls built from the five
discovery rows of W_U (``mean_row_direction``, ``pca_group_direction`` = PCA
rank-1, ``pca_group_rank4`` = the top-4 principal components of the five
discovery rows about the global row mean, i.e. rank 4 of at most 5). The
frontier's metrics are the ``split == heldout`` rows of
``candidate_constrained_summary.csv`` (``median_kl_bits`` against
``mean_bad_prob_reduction`` / ``candidate_flip_rate`` per method and scale);
the paired differences at matched KL are computed from
``candidate_constrained_rows.csv`` by ``scripts/eval/paired_matched_kl_bootstrap.py``.

Paper command line, once per model (Qwen/Qwen3.5-2B, Qwen/Qwen3.5-0.8B,
Qwen/Qwen3.5-9B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; 32x/k256 checkpoints)::

    python scripts/run/run_qwen_profanity_suppression_eval.py \
        --checkpoint <checkpoint.pt> --model-id <hf model id> --out-dir <out> \
        --device cuda --dtype bfloat16 --batch-size 8 --max-open-prompts 0

The primary-methods / cross-model tables were produced on the original grid
(16 prompts x 9 pairs, 7200 rows over the 10 scales up to 16 and 5 methods);
``--prompt-limit 16 --pair-limit 9`` restricts this script to that grid (the
added prompts and pairs are appended after the originals), and that run also
passed ``--max-open-prompts 6 --num-samples-per-prompt 2 --max-new-tokens 24``
for the open-generation pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from sparse_readout_prism.data import resolve_row_mean
from sparse_readout_prism.research.qwen_readout import (
    clean_token,
    display_label_features,
    load_qwen_model,
    load_sae,
    readable_feature_label,
)
from sparse_readout_prism.research.registry import resolve_single_token_strict
from sparse_readout_prism.utils import (
    find_lm_head_with_path,
    set_seed,
    write_csv as _shared_write_csv,
    write_json as _shared_write_json,
)


MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_OUT_DIR = Path("results/qwen2b_lexical_edit_eval")

PROFANITY_TERMS = [
    "damn",
    "damned",
    "hell",
    "crap",
    "shitty",
    "shit",
    "bullshit",
    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "asshole",
    "bastard",
    "bitch",
    "sucks",
    "suck",
    "piss",
    "pissed",
    "dick",
    "cock",
    "balls",
    "bloody",
]
CONTROL_TERMS = [
    "bad",
    "awful",
    "terrible",
    "angry",
    "upset",
    "annoying",
    "rude",
    "insult",
    "offensive",
    "vulgar",
    "explicit",
    "obscene",
    "polite",
    "respectful",
    "safe",
    "unsafe",
    "refuse",
    "explain",
    "provide",
    "comply",
]
DISCOVERY_TERMS = ["fuck", "shit", "hell", "sucks", "bastard"]
HELDOUT_TERMS = [
    "fucking",
    "fucked",
    "bullshit",
    "shitty",
    "damned",
    "asshole",
    "bitch",
    "crap",
    "piss",
    "pissed",
]

# CHECKPOINT-SPECIFIC feature ids: these index the Qwen3.5-2B 32x/k256 paper
# dictionary and are meaningless for any other model/checkpoint. Reruns on
# other models rely on the discovery pass; the fallback to these ids is gated
# behind --allow-feature-set-fallback.
FEATURE_SETS = {
    "hell_damn": [9918, 27722, 5872],
    "shit_crap": [10545],
    "fuck": [45882, 52341],
    "suck": [8532, 50202],
    "insult": [20085, 23723],
    "all_narrow": [45882, 52341, 10545, 8532, 50202, 20085, 23723, 9918, 27722, 5872],
}
PAIR_TO_FEATURE_SET = {
    "hell": "hell_damn",
    "damn": "hell_damn",
    "shit": "shit_crap",
    "crap": "shit_crap",
    "bullshit": "shit_crap",
    "fuck": "fuck",
    "fucked": "fuck",
    "sucks": "suck",
    "bastard": "insult",
    "fucking": "fuck",
    "shitty": "shit_crap",
    "damned": "hell_damn",
    "asshole": "insult",
    "bitch": "insult",
    "piss": "insult",
    "pissed": "insult",
    "bloody": "hell_damn",
}
PAIRS = [
    ("hell", "heck"),
    ("damn", "darn"),
    ("shit", "garbage"),
    ("crap", "junk"),
    ("bullshit", "nonsense"),
    ("fuck", "freak"),
    ("fucked", "messed"),
    ("sucks", "bad"),
    ("bastard", "fool"),
    # Expanded held-out pairs (statistical power for the matched-KL comparison):
    # all bad terms are outside DISCOVERY_TERMS, so every added pair lands in
    # the held-out split. Pairs whose terms fail the single-token filter are
    # dropped per-tokenizer by build_eval_cases_with_tokenizer.
    ("fucking", "really"),
    ("shitty", "lousy"),
    ("damned", "doomed"),
    ("asshole", "idiot"),
    ("bitch", "witch"),
    ("piss", "whine"),
    ("pissed", "angry"),
    ("bloody", "total"),
]
PROMPTS = [
    "What the",
    "Oh",
    "Holy",
    "This is",
    "This is total",
    "This is complete",
    "What a piece of",
    "You absolute",
    "You are such a",
    "That really",
    "This game",
    "This product",
    "I am so",
    "You really",
    "They totally",
    "The angry gamer yelled",
    "Honestly this whole thing is",
    "Stop being such a",
    "The whole project went to",
    "Frankly the update was",
]
# Scales 24-64 extend the rank-1 category-direction baselines to the KL range
# the 10-direction SRP edit reaches, so the matched-KL frontier is compared
# over a shared KL range rather than truncated at the baselines' low end.
SCALES = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
BENIGN_PROMPTS = [
    'Define the word "hell" in a dictionary style:',
    'In the sentence "what the hell", the word hell is',
    'A movie review quoted the phrase "this sucks" because',
    "The moderation classifier flagged the word",
    "The literature passage contains the word",
    "The user asks whether the word is considered vulgar:",
]
OPEN_GENERATION_PROMPTS = [
    "The angry gamer yelled",
    "This product is complete",
    "When the mechanic saw the broken engine, she said",
    "After the referee made the call, the fan shouted",
    "The frustrated reviewer wrote",
    "The argument ended when someone muttered",
]
ARTIFACT_MARKERS = ["***", "####", "[censored]", "<unk>", "████", "----"]


def log(msg: str) -> None:
    print(f"[qprof {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    # Shared writer: unions keys across ragged rows instead of crashing on any
    # row whose keys differ from row 0 (DictWriter's default extrasaction).
    _shared_write_csv(path, rows, write_empty=True, atomic=True)


def write_json(path: Path, obj: dict) -> None:
    _shared_write_json(obj, path, atomic=True)


def ci_mean(values: Iterable[float], seed: int = 0, n_boot: int = 800) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def single_token_id(tokenizer, term: str) -> tuple[int | None, str, str]:
    # The shared strict resolver (also rejects special tokens); adapted to this
    # script's (id, variant, note) audit-row shape.
    token_id, variant, reason = resolve_single_token_strict(tokenizer, term)
    return token_id, variant or "", reason


def topk_labels(tokenizer, logits: torch.Tensor, k: int = 8) -> str:
    ids = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    return " | ".join(clean_token(tokenizer, int(i)) for i in ids)


def pair_prob(logits: torch.Tensor, bad_id: int, good_id: int) -> float:
    pair = torch.stack([logits[bad_id], logits[good_id]])
    return float(torch.softmax(pair, dim=0)[0].item())


def kl_bits(before: torch.Tensor, after: torch.Tensor) -> float:
    p_log = F.log_softmax(before.float(), dim=-1)
    q_log = F.log_softmax(after.float(), dim=-1)
    return float(torch.sum(torch.exp(p_log) * (p_log - q_log)).item() / math.log(2.0))


def topk_overlap(before: torch.Tensor, after: torch.Tensor, k: int = 50) -> float:
    a = set(torch.topk(before, k=min(k, before.numel())).indices.tolist())
    b = set(torch.topk(after, k=min(k, after.numel())).indices.tolist())
    return float(len(a & b) / max(1, len(a)))


def stable_row_id(*parts: object) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Loaded:
    model: object
    tokenizer: object
    decoder: torch.Tensor
    encoder_w: torch.Tensor
    encoder_b: torch.Tensor
    W_U: torch.Tensor
    row_mean: torch.Tensor
    labels: dict[int, list[str]]


def load_all(args: argparse.Namespace) -> Loaded:
    log(f"loading tokenizer/model: {args.model_id}")
    # Shared loader: multimodal-first auto-class order (Qwen3.5 only loads via
    # AutoModelForImageTextToText), frozen params, eval mode.
    model, tokenizer = load_qwen_model(
        args.model_id,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)

    log(f"loading SAE: {args.checkpoint}")
    decoder, encoder_w, encoder_b, _config, ckpt_row_mean = load_sae(args.checkpoint)
    lm_head, _path = find_lm_head_with_path(model)
    tensor_device = args.device if args.tensor_device == "auto" else args.tensor_device
    W_U = lm_head.weight.detach().float().to(tensor_device)
    # Center against the exact mean the dictionary was trained with (stored in
    # the checkpoint); falls back to the live-model mean only for legacy
    # checkpoints that predate the stored row_mean.
    row_mean = resolve_row_mean(W_U, ckpt={"row_mean": ckpt_row_mean}).to(tensor_device)
    decoder = decoder.to(tensor_device)
    encoder_w = encoder_w.to(tensor_device)
    encoder_b = encoder_b.to(tensor_device)
    n_features = decoder.shape[0]
    label_ids = sorted({fid for ids in FEATURE_SETS.values() for fid in ids if fid < n_features})
    log(f"tensor_device={tensor_device}; labeling {len(label_ids)} FEATURE_SETS ids (D={n_features})")
    labels = display_label_features(
        W=W_U,
        row_mean=row_mean,
        feature_ids=label_ids,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        tokenizer=tokenizer,
        top_tokens=8,
        chunk_size=args.label_chunk_size,
    )
    return Loaded(model, tokenizer, decoder, encoder_w, encoder_b, W_U, row_mean, labels)


@torch.no_grad()
def next_logits(model, tokenizer, prompt: str, device: str, keep_on_device: bool = False) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc).logits[0, -1].float().detach()
    return out if keep_on_device else out.cpu()


@torch.no_grad()
def next_logits_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    batch_size: int,
    keep_on_device: bool = False,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            logits = model(**enc).logits[:, -1, :].float().detach()
            if not keep_on_device:
                logits = logits.cpu()
            for prompt, row in zip(batch, logits):
                out[prompt] = row
    finally:
        tokenizer.padding_side = padding_side
    return out


def suppress_vec(decoder: torch.Tensor, feature_ids: list[int], scale: float) -> torch.Tensor:
    idx = torch.tensor(feature_ids, dtype=torch.long, device=decoder.device)
    directions = decoder[idx].float()
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return -float(scale) * directions.sum(dim=0)


def category_direction_units(
    W_U: torch.Tensor, discovery_ids: list[int], row_mean: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-row and top-PCA unit directions from the discovery tokens' unembedding
    rows, centered by the shared row mean (parity with the readout SAE, which is
    trained on row-mean-centered rows). Both point toward the discovery side, so
    subtracting a positive multiple suppresses those tokens (matching the sign
    convention of ``suppress_vec``). These are the label-free "mean row / PCA
    directions of the same token group" controls."""
    idx = torch.tensor(discovery_ids, dtype=torch.long, device=W_U.device)
    rows = W_U[idx].float()  # (n_disc, d_model)
    centered = rows - row_mean.to(rows.dtype)  # deviations from the global row mean
    m = centered.mean(dim=0)  # (d_model,)
    mean_dir = m / m.norm().clamp_min(1e-12)
    # Top principal component about the global-mean origin (uncentered SVD; rank
    # <= n_disc). Deterministic: torch.linalg.svd, not the randomized pca_lowrank.
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    pc = vh[0]  # (d_model,)
    if torch.dot(pc, mean_dir) < 0:  # orient toward the discovery ("profane") side
        pc = -pc
    pca_dir = pc / pc.norm().clamp_min(1e-12)
    # PC set of rank <= 4 (capped at the number of discovery rows): the
    # capacity-fair "PCA directions" of the token group, each unit-norm and
    # oriented toward the discovery side.
    n_pc = min(4, vh.shape[0])
    pca_dirs = vh[:n_pc].clone()
    for j in range(n_pc):
        if torch.dot(pca_dirs[j], mean_dir) < 0:
            pca_dirs[j] = -pca_dirs[j]
        pca_dirs[j] = pca_dirs[j] / pca_dirs[j].norm().clamp_min(1e-12)
    return mean_dir, pca_dir, pca_dirs


def random_feature_ids(n_features: int, n_total: int, seed: int = 17) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(n_total), k=n_features)


def calibrated_token_bias_delta(
    feature_delta: torch.Tensor,
    audited_token_ids: list[int],
    fallback_strength: float,
) -> float:
    if not audited_token_ids:
        return float(fallback_strength)
    reductions = -feature_delta[audited_token_ids].float().detach().cpu().numpy()
    reductions = reductions[np.isfinite(reductions)]
    if len(reductions) == 0:
        return float(fallback_strength)
    return float(max(0.0, reductions.mean()))


def apply_method(
    *,
    logits: torch.Tensor,
    W_U: torch.Tensor,
    decoder: torch.Tensor,
    method: str,
    feature_ids: list[int],
    random_ids: list[int],
    scale: float,
    token_ids: list[int],
    audited_ids: list[int],
    mean_dir: torch.Tensor | None = None,
    pca_dir: torch.Tensor | None = None,
    pca_dirs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if method == "none":
        return logits.clone(), {"token_bias_strength": 0.0, "intervention_norm": 0.0}
    if method == "feature_suppression":
        v = suppress_vec(decoder, feature_ids, scale)
        return logits + v @ W_U.T, {
            "token_bias_strength": 0.0,
            "intervention_norm": float(v.norm().item()),
        }
    if method == "random_feature_suppression":
        v_ref = suppress_vec(decoder, feature_ids, scale)
        v = suppress_vec(decoder, random_ids, scale)
        v = v * (v_ref.norm() / v.norm().clamp_min(1e-12))
        return logits + v @ W_U.T, {
            "token_bias_strength": 0.0,
            "intervention_norm": float(v.norm().item()),
        }
    if method == "mean_row_direction":
        if mean_dir is None:
            raise ValueError("mean_row_direction requires mean_dir")
        v = -float(scale) * mean_dir
        return logits + v @ W_U.T, {
            "token_bias_strength": 0.0,
            "intervention_norm": float(v.norm().item()),
        }
    if method == "pca_group_direction":
        if pca_dir is None:
            raise ValueError("pca_group_direction requires pca_dir")
        v = -float(scale) * pca_dir
        return logits + v @ W_U.T, {
            "token_bias_strength": 0.0,
            "intervention_norm": float(v.norm().item()),
        }
    if method == "pca_group_rank4":
        if pca_dirs is None:
            raise ValueError("pca_group_rank4 requires pca_dirs")
        v = -float(scale) * pca_dirs.sum(dim=0)  # summed unit PCs, as suppress_vec
        return logits + v @ W_U.T, {
            "token_bias_strength": 0.0,
            "intervention_norm": float(v.norm().item()),
        }
    if method.startswith("token_logit_bias"):
        v_ref = suppress_vec(decoder, feature_ids, scale)
        feature_delta = v_ref @ W_U.T
        strength = calibrated_token_bias_delta(feature_delta, audited_ids, fallback_strength=scale)
        edited = logits.clone()
        if token_ids:
            edited[torch.tensor(token_ids, dtype=torch.long, device=edited.device)] -= strength
        return edited, {
            "token_bias_strength": strength,
            "intervention_norm": float(strength * math.sqrt(max(1, len(token_ids)))),
        }
    raise ValueError(f"unknown method {method}")


def token_audit(tokenizer) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    ids: dict[str, int] = {}
    for group, terms in (("profanity", PROFANITY_TERMS), ("control", CONTROL_TERMS)):
        for term in terms:
            token_id, variant, note = single_token_id(tokenizer, term)
            rows.append(
                {
                    "group": group,
                    "term": term,
                    "token_id": "" if token_id is None else token_id,
                    "token_label": "" if token_id is None else clean_token(tokenizer, token_id),
                    "tokenization_variant": variant,
                    "note": note,
                }
            )
            if token_id is not None:
                ids[term] = token_id
    return rows, ids


@torch.no_grad()
def feature_discovery_audit(loaded: Loaded, token_ids: dict[str, int], topn: int = 20) -> tuple[list[dict], list[dict]]:
    vocab_for_labels = min(loaded.W_U.shape[0], len(loaded.tokenizer))
    rows: list[dict] = []
    feature_acc: dict[int, dict[str, object]] = {}
    for group, terms in (("profanity", PROFANITY_TERMS), ("control", CONTROL_TERMS)):
        for term in terms:
            token_id = token_ids.get(term)
            if token_id is None or token_id >= vocab_for_labels:
                continue
            row = loaded.W_U[token_id : token_id + 1]
            x = row - loaded.row_mean
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
            acts = F.relu(x @ loaded.encoder_w.T + loaded.encoder_b)[0]
            values, indices = torch.topk(acts, k=min(topn, acts.numel()))
            for rank, (fid_t, act_t) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                fid = int(fid_t)
                act = float(act_t)
                rows.append(
                    {
                        "group": group,
                        "term": term,
                        "token_id": int(token_id),
                        "token_label": clean_token(loaded.tokenizer, token_id),
                        "rank": rank,
                        "feature_id": fid,
                        "activation": act,
                    }
                )
                acc = feature_acc.setdefault(
                    fid,
                    {
                        "feature_id": fid,
                        "profane_token_count_top20": 0,
                        "control_token_count_top20": 0,
                        "profane_activation_sum": 0.0,
                        "control_activation_sum": 0.0,
                        "max_profane_activation": 0.0,
                        "profane_tokens": set(),
                        "control_tokens": set(),
                    },
                )
                if group == "profanity":
                    acc["profane_token_count_top20"] = int(acc["profane_token_count_top20"]) + 1
                    acc["profane_activation_sum"] = float(acc["profane_activation_sum"]) + act
                    acc["max_profane_activation"] = max(float(acc["max_profane_activation"]), act)
                    acc["profane_tokens"].add(term)
                else:
                    acc["control_token_count_top20"] = int(acc["control_token_count_top20"]) + 1
                    acc["control_activation_sum"] = float(acc["control_activation_sum"]) + act
                    acc["control_tokens"].add(term)

    feature_ids = sorted(feature_acc)
    labels = display_label_features(
        W=loaded.W_U,
        row_mean=loaded.row_mean,
        feature_ids=feature_ids,
        encoder_w=loaded.encoder_w,
        encoder_b=loaded.encoder_b,
        tokenizer=loaded.tokenizer,
        top_tokens=8,
        chunk_size=16384,
    )
    summary: list[dict] = []
    for fid, acc in feature_acc.items():
        prof = float(acc["profane_activation_sum"])
        ctrl = float(acc["control_activation_sum"])
        lab = readable_feature_label(labels.get(fid, []), fid)
        prof_tokens = sorted(acc["profane_tokens"])
        ctrl_tokens = sorted(acc["control_tokens"])
        summary.append(
            {
                "feature_id": fid,
                "feature_label": lab,
                "profane_token_count_top20": int(acc["profane_token_count_top20"]),
                "control_token_count_top20": int(acc["control_token_count_top20"]),
                "profane_activation_sum": prof,
                "control_activation_sum": ctrl,
                "profanity_specificity": prof / (prof + ctrl + 1e-9),
                "max_profane_activation": float(acc["max_profane_activation"]),
                "top_token_labels": ";".join(labels.get(fid, [])[:8]),
                "profane_tokens": ";".join(prof_tokens),
                "control_tokens": ";".join(ctrl_tokens),
            }
        )
    summary.sort(
        key=lambda r: (
            float(r["profane_activation_sum"]),
            float(r["profanity_specificity"]),
        ),
        reverse=True,
    )
    for row in rows:
        row["feature_label"] = readable_feature_label(labels.get(int(row["feature_id"]), []), int(row["feature_id"]))
    return rows, summary


def select_discovery_features(token_feature_rows: list[dict], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for row in token_feature_rows:
        if row["group"] != "profanity" or row["term"] not in DISCOVERY_TERMS:
            continue
        fid = int(row["feature_id"])
        scores[fid] = scores.get(fid, 0.0) + float(row["activation"])
    ordered = [fid for fid, _score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ordered[:k]


def build_eval_cases_with_tokenizer(tokenizer, token_ids: dict[str, int]) -> list[dict]:
    cases = []
    for prompt in PROMPTS:
        for bad, good in PAIRS:
            bad_id = token_ids.get(bad)
            good_id, _variant, _note = single_token_id(tokenizer, good)
            if bad_id is None or good_id is None:
                continue
            cases.append(
                {
                    "id": stable_row_id(prompt, bad, good),
                    "prompt": prompt,
                    "bad": bad,
                    "good": good,
                    "bad_id": bad_id,
                    "good_id": good_id,
                    "feature_set": PAIR_TO_FEATURE_SET[bad],
                    "split": "discovery" if bad in DISCOVERY_TERMS else "heldout",
                }
            )
    return cases


def candidate_rows(
    loaded: Loaded,
    cases: list[dict],
    token_ids: dict[str, int],
    feature_ids: list[int],
    methods: list[str],
    scales: list[float],
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    discovery_ids = [token_ids[t] for t in DISCOVERY_TERMS if t in token_ids]
    all_profane_ids = [token_ids[t] for t in PROFANITY_TERMS if t in token_ids]
    random_ids = random_feature_ids(len(feature_ids), loaded.decoder.shape[0], seed=args.random_seed)
    mean_dir, pca_dir, pca_dirs = (None, None, None)
    if discovery_ids:
        mean_dir, pca_dir, pca_dirs = category_direction_units(loaded.W_U, discovery_ids, loaded.row_mean)
    unique_prompts = sorted({case["prompt"] for case in cases})
    log(f"computing next-token logits for {len(unique_prompts)} unique prompts in batches of {args.batch_size}")
    prompt_logits = next_logits_batch(
        loaded.model,
        loaded.tokenizer,
        unique_prompts,
        args.device,
        args.batch_size,
        keep_on_device=(loaded.W_U.device.type == "cuda"),
    )
    log("candidate prompt logits cached")
    logits_cache = {case["id"]: prompt_logits[case["prompt"]] for case in cases}
    base_top8_cache: dict[str, str] = {}
    base_topk_cache: dict[str, set[int]] = {}
    total_rows = len(cases) * len(scales) * len(methods)
    completed = 0
    t_start = time.time()
    for case_idx, case in enumerate(cases):
        logits = logits_cache[case["id"]]
        bad_id = int(case["bad_id"])
        good_id = int(case["good_id"])
        base_margin = float((logits[bad_id] - logits[good_id]).item())
        base_prob = pair_prob(logits, bad_id, good_id)
        if case["id"] not in base_top8_cache:
            base_top8_cache[case["id"]] = topk_labels(loaded.tokenizer, logits)
            base_topk_cache[case["id"]] = set(torch.topk(logits, k=min(50, logits.numel())).indices.tolist())
        base_top8 = base_top8_cache[case["id"]]
        base_topk_set = base_topk_cache[case["id"]]
        for scale in scales:
            for method in methods:
                ids = []
                audited_ids = all_profane_ids
                if method == "token_logit_bias_discovery":
                    ids = discovery_ids
                    audited_ids = discovery_ids
                elif method == "token_logit_bias_oracle":
                    ids = all_profane_ids
                    audited_ids = all_profane_ids
                edited, meta = apply_method(
                    logits=logits,
                    W_U=loaded.W_U,
                    decoder=loaded.decoder,
                    method=method,
                    feature_ids=feature_ids,
                    random_ids=random_ids,
                    scale=scale,
                    token_ids=ids,
                    audited_ids=audited_ids,
                    mean_dir=mean_dir,
                    pca_dir=pca_dir,
                    pca_dirs=pca_dirs,
                )
                edit_margin = float((edited[bad_id] - edited[good_id]).item())
                edit_prob = pair_prob(edited, bad_id, good_id)
                rows.append(
                    {
                        "id": case["id"],
                        "prompt": case["prompt"],
                        "bad": case["bad"],
                        "good": case["good"],
                        "split": case["split"],
                        "method": method,
                        "feature_set": "discovery_selected",
                        "feature_ids": ";".join(str(fid) for fid in feature_ids),
                        "scale": scale,
                        "baseline_margin_bad_minus_good": base_margin,
                        "suppressed_margin_bad_minus_good": edit_margin,
                        "delta_margin": edit_margin - base_margin,
                        "baseline_bad_prob_pair": base_prob,
                        "suppressed_bad_prob_pair": edit_prob,
                        "bad_prob_reduction": base_prob - edit_prob,
                        "baseline_choice": case["bad"] if base_margin > 0 else case["good"],
                        "suppressed_choice": case["bad"] if edit_margin > 0 else case["good"],
                        "flip": int(base_margin > 0 and edit_margin < 0),
                        "full_vocab_kl_bits": kl_bits(logits, edited),
                        "top_k_overlap": topk_overlap_from_base(base_topk_set, edited, k=50),
                        "top8_before": base_top8,
                        "top8_after": topk_labels(loaded.tokenizer, edited),
                        **meta,
                    }
                )
                completed += 1
                if completed % 500 == 0 or completed == total_rows:
                    elapsed = time.time() - t_start
                    rate = completed / max(elapsed, 1e-9)
                    eta = (total_rows - completed) / max(rate, 1e-9)
                    log(f"candidate {completed}/{total_rows} ({rate:.1f} rows/s, eta {eta:.0f}s)")
    return rows


def topk_overlap_from_base(base_topk_set: set[int], after: torch.Tensor, k: int = 50) -> float:
    b = set(torch.topk(after, k=min(k, after.numel())).indices.tolist())
    return float(len(base_topk_set & b) / max(1, len(base_topk_set)))


def summarize_candidate_rows(rows: list[dict]) -> tuple[list[dict], dict[str, float]]:
    groups: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], float(row["scale"]), row["split"]), []).append(row)
    out: list[dict] = []
    for (method, scale, split), rs in sorted(groups.items()):
        flips = [float(r["flip"]) for r in rs]
        red = [float(r["bad_prob_reduction"]) for r in rs]
        kl = [float(r["full_vocab_kl_bits"]) for r in rs]
        flo, fhi = ci_mean(flips)
        rlo, rhi = ci_mean(red)
        out.append(
            {
                "method": method,
                "scale": scale,
                "split": split,
                "n": len(rs),
                "candidate_flip_rate": float(np.mean(flips)) if flips else float("nan"),
                "candidate_flip_rate_ci_lo": flo,
                "candidate_flip_rate_ci_hi": fhi,
                "mean_bad_prob_reduction": float(np.mean(red)) if red else float("nan"),
                "mean_bad_prob_reduction_ci_lo": rlo,
                "mean_bad_prob_reduction_ci_hi": rhi,
                "median_kl_bits": float(np.median(kl)) if kl else float("nan"),
                "mean_top_k_overlap": float(np.mean([float(r["top_k_overlap"]) for r in rs])) if rs else float("nan"),
            }
        )
    operating = choose_operating_point(out)
    return out, operating


def choose_operating_point(summary_rows: list[dict]) -> dict[str, float]:
    feature_rows = [r for r in summary_rows if r["method"] == "feature_suppression" and r["split"] == "discovery"]
    feature_rows.sort(key=lambda r: float(r["scale"]))
    for row in feature_rows:
        if float(row["candidate_flip_rate"]) >= 0.80 and float(row["median_kl_bits"]) < 0.10:
            return {"selected_scale": float(row["scale"]), "selected_by_rule": 1.0}
    if not feature_rows:
        return {"selected_scale": 1.0, "selected_by_rule": 0.0}
    best = max(
        feature_rows,
        key=lambda r: (float(r["candidate_flip_rate"]), -float(r["median_kl_bits"])),
    )
    return {"selected_scale": float(best["scale"]), "selected_by_rule": 0.0}


def heldout_generalization_rows(candidate_summary: list[dict], feature_count: int, token_count: int) -> list[dict]:
    rows: list[dict] = []
    by_key = {(r["method"], r["scale"], r["split"]): r for r in candidate_summary}
    for method in sorted({r["method"] for r in candidate_summary}):
        for scale in sorted({float(r["scale"]) for r in candidate_summary if r["method"] == method}):
            d = by_key.get((method, scale, "discovery"), {})
            h = by_key.get((method, scale, "heldout"), {})
            rows.append(
                {
                    "method": method,
                    "scale": scale,
                    "discovery_flip_rate": d.get("candidate_flip_rate", float("nan")),
                    "heldout_flip_rate": h.get("candidate_flip_rate", float("nan")),
                    "discovery_bad_prob_reduction": d.get("mean_bad_prob_reduction", float("nan")),
                    "heldout_bad_prob_reduction": h.get("mean_bad_prob_reduction", float("nan")),
                    "feature_count": feature_count if method == "feature_suppression" else 0,
                    "token_count_covered": token_count,
                }
            )
    return rows


def baseline_comparison_rows(candidate_summary: list[dict], selected_scale: float) -> list[dict]:
    rows = [r for r in candidate_summary if abs(float(r["scale"]) - selected_scale) < 1e-9]
    out: list[dict] = []
    for method in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == method]
        out.append(
            {
                "method": method,
                "scale": selected_scale,
                "n_splits": len(rs),
                "mean_flip_rate": float(np.mean([float(r["candidate_flip_rate"]) for r in rs])) if rs else float("nan"),
                "mean_bad_prob_reduction": float(np.mean([float(r["mean_bad_prob_reduction"]) for r in rs]))
                if rs
                else float("nan"),
                "median_kl_bits": float(np.median([float(r["median_kl_bits"]) for r in rs])) if rs else float("nan"),
                "mean_top_k_overlap": float(np.mean([float(r["mean_top_k_overlap"]) for r in rs]))
                if rs
                else float("nan"),
            }
        )
    return out


def dose_response_rows(candidate_summary: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in candidate_summary:
        if row["split"] != "discovery":
            continue
        out.append(
            {
                "method": row["method"],
                "scale": row["scale"],
                "candidate_flip_rate": row["candidate_flip_rate"],
                "mean_bad_token_probability_reduction": row["mean_bad_prob_reduction"],
                "median_kl_bits": row["median_kl_bits"],
                "benign_regression_rate": "",
            }
        )
    return out


def add_benign_rates_to_dose(dose: list[dict], benign: list[dict], selected_scale: float) -> list[dict]:
    by_method: dict[str, list[float]] = {}
    for row in benign:
        by_method.setdefault(str(row["method"]), []).append(float(row["top1_changed"]))
    rates = {m: float(np.mean(vals)) for m, vals in by_method.items() if vals}
    for row in dose:
        if abs(float(row["scale"]) - selected_scale) < 1e-9 and row["method"] in rates:
            row["benign_regression_rate"] = rates[row["method"]]
    return dose


def benign_regression_rows(
    loaded: Loaded,
    token_ids: dict[str, int],
    feature_ids: list[int],
    scale: float,
    args: argparse.Namespace,
) -> list[dict]:
    random_ids = random_feature_ids(len(feature_ids), loaded.decoder.shape[0], seed=args.random_seed)
    rows: list[dict] = []
    target_terms = ["hell", "sucks", "vulgar", "obscene", "fuck"]
    target_ids = {t: token_ids.get(t) or single_token_id(loaded.tokenizer, t)[0] for t in target_terms}
    all_profane_ids = [token_ids[t] for t in PROFANITY_TERMS if t in token_ids]
    keep_on_device = loaded.W_U.device.type == "cuda"
    for prompt in BENIGN_PROMPTS:
        logits = next_logits(
            loaded.model,
            loaded.tokenizer,
            prompt,
            args.device,
            keep_on_device=keep_on_device,
        )
        base_top = int(torch.argmax(logits).item())
        for method in [
            "feature_suppression",
            "token_logit_bias_oracle",
            "random_feature_suppression",
        ]:
            edited, meta = apply_method(
                logits=logits,
                W_U=loaded.W_U,
                decoder=loaded.decoder,
                method=method,
                feature_ids=feature_ids,
                random_ids=random_ids,
                scale=scale,
                token_ids=all_profane_ids if method == "token_logit_bias_oracle" else [],
                audited_ids=all_profane_ids,
            )
            edit_top = int(torch.argmax(edited).item())
            for term, tid in target_ids.items():
                if tid is None:
                    continue
                rows.append(
                    {
                        "prompt": prompt,
                        "target_word": term,
                        "target_token_id": tid,
                        "method": method,
                        "scale": scale,
                        "baseline_target_logit": float(logits[tid].item()),
                        "edited_target_logit": float(edited[tid].item()),
                        "target_word_margin_reduction": float((logits[tid] - edited[tid]).item()),
                        "top1_changed": int(base_top != edit_top),
                        "baseline_top1": clean_token(loaded.tokenizer, base_top),
                        "edited_top1": clean_token(loaded.tokenizer, edit_top),
                        "full_vocab_kl_bits": kl_bits(logits, edited),
                        "top_k_overlap": topk_overlap(logits, edited, k=50),
                        "artifact_rate": 0.0,
                        **meta,
                    }
                )
    return rows


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    suppress: torch.Tensor | None = None,
) -> str:
    handle = None
    if suppress is not None:
        lm_head, _path = find_lm_head_with_path(model)

        def pre_hook(_module, inputs):
            h = inputs[0]
            h2 = h.clone()
            h2[:, -1, :] = h2[:, -1, :] + suppress.to(h2.device, dtype=h2.dtype)
            return (h2,)

        handle = lm_head.register_forward_pre_hook(pre_hook)
    try:
        enc = tokenizer(prompt, return_tensors="pt").to(args.device)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


def open_generation_rows(
    loaded: Loaded,
    token_ids: dict[str, int],
    feature_ids: list[int],
    selected_scale: float,
    args: argparse.Namespace,
) -> list[dict]:
    if args.max_open_prompts <= 0:
        return []
    rows: list[dict] = []
    random_ids = random_feature_ids(len(feature_ids), loaded.decoder.shape[0], seed=args.random_seed)
    all_profane = set(PROFANITY_TERMS)
    conditions = [
        ("baseline", None),
        (
            "feature_suppression_conservative",
            suppress_vec(loaded.decoder, feature_ids, selected_scale),
        ),
        (
            "feature_suppression_stronger",
            suppress_vec(loaded.decoder, feature_ids, selected_scale * 2.0),
        ),
        (
            "random_feature_suppression",
            suppress_vec(loaded.decoder, random_ids, selected_scale),
        ),
    ]
    prompts = OPEN_GENERATION_PROMPTS[: args.max_open_prompts]
    for prompt in prompts:
        for condition, vec in conditions:
            for sample_idx in range(args.num_samples_per_prompt):
                text = generate_one(loaded.model, loaded.tokenizer, prompt, args, suppress=vec)
                low = text.lower()
                hits = [t for t in all_profane if t in low]
                artifacts = [m for m in ARTIFACT_MARKERS if m.lower() in low]
                rows.append(
                    {
                        "prompt": prompt,
                        "condition": condition,
                        "sample_idx": sample_idx,
                        "continuation": text.replace("\n", "\\n"),
                        "audited_profanity_occurrence": int(bool(hits)),
                        "audited_terms_hit": ";".join(sorted(hits)),
                        "repetition_rate": repetition_rate(text),
                        "markdown_censor_artifact_rate": float(bool(artifacts)),
                        "artifact_markers": ";".join(artifacts),
                        "average_continuation_length": len(text.split()),
                    }
                )
    return rows


def repetition_rate(text: str) -> float:
    toks = text.lower().split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return 1.0 - (len(set(bigrams)) / max(1, len(bigrams)))


def write_summary_md(
    path: Path,
    manifest: dict,
    op: dict,
    candidate_summary: list[dict],
    benign_rows: list[dict],
) -> None:
    selected = float(op["selected_scale"])
    fs_rows = [
        r
        for r in candidate_summary
        if r["method"] == "feature_suppression" and abs(float(r["scale"]) - selected) < 1e-9
    ]
    held = next((r for r in fs_rows if r["split"] == "heldout"), None)
    disc = next((r for r in fs_rows if r["split"] == "discovery"), None)
    benign_rate = float(np.mean([float(r["top1_changed"]) for r in benign_rows])) if benign_rows else float("nan")
    text = [
        "# Qwen Profanity Suppression Evaluation Summary",
        "",
        "This evaluation supports only the narrow lexical-control claim: audited readout-feature suppression can reduce or flip selected profanity-token continuations without retraining. It is not a toxicity, alignment, or safety evaluation.",
        "",
        f"- Model: `{manifest['model_id']}`",
        f"- SAE checkpoint: `{manifest['checkpoint']}`",
        f"- Selected scale: `{selected:g}` (`selected_by_rule={int(op['selected_by_rule'])}`)",
        f"- Discovery feature count: `{manifest['selected_feature_count']}`",
    ]
    if disc:
        text.append(
            f"- Discovery candidate flip rate: `{float(disc['candidate_flip_rate']):.3f}`; mean bad-prob reduction: `{float(disc['mean_bad_prob_reduction']):.3f}`"
        )
    if held:
        text.append(
            f"- Held-out candidate flip rate: `{float(held['candidate_flip_rate']):.3f}`; mean bad-prob reduction: `{float(held['mean_bad_prob_reduction']):.3f}`"
        )
    text.extend(
        [
            f"- Benign top-1 regression rate at selected scale: `{benign_rate:.3f}`",
            "",
            "Always-on suppression is intentionally treated as too blunt for applications that must quote or discuss profanity. Use the benign-regression rows to motivate context gating.",
        ]
    )
    path.write_text("\n".join(text) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--revision", default=None, help="HF weight revision to pin (default: latest)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained readout-feature SAE checkpoint.pt (qwen2b 32x/k256 for "
        "tab:lexical-control-primary-methods; per-model checkpoints for "
        "tab:lexical-control-cross-model-results are in "
        "configs/registries/result1_query_fidelity_cluster.yaml).",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--tensor-device",
        default="auto",
        help="Device for W_U / SAE tensors and intervention math. 'auto' follows --device.",
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--allow-feature-set-fallback",
        action="store_true",
        help="if discovery selects nothing, fall back to the hardcoded Qwen3.5-2B 32x/k256 FEATURE_SETS ids",
    )
    ap.add_argument("--selected-feature-count", type=int, default=10)
    ap.add_argument("--random-seed", type=int, default=17)
    ap.add_argument("--label-chunk-size", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Debug only: limit prompt templates.",
    )
    ap.add_argument(
        "--pair-limit",
        type=int,
        default=None,
        help="Debug only: limit candidate pairs.",
    )
    ap.add_argument(
        "--max-open-prompts",
        type=int,
        default=0,
        help="0 writes an empty open_generation_rows.csv.",
    )
    ap.add_argument("--num-samples-per-prompt", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()
    set_seed(args.random_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_all(args)
    token_rows, token_ids = token_audit(loaded.tokenizer)
    write_csv(args.out_dir / "tokenization_audit.csv", token_rows)

    log("running feature discovery audit")
    feature_token_rows, feature_summary = feature_discovery_audit(loaded, token_ids)
    write_csv(args.out_dir / "feature_token_rows.csv", feature_token_rows)
    write_csv(args.out_dir / "feature_summary.csv", feature_summary)

    n_features = loaded.decoder.shape[0]
    selected_features = select_discovery_features(feature_token_rows, args.selected_feature_count)
    if not selected_features:
        if not args.allow_feature_set_fallback:
            raise SystemExit(
                "feature discovery selected nothing; the FEATURE_SETS fallback ids are "
                "specific to the Qwen3.5-2B 32x/k256 paper checkpoint and would be wrong "
                "for this model — pass --allow-feature-set-fallback only if that is the "
                "checkpoint you are evaluating"
            )
        # Qwen-2B ids; only valid on dictionaries wide enough to contain them.
        selected_features = [f for f in FEATURE_SETS["all_narrow"] if f < n_features]
    selected_features = [f for f in selected_features if f < n_features][: args.selected_feature_count]
    log(f"selected discovery features (D={n_features}): {selected_features}")

    cases = build_eval_cases_with_tokenizer(loaded.tokenizer, token_ids)
    if args.prompt_limit is not None:
        keep_prompts = set(PROMPTS[: args.prompt_limit])
        cases = [case for case in cases if case["prompt"] in keep_prompts]
    if args.pair_limit is not None:
        keep_pairs = {bad for bad, _good in PAIRS[: args.pair_limit]}
        cases = [case for case in cases if case["bad"] in keep_pairs]
    methods = [
        "none",
        "feature_suppression",
        "token_logit_bias_discovery",
        "token_logit_bias_oracle",
        "random_feature_suppression",
        "mean_row_direction",
        "pca_group_direction",
        "pca_group_rank4",
    ]
    log(f"evaluating {len(cases)} candidate-constrained rows across {len(methods)} methods")
    cand = candidate_rows(loaded, cases, token_ids, selected_features, methods, SCALES, args)
    write_csv(args.out_dir / "candidate_constrained_rows.csv", cand)
    cand_summary, operating = summarize_candidate_rows(cand)
    write_csv(args.out_dir / "candidate_constrained_summary.csv", cand_summary)
    write_json(args.out_dir / "operating_point.json", operating)

    heldout = heldout_generalization_rows(
        cand_summary,
        feature_count=len(selected_features),
        token_count=len([token_ids[t] for t in DISCOVERY_TERMS if t in token_ids]),
    )
    write_csv(args.out_dir / "heldout_generalization_rows.csv", heldout)
    baseline = baseline_comparison_rows(cand_summary, float(operating["selected_scale"]))
    write_csv(args.out_dir / "baseline_comparison.csv", baseline)

    log("running benign regression checks")
    benign = benign_regression_rows(loaded, token_ids, selected_features, float(operating["selected_scale"]), args)
    write_csv(args.out_dir / "benign_regression_rows.csv", benign)
    dose = add_benign_rates_to_dose(dose_response_rows(cand_summary), benign, float(operating["selected_scale"]))
    write_csv(args.out_dir / "dose_response.csv", dose)

    log("running open-generation pilot" if args.max_open_prompts > 0 else "skipping open-generation pilot")
    open_rows = open_generation_rows(loaded, token_ids, selected_features, float(operating["selected_scale"]), args)
    write_csv(args.out_dir / "open_generation_rows.csv", open_rows)

    manifest = {
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "out_dir": str(args.out_dir),
        "scales": SCALES,
        "feature_sets": FEATURE_SETS,
        "selected_features": selected_features,
        "selected_feature_count": len(selected_features),
        "methods": methods,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "open_generation_prompt_count": args.max_open_prompts,
        "non_claims": ["toxicity", "alignment", "broad AI safety"],
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_summary_md(args.out_dir / "summary.md", manifest, operating, cand_summary, benign)

    # Tables (baseline_comparison.csv / candidate_constrained_summary.csv) are the
    # only paper-facing outputs (Appendix N tab:lexical-control-*); the eval emits
    # no paper figure, so it does not render or copy any plots.
    log(f"done: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
