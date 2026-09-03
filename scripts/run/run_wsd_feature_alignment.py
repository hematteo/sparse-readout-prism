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
``scoring_summary.json`` (contexts scored, median target rank, row
coverage, prompts truncated) and ``metrics.json`` (full-vector centroid
references computed on the same bundle, with paired per-word bootstrap
differences).

Dataset
-------
CoarseWSD-20 (Loureiro, Rezaee, Pilehvar and Camacho-Collados, 2021,
Computational Linguistics 47(2), doi:10.1162/coli_a_00405). ``--data-root``
is a checkout of https://github.com/danlou/bert-disambiguation; the loader
looks for ``<root>/data/CoarseWSD-20/<word>/{train,test}.{data,gold}.txt``
plus ``class_map.txt`` (``<root>`` or ``<root>/CoarseWSD-20`` also work). The
shipped train/test split is kept. The target occurrence is replaced by
``[BLANK]`` and the sentence is wrapped in ``COARSEWSD_TEMPLATE``; the
target's leading-space token is scored at the prompt's final position, so it
is a genuine next-token readout even when the disambiguating evidence
originally followed the target.

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
``metrics.json`` from a frozen bundle without model inference; it writes
``analysis_config.json`` and leaves the scoring run's ``run_config.json``
untouched.

Gold labels are never used to fit the dictionary or to select features.

``--centering live`` (default, the paper's runs) centres unembedding rows on
the full-vocabulary mean of the live model's LM head; ``--centering trained``
uses the dictionary's stored training mean (``row_mean`` in the checkpoint),
falling back to the mean over the tokenizer's text-token rows.

Fixed in 0.2.1: prompts longer than ``--max-length`` are truncated from the
left, so the cloze cue at the end of the prompt survives (they were truncated
from the right; ``scoring_summary.json`` now records how many prompts were
truncated). The AmbiStory dataset mode, ``--position-mode occurrence`` and the
``--anchors`` bundle block, none of which the paper reports, were removed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparse_readout_prism.data import center_normalize_rows, centering_mean, token_mask_from_tokenizer
from sparse_readout_prism.research.qwen_readout import encode_topk, load_sae
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.research.wsd import (
    bootstrap_mean,
    l2_normalize,
    load_bundle,
    percentile_ci,
    safe_spearman,
    shuffled_srp,
    word_splits,
)
from sparse_readout_prism.utils import find_lm_head, load_causal_lm, set_seed, write_json, write_jsonl


COARSEWSD_TEMPLATE = """Read the sentence and recover the missing word.

{sentence}

The missing word is:"""

METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "pairwise_auc", "ari", "nmi")
COMPARISON_BASELINES = ("shuffled_srp", "support_projection", "hidden")


def log(message: str) -> None:
    print(f"[wsd {time.strftime('%H:%M:%S')}] {message}", flush=True)


def stable_id(*parts: str, n: int = 16) -> str:
    text = "\x1f".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


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


def configure_tokenizer(tokenizer) -> None:
    """Left padding keeps the scored position last in a batch; left truncation
    keeps the cloze cue (the prompt's tail) when a prompt exceeds ``--max-length``."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"


def load_model(model_id: str, device: str, dtype: str):
    """Frozen causal LM + tokenizer configured by ``configure_tokenizer``."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model, tokenizer = load_causal_lm(model_id, dtype=torch_dtype, device_map=None)
    model.to(device)
    configure_tokenizer(tokenizer)
    return model, tokenizer


def checkpoint_k(path: Path, sae_config: dict[str, Any]) -> int:
    """The dictionary's k: ``config.factorizer.k`` when ``load_sae`` surfaced a
    config, else the runner checkpoint's top-level ``factorizer.k`` (read with
    ``mmap`` so the weights are not loaded twice); 256 when neither names it."""
    cfg = sae_config.get("factorizer") or {}
    if "k" not in cfg:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer", {})
    return int(cfg.get("k", 256))


def scoring_row_mean(w_u: torch.Tensor, tokenizer, mode: str, ckpt_row_mean: torch.Tensor | None) -> torch.Tensor:
    """Centering mean on ``w_u``'s device under ``--centering``: ``live`` is the
    full-vocabulary mean of the live LM head; ``trained`` is the checkpoint's
    stored ``row_mean``, else the mean over the tokenizer's text-token rows."""
    token_mask = None
    if mode == "trained":
        token_mask = token_mask_from_tokenizer(tokenizer, w_u.shape[0]).to(w_u.device)
    ckpt = {"row_mean": ckpt_row_mean} if ckpt_row_mean is not None else None
    return centering_mean(w_u, mode=mode, token_mask=token_mask, ckpt=ckpt).to(w_u.device)


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
    """Top-k codes of each target's centred, unit-normalised unembedding row.

    ``w_enc`` is the (d_features, d_model) encoder weight from ``load_sae``.
    The paper's run multiplied against its (d_model, d_features) transpose
    held contiguously, and gemm summation order follows the operand layout, so
    the same layout is handed to ``encode_topk`` (``w_enc.T.contiguous().T``
    is that layout viewed back) to reproduce the frozen bundles bit for bit
    rather than to the last float32 ulp.
    """
    encoder_w = w_enc.T.contiguous().T
    info: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for target in sorted(set(targets)):
        resolved = resolve_single_token(tokenizer, target)
        if resolved is None:
            skipped[target] = "multi_token"
            continue
        token_id, continuation = resolved
        row_norms, x = center_normalize_rows(w_u[token_id][None], row_mean)
        row_norm = row_norms[0]
        code = encode_topk(x, encoder_w, b_enc, k)[0]
        # Bundle columns are the k codes in descending order (torch.topk's order).
        values, indices = torch.topk(code, k=min(k, code.numel()))
        beta = row_norm * values
        recon = beta @ w_dec_unit[indices]
        row = w_u[token_id] - row_mean
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
    n_truncated = 0
    max_prompt_tokens = 0

    def pre_hook(_module, inputs):
        grabbed.append(inputs[0].detach())

    hook = lm_head.register_forward_pre_hook(pre_hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(kept), batch_size):
                batch = kept[start : start + batch_size]
                prompts = [row["prompt"] for row in batch]
                lengths = [len(ids) for ids in tokenizer(prompts, padding=False, truncation=False)["input_ids"]]
                n_truncated += sum(length > max_length for length in lengths)
                max_prompt_tokens = max(max_prompt_tokens, *lengths)
                enc = tokenizer(
                    prompts,
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
        "truncation": {
            "max_length": max_length,
            "truncation_side": tokenizer.truncation_side,
            "n_truncated_prompts": int(n_truncated),
            "max_prompt_tokens": int(max_prompt_tokens),
        },
    }


def method_matrices(bundle: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    hidden = bundle["hidden"].float().numpy()
    projection = bundle["projection"].float().numpy()
    contribution = bundle["contribution"].float().numpy()
    beta = bundle["beta"].float().numpy()
    return {
        "srp": l2_normalize(contribution),
        "support_projection": l2_normalize(projection),
        "hidden": l2_normalize(hidden),
        "shuffled_srp": l2_normalize(shuffled_srp(projection, beta, bundle["metadata"], seed)),
        "token_only": l2_normalize(beta),
    }


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
    splits = word_splits(metadata)
    metrics: dict[str, Any] = {"dataset": "coarsewsd20", "methods": {}, "comparisons": {}}
    srp_predictions: list[dict[str, Any]] = []
    per_word_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for method, matrix in matrices.items():
        per_word: dict[str, dict[str, float]] = {}
        for split in splits:
            centroids = []
            for sense in split.senses:
                centroid = matrix[split.train_idx[split.train_y == sense]].mean(axis=0)
                centroid /= max(np.linalg.norm(centroid), 1e-12)
                centroids.append(centroid)
            centroid_matrix = np.stack(centroids)
            scores = matrix[split.test_idx] @ centroid_matrix.T
            predictions = np.array([split.senses[i] for i in scores.argmax(axis=1)])
            cluster_pred = fit_train_clusters(
                matrix[split.train_idx],
                matrix[split.test_idx],
                n_clusters=len(split.senses),
                seed=seed,
            )
            per_word[split.word] = {
                "n_train": len(split.train_idx),
                "n_test": len(split.test_idx),
                "n_senses": len(split.senses),
                "accuracy": float(accuracy_score(split.test_y, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(split.test_y, predictions)),
                "macro_f1": float(f1_score(split.test_y, predictions, average="macro")),
                "pairwise_auc": sampled_pair_auc(matrix[split.test_idx], split.test_y, seed),
                "ari": float(adjusted_rand_score(split.test_y, cluster_pred)),
                "nmi": float(normalized_mutual_info_score(split.test_y, cluster_pred)),
                "row_relative_error": float(metadata[split.test_idx[0]]["row_relative_error"]),
            }
            if method == "srp":
                for local, idx in enumerate(split.test_idx):
                    srp_predictions.append(
                        {
                            "item_id": metadata[idx]["item_id"],
                            "target": split.word,
                            "gold_sense": str(split.test_y[local]),
                            "predicted_sense": str(predictions[local]),
                            "correct": int(split.test_y[local] == predictions[local]),
                            "target_rank": int(bundle["target_rank"][idx]),
                            "target_logprob": float(bundle["target_logprob"][idx]),
                        }
                    )
        aggregate: dict[str, Any] = {"n_words": len(per_word)}
        rng = np.random.default_rng(seed)
        word_names = sorted(per_word)
        for metric_name in METRIC_NAMES:
            values = np.array([per_word[word][metric_name] for word in word_names], dtype=float)
            values = values[np.isfinite(values)]
            aggregate[metric_name] = float(values.mean()) if len(values) else float("nan")
            aggregate[f"{metric_name}_ci"] = percentile_ci(bootstrap_mean(values, rng, n_boot) if len(values) else [])
        gated_words = [w for w in word_names if per_word[w]["row_relative_error"] < 0.5]
        aggregate["row_gate_words"] = len(gated_words)
        aggregate["row_gate_fraction"] = len(gated_words) / max(len(word_names), 1)
        aggregate["row_gate"] = {
            metric_name: (
                float(np.mean([per_word[w][metric_name] for w in gated_words])) if gated_words else float("nan")
            )
            for metric_name in METRIC_NAMES
        }
        metrics["methods"][method] = {"aggregate": aggregate, "per_word": per_word}
        per_word_by_method[method] = per_word

    # CoarseWSD words are the independent units. Bootstrap paired per-word
    # differences so the intervals answer whether SRP beats each control.
    for comparison_index, baseline in enumerate(COMPARISON_BASELINES):
        common_words = sorted(set(per_word_by_method["srp"]) & set(per_word_by_method[baseline]))
        comparison: dict[str, Any] = {"n_words": len(common_words)}
        rng = np.random.default_rng(seed + 200 + comparison_index)
        for metric_name in METRIC_NAMES:
            differences = np.array(
                [
                    per_word_by_method["srp"][word][metric_name] - per_word_by_method[baseline][word][metric_name]
                    for word in common_words
                ],
                dtype=float,
            )
            differences = differences[np.isfinite(differences)]
            comparison[f"{metric_name}_difference"] = float(differences.mean()) if len(differences) else float("nan")
            comparison[f"{metric_name}_difference_ci"] = percentile_ci(
                bootstrap_mean(differences, rng, n_boot) if len(differences) else []
            )
        metrics["comparisons"][f"srp_minus_{baseline}"] = comparison
    write_jsonl(out_dir / "predictions.jsonl", srp_predictions)
    return metrics


def summarize_scoring(bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    exact = bundle["exact_logit"].numpy()
    recon = bundle["reconstructed_logit"].numpy()
    ranks = bundle["target_rank"].numpy()
    summary = {
        "n_scored": len(metadata),
        "n_targets": len({row["target"] for row in metadata}),
        "median_target_rank": float(np.median(ranks)),
        "target_top1_fraction": float(np.mean(ranks == 1)),
        "target_top10_fraction": float(np.mean(ranks <= 10)),
        "median_absolute_logit_residual": float(np.median(np.abs(exact - recon))),
        "mean_absolute_logit_residual": float(np.mean(np.abs(exact - recon))),
        "row_gate_fraction_items": float(np.mean([row["row_relative_error"] < 0.5 for row in metadata])),
    }
    if "truncation" in bundle:
        summary["truncation"] = dict(bundle["truncation"])
    return summary


def self_test() -> int:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("coarsewsd20",), default="coarsewsd20")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="a bert-disambiguation checkout (or its data/CoarseWSD-20 dir)",
    )
    parser.add_argument("--model-id", help="Hugging Face model id")
    parser.add_argument("--checkpoint", type=Path, help="trained readout dictionary checkpoint (.pt)")
    parser.add_argument("--out-dir", type=Path, help="output dir: representations.pt, audit/scoring/metrics/run_config")
    parser.add_argument("--splits", default="train,test", help="comma-separated CoarseWSD-20 splits")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-word", type=int, default=None)
    parser.add_argument("--k", type=int, default=None, help="top-k codes per row; default: the checkpoint's k")
    parser.add_argument(
        "--centering",
        choices=("live", "trained"),
        default="live",
        help="row centering mean: live = full-vocabulary mean of the live LM head (the paper's runs); "
        "trained = the checkpoint's stored row_mean, else the tokenizer's text-token rows",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    required = (args.data_root, args.model_id, args.checkpoint, args.out_dir)
    if not args.analyze_only and not all(required):
        raise SystemExit("data-root, model-id, checkpoint, and out-dir are required")
    if args.analyze_only and args.out_dir is None:
        raise SystemExit("--out-dir is required with --analyze-only")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    representation_path = args.out_dir / "representations.pt"
    provenance = run_provenance(args)
    if args.analyze_only:
        bundle = load_bundle(representation_path)
    else:
        set_seed(args.seed)
        splits = [part.strip() for part in args.splits.split(",") if part.strip()]
        items, data_audit = load_coarsewsd(args.data_root, splits, args.limit, args.limit_per_word)
        log(f"loaded {len(items)} items ({data_audit})")
        log(f"loading {args.model_id}")
        model, tokenizer = load_model(args.model_id, args.device, args.dtype)
        lm_head = find_lm_head(model)
        w_u = lm_head.weight.detach().float().to(args.device)
        w_dec_unit, w_enc, b_enc, sae_config, ckpt_row_mean = load_sae(args.checkpoint)
        k = args.k if args.k is not None else checkpoint_k(args.checkpoint, sae_config)
        row_mean = scoring_row_mean(w_u, tokenizer, args.centering, ckpt_row_mean)
        w_dec_unit = w_dec_unit.to(args.device)
        w_enc = w_enc.to(args.device)
        b_enc = b_enc.to(args.device)
        if w_dec_unit.shape[1] != w_u.shape[1]:
            raise ValueError(f"dictionary/model width mismatch: {w_dec_unit.shape[1]} vs {w_u.shape[1]}")
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
        del w_enc, b_enc
        audit = {**data_audit, "tokenization_and_rows": token_audit, "k": k}
        write_json(audit, args.out_dir / "audit.json")
        log(f"single-token coverage {token_audit['n_targets_single_token']}/{token_audit['n_targets_requested']}")
        bundle = score_items(
            items,
            model,
            tokenizer,
            lm_head,
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
            "centering": args.centering,
        }
        torch.save(bundle, representation_path)
        write_json(summarize_scoring(bundle), args.out_dir / "scoring_summary.json")

    metrics = analyze_coarsewsd(bundle, args.out_dir, args.n_boot, args.seed)
    metrics["run"] = bundle["run"]
    metrics["scoring"] = summarize_scoring(bundle)
    metrics["provenance"] = provenance
    write_json(metrics, args.out_dir / "metrics.json")
    config = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "coarsewsd_template": COARSEWSD_TEMPLATE,
        "provenance": provenance,
    }
    write_json(config, args.out_dir / ("analysis_config.json" if args.analyze_only else "run_config.json"))
    log(f"wrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
