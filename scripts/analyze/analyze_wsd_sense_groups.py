#!/usr/bin/env python3
"""Per-sense feature groups on CoarseWSD-20 readout bundles (``tab:app-sense-alignment``).

For each ambiguous word, select ``--group-size`` dictionary features per sense
on the train split, then assign each held-out context to the sense whose group
carries the larger summed contribution. Feature scales are standardized by
train-split statistics alone (``--raw-scale`` disables this); no test label
enters the selection. Summary fields, all balanced accuracy averaged over
words, and the table row each one backs:

  word_mean_balanced               Selected groups
  word_mean_null_balanced          Permutation null (train sense labels permuted
                                   ``--n-null`` times, the selection repeated each time)
  word_mean_majority_balanced      Majority
  words_beating_null_p95_balanced  Words above null p95
  words_beating_majority_balanced  Words above majority
  word_mean_full_account_balanced  Full account, no selection (nearest centroid on the
                                   full train-standardized contribution vector)
  word_mean_hidden_balanced        Hidden state (nearest centroid on the full
                                   train-standardized decoded state)

Per-word entries carry the same quantities plus stratified bootstrap intervals,
unbalanced accuracy, the selected feature positions and the local score criterion
(|exact - reconstructed| / (|exact| + 0.5) < 0.5) coverage.

``--selector mean_diff`` (default) ranks features by the train-mean contribution
gap between a sense and the other senses; ``top1_freq`` ranks by how often a
feature carries the largest contribution on that sense's train instances.
Senses are processed in descending train count and a position taken by an
earlier sense is skipped, so groups are disjoint.

``--basis`` swaps the dictionary contributions for a direct-W_U-geometry
account (``pca256`` / ``ridge128`` / ``knn128``, sized by ``--pca-rank`` /
``--neighbor-k`` / ``--ridge-lam``) under the identical selection and
prediction protocol, rebuilt from the bundle's stored decoded states with no
model forward pass. These need the unembedding matrix, read from ``--w-u`` (an
extraction payload with ``W_U_orig``, as written by
``scripts/data/extract_model_readout.py``) or, failing that, from the local
Hugging Face cache of ``--model-id`` at ``--revision`` (no download). Their
rows are centred under ``--centering`` (``live``, the default: the
full-vocabulary mean of that matrix; ``trained``: the ``--checkpoint``'s stored
``row_mean``, else the mean over the payload's ``token_mask`` or the
tokenizer's text-token rows). The paper's table reports ``srp`` only.

Reads the ``representations.pt`` bundles written by
``scripts/run/run_wsd_feature_alignment.py``; CPU-only, deterministic under
``--seed``. Paper run, once per model bundle (Qwen3.5-2B, Qwen3.5-9B,
DeepSeek-R1-Distill-Llama-8B), writing ``<out stem>_g{1,2,4,8}.json``:

    python scripts/analyze/analyze_wsd_sense_groups.py \
        --bundle results/wsd_core/coarsewsd20_qwen2b/representations.pt \
        --out results/wsd_sense_groups/coarsewsd20_qwen2b__srp.json \
        --group-sizes 1,2,4,8 --selector mean_diff --n-null 200 --n-boot 2000 --seed 0

The table reads the ``_g8`` file; the group-size sweep in the appendix prose
reads all four.

Fixed in 0.2.1: the cached Hugging Face weights are resolved through
``snapshot_download(local_files_only=True)`` at ``--revision`` (default: the
cached ``main`` ref) instead of the lexicographically last snapshot directory;
the direct-geometry centering mean goes through ``data.centering_mean`` under
``--centering``; the bundle reader and split rule moved to
``sparse_readout_prism.research.wsd``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparse_readout_prism.data import centering_mean, token_mask_from_tokenizer
from sparse_readout_prism.research.qwen_readout import load_sae
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.research.wsd import load_bundle, percentile_ci, word_splits
from sparse_readout_prism.utils import write_json


def load_wu_from_artifact(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unembedding matrix (V, d_model) fp32 and the ``token_mask`` (if stored) from an extraction payload."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    w = payload.get("W_U_orig", payload.get("W_U"))
    if w is None:
        raise KeyError(f"{path}: no 'W_U_orig'/'W_U' key (keys: {list(payload)})")
    token_mask = payload.get("token_mask")
    return w.float(), (token_mask.bool() if token_mask is not None else None)


def load_wu(model_id: str, revision: str | None = None) -> torch.Tensor:
    """LM-head weight (V, d_model) fp32 from the local Hugging Face cache, no download.

    ``huggingface_hub.snapshot_download(..., local_files_only=True)`` resolves
    ``revision`` (default: the cached ``main`` ref) to the snapshot directory;
    only the tensor is read from the safetensors shards, falling back to the
    tied embedding when no ``lm_head.weight`` exists (e.g. Qwen3.5-2B,
    ``tie_word_embeddings``).
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    shards = sorted(snap.glob("*.safetensors"))
    for suffix in ("lm_head.weight", "embed_tokens.weight"):
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    if key.endswith(suffix):
                        w = f.get_tensor(key).float()
                        print(f"[wu] {model_id}: {key} {tuple(w.shape)} from {shard.name}")
                        return w
    raise KeyError(f"no lm_head/embed_tokens weight in {snap}")


def geometry_contributions(
    basis: str,
    W: torch.Tensor,  # (V, d_model) fp32
    row_mean: torch.Tensor,  # (d_model,) centering mean, from data.centering_mean
    token_ids: list[int],
    hidden_by_word: dict[int, torch.Tensor],  # token_id -> (n_word, d_model) fp32
    pca_rank: int,
    neighbor_k: int,
    ridge_lam: float,
    torch_seed: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-word contribution matrices for a direct-geometry basis.

    Vectorized form of the baseline runner's methods in
    ``scripts/run/run_readout_baseline_comparisons.py`` (script classes, so
    they cannot be imported here): ``pca256`` mirrors ``DensePCAMethod``
    (``svd_lowrank`` on the centred, unit-normalised rows, q = rank + 16,
    niter = 4), ``ridge128`` mirrors ``NearestRowRidgeMethod`` (signed ridge fit
    on the top-k cosine neighbours, self excluded) and ``knn128`` mirrors
    ``KNNBasisMethod`` (the cosine-weighted neighbourhood mean rescaled by the
    target's projection onto it). Rows are centred on ``row_mean`` and
    unit-normalised, the target row is rebuilt in the basis, and
    contribution_j(h) = row_norm * coeff_j * (h . basis_j). Returns
    {token_id: {contribution (n, F), exact (n,), recon (n,)}}.
    """
    centered = W - row_mean
    row_norm_all = centered.norm(dim=1).clamp_min(1e-8)
    row_normalized_all = centered / row_norm_all[:, None]
    del centered

    v_pca = None
    if basis == "pca256":
        torch.manual_seed(torch_seed)
        q = min(pca_rank + 16, min(row_normalized_all.shape) - 1)
        _, _, vt = torch.svd_lowrank(row_normalized_all, q=q, niter=4)
        v_pca = vt[:, :pca_rank].T.contiguous()  # (pca_rank, d_model)

    out: dict[int, dict[str, np.ndarray]] = {}
    for tid in token_ids:
        x = row_normalized_all[tid]  # (d_model,)
        row_norm = row_norm_all[tid]
        w_row = W[tid]
        h = hidden_by_word[tid]  # (n, d_model)
        if basis == "pca256":
            coeffs = v_pca @ x  # (pca_rank,)
            contribution = row_norm * (h @ v_pca.T) * coeffs  # (n, pca_rank)
        else:
            sims = row_normalized_all @ x  # (V,)
            sims[tid] = -float("inf")  # exclude self
            top = torch.topk(sims, k=neighbor_k)
            basis_rows = row_normalized_all[top.indices]  # (neighbor_k, d_model)
            if basis == "ridge128":
                a = basis_rows @ basis_rows.T + ridge_lam * torch.eye(neighbor_k)
                coeffs = torch.linalg.solve(a, basis_rows @ x)  # (neighbor_k,)
            elif basis == "knn128":
                w = top.values.clamp_min(0.0)
                w = w / w.sum().clamp_min(1e-8)
                mean_nbr = w @ basis_rows  # (d_model,)
                gamma = (x @ mean_nbr) / (mean_nbr @ mean_nbr).clamp_min(1e-8)
                coeffs = gamma * w
            else:
                raise ValueError(f"unknown basis {basis}")
            contribution = row_norm * (h @ basis_rows.T) * coeffs  # (n, neighbor_k)
        exact = h @ w_row  # (n,)
        recon = h @ row_mean + contribution.sum(dim=1)  # (n,)
        out[tid] = {
            "contribution": contribution.numpy(),
            "exact": exact.numpy(),
            "recon": recon.numpy(),
        }
    return out


def select_anchors(
    train_contribution: np.ndarray,
    senses: list[str],
    train_y: np.ndarray,
    selector: str = "mean_diff",
    group_size: int = 1,
) -> dict[str, list[int]]:
    """Greedy per-sense feature-group assignment (group_size=1: one feature).

    selector="top1_freq": the feature most often carrying the largest
    contribution on that sense's train instances.
    selector="mean_diff": the feature with the largest train-mean contribution
    gap between this sense and the other senses.
    Senses are processed in descending train count; a feature position taken by
    an earlier sense is skipped. Ties break by position index for determinism.
    """
    order = sorted(senses, key=lambda s: (-int((train_y == s).sum()), s))
    taken: set[int] = set()
    anchors: dict[str, list[int]] = {}
    top1 = train_contribution.argmax(axis=1)
    for sense in order:
        mask = train_y == sense
        if selector == "top1_freq":
            counts = Counter(top1[mask].tolist())
            ranked = [pos for pos, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        else:
            gap = train_contribution[mask].mean(axis=0) - train_contribution[~mask].mean(axis=0)
            ranked = np.argsort(-gap).tolist()
        group: list[int] = []
        for pos in ranked:
            if pos not in taken:
                group.append(int(pos))
                taken.add(int(pos))
                if len(group) >= group_size:
                    break
        if not group:  # every candidate taken: fall back to first free position
            free = [p for p in range(train_contribution.shape[1]) if p not in taken]
            group = [free[0] if free else int(top1[mask][0])]
            taken.add(group[0])
        anchors[sense] = group
    return anchors


def word_accuracy(
    contribution: np.ndarray, anchors: dict[str, list[int]], senses: list[str], gold: np.ndarray
) -> tuple[float, np.ndarray]:
    anchor_matrix = np.stack([contribution[:, anchors[s]].sum(axis=1) for s in senses], axis=1)  # (n, n_senses)
    predictions = np.array([senses[i] for i in anchor_matrix.argmax(axis=1)])
    return float((predictions == gold).mean()), predictions


def balanced_accuracy(predictions: np.ndarray, gold: np.ndarray) -> float:
    """Mean per-sense recall over the senses present in gold."""
    recalls = [float((predictions[gold == s] == s).mean()) for s in np.unique(gold)]
    return float(np.mean(recalls))


def ncm_balanced(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, test_y: np.ndarray) -> float:
    """Nearest class-centroid on train-standardized features: the
    no-selection reference (how much sense information the representation
    carries when the whole account is used)."""
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0) + 1e-6
    ztr = (train_x - mu) / sd
    zte = (test_x - mu) / sd
    senses = sorted(set(train_y))
    cents = np.stack([ztr[train_y == s].mean(axis=0) for s in senses])
    dist = ((zte[:, None, :] - cents[None]) ** 2).sum(axis=-1)  # (n_test, n_senses)
    pred = np.array([senses[i] for i in dist.argmin(axis=1)])
    return balanced_accuracy(pred, test_y)


def geometry_row_mean(
    W: torch.Tensor,
    mode: str,
    *,
    token_mask: torch.Tensor | None,
    checkpoint: Path | None,
    model_id: str,
    revision: str | None,
) -> torch.Tensor:
    """``--centering`` for the direct-geometry controls, via ``data.centering_mean``.

    ``live``: the full-vocabulary mean of ``W``. ``trained``: the checkpoint's
    stored ``row_mean`` when ``--checkpoint`` is given, else the mean over
    ``token_mask`` (the ``--w-u`` payload's, else built from ``model_id``'s
    tokenizer in the local cache).
    """
    ckpt = None
    if mode == "trained":
        if checkpoint is not None:
            *_rest, ckpt_row_mean = load_sae(checkpoint)
            ckpt = {"row_mean": ckpt_row_mean} if ckpt_row_mean is not None else None
        if token_mask is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
            token_mask = token_mask_from_tokenizer(tok, W.shape[0])
    return centering_mean(W, mode=mode, token_mask=token_mask, ckpt=ckpt)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bundle", type=Path, required=True, help="representations.pt from run_wsd_feature_alignment.py"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="summary JSON (with --group-sizes: <stem>_g<k>.json each)"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selector", choices=["top1_freq", "mean_diff"], default="mean_diff")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument(
        "--group-sizes",
        default=None,
        help="comma list (e.g. 1,2,4,8): sweep group sizes in one run, writing <out>_g<k>.json each; "
        "shares the basis/W_U preparation across sizes",
    )
    parser.add_argument(
        "--basis",
        choices=["srp", "pca256", "ridge128", "knn128"],
        default="srp",
        help="srp: bundle contributions; others: direct-W_U geometry controls",
    )
    parser.add_argument("--model-id", default=None, help="override bundle run.model_id for --basis controls")
    parser.add_argument("--revision", default=None, help="HF revision of --model-id in the local cache (default: main)")
    parser.add_argument(
        "--w-u",
        type=Path,
        default=None,
        help="extraction payload (.pt with W_U_orig) for --basis controls; default: local HF cache of --model-id",
    )
    parser.add_argument(
        "--centering",
        choices=("live", "trained"),
        default="live",
        help="row centering for --basis controls: live = full-vocabulary mean of W_U (the paper's runs); "
        "trained = --checkpoint's stored row_mean, else the token_mask mean",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="dictionary checkpoint whose row_mean --centering trained uses"
    )
    parser.add_argument("--pca-rank", type=int, default=256)
    parser.add_argument("--neighbor-k", type=int, default=128)
    parser.add_argument("--ridge-lam", type=float, default=1e-3)
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="skip train-statistic standardization of contributions (default: standardize)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provenance = run_provenance(args)

    bundle = load_bundle(args.bundle)
    metadata = bundle["metadata"]
    contribution = bundle["contribution"].float().numpy()  # (n_items, k_support)
    exact = bundle["exact_logit"].numpy()
    recon = bundle["reconstructed_logit"].numpy()

    if args.basis != "srp":
        model_id = args.model_id or bundle["run"]["model_id"]
        token_mask = None
        if args.w_u is not None:
            wu, token_mask = load_wu_from_artifact(args.w_u)
        else:
            wu = load_wu(model_id, args.revision)
        row_mean = geometry_row_mean(
            wu,
            args.centering,
            token_mask=token_mask,
            checkpoint=args.checkpoint,
            model_id=model_id,
            revision=args.revision,
        )
        hidden = bundle["hidden"].float()  # (n_items, d_model)
        word_tid: dict[str, int] = {}
        word_idx: dict[str, list[int]] = {}
        for i, row in enumerate(metadata):
            word_tid[row["target"]] = int(row["token_id"])
            word_idx.setdefault(row["target"], []).append(i)
        hidden_by_word = {word_tid[w]: hidden[np.array(idx)] for w, idx in word_idx.items()}
        geo = geometry_contributions(
            args.basis,
            wu,
            row_mean,
            [word_tid[w] for w in sorted(word_idx)],
            hidden_by_word,
            args.pca_rank,
            args.neighbor_k,
            args.ridge_lam,
            args.seed,
        )
        n_feat = args.pca_rank if args.basis == "pca256" else args.neighbor_k
        contribution = np.zeros((len(metadata), n_feat), dtype=np.float32)
        geo_exact = np.zeros(len(metadata), dtype=np.float32)
        recon = np.zeros(len(metadata), dtype=np.float32)
        for w, idx in word_idx.items():
            g = geo[word_tid[w]]
            contribution[np.array(idx)] = g["contribution"]
            geo_exact[np.array(idx)] = g["exact"]
            recon[np.array(idx)] = g["recon"]
        # W_U extraction sanity: fp32 rebuild of h.w_t vs the runtime logit
        exact_gap = float(np.max(np.abs(geo_exact - exact)))
        print(f"[wu] max |exact(rebuilt) - exact(bundle)| = {exact_gap:.4f}")
        exact = geo_exact

    rho = np.abs(exact - recon) / (np.abs(exact) + 0.5)  # floored rho, paper convention
    gate = rho < 0.5

    group_sizes = [int(x) for x in args.group_sizes.split(",")] if args.group_sizes else [args.group_size]

    hidden_all = bundle["hidden"].float().numpy()  # (n_items, d_model)
    word_cache: dict[str, dict] = {}
    for split in word_splits(metadata):
        train_idx, test_idx = split.train_idx, split.test_idx
        train_y, test_y = split.train_y, split.test_y
        train_c = contribution[train_idx]
        test_c = contribution[test_idx]
        if not args.raw_scale:
            # Features have incommensurate contribution scales; standardize by
            # train-split statistics (identically for every basis, no leakage)
            # so the cross-feature argmax compares deviations, not magnitudes.
            mu = train_c.mean(axis=0)
            sd = train_c.std(axis=0) + 1e-6
            train_c = (train_c - mu) / sd
            test_c = (test_c - mu) / sd
        word_cache[split.word] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_y": train_y,
            "test_y": test_y,
            "senses": split.senses,
            "train_c": train_c,
            "test_c": test_c,
            # basis-independent references, computed once per word
            "full_account_balanced": ncm_balanced(contribution[train_idx], contribution[test_idx], train_y, test_y),
            "hidden_balanced": ncm_balanced(hidden_all[train_idx], hidden_all[test_idx], train_y, test_y),
        }

    for group_size in group_sizes:
        run_group(args, group_size, word_cache, gate, provenance)
    return 0


def run_group(args, group_size: int, word_cache: dict, gate: np.ndarray, provenance: dict[str, Any]) -> None:
    rng = np.random.default_rng(args.seed)
    per_word: dict[str, dict] = {}
    pooled_correct: list[int] = []
    pooled_correct_gated: list[int] = []

    for word, wc in word_cache.items():
        train_idx, test_idx = wc["train_idx"], wc["test_idx"]
        train_y, test_y = wc["train_y"], wc["test_y"]
        senses = wc["senses"]
        train_c, test_c = wc["train_c"], wc["test_c"]

        anchors = select_anchors(train_c, senses, train_y, args.selector, group_size)
        acc, predictions = word_accuracy(test_c, anchors, senses, test_y)
        bal_acc = balanced_accuracy(predictions, test_y)
        majority = max(senses, key=lambda s: int((train_y == s).sum()))
        majority_acc = float((test_y == majority).mean())
        majority_bal = balanced_accuracy(np.full_like(test_y, majority), test_y)

        # bootstrap CI over test instances (stratified by sense for balanced acc)
        boot = np.empty(args.n_boot)
        boot_bal = np.empty(args.n_boot)
        correct = (predictions == test_y).astype(float)
        sense_pools = {s: np.flatnonzero(test_y == s) for s in np.unique(test_y)}
        for b in range(args.n_boot):
            sample = rng.integers(0, len(correct), len(correct))
            boot[b] = correct[sample].mean()
            strat = np.concatenate([pool[rng.integers(0, len(pool), len(pool))] for pool in sense_pools.values()])
            boot_bal[b] = balanced_accuracy(predictions[strat], test_y[strat])
        ci = percentile_ci(boot)
        bal_ci = percentile_ci(boot_bal)

        # label-shuffle null: re-select anchors under permuted train labels
        null_accs = np.empty(args.n_null)
        null_bals = np.empty(args.n_null)
        for n in range(args.n_null):
            shuffled_y = rng.permutation(train_y)
            null_anchors = select_anchors(train_c, senses, shuffled_y, args.selector, group_size)
            null_accs[n], null_pred = word_accuracy(test_c, null_anchors, senses, test_y)
            null_bals[n] = balanced_accuracy(null_pred, test_y)

        test_gate = gate[test_idx]
        gated_acc = float(correct[test_gate].mean()) if test_gate.any() else float("nan")
        gated_bal = balanced_accuracy(predictions[test_gate], test_y[test_gate]) if test_gate.any() else float("nan")
        pooled_correct.extend(correct.astype(int).tolist())
        pooled_correct_gated.extend(correct[test_gate].astype(int).tolist())

        per_word[word] = {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_senses": len(senses),
            "anchors": {s: [int(p) for p in anchors[s]] for s in senses},
            "accuracy": acc,
            "accuracy_ci": ci,
            "balanced_accuracy": bal_acc,
            "balanced_accuracy_ci": bal_ci,
            "majority_accuracy": majority_acc,
            "majority_balanced_accuracy": majority_bal,
            "chance_balanced_accuracy": 1.0 / len(senses),
            "null_accuracy_mean": float(null_accs.mean()),
            "null_accuracy_p95": float(np.percentile(null_accs, 95)),
            "null_balanced_mean": float(null_bals.mean()),
            "null_balanced_p95": float(np.percentile(null_bals, 95)),
            "beats_null_p95": bool(acc > np.percentile(null_accs, 95)),
            "beats_null_p95_balanced": bool(bal_acc > np.percentile(null_bals, 95)),
            "gate_fraction": float(test_gate.mean()),
            "gated_accuracy": gated_acc,
            "gated_balanced_accuracy": gated_bal,
            "full_account_balanced": wc["full_account_balanced"],
            "hidden_balanced": wc["hidden_balanced"],
        }

    names = sorted(per_word)
    word_acc = np.array([per_word[w]["accuracy"] for w in names])
    word_bal = np.array([per_word[w]["balanced_accuracy"] for w in names])
    word_majority = np.array([per_word[w]["majority_accuracy"] for w in names])
    word_majority_bal = np.array([per_word[w]["majority_balanced_accuracy"] for w in names])
    word_chance_bal = np.array([per_word[w]["chance_balanced_accuracy"] for w in names])
    word_null = np.array([per_word[w]["null_accuracy_mean"] for w in names])
    word_null_bal = np.array([per_word[w]["null_balanced_mean"] for w in names])
    boot_mean = np.empty(args.n_boot)
    boot_mean_bal = np.empty(args.n_boot)
    for b in range(args.n_boot):
        sample = rng.integers(0, len(word_acc), len(word_acc))
        boot_mean[b] = word_acc[sample].mean()
        boot_mean_bal[b] = word_bal[sample].mean()

    summary = {
        "bundle": str(args.bundle),
        "basis": args.basis,
        "selector": args.selector,
        "group_size": group_size,
        "seed": args.seed,
        "n_words": len(names),
        "word_mean_accuracy": float(word_acc.mean()),
        "word_mean_accuracy_ci": percentile_ci(boot_mean),
        "standardized": not args.raw_scale,
        "word_mean_balanced": float(word_bal.mean()),
        "word_mean_balanced_ci": percentile_ci(boot_mean_bal),
        "word_mean_majority": float(word_majority.mean()),
        "word_mean_majority_balanced": float(word_majority_bal.mean()),
        "word_mean_chance_balanced": float(word_chance_bal.mean()),
        "word_mean_null": float(word_null.mean()),
        "word_mean_null_balanced": float(word_null_bal.mean()),
        "words_beating_null_p95": int(sum(per_word[w]["beats_null_p95"] for w in names)),
        "words_beating_null_p95_balanced": int(sum(per_word[w]["beats_null_p95_balanced"] for w in names)),
        "words_beating_majority_balanced": int(
            sum(per_word[w]["balanced_accuracy"] > per_word[w]["majority_balanced_accuracy"] for w in names)
        ),
        "word_mean_full_account_balanced": float(np.mean([per_word[w]["full_account_balanced"] for w in names])),
        "word_mean_hidden_balanced": float(np.mean([per_word[w]["hidden_balanced"] for w in names])),
        "pooled_accuracy": float(np.mean(pooled_correct)),
        "pooled_gated_accuracy": float(np.mean(pooled_correct_gated)),
        "gate_fraction_overall": float(len(pooled_correct_gated) / max(len(pooled_correct), 1)),
        "per_word": per_word,
        "provenance": provenance,
    }

    out = args.out if args.group_sizes is None else args.out.with_name(f"{args.out.stem}_g{group_size}.json")
    write_json(summary, out, atomic=True)
    print(
        f"{args.bundle.parent.name} [{args.basis} g={group_size}]: words={summary['n_words']} "
        f"bal acc={summary['word_mean_balanced']:.3f} "
        f"CI[{summary['word_mean_balanced_ci'][0]:.3f},{summary['word_mean_balanced_ci'][1]:.3f}] "
        f"(chance={summary['word_mean_chance_balanced']:.3f} "
        f"null={summary['word_mean_null_balanced']:.3f} "
        f"beat_null={summary['words_beating_null_p95_balanced']}/{summary['n_words']}) | "
        f"raw acc={summary['word_mean_accuracy']:.3f} "
        f"majority={summary['word_mean_majority']:.3f} "
        f"gated acc={summary['pooled_gated_accuracy']:.3f} "
        f"(gate {summary['gate_fraction_overall']:.2f}) | "
        f"full acct={summary['word_mean_full_account_balanced']:.3f} "
        f"hidden={summary['word_mean_hidden_balanced']:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
