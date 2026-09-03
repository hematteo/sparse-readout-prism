#!/usr/bin/env python3
"""Truncated-display sense classifier on CoarseWSD-20 readout bundles.

Backs the "classifier framing" paragraph of the sense-labelled evaluation
appendix (``app:sense-labelled-evaluation``). Per word, a nearest-centroid
classifier is fit on the train split over the ``--primary-k`` features with
the largest absolute value, retaining each displayed term's signed value
(``weighted``) or only its sign (``signed``), and evaluated on the held-out
split. Four sources are compared under the same encoding:

  srp                   signed contributions  z_{v,i} * p_i(h)
  projection_only       unweighted projections p_i(h)
  shuffled_srp          row coefficients permuted within the word (per-word seed)
  random_srp_features   k feature positions drawn at random per word

The paragraph quotes ``primary.methods.<source>.accuracy`` (projections
against contributions, with the shuffled and random sources alongside) at
``--primary-encoding weighted --primary-k 10``; ``balanced_accuracy`` and
``macro_f1`` are written next to it, with per-word bootstrap intervals,
paired per-word differences against the ``srp`` source and a null-seed
sensitivity for the two randomized sources. Secondary curves cover every
``--ks`` value under both encodings. Feature-level readings are restricted to
the local score criterion: sign preservation and
|residual| / (|centered exact score| + 0.5) < 0.5.

Reads the ``representations.pt`` bundles written by
``scripts/run/run_wsd_feature_alignment.py``; CPU-only, deterministic under
``--seed``. Paper run, once per model bundle (Qwen3.5-2B, Qwen3.5-9B,
DeepSeek-R1-Distill-Llama-8B), all values at their defaults:

    python scripts/analyze/analyze_wsd_classifier_framing.py \
        --bundle results/wsd_core/coarsewsd20_qwen2b/representations.pt \
        --out results/wsd_classifier_framing/coarsewsd20_qwen2b.json \
        --ks 5,10,20,50 --primary-k 10 --primary-encoding weighted \
        --n-boot 5000 --n-null-seeds 20 --seed 0

Fixed in 0.2.1: the AmbiStory bundle path, which the paper does not use, was
removed; the bundle reader, shuffle null and bootstrap helpers moved to
``sparse_readout_prism.research.wsd``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.research.wsd import (
    bootstrap_mean,
    l2_normalize,
    load_bundle,
    percentile_ci,
    shuffled_srp,
    stable_seed,
    word_splits,
)
from sparse_readout_prism.utils import write_json


METHODS = ("srp", "projection_only", "shuffled_srp", "random_srp_features")
ENCODINGS = ("signed", "weighted")
COARSE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def score_gate(bundle: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    contribution = bundle["contribution"].float().numpy()
    exact = bundle["exact_logit"].float().numpy()
    reconstructed = bundle["reconstructed_logit"].float().numpy()
    feature_sum = contribution.sum(axis=1)
    base = reconstructed - feature_sum
    centered_exact = exact - base
    residual = exact - reconstructed
    rho = np.abs(residual) / (np.abs(centered_exact) + 0.5)
    sign_match = np.sign(centered_exact) == np.sign(feature_sum)
    passed = sign_match & (rho < 0.5)
    return passed, {
        "n_items": int(len(passed)),
        "n_pass": int(passed.sum()),
        "pass_fraction": float(passed.mean()),
        "sign_match_fraction": float(sign_match.mean()),
        "median_rho_plus_0p5": float(np.median(rho)),
    }


def source_matrix(bundle: dict[str, Any], method: str, seed: int) -> np.ndarray:
    projection = bundle["projection"].float().numpy()
    contribution = bundle["contribution"].float().numpy()
    if method == "srp" or method == "random_srp_features":
        return contribution
    if method == "projection_only":
        return projection
    if method != "shuffled_srp":
        raise KeyError(method)
    return shuffled_srp(projection, bundle["beta"].float().numpy(), bundle["metadata"], seed)


def encode_top_features(
    bundle: dict[str, Any],
    method: str,
    k: int,
    encoding: str,
    seed: int,
) -> np.ndarray:
    matrix = source_matrix(bundle, method, seed)
    n_rows, width = matrix.shape
    if not 1 <= k <= width:
        raise ValueError(f"invalid k={k} for width={width}")
    selected = np.empty((n_rows, k), dtype=np.int64)
    if method == "random_srp_features":
        by_target: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(bundle["metadata"]):
            by_target[str(row["target"])].append(i)
        for target, rows in by_target.items():
            rng = np.random.default_rng(stable_seed(f"random:{target}", seed))
            choice = rng.permutation(width)[:k]
            selected[np.asarray(rows)] = choice[None, :]
    else:
        selected = np.argpartition(np.abs(matrix), width - k, axis=1)[:, -k:]

    encoded = np.zeros_like(matrix, dtype=np.float32)
    row_index = np.arange(n_rows)[:, None]
    values = matrix[row_index, selected]
    if encoding == "signed":
        values = np.sign(values)
    elif encoding != "weighted":
        raise KeyError(encoding)
    encoded[row_index, selected] = values
    return l2_normalize(encoded)


def summarize_null_values(values: list[float], observed: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n_seeds": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "q05_q95": [float(v) for v in np.quantile(array, [0.05, 0.95])],
        "observed_srp": float(observed),
        "fraction_null_at_least_observed": float(np.mean(array >= observed)),
    }


def coarse_per_word(bundle: dict[str, Any], encoded: np.ndarray, gate: np.ndarray) -> dict[str, dict[str, float]]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    per_word: dict[str, dict[str, float]] = {}
    for split in word_splits(bundle["metadata"], keep=gate):
        centroids = []
        for sense in split.senses:
            centroid = encoded[split.train_idx[split.train_y == sense]].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            centroids.append(centroid)
        scores = encoded[split.test_idx] @ np.stack(centroids).T
        predictions = np.asarray([split.senses[index] for index in scores.argmax(axis=1)])
        per_word[split.word] = {
            "n_train": len(split.train_idx),
            "n_test": len(split.test_idx),
            "n_senses": len(split.senses),
            "accuracy": float(accuracy_score(split.test_y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(split.test_y, predictions)),
            "macro_f1": float(f1_score(split.test_y, predictions, average="macro")),
        }
    return per_word


def summarize_coarse(per_word: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "n_words": len(per_word),
        "n_train": int(sum(row["n_train"] for row in per_word.values())),
        "n_test": int(sum(row["n_test"] for row in per_word.values())),
        **{metric: float(np.mean([row[metric] for row in per_word.values()])) for metric in COARSE_METRICS},
    }


def paired_word_comparison(
    srp: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    words = sorted(set(srp) & set(baseline))
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"n_words": len(words)}
    for metric in COARSE_METRICS:
        differences = np.asarray([srp[word][metric] - baseline[word][metric] for word in words], dtype=float)
        result[f"{metric}_difference"] = float(differences.mean())
        result[f"{metric}_difference_ci"] = percentile_ci(bootstrap_mean(differences, rng, n_boot))
        result[f"{metric}_positive_words"] = int((differences > 0).sum())
    return result


def bootstrap_coarse_method(per_word: dict[str, dict[str, float]], n_boot: int, seed: int) -> dict[str, Any]:
    result = summarize_coarse(per_word)
    values = np.asarray([row["macro_f1"] for row in per_word.values()], dtype=float)
    rng = np.random.default_rng(seed)
    result["macro_f1_ci"] = percentile_ci(bootstrap_mean(values, rng, n_boot))
    return result


def coarse_null_seed_sensitivity(
    bundle: dict[str, Any],
    gate: np.ndarray,
    primary_k: int,
    primary_encoding: str,
    n_null_seeds: int,
    seed: int,
    observed: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("shuffled_srp", "random_srp_features"):
        values: list[float] = []
        for offset in range(n_null_seeds):
            encoded = encode_top_features(bundle, method, primary_k, primary_encoding, seed + offset)
            summary = summarize_coarse(coarse_per_word(bundle, encoded, gate))
            values.append(float(summary["macro_f1"]))
        result[method] = {"macro_f1": summarize_null_values(values, float(observed["macro_f1"]))}
    return result


def analyze_coarse(
    bundle: dict[str, Any],
    ks: list[int],
    primary_k: int,
    primary_encoding: str,
    n_boot: int,
    n_null_seeds: int,
    seed: int,
) -> dict[str, Any]:
    gate, gate_summary = score_gate(bundle)
    metrics: dict[str, Any] = {
        "dataset": "coarsewsd20",
        "question": "Do top contributing features predict held-out word senses?",
        "fidelity_gate": gate_summary,
        "results": {},
        "primary": {
            "encoding": primary_encoding,
            "k": primary_k,
            "methods": {},
            "comparisons": {},
        },
    }
    primary_per_word: dict[str, dict[str, dict[str, float]]] = {}
    for encoding in ENCODINGS:
        metrics["results"][encoding] = {}
        for k in ks:
            metrics["results"][encoding][str(k)] = {}
            for method in METHODS:
                encoded = encode_top_features(bundle, method, k, encoding, seed)
                per_word = coarse_per_word(bundle, encoded, gate)
                metrics["results"][encoding][str(k)][method] = {
                    "aggregate": summarize_coarse(per_word),
                    "per_word": per_word,
                }
                if encoding == primary_encoding and k == primary_k:
                    primary_per_word[method] = per_word
    for method_index, method in enumerate(METHODS):
        metrics["primary"]["methods"][method] = bootstrap_coarse_method(
            primary_per_word[method], n_boot, seed + 150 + method_index
        )
    for comparison_index, baseline in enumerate(METHODS[1:]):
        metrics["primary"]["comparisons"][f"srp_minus_{baseline}"] = paired_word_comparison(
            primary_per_word["srp"],
            primary_per_word[baseline],
            n_boot,
            seed + 200 + comparison_index,
        )
    metrics["primary"]["null_seed_sensitivity"] = coarse_null_seed_sensitivity(
        bundle,
        gate,
        primary_k,
        primary_encoding,
        n_null_seeds,
        seed,
        metrics["primary"]["methods"]["srp"],
    )
    return metrics


def self_test() -> int:
    matrix = np.asarray([[1.0, -4.0, 3.0, 0.1]], dtype=np.float32)
    bundle = {
        "projection": torch.from_numpy(matrix),
        "contribution": torch.from_numpy(matrix),
        "beta": torch.ones_like(torch.from_numpy(matrix)),
        "feature_ids": torch.arange(4)[None],
        "metadata": [{"target": "bank"}],
    }
    signed = encode_top_features(bundle, "srp", 2, "signed", 0)
    assert signed[0, 1] < 0 and signed[0, 2] > 0
    assert signed[0, 0] == 0 and signed[0, 3] == 0
    assert np.isclose(np.linalg.norm(signed[0]), 1.0)
    weighted = encode_top_features(bundle, "srp", 2, "weighted", 0)
    assert abs(weighted[0, 1]) > abs(weighted[0, 2]) > 0
    print("self-test passed")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, help="representations.pt from run_wsd_feature_alignment.py")
    parser.add_argument("--out", type=Path, help="metrics JSON")
    parser.add_argument("--ks", default="5,10,20,50")
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--primary-encoding", choices=ENCODINGS, default="weighted")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-null-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.bundle is None or args.out is None:
        raise SystemExit("--bundle and --out are required")
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    if args.primary_k not in ks:
        raise SystemExit("--primary-k must be included in --ks")
    provenance = run_provenance(args)
    bundle = load_bundle(args.bundle)
    metrics = analyze_coarse(
        bundle,
        ks,
        args.primary_k,
        args.primary_encoding,
        args.n_boot,
        args.n_null_seeds,
        args.seed,
    )
    metrics["run"] = bundle["run"]
    metrics["analysis"] = {
        "bundle": str(args.bundle),
        "ks": ks,
        "primary_k": args.primary_k,
        "primary_encoding": args.primary_encoding,
        "n_boot": args.n_boot,
        "n_null_seeds": args.n_null_seeds,
        "seed": args.seed,
        "methods": list(METHODS),
        "encodings": list(ENCODINGS),
    }
    metrics["provenance"] = provenance
    write_json(metrics, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
