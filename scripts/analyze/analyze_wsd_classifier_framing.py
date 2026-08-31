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
the local score gate: sign preservation and
|residual| / (|centered exact score| + 0.5) < 0.5.

AmbiStory bundles go through the same encoding path (story display against
the two gloss-anchor displays); the paper uses CoarseWSD-20 only.

Reads the ``representations.pt`` bundles written by
``scripts/run/run_wsd_feature_alignment.py``; CPU-only, deterministic under
``--seed``. Paper run, once per model bundle (Qwen3.5-2B, Qwen3.5-9B,
DeepSeek-R1-Distill-Llama-8B), all values at their defaults:

    python scripts/analyze/analyze_wsd_classifier_framing.py \
        --bundle results/wsd_core/coarsewsd20_qwen2b/representations.pt \
        --out results/wsd_classifier_framing/coarsewsd20_qwen2b.json \
        --ks 5,10,20,50 --primary-k 10 --primary-encoding weighted \
        --n-boot 5000 --n-null-seeds 20 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


METHODS = ("srp", "projection_only", "shuffled_srp", "random_srp_features")
ENCODINGS = ("signed", "weighted")


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) + seed) % (2**32)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


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


def validate_feature_alignment(bundle: dict[str, Any]) -> None:
    metadata = bundle["metadata"]
    feature_ids = bundle["feature_ids"].numpy()
    first_by_target: dict[str, np.ndarray] = {}
    for row, ids in zip(metadata, feature_ids):
        target = str(row["target"])
        if target in first_by_target:
            if not np.array_equal(first_by_target[target], ids):
                raise ValueError(f"feature coordinates change within target {target!r}")
        else:
            first_by_target[target] = ids.copy()


def source_matrix(bundle: dict[str, Any], method: str, seed: int) -> np.ndarray:
    projection = bundle["projection"].float().numpy()
    contribution = bundle["contribution"].float().numpy()
    beta = bundle["beta"].float().numpy()
    metadata = bundle["metadata"]
    if method == "srp" or method == "random_srp_features":
        return contribution
    if method == "projection_only":
        return projection
    if method != "shuffled_srp":
        raise KeyError(method)
    shuffled = np.empty_like(contribution)
    by_target: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        by_target[str(row["target"])].append(i)
    for target, indices in by_target.items():
        rng = np.random.default_rng(stable_seed(target, seed))
        permutation = rng.permutation(beta.shape[1])
        idx = np.asarray(indices)
        shuffled[idx] = projection[idx] * beta[idx][:, permutation]
    return shuffled


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
    norms = np.linalg.norm(encoded, axis=1, keepdims=True)
    return encoded / np.maximum(norms, 1e-12)


def safe_spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) < 1e-8 or np.ptp(y) < 1e-8:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cluster_bootstrap_delta(
    rows: list[dict[str, Any]],
    cluster_key: str,
    statistic,
    n_boot: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sample: list[dict[str, Any]] = []
        for selected in rng.choice(keys, size=len(keys), replace=True):
            sample.extend(grouped[str(selected)])
        value = float(statistic(sample))
        if np.isfinite(value):
            values.append(value)
    return values


def confidence_interval(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def ambistory_rows(bundle: dict[str, Any], encoded: np.ndarray, gate: np.ndarray) -> list[dict[str, Any]]:
    metadata = bundle["metadata"]
    anchor_index = {
        (str(row["target"]), str(row["anchor_key"])): i for i, row in enumerate(metadata) if row["kind"] == "anchor"
    }
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(metadata):
        if row["kind"] != "story" or not gate[i]:
            continue
        anchor_ids = [anchor_index.get((str(row["target"]), str(key))) for key in row["anchor_keys"]]
        if any(index is None or not gate[int(index)] for index in anchor_ids):
            continue
        similarities = [float(encoded[i] @ encoded[int(index)]) for index in anchor_ids]
        human_margin = float(row["human_margin"])
        predicted_margin = similarities[0] - similarities[1]
        rows.append(
            {
                "item_id": str(row["item_id"]),
                "setup_id": str(row["setup_id"]),
                "target": str(row["target"]),
                "human_margin": human_margin,
                "predicted_margin": predicted_margin,
                "eligible_gap1": int(abs(human_margin) >= 1.0),
                "predicted_tie": int(abs(predicted_margin) < 1e-12),
                "correct": int(human_margin != 0 and np.sign(human_margin) == np.sign(predicted_margin)),
                "tie_half_score": (
                    0.5
                    if abs(predicted_margin) < 1e-12
                    else float(human_margin != 0 and np.sign(human_margin) == np.sign(predicted_margin))
                ),
            }
        )
    return rows


def summarize_ambistory_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible_gap1"]]
    eligible_non_ties = [row for row in eligible if not row["predicted_tie"]]
    non_ties = [row for row in rows if row["human_margin"] != 0]
    return {
        "n_contexts": len(rows),
        "n_setups": len({row["setup_id"] for row in rows}),
        "n_targets": len({row["target"] for row in rows}),
        "preference_spearman": safe_spearman(
            [row["human_margin"] for row in rows],
            [row["predicted_margin"] for row in rows],
        ),
        "preferred_sense_accuracy_gap1": (
            float(np.mean([row["tie_half_score"] for row in eligible])) if eligible else float("nan")
        ),
        "preferred_sense_accuracy_gap1_non_abstain": (
            float(np.mean([row["correct"] for row in eligible_non_ties])) if eligible_non_ties else float("nan")
        ),
        "preferred_sense_coverage_gap1": (len(eligible_non_ties) / len(eligible) if eligible else float("nan")),
        "n_gap1": len(eligible),
        "preferred_sense_accuracy_non_tie": (
            float(np.mean([row["correct"] for row in non_ties])) if non_ties else float("nan")
        ),
    }


def paired_ambistory_comparison(
    srp_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_id = {row["item_id"]: row for row in baseline_rows}
    paired = []
    for srp in srp_rows:
        baseline = baseline_by_id.get(srp["item_id"])
        if baseline is None:
            continue
        paired.append(
            {
                "setup_id": srp["setup_id"],
                "human_margin": srp["human_margin"],
                "eligible_gap1": srp["eligible_gap1"],
                "srp_margin": srp["predicted_margin"],
                "baseline_margin": baseline["predicted_margin"],
                "srp_correct": srp["correct"],
                "baseline_correct": baseline["correct"],
                "srp_tie_half_score": srp["tie_half_score"],
                "baseline_tie_half_score": baseline["tie_half_score"],
            }
        )

    def rho_delta(sample):
        human = [row["human_margin"] for row in sample]
        return safe_spearman(human, [row["srp_margin"] for row in sample]) - safe_spearman(
            human, [row["baseline_margin"] for row in sample]
        )

    eligible = [row for row in paired if row["eligible_gap1"]]

    def accuracy_delta(sample):
        return float(np.mean([row["srp_tie_half_score"] for row in sample])) - float(
            np.mean([row["baseline_tie_half_score"] for row in sample])
        )

    return {
        "n_contexts": len(paired),
        "preference_spearman_difference": float(rho_delta(paired)),
        "preference_spearman_difference_ci": confidence_interval(
            cluster_bootstrap_delta(paired, "setup_id", rho_delta, n_boot, seed)
        ),
        "preferred_sense_accuracy_gap1_difference": float(accuracy_delta(eligible)),
        "preferred_sense_accuracy_gap1_difference_ci": confidence_interval(
            cluster_bootstrap_delta(eligible, "setup_id", accuracy_delta, n_boot, seed + 1)
        ),
    }


def bootstrap_ambistory_method(rows: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible_gap1"]]

    def rho(sample):
        return safe_spearman(
            [row["human_margin"] for row in sample],
            [row["predicted_margin"] for row in sample],
        )

    def accuracy(sample):
        return float(np.mean([row["tie_half_score"] for row in sample]))

    return {
        **summarize_ambistory_rows(rows),
        "preference_spearman_ci": confidence_interval(cluster_bootstrap_delta(rows, "setup_id", rho, n_boot, seed)),
        "preferred_sense_accuracy_gap1_ci": confidence_interval(
            cluster_bootstrap_delta(eligible, "setup_id", accuracy, n_boot, seed + 1)
        ),
    }


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


def ambistory_null_seed_sensitivity(
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
        rho_values: list[float] = []
        accuracy_values: list[float] = []
        for offset in range(n_null_seeds):
            encoded = encode_top_features(bundle, method, primary_k, primary_encoding, seed + offset)
            summary = summarize_ambistory_rows(ambistory_rows(bundle, encoded, gate))
            rho_values.append(float(summary["preference_spearman"]))
            accuracy_values.append(float(summary["preferred_sense_accuracy_gap1"]))
        result[method] = {
            "preference_spearman": summarize_null_values(rho_values, float(observed["preference_spearman"])),
            "preferred_sense_accuracy_gap1": summarize_null_values(
                accuracy_values, float(observed["preferred_sense_accuracy_gap1"])
            ),
        }
    return result


def analyze_ambistory(
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
        "dataset": "ambistory",
        "question": "Do top contributing features identify the human-preferred sense?",
        "fidelity_gate": gate_summary,
        "results": {},
        "primary": {
            "encoding": primary_encoding,
            "k": primary_k,
            "methods": {},
            "comparisons": {},
        },
    }
    primary_rows: dict[str, list[dict[str, Any]]] = {}
    for encoding in ENCODINGS:
        metrics["results"][encoding] = {}
        for k in ks:
            metrics["results"][encoding][str(k)] = {}
            for method in METHODS:
                encoded = encode_top_features(bundle, method, k, encoding, seed)
                rows = ambistory_rows(bundle, encoded, gate)
                metrics["results"][encoding][str(k)][method] = summarize_ambistory_rows(rows)
                if encoding == primary_encoding and k == primary_k:
                    primary_rows[method] = rows
    for method_index, method in enumerate(METHODS):
        metrics["primary"]["methods"][method] = bootstrap_ambistory_method(
            primary_rows[method], n_boot, seed + 50 + method_index * 2
        )
    for comparison_index, baseline in enumerate(METHODS[1:]):
        metrics["primary"]["comparisons"][f"srp_minus_{baseline}"] = paired_ambistory_comparison(
            primary_rows["srp"],
            primary_rows[baseline],
            n_boot,
            seed + 100 + comparison_index * 10,
        )
    metrics["primary"]["null_seed_sensitivity"] = ambistory_null_seed_sensitivity(
        bundle,
        gate,
        primary_k,
        primary_encoding,
        n_null_seeds,
        seed,
        metrics["primary"]["methods"]["srp"],
    )
    return metrics


def coarse_per_word(bundle: dict[str, Any], encoded: np.ndarray, gate: np.ndarray) -> dict[str, dict[str, float]]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    metadata = bundle["metadata"]
    words = sorted({str(row["target"]) for row in metadata})
    per_word: dict[str, dict[str, float]] = {}
    for word in words:
        train_idx = [
            i for i, row in enumerate(metadata) if str(row["target"]) == word and row["split"] == "train" and gate[i]
        ]
        test_idx = [
            i for i, row in enumerate(metadata) if str(row["target"]) == word and row["split"] == "test" and gate[i]
        ]
        if not train_idx or not test_idx:
            continue
        train_y = np.asarray([str(metadata[i]["sense"]) for i in train_idx])
        test_y = np.asarray([str(metadata[i]["sense"]) for i in test_idx])
        senses = sorted(set(train_y))
        if len(senses) < 2 or not set(test_y).issubset(set(senses)):
            continue
        centroids = []
        for sense in senses:
            centroid = encoded[np.asarray(train_idx)[train_y == sense]].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            centroids.append(centroid)
        scores = encoded[test_idx] @ np.stack(centroids).T
        predictions = np.asarray([senses[index] for index in scores.argmax(axis=1)])
        per_word[word] = {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_senses": len(senses),
            "accuracy": float(accuracy_score(test_y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
            "macro_f1": float(f1_score(test_y, predictions, average="macro")),
        }
    return per_word


def summarize_coarse(per_word: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "n_words": len(per_word),
        "n_train": int(sum(row["n_train"] for row in per_word.values())),
        "n_test": int(sum(row["n_test"] for row in per_word.values())),
        **{
            metric: float(np.mean([row[metric] for row in per_word.values()]))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        },
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
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        differences = np.asarray([srp[word][metric] - baseline[word][metric] for word in words], dtype=float)
        boot = [float(rng.choice(differences, size=len(differences), replace=True).mean()) for _ in range(n_boot)]
        result[f"{metric}_difference"] = float(differences.mean())
        result[f"{metric}_difference_ci"] = confidence_interval(boot)
        result[f"{metric}_positive_words"] = int((differences > 0).sum())
    return result


def bootstrap_coarse_method(per_word: dict[str, dict[str, float]], n_boot: int, seed: int) -> dict[str, Any]:
    result = summarize_coarse(per_word)
    values = np.asarray([row["macro_f1"] for row in per_word.values()], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n_boot)]
    result["macro_f1_ci"] = confidence_interval(boot)
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
    validate_feature_alignment(bundle)
    print("self-test passed")
    return 0


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.bundle is None or args.out is None:
        raise SystemExit("--bundle and --out are required")
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    if args.primary_k not in ks:
        raise SystemExit("--primary-k must be included in --ks")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    validate_feature_alignment(bundle)
    dataset = str(bundle["run"]["dataset"])
    if dataset == "ambistory":
        metrics = analyze_ambistory(
            bundle,
            ks,
            args.primary_k,
            args.primary_encoding,
            args.n_boot,
            args.n_null_seeds,
            args.seed,
        )
    elif dataset == "coarsewsd20":
        metrics = analyze_coarse(
            bundle,
            ks,
            args.primary_k,
            args.primary_encoding,
            args.n_boot,
            args.n_null_seeds,
            args.seed,
        )
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")
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
    write_json(args.out, metrics)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
