"""CoarseWSD-20 bundle and statistics helpers for the sense-labelled evaluation.

Shared by ``scripts/run/run_wsd_feature_alignment.py`` (writes the
``representations.pt`` bundle and the full-vector centroid references),
``scripts/analyze/analyze_wsd_sense_groups.py`` (``tab:app-sense-alignment``)
and ``scripts/analyze/analyze_wsd_classifier_framing.py`` (the classifier
framing paragraph). Before 0.2.1 each script carried its own copy of the
bundle reader, the per-word train/test split rule, the row-coefficient shuffle
null and the bootstrap helpers; the definitions here are those copies merged
without changing a number (``tests/test_wsd_sense_groups.py`` pins the
pre-merge outputs).

Bundle layout (``representations.pt``): ``metadata`` (one dict per scored
context: ``item_id``, ``split``, ``target``, ``sense``, ``token_id``,
``row_relative_error``, ...), ``run`` (dataset, model, checkpoint, k, ...) and
row-aligned tensors -- ``hidden`` (n, d_model), ``projection`` /
``contribution`` / ``beta`` (n, k) in the target's feature order,
``feature_ids`` (n, k), ``exact_logit`` / ``reconstructed_logit`` /
``target_logprob`` (n,) and ``target_rank`` (n,).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

BUNDLE_TENSOR_KEYS = (
    "hidden",
    "projection",
    "contribution",
    "beta",
    "feature_ids",
    "exact_logit",
    "reconstructed_logit",
    "target_logprob",
    "target_rank",
)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Read a ``representations.pt`` bundle and validate it.

    ``weights_only=True``: the bundle holds tensors, the metadata list of
    primitive dicts and the ``run`` dict, so it loads without unpickling code.
    """
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Raise if the bundle is not a row-aligned CoarseWSD-20 bundle.

    Checks the required keys, that every tensor has one row per metadata
    entry, that the bundle is a CoarseWSD-20 run (the AmbiStory mode was
    removed in 0.2.1) and that the feature coordinates do not change within a
    target -- the analyses compare columns across a word's rows.
    """
    missing = [key for key in ("metadata", "run", *BUNDLE_TENSOR_KEYS) if key not in bundle]
    if missing:
        raise KeyError(f"bundle is missing {missing}")
    n_rows = len(bundle["metadata"])
    for key in BUNDLE_TENSOR_KEYS:
        if int(bundle[key].shape[0]) != n_rows:
            raise ValueError(f"bundle[{key!r}] has {int(bundle[key].shape[0])} rows for {n_rows} metadata entries")
    dataset = bundle["run"].get("dataset")
    if dataset != "coarsewsd20":
        raise ValueError(f"unsupported dataset {dataset!r}: only CoarseWSD-20 bundles are supported")
    first_by_target: dict[str, np.ndarray] = {}
    for row, ids in zip(bundle["metadata"], bundle["feature_ids"].numpy()):
        target = str(row["target"])
        if target in first_by_target:
            if not np.array_equal(first_by_target[target], ids):
                raise ValueError(f"feature coordinates change within target {target!r}")
        else:
            first_by_target[target] = ids.copy()


@dataclass(frozen=True)
class WordSplit:
    """Row indices and sense labels of one word's train and test contexts."""

    word: str
    train_idx: np.ndarray  # (n_train,) bundle row indices
    test_idx: np.ndarray  # (n_test,)
    train_y: np.ndarray  # (n_train,) sense labels
    test_y: np.ndarray  # (n_test,)
    senses: list  # sorted train senses


def word_splits(metadata: list[dict[str, Any]], keep: np.ndarray | None = None) -> list[WordSplit]:
    """Per-word train/test splits in sorted word order, under the rule all three scripts apply.

    A word is skipped when either split is empty, when fewer than two senses
    occur in train, or when a test sense is absent from train. ``keep`` (one
    bool per bundle row) restricts the rows first; the classifier framing
    passes its score gate.
    """
    words = sorted({str(row["target"]) for row in metadata})
    splits: list[WordSplit] = []
    for word in words:
        train_idx = np.array(
            [
                i
                for i, row in enumerate(metadata)
                if str(row["target"]) == word and row["split"] == "train" and (keep is None or keep[i])
            ],
            dtype=np.int64,
        )
        test_idx = np.array(
            [
                i
                for i, row in enumerate(metadata)
                if str(row["target"]) == word and row["split"] == "test" and (keep is None or keep[i])
            ],
            dtype=np.int64,
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_y = np.array([str(metadata[i]["sense"]) for i in train_idx])
        test_y = np.array([str(metadata[i]["sense"]) for i in test_idx])
        senses = sorted(set(train_y))
        if len(senses) < 2 or not set(test_y).issubset(set(senses)):
            continue
        splits.append(WordSplit(word, train_idx, test_idx, train_y, test_y, senses))
    return splits


def stable_seed(text: str, seed: int) -> int:
    """Per-target RNG seed: ``(int(sha1(text)[:8], 16) + seed) % 2**32``.

    The classifier framing's form. The run script previously used
    ``seed + int(sha1(text)[:8], 16)`` without the modulus; the two agree for
    ``seed=0`` (the paper's runs) and whenever the sum stays below ``2**32``.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) + seed) % (2**32)


def shuffled_srp(projection: np.ndarray, beta: np.ndarray, metadata: list[dict[str, Any]], seed: int) -> np.ndarray:
    """Row-coefficient shuffle null: ``projection * beta[:, permutation]`` with one permutation per target.

    ``projection`` and ``beta`` are the bundle's (n, k) matrices; the
    permutation of each target's columns is drawn from
    ``default_rng(stable_seed(target, seed))``.
    """
    shuffled = np.empty_like(projection)
    by_target: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        by_target[str(row["target"])].append(i)
    for target in sorted(by_target):
        rng = np.random.default_rng(stable_seed(target, seed))
        permutation = rng.permutation(beta.shape[1])
        idx = np.asarray(by_target[target])
        shuffled[idx] = projection[idx] * beta[idx][:, permutation]
    return shuffled


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def safe_spearman(x: Any, y: Any) -> float:
    """Spearman rho, nan below three points or when either input's range is under 1e-8.

    Differs from ``utils.spearman`` only in the degeneracy guard: this one
    tests the range (``np.ptp < 1e-8``), ``utils.spearman`` the standard
    deviation (``< 1e-12``), so inputs varying by less than 1e-8 are nan here
    and a correlation there. Kept as the WSD scripts' guard so their outputs do
    not move.
    """
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) < 1e-8 or np.ptp(y) < 1e-8:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    cluster_key: str,
    stat_fn: Callable[[list[dict[str, Any]]], float],
    n_boot: int,
    seed: int,
) -> list[float]:
    """Resample clusters (rows grouped by ``cluster_key``) with replacement; non-finite statistics are dropped."""
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


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """``n_boot`` means of with-replacement resamples of ``values`` (the per-word bootstrap)."""
    return [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n_boot)]


def percentile_ci(values: Any) -> list[float]:
    """2.5 / 97.5 percentiles of a bootstrap sample; ``[nan, nan]`` when empty."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]
