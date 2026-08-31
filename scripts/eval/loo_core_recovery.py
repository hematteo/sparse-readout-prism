#!/usr/bin/env python3
"""Held-out recovery of the stable grouping core (Appendix I, ``tab:app-loo-core-recovery``).

Which methods recover reproducible grouping structure, on identical targets?
For each curated contrast and each side, every seed's dictionary yields a side
token-set (top-M features per sign, top-R centred rows per feature). For each
held-out seed ``i`` the core is the set of tokens present in the side sets of
ALL remaining seeds (leave-one-out construction; with three seeds, 2 of 2), and

    srp_loo_recall      = |core_i & set_i| / |core_i|     held-out dictionary, same recipe
    knn_loo_recall      = |core_i & K|     / |core_i|     weighted-kNN side set on the rows
    cluster_loo_recall  = |core_i & CL|    / |core_i|     hard k-means clustering of the rows
    reference recall    = |core_i & R|     / |core_i|     optional dictionary from another recipe

all scored on the same cores. The kNN side set of a target row is its top
``--side-n`` cosine neighbours among the centred, normalised rows; the cluster
side set is the target's k-means cluster (spherical k-means with
``--n-clusters`` centroids, ``--kmeans-iters`` iterations, seeded), truncated to
the ``--side-n`` members closest to the centroid. 95% CIs are a
contrast-clustered bootstrap (resample contrasts, keep all their cells,
``--n-boot`` replicates) for each method's mean recall and for the SRP/kNN ratio.

Two conventions of the paper's run are kept and made explicit:

* The kNN side set excludes the target row itself (``--knn-exclude-self``,
  the default). The SRP side sets (top-R rows of the contrast's features) and
  the cluster side set (the target's cluster members) do not exclude it, so
  the target token can sit in a core and be recoverable by SRP and by the
  cluster control but never by kNN. ``--no-knn-exclude-self`` admits the
  target's own row (cosine 1, so it always takes one of the ``--side-n``
  slots).
* Cluster membership is the assignment that produced the last k-means update
  (``spherical_kmeans_unit(..., final_assignment=False)``), one Lloyd step
  behind the centroids the members are ranked against. The ``row_cluster_*``
  baselines of the direct-geometry grid assign against the final centroids.

``--centering {live,trained}`` selects the centering mean: ``live`` (default)
is the full-vocabulary mean of the payload's ``W_U``, how the paper's runs were
computed; ``trained`` is the dictionaries' stored training mean. The output
JSON carries a ``provenance`` block (command line, args, git hash, versions).

Dictionary-family discipline: ``--dictionaries`` must come from ONE recipe (the
seed-variation family trained from ``configs/sweeps/qwen35_2b_seedvar_base.yaml``);
the script refuses dictionaries that differ in width or ``k``. A dictionary
from a different recipe enters only through ``--reference-dict`` and is scored
on the same cores without contributing to them (Appendix I, cross-recipe
disclosure).

Paper runs (Qwen3.5-2B, three seeds per width, curated A/B bank, all defaults):

    uv run python scripts/eval/loo_core_recovery.py \\
        --dictionaries results/qwen35_2b_seedvar/qwen2b_d65536_k256_s0/checkpoint.pt \\
                       results/qwen35_2b_seedvar/qwen2b_d65536_k256_s1/checkpoint.pt \\
                       results/qwen35_2b_seedvar/qwen2b_d65536_k256_s2/checkpoint.pt \\
        --w-u "$SRP_ARCHIVE_ROOT/data/qwen35-2b/qwen35_2b.pt" \\
        --bank data/query_banks/qwen_gemma_result1_curated_ab.jsonl \\
        --width-tag 32x \\
        --reference-dict "$SRP_ARCHIVE_ROOT/results/qwen35_2b_sae/converge/topk_d65536_32x_k256_rowseeded_hybrid_lamramp_s0/checkpoint.pt" \\
        --out results/seed_stability/loo_core_recovery_32x.json

    (16x: the ``qwen2b_d32768_k128_s{0,1,2}`` checkpoints, ``--width-tag 16x``,
    no ``--reference-dict``.)

The k-means fit over the full vocabulary wants a GPU (or Apple MPS); the rest
runs on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.research.row_geometry import spherical_kmeans_unit
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.research.seed_stability import (
    center_rows,
    contrast_features,
    feature_token_set,
    load_contrast_pairs,
    load_dictionary,
    load_readout,
    resolve_centering,
    token_strings,
)
from sparse_readout_prism.utils import resolve_device, to_jsonable


def knn_side_sets(
    W_n: torch.Tensor, pairs: list[tuple], side_n: int, tok, cache: dict[int, str], *, exclude_self: bool
) -> list[tuple[set[str], set[str]]]:
    """Per contrast, the ``(A, B)`` token sets of each target's top ``side_n`` cosine neighbours."""
    out = []
    for _a, _b, ia, ib in pairs:
        sides = []
        for i in (ia, ib):
            sims = W_n @ W_n[i]
            if exclude_self:
                sims[i] = -1
            sides.append(token_strings(tok, torch.topk(sims, side_n).indices.tolist(), cache))
        out.append(tuple(sides))
    return out


def run(
    dicts: list,
    W: torch.Tensor,
    tok,
    pairs: list[tuple],
    device,
    *,
    row_mean: torch.Tensor,
    top_m: int,
    top_r: int,
    side_n: int,
    n_clusters: int,
    kmeans_iters: int,
    kmeans_seed: int,
    n_boot: int,
    bootstrap_seed: int,
    reference_dict=None,
    width_tag: str | None,
    knn_exclude_self: bool = True,
) -> dict:
    W_c, rn, W_n = center_rows(W, row_mean)
    cache: dict[int, str] = {}
    n_seeds = len(dicts)

    def dict_side_sets(d) -> list[tuple[set, set]]:
        out = []
        for _a, _b, ia, ib in pairs:
            top_pos, top_neg = contrast_features(W_n, rn, d, ia, ib, top_m)
            out.append(
                (
                    feature_token_set(top_pos, d.decoder, W_c, tok, top_r, cache),
                    feature_token_set(top_neg, d.decoder, W_c, tok, top_r, cache),
                )
            )
        return out

    side_sets = []
    for si, d in enumerate(dicts):
        side_sets.append(dict_side_sets(d))
        print(f"seed {si} side sets done", flush=True)

    knn_sets = knn_side_sets(W_n, pairs, side_n, tok, cache, exclude_self=knn_exclude_self)

    ref_sets = None
    if reference_dict is not None:
        ref_sets = dict_side_sets(reference_dict)
        print("reference dictionary side sets done", flush=True)

    print(f"fitting k-means n_clusters={n_clusters} on {device} ...", flush=True)
    C, assign = spherical_kmeans_unit(
        W_n.to(device), n_clusters, kmeans_seed, iters=kmeans_iters, return_assignments=True, final_assignment=False
    )
    C, assign = C.cpu(), assign.cpu()

    def cluster_side(i: int) -> set[str]:
        members = (assign == assign[i]).nonzero().flatten()
        if len(members) > side_n:
            sims = W_n[members] @ C[assign[i]]
            members = members[torch.topk(sims, side_n).indices]
        return token_strings(tok, members.tolist(), cache)

    cluster_sets = [(cluster_side(ia), cluster_side(ib)) for _a, _b, ia, ib in pairs]
    print("cluster side sets done", flush=True)

    srp_loo, knn_loo, cl_loo, ref_loo, core_sizes, cell_ci = [], [], [], [], [], []
    for ci in range(len(pairs)):
        for side in (0, 1):
            sets = [side_sets[si][ci][side] for si in range(n_seeds)]
            K = knn_sets[ci][side]
            CL = cluster_sets[ci][side]
            for held in range(n_seeds):
                others = [sets[j] for j in range(n_seeds) if j != held]
                core = set.intersection(*others)
                if not core:
                    continue
                cell_ci.append(ci)
                core_sizes.append(len(core))
                srp_loo.append(len(core & sets[held]) / len(core))
                knn_loo.append(len(core & K) / len(core))
                cl_loo.append(len(core & CL) / len(core))
                if ref_sets is not None:
                    ref_loo.append(len(core & ref_sets[ci][side]) / len(core))

    def m(v):
        return round(float(np.mean(v)), 4) if v else None

    # contrast-clustered bootstrap CIs (resample contrasts, keep all their cells)
    cell_idx = np.array(cell_ci)
    arrs = dict(srp=np.array(srp_loo), knn=np.array(knn_loo), cluster=np.array(cl_loo))
    brng = np.random.default_rng(bootstrap_seed)
    n_c = len(pairs)
    boots: dict[str, list[float]] = {k: [] for k in arrs}
    boots["ratio_srp_knn"] = []
    for _ in range(n_boot):
        samp = brng.integers(0, n_c, n_c)
        mask_counts = np.bincount(samp, minlength=n_c)
        w = mask_counts[cell_idx].astype(float)
        if w.sum() == 0:
            continue
        means = {k: float((v * w).sum() / w.sum()) for k, v in arrs.items()}
        for k in arrs:
            boots[k].append(means[k])
        boots["ratio_srp_knn"].append(means["srp"] / max(means["knn"], 1e-9))

    def ci95(k: str) -> list[float]:
        v = np.array(boots[k])
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return dict(
        width=width_tag,
        n_loo_cells=len(srp_loo),
        core_median_size=float(np.median(core_sizes)),
        srp_heldout_seed_recall=m(srp_loo),
        srp_ci95=ci95("srp"),
        knn_recall_same_cores=m(knn_loo),
        knn_ci95=ci95("knn"),
        cluster_recall_same_cores=m(cl_loo),
        cluster_ci95=ci95("cluster"),
        paper_dict_recall_same_cores=m(ref_loo),
        ratio_srp_over_knn=round(float(np.mean(srp_loo) / np.mean(knn_loo)), 3),
        ratio_ci95=ci95("ratio_srp_knn"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dictionaries",
        nargs="+",
        type=Path,
        required=True,
        help="checkpoint.pt of each seed, one recipe, at least three (paper: seeds 0, 1, 2 of one width)",
    )
    ap.add_argument("--w-u", type=Path, required=True, help="extraction payload {W_U_orig, token_mask, ...}")
    ap.add_argument("--bank", type=Path, required=True, help="curated A/B JSONL bank (target_a / target_b)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B", help="HF tokenizer id or local path")
    ap.add_argument(
        "--centering",
        choices=("live", "trained"),
        default="live",
        help="centering mean: full-vocabulary mean of W_U (live, the paper's runs) or the dictionaries' stored "
        "training mean (trained)",
    )
    ap.add_argument("--width-tag", default=None, help='label stored in the output, e.g. "32x"')
    ap.add_argument(
        "--reference-dict",
        type=Path,
        default=None,
        help="optional dictionary from another recipe, scored on the same cores (paper: 32x run only)",
    )
    ap.add_argument("--max-contrasts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0, help="contrast-order seed")
    ap.add_argument("--top-m", type=int, default=8, help="features per side")
    ap.add_argument("--top-r", type=int, default=12, help="rows per feature for the token summary")
    ap.add_argument("--side-n", type=int, default=96, help="tokens per kNN / cluster side set")
    ap.add_argument(
        "--knn-exclude-self",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop the target row from its own kNN side set (the paper's run); --no-knn-exclude-self admits it",
    )
    ap.add_argument("--n-clusters", type=int, default=16384)
    ap.add_argument("--kmeans-iters", type=int, default=12)
    ap.add_argument("--kmeans-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=1)
    ap.add_argument("--device", type=str, default=None, help="device for the k-means fit")
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    args = ap.parse_args()

    if len(args.dictionaries) < 3:
        ap.error("--dictionaries needs at least three checkpoints (leave-one-out cores need two remaining seeds)")

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    W, _, token_mask = load_readout(args.w_u)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    dicts = [load_dictionary(p) for p in args.dictionaries]
    for si, d in enumerate(dicts):
        print(f"seed {si}: D={d.decoder.shape[0]} k={d.k}")
    if len({(d.decoder.shape[0], d.k) for d in dicts}) != 1:
        raise SystemExit("dictionaries differ in width or k; pass checkpoints from one dictionary family")
    reference = load_dictionary(args.reference_dict) if args.reference_dict is not None else None
    row_mean = resolve_centering(W, dicts, args.centering, token_mask, tok=tok)

    pairs = load_contrast_pairs(args.bank, tok, rng, args.max_contrasts)
    print(f"contrasts: {len(pairs)} unique single-token pairs")

    out = run(
        dicts,
        W,
        tok,
        pairs,
        device,
        row_mean=row_mean,
        top_m=args.top_m,
        top_r=args.top_r,
        side_n=args.side_n,
        n_clusters=args.n_clusters,
        kmeans_iters=args.kmeans_iters,
        kmeans_seed=args.kmeans_seed,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
        reference_dict=reference,
        width_tag=args.width_tag,
        knn_exclude_self=args.knn_exclude_self,
    )
    print(json.dumps(to_jsonable(out), indent=2))
    out["provenance"] = run_provenance(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(out), indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
