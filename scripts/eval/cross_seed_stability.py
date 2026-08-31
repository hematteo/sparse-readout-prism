#!/usr/bin/env python3
"""Cross-seed stability of contrast explanations (Appendix I, ``tab:app-cross-seed-stability``).

Given several dictionaries trained from one recipe with different seeds (same
width, sparsity and preprocessing), decompose every curated single-token
contrast ``q = w_A - w_B`` with each dictionary and ask whether the same token
families land on the same side of the contrast across seeds. For each seed a
contrast yields row-side coefficients ``beta_i = rn_A z_A,i - rn_B z_B,i``; the
top-M positive and top-M negative features summarise each side by the token
set of their top-R centred unembedding rows.

Computes exactly five quantities:

    same-side Jaccard        J(T_pos^s1, T_pos^s2), J(T_neg^s1, T_neg^s2)   per seed pair and contrast
    cross-side leakage       J(T_pos^s1, T_neg^s2)                          should be ~0
    cross-contrast null      J(T_pos^s1(c), T_pos^s2(c')),  c' != c random   null anchor
    matched decoder cosine   nearest-neighbour decoder cosine between seeds, for a
                             random feature sample and for the features the contrasts use
    matched-projection corr  correlation between projections of held-out hidden states
                             (``h_LN`` in the extraction payload) onto each used feature
                             and onto its nearest decoder match in the other seed

plus the fraction of contrasts whose mean same-side Jaccard exceeds the
cross-contrast null's 90th percentile. Everything is local to the extraction
payload and the checkpoints; no model forward pass.

Dictionary-family discipline: pass dictionaries from ONE recipe (the
seed-variation family trained from ``configs/sweeps/qwen35_2b_seedvar_base.yaml``).
The script refuses dictionaries that differ in width or ``k``. Cross-recipe
agreement is weaker and is reported separately (Appendix I).

Paper runs (Qwen3.5-2B, three seeds per width, curated A/B bank, all defaults):

    uv run python scripts/eval/cross_seed_stability.py \\
        --dictionaries results/qwen35_2b_seedvar/qwen2b_d65536_k256_s0/checkpoint.pt \\
                       results/qwen35_2b_seedvar/qwen2b_d65536_k256_s1/checkpoint.pt \\
                       results/qwen35_2b_seedvar/qwen2b_d65536_k256_s2/checkpoint.pt \\
        --w-u "$SRP_ARCHIVE_ROOT/data/qwen35-2b/qwen35_2b.pt" \\
        --bank data/query_banks/qwen_gemma_result1_curated_ab.jsonl \\
        --width-tag 32x --out results/seed_stability/cross_seed_stability_32x.json

    (16x: the ``qwen2b_d32768_k128_s{0,1,2}`` checkpoints, ``--width-tag 16x``.)

Under the Qwen3.5 tokenizer the 150-contrast cap resolves to 63 unique
single-token pairs of the curated bank (126 signed side-sets).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.research.seed_stability import (
    center_rows,
    contrast_features,
    feature_token_set,
    jaccard,
    load_contrast_pairs,
    load_dictionary,
    load_readout,
)
from sparse_readout_prism.utils import to_jsonable


def _stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p10=float(np.percentile(v, 10)), n=int(len(v)))


def _unit(d: torch.Tensor) -> torch.Tensor:
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)


def run(
    dicts: list,
    labels: list[int],
    W: torch.Tensor,
    h_LN: torch.Tensor | None,
    tok,
    pairs: list[tuple],
    rng: np.random.Generator,
    *,
    top_m: int,
    top_r: int,
    n_sample: int,
    n_hidden: int,
    width_tag: str | None,
) -> dict:
    W_c, rn, W_n = center_rows(W)
    n_seeds = len(dicts)
    cache: dict[int, str] = {}

    # per-seed per-contrast side token-sets + used features
    side_sets: list[list[tuple[set, set]]] = [[] for _ in dicts]  # [seed][ci] = (T_pos, T_neg)
    used_feats: list[set[int]] = [set() for _ in dicts]
    for si, d in enumerate(dicts):
        dec = d[0]
        for _a, _b, ia, ib in pairs:
            top_pos, top_neg = contrast_features(W_n, rn, d, ia, ib, top_m)
            used_feats[si].update(top_pos + top_neg)
            side_sets[si].append(
                (
                    feature_token_set(top_pos, dec, W_c, tok, top_r, cache),
                    feature_token_set(top_neg, dec, W_c, tok, top_r, cache),
                )
            )
        print(f"seed {labels[si]}: decomposed {len(pairs)} contrasts", flush=True)

    # explanation-level reproduction
    same_side, cross_side, cross_contrast = [], [], []
    for s1 in range(n_seeds):
        for s2 in range(s1 + 1, n_seeds):
            for ci in range(len(pairs)):
                p1, n1 = side_sets[s1][ci]
                p2, n2 = side_sets[s2][ci]
                same_side.append(jaccard(p1, p2))
                same_side.append(jaccard(n1, n2))
                cross_side.append(jaccard(p1, n2))
                cj = int(rng.integers(len(pairs) - 1))
                cj = cj + 1 if cj >= ci else cj
                cross_contrast.append(jaccard(p1, side_sets[s2][cj][0]))

    # fraction of contrasts whose same-side overlap beats the cross-contrast null p90
    null90 = float(np.percentile(cross_contrast, 90))
    per_contrast = np.asarray(same_side).reshape(-1, 2).mean(axis=1)  # mean of pos/neg per (pair, ci)
    frac_above_null = float((per_contrast > null90).mean())

    # basis-level nearest-neighbour cosine
    basis = {}
    for s1 in range(n_seeds):
        for s2 in range(s1 + 1, n_seeds):
            d1, d2 = dicts[s1][0], dicts[s2][0]
            d1n, d2n = _unit(d1), _unit(d2)
            samp = torch.tensor(rng.choice(d1.shape[0], min(n_sample, d1.shape[0]), replace=False))
            nn_rand = (d1n[samp] @ d2n.T).max(dim=1).values
            used = torch.tensor(sorted(used_feats[s1]))
            nn_used = (d1n[used] @ d2n.T).max(dim=1).values
            basis[f"s{labels[s1]}-s{labels[s2]}"] = dict(
                nn_cos_random=_stats(nn_rand.tolist()), nn_cos_used=_stats(nn_used.tolist())
            )

    # projection correlation on held-out hidden states
    proj = {}
    if h_LN is not None:
        H = h_LN.reshape(-1, h_LN.shape[-1]).float()  # (n_states, d_model)
        H = H[torch.tensor(rng.choice(H.shape[0], min(n_hidden, H.shape[0]), replace=False))]
        for s1 in range(n_seeds):
            for s2 in range(s1 + 1, n_seeds):
                d1, d2 = dicts[s1][0], dicts[s2][0]
                used = torch.tensor(sorted(used_feats[s1]))
                match = (_unit(d1[used]) @ _unit(d2).T).argmax(dim=1)
                p1 = H @ d1[used].T
                p2 = H @ d2[match].T
                r = torch.corrcoef(torch.stack([p1.flatten(), p2.flatten()]))[0, 1]
                proj[f"s{labels[s1]}-s{labels[s2]}"] = float(r)

    return dict(
        width=width_tag,
        k=int(dicts[0][3]),
        n_contrasts=len(pairs),
        top_m=top_m,
        top_r=top_r,
        same_side=_stats(same_side),
        cross_side=_stats(cross_side),
        cross_contrast_null=_stats(cross_contrast),
        null_p90=null90,
        frac_contrasts_above_null_p90=frac_above_null,
        basis_nn_cosine=basis,
        matched_projection_corr=proj,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dictionaries",
        nargs="+",
        type=Path,
        required=True,
        help="checkpoint.pt of each seed, one recipe (paper: seeds 0, 1, 2 of one width)",
    )
    ap.add_argument("--seed-labels", nargs="+", type=int, default=None, help="labels for output keys (default 0..n-1)")
    ap.add_argument("--w-u", type=Path, required=True, help="extraction payload {W_U_orig, h_LN, ...}")
    ap.add_argument("--bank", type=Path, required=True, help="curated A/B JSONL bank (target_a / target_b)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B", help="HF tokenizer id or local path")
    ap.add_argument("--width-tag", default=None, help='label stored in the output, e.g. "32x"')
    ap.add_argument("--max-contrasts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-m", type=int, default=8, help="features per side")
    ap.add_argument("--top-r", type=int, default=12, help="rows per feature for the token summary")
    ap.add_argument("--n-sample", type=int, default=4096, help="random features for the basis-level NN cosine")
    ap.add_argument("--n-hidden", type=int, default=512, help="held-out hidden states for the projection check")
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    args = ap.parse_args()

    if len(args.dictionaries) < 2:
        ap.error("--dictionaries needs at least two checkpoints")
    labels = args.seed_labels if args.seed_labels is not None else list(range(len(args.dictionaries)))
    if len(labels) != len(args.dictionaries):
        ap.error("--seed-labels must have one entry per dictionary")

    rng = np.random.default_rng(args.seed)
    W, h_LN = load_readout(args.w_u)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    dicts = [load_dictionary(p) for p in args.dictionaries]
    for lab, d in zip(labels, dicts):
        print(f"seed {lab}: D={d[0].shape[0]} k={d[3]}")
    if len({(d[0].shape[0], d[3]) for d in dicts}) != 1:
        raise SystemExit("dictionaries differ in width or k; pass checkpoints from one dictionary family")

    pairs = load_contrast_pairs(args.bank, tok, rng, args.max_contrasts)
    print(f"contrasts: {len(pairs)} unique single-token pairs")

    out = run(
        dicts,
        labels,
        W,
        h_LN,
        tok,
        pairs,
        rng,
        top_m=args.top_m,
        top_r=args.top_r,
        n_sample=args.n_sample,
        n_hidden=args.n_hidden,
        width_tag=args.width_tag,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(out), indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "basis_nn_cosine"}, indent=2))
    print(
        "basis NN-cosine (used features):",
        {k: round(v["nn_cos_used"]["median"], 3) for k, v in out["basis_nn_cosine"].items()},
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
