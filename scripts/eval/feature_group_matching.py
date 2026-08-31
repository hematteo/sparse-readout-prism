#!/usr/bin/env python3
"""Cross-seed feature-group matching (Appendix I, ``tab:app-feature-group-matching``).

Contrast-level explanations reproduce across seeds while feature indices
reshuffle. This script asks whether the feature-level token GROUPS themselves
have an equivalent in another seed's dictionary, i.e. whether the grouping
structure is re-indexed rather than seed noise (cf. Paulo & Belrose 2025,
arXiv:2501.16615).

Protocol
  * Query population: features USED by the curated contrast bank in the query
    seed (top-M positive + top-M negative per contrast, the same construction
    as ``cross_seed_stability.py``); ``--n-query`` are sampled per seed pair
    among those with at least ``--min-group-size`` tokens.
  * Group of a feature = token strings of its top-R centred unembedding rows
    by decoder similarity (identical to the token summary used elsewhere).
  * Match: best-of-D search over ALL candidate-seed features by Jaccard of
    token groups. Secondary: greedy union of up to ``--greedy-k`` candidate
    groups drawn from the ``--greedy-pool`` best single matches (feature
    splitting), scored by recall of the query group.
  * Null: ``--n-null`` pseudo-groups of top-R tokens drawn from the empirical
    token-frequency distribution over the candidate seed's groups, run through
    the IDENTICAL best-of-D search, which prices in the inflation of searching
    a 65,536-entry dictionary.
  * Criteria, fixed before the analysis: a query group reproduces if its best
    single Jaccard exceeds the null's 99th percentile; strong equivalence is a
    best single Jaccard of at least 0.5. Full distributions are stored either way.

``--direction-check`` adds the direction-level check of the below-null tail:
for every query group at or below the null p99, the maximum decoder cosine
over the full candidate dictionary, against a null of ``--direction-null``
random unit directions passed through the same maximum. It writes
``max_decoder_cosine`` into those per-query records and a ``direction_check``
summary per seed pair (counts above the direction null p99 and at cosine >= 0.5).

Seed pairs are all ordered pairs (query, candidate) with query index below
candidate index, so three dictionaries give s0->s1, s0->s2, s1->s2.

Dictionary-family discipline: pass dictionaries from ONE recipe (the
seed-variation family trained from ``configs/sweeps/qwen35_2b_seedvar_base.yaml``);
the script refuses dictionaries that differ in width or ``k``.

Paper runs (Qwen3.5-2B, three seeds per width, curated A/B bank, all defaults):

    uv run python scripts/eval/feature_group_matching.py \\
        --dictionaries results/qwen35_2b_seedvar/qwen2b_d65536_k256_s0/checkpoint.pt \\
                       results/qwen35_2b_seedvar/qwen2b_d65536_k256_s1/checkpoint.pt \\
                       results/qwen35_2b_seedvar/qwen2b_d65536_k256_s2/checkpoint.pt \\
        --w-u "$SRP_ARCHIVE_ROOT/data/qwen35-2b/qwen35_2b.pt" \\
        --bank data/query_banks/qwen_gemma_result1_curated_ab.jsonl \\
        --width-tag 32x --direction-check \\
        --out results/seed_stability/feature_group_matching_32x.json

    (16x: the ``qwen2b_d32768_k128_s{0,1,2}`` checkpoints, ``--width-tag 16x``.)
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sparse_readout_prism.research.seed_stability import (
    center_rows,
    contrast_features,
    load_contrast_pairs,
    load_dictionary,
    load_readout,
    token_strings,
)
from sparse_readout_prism.utils import resolve_device, to_jsonable


def all_feature_groups(dec: torch.Tensor, W_c: torch.Tensor, tok, device, cache: dict, top_r: int) -> list[frozenset]:
    """Token group (frozenset of strings) of every feature: its top-R centred rows."""
    D = dec.shape[0]
    Wt = W_c.T.to(device)  # (d_model, vocab)
    top_rows = torch.empty((D, top_r), dtype=torch.long)
    for lo in range(0, D, 2048):
        chunk = dec[lo : lo + 2048].to(device)
        sims = chunk @ Wt  # (chunk, vocab)
        top_rows[lo : lo + 2048] = sims.topk(top_r, dim=1).indices.cpu()
        del sims, chunk
    return [frozenset(token_strings(tok, top_rows[f].tolist(), cache)) for f in range(D)]


def best_matches(query: frozenset, inv_index: dict, groups: list[frozenset], greedy_pool: int, greedy_k: int):
    """``(best_jaccard, best_feature, greedy recall@1..greedy_k)`` over all candidate groups."""
    counts: Counter = Counter()
    for t in query:
        for f in inv_index.get(t, ()):
            counts[f] += 1
    if not counts:
        return 0.0, -1, [0.0] * greedy_k
    scored = sorted(
        ((inter / (len(query) + len(groups[f]) - inter), f) for f, inter in counts.items()),
        reverse=True,
    )
    best_j, best_f = scored[0]
    pool = [f for _, f in scored[:greedy_pool]]
    covered: set = set()
    recalls = []
    for _ in range(greedy_k):
        gain_best, f_best = 0, None
        for f in pool:
            gain = len((query & groups[f]) - covered)
            if gain > gain_best:
                gain_best, f_best = gain, f
        if f_best is not None:
            covered |= query & groups[f_best]
            pool.remove(f_best)
        recalls.append(len(covered) / max(1, len(query)))
    return best_j, best_f, recalls


def _stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return dict(
        mean=float(v.mean()),
        median=float(np.median(v)),
        p90=float(np.percentile(v, 90)),
        p99=float(np.percentile(v, 99)),
        n=int(len(v)),
    )


def _unit(d: torch.Tensor) -> torch.Tensor:
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)


def run(
    dicts: list,
    labels: list[int],
    W: torch.Tensor,
    tok,
    pairs: list[tuple],
    rng: np.random.Generator,
    device,
    *,
    top_m: int,
    top_r: int,
    n_query: int,
    n_null: int,
    greedy_pool: int,
    greedy_k: int,
    min_group_size: int,
    width_tag: str | None,
    direction_check: bool,
    direction_null: int,
    direction_seed: int,
) -> dict:
    W_c, rn, W_n = center_rows(W)
    cache: dict[int, str] = {}

    # token groups of every feature, every seed
    groups = [all_feature_groups(d[0], W_c, tok, device, cache, top_r) for d in dicts]
    print("feature groups computed", flush=True)

    # used features per seed (same construction as cross_seed_stability.py)
    used: list[set[int]] = [set() for _ in dicts]
    for si, d in enumerate(dicts):
        for _a, _b, ia, ib in pairs:
            top_pos, top_neg = contrast_features(W_n, rn, d, ia, ib, top_m)
            used[si].update(top_pos)
            used[si].update(top_neg)
    print("used features:", [len(u) for u in used])

    seed_pairs = list(itertools.combinations(range(len(dicts)), 2))  # (query seed, candidate seed)
    unit_decs = [_unit(d[0]) for d in dicts]
    results: dict[str, dict] = {}
    for qs, cs in seed_pairs:
        cand_groups = groups[cs]
        inv: dict[str, list[int]] = {}
        for f, g in enumerate(cand_groups):
            for t in g:
                inv.setdefault(t, []).append(f)

        # frequency-matched null token pool
        tok_freq = Counter(t for g in cand_groups for t in g)
        pool_toks = list(tok_freq)
        pool_p = np.array([tok_freq[t] for t in pool_toks], dtype=float)
        pool_p /= pool_p.sum()

        # queries: sampled used features of the query seed with large-enough groups
        cands = [f for f in sorted(used[qs]) if len(groups[qs][f]) >= min_group_size]
        qs_feats = rng.choice(cands, size=min(n_query, len(cands)), replace=False)

        q_best, q_rec, q_cos, q_records = [], [], [], []
        dq, dc = unit_decs[qs], unit_decs[cs]
        for f in qs_feats:
            f = int(f)
            bj, bf, recalls = best_matches(groups[qs][f], inv, cand_groups, greedy_pool, greedy_k)
            q_best.append(bj)
            q_rec.append(recalls)
            q_cos.append(float(dq[f] @ dc[bf]) if bf >= 0 else 0.0)
            q_records.append(
                dict(
                    feature=f,
                    query_tokens=sorted(groups[qs][f]),
                    best_jaccard=round(bj, 4),
                    best_feature=int(bf),
                    best_match_tokens=sorted(cand_groups[bf]) if bf >= 0 else [],
                    recall3=round(recalls[-1], 4),
                    cosine=round(q_cos[-1], 4),
                )
            )

        n_best, n_rec = [], []
        for _ in range(n_null):
            g = frozenset(rng.choice(pool_toks, size=top_r, replace=False, p=pool_p))
            bj, _, recalls = best_matches(g, inv, cand_groups, greedy_pool, greedy_k)
            n_best.append(bj)
            n_rec.append(recalls)

        null99 = float(np.percentile(n_best, 99))
        rec3 = [r[-1] for r in q_rec]
        n_rec3 = [r[-1] for r in n_rec]
        res = dict(
            n_query=int(len(qs_feats)),
            best_jaccard=_stats(q_best),
            null_best_jaccard=_stats(n_best),
            frac_above_null_p99=float((np.array(q_best) > null99).mean()),
            frac_jaccard_ge_050=float((np.array(q_best) >= 0.5).mean()),
            frac_jaccard_ge_025=float((np.array(q_best) >= 0.25).mean()),
            recall_at_1=_stats([r[0] for r in q_rec]),
            recall_at_3=_stats(rec3),
            null_recall_at_3=_stats(n_rec3),
            frac_recall3_above_null_p99=float((np.array(rec3) > np.percentile(n_rec3, 99)).mean()),
            matched_decoder_cosine=_stats(q_cos),
            null99=null99,
            per_query=q_records,
        )
        key = f"s{labels[qs]}->s{labels[cs]}"
        results[key] = res
        print(
            f"{key}: best-J med {res['best_jaccard']['median']:.3f} "
            f"(null p99 {null99:.3f}), >null {res['frac_above_null_p99']:.2f}, "
            f">=0.5 {res['frac_jaccard_ge_050']:.2f}, recall@3 med {res['recall_at_3']['median']:.3f} "
            f"(null {res['null_recall_at_3']['median']:.3f})",
            flush=True,
        )

    if direction_check:
        drng = np.random.default_rng(direction_seed)
        d_model = dicts[0][0].shape[1]
        for qs, cs in seed_pairs:
            key = f"s{labels[qs]}->s{labels[cs]}"
            r = results[key]
            fails = [q for q in r["per_query"] if q["best_jaccard"] <= r["null99"]]
            dq, dc = unit_decs[qs], unit_decs[cs]
            g = torch.tensor(drng.standard_normal((direction_null, d_model)), dtype=torch.float32)
            g = g / g.norm(dim=1, keepdim=True)
            null_max = (g @ dc.T).max(dim=1).values
            cos99 = float(np.percentile(null_max.numpy(), 99))
            n_dir = n_half = 0
            for q in fails:
                maxcos = float((dq[q["feature"]] @ dc.T).max())
                q["max_decoder_cosine"] = round(maxcos, 4)
                n_dir += int(maxcos > cos99)
                n_half += int(q["max_decoder_cosine"] >= 0.5)
            r["direction_check"] = dict(
                n_below_null_p99=len(fails),
                n_null=direction_null,
                direction_null_p99=cos99,
                n_above_direction_null_p99=n_dir,
                n_decoder_cosine_ge_0p5=n_half,
            )
            print(
                f"{key}: {len(fails)} token-level failures | direction-null p99 = {cos99:.3f} | "
                f"above direction null: {n_dir}/{len(fails)} | cosine >= 0.5: {n_half}/{len(fails)}"
            )
            for q in sorted(fails, key=lambda x: -x["max_decoder_cosine"]):
                mark = "DIR-MATCH" if q["max_decoder_cosine"] > cos99 else "no match "
                print(
                    f"  [{mark}] feat {q['feature']} maxcos={q['max_decoder_cosine']:.2f} "
                    f"bestJ={q['best_jaccard']:.2f} :: {', '.join(q['query_tokens'][:8])}"
                )

    return dict(width=width_tag, top_r=top_r, top_m=top_m, n_null=n_null, pairs=results)


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
    ap.add_argument("--w-u", type=Path, required=True, help="extraction payload {W_U_orig, ...}")
    ap.add_argument("--bank", type=Path, required=True, help="curated A/B JSONL bank (target_a / target_b)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B", help="HF tokenizer id or local path")
    ap.add_argument("--width-tag", default=None, help='label stored in the output, e.g. "32x"')
    ap.add_argument("--max-contrasts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-m", type=int, default=8, help="features per side when collecting used features")
    ap.add_argument("--top-r", type=int, default=12, help="rows per feature -> token group")
    ap.add_argument("--n-query", type=int, default=100, help="query features sampled per seed pair")
    ap.add_argument("--n-null", type=int, default=500, help="frequency-matched pseudo-groups per seed pair")
    ap.add_argument("--greedy-pool", type=int, default=200, help="best single matches eligible for the greedy union")
    ap.add_argument("--greedy-k", type=int, default=3, help="maximum features in the greedy union (recall@k)")
    ap.add_argument("--min-group-size", type=int, default=4, help="minimum tokens in a query group")
    ap.add_argument("--direction-check", action="store_true", help="direction-level check of the below-null tail")
    ap.add_argument("--direction-null", type=int, default=500, help="random unit directions for the direction null")
    ap.add_argument("--direction-seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, help="device for the all-feature row scoring")
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    args = ap.parse_args()

    if len(args.dictionaries) < 2:
        ap.error("--dictionaries needs at least two checkpoints")
    labels = args.seed_labels if args.seed_labels is not None else list(range(len(args.dictionaries)))
    if len(labels) != len(args.dictionaries):
        ap.error("--seed-labels must have one entry per dictionary")

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")
    W, _ = load_readout(args.w_u)
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
        tok,
        pairs,
        rng,
        device,
        top_m=args.top_m,
        top_r=args.top_r,
        n_query=args.n_query,
        n_null=args.n_null,
        greedy_pool=args.greedy_pool,
        greedy_k=args.greedy_k,
        min_group_size=args.min_group_size,
        width_tag=args.width_tag,
        direction_check=args.direction_check,
        direction_null=args.direction_null,
        direction_seed=args.direction_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(out), indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
