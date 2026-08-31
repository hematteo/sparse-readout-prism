"""Shared pipeline for the cross-seed stability analyses (Appendix I).

Backs three entry points: ``scripts/eval/cross_seed_stability.py``
(``tab:app-cross-seed-stability``), ``scripts/eval/feature_group_matching.py``
(``tab:app-feature-group-matching``) and ``scripts/eval/loo_core_recovery.py``
(``tab:app-loo-core-recovery``). All three compare token summaries produced by
independently seeded dictionaries of one training recipe, so the steps that
produce those summaries must be identical across the scripts and live here:

* single-token A/B contrasts parsed from a curated JSONL bank (the bare form
  is tried before the space-prefixed form, via
  ``row_geometry.resolve_single_token_bare_first``; the first occurrence of a
  pair wins, the list is shuffled once with the caller's generator and capped),
* rows centred against an explicit centering mean and per-row normalised with
  ``data.center_normalize_rows``; the mean is chosen by ``resolve_centering``,
  the scripts' ``--centering {live,trained}`` switch (``live`` = the
  full-vocabulary mean of ``W_U``, how the paper's runs were computed;
  ``trained`` = the dictionaries' stored training mean, through
  ``data.centering_mean``),
* TopK codes from a dictionary's encoder (``qwen_readout.encode_topk``, the
  one top-k kernel) and the contrast coefficient
  ``beta = rn_A * z_A - rn_B * z_B`` for the contrast ``w_A - w_B``,
* the top-M features per sign, and the token set of a feature's top-R
  centred rows (decoded, stripped, lower-cased, printable strings only),
* the summary statistics and decoder unit-normalisation the scripts share.

Checkpoints are read with ``qwen_readout.load_sae`` (``load_factorizer``
underneath: ``weights_only=True``, TopK architecture checked), which also
surfaces the stored training ``row_mean``.

Changed in 0.2.1:

* Top-k selection routes through ``factorizers.topk_mask`` and keeps exactly
  ``k`` codes per row. The previous rule zeroed activations *below* the k-th
  largest value and so kept every activation tied with it; the two agree
  unless positive activations tie exactly at the boundary (zero codes
  contribute nothing to ``beta`` either way), which does not happen for a
  trained encoder on real rows.
* The legacy ``W_dec`` / ``W_enc`` / ``b_enc`` checkpoint layout is no longer
  read. Every dictionary of the paper runs, the cross-recipe
  ``--reference-dict`` included, is in the runner's ``model_state_dict``
  schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from sparse_readout_prism.data import center_normalize_rows, centering_mean, token_mask_from_tokenizer
from sparse_readout_prism.research.qwen_readout import encode_topk, load_sae
from sparse_readout_prism.research.row_geometry import resolve_single_token_bare_first


class Dictionary(NamedTuple):
    """One seed's dictionary as float32 CPU tensors (indexable like the former 4-tuple)."""

    decoder: torch.Tensor  # (d_features, d_model)
    encoder_w: torch.Tensor  # (d_features, d_model)
    encoder_b: torch.Tensor  # (d_features,)
    k: int
    row_mean: torch.Tensor | None  # (d_model,) training centering mean; None for checkpoints predating it


def load_dictionary(path: str | Path) -> Dictionary:
    """Read a TopK checkpoint through ``load_sae``; ``k`` comes from its factorizer config (default 256)."""
    decoder, encoder_w, encoder_b, _config, row_mean = load_sae(path)
    # load_sae hands back the training-config block, which runner checkpoints
    # do not carry; k lives in the top-level factorizer block. mmap keeps this
    # second open from reading the tensors again.
    meta = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg = meta.get("factorizer") or meta.get("config", {}).get("factorizer", {})
    return Dictionary(decoder, encoder_w, encoder_b, int(cfg.get("k", 256)), row_mean)


def load_readout(path: str | Path) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """``(W_U, h_LN, token_mask)`` from an extraction payload; ``h_LN`` and ``token_mask`` may be None."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    W = payload.get("W_U_orig", payload.get("W_U"))
    if W is None:
        raise KeyError(f"{path}: no W_U_orig / W_U in payload")
    return W.float(), payload.get("h_LN"), payload.get("token_mask")


def resolve_centering(
    W: torch.Tensor, dicts: list[Dictionary], mode: str, token_mask: torch.Tensor | None, tok=None
) -> torch.Tensor:
    """Centering mean of a dictionary family under ``--centering {live,trained}``.

    ``trained`` uses the stored training mean of the dictionaries (one recipe,
    so every checkpoint must store the same tensor; a disagreement is refused)
    and otherwise the text-token mean over ``token_mask`` (the payload's, or
    one rebuilt from ``tok`` when the payload predates the mask). ``live`` is
    the full-vocabulary mean of ``W``.
    """
    stored = [d.row_mean for d in dicts if d.row_mean is not None]
    if mode == "trained":
        if any(not torch.equal(stored[0], m) for m in stored[1:]):
            raise ValueError("dictionaries store different training row_mean tensors; pass checkpoints from one recipe")
        if token_mask is None and tok is not None:
            token_mask = token_mask_from_tokenizer(tok, W.shape[0])
    ckpt = {"row_mean": stored[0]} if stored else None
    return centering_mean(W, mode=mode, token_mask=token_mask, ckpt=ckpt)


def center_rows(W: torch.Tensor, row_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centring against ``row_mean`` and per-row normalisation: ``(W_c, row_norms, W_n)``."""
    rn, W_n = center_normalize_rows(W, row_mean)
    return W - row_mean, rn, W_n


def unit_rows(d: torch.Tensor) -> torch.Tensor:
    """Rows of ``d`` scaled to unit norm (1e-8 floor)."""
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-8)


def summary_stats(v, percentiles: tuple[int, ...]) -> dict:
    """``{mean, median, p<q>..., n}`` of ``v``, in the key order the scripts' JSON outputs use."""
    v = np.asarray(v, dtype=float)
    out = dict(mean=float(v.mean()), median=float(np.median(v)))
    for q in percentiles:
        out[f"p{q}"] = float(np.percentile(v, q))
    out["n"] = int(len(v))
    return out


def load_contrast_pairs(bank_path: str | Path, tok, rng: np.random.Generator, max_contrasts: int) -> list[tuple]:
    """Unique single-token ``(a, b, id_a, id_b)`` contrasts of a bank, shuffled with ``rng`` and capped."""
    recs = [json.loads(line) for line in Path(bank_path).read_text().splitlines() if line.strip()]
    pairs: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for r in recs:
        a, b = r.get("target_a"), r.get("target_b")
        if not a or not b or (a, b) in seen:
            continue
        ia, ib = resolve_single_token_bare_first(tok, a), resolve_single_token_bare_first(tok, b)
        if ia is None or ib is None or ia == ib:
            continue
        seen.add((a, b))
        pairs.append((a, b, ia, ib))
    rng.shuffle(pairs)
    return pairs[:max_contrasts]


def contrast_features(
    W_n: torch.Tensor, rn: torch.Tensor, dictionary: Dictionary, ia: int, ib: int, top_m: int
) -> tuple[list[int], list[int]]:
    """Top-M positive and top-M negative feature ids of the contrast row ``ia`` minus row ``ib``."""
    z = encode_topk(torch.stack([W_n[ia], W_n[ib]]), dictionary.encoder_w, dictionary.encoder_b, dictionary.k)
    beta = rn[ia] * z[0] - rn[ib] * z[1]  # (d_features,)
    return torch.topk(beta, top_m).indices.tolist(), torch.topk(-beta, top_m).indices.tolist()


def token_strings(tok, ids, cache: dict[int, str] | None = None) -> set[str]:
    """Decoded, stripped, lower-cased, printable token strings of ``ids``."""
    out: set[str] = set()
    for r in ids:
        r = int(r)
        s = None if cache is None else cache.get(r)
        if s is None:
            s = tok.decode([r]).strip().lower()
            if cache is not None:
                cache[r] = s
        if s and s.isprintable():
            out.add(s)
    return out


def feature_token_set(
    fids, dec: torch.Tensor, W_c: torch.Tensor, tok, top_r: int, cache: dict[int, str] | None = None
) -> set[str]:
    """Union over ``fids`` of the token strings of each feature's top-R centred rows.

    One matrix-vector product per feature, as in the paper's runs; a single
    GEMM over the unique feature ids would not reproduce the same top-R
    orderings bit for bit (different accumulation order), so it is not used.
    """
    out: set[str] = set()
    for f in fids:
        rows = torch.topk(W_c @ dec[f], top_r).indices.tolist()
        out |= token_strings(tok, rows, cache)
    return out


def jaccard(a, b) -> float:
    return len(a & b) / max(1, len(a | b))
