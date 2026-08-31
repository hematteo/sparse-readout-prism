"""Shared pipeline for the cross-seed stability analyses (Appendix I).

Backs three entry points: ``scripts/eval/cross_seed_stability.py``
(``tab:app-cross-seed-stability``), ``scripts/eval/feature_group_matching.py``
(``tab:app-feature-group-matching``) and ``scripts/eval/loo_core_recovery.py``
(``tab:app-loo-core-recovery``). All three compare token summaries produced by
independently seeded dictionaries of one training recipe, so the steps that
produce those summaries must be identical across the scripts and live here:

* single-token A/B contrasts parsed from a curated JSONL bank (the bare form
  is tried before the space-prefixed form, the first occurrence of a pair
  wins, the list is shuffled once with the caller's generator and capped),
* rows centred against the full-vocabulary mean of ``W_U`` and per-row
  normalised,
* TopK codes from a dictionary's encoder and the contrast coefficient
  ``beta = rn_A * z_A - rn_B * z_B`` for the contrast ``w_A - w_B``,
* the top-M features per sign, and the token set of a feature's top-R
  centred rows (decoded, stripped, lower-cased, printable strings only).

Checkpoints are read as raw tensors (``decoder`` / ``encoder.weight`` /
``encoder.bias`` from ``model_state_dict``) so the scripts also accept the
legacy ``W_dec`` / ``W_enc`` / ``b_enc`` layout of an archived reference
dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

Dictionary = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


def load_dictionary(path: str | Path) -> Dictionary:
    """``(decoder, encoder_w, encoder_b, k)`` as float32 CPU tensors.

    ``decoder`` and ``encoder_w`` are ``(d_features, d_model)``; ``k`` is read
    from the checkpoint's factorizer config (default 256 when absent).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "W_dec" in ckpt:  # legacy raw-tensor layout
        dec = ckpt["W_dec"].float()
        enc_w = ckpt["W_enc"].float().T.contiguous()
        bias = ckpt.get("b_enc")
        enc_b = bias.float() if bias is not None else torch.zeros(enc_w.shape[0])
        k = int(ckpt.get("k") or ckpt.get("config", {}).get("k") or 256)
        return dec, enc_w, enc_b, k
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"{path}: no model_state_dict / state_dict (or legacy W_dec) in checkpoint")
    dec = state["decoder"].float()
    enc_w = state["encoder.weight"].float()
    enc_b = state.get("encoder.bias", torch.zeros(enc_w.shape[0])).float()
    cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer", {})
    return dec, enc_w, enc_b, int(cfg.get("k", 256))


def load_readout(path: str | Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(W_U, h_LN)`` from an extraction payload ``{W_U_orig, h_LN, ...}``; ``h_LN`` may be None."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    W = payload.get("W_U_orig", payload.get("W_U"))
    if W is None:
        raise KeyError(f"{path}: no W_U_orig / W_U in payload")
    return W.float(), payload.get("h_LN")


def center_rows(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-vocabulary centring and per-row normalisation: ``(W_c, row_norms, W_n)``."""
    W_c = W - W.mean(0)
    rn = W_c.norm(dim=1).clamp_min(1e-8)
    return W_c, rn, W_c / rn[:, None]


def topk_codes(x: torch.Tensor, enc_w: torch.Tensor, enc_b: torch.Tensor, k: int) -> torch.Tensor:
    """ReLU encoder activations, zeroed below the k-th largest value of each row."""
    acts = torch.relu(x @ enc_w.T + enc_b)
    if k < acts.shape[-1]:
        thresh = acts.topk(k, dim=-1).values[..., -1:]
        acts = torch.where(acts >= thresh, acts, torch.zeros_like(acts))
    return acts


def single_token_id(tok, term: str) -> int | None:
    """Token id of ``term`` if it (or ``" " + term``) is a single token, else None."""
    for variant in (term, " " + term):
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def load_contrast_pairs(bank_path: str | Path, tok, rng: np.random.Generator, max_contrasts: int) -> list[tuple]:
    """Unique single-token ``(a, b, id_a, id_b)`` contrasts of a bank, shuffled with ``rng`` and capped."""
    recs = [json.loads(line) for line in Path(bank_path).read_text().splitlines() if line.strip()]
    pairs: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for r in recs:
        a, b = r.get("target_a"), r.get("target_b")
        if not a or not b or (a, b) in seen:
            continue
        ia, ib = single_token_id(tok, a), single_token_id(tok, b)
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
    _, enc_w, enc_b, k = dictionary
    z = topk_codes(torch.stack([W_n[ia], W_n[ib]]), enc_w, enc_b, k)  # (2, d_features)
    beta = rn[ia] * z[0] - rn[ib] * z[1]
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
    """Union over ``fids`` of the token strings of each feature's top-R centred rows."""
    out: set[str] = set()
    for f in fids:
        rows = torch.topk(W_c @ dec[f], top_r).indices.tolist()
        out |= token_strings(tok, rows, cache)
    return out


def jaccard(a, b) -> float:
    return len(a & b) / max(1, len(a | b))
