"""Shared Qwen readout toolkit: model/SAE loading, encode/topk, label display.

Extracted from ``research.figures.compute_sae_paper_examples`` so the figure
renderer, the data miners, and the analysis scripts share one definition of the
loaders and the feature-label helpers instead of importing them from a figure
module (a figures->run / figures->data dependency inversion).

The model loader is Qwen-bound: it uses ``AutoModelForImageTextToText`` (the
Qwen3.5-2B architecture's auto-class), not the generic causal-LM class, hence
the ``qwen`` in the module name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sparse_readout_prism.token_display import clean_token, is_display_token
from sparse_readout_prism.utils import find_lm_head_with_path as find_lm_head

__all__ = [
    "DISPLAY_EXCLUDE_RE",
    "DISPLAY_TOKEN_RE",
    "ContrastDecomposition",
    "clean_token",
    "collect_readout_state",
    "collect_readout_states_batched",
    "decompose_row_contrast",
    "display_label_features",
    "encode_topk",
    "fallback_feature_label",
    "find_lm_head",
    "load_qwen_model",
    "load_sae",
    "parse_single_token",
    "readable_feature_label",
    "token_summary",
]


@dataclass
class ContrastDecomposition:
    """Per-feature decomposition of a two-row (A-vs-B) centered readout contrast.

    Fields mirror the locals the A/B miners consume downstream:
    ``z`` is the full two-row top-k code (so ``z[0, fid]``/``z[1, fid]`` give the
    per-side activations), ``coeff`` is the full per-feature contrast coefficient
    ``norms[0]*z[0] - norms[1]*z[1]`` (indexed by ``active`` downstream),
    ``active`` are the nonzero coefficient indices, ``feature_scores = h @ decoder[active].T``,
    ``contrib = coeff[active] * feature_scores``, ``feature_sum = contrib.sum()`` and
    ``residual = exact - feature_sum``.
    """

    z: torch.Tensor
    coeff: torch.Tensor
    active: torch.Tensor
    feature_scores: torch.Tensor
    contrib: torch.Tensor
    feature_sum: float
    residual: float


@torch.no_grad()
def decompose_row_contrast(
    *,
    W: torch.Tensor,
    target_a: int,
    target_b: int,
    row_mean: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    decoder: torch.Tensor,
    h: torch.Tensor,
    k: int,
    exact: float,
) -> ContrastDecomposition:
    """Decompose a centered A-vs-B readout contrast into sparse feature terms.

    Centers rows ``a``/``b`` of ``W`` against ``row_mean``, per-row normalizes,
    top-k encodes, forms the contrast coefficient, and projects the hidden state
    ``h`` onto the active decoder rows. Behaviour-preserving extraction of the
    identical block in the Qwen2B A/B paper-example and interesting-feature miners.
    """
    rows = W[[target_a, target_b]].float()
    centered = rows - row_mean
    norms = centered.norm(dim=1).clamp_min(1e-8)
    x = centered / norms[:, None]
    z = encode_topk(x, encoder_w, encoder_b, k=k)
    coeff = norms[0] * z[0] - norms[1] * z[1]
    active = torch.nonzero(coeff != 0, as_tuple=False).flatten()
    feature_scores = h @ decoder[active].T
    contrib = coeff[active] * feature_scores
    feature_sum = float(contrib.sum())
    residual = exact - feature_sum
    return ContrastDecomposition(
        z=z,
        coeff=coeff,
        active=active,
        feature_scores=feature_scores,
        contrib=contrib,
        feature_sum=feature_sum,
        residual=residual,
    )


DISPLAY_EXCLUDE_RE = re.compile(r"(<\||\|>|�|\\n|\\t)")
DISPLAY_TOKEN_RE = re.compile(r"[A-Za-z0-9]")


def parse_single_token(tokenizer, raw: str) -> int:
    ids = tokenizer.encode(raw, add_special_tokens=False)
    if len(ids) != 1:
        labels = [clean_token(tokenizer, token_id) for token_id in ids]
        raise ValueError(f"{raw!r} tokenized to {ids} = {labels}; expected one token")
    return int(ids[0])


def load_qwen_model(model_id: str, dtype: torch.dtype):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def load_sae(checkpoint) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    decoder = state["decoder"].float().contiguous()
    encoder_w = state["encoder.weight"].float().contiguous()
    encoder_b = state["encoder.bias"].float().contiguous()
    decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return decoder, encoder_w, encoder_b, ckpt.get("config", {})


@torch.no_grad()
def encode_topk(x: torch.Tensor, encoder_w: torch.Tensor, encoder_b: torch.Tensor, k: int) -> torch.Tensor:
    acts = F.relu(x @ encoder_w.T + encoder_b)
    kk = min(int(k), acts.shape[-1])
    values, indices = torch.topk(acts, k=kk, dim=-1)
    code = torch.zeros_like(acts)
    code.scatter_(dim=-1, index=indices, src=values)
    return code


def readable_feature_label(labels: list[str], feature_id: int, max_len: int = 36) -> str:
    parts = [part for part in labels if part]
    if not parts:
        return f"f{feature_id}"
    label = f"f{feature_id} " + "/".join(parts[:3])
    label = re.sub(r"\s+", " ", label).strip()
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "..."


def fallback_feature_label(row: dict[str, object], feature_id: int) -> str:
    token = str(row.get("contrast_token_hint", "")).strip()
    if token:
        return f"f{feature_id} {token}"
    return f"f{feature_id}"


def token_summary(row: dict[str, object], *, max_tokens: int = 4) -> str:
    tokens = [part.strip() for part in str(row.get("feature_top_tokens", "")).split(";") if part.strip()]
    if tokens:
        return ", ".join(tokens[:max_tokens])
    hint = str(row.get("contrast_token_hint", "")).strip()
    if hint:
        return f"{hint} row"
    return "contrast row"


@torch.no_grad()
def display_label_features(
    *,
    W: torch.Tensor,
    row_mean: torch.Tensor,
    feature_ids: list[int],
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer,
    top_tokens: int,
    chunk_size: int,
) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {int(fid): [] for fid in feature_ids}
    if not feature_ids:
        return labels
    device = W.device
    fids = torch.tensor(feature_ids, dtype=torch.long, device=encoder_w.device)
    enc = encoder_w[fids].contiguous().to(device)
    bias = encoder_b[fids].contiguous().to(device)
    row_mean = row_mean.to(device)
    vocab_for_labels = min(W.shape[0], len(tokenizer))
    candidate_pool = min(vocab_for_labels, max(top_tokens * 4096, 16384))
    scores = torch.full((len(feature_ids), candidate_pool), -float("inf"), device=device)
    ids = torch.full((len(feature_ids), candidate_pool), -1, dtype=torch.long, device=device)
    for start in range(0, vocab_for_labels, chunk_size):
        rows = W[start : start + chunk_size].float()
        centered = rows - row_mean
        norms = centered.norm(dim=1).clamp_min(1e-8)
        x = centered / norms[:, None]
        chunk_scores = F.relu(x @ enc.T + bias).T
        merged_scores = torch.cat([scores, chunk_scores], dim=1)
        chunk_ids = torch.arange(start, start + rows.shape[0], dtype=torch.long, device=device).expand(
            len(feature_ids), -1
        )
        merged_ids = torch.cat([ids, chunk_ids], dim=1)
        scores, keep = torch.topk(merged_scores, k=scores.shape[1], dim=1)
        ids = torch.gather(merged_ids, 1, keep)
    for row_idx, fid in enumerate(feature_ids):
        seen: set[str] = set()
        out: list[str] = []
        for token_id in ids[row_idx].tolist():
            token = clean_token(tokenizer, int(token_id))
            if not is_display_token(token):
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= top_tokens:
                break
        labels[int(fid)] = out
    return labels


@torch.no_grad()
def collect_readout_state(model, tokenizer, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lm_head, _path = find_lm_head(model)
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        captured["h"] = (inputs[0] if isinstance(inputs, tuple) else inputs).detach()

    handle = lm_head.register_forward_pre_hook(pre_hook)
    try:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model(**enc)
    finally:
        handle.remove()
    h = captured["h"][0, -1].float().cpu().contiguous()
    logits = out.logits[0, -1].float().cpu().contiguous()
    return h, logits


@torch.no_grad()
def collect_readout_states_batched(model, tokenizer, prompts: list[str], device: torch.device, batch_size: int):
    """Batched last-position readout states + logits for a list of prompts.

    Like ``collect_readout_state`` but runs prompts in padded batches and reads
    the last non-pad position per row. Returns ``(states, logits)`` lists of
    per-prompt CPU tensors (each ``(d_model,)`` / ``(vocab,)``).
    """
    lm_head, _ = find_lm_head(model)
    states: list[torch.Tensor] = []
    logits_out: list[torch.Tensor] = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        captured: dict[str, torch.Tensor] = {}

        def pre_hook(_module, inputs):
            captured["h"] = (inputs[0] if isinstance(inputs, tuple) else inputs).detach()

        handle = lm_head.register_forward_pre_hook(pre_hook)
        try:
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = model(**enc)
        finally:
            handle.remove()
        last_idx = enc["attention_mask"].sum(dim=1).cpu() - 1
        hidden = captured["h"].float().cpu()
        logits = out.logits.float().cpu()
        for i, idx in enumerate(last_idx.tolist()):
            states.append(hidden[i, idx].contiguous())
            logits_out.append(logits[i, idx].contiguous())
    return states, logits_out
