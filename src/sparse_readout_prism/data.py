from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class PrismDataset:
    W_U: torch.Tensor
    rows_normalized: torch.Tensor
    row_mean: torch.Tensor
    row_norms: torch.Tensor
    hidden_train: torch.Tensor
    hidden_val: torch.Tensor
    h_LN_grid: torch.Tensor
    row_token_ids: torch.Tensor
    # Placeholder "tok_<id>" strings; real decoded labels come from a tokenizer
    # later (research.qwen_readout.display_label_features), never from here.
    token_labels: list[str]
    top_targets: torch.Tensor
    top_counts: torch.Tensor
    source_path: str
    used_fallback: bool
    # True when val_hidden could not be carved out and val metrics are computed
    # on (a prefix of) the training rows — see load_prism_dataset.
    val_overlaps_train: bool = False

    @property
    def vocab_size(self) -> int:
        return int(self.W_U.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_U.shape[1])


def load_prism_dataset(config: dict[str, Any], seed: int) -> PrismDataset:
    """Load dataset.

    `seed` is the legacy single seed (back-compat). For paired-architecture
    comparisons, set `data.data_seed` in the config — this controls hidden-state
    sampling and row-subset selection. The model `init_seed` is consumed
    elsewhere (see set_seed in run_experiment) so the val split is identical
    across cells that share `data_seed`.
    """
    data_cfg = config.get("data", {})
    # expandvars so configs can reference ${SRP_SSD_ROOT}/... etc.; an unset var
    # stays literal -> not a file -> synthetic fallback below.
    path_str = os.path.expandvars(data_cfg.get("path") or "")
    path = Path(path_str) if path_str else None
    fallback = data_cfg.get("fallback", "synthetic")
    max_rows = data_cfg.get("max_rows")
    max_hidden = int(data_cfg.get("max_hidden") or 512)
    val_hidden = int(data_cfg.get("val_hidden") or min(128, max_hidden))
    data_seed = int(data_cfg.get("data_seed", seed))

    used_fallback = False
    token_mask = None
    if path is not None and path.is_file():
        raw = torch.load(path, map_location="cpu", weights_only=True)
        W_U = raw.get("W_U_orig", raw.get("W_U"))
        h_LN = raw.get("h_LN")
        token_mask = raw.get("token_mask")
        if W_U is None or h_LN is None:
            raise KeyError(f"{path} must contain W_U_orig/W_U and h_LN")
        source_path = str(path)
    elif fallback == "synthetic":
        W_U, h_LN = make_synthetic_data(seed=data_seed)
        source_path = "synthetic"
        used_fallback = True
    elif fallback == "error":
        raise FileNotFoundError(
            f"data.path {path!r} does not exist and fallback is set to 'error'. "
            "Sync the dataset to the cluster or change the config."
        )
    else:
        raise FileNotFoundError(path)

    W_U = W_U.detach().float().cpu()
    h_LN = h_LN.detach().float().cpu()
    if h_LN.shape[-1] != W_U.shape[-1]:
        raise ValueError(f"hidden dim {h_LN.shape[-1]} does not match W_U dim {W_U.shape[-1]}")

    # Filter non-text rows (vision/special tokens for multimodal vocabs).
    # token_mask survives as a row-level filter applied BEFORE choose_row_subset
    # so frequency stats are computed only over kept tokens.
    if token_mask is not None and data_cfg.get("apply_token_mask", True):
        token_mask = token_mask.bool().cpu()
        if token_mask.numel() != W_U.shape[0]:
            raise ValueError(f"token_mask length {token_mask.numel()} does not match vocab {W_U.shape[0]}")
        # We keep a global vocab-sized indexing intact for top_targets later,
        # so instead of slicing W_U here, we override choose_row_subset's
        # candidate pool via max_rows + the mask.
    else:
        token_mask = None  # apply_token_mask: false really does disable it

    hidden_flat = h_LN.reshape(-1, h_LN.shape[-1])
    # Drop padding zeros from extracted-corpus h_LN where short docs were padded.
    keep = hidden_flat.norm(dim=1) > 1e-6
    if keep.any() and keep.sum() < hidden_flat.shape[0]:
        hidden_flat = hidden_flat[keep]
    hidden_sample = sample_rows(hidden_flat, max_hidden + val_hidden, seed=data_seed + 11)
    hidden_train = hidden_sample[:max_hidden].contiguous()
    hidden_val = hidden_sample[max_hidden : max_hidden + val_hidden].contiguous()
    val_overlaps_train = False
    if hidden_val.numel() == 0:
        # Corpus smaller than max_hidden: fall back to a training prefix so val
        # metrics still compute, but flag it — they are NOT held out.
        hidden_val = hidden_train[: min(64, hidden_train.shape[0])].contiguous()
        val_overlaps_train = True

    row_ids = choose_row_subset(W_U, hidden_train, max_rows=max_rows, seed=data_seed + 23, token_mask=token_mask)
    W_U = W_U[row_ids].contiguous()
    row_token_ids = row_ids.cpu()
    token_labels = [f"tok_{int(i)}" for i in row_token_ids]

    row_mean, row_norms, rows_normalized = preprocess_rows(W_U)
    top_targets, top_counts = compute_top_targets(W_U, hidden_train, top_n=8)

    return PrismDataset(
        W_U=W_U,
        rows_normalized=rows_normalized,
        row_mean=row_mean,
        row_norms=row_norms,
        hidden_train=hidden_train,
        hidden_val=hidden_val,
        h_LN_grid=h_LN,
        row_token_ids=row_token_ids,
        token_labels=token_labels,
        top_targets=top_targets,
        top_counts=top_counts,
        source_path=source_path,
        used_fallback=used_fallback,
        val_overlaps_train=val_overlaps_train,
    )


def make_synthetic_data(
    seed: int,
    vocab_size: int = 1024,
    d_model: int = 128,
    layers: int = 8,
    positions: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(96, d_model, generator=gen)
    token_codes = torch.randn(vocab_size, 96, generator=gen)
    token_codes = F.relu(token_codes)
    token_codes[token_codes < token_codes.quantile(0.95)] = 0
    W_U = token_codes @ latent / (96**0.5)
    W_U += 0.05 * torch.randn(vocab_size, d_model, generator=gen)
    h = torch.randn(layers, positions, d_model, generator=gen)
    h = h + 0.15 * torch.sin(torch.linspace(0, 6.28, layers)).view(-1, 1, 1)
    return W_U.float(), h.float()


def sample_rows(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    # Shuffle even when keeping everything: callers split the result into
    # train/val by position, so an unshuffled pass-through would silently make
    # the split a corpus-order prefix in the small-data regime.
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=gen)[: min(n, x.shape[0])]
    return x[idx]


def preprocess_rows(W_U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center + per-row normalize a full row matrix. Returns (row_mean, row_norms, rows_normalized)."""
    row_mean = W_U.mean(dim=0)
    row_norms, rows_normalized = center_normalize_rows(W_U, row_mean)
    return row_mean, row_norms, rows_normalized


def center_normalize_rows(rows: torch.Tensor, row_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center ``rows`` against a given ``row_mean`` and per-row L2-normalize.

    Returns ``(norms, rows_normalized)``. The one definition of the training
    preprocessing applied to any row subset at decompose time — the 1e-8 norm
    floor here must stay in sync with training, so do not reimplement inline.
    """
    centered = rows - row_mean
    norms = centered.norm(dim=1).clamp_min(1e-8)
    return norms, centered / norms[:, None]


def resolve_row_mean(
    W_U: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    ckpt: dict[str, Any] | None = None,
) -> torch.Tensor:
    """The centering mean matched to how the dictionary was trained.

    Preference order: the checkpoint's stored ``row_mean`` (exact training
    value), else the mean over ``token_mask``-kept rows (how a masked
    extraction trains), else the full-vocab mean. Evaluation scripts must use
    this rather than ad-hoc ``W_U.mean(dim=0)`` — centering against a different
    mean than training silently changes every decomposition on masked
    (multimodal) vocabularies.
    """
    if ckpt is not None:
        stored = ckpt.get("row_mean")
        if stored is not None:
            return stored.detach().float().cpu()
    if token_mask is not None:
        return W_U[token_mask.bool()].float().mean(dim=0).cpu()
    return W_U.float().mean(dim=0).cpu()


def choose_row_subset(
    W_U: torch.Tensor,
    hidden: torch.Tensor,
    max_rows: int | None,
    seed: int,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    vocab = W_U.shape[0]

    # Restrict candidate pool to allowed tokens if a mask is given.
    if token_mask is not None:
        allowed = token_mask.bool()
        if allowed.numel() != vocab:
            raise ValueError("token_mask size mismatch")
    else:
        allowed = torch.ones(vocab, dtype=torch.bool)
    allowed_idx = torch.nonzero(allowed, as_tuple=False).squeeze(1)

    if max_rows is None or max_rows >= allowed_idx.numel():
        return allowed_idx

    keep = int(max_rows)
    top_targets, counts = compute_top_targets(W_U, hidden[: min(hidden.shape[0], 256)], top_n=5)
    del top_targets
    # Zero-out counts for disallowed tokens so frequency-ranked picks stay clean.
    counts = counts.clone()
    counts[~allowed] = 0
    ranked = torch.argsort(counts, descending=True)
    active = ranked[counts[ranked] > 0]
    take_active = active[: min(active.numel(), keep // 2)]

    gen = torch.Generator().manual_seed(seed)
    perm = allowed_idx[torch.randperm(allowed_idx.numel(), generator=gen)]
    mask = torch.ones(vocab, dtype=torch.bool)
    mask[take_active] = False
    fill = perm[mask[perm]][: keep - take_active.numel()]
    row_ids = torch.cat([take_active, fill]).unique()
    if row_ids.numel() < keep:
        remaining = perm[~torch.isin(perm, row_ids)][: keep - row_ids.numel()]
        row_ids = torch.cat([row_ids, remaining])
    return row_ids[:keep]


def compute_top_targets(
    W_U: torch.Tensor,
    hidden: torch.Tensor,
    top_n: int = 8,
    hidden_batch: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    top_n = min(top_n, W_U.shape[0])
    targets = []
    counts = torch.zeros(W_U.shape[0], dtype=torch.long)
    Wt = W_U.T.contiguous()
    for start in range(0, hidden.shape[0], hidden_batch):
        h = hidden[start : start + hidden_batch]
        logits = h @ Wt
        top = torch.topk(logits, k=top_n, dim=1).indices.cpu()
        targets.append(top)
        counts.scatter_add_(0, top.reshape(-1), torch.ones(top.numel(), dtype=torch.long))
    return torch.cat(targets, dim=0), counts


def row_sampling_weights(dataset: PrismDataset, mode: str) -> torch.Tensor:
    mode = mode or "uniform"
    if mode == "uniform":
        weights = torch.ones(dataset.vocab_size)
    elif mode == "frequency":
        weights = dataset.top_counts.float().clamp_min(1.0)
    elif mode in {"frequency_sqrt", "frequency_pow_0.5"}:
        weights = dataset.top_counts.float().clamp_min(1.0).pow(0.5)
    elif mode == "frequency_pow_0.75":
        weights = dataset.top_counts.float().clamp_min(1.0).pow(0.75)
    elif mode.startswith("hybrid_") and "freq_" in mode and mode.endswith("uniform"):
        mix = mode.removeprefix("hybrid_").removesuffix("uniform")
        freq_raw, uniform_raw = mix.split("freq_", 1)
        freq_pct = float(freq_raw)
        uniform_pct = float(uniform_raw)
        total_pct = max(freq_pct + uniform_pct, 1e-8)
        freq_weight = freq_pct / total_pct
        freq = dataset.top_counts.float().clamp_min(1.0)
        freq = freq / freq.sum()
        uniform = torch.ones(dataset.vocab_size) / dataset.vocab_size
        weights = freq_weight * freq + (1.0 - freq_weight) * uniform
    elif mode == "top1_sqrt_frequency":
        top1 = dataset.top_targets[:, 0]
        counts = torch.zeros(dataset.vocab_size, dtype=torch.float32)
        counts.scatter_add_(0, top1, torch.ones_like(top1, dtype=torch.float32))
        weights = counts.clamp_min(1.0).sqrt()
    elif mode == "top5_sqrt_frequency":
        counts = torch.zeros(dataset.vocab_size, dtype=torch.float32)
        top = dataset.top_targets[:, : min(5, dataset.top_targets.shape[1])].reshape(-1)
        counts.scatter_add_(0, top, torch.ones_like(top, dtype=torch.float32))
        weights = counts.clamp_min(1.0).sqrt()
    else:
        raise ValueError(f"Unknown row_sampling mode: {mode}")
    return weights / weights.sum()
