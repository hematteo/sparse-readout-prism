from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from sparse_readout_prism.data import PrismDataset
from sparse_readout_prism.decompose import batch_decomposition_identity_error
from sparse_readout_prism.factorizers import SAEBase


@torch.no_grad()
def reconstruct_normalized_rows(
    model: SAEBase,
    rows_normalized: torch.Tensor,
    k: int,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Returns (recon, usage, per_row_l0)."""
    recons = []
    usage = torch.zeros(model.d_features, dtype=torch.float64)
    per_row_l0: list[int] = []
    device = next(model.parameters()).device
    for start in range(0, rows_normalized.shape[0], batch_size):
        x = rows_normalized[start : start + batch_size].to(device)
        batch = model(x, k=k)
        recons.append(batch.reconstruction.detach().cpu())
        code_cpu = batch.code.detach().cpu()
        nonzero = code_cpu > 0
        usage += nonzero.sum(dim=0).double()
        per_row_l0.extend(nonzero.sum(dim=1).tolist())
    return torch.cat(recons, dim=0), usage, per_row_l0


def resolve_eval_k(config: dict[str, Any], model: SAEBase) -> int:
    """The k used at evaluation time: evaluation.k, else factorizer.k, else model.k.

    The single definition of that fallback chain (evaluate + runner both use it).
    The `or` chain is deliberate: an explicit ``evaluation.k: 0``/``null`` means
    "defer to the factorizer/model k", not "k = 0".
    """
    return int(config.get("evaluation", {}).get("k") or config.get("factorizer", {}).get("k", model.k))


@torch.no_grad()
def evaluate_model(
    model: SAEBase,
    dataset: PrismDataset,
    config: dict[str, Any],
    device: torch.device,
    *,
    return_usage: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], torch.Tensor]:
    """Compute the full eval-metric dict; ``return_usage=True`` also returns the
    per-feature usage counts from the reconstruction pass (so callers such as
    the runner don't repeat the whole O(rows x d_features) pass for them)."""
    eval_cfg = config.get("evaluation", {})
    k = resolve_eval_k(config, model)
    max_eval_rows = eval_cfg.get("max_eval_rows")
    max_eval_hidden = int(eval_cfg.get("max_eval_hidden") or dataset.hidden_val.shape[0])
    contribution_top_k = int(eval_cfg.get("contribution_top_k") or 8)
    topk_overlap = int(eval_cfg.get("topk_overlap") or 5)

    n_rows = min(int(max_eval_rows or dataset.vocab_size), dataset.vocab_size)
    rows_normalized = dataset.rows_normalized[:n_rows]
    W_U = dataset.W_U[:n_rows]
    row_norms = dataset.row_norms[:n_rows]

    model.eval()
    recon_x, usage, per_row_l0 = reconstruct_normalized_rows(model, rows_normalized, k=k)
    recon_W = dataset.row_mean[None, :] + row_norms[:, None] * recon_x

    residual_x = rows_normalized - recon_x
    row_centered_ev = 1.0 - (residual_x.pow(2).sum() / rows_normalized.pow(2).sum().clamp_min(1e-12)).item()
    row_centered_cosine = F.cosine_similarity(rows_normalized, recon_x, dim=1).mean().item()

    hidden = dataset.hidden_val[:max_eval_hidden]
    logits_orig = hidden @ W_U.T
    logits_rec = hidden @ recon_W.T

    logp = F.log_softmax(logits_orig, dim=1)
    logq = F.log_softmax(logits_rec, dim=1)
    p = logp.exp()
    val_logit_kl_bits_mean = (p * (logp - logq)).sum(dim=1).mean().item() / math.log(2.0)

    top1_orig = logits_orig.argmax(dim=1)
    top1_rec = logits_rec.argmax(dim=1)
    val_top1_match = (top1_orig == top1_rec).float().mean().item()

    k_overlap = min(topk_overlap, W_U.shape[0])
    top_orig = torch.topk(logits_orig, k=k_overlap, dim=1).indices
    top_rec = torch.topk(logits_rec, k=k_overlap, dim=1).indices
    overlap = []
    for a, b in zip(top_orig, top_rec, strict=False):
        overlap.append(len(set(a.tolist()) & set(b.tolist())) / float(k_overlap))
    val_top5_overlap = float(sum(overlap) / len(overlap)) if overlap else 0.0

    top_rows = top1_orig
    orig_top_logits = logits_orig[torch.arange(hidden.shape[0]), top_rows]
    rec_top_logits = logits_rec[torch.arange(hidden.shape[0]), top_rows]
    residual_logits = orig_top_logits - rec_top_logits
    top1_logit_residual_abs_mean = residual_logits.abs().mean().item()
    top1_logit_residual_frac_mean = (residual_logits.abs() / orig_top_logits.abs().clamp_min(1e-6)).mean().item()

    coverage_pos, coverage_abs = contribution_coverages(
        model=model,
        hidden=hidden.to(device),
        rows_normalized=rows_normalized[top_rows].to(device),
        row_norms=row_norms[top_rows].to(device),
        k=k,
        contribution_top_k=contribution_top_k,
    )

    identity_n = min(hidden.shape[0], top_rows.shape[0])
    identity_errors = batch_decomposition_identity_error(
        h=hidden[:identity_n].to(device),
        W_rows=W_U[top_rows[:identity_n]].to(device),
        row_mean=dataset.row_mean.to(device),
        row_norms=row_norms[top_rows[:identity_n]].to(device),
        rows_normalized=rows_normalized[top_rows[:identity_n]].to(device),
        model=model,
        k=k,
    )

    dead_feature_rate = (usage == 0).double().mean().item()
    usage_total = usage.sum().clamp_min(1.0)
    p_usage = usage / usage_total
    nonzero_p = p_usage[p_usage > 0]
    feature_usage_entropy = (
        (-(nonzero_p * nonzero_p.log()).sum() / math.log(max(2, model.d_features))).item() if nonzero_p.numel() else 0.0
    )

    # L0 distribution (eval-time, on the eval rows). Per-row count of features
    # used by the encoder pass.
    if per_row_l0:
        l0_tensor = torch.tensor(per_row_l0, dtype=torch.float64)
        mean_l0 = float(l0_tensor.mean().item())
        l0_p10 = float(torch.quantile(l0_tensor, 0.10).item())
        l0_p50 = float(torch.quantile(l0_tensor, 0.50).item())
        l0_p90 = float(torch.quantile(l0_tensor, 0.90).item())
    else:
        mean_l0 = l0_p10 = l0_p50 = l0_p90 = float("nan")

    # Rare/always-on rates: feature firing on <0.1% / >50% of eval rows.
    fire_rate = usage / max(1, n_rows)
    rare_feature_rate = float((fire_rate < 1e-3).double().mean().item())
    always_on_feature_rate = float((fire_rate > 0.5).double().mean().item())

    # Duplicate decoder: fraction of rows whose max cosine to any other row is ≥0.9.
    duplicate_decoder_frac = _duplicate_decoder_frac(model.decoder, threshold=0.9)

    metrics: dict[str, Any] = {
        "k": k,
        "eval_rows": n_rows,
        "eval_hidden": int(hidden.shape[0]),
        "contribution_top_k": contribution_top_k,
        "topk_overlap": topk_overlap,
        "row_centered_ev": row_centered_ev,
        "row_centered_cosine": row_centered_cosine,
        "top1_logit_residual_frac_mean": top1_logit_residual_frac_mean,
        "top1_logit_residual_abs_mean": top1_logit_residual_abs_mean,
        # Key names carry the configured widths (default 8 / 5) so a non-default
        # `contribution_top_k` / `topk_overlap` can't be mislabeled as top8/top5.
        f"top{contribution_top_k}_positive_contrib_coverage_mean": coverage_pos,
        f"top{contribution_top_k}_abs_contrib_coverage_mean": coverage_abs,
        "val_logit_kl_bits_mean": val_logit_kl_bits_mean,
        "val_top1_match": val_top1_match,
        f"val_top{topk_overlap}_overlap": val_top5_overlap,
        "dead_feature_rate": dead_feature_rate,
        "feature_usage_entropy": feature_usage_entropy,
        "mean_l0": mean_l0,
        "l0_p10": l0_p10,
        "l0_p50": l0_p50,
        "l0_p90": l0_p90,
        "rare_feature_rate": rare_feature_rate,
        "always_on_feature_rate": always_on_feature_rate,
        "duplicate_decoder_frac": duplicate_decoder_frac,
        "identity_abs_error_max": identity_errors.max().item(),
        "identity_abs_error_mean": identity_errors.mean().item(),
        "finite_metrics": True,
    }
    metrics["selection_score"] = selection_score(metrics, config)
    if return_usage:
        return metrics, usage
    return metrics


@torch.no_grad()
def _duplicate_decoder_frac(decoder: torch.Tensor, threshold: float = 0.9, chunk: int = 4096) -> float:
    """Fraction of decoder rows whose max cosine to any *other* row is >= threshold.

    Computed in chunks to avoid O(D^2) memory at D≈25k.
    """
    D = decoder.shape[0]
    if D <= 1:
        return 0.0
    d_norm = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
    d_norm_cpu = d_norm.detach().cpu()
    dup = 0
    for start in range(0, D, chunk):
        end = min(start + chunk, D)
        block = d_norm_cpu[start:end]  # (b, d_model)
        sim = block @ d_norm_cpu.T  # (b, D)
        # Mask self-similarity.
        idx = torch.arange(start, end)
        sim[torch.arange(end - start), idx] = -1.0
        block_max = sim.max(dim=1).values
        dup += int((block_max >= threshold).sum().item())
    return dup / D


@torch.no_grad()
def contribution_coverages(
    model: SAEBase,
    hidden: torch.Tensor,
    rows_normalized: torch.Tensor,
    row_norms: torch.Tensor,
    k: int,
    contribution_top_k: int,
) -> tuple[float, float]:
    code = model.encode(rows_normalized, k=k)
    feature_scores = hidden @ model.decoder.T
    contrib = row_norms[:, None] * code * feature_scores
    pos = contrib.clamp_min(0)
    abs_contrib = contrib.abs()
    kk = min(contribution_top_k, contrib.shape[1])

    pos_total = pos.sum(dim=1)
    pos_top = torch.topk(pos, k=kk, dim=1).values.sum(dim=1)
    pos_cov = torch.where(pos_total > 1e-12, pos_top / pos_total, torch.ones_like(pos_total))

    abs_total = abs_contrib.sum(dim=1)
    abs_top = torch.topk(abs_contrib, k=kk, dim=1).values.sum(dim=1)
    abs_cov = torch.where(abs_total > 1e-12, abs_top / abs_total, torch.ones_like(abs_total))
    return pos_cov.mean().item(), abs_cov.mean().item()


def selection_score(metrics: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    """Model-selection score used by the sweep configs to rank cells within a grid.

    Sweep plumbing, not a paper metric: the weights are ad hoc (chosen once
    during the Appendix C–E frontier sweeps and frozen), and the sentinel tiers
    order failure modes — -1e9 non-finite metrics, -1e6 dead dictionary,
    -1e5 KL/identity gate failures — so broken runs sort below every finite
    score. Gate thresholds come from ``autoresearch.score_thresholds`` in the
    sweep configs (see configs/sweeps/README.md); the identity gate (1e-3) is
    fixed because a violated identity means the decomposition is wrong, full stop.
    """
    if not all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float))):
        return -1e9
    thresholds = ((config or {}).get("autoresearch", {}) or {}).get("score_thresholds", {})
    max_dead = float(thresholds.get("max_dead_feature_rate", 0.98))
    max_kl = float(thresholds.get("max_val_logit_kl_bits_mean", 20.0))
    if metrics.get("dead_feature_rate", 1.0) > max_dead:
        return -1e6 + (1.0 - metrics.get("dead_feature_rate", 1.0))
    if metrics.get("val_logit_kl_bits_mean", 1e9) > max_kl:
        return -1e5 - metrics.get("val_logit_kl_bits_mean", 0.0)
    if metrics.get("identity_abs_error_max", 1.0) > 1e-3:
        return -1e5 - metrics.get("identity_abs_error_max", 0.0)

    # Coverage/overlap key names carry the configured widths (see evaluate_model).
    ck = int(metrics.get("contribution_top_k", 8))
    ov = int(metrics.get("topk_overlap", 5))
    return float(
        1.5 * metrics.get("row_centered_ev", 0.0)
        + 0.5 * metrics.get("row_centered_cosine", 0.0)
        + 0.7 * metrics.get(f"top{ck}_abs_contrib_coverage_mean", 0.0)
        + 0.4 * metrics.get(f"top{ck}_positive_contrib_coverage_mean", 0.0)
        + 0.5 * metrics.get("val_top1_match", 0.0)
        + 0.3 * metrics.get(f"val_top{ov}_overlap", 0.0)
        - 0.8 * metrics.get("top1_logit_residual_frac_mean", 0.0)
        - 0.03 * metrics.get("val_logit_kl_bits_mean", 0.0)
        - 0.2 * metrics.get("dead_feature_rate", 0.0)
    )


@torch.no_grad()
def label_features(
    model: SAEBase,
    dataset: PrismDataset,
    usage: torch.Tensor | None = None,
    top_n: int = 6,
    max_features: int | None = None,
) -> list[dict[str, Any]]:
    """Label each learned feature with its top-scoring vocabulary tokens."""
    device = next(model.parameters()).device
    rows = dataset.rows_normalized.to(device)
    decoder = model.decoder.detach()
    n_features = model.d_features if max_features is None else min(max_features, model.d_features)
    # Only score the features we actually label; the full (vocab, d_features)
    # matrix is ~64 GB for Ministral (131k×131k×4B) and OOMs an A40.
    scores = rows @ decoder[:n_features].T
    labels = []
    for feature_id in range(n_features):
        top = torch.topk(scores[:, feature_id], k=min(top_n, scores.shape[0])).indices.cpu()
        token_ids = [int(dataset.row_token_ids[i]) for i in top]
        token_labels = [dataset.token_labels[i] for i in top]
        label = "/".join(token_labels[:3])
        item: dict[str, Any] = {
            "feature_id": feature_id,
            "label": label,
            "top_token_ids": token_ids,
            "top_token_labels": token_labels,
        }
        if usage is not None:
            item["usage_count"] = float(usage[feature_id].item())
        labels.append(item)
    return labels
