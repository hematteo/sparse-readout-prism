from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch

from sparse_readout_prism.data import PrismDataset, row_sampling_weights
from sparse_readout_prism.factorizers import SAEBase, reconstruction_loss


def train_factorizer(
    model: SAEBase,
    dataset: PrismDataset,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    factor_cfg = config.get("factorizer", {})
    steps = int(train_cfg.get("steps", 100))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    lambda_prism = float(train_cfg.get("lambda_prism", 0.0))
    prism_top_n = int(train_cfg.get("prism_top_n", 1))
    code_l2 = float(train_cfg.get("code_l2", 0.0))
    log_every = int(train_cfg.get("log_every", 50))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 500))

    architecture = factor_cfg.get("architecture", "topk")
    k = int(factor_cfg.get("k", getattr(model, "k", model.d_features)))
    k_values = [int(x) for x in factor_cfg.get("k_values", [])]
    weights = [float(x) for x in factor_cfg.get("matryoshka_weights", [])]
    if weights and len(weights) != len(k_values):
        weights = None

    # Architecture-specific sparsity coefficients (raw value × coefficient added to loss).
    l0_coeff = float(train_cfg.get("l0_coeff", 0.0))  # JumpReLU
    l1_coeff = float(train_cfg.get("l1_coeff", 0.0))  # L1 ReLU
    gate_l1_coeff = float(train_cfg.get("gate_l1_coeff", 0.0))  # Gated SAE sparsity
    gated_aux_coeff = float(train_cfg.get("gated_aux_coeff", 1.0))  # Gated aux recon weight

    # L0 controller: a multiplicative proportional controller that adapts the
    # sparsity coefficient each step so achieved L0 tracks a target. Makes the
    # coefficient-driven architectures (jumprelu, l1_relu) land at a pinned L0
    # regardless of training horizon — required for a matched-L0 frontier.
    # TopK-family architectures are unaffected (k IS the L0 control).
    ctrl_cfg = dict(train_cfg.get("l0_controller", {}) or {})
    ctrl_enabled = bool(ctrl_cfg.get("enabled", False)) and architecture in {
        "jumprelu",
        "l1_relu",
    }
    ctrl_target = float(ctrl_cfg.get("target_l0", 0.0))
    ctrl_kp = float(ctrl_cfg.get("kp", 0.2))
    ctrl_ema = float(ctrl_cfg.get("ema", 0.9))
    ctrl_warmup = int(ctrl_cfg.get("warmup_steps", 200))
    ctrl_period = max(1, int(ctrl_cfg.get("control_period", 10)))
    # Clip relative error so a single update can move the coefficient by at most
    # exp(kp * err_clip) — prevents integral windup when |L0 - target| is huge
    # at init (e.g. L1 starts near-dense).
    ctrl_err_clip = float(ctrl_cfg.get("err_clip", 0.3))
    ctrl_coeff = float(ctrl_cfg.get("coeff_init", l0_coeff or l1_coeff or 1e-4))
    ctrl_coeff_min = float(ctrl_cfg.get("coeff_min", 1e-8))
    # JumpReLU's L0-estimate penalty scales with d_features (hundreds), so the
    # same coeff is far more lethal than L1's (~tens). JumpReLU collapse is also
    # irreversible (STE gradient is zero outside the bandwidth window), so cap
    # it tighter to make overshoot impossible.
    _default_coeff_max = 0.01 if architecture == "jumprelu" else 0.05
    ctrl_coeff_max = float(ctrl_cfg.get("coeff_max", _default_coeff_max))
    ctrl_l0_ema: float | None = None

    x_rows = dataset.rows_normalized.to(device)
    W_U = dataset.W_U.to(device)
    row_mean = dataset.row_mean.to(device)
    row_norms = dataset.row_norms.to(device)
    hidden = dataset.hidden_train.to(device)
    top_targets = dataset.top_targets[:, : max(1, min(prism_top_n, dataset.top_targets.shape[1]))].to(device)
    sample_weights = row_sampling_weights(dataset, train_cfg.get("row_sampling", "uniform")).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[dict[str, Any]] = []
    elapsed_prev = 0.0
    start_step = 1

    ckpt_path = (output_dir / "train_ckpt.pt") if output_dir is not None else None
    if ckpt_path is not None and ckpt_path.exists():
        try:
            # train_ckpt.pt is internal resume scratch this process wrote itself
            # (optimizer + RNG state) and deletes at run end — never shipped or
            # downloaded — so weights_only=False is fine here, unlike the released
            # checkpoint loaders which use weights_only=True.
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            opt.load_state_dict(state["opt_state_dict"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            cuda_rng = state.get("cuda_rng_state")
            if cuda_rng is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in cuda_rng])
            start_step = int(state["step"]) + 1
            losses = list(state.get("losses", []))
            elapsed_prev = float(state.get("elapsed_seconds", 0.0))
            if ctrl_enabled and "ctrl_coeff" in state:
                ctrl_coeff = float(state["ctrl_coeff"])
                ctrl_l0_ema = state.get("ctrl_l0_ema")
            print(
                f"[resume] {ckpt_path.parent.name}: step {start_step - 1}/{steps} ({elapsed_prev:.1f}s already)",
                flush=True,
            )
        except Exception as exc:
            print(f"[resume] failed to load {ckpt_path}: {exc}; starting from scratch", flush=True)
            start_step = 1
            losses = []
            elapsed_prev = 0.0

    start_time = time.time()
    if start_step > steps:
        return {
            "losses": losses,
            "train_seconds": elapsed_prev,
            "mean_l0_train": float("nan"),
            "steps_run": 0,
            "batch_size": batch_size,
        }
    l0_running = 0.0
    l0_running_count = 0
    for step in range(start_step, steps + 1):
        row_idx = torch.multinomial(sample_weights, num_samples=batch_size, replacement=True)
        x = x_rows[row_idx]
        rec_loss, code, aux_loss, aux_info = reconstruction_loss(
            model,
            x,
            architecture=architecture,
            k=k,
            k_values=k_values,
            weights=weights,
        )
        # Measured L0 this step (used by controller and end-of-run reporting).
        with torch.no_grad():
            step_l0 = float((code > 0).float().sum(dim=-1).mean().item())

        # L0 controller update (jumprelu/l1_relu only). Multiplicative
        # proportional law on relative error, EMA-smoothed, post-warmup.
        if ctrl_enabled:
            ctrl_l0_ema = step_l0 if ctrl_l0_ema is None else ctrl_ema * ctrl_l0_ema + (1.0 - ctrl_ema) * step_l0
            if step > ctrl_warmup and ctrl_target > 0 and step % ctrl_period == 0:
                rel_err = (ctrl_l0_ema - ctrl_target) / ctrl_target
                rel_err = max(-ctrl_err_clip, min(ctrl_err_clip, rel_err))
                # L0 too high (not sparse enough) -> raise the penalty coeff.
                ctrl_coeff *= math.exp(ctrl_kp * rel_err)
                ctrl_coeff = min(max(ctrl_coeff, ctrl_coeff_min), ctrl_coeff_max)

        loss = rec_loss
        sparsity_term = x.new_tensor(0.0)
        eff_l0_coeff = ctrl_coeff if ctrl_enabled else l0_coeff
        eff_l1_coeff = ctrl_coeff if ctrl_enabled else l1_coeff
        if architecture == "jumprelu" and eff_l0_coeff > 0:
            sparsity_term = eff_l0_coeff * aux_loss
            loss = loss + sparsity_term
        elif architecture == "l1_relu" and eff_l1_coeff > 0:
            sparsity_term = eff_l1_coeff * aux_loss
            loss = loss + sparsity_term
        elif architecture == "gated":
            if gate_l1_coeff > 0:
                sparsity_term = gate_l1_coeff * aux_loss
                loss = loss + sparsity_term
            if "gated_aux_recon_mse" in aux_info:
                loss = loss + gated_aux_coeff * aux_info["gated_aux_recon_mse"]

        # Track training-time L0 (last 1k steps average is reported at end).
        if step >= steps - 1000:
            l0_running += step_l0
            l0_running_count += 1

        prism_loss = x.new_tensor(0.0)
        if lambda_prism > 0 and hidden.numel() > 0:
            prism_loss = selected_prism_loss(
                model=model,
                W_U=W_U,
                rows_normalized=x_rows,
                row_mean=row_mean,
                row_norms=row_norms,
                hidden=hidden,
                top_targets=top_targets,
                batch_size=min(batch_size, hidden.shape[0]),
                k=k,
            )
            loss = loss + lambda_prism * prism_loss
        if code_l2 > 0:
            loss = loss + code_l2 * code.pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Renormalize decoder for architectures that depend on it. JumpReLU
        # treats threshold as the sparsity knob; renormalizing would couple it.
        if architecture not in {"jumprelu"}:
            model.normalize_decoder_()

        if step == 1 or step % log_every == 0 or step == steps:
            losses.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu().item()),
                    "reconstruction_loss": float(rec_loss.detach().cpu().item()),
                    "prism_loss": float(prism_loss.detach().cpu().item()),
                }
            )

        if ckpt_path is not None and checkpoint_every > 0 and (step % checkpoint_every == 0 or step == steps):
            _save_train_checkpoint(
                ckpt_path,
                step=step,
                model=model,
                opt=opt,
                losses=losses,
                elapsed_seconds=elapsed_prev + (time.time() - start_time),
                ctrl_coeff=ctrl_coeff if ctrl_enabled else None,
                ctrl_l0_ema=ctrl_l0_ema if ctrl_enabled else None,
            )
    train_seconds = elapsed_prev + (time.time() - start_time)
    mean_l0_train = (l0_running / l0_running_count) if l0_running_count > 0 else float("nan")
    return {
        "losses": losses,
        "train_seconds": train_seconds,
        "mean_l0_train": mean_l0_train,
        "steps_run": max(0, steps - start_step + 1),
        "batch_size": batch_size,
        "l0_controller_enabled": ctrl_enabled,
        "l0_controller_final_coeff": ctrl_coeff if ctrl_enabled else None,
        "l0_controller_target": ctrl_target if ctrl_enabled else None,
    }


def _save_train_checkpoint(
    path: Path,
    step: int,
    model: SAEBase,
    opt: torch.optim.Optimizer,
    losses: list[dict[str, Any]],
    elapsed_seconds: float,
    ctrl_coeff: float | None = None,
    ctrl_l0_ema: float | None = None,
) -> None:
    state = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "losses": losses,
        "elapsed_seconds": float(elapsed_seconds),
    }
    if ctrl_coeff is not None:
        state["ctrl_coeff"] = float(ctrl_coeff)
    if ctrl_l0_ema is not None:
        state["ctrl_l0_ema"] = float(ctrl_l0_ema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def selected_prism_loss(
    model: SAEBase,
    W_U: torch.Tensor,
    rows_normalized: torch.Tensor,
    row_mean: torch.Tensor,
    row_norms: torch.Tensor,
    hidden: torch.Tensor,
    top_targets: torch.Tensor,
    batch_size: int,
    k: int,
) -> torch.Tensor:
    h_idx = torch.randint(0, hidden.shape[0], (batch_size,), device=hidden.device)
    rank_idx = torch.randint(0, top_targets.shape[1], (batch_size,), device=hidden.device)
    tok_idx = top_targets[h_idx, rank_idx]
    x_t = rows_normalized[tok_idx]
    batch = model(x_t, k=k)
    recon_w = row_mean[None, :] + row_norms[tok_idx, None] * batch.reconstruction
    residual = W_U[tok_idx] - recon_w
    h = hidden[h_idx]
    residual_logit = (h * residual).sum(dim=1)
    original_logit = (h * W_U[tok_idx]).sum(dim=1).abs().clamp_min(1e-5)
    return ((residual_logit / original_logit) ** 2).mean()
