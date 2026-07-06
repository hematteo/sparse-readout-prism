from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FactorizerBatch:
    reconstruction: torch.Tensor  # (B, d_model)
    code: torch.Tensor  # (B, d_features), post-sparsity
    aux_loss: torch.Tensor = field(default_factory=lambda: torch.zeros(()))
    # Gradient-bearing auxiliary reconstruction (GatedSAE's activation-only
    # path); kept out of `info` so that dict stays JSON-safe scalars only.
    aux_recon: torch.Tensor | None = None  # (B, d_model)
    info: dict[str, float] = field(default_factory=dict)


def topk_mask(acts: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` largest activations per row, zero the rest.

    The one top-k selection kernel: ``TopKSAE.encode`` and the raw-tensor
    ``research.qwen_readout.encode_topk`` path both route through it, so the
    tie-breaking/selection semantics cannot drift between library and scripts.
    """
    kk = min(int(k), acts.shape[-1])
    if kk >= acts.shape[-1]:
        return acts
    values, indices = torch.topk(acts, k=kk, dim=-1)
    code = torch.zeros_like(acts)
    code.scatter_(dim=-1, index=indices, src=values)
    return code


class SAEBase(nn.Module):
    """Common base for all row factorizers. Public so it can name the model type
    in the signatures of ``build_factorizer`` / ``load_factorizer`` /
    ``decompose_token_logit`` (any architecture is accepted, not just TopK)."""

    architecture: str = "base"

    def __init__(self, d_model: int, d_features: int, encoder_init_scale: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_features = int(d_features)
        self.encoder = nn.Linear(d_model, d_features)
        self.decoder = nn.Parameter(torch.empty(d_features, d_model))
        nn.init.normal_(self.encoder.weight, std=encoder_init_scale)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder, std=d_model**-0.5)
        self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return code @ self.decoder

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, k: int | None = None) -> FactorizerBatch:
        code = self.encode(x, k=k)
        return FactorizerBatch(reconstruction=self.decode(code), code=code)


class TopKSAE(SAEBase):
    """Per-row TopK selection. Backs both `topk` and `matryoshka_topk`."""

    architecture = "topk"

    def __init__(
        self,
        d_model: int,
        d_features: int,
        k: int,
        encoder_init_scale: float = 0.02,
    ) -> None:
        super().__init__(d_model, d_features, encoder_init_scale)
        self.k = int(k)

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        # `k or self.k`: k=0/None both fall back to the trained k on purpose.
        return topk_mask(F.relu(self.encoder(x)), int(k or self.k))


class BatchTopKSAE(SAEBase):
    """BatchTopK SAE (Bussmann et al. 2024).

    Training: select top (B*k) activations across the entire (batch, features) plane.
    Eval: per-row threshold inherited from an EMA of training-time activation cutoffs,
    so eval can run on single rows without a batch.
    """

    architecture = "batch_topk"

    def __init__(
        self,
        d_model: int,
        d_features: int,
        k: int,
        encoder_init_scale: float = 0.02,
        threshold_momentum: float = 0.995,
    ) -> None:
        super().__init__(d_model, d_features, encoder_init_scale)
        self.k = int(k)
        self.threshold_momentum = float(threshold_momentum)
        # Running threshold: smallest selected activation across batches.
        self.register_buffer("activation_threshold", torch.tensor(0.0))
        self.register_buffer("threshold_initialized", torch.tensor(0, dtype=torch.long))

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        kk = min(int(k or self.k), self.d_features)
        acts = F.relu(self.encoder(x))
        if self.training:
            B = acts.shape[0]
            flat = acts.reshape(-1)
            n_keep = min(B * kk, flat.numel())
            if n_keep <= 0:
                return torch.zeros_like(acts)
            values, indices = torch.topk(flat, k=n_keep)
            mask = torch.zeros_like(flat)
            mask.scatter_(0, indices, 1.0)
            code = (acts.reshape(-1) * mask).reshape_as(acts)
            with torch.no_grad():
                cutoff = values.min().detach()
                if int(self.threshold_initialized.item()) == 0:
                    self.activation_threshold.copy_(cutoff)
                    self.threshold_initialized.fill_(1)
                else:
                    m = self.threshold_momentum
                    self.activation_threshold.mul_(m).add_(cutoff, alpha=1.0 - m)
            return code
        # Eval: per-row threshold.
        thresh = self.activation_threshold
        return torch.where(acts > thresh, acts, torch.zeros_like(acts))


class JumpReLUSAE(SAEBase):
    """JumpReLU SAE (Rajamanoharan et al. 2024).

    Per-feature learnable threshold theta. f(x) = relu(z) * I[relu(z) > theta]
    where z = W_enc x + b. L0 penalty trained with a rectangle-kernel STE.
    """

    architecture = "jumprelu"

    def __init__(
        self,
        d_model: int,
        d_features: int,
        encoder_init_scale: float = 0.02,
        log_threshold_init: float = -6.0,
        bandwidth: float = 0.5,
    ) -> None:
        super().__init__(d_model, d_features, encoder_init_scale)
        self.log_threshold = nn.Parameter(torch.full((d_features,), float(log_threshold_init)))
        # `bandwidth` is now a FRACTION of the running RMS of positive
        # activations, not an absolute value. A fixed absolute bandwidth is
        # mis-scaled across models/d_features: too small -> the rectangle-STE
        # gradient window misses almost every feature -> features never get a
        # gradient -> permanent dead collapse (observed at d=16384 on 160M).
        # Scaling to the activation RMS makes the gradient window track the
        # data scale so features stay trainable.
        self.bandwidth = float(bandwidth)
        self.register_buffer("act_scale_ema", torch.tensor(1.0))
        self.register_buffer("act_scale_init", torch.tensor(0, dtype=torch.long))
        # k attribute exists for API parity with TopK consumers (used in evaluate).
        # JumpReLU does not enforce a hard k.
        self.k = int(d_features)

    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def _effective_h(self, z: torch.Tensor, update: bool = True) -> float:
        """Bandwidth in activation units = fraction * EMA(RMS of positive z).

        ``update=False`` reads the EMA without mutating it — used by ``encode``
        so the eval/decompose path is side-effect-free even if the module was
        left in ``train()`` mode.
        """
        with torch.no_grad():
            pos = z[z > 0]
            rms = pos.pow(2).mean().sqrt() if pos.numel() > 0 else torch.as_tensor(1.0, device=z.device)
            if self.training and update:
                if int(self.act_scale_init.item()) == 0:
                    self.act_scale_ema.copy_(rms.detach())
                    self.act_scale_init.fill_(1)
                else:
                    self.act_scale_ema.mul_(0.99).add_(rms.detach(), alpha=0.01)
                scale = float(self.act_scale_ema.item())
            else:
                # Eval: forward is just the indicator; h is irrelevant, but use
                # the learned EMA for consistency.
                scale = float(self.act_scale_ema.item())
        return max(self.bandwidth * scale, 1e-8)

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        z = F.relu(self.encoder(x))
        theta = self.threshold().unsqueeze(0)
        mask = _RectangleSTE.apply(z - theta, self._effective_h(z, update=False))
        return z * mask

    def forward(self, x: torch.Tensor, k: int | None = None) -> FactorizerBatch:
        z = F.relu(self.encoder(x))
        theta = self.threshold().unsqueeze(0)
        gate_mask = _RectangleSTE.apply(z - theta, self._effective_h(z))
        code = z * gate_mask
        recon = self.decode(code)
        l0_estimate = gate_mask.sum(dim=-1).mean()
        return FactorizerBatch(
            reconstruction=recon,
            code=code,
            aux_loss=l0_estimate,
            info={"mean_l0_train": float(l0_estimate.detach().cpu().item())},
        )


class _RectangleSTE(torch.autograd.Function):
    """Heaviside forward with a rectangle-kernel backward (bandwidth=h)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, h: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.h = float(h)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        h = ctx.h
        kernel = ((x.abs() <= h / 2).to(x.dtype)) / max(h, 1e-12)
        return grad_output * kernel, None


class GatedSAE(SAEBase):
    """Gated SAE (Rajamanoharan et al. 2024).

    Two-branch encoder: an activation branch decides which features fire and a magnitude
    branch sets their value. We share the encoder weight (tied), per the paper's
    weight-sharing variant: W_mag = exp(r) * W_gate, b_mag and b_gate separate.
    """

    architecture = "gated"

    def __init__(
        self,
        d_model: int,
        d_features: int,
        encoder_init_scale: float = 0.02,
    ) -> None:
        super().__init__(d_model, d_features, encoder_init_scale)
        self.gate_bias = nn.Parameter(torch.zeros(d_features))
        self.log_r = nn.Parameter(torch.zeros(d_features))
        # k attribute exists for API parity with TopK consumers (see JumpReLU);
        # Gated does not enforce a hard k.
        self.k = int(d_features)

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        # NOTE: SAEBase's `encoder.bias` is repurposed as the MAGNITUDE-branch
        # bias here (the gate branch has its own `gate_bias`); the shared
        # `encoder.weight` is the paper's weight-tying variant.
        z = x @ self.encoder.weight.T  # (B, d_features)
        gate_pre = z + self.gate_bias
        gate_active = (gate_pre > 0).to(z.dtype)
        mag_pre = self.log_r.exp() * z + self.encoder.bias
        mag = F.relu(mag_pre)
        return gate_active * mag

    def forward(self, x: torch.Tensor, k: int | None = None) -> FactorizerBatch:
        z = x @ self.encoder.weight.T
        gate_pre = z + self.gate_bias
        gate_active = (gate_pre > 0).to(z.dtype)
        mag_pre = self.log_r.exp() * z + self.encoder.bias
        mag = F.relu(mag_pre)
        code = gate_active * mag

        # Aux losses per the paper.
        gate_sparsity = F.relu(gate_pre).sum(dim=-1).mean()  # L1 on active units
        # Auxiliary reconstruction with activation-only path (decoder is stop-grad,
        # but the gating branch must keep grads so encoder.weight + gate_bias
        # actually learn through the aux MSE).
        aux_recon = F.relu(gate_pre) @ self.decoder.detach()
        aux_loss = gate_sparsity  # the recon-aux term is added in train.py via aux_recon
        return FactorizerBatch(
            reconstruction=self.decode(code),
            code=code,
            aux_loss=aux_loss,
            aux_recon=aux_recon,
            info={
                "mean_l0_train": float((gate_active.sum(dim=-1).float().mean()).cpu().item()),
            },
        )


class L1ReLUSAE(SAEBase):
    """Vanilla L1-ReLU SAE (Bricken et al. 2023).

    f(x) = ReLU(W_enc x + b). Aux loss = sum_i |f_i| * ||W_dec_i||.
    Decoder is renormalized after each step (standard trick), so the norm factor
    is approximately 1; we still multiply explicitly so the loss tracks correctly
    if normalization is disabled.
    """

    architecture = "l1_relu"

    def __init__(
        self,
        d_model: int,
        d_features: int,
        encoder_init_scale: float = 0.02,
    ) -> None:
        super().__init__(d_model, d_features, encoder_init_scale)
        self.k = int(d_features)

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def forward(self, x: torch.Tensor, k: int | None = None) -> FactorizerBatch:
        code = self.encode(x)
        decoder_norms = self.decoder.norm(dim=1).detach()  # (d_features,)
        l1 = (code * decoder_norms).sum(dim=-1).mean()
        return FactorizerBatch(
            reconstruction=self.decode(code),
            code=code,
            aux_loss=l1,
            info={"mean_l0_train": float((code > 0).float().sum(dim=-1).mean().cpu().item())},
        )


_ARCHS = {
    "topk": TopKSAE,
    "matryoshka_topk": TopKSAE,
    "batch_topk": BatchTopKSAE,
    "jumprelu": JumpReLUSAE,
    "gated": GatedSAE,
    "l1_relu": L1ReLUSAE,
}


def build_factorizer(config: dict[str, Any], d_model: int) -> SAEBase:
    cfg = config.get("factorizer", {})
    arch = cfg.get("architecture", "topk")
    if arch not in _ARCHS:
        raise ValueError(f"Unsupported factorizer architecture: {arch}")
    cls = _ARCHS[arch]
    d_features = int(cfg.get("d_features", 1024))
    encoder_init_scale = float(cfg.get("encoder_init_scale", 0.02))
    if arch in {"topk", "matryoshka_topk"}:
        model = cls(
            d_model=d_model,
            d_features=d_features,
            k=int(cfg.get("k", 32)),
            encoder_init_scale=encoder_init_scale,
        )
    elif arch == "batch_topk":
        model = cls(
            d_model=d_model,
            d_features=d_features,
            k=int(cfg.get("k", 32)),
            encoder_init_scale=encoder_init_scale,
            threshold_momentum=float(cfg.get("threshold_momentum", 0.995)),
        )
    elif arch == "jumprelu":
        model = cls(
            d_model=d_model,
            d_features=d_features,
            encoder_init_scale=encoder_init_scale,
            log_threshold_init=float(cfg.get("log_threshold_init", -6.0)),
            bandwidth=float(cfg.get("bandwidth", 0.5)),
        )
    elif arch == "gated":
        model = cls(
            d_model=d_model,
            d_features=d_features,
            encoder_init_scale=encoder_init_scale,
        )
    elif arch == "l1_relu":
        model = cls(
            d_model=d_model,
            d_features=d_features,
            encoder_init_scale=encoder_init_scale,
        )
    else:
        raise AssertionError("unreachable")
    # Instance attribute shadowing the class attribute on purpose: it is how a
    # TopKSAE built as "matryoshka_topk" stays distinguishable from "topk"
    # downstream (train.py dispatches the matryoshka loss on it).
    model.architecture = arch
    return model


def load_factorizer(
    checkpoint: str | Any,
    *,
    d_model: int | None = None,
    factorizer_config: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    freeze: bool = False,
    eval_mode: bool = True,
) -> SAEBase:
    """Rebuild a factorizer from a training checkpoint and load its weights.

    The single reader matched to ``runner.py``'s checkpoint writer; scripts that
    re-implement the build + ``load_state_dict`` + ``eval`` sequence should call
    this instead so the checkpoint-schema handling lives in one place.

    ``checkpoint`` is a path or an already-loaded dict. The factorizer config is
    taken from ``factorizer_config`` if given, else from the checkpoint
    (``ckpt['factorizer']`` or ``ckpt['config']['factorizer']``). Weights come
    from ``ckpt['model_state_dict']`` or ``ckpt['state_dict']`` (or the dict
    itself if it looks like a bare state dict). ``d_model`` is inferred from
    ``encoder.weight`` when not given.
    """
    # weights_only=True: run_experiment writes only tensors + primitive dicts
    # (state dict, factorizer/evaluation config, row stats, metrics), so released
    # checkpoints load without executing pickle — the safe default for artifacts
    # downloaded from the Hub.
    ckpt = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = factorizer_config
    if cfg is None:
        cfg = ckpt.get("factorizer") or ckpt.get("config", {}).get("factorizer")
    if cfg is None:
        raise KeyError("load_factorizer: no factorizer config (pass factorizer_config= or embed it in the checkpoint)")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if d_model is None:
        w = state.get("encoder.weight")
        if w is None:
            raise ValueError("load_factorizer: cannot infer d_model from state dict; pass d_model=")
        d_model = int(w.shape[1])
    model = build_factorizer({"factorizer": cfg}, d_model=int(d_model))
    if device is not None:
        model = model.to(device)
    model.load_state_dict(state)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    if eval_mode:
        model.eval()
    return model


def reconstruction_loss(
    model: SAEBase,
    x: torch.Tensor,
    architecture: str,
    k: int,
    k_values: list[int] | None = None,
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Returns (rec_loss, code, aux_loss, info). aux_loss is the architecture's
    own sparsity/auxiliary term (raw, no coefficient applied)."""
    if architecture == "matryoshka_topk" and k_values:
        if weights is None:
            weights = [1.0] * len(k_values)
        total = x.new_tensor(0.0)
        last_code = None
        denom = float(sum(weights))
        for kk, ww in zip(k_values, weights, strict=False):
            batch = model(x, k=int(kk))
            total = total + float(ww) * F.mse_loss(batch.reconstruction, x)
            last_code = batch.code
        assert last_code is not None
        return total / denom, last_code, x.new_tensor(0.0), {}

    batch = model(x, k=k)
    rec = F.mse_loss(batch.reconstruction, x)
    aux = batch.aux_loss
    info = dict(batch.info)
    if architecture == "gated" and batch.aux_recon is not None:
        # Aux reconstruction (activation-only path) — standard Gated SAE objective.
        # Gradient-bearing: train.py adds it to the loss (the only non-scalar entry).
        info["gated_aux_recon_mse"] = F.mse_loss(batch.aux_recon, x)
    return rec, batch.code, aux, info
