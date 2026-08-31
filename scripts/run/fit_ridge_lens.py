#!/usr/bin/env python3
"""Fit corpus-conditioned ridge-translator lenses, one per source layer (tuned-lens style).

The second lens family of the cross-lens study (``tab:app-cross-lens-extension``
rows "EN--ZH, ridge translator" and "Jacobian vs ridge, EN";
``fig:cross-lens-extension``). Per source layer ``l`` the script solves the
bias-free ridge regression

    W_l = argmin_W || X_l W^T - Y ||_F^2 + lambda ||W||_F^2

where ``X_l`` are the block-output residuals at layer ``l`` and ``Y`` the
final-block residuals of the same tokens, over the same seeded C4 prompts used to
fit the corresponding Jacobian lens (``scripts/run/fit_jlens.py``). The result is
stored in the JacobianLens container (transport = residual @ W.T, fp16), so
``scripts/run/run_cross_lens_readouts.py`` and the aggregators consume it
unchanged.

Protocol:
  - source layers and d_model are copied from a reference Jacobian lens file
    (``--layers-from``);
  - fit on prompts [0, n_prompts - holdout), lambda selected per layer on the
    holdout tail by translation R^2, then refit on all n_prompts prompts at the
    selected lambda;
  - lambda grid: g * tr(XtX) / (N * d) for g in {1e-3, 1e-2, 1e-1, 1, 10};
  - diagnostics reported, not enforced: per-layer holdout R^2, and top-1
    agreement between decode(transport(h_l)) and the model's final logits on the
    last (up to) 5 holdout prompts at the two deepest fitted layers (written to
    ``<out>.report.json`` together with a ``provenance`` block).

Lambda grid, as run. The multipliers ``g`` are scaled per token
(``lambda = g * tr(XtX) / (N * d)``, ``N`` = tokens in the fit prompts) but the
penalty is added to the N-token Gram sum ``XtX`` itself, so relative to the Gram
matrix the effective shrinkage is ``g / N`` -- small at every grid point -- and
both paper fits selected the grid's top value ``g = 10`` at all twelve layers
(``lambda_mult`` in the shipped ``.report.json`` files). The math is kept
exactly as the paper ran it; widening or rescaling the grid changes the fitted
translators and is a paper decision, not a code fix.

Resumable: the second-moment accumulators checkpoint every 10 prompts (atomic)
under ``<ckpt-dir>/<tag>_{fit,hold}.ckpt``, guarded by ``<ckpt-dir>/<tag>.meta.json``
(model id, prompt manifest, sha1 of the prompt list, ``n_prompts``, ``holdout``,
``tag``, layers, ``max_seq_len``): existing checkpoints are refused when the
sidecar differs. The script exits early if ``--out`` already exists. ``jlens``
is imported lazily (``uv sync --extra lens``); the model is loaded through
``research.cross_lens.load_lens_model`` on ``--device`` (default ``cuda``).

Paper runs (Qwen3.5-9B, layer set copied from the English-fitted Jacobian lens)::

  fit_ridge_lens.py --model-id Qwen/Qwen3.5-9B --prompts-json <out>/c4_prompts_en_seed0.json \
      --n-prompts 100 --holdout 10 --layers-from <out>/qwen35_9b_jlens_en_seed0_n100.pt \
      --ckpt-dir <ckpt> --tag ridge_en --out <out>/qwen35_9b_ridgelens_en_seed0_n100.pt
  fit_ridge_lens.py --model-id Qwen/Qwen3.5-9B --prompts-json <out>/c4_prompts_zh_seed0.json \
      --n-prompts 100 --holdout 10 --layers-from <out>/qwen35_9b_jlens_en_seed0_n100.pt \
      --ckpt-dir <ckpt> --tag ridge_zh --out <out>/qwen35_9b_ridgelens_zh_seed0_n100.pt

Fixed in 0.2.1:
  - ``--holdout`` must be >= 1 and leave at least one fit prompt (clear
    ``SystemExit``): with 0 the holdout accumulator was empty and lambda
    selection crashed after the full fit pass.
  - The deep-layer top-1 agreement diagnostic sliced ``prompts[-5:]``, which
    reads into the fit prompts whenever ``--holdout`` < 5; it now slices the
    holdout tail, ``prompts[n_fit:][-5:]``. For the paper's ``--holdout 10`` the
    two slices coincide, so the shipped ``deep_top1_agreement`` numbers stand.
  - Accumulator resume is guarded by the ``<tag>.meta.json`` sidecar described
    above; ``<out>.report.json`` gains a ``provenance`` block.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sparse_readout_prism.research.cross_lens import (
    check_resume_meta,
    import_jlens,
    load_lens_model,
    prompt_list_sha1,
    write_resume_meta,
)
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import resolve_device

LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def log(msg: str) -> None:
    print(f"[ridge_lens] {msg}", flush=True)


def _activation_recorder():
    import_jlens()
    from jlens.hooks import ActivationRecorder

    return ActivationRecorder


def diagnostic_prompts(prompts: list[str], n_fit: int) -> list[str]:
    """The (up to) five holdout prompts the deep top-1 agreement diagnostic scores."""
    return prompts[n_fit:][-5:]


@torch.no_grad()
def accumulate(model, prompts, layers, final_layer, max_seq_len, ckpt_path, *, device, every=10):
    """One pass over prompts, accumulating XtX / XtY / YtY traces per layer."""
    ActivationRecorder = _activation_recorder()

    record_at = sorted(set(layers) | {final_layer})

    state = None
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        log(f"[resume] accumulator checkpoint at prompt {state['next_prompt']}")

    acc = None
    start = 0
    if state is not None:
        acc = {l: {k: v.to(device) for k, v in per.items()} for l, per in state["acc"].items()}
        start = state["next_prompt"]

    for pi in range(start, len(prompts)):
        with ActivationRecorder(model.layers, at=record_at) as rec:
            input_ids = model.encode(prompts[pi], max_length=max_seq_len)
            model.forward(input_ids)
            acts = {i: rec.activations[i].detach() for i in record_at}
        y = acts[final_layer][0].float()  # (seq, d_model)
        if acc is None:
            d = y.shape[1]
            acc = {
                l: {
                    "xtx": torch.zeros(d, d, device=device),
                    "xty": torch.zeros(d, d, device=device),
                    "ytr": torch.zeros((), device=device),
                    "n": torch.zeros((), device=device),
                }
                for l in layers
            }
        for l in layers:
            x = acts[l][0].float()  # (seq, d_model)
            acc[l]["xtx"] += x.T @ x
            acc[l]["xty"] += x.T @ y
            acc[l]["ytr"] += (y * y).sum()
            acc[l]["n"] += y.shape[0]
        if (pi + 1) % every == 0 or pi == len(prompts) - 1:
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(
                {
                    "next_prompt": pi + 1,
                    "acc": {l: {k: v.cpu() for k, v in per.items()} for l, per in acc.items()},
                },
                tmp,
            )
            os.replace(tmp, ckpt_path)
            log(f"prompt {pi + 1}/{len(prompts)} accumulated")
    return acc


def solve(xtx: torch.Tensor, xty: torch.Tensor, lam: float) -> torch.Tensor:
    d = xtx.shape[0]
    reg = xtx + lam * torch.eye(d, device=xtx.device)
    return torch.linalg.solve(reg, xty).T.contiguous()  # transport = x @ W.T


def r2(acc_h, W) -> float:
    """Holdout R^2 from accumulated second moments (vs zero baseline)."""
    xtx, xty, ytr = acc_h["xtx"], acc_h["xty"], acc_h["ytr"]
    sse = torch.einsum("ij,jk,ik->", W, xtx, W) - 2.0 * torch.einsum("ij,ji->", W, xty) + ytr
    return float(1.0 - sse / ytr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--prompts-json", required=True, help="seeded prompt dump from fit_jlens.py prompts")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=10, help="holdout tail for lambda selection (>= 1)")
    ap.add_argument("--layers-from", required=True, help="reference Jacobian lens .pt (source layers, d_model)")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--ckpt-dir", required=True, help="directory for the resumable accumulator checkpoints")
    ap.add_argument("--out", required=True, help="output lens .pt (JacobianLens container)")
    ap.add_argument("--tag", default="ridge", help="checkpoint file prefix inside --ckpt-dir")
    ap.add_argument("--device", default="cuda", help="torch device for the model and accumulators (paper: cuda)")
    args = ap.parse_args(argv)
    if args.holdout < 1:
        raise SystemExit(f"--holdout must be >= 1 (got {args.holdout}): lambda is selected on the holdout prompts")

    out = Path(args.out)
    if out.exists():
        log(f"[resume] {out} exists; done")
        return 0

    ref = torch.load(args.layers_from, map_location="cpu", weights_only=True)
    layers = sorted(int(l) for l in ref["source_layers"])
    d_model = int(ref["d_model"])
    log(f"layers={layers} d_model={d_model}")

    payload = json.loads(Path(args.prompts_json).read_text())
    prompts = payload["prompts"][: args.n_prompts]
    n_fit = len(prompts) - args.holdout
    if n_fit < 1:
        raise SystemExit(f"--holdout {args.holdout} leaves no fit prompts ({len(prompts)} prompts available)")
    log(f"{len(prompts)} prompts ({n_fit} fit + {args.holdout} holdout)")

    ckpt_dir = Path(args.ckpt_dir)
    fit_ckpt, hold_ckpt = ckpt_dir / f"{args.tag}_fit.ckpt", ckpt_dir / f"{args.tag}_hold.ckpt"
    meta = {
        "model_id": args.model_id,
        "prompts_json": args.prompts_json,
        "prompts_sha1": prompt_list_sha1(prompts),
        "n_prompts": len(prompts),
        "holdout": args.holdout,
        "tag": args.tag,
        "layers": layers,
        "max_seq_len": args.max_seq_len,
    }
    meta_path = ckpt_dir / f"{args.tag}.meta.json"
    check_resume_meta(meta_path, meta, [fit_ckpt, hold_ckpt])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_resume_meta(meta_path, meta)

    device = resolve_device(args.device)
    model = load_lens_model(args.model_id, device=device).lens_model
    final_layer = model.n_layers - 1

    acc_fit = accumulate(model, prompts[:n_fit], layers, final_layer, args.max_seq_len, fit_ckpt, device=device)
    acc_hold = accumulate(model, prompts[n_fit:], layers, final_layer, args.max_seq_len, hold_ckpt, device=device)

    report, W_final = {}, {}
    for l in layers:
        scale = float(acc_fit[l]["xtx"].diagonal().sum() / (acc_fit[l]["n"] * d_model))
        best = None
        for g in LAMBDA_GRID:
            lam = g * scale
            W = solve(acc_fit[l]["xtx"], acc_fit[l]["xty"], lam)
            score = r2(acc_hold[l], W)
            if best is None or score > best[2]:
                best = (g, lam, score)
        g, lam, score = best
        # Refit on fit+holdout at the selected lambda (same lambda scale).
        xtx = acc_fit[l]["xtx"] + acc_hold[l]["xtx"]
        xty = acc_fit[l]["xty"] + acc_hold[l]["xty"]
        W_final[l] = solve(xtx, xty, lam)
        report[l] = {"lambda_mult": g, "lambda": lam, "holdout_r2": score}
        log(f"layer {l}: g={g} holdout_R2={score:.4f}")

    # Diagnostic: top-1 agreement of decoded transported states vs model
    # logits on the last (up to) 5 holdout prompts, two deepest fitted layers.
    ActivationRecorder = _activation_recorder()

    deep = layers[-2:]
    agree = {l: [0, 0] for l in deep}
    with torch.no_grad():
        for prompt in diagnostic_prompts(prompts, n_fit):
            with ActivationRecorder(model.layers, at=sorted(set(deep) | {final_layer})) as rec:
                input_ids = model.encode(prompt, max_length=args.max_seq_len)
                model.forward(input_ids)
                acts = {i: rec.activations[i].detach() for i in rec.activations}
            ml = model.unembed(acts[final_layer][0].float()).argmax(-1)
            for l in deep:
                tl = model.unembed((acts[l][0].float() @ W_final[l].T)).argmax(-1)
                agree[l][0] += int((tl == ml.to(tl.device)).sum())
                agree[l][1] += int(ml.numel())
    for l in deep:
        k, n = agree[l]
        report[l]["deep_top1_agreement"] = k / max(n, 1)
        log(f"layer {l}: transported-vs-model top1 agreement {k}/{n} = {k / max(n, 1):.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(
        {
            "J": {l: W_final[l].cpu().to(torch.float16) for l in layers},
            "n_prompts": len(prompts),
            "source_layers": layers,
            "d_model": d_model,
        },
        tmp,
    )
    os.replace(tmp, out)
    Path(str(out) + ".report.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "prompts_json": args.prompts_json,
                "n_prompts": len(prompts),
                "holdout": args.holdout,
                "lambda_grid": LAMBDA_GRID,
                "layers": {str(l): report[l] for l in layers},
                "provenance": run_provenance(args),
            },
            indent=1,
        )
    )
    log(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
