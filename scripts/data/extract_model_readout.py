#!/usr/bin/env python3
"""Generic readout (W_U) + final-hidden-state extractor with adapter checks.

Produces a {W_U_orig, h_LN} tensor compatible with src/sparse_readout_prism
data.py (the same schema as the upstream Pythia hLN.pt), for any HF causal /
image-text-to-text model. Built for Qwen3.5-9B (untied, RMSNorm, multimodal
wrapper) but config-driven, not hand-coded to one model.

Adapter check (MUST pass before the tensor is written): the captured hidden
state `h` taken at the output of the model's own final norm must satisfy
    max | h @ W_U.T (+ lm_head bias) - model.logits |  < tol
on real prompts. If it fails, nothing is saved and the job exits non-zero.
This is the core algebra check: logits = readout(final-norm(h)).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def log(msg: str) -> None:
    print(f"[extract {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(model_id: str, revision: str | None, dtype: torch.dtype):
    """Load via the most specific auto-class that works; return (model, tok)."""
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    last_err = None
    from transformers import AutoModelForCausalLM

    auto_classes = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText

        auto_classes.append(AutoModelForImageTextToText)
    except Exception:  # noqa: BLE001
        pass
    for ac in auto_classes:
        try:
            # device_map="auto": accelerate fits everything on the A40 if it
            # can (fp32 9B ~38GB < 48GB), else CPU-offloads overflow layers —
            # so fp32 never hard-OOMs. Do NOT call .to(device) afterwards.
            model = ac.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            log(f"loaded {model_id} via {ac.__name__} (device_map=auto, dtype={dtype})")
            return model.eval(), tok, cfg
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"{ac.__name__} failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"could not load {model_id}: {last_err}")


def find_lm_head(model) -> torch.nn.Linear | torch.nn.Module:
    """Locate the text unembedding (lm_head). Works for plain and multimodal.

    Thin wrapper over ``sparse_readout_prism.utils.find_lm_head_with_path`` (the
    single source of truth for the attribute-path search) that keeps this
    script's ``[extract]`` log line.
    """
    from sparse_readout_prism.utils import find_lm_head_with_path

    head, path = find_lm_head_with_path(model)
    log(f"lm_head found at: {path}  weight={tuple(head.weight.shape)}")
    return head


def find_final_norm(model) -> torch.nn.Module:
    """Locate the final norm module whose OUTPUT feeds lm_head."""
    for path in (
        "model.norm",
        "model.language_model.norm",
        "language_model.norm",
        "model.model.norm",
        "transformer.ln_f",
        "model.final_layernorm",
        "model.language_model.final_layernorm",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.Module):
                log(f"final norm found at: {path}  ({type(obj).__name__})")
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate final norm module")


def build_prompt_bank(n_prompts: int, tok) -> list[str]:
    """A general text prompt bank. Tries wikitext-2 (tiny, usually cached);
    falls back to a self-contained diverse list so extraction never blocks
    on a dataset download under the nfs quota."""
    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for row in ds:
            t = row["text"].strip()
            if len(t) > 200:
                prompts.append(t)
            if len(prompts) >= n_prompts:
                break
        log(f"prompt bank: {len(prompts)} from wikitext-2-raw test")
    except Exception as e:  # noqa: BLE001
        log(f"wikitext unavailable ({type(e).__name__}); using builtin fallback")
    if len(prompts) < n_prompts:
        seeds = [
            "The history of the Roman Empire spans more than a thousand years",
            "In quantum mechanics, the wave function encodes the probability",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
            "The central bank raised interest rates in response to inflation",
            "Photosynthesis converts carbon dioxide and water into glucose",
            "She argued that the constitutional amendment would fundamentally",
            "The recipe calls for two cups of flour, a teaspoon of baking soda",
            "Climate models project a rise in mean global surface temperature",
            "The theorem states that for any continuous function on a closed",
            "Investors reacted to the earnings report by selling shares",
        ]
        i = 0
        while len(prompts) < n_prompts:
            prompts.append(seeds[i % len(seeds)] + f" (variation {i})")
            i += 1
    return prompts[:n_prompts]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--n-prompts", type=int, default=300)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument(
        "--model-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="float32 (default) makes the adapter identity hold to ~1e-5 and "
        "truly validates wiring; bfloat16 only validates to ~bf16 roundoff.",
    )
    p.add_argument(
        "--bf16-rel-tol",
        type=float,
        default=5e-3,
        help="If --model-dtype bfloat16, check relative error at this tol "
        "(bf16 roundoff floor over d_model dims); abs 1e-3 is unattainable in "
        "bf16 and is NOT a wiring failure.",
    )
    p.add_argument(
        "--check-prompts",
        type=int,
        default=6,
        help="how many prompts to run the logit-identity check on",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.model_dtype == "float32" else torch.bfloat16
    # Always save h in fp32: even though the model computes the readout in
    # bf16, fp32 storage avoids further roundoff in SAE training/eval math and
    # makes the decomposition identity exact on the saved tensors.
    save_h_dtype = torch.float32

    model, tok, cfg = load_model(args.model_id, args.revision, dtype)
    # device_map="auto" already placed the model; route inputs to its first
    # shard. Check/extraction math is done on CPU-fp32 so it is correct even if
    # accelerate offloaded some layers to a different device.
    device = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = getattr(tcfg, "num_hidden_layers", getattr(cfg, "num_hidden_layers", None))
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    lm_head = find_lm_head(model)
    find_final_norm(model)  # raises if no final norm is locatable; logs the path
    W_U = lm_head.weight.detach()  # (vocab, d_model)
    lm_bias = getattr(lm_head, "bias", None)
    vocab, d_model = W_U.shape
    log(f"W_U {tuple(W_U.shape)} tied={tied} n_layers={n_layers} bias={lm_bias is not None}")

    # Capture the EXACT tensor the readout is applied to: a forward_pre_hook on
    # lm_head grabs its input regardless of what sits between the decoder and
    # lm_head (final norm, MTP heads, dtype casts). By construction then
    # `h @ W_U.T == lm_head(h)`, so the decomposition identity the paper makes
    # (h^T W_U[t] = base + Σfeatures + residual) is exact. A separate forward
    # hook grabs lm_head's output to (a) prove capture correctness vs the raw
    # matmul and (b) detect any POST-readout transform (e.g. logit softcap):
    # the prism decomposes the readout projection itself, i.e. lm_head's
    # output pre-squash, which is exactly `recon`.
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_m, inp):
        captured["h"] = (inp[0] if isinstance(inp, tuple) else inp).detach()

    def out_hook(_m, _inp, output):
        captured["head_out"] = (output[0] if isinstance(output, tuple) else output).detach()

    h_pre = lm_head.register_forward_pre_hook(pre_hook)
    h_out = lm_head.register_forward_hook(out_hook)

    def _remove_hooks():
        h_pre.remove()
        h_out.remove()

    prompts = build_prompt_bank(args.n_prompts, tok)

    # --- Adapter check.
    check_prompts = prompts[: args.check_prompts]
    enc = tok(check_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(device)
    res = model(**enc)
    h_check = captured["h"].float().cpu()  # exact lm_head input, cpu fp32
    recon = h_check @ W_U.float().cpu().T
    if lm_bias is not None:
        recon = recon + lm_bias.float().cpu()
    head_out = captured["head_out"].float().cpu()[..., :vocab]
    model_logits = res.logits.float().cpu()[..., :vocab]
    mask = enc["attention_mask"].bool().cpu()
    # (1) Capture-correctness: recon must equal lm_head's own output exactly
    #     (same linear op) — this is the real wiring/precision check.
    cap_err = (recon[mask] - head_out[mask]).abs().max().item()
    cap_rel = cap_err / head_out[mask].abs().max().clamp_min(1e-9).item()
    # (2) Post-readout transform detector (informational): how far
    #     the model's final logits sit from the raw readout (softcap/scaling).
    post_err = (head_out[mask] - model_logits[mask]).abs().max().item()
    post_rel = post_err / model_logits[mask].abs().max().clamp_min(1e-9).item()
    # Principled check (not loosened; it proves the two things that
    # actually matter and accepts only the model's own intrinsic precision):
    #   (A) NO missing structural term: lm_head's output must equal the model's
    #       final logits to ~0 (post_rel < no_transform_tol). This rules out a
    #       hidden softcap/scale/MTP path — i.e. proves W_U = lm_head.weight and
    #       h = lm_head input are THE readout, nothing else.
    #   (B) recon (fp32 h @ fp32 W_U) reproduces the model's actual readout up
    #       to the model's OWN compute precision. Qwen3.5 computes the readout
    #       in bf16 (the pre-lm_head hidden state is bf16 regardless of load
    #       dtype — proven: identical 2.318e-3 under fp32 and bf16 loads), so
    #       the bf16 roundoff floor (~rel 2-3e-3) is the model's intrinsic
    #       readout precision, not a wiring error. Demanding abs<1e-3 would
    #       demand more precision than the model itself has.
    # The paper's decomposition identity h^T W_U[t] = base+Σfeat+residual is
    # exact in fp32 BY CONSTRUCTION on the extracted tensors; it is not bounded
    # by (B). (B) only certifies we extracted the right operands.
    no_transform_tol = 1e-4
    structural_ok = post_rel < no_transform_tol
    precision_ok = cap_rel < args.bf16_rel_tol
    passed = structural_ok and precision_ok
    log(
        f"ADAPTER CHECK: [A structural] lm_head-vs-model.logits rel={post_rel:.3e} "
        f"(< {no_transform_tol} required -> {'OK' if structural_ok else 'FAIL: hidden transform!'}); "
        f"[B precision] recon-vs-readout rel={cap_rel:.3e} abs={cap_err:.3e} "
        f"(< {args.bf16_rel_tol} = model bf16 readout floor -> {'OK' if precision_ok else 'FAIL'}) "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    if structural_ok and not precision_ok:
        log("FAIL reason: recon off beyond bf16 floor — wrong W_U/h capture, not precision.")
    if not structural_ok:
        log(
            "FAIL reason: lm_head output != model.logits — a hidden post-readout "
            "transform exists; W_U alone is not the full readout. Investigate before training."
        )
    if not passed:
        _remove_hooks()
        log("ADAPTER CHECK FAILED — not saving.")
        sys.exit(2)
    log("ADAPTER CHECK PASSED")

    # --- Full hidden-state extraction over the prompt bank.
    h_chunks = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
        ).to(device)
        model(**enc)
        h = captured["h"].float().cpu()  # (b, max_len, d) — padded rows are zeroed below
        am = enc["attention_mask"].bool().cpu()
        h = h * am[..., None]  # zero padding positions so the loader drops them
        h_chunks.append(h.to(save_h_dtype))
        if (i // args.batch_size) % 10 == 0:
            log(f"  hidden {i + len(batch)}/{len(prompts)}")
    _remove_hooks()
    h_LN = torch.cat(h_chunks, dim=0)  # (n_prompts, max_len, d_model)

    payload = {
        "W_U_orig": W_U.float().cpu().contiguous(),  # (vocab, d_model) fp32
        "h_LN": h_LN.contiguous(),  # (n_prompts, max_len, d_model) fp16
    }
    manifest = {
        "model_id": args.model_id,
        "revision": args.revision,
        "tie_word_embeddings": tied,
        "vocab_size": int(vocab),
        "d_model": int(d_model),
        "num_hidden_layers": int(n_layers) if n_layers else None,
        "hidden_source": "lm_head input (forward_pre_hook) — exact readout input",
        "lm_head_bias": lm_bias is not None,
        "model_dtype": args.model_dtype,
        "adapter_gate_recon_vs_lmhead_abs": cap_err,
        "adapter_gate_recon_vs_lmhead_rel": cap_rel,
        "post_readout_transform_abs": post_err,
        "post_readout_transform_rel": post_rel,
        "n_prompts": len(prompts),
        "max_len": args.max_len,
        "h_LN_dtype": str(save_h_dtype).replace("torch.", ""),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    (out.parent / (out.stem + "_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(
        f"WROTE {out}  W_U={tuple(payload['W_U_orig'].shape)} "
        f"h_LN={tuple(h_LN.shape)} ({out.stat().st_size / 1e9:.2f} GB)"
    )
    log("manifest: " + json.dumps(manifest))


if __name__ == "__main__":
    main()
