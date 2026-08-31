#!/usr/bin/env python3
"""Per-layer lens readings and readout decompositions over a prompt bank (raw dump).

The producer for every cross-lens artifact: ``fig:cross-lens-butterfly``,
``fig:cross-lens-extension``, ``tab:app-cross-lens-families``,
``tab:app-cross-lens-extension`` and ``tab:app-cross-lens-de-antonym-layers``.
For each prompt in the bank the script records the greedy continuation, then per
fitted layer and per scored position: the lens's top-5 tokens with their logits,
and the Sparse Readout Prism decomposition of every probed target token's
transported logit (base term, feature sum, residual and the top signed feature
contributions), optionally also of the lens's own top-1 token
(``--decompose-top1``). Aggregation is offline:
``scripts/eval/aggregate_cross_lens_en_zh.py``,
``scripts/eval/aggregate_cross_lens_en_de.py``,
``scripts/analyze/cross_lens_three_lens_prompt.py`` and
``scripts/figures/compute_cross_lens_shared_feature.py`` all read this dump.

Transport. ``--lens`` is a fitted lens ``.pt`` in the JacobianLens container (a
Jacobian lens from ``scripts/run/fit_jlens.py`` or a ridge translator from
``scripts/run/fit_ridge_lens.py``): the hooked block output ``h_l`` at each scored
position is transported with ``J_l @ h_l``, passed through the model's final norm,
and read through the LM head. ``--lens identity`` skips the transport (a plain
logit-lens reading on the same layer set, copied from ``--layers-from``). A
position-alignment guard checks on the first prompt that the hooked state
reproduces the lens's own logits at every layer.

Decomposition basis. ``--checkpoint`` is the k=128 seed-0 readout dictionary for
Qwen3.5-9B (8x, D=32768), Hugging Face ``hematteo/sparse-readout-prism`` file
``qwen3.5-9b/k128_8x/checkpoint.pt``, a different operating point from the main
tables' 32x/k=256 dictionaries; it is read with ``research.qwen_readout.load_sae``
and ``--k`` defaults to the checkpoint's trained k (an explicit different value
warns). Rows of the LM head are centred and unit-normalised before encoding
(``data.center_normalize_rows``), and the contribution of feature ``i`` to the
score of token ``t`` is ``||W_t - mu|| * code_i(t) * (h . d_i)``. ``--centering``
picks ``mu``: ``live`` (default, the paper) is the full-vocabulary mean of the
live bf16 -> fp32 head; ``trained`` is the checkpoint's stored ``row_mean``, or
the mean over the tokenizer's text-token rows when the checkpoint predates it.

Bank schema: ``{"prompts": [{"id", "group", "prompt", "targets": [...], ...}]}``
(see ``data/cross_lens/README.md``). Every target string is probed by the id of
its first token. ``jlens`` is imported lazily (``uv sync --extra lens``); the
model is loaded through ``research.cross_lens.load_lens_model`` on ``--device``
(default ``cuda``). The dump keeps its ``sae`` key (the checkpoint path) so
existing readers of the schema are unaffected; ``<out>.manifest.json`` records
the resolved ``k``, the centering mode and full provenance.

Paper runs (Qwen3.5-9B, one run per lens and bank, final prompt position only)::

  run_cross_lens_readouts.py --model-id Qwen/Qwen3.5-9B --lens <lens.pt> --checkpoint <checkpoint.pt> \
      --k 128 --prompts data/cross_lens/cross_lens_prompts_en_zh.json --n-positions 1 \
      --decompose-top1 --out <dumps>/en_zh__<lens>.json

  Lenses: the English-, Chinese- and German-fitted Jacobian lenses (n=100 and
  n=300) and the English- and Chinese-fitted ridge translators. Banks:
  ``data/cross_lens/cross_lens_prompts_en_zh.json`` (EN-ZH cells) and
  ``data/cross_lens/cross_lens_prompts_en_de.json`` (EN-DE cells). The three-lens
  worked example reads ``data/cross_lens/cross_lens_three_lens_prompts.json`` with
  ``--top-feats 10`` under the EN, ZH and DE n=100 lenses, plus
  ``--lens identity --layers-from <EN lens>`` for the lens-free reading.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch

from sparse_readout_prism.research.cross_lens import (
    CENTERING_MODES,
    centering_row_mean,
    decompose_token,
    feature_top_tokens,
    import_jlens,
    load_lens_model,
    resolve_k,
)
from sparse_readout_prism.research.qwen_readout import load_sae
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import resolve_device, write_json


def log(msg: str) -> None:
    print(f"[readouts] {msg}", flush=True)


@torch.no_grad()
def run(args, *, device: torch.device, sae: tuple, k: int) -> dict:
    jlens = import_jlens()

    loaded = load_lens_model(args.model_id, device=device)
    hf, tok, model = loaded.hf, loaded.tok, loaded.lens_model
    # Lens-free control: J = I at every layer, i.e. no transport at all. This
    # is the floor the cross-lens agreement has to beat: how often two surface
    # forms share a dominant readout feature when no lens is involved.
    identity = args.lens == "identity"
    if identity:
        ref = jlens.JacobianLens.load(args.layers_from)
        layers = sorted(ref.jacobians)
        del ref
        lens, J = None, None
    else:
        lens = jlens.JacobianLens.load(args.lens)
        layers = sorted(lens.jacobians)
        J = {l: lens.jacobians[l].float().to(device) for l in layers}
    W = loaded.lm_head.weight.detach().float().to(device)  # (vocab, d_model)
    final_norm = loaded.final_norm
    decoder, encoder_w, encoder_b, _config, ckpt_row_mean = sae
    row_mean = centering_row_mean(W, args.centering, tok=tok, ckpt_row_mean=ckpt_row_mean)  # (d_model,)
    decoder, encoder_w, encoder_b = decoder.to(device), encoder_w.to(device), encoder_b.to(device)
    decompose = partial(
        decompose_token,
        W=W,
        row_mean=row_mean,
        decoder=decoder,
        encoder_w=encoder_w,
        encoder_b=encoder_b,
        k=k,
        top_feats=args.top_feats,
    )

    spec = json.loads(Path(args.prompts).read_text())["prompts"]
    positions = list(range(-args.n_positions, 0))
    log(f"{len(spec)} prompts, layers={layers}, positions={positions}")

    captured = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles = [model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    records = []
    seen = set()
    for vi, v in enumerate(spec):
        ids = tok(v["prompt"], return_tensors="pt").input_ids.to(device)
        greedy = hf.generate(ids, max_new_tokens=8, do_sample=False)[0, ids.shape[1] :]
        continuation = tok.decode(greedy)
        # First token id of each target string (multi-token targets probed by head).
        target_ids = {}
        for t in v["targets"]:
            t_ids = tok(t, add_special_tokens=False).input_ids
            if t_ids:
                target_ids[t] = t_ids[0]

        if identity:
            with torch.no_grad():
                hf(ids)
            per_layer = {}
            for l in layers:
                hs = torch.stack([final_norm(captured[l][0, pos].to(torch.bfloat16)).float() for pos in positions])
                per_layer[l] = hs @ W.T
        else:
            per_layer, final_logits, _ = lens.apply(model, v["prompt"], positions=positions)
        rec = {
            "id": v["id"],
            "group": v["group"],
            "continuation": continuation,
            "intermediate": v.get("intermediate"),
            "answer": v.get("answer"),
            "targets": {t: int(i) for t, i in target_ids.items()},
            "layers": {},
        }
        for l in layers:
            ll = per_layer[l].float().to(device)  # (n_positions, vocab)
            lrec = {}
            for pi, pos in enumerate(positions):
                src = captured[l][0, pos].float()
                transported = src if identity else J[l] @ src
                h_t = final_norm(transported.to(torch.bfloat16)).float()
                logits_p = ll[pi]
                if vi == 0:
                    # Position-alignment guard: the hooked state at `pos` must
                    # reproduce lens.apply's logits at slot `pi`, else the
                    # decomposition and the lens fields are silently misaligned.
                    rel = float((h_t @ W.T - logits_p).abs().max() / logits_p.abs().max().clamp_min(1e-6))
                    if rel > 5e-2:
                        raise RuntimeError(f"position misalignment at layer {l} pos {pos}: rel={rel:.3f}")
                top5 = logits_p.topk(5)
                entry = {
                    "top5": [
                        (tok.decode([t]), round(float(s), 2))
                        for t, s in zip(top5.indices.tolist(), top5.values.tolist())
                    ],
                    "targets": {},
                }
                if args.decompose_top1:
                    d1 = decompose(h_t, int(top5.indices[0]))
                    d1["top_features"] = d1["top_features"][:8]
                    entry["top1_decomp"] = d1
                    seen.update(f["id"] for f in d1["top_features"])
                for t, tid in target_ids.items():
                    d = decompose(h_t, tid)
                    d["lens_logit"] = float(logits_p[tid])
                    d["lens_rank"] = int((logits_p > logits_p[tid]).sum())
                    entry["targets"][t] = d
                    seen.update(f["id"] for f in d["top_features"])
                lrec[str(pos)] = entry
            rec["layers"][str(l)] = lrec
        records.append(rec)
        log(f"{vi + 1}/{len(spec)}: {v['id']} -> {continuation!r}")

    for h in handles:
        h.remove()
    log(f"labeling {len(seen)} unique features")
    labels = feature_top_tokens(W, row_mean, seen, encoder_w, encoder_b, tok, top_tokens=12)

    return {
        "model": args.model_id,
        "lens": args.lens,
        "sae": args.checkpoint,
        "positions": positions,
        "layers": layers,
        "records": records,
        "feature_top_tokens": labels,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--lens",
        required=True,
        help="path to a fitted lens .pt (JacobianLens container), or the literal 'identity' for the "
        "lens-free control (no transport; plain logit-lens readout)",
    )
    p.add_argument(
        "--layers-from",
        default=None,
        help="required with --lens identity: a fitted lens .pt whose layer set the control copies, "
        "so the comparison is layer-matched",
    )
    p.add_argument(
        "--checkpoint", required=True, help="readout dictionary checkpoint.pt (paper: Qwen3.5-9B k=128 seed 0, 8x)"
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="active features per row code (default: the checkpoint's trained k; a different value warns)",
    )
    p.add_argument(
        "--centering",
        choices=CENTERING_MODES,
        default="live",
        help="centering mean: live = full-vocabulary mean of the live head (paper), "
        "trained = the checkpoint's row_mean (else the tokenizer's text-token mean)",
    )
    p.add_argument("--prompts", required=True, help="prompt bank JSON ({'prompts': [...]})")
    p.add_argument("--n-positions", type=int, default=4, help="score the last n prompt positions")
    p.add_argument("--decompose-top1", action="store_true", help="also decompose the lens's own top-1 token")
    p.add_argument("--top-feats", type=int, default=10, help="signed feature contributions kept per decomposition")
    p.add_argument("--device", default="cuda", help="torch device for the model and the dictionary (paper: cuda)")
    p.add_argument("--out", required=True, help="output dump JSON")
    args = p.parse_args(argv)
    if args.lens == "identity" and not args.layers_from:
        p.error("--lens identity requires --layers-from <fitted lens .pt>")

    sae = load_sae(args.checkpoint)
    k = resolve_k(args.k, sae[3])
    payload = run(args, device=resolve_device(args.device), sae=sae, k=k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, out)
    write_json(
        {
            "n_records": len(payload["records"]),
            "layers": payload["layers"],
            "positions": payload["positions"],
            "k": k,
            "centering": args.centering,
            "provenance": run_provenance(args),
        },
        out.with_suffix(".manifest.json"),
    )
    log(f"wrote {len(payload['records'])} prompts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
