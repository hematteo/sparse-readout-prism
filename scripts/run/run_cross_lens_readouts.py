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

Decomposition basis. ``--sae`` is the k=128 seed-0 readout dictionary for
Qwen3.5-9B (8x, D=32768), Hugging Face ``hematteo/sparse-readout-prism`` file
``qwen3.5-9b/k128_8x/checkpoint.pt``, a different operating point from the main
tables' 32x/k=256 dictionaries. Rows of the LM head are centred on the vocabulary
mean and unit-normalised before encoding, and the contribution of feature ``i`` to
the score of token ``t`` is ``||W_t - mu|| * code_i(t) * (h . d_i)``.

Bank schema: ``{"prompts": [{"id", "group", "prompt", "targets": [...], ...}]}``
(see ``data/cross_lens/README.md``). Every target string is probed by the id of
its first token. ``jlens`` is imported lazily (``uv sync --extra lens``).

Paper runs (Qwen3.5-9B, one run per lens and bank, final prompt position only)::

  run_cross_lens_readouts.py --model-id Qwen/Qwen3.5-9B --lens <lens.pt> --sae <checkpoint.pt> \
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
from pathlib import Path

import torch
import torch.nn.functional as F

from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import write_json


def log(msg: str) -> None:
    print(f"[readouts] {msg}", flush=True)


def _import_jlens():
    try:
        import jlens
    except ImportError as e:
        raise ImportError(
            "jlens (the Jacobian-lens reference implementation, Apache-2.0, "
            "github.com/anthropics/jacobian-lens) is not installed; install it with `uv sync --extra lens`"
        ) from e
    return jlens


def load_sae(checkpoint: Path):
    """Raw TopK dictionary tensors from a training checkpoint (decoder rows unit-normalised)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    decoder = state["decoder"].float().contiguous()  # (d_features, d_model)
    encoder_w = state["encoder.weight"].float().contiguous()  # (d_features, d_model)
    encoder_b = state["encoder.bias"].float().contiguous()  # (d_features,)
    decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return decoder, encoder_w, encoder_b


def encode_topk(x, encoder_w, encoder_b, k):
    acts = F.relu(x @ encoder_w.T + encoder_b)
    values, indices = torch.topk(acts, k=min(k, acts.shape[-1]), dim=-1)
    code = torch.zeros_like(acts)
    code.scatter_(dim=-1, index=indices, src=values)
    return code


@torch.no_grad()
def feature_top_tokens(W, row_mean, feature_ids, encoder_w, encoder_b, tokenizer, top_tokens=12, chunk=8192):
    """Top unembedding rows (by encoder activation) for each feature id, as decoded strings."""
    if not feature_ids:
        return {}
    device = W.device
    fids = torch.tensor(sorted(feature_ids), dtype=torch.long)
    enc = encoder_w[fids].to(device)
    bias = encoder_b[fids].to(device)
    best_scores = torch.full((len(fids), top_tokens), -float("inf"), device=device)
    best_ids = torch.zeros((len(fids), top_tokens), dtype=torch.long, device=device)
    for start in range(0, W.shape[0], chunk):
        rows = W[start : start + chunk].float()
        centered = rows - row_mean
        x = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scores = F.relu(x @ enc.T + bias).T
        merged = torch.cat([best_scores, scores], dim=1)
        ids = torch.arange(start, start + rows.shape[0], device=device).expand(len(fids), -1)
        merged_ids = torch.cat([best_ids, ids], dim=1)
        best_scores, keep = torch.topk(merged, k=top_tokens, dim=1)
        best_ids = torch.gather(merged_ids, 1, keep)
    return {
        int(fid): [tokenizer.decode([t]) for t, s in zip(best_ids[i].tolist(), best_scores[i].tolist()) if s > 0]
        for i, fid in enumerate(fids.tolist())
    }


@torch.no_grad()
def run(args) -> dict:
    import transformers

    jlens = _import_jlens()

    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).cuda().eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = jlens.from_hf(hf, tok)
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
        J = {l: lens.jacobians[l].float().cuda() for l in layers}
    W = hf.get_output_embeddings().weight.detach().float().cuda()  # (vocab, d_model)
    row_mean = W.mean(dim=0)
    final_norm = hf.model.norm
    decoder, encoder_w, encoder_b = load_sae(Path(args.sae))
    decoder, encoder_w, encoder_b = decoder.cuda(), encoder_w.cuda(), encoder_b.cuda()

    spec = json.loads(Path(args.prompts).read_text())["prompts"]
    positions = list(range(-args.n_positions, 0))
    log(f"{len(spec)} prompts, layers={layers}, positions={positions}")

    captured = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles = [model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    def decompose(h_state, token_id):
        W_row = W[token_id]
        centered = W_row - row_mean
        norm = centered.norm().clamp_min(1e-8)
        code = encode_topk((centered / norm)[None, :], encoder_w, encoder_b, args.k)[0]
        contributions = norm * code * (h_state @ decoder.T)
        base = float(h_state @ row_mean)
        feat_sum = float(contributions.sum())
        original = float(h_state @ W_row)
        active = torch.nonzero(contributions != 0).flatten()
        top = active[contributions[active].abs().argsort(descending=True)[: args.top_feats]]
        return {
            "original_logit": original,
            "base": base,
            "feature_sum": feat_sum,
            "residual": original - base - feat_sum,
            "top_features": [{"id": int(f), "contribution": float(contributions[f])} for f in top],
        }

    records = []
    seen = set()
    for vi, v in enumerate(spec):
        ids = tok(v["prompt"], return_tensors="pt").input_ids.cuda()
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
            ll = per_layer[l].float().cuda()  # (n_positions, vocab)
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
    labels = feature_top_tokens(W, row_mean, seen, encoder_w, encoder_b, tok)

    return {
        "model": args.model_id,
        "lens": args.lens,
        "sae": args.sae,
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
    p.add_argument("--sae", required=True, help="readout SAE checkpoint.pt (paper: Qwen3.5-9B k=128 seed 0, 8x)")
    p.add_argument("--k", type=int, default=128, help="active features per row code")
    p.add_argument("--prompts", required=True, help="prompt bank JSON ({'prompts': [...]})")
    p.add_argument("--n-positions", type=int, default=4, help="score the last n prompt positions")
    p.add_argument("--decompose-top1", action="store_true", help="also decompose the lens's own top-1 token")
    p.add_argument("--top-feats", type=int, default=10, help="signed feature contributions kept per decomposition")
    p.add_argument("--out", required=True, help="output dump JSON")
    args = p.parse_args(argv)
    if args.lens == "identity" and not args.layers_from:
        p.error("--lens identity requires --layers-from <fitted lens .pt>")

    payload = run(args)

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
            "provenance": run_provenance(args),
        },
        out.with_suffix(".manifest.json"),
    )
    log(f"wrote {len(payload['records'])} prompts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
