#!/usr/bin/env python3
"""The published antonym prompt, layer by layer, under one fitted lens (``tab:app-cross-lens-antonym-layers``).

``run`` reads the prompt ``"小"的反义词是"`` (model answer 大) through one fitted
lens on Qwen3.5-9B and records, per fitted layer and at the final position: the
top-10 tokens with softmax shares under (a) the lens transport and (b) the
identity transport (logit lens, same final norm), plus the Sparse Readout Prism
decomposition of the English and Chinese answer forms (' big', 'big', '大', with
the ' small' / '小' contrasts) on the transported state, with the top signed
feature contributions itemised. It is run once per lens. ``table`` then assembles
the layer x lens table of top-1 token and softmax share (and the dominant feature
of each probed form under each lens) from the per-lens JSON files.

Decomposition basis: ``--checkpoint``, the k=128 seed-0 readout dictionary for
Qwen3.5-9B (8x, D=32768), Hugging Face ``hematteo/sparse-readout-prism`` file
``qwen3.5-9b/k128_8x/checkpoint.pt``, read with ``research.qwen_readout.load_sae``;
``--k`` defaults to the checkpoint's trained k (a different explicit value
warns) and ``--centering`` selects the centering mean (``live`` = full-vocabulary
mean of the live head, the paper; ``trained`` = the checkpoint's ``row_mean`` or
the tokenizer's text-token mean). ``jlens`` is imported lazily
(``uv sync --extra lens``); the model is loaded through
``research.cross_lens.load_lens_model`` on ``--device`` (default ``cuda``).
``run`` writes ``<out stem>.manifest.json`` (resolved k, centering, provenance)
next to its JSON; ``table`` writes ``<out-csv stem>.manifest.json`` next to its CSVs.

Paper runs::

  cross_lens_antonym_layers.py run --model-id Qwen/Qwen3.5-9B \
      --lens <out>/qwen35_9b_jlens_en_seed0_n100.pt --checkpoint <checkpoint.pt> --k 128 \
      --out <out>/antonym_layers_en.json
  cross_lens_antonym_layers.py run --model-id Qwen/Qwen3.5-9B \
      --lens <out>/qwen35_9b_jlens_zh_seed0_n100.pt --checkpoint <checkpoint.pt> --k 128 \
      --out <out>/antonym_layers_zh.json
  cross_lens_antonym_layers.py table --dump EN=<out>/antonym_layers_en.json \
      --dump ZH=<out>/antonym_layers_zh.json --layers 24,26,29,final \
      --out-csv <out>/antonym_layers_table.csv --out-features-csv <out>/antonym_layers_features.csv
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
    parse_dump_args,
    resolve_k,
    write_manifest,
)
from sparse_readout_prism.research.qwen_readout import load_sae
from sparse_readout_prism.utils import resolve_device, write_csv


def log(msg):
    print(f"[antonym] {msg}", flush=True)


@torch.no_grad()
def cmd_run(args) -> None:
    jlens = import_jlens()

    device = resolve_device(args.device)
    loaded = load_lens_model(args.model_id, device=device)
    tok, model = loaded.tok, loaded.lens_model
    lens = jlens.JacobianLens.load(args.lens)
    layers = sorted(lens.jacobians)
    J = {l: lens.jacobians[l].float().to(device) for l in layers}
    W = loaded.lm_head.weight.detach().float().to(device)  # (vocab, d_model)
    final_norm = loaded.final_norm
    decoder, encoder_w, encoder_b, config, ckpt_row_mean = load_sae(args.checkpoint)
    k = resolve_k(args.k, config)
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
        top_feats=16,
    )

    captured = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[("out", idx)] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles = [model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    def norm_hook(module, inputs, output):
        if output.ndim == 3:
            captured["final_norm_out"] = output.detach()

    handles.append(final_norm.register_forward_hook(norm_hook))

    per_layer, final_logits, _ = lens.apply(model, args.prompt, positions=[-1])
    h_ref = captured["final_norm_out"][0, -1].float().clone()
    final_logits = final_logits.float().squeeze().to(device)
    greedy = tok.decode([int(final_logits.argmax())])
    log(f"prompt={args.prompt!r} greedy={greedy!r}")

    def top10(logits):
        pr = torch.softmax(logits, -1)
        v, i = torch.topk(pr, 10)
        return [[tok.decode([t]), round(float(x) * 100, 1)] for t, x in zip(i.tolist(), v.tolist())]

    def probe_ids():
        out = {}
        for text in (" big", "big", "大", " small", "小"):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out[text] = ids[0]
        return out

    probes = probe_ids()
    log(f"single-token probes: {list(probes)}")

    rows, seen = {}, set()
    for l in layers:
        h = captured[("out", l)][0, -1].float()
        h_j = final_norm((J[l] @ h).to(torch.bfloat16)).float()
        h_id = final_norm(h.to(torch.bfloat16)).float()  # logit lens: identity transport
        entry = {
            "jlens_top10": top10(h_j @ W.T),
            "logitlens_top10": top10(h_id @ W.T),
            "targets": {t: decompose(h_j, i) for t, i in probes.items()},
        }
        rows[str(l)] = entry
        for t in entry["targets"].values():
            seen.update(f["id"] for f in t["top_features"])
    rows["final"] = {
        "jlens_top10": top10(final_logits),
        "logitlens_top10": top10(final_logits),
        "targets": {t: decompose(h_ref, i) for t, i in probes.items()},
    }
    for t in rows["final"]["targets"].values():
        seen.update(f["id"] for f in t["top_features"])
    for h in handles:
        h.remove()

    labels = feature_top_tokens(W, row_mean, seen, encoder_w, encoder_b, tok, top_tokens=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model_id,
        "lens": args.lens,
        "sae": args.checkpoint,
        "prompt": args.prompt,
        "greedy": greedy,
        "layers": layers,
        "rows": rows,
        "feature_top_tokens": labels,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, out)
    write_manifest(out, args, prompt=args.prompt, greedy=greedy, layers=layers, k=k, centering=args.centering)

    for L in [str(l) for l in layers] + ["final"]:
        jl = " · ".join(f"{t}({p}%)" for t, p in rows[L]["jlens_top10"][:3])
        ll = " · ".join(f"{t}({p}%)" for t, p in rows[L]["logitlens_top10"][:3])
        log(f"L{L:>5}  lens: {jl}   |   logit-lens: {ll}")
    log(f"wrote -> {out}")


def antonym_layer_table(dumps: dict[str, dict], layers: list[str]) -> tuple[list[dict], list[dict]]:
    """Layer x lens rows of top-1 token / softmax share, and per-probe dominant features.

    ``dumps`` maps a lens label to the JSON payload written by ``run``. Layers are
    the string keys of ``rows`` (fitted layer indices and ``"final"``).
    """
    table, features = [], []
    for L in layers:
        for label, d in dumps.items():
            r = d["rows"][L]
            lens_top, logit_top = r["jlens_top10"][0], r["logitlens_top10"][0]
            table.append(
                {
                    "layer": L,
                    "lens": label,
                    "top1_token": lens_top[0],
                    "top1_softmax_pct": lens_top[1],
                    "logit_lens_top1_token": logit_top[0],
                    "logit_lens_top1_softmax_pct": logit_top[1],
                    "top3_tokens": " | ".join(t for t, _ in r["jlens_top10"][:3]),
                }
            )
            for target, dec in r["targets"].items():
                top = dec["top_features"][0] if dec["top_features"] else {"id": None, "contribution": None}
                features.append(
                    {
                        "layer": L,
                        "lens": label,
                        "target": target,
                        "original_logit": dec["original_logit"],
                        "feature_sum": dec["feature_sum"],
                        "residual": dec["residual"],
                        "dominant_feature": top["id"],
                        "dominant_contribution": top["contribution"],
                    }
                )
    return table, features


def cmd_table(args) -> None:
    dumps = {label: json.loads(path.read_text()) for label, path in parse_dump_args(args.dump)}
    first = next(iter(dumps.values()))
    layers = [s.strip() for s in args.layers.split(",") if s.strip()] if args.layers else list(first["rows"])
    table, features = antonym_layer_table(dumps, layers)
    labels = list(dumps)
    print(f"prompt={first['prompt']!r}")
    print("layer  " + "".join(f"{lb:>28s}" for lb in labels))
    for L in layers:
        cells = [t for t in table if t["layer"] == L]
        print(f"{L:>5s}  " + "".join(f"{c['top1_token']!r} {c['top1_softmax_pct']}%".rjust(28) for c in cells))
    written = []
    if args.out_csv:
        write_csv(args.out_csv, table)
        written.append(args.out_csv)
        print(f"wrote {args.out_csv}")
    if args.out_features_csv:
        write_csv(args.out_features_csv, features)
        written.append(args.out_features_csv)
        print(f"wrote {args.out_features_csv}")
    if written:
        write_manifest(written[0], args, prompt=first["prompt"], layers=layers, lenses=labels)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="read the prompt through one fitted lens (GPU)")
    sp.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    sp.add_argument("--lens", required=True, help="fitted lens .pt (JacobianLens container)")
    sp.add_argument(
        "--checkpoint", required=True, help="readout dictionary checkpoint.pt (paper: Qwen3.5-9B k=128 seed 0, 8x)"
    )
    sp.add_argument(
        "--k",
        type=int,
        default=None,
        help="active features per row code (default: the checkpoint's trained k; a different value warns)",
    )
    sp.add_argument(
        "--centering",
        choices=CENTERING_MODES,
        default="live",
        help="centering mean: live = full-vocabulary mean of the live head (paper), "
        "trained = the checkpoint's row_mean (else the tokenizer's text-token mean)",
    )
    sp.add_argument("--prompt", default='"小"的反义词是"')
    sp.add_argument("--device", default="cuda", help="torch device for the model and the dictionary (paper: cuda)")
    sp.add_argument("--out", required=True, help="output JSON")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("table", help="assemble the layer x lens table from per-lens JSON files")
    sp.add_argument("--dump", action="append", required=True, help="LABEL=path, repeatable (e.g. EN=..., ZH=...)")
    sp.add_argument("--layers", default="24,26,29,final", help="comma-separated layer keys (default: paper rows)")
    sp.add_argument("--out-csv", default=None, help="top-1 token and softmax share per layer x lens")
    sp.add_argument("--out-features-csv", default=None, help="dominant feature per layer x lens x probed form")
    sp.set_defaults(fn=cmd_table)

    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
