#!/usr/bin/env python3
"""Fit the Jacobian lens (jlens) on a seeded C4 prompt sample: prompts, smoke, fit, merge.

Produces the fitted lenses behind the cross-lens study (``fig:cross-lens-butterfly``,
``fig:cross-lens-extension``, ``tab:app-cross-lens-families``,
``tab:app-cross-lens-extension``, ``tab:app-cross-lens-antonym-layers``,
``tab:app-cross-lens-de-antonym-layers``): one Jacobian lens per fitting corpus
(English, Chinese and German C4) with identical seeds, source layers and procedure,
at two fitting budgets (100 and 300 prompts, both read from the head of the same
seeded dump).

Subcommands, run in this order:

  prompts  seeded streaming C4 prompt dump (idempotent, atomic write)
  smoke    tiny fit on a small model with finite / save-load / apply checks
  fit      one prompt shard on one GPU, resumable through the jlens checkpoint
  merge    n_prompts-weighted merge of the shard lenses plus a half-vs-full
           convergence report per layer

The estimator is the reference implementation of the Jacobian lens (``jlens``,
Apache-2.0, https://github.com/anthropics/jacobian-lens); this script only
orchestrates sharding and storage. Jacobians accumulate in fp32 and are saved in
fp16. Install the dependency with ``uv sync --extra lens``; the import is lazy so
the rest of the repository installs and tests without it.

Source layers are ``pick_source_layers(n_layers, --n-layers)``: evenly spaced over
5-95% depth, excluding layer 0 and the final (target) layer. For Qwen3.5-9B and
``--n-layers 12`` this gives {2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 29}.

Paper runs (Qwen3.5-9B, four shards on four GPUs, ``--dim-batch 8``)::

  # seeded prompt dumps, 1000 prompts each (the English dump keeps the default --min-chars 800)
  fit_jlens.py prompts --prompts-json <out>/c4_prompts_en_seed0.json --n-prompts 1000 --seed 0
  fit_jlens.py prompts --prompts-json <out>/c4_prompts_zh_seed0.json --n-prompts 1000 --seed 0 \
      --min-chars 300 --c4-config zh
  fit_jlens.py prompts --prompts-json <out>/c4_prompts_de_seed0.json --n-prompts 1000 --seed 0 \
      --min-chars 300 --c4-config de
  # smoke on a small model before the 9B fit
  fit_jlens.py smoke --model-id Qwen/Qwen3.5-0.8B --prompts-json <out>/c4_prompts_en_seed0.json \
      --out <ckpt>/smoke_0p8b.lens.pt --dim-batch 16
  # one shard per GPU (i in 0..3), then merge; --n-prompts 100 for the main lenses and
  # --n-prompts 300 for the larger-corpus refits
  fit_jlens.py fit --model-id Qwen/Qwen3.5-9B --prompts-json <out>/c4_prompts_en_seed0.json \
      --n-prompts 100 --shard $i --num-shards 4 --n-layers 12 --dim-batch 8 \
      --ckpt-dir <ckpt> --out-dir <ckpt>
  fit_jlens.py merge --shard-dir <ckpt> --num-shards 4 --out <out>/qwen35_9b_jlens_en_seed0_n100.pt

The same fit + merge with the zh / de dumps produce the Chinese- and German-fitted
lenses (``qwen35_9b_jlens_zh_seed0_n100.pt``, ``qwen35_9b_jlens_de_seed0_n100.pt``),
and with ``--n-prompts 300`` the three ``*_n300.pt`` refits.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from sparse_readout_prism.utils import git_commit


def log(msg: str) -> None:
    print(f"[fit_jlens] {msg}", flush=True)


def _import_jlens():
    try:
        import jlens
    except ImportError as e:
        raise ImportError(
            "jlens (the Jacobian-lens reference implementation, Apache-2.0, "
            "github.com/anthropics/jacobian-lens) is not installed; install it with `uv sync --extra lens`"
        ) from e
    return jlens


def load_model(model_id: str):
    import transformers

    jlens = _import_jlens()
    hf = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    return jlens.from_hf(hf, tok)


def pick_source_layers(n_layers: int, n_pick: int) -> list[int]:
    # Evenly spaced over 5-95% depth; excludes layer 0 and the final layer
    # (target). Deterministic given (n_layers, n_pick).
    fracs = [(i + 1) / (n_pick + 1) for i in range(n_pick)]
    layers = sorted({max(1, min(n_layers - 2, round(f * (n_layers - 1)))) for f in fracs})
    return layers


def cmd_prompts(args) -> None:
    out = Path(args.prompts_json)
    if out.exists():
        log(f"[resume] prompts exist: {out}")
        return
    from datasets import load_dataset

    config = args.dataset_config if args.dataset_config is not None else args.c4_config
    ds = load_dataset(args.dataset, config, split=args.dataset_split, streaming=True, revision=args.revision)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    prompts = []
    scanned = 0
    for ex in ds:
        scanned += 1
        text = (ex[args.text_field] or "").strip()
        if len(text) < args.min_chars:
            continue
        prompts.append(text[:4000])  # jlens.fit truncates to its max_seq_len tokens
        if len(prompts) >= args.n_prompts:
            break
    if len(prompts) < args.n_prompts:
        raise RuntimeError(
            f"only {len(prompts)}/{args.n_prompts} prompts after scanning {scanned} records "
            f"({args.dataset} {config}); widen the stream or relax --min-chars"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": f"{args.dataset} {config} streaming",
        "dataset": args.dataset,
        "dataset_config": config,
        "dataset_split": args.dataset_split,
        "revision": args.revision,
        "text_field": args.text_field,
        "records_scanned": scanned,
        "seed": args.seed,
        "min_chars": args.min_chars,
        "n": len(prompts),
        "prompts": prompts,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    log(f"wrote {len(prompts)} prompts -> {out}")


def read_prompts(path: str, n: int | None = None, start: int = 0) -> list[str]:
    prompts = json.loads(Path(path).read_text())["prompts"]
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    prompts = prompts[start:]
    return prompts[:n] if n else prompts


def cmd_smoke(args) -> None:
    jlens = _import_jlens()

    model = load_model(args.model_id)
    prompts = read_prompts(args.prompts_json, 2)
    layers = pick_source_layers(model.n_layers, 4)
    t0 = time.time()
    lens = jlens.fit(model, prompts, source_layers=layers, dim_batch=args.dim_batch)
    dt = time.time() - t0
    for layer, J in lens.jacobians.items():
        assert torch.isfinite(J).all(), f"non-finite Jacobian at layer {layer}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(out))
    jlens.JacobianLens.load(str(out))
    per_layer, final_logits, _ = lens.apply(model, "The capital of France is", positions=[-1])
    assert torch.isfinite(final_logits).all()
    for layer, logits in per_layer.items():
        assert torch.isfinite(logits).all(), f"non-finite lens logits at layer {layer}"
    log(
        f"smoke OK: d_model={model.d_model}, layers={layers}, "
        f"{dt:.1f}s for 2 prompts ({dt / (2 * model.d_model) * 1e3:.2f} ms per prompt-dim)"
    )


def cmd_fit(args) -> None:
    jlens = _import_jlens()

    prompts = read_prompts(args.prompts_json, args.n_prompts, start=args.start)
    shard_prompts = prompts[args.shard :: args.num_shards]
    model = load_model(args.model_id)
    layers = pick_source_layers(model.n_layers, args.n_layers)
    ckpt = Path(args.ckpt_dir) / f"shard{args.shard}.ckpt.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"shard{args.shard}.lens.pt"
    if out.exists():
        log(f"[resume] shard {args.shard} lens exists: {out}")
        return
    log(
        f"git={git_commit()} model={args.model_id} shard={args.shard}/{args.num_shards} "
        f"prompt_slice=[{args.start},{args.start + args.n_prompts}) prompts={len(shard_prompts)} "
        f"layers={layers} dim_batch={args.dim_batch} ckpt={ckpt}"
    )
    t0 = time.time()
    # Retry OUTSIDE the except block: the exception traceback pins the failed
    # attempt's autograd graph, so an in-except retry OOMs against ghost memory.
    lens = None
    for dim_batch in (args.dim_batch, args.dim_batch // 2, args.dim_batch // 4):
        if dim_batch < 2:
            break
        oom = False
        try:
            lens = jlens.fit(
                model, shard_prompts, source_layers=layers, dim_batch=dim_batch, checkpoint_path=str(ckpt), resume=True
            )
        except torch.cuda.OutOfMemoryError:
            oom = True
        if not oom:
            break
        log(f"OOM at dim_batch={dim_batch}; freeing and halving")
        gc.collect()
        torch.cuda.empty_cache()
    if lens is None:
        raise RuntimeError("all dim_batch fallbacks OOMed")
    for layer, J in lens.jacobians.items():
        assert torch.isfinite(J).all(), f"non-finite Jacobian at layer {layer}"
    tmp = str(out) + ".tmp"
    lens.save(tmp)
    os.replace(tmp, out)
    log(f"shard {args.shard} done: {lens.n_prompts} prompts in {(time.time() - t0) / 3600:.2f} h -> {out}")


def cmd_merge(args) -> None:
    jlens = _import_jlens()

    final = Path(args.out)
    if final.exists():
        log(f"[resume] merged lens exists: {final}")
        return
    lenses = [jlens.JacobianLens.load(str(Path(args.shard_dir) / f"shard{i}.lens.pt")) for i in range(args.num_shards)]
    merged = jlens.JacobianLens.merge(lenses)
    # Convergence report: first half of the shards vs all of them. Small
    # relative deltas mean the prompt budget has saturated.
    half = jlens.JacobianLens.merge(lenses[: max(1, args.num_shards // 2)])
    for layer in sorted(merged.jacobians):
        rel = (half.jacobians[layer].float() - merged.jacobians[layer].float()).norm()
        rel = rel / merged.jacobians[layer].float().norm()
        log(f"convergence layer {layer}: ||J_half - J_full|| / ||J_full|| = {rel:.4f}")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(final) + ".tmp"
    merged.save(tmp)
    os.replace(tmp, final)
    log(f"merged {sum(lens.n_prompts for lens in lenses)} prompts, {len(merged.jacobians)} layers -> {final}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prompts", help="seeded streaming C4 prompt dump")
    sp.add_argument("--prompts-json", required=True, help="output manifest ({'prompts': [...], ...})")
    sp.add_argument("--n-prompts", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--min-chars", type=int, default=800)
    sp.add_argument("--c4-config", default="en", help="C4 language config, e.g. en, zh, de")
    sp.add_argument("--dataset", default="allenai/c4", help="HF dataset id")
    sp.add_argument("--dataset-config", default=None, help="dataset config; falls back to --c4-config")
    sp.add_argument("--dataset-split", default="train")
    sp.add_argument("--revision", default=None, help="optional dataset revision pin (the paper runs used none)")
    sp.add_argument("--text-field", default="text", help="record field holding the document text")
    sp.set_defaults(fn=cmd_prompts)

    sp = sub.add_parser("smoke", help="tiny fit with finite / save-load / apply checks")
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--prompts-json", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--dim-batch", type=int, default=16)
    sp.set_defaults(fn=cmd_smoke)

    sp = sub.add_parser("fit", help="fit one prompt shard on one GPU")
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--prompts-json", required=True)
    sp.add_argument("--n-prompts", type=int, required=True)
    sp.add_argument("--start", type=int, default=0, help="offset into the frozen prompt manifest")
    sp.add_argument("--shard", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--n-layers", type=int, default=12)
    sp.add_argument("--dim-batch", type=int, default=16)
    sp.add_argument("--ckpt-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(fn=cmd_fit)

    sp = sub.add_parser("merge", help="merge shard lenses into one lens file")
    sp.add_argument("--shard-dir", required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_merge)

    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
