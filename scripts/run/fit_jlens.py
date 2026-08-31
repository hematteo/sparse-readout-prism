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
the rest of the repository installs and tests without it. The model is loaded
through ``research.cross_lens.load_lens_model`` (``utils.load_causal_lm`` on
``--device``, default ``cuda``).

Source layers are ``pick_source_layers(n_layers, --n-layers)``: evenly spaced over
5-95% depth, excluding layer 0 and the final (target) layer. For Qwen3.5-9B and
``--n-layers 12`` this gives {2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 29}.

Resume safety. ``fit`` writes ``shard{i}.meta.json`` next to the shard lens
(model id, prompt manifest, sha1 of the prompt slice ``[start, start+n_prompts)``,
``n_prompts``, ``start``, shard index and count, ``n_layers``) and refuses to
reuse an existing ``shard{i}.lens.pt`` or ``shard{i}.ckpt.pt`` whose sidecar
differs. ``merge`` requires the sidecars, checks that every shard agrees on
them, and records the shared values (plus per-shard prompt counts and
provenance) in ``<out stem>.meta.json``.

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

Fixed in 0.2.1:
  - The out-of-memory fallback in ``fit`` stopped once ``dim_batch < 2``, so
    ``--dim-batch 1`` was never attempted and ``2`` / ``3`` got no fallback; it
    now tries the requested value and halves down to 1 (8 -> 8, 4, 2, 1; 3 -> 3, 1).
  - Shard resume only checked that ``shard{i}.lens.pt`` existed, so a re-run
    with a different prompt slice, budget or layer count reused a stale shard
    or resumed jlens's checkpoint into a mixed fit; see "Resume safety" above.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from sparse_readout_prism.research.cross_lens import (
    check_resume_meta,
    cli_args,
    import_jlens,
    load_lens_model,
    prompt_list_sha1,
    write_resume_meta,
)
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import git_commit, read_json, resolve_device, write_json

# The fit parameters every shard of one lens must share (checked by ``merge``).
SHARED_META_KEYS = ("model_id", "prompts_json", "prompts_sha1", "n_prompts", "start", "num_shards", "n_layers")


def log(msg: str) -> None:
    print(f"[fit_jlens] {msg}", flush=True)


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
    jlens = import_jlens()

    model = load_lens_model(args.model_id, device=resolve_device(args.device)).lens_model
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


def dim_batch_schedule(dim_batch: int) -> list[int]:
    """The requested ``dim_batch``, then halved down to 1: 8 -> [8, 4, 2, 1], 3 -> [3, 1], 1 -> [1]."""
    if dim_batch < 1:
        raise ValueError(f"--dim-batch must be >= 1, got {dim_batch}")
    schedule = []
    while dim_batch >= 1:
        schedule.append(dim_batch)
        dim_batch //= 2
    return schedule


def fit_with_fallback(fit, model, prompts: list[str], layers: list[int], dim_batch: int, ckpt: Path):
    """``fit`` (``jlens.fit``) at each :func:`dim_batch_schedule` value until one does not OOM.

    Every attempt resumes jlens's own checkpoint at ``ckpt``, so a fallback
    continues from the prompts already accumulated. The retry happens after
    the except block has closed: the exception traceback pins the failed
    attempt's autograd graph, and an in-except retry OOMs against that ghost
    memory.
    """
    for db in dim_batch_schedule(dim_batch):
        try:
            return fit(model, prompts, source_layers=layers, dim_batch=db, checkpoint_path=str(ckpt), resume=True)
        except torch.cuda.OutOfMemoryError:
            pass
        log(f"OOM at dim_batch={db}; freeing and halving")
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError("all dim_batch fallbacks OOMed")


def shard_meta(args, prompts: list[str]) -> dict:
    """The parameters that define one shard's fit, stored in ``shard{i}.meta.json``."""
    return {
        "model_id": args.model_id,
        "prompts_json": args.prompts_json,
        "prompts_sha1": prompt_list_sha1(prompts),
        "n_prompts": args.n_prompts,
        "start": args.start,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_layers": args.n_layers,
    }


def cmd_fit(args) -> None:
    jlens = import_jlens()

    prompts = read_prompts(args.prompts_json, args.n_prompts, start=args.start)
    shard_prompts = prompts[args.shard :: args.num_shards]
    ckpt = Path(args.ckpt_dir) / f"shard{args.shard}.ckpt.pt"
    out = Path(args.out_dir) / f"shard{args.shard}.lens.pt"
    meta_path = Path(args.out_dir) / f"shard{args.shard}.meta.json"
    meta = shard_meta(args, prompts)
    check_resume_meta(meta_path, meta, [out, ckpt])
    if out.exists():
        log(f"[resume] shard {args.shard} lens exists: {out}")
        return
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    write_resume_meta(meta_path, meta)

    model = load_lens_model(args.model_id, device=resolve_device(args.device)).lens_model
    layers = pick_source_layers(model.n_layers, args.n_layers)
    log(
        f"git={git_commit()} model={args.model_id} shard={args.shard}/{args.num_shards} "
        f"prompt_slice=[{args.start},{args.start + args.n_prompts}) prompts={len(shard_prompts)} "
        f"layers={layers} dim_batch={args.dim_batch} ckpt={ckpt}"
    )
    t0 = time.time()
    lens = fit_with_fallback(jlens.fit, model, shard_prompts, layers, args.dim_batch, ckpt)
    for layer, J in lens.jacobians.items():
        assert torch.isfinite(J).all(), f"non-finite Jacobian at layer {layer}"
    tmp = str(out) + ".tmp"
    lens.save(tmp)
    os.replace(tmp, out)
    log(f"shard {args.shard} done: {lens.n_prompts} prompts in {(time.time() - t0) / 3600:.2f} h -> {out}")


def check_shard_metas_agree(metas: list[dict], *, num_shards: int) -> dict:
    """The fit parameters shared by all shard sidecars; raises naming any they disagree on."""
    diffs = []
    for key in SHARED_META_KEYS:
        vals = [m.get(key) for m in metas]
        if any(v != vals[0] for v in vals):
            diffs.append(f"{key}: {vals}")
    if [m.get("shard") for m in metas] != list(range(len(metas))):
        diffs.append(f"shard: {[m.get('shard') for m in metas]} (expected 0..{len(metas) - 1})")
    if metas[0].get("num_shards") != num_shards:
        diffs.append(f"num_shards: sidecars say {metas[0].get('num_shards')}, --num-shards is {num_shards}")
    if diffs:
        raise RuntimeError("shard sidecars disagree; these shards are not one fit\n  " + "\n  ".join(diffs))
    return {k: metas[0].get(k) for k in SHARED_META_KEYS}


def cmd_merge(args) -> None:
    jlens = import_jlens()

    final = Path(args.out)
    if final.exists():
        log(f"[resume] merged lens exists: {final}")
        return
    shard_dir = Path(args.shard_dir)
    metas = []
    for i in range(args.num_shards):
        meta_path = shard_dir / f"shard{i}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing: shards written before 0.2.1 carry no sidecar, so their fit cannot be "
                "verified; refit the shard (or write the sidecar by hand from the run log)"
            )
        metas.append(read_json(meta_path))
    shared = check_shard_metas_agree(metas, num_shards=args.num_shards)
    lenses = [jlens.JacobianLens.load(str(shard_dir / f"shard{i}.lens.pt")) for i in range(args.num_shards)]
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
    write_json(
        {
            **shared,
            "shard_n_prompts": [lens.n_prompts for lens in lenses],
            "merged_n_prompts": merged.n_prompts,
            "layers": merged.source_layers,
            "provenance": run_provenance(cli_args(args)),
        },
        final.with_suffix(".meta.json"),
    )
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
    sp.add_argument("--device", default="cuda", help="torch device for the model (paper: cuda)")
    sp.set_defaults(fn=cmd_smoke)

    sp = sub.add_parser("fit", help="fit one prompt shard on one GPU")
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--prompts-json", required=True)
    sp.add_argument("--n-prompts", type=int, required=True)
    sp.add_argument("--start", type=int, default=0, help="offset into the frozen prompt manifest")
    sp.add_argument("--shard", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--n-layers", type=int, default=12)
    sp.add_argument("--dim-batch", type=int, default=16, help="output dims per backward pass; halved on OOM down to 1")
    sp.add_argument("--ckpt-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--device", default="cuda", help="torch device for the model (paper: cuda)")
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
