"""Cross-lens study toolkit: lens-model loading, the readout decomposition, and the aggregation core.

Shared by the lens fitters (``scripts/run/fit_jlens.py``, ``scripts/run/fit_ridge_lens.py``),
the readout runner (``scripts/run/run_cross_lens_readouts.py``), the antonym
layer study (``scripts/analyze/cross_lens_antonym_layers.py``), the two
aggregators (``scripts/eval/aggregate_cross_lens_en_{zh,de}.py``), the
three-lens table and shared-feature scripts, and the EN-DE bank builder. Until
0.2.1 each of those carried its own copy of the pieces below.

* :func:`import_jlens` -- lazy import of the Jacobian-lens reference
  implementation (``uv sync --extra lens``).
* :func:`load_lens_model` / :func:`find_final_norm` -- one model loader for the
  GPU scripts, built on ``utils.load_causal_lm`` and ``utils.find_lm_head``.
* :func:`decompose_token` / :func:`feature_top_tokens` -- the per-token Sparse
  Readout Prism decomposition of a transported state and the raw-decode feature
  labels the paper's dumps carry.
* :func:`resolve_k` / :func:`centering_row_mean` -- the ``--k`` default and the
  ``--centering {live,trained}`` switch of the two dictionary-reading scripts.
* :func:`majority_same` / :func:`token_diverges` -- the mid-band majority votes,
  with ``rule="half"`` (the paper: at least half of the scored layers) or
  ``rule="strict"`` (more than half).
* :func:`aggregate_pair` / :func:`print_summary` / :func:`write_summary` -- the
  aggregation core, parameterised by a :class:`PairSpec` so the EN-ZH and EN-DE
  entry points differ only in their surface-language call; with
  :func:`load_dump_records` / :func:`load_bank_items` / :func:`parse_dump_args`
  as the dump and bank readers.
* :func:`check_resume_meta` / :func:`write_resume_meta` -- the sidecar that
  makes shard / accumulator resume refuse a different fit.
* :func:`write_manifest` / :func:`cli_args` -- ``<stem>.manifest.json`` with
  ``run_provenance`` for the CSV-writing scripts (subcommand callables stripped).
* :func:`fold_text` -- casefold + eszett + diacritic folding (bank builder and
  the EN-DE lexical call).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sparse_readout_prism.data import center_normalize_rows, centering_mean, token_mask_from_tokenizer
from sparse_readout_prism.research.qwen_readout import encode_topk
from sparse_readout_prism.research.run_io import run_provenance
from sparse_readout_prism.utils import find_lm_head, load_causal_lm, write_json

MIDBAND = ["21", "24", "26", "29"]
POS = "-1"
AGREEMENT_RULES = ("half", "strict")
NULL_POPULATIONS = ("all", "cross")
CENTERING_MODES = ("live", "trained")


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #


def import_jlens():
    try:
        import jlens
    except ImportError as e:
        raise ImportError(
            "jlens (the Jacobian-lens reference implementation, Apache-2.0, "
            "github.com/anthropics/jacobian-lens) is not installed; install it with `uv sync --extra lens`"
        ) from e
    return jlens


_FINAL_NORM_PATHS = ("model.norm", "model.language_model.norm", "language_model.norm", "norm")


def find_final_norm(model: torch.nn.Module) -> torch.nn.Module:
    """The final pre-unembedding norm of a HF causal LM (plain or multimodal wrapper)."""
    for path in _FINAL_NORM_PATHS:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            return obj
    raise RuntimeError(
        f"could not locate the final norm on {type(model).__name__} (tried {', '.join(_FINAL_NORM_PATHS)})"
    )


@dataclass
class LoadedLensModel:
    """A HF model on its device plus the handles the cross-lens scripts read through."""

    hf: torch.nn.Module
    tok: Any
    lens_model: Any  # jlens.HFLensModel over ``hf``
    lm_head: torch.nn.Module
    final_norm: torch.nn.Module
    device: torch.device


def load_lens_model(
    model_id: str, *, device: torch.device | str, dtype: torch.dtype = torch.bfloat16
) -> LoadedLensModel:
    """Load ``model_id`` through ``utils.load_causal_lm`` and wrap it for ``jlens``.

    ``load_causal_lm(device_map=None)`` loads on CPU with frozen parameters; the
    model is then moved to ``device`` whole (the paper runs: one 9B model per
    GPU in bf16). ``jlens.from_hf`` locates the residual stack; ``lm_head`` and
    ``final_norm`` are the repo-wide ``utils.find_lm_head`` and
    :func:`find_final_norm` so the readout matrix and the norm applied to
    transported states come from the same places every other script uses.
    """
    jlens = import_jlens()
    device = torch.device(device)
    hf, tok = load_causal_lm(model_id, dtype=dtype, device_map=None)
    hf = hf.to(device).eval()
    lens_model = jlens.from_hf(hf, tok)
    return LoadedLensModel(
        hf=hf,
        tok=tok,
        lens_model=lens_model,
        lm_head=find_lm_head(hf),
        final_norm=find_final_norm(hf),
        device=device,
    )


# --------------------------------------------------------------------------- #
# dictionary: k, centering, decomposition, labels
# --------------------------------------------------------------------------- #


def resolve_k(requested: int | None, config: dict[str, Any]) -> int:
    """Active features per code: the checkpoint's ``config['factorizer']['k']`` unless overridden.

    An explicit ``requested`` value that differs from the checkpoint's wins but
    raises a ``UserWarning`` -- the dictionary was trained at its own k and a
    different one changes every code.
    """
    trained = (config or {}).get("factorizer", {}).get("k")
    if requested is None:
        if trained is None:
            raise ValueError("checkpoint config carries no factorizer.k; pass --k explicitly")
        return int(trained)
    if trained is not None and int(trained) != int(requested):
        warnings.warn(
            f"--k {requested} differs from the checkpoint's trained k={trained}; codes will not match training",
            UserWarning,
            stacklevel=2,
        )
    return int(requested)


def centering_row_mean(W: torch.Tensor, mode: str, *, tok: Any, ckpt_row_mean: torch.Tensor | None) -> torch.Tensor:
    """``data.centering_mean`` for scripts that only have the live model.

    ``live`` is the full-vocabulary mean of the live (bf16 -> fp32) head, which
    is how the paper's cross-lens dumps were computed. ``trained`` is the
    checkpoint's stored ``row_mean`` when it has one, else the mean over the
    text-token rows ``data.token_mask_from_tokenizer(tok, vocab)`` (how a masked
    extraction trains). Returned on ``W``'s device.
    """
    if mode not in CENTERING_MODES:
        raise ValueError(f"unknown centering mode {mode!r} (expected one of {CENTERING_MODES})")
    token_mask = None
    if mode == "trained" and ckpt_row_mean is None:
        token_mask = token_mask_from_tokenizer(tok, W.shape[0])
    return centering_mean(W, mode=mode, token_mask=token_mask, ckpt={"row_mean": ckpt_row_mean}).to(W.device)


@torch.no_grad()
def decompose_token(
    h_state: torch.Tensor,
    token_id: int,
    *,
    W: torch.Tensor,
    row_mean: torch.Tensor,
    decoder: torch.Tensor,
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    k: int,
    top_feats: int,
) -> dict[str, Any]:
    """Sparse Readout Prism decomposition of the score ``h_state . W[token_id]``.

    Row ``t`` is centred on ``row_mean`` and unit-normalised
    (``data.center_normalize_rows``), top-k encoded, and feature ``i``
    contributes ``||W_t - mu|| * code_i(t) * (h . d_i)``. Returns the dump entry
    ``{original_logit, base, feature_sum, residual, top_features}`` where
    ``top_features`` are the ``top_feats`` largest |contribution| features with
    their signed contributions (largest first).
    """
    W_row = W[token_id]
    norms, x = center_normalize_rows(W_row[None, :], row_mean)
    code = encode_topk(x, encoder_w, encoder_b, k)[0]
    contributions = norms[0] * code * (h_state @ decoder.T)
    base = float(h_state @ row_mean)
    feat_sum = float(contributions.sum())
    original = float(h_state @ W_row)
    active = torch.nonzero(contributions != 0).flatten()
    top = active[contributions[active].abs().argsort(descending=True)[:top_feats]]
    return {
        "original_logit": original,
        "base": base,
        "feature_sum": feat_sum,
        "residual": original - base - feat_sum,
        "top_features": [{"id": int(f), "contribution": float(contributions[f])} for f in top],
    }


@torch.no_grad()
def feature_top_tokens(
    W: torch.Tensor,
    row_mean: torch.Tensor,
    feature_ids: Iterable[int],
    encoder_w: torch.Tensor,
    encoder_b: torch.Tensor,
    tokenizer: Any,
    *,
    top_tokens: int = 12,
    chunk: int = 8192,
) -> dict[int, list[str]]:
    """Top unembedding rows (by encoder activation) per feature id, as raw decoded strings.

    Labels are ``tokenizer.decode([id])`` verbatim, in activation order, rows
    with zero activation dropped, no display cleaning and no de-duplication.
    That is exactly what the paper's dumps carry under ``feature_top_tokens``;
    the display-cleaned ``research.qwen_readout.display_label_features`` would
    change those lists, so it is deliberately not used here.
    """
    feature_ids = sorted(set(int(f) for f in feature_ids))
    if not feature_ids:
        return {}
    device = W.device
    fids = torch.tensor(feature_ids, dtype=torch.long)
    enc = encoder_w[fids].to(device)
    bias = encoder_b[fids].to(device)
    best_scores = torch.full((len(fids), top_tokens), -float("inf"), device=device)
    best_ids = torch.zeros((len(fids), top_tokens), dtype=torch.long, device=device)
    for start in range(0, W.shape[0], chunk):
        rows = W[start : start + chunk].float()
        _norms, x = center_normalize_rows(rows, row_mean)
        scores = F.relu(x @ enc.T + bias).T
        merged = torch.cat([best_scores, scores], dim=1)
        ids = torch.arange(start, start + rows.shape[0], device=device).expand(len(fids), -1)
        merged_ids = torch.cat([best_ids, ids], dim=1)
        best_scores, keep = torch.topk(merged, k=top_tokens, dim=1)
        best_ids = torch.gather(merged_ids, 1, keep)
    return {
        fid: [tokenizer.decode([t]) for t, s in zip(best_ids[i].tolist(), best_scores[i].tolist()) if s > 0]
        for i, fid in enumerate(feature_ids)
    }


# --------------------------------------------------------------------------- #
# offline: dumps, votes, aggregation
# --------------------------------------------------------------------------- #


def parse_dump_args(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``LABEL=path`` arguments, preserving order."""
    out = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise ValueError(f"--dump expects LABEL=path, got {spec!r}")
        out.append((label, Path(path)))
    return out


def load_dump_records(path: str | Path) -> dict[str, dict]:
    """``{prompt id: record}`` of a readout dump from ``run_cross_lens_readouts.py``."""
    return {r["id"]: r for r in json.loads(Path(path).read_text())["records"]}


def load_bank_items(path: str | Path) -> dict[str, dict]:
    """``{prompt id: item}`` of a cross-lens prompt bank; duplicate ids are an error."""
    items = json.loads(Path(path).read_text())["prompts"]
    ids = [v["id"] for v in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate prompt ids in {path}: {dupes}")
    return {v["id"]: v for v in items}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def top5_at(rec: dict, layer: str) -> list | None:
    return rec["layers"].get(layer, {}).get(POS, {}).get("top5")


def dom_feat(rec: dict, layer: str, target: str) -> int | None:
    """Dominant feature id (largest |contribution|) of ``target``'s decomposition at ``layer``."""
    t = rec["layers"].get(layer, {}).get(POS, {}).get("targets", {}).get(target)
    if not t or not t.get("top_features"):
        return None
    return t["top_features"][0]["id"]


def _majority(count: int, total: int, *, rule: str) -> bool:
    if rule == "half":
        return count >= (total + 1) // 2
    if rule == "strict":
        return 2 * count > total
    raise ValueError(f"unknown agreement rule {rule!r} (expected one of {AGREEMENT_RULES})")


def majority_same(rec_a: dict, rec_b: dict, target_a: str, target_b: str, *, rule: str) -> bool | None:
    """Mid-band vote: is dom(target_a in rec_a) the same feature as dom(target_b in rec_b)?

    Layers where either decomposition is missing are skipped; ``None`` when none
    remain. ``rule="half"`` passes on at least ``ceil(n/2)`` agreeing layers (the
    paper: 2 of 4), ``rule="strict"`` needs more than half (3 of 4).
    """
    same = total = 0
    for L in MIDBAND:
        fa, fb = dom_feat(rec_a, L, target_a), dom_feat(rec_b, L, target_b)
        if fa is None or fb is None:
            continue
        total += 1
        same += int(fa == fb)
    if total == 0:
        return None
    return _majority(same, total, rule=rule)


def token_diverges(rec_a: dict, rec_b: dict, *, rule: str) -> bool | None:
    """Mid-band vote: do the two lenses' top-1 token strings differ? Same ``rule`` as :func:`majority_same`."""
    diff = tot = 0
    for L in MIDBAND:
        ta, tb = top5_at(rec_a, L), top5_at(rec_b, L)
        if ta and tb:
            tot += 1
            diff += int(ta[0][0] != tb[0][0])
    if tot == 0:
        return None
    return _majority(diff, tot, rule=rule)


@dataclass(frozen=True)
class PairSpec:
    """What distinguishes one language pair's aggregation from the other's.

    ``tag_b`` suffixes the slot-B output keys (``cross_form_zh`` / ``_de``);
    ``surface_call`` labels a lens's top-1 token given the bank item
    (script for EN-ZH, lexical for EN-DE) into column ``top1_<surface_column>_<tag>``;
    ``split_labels`` is the (slot-A, slot-B) label pair counted as a lens-only
    split; ``divergence`` adds the EN-DE top-1 string divergence rate.
    """

    tag_b: str
    cross_groups: tuple[str, ...]
    surface_column: str
    surface_call: Callable[[str, dict], str]
    split_labels: tuple[str, str]
    split_caption: str
    surface_caption: str
    divergence: bool = False


def aggregate_pair(
    records_a: dict[str, dict],
    records_b: dict[str, dict],
    bank: dict[str, dict],
    spec: PairSpec,
    *,
    seed: int,
    rule: str = "half",
    null_population: str = "all",
) -> dict:
    """Per-prompt rows, per-family and pooled agreement, null floors and the surface split.

    ``rule`` is the majority rule of every vote (headline, within-lens
    cross-form, both nulls, divergence). ``null_population`` selects the rows
    pooled into the unrelated-token floor: ``"all"`` (every bank item, the
    paper's EN-DE 204 comparisons) or ``"cross"`` (control families excluded,
    which is the EN-ZH population where controls carry no nulls). Key order of
    the returned dict and of each row is the dump format; do not reorder.
    """
    if null_population not in NULL_POPULATIONS:
        raise ValueError(f"unknown null population {null_population!r} (expected one of {NULL_POPULATIONS})")
    col = f"top1_{spec.surface_column}"
    ids = sorted(set(records_a) & set(records_b) & set(bank))

    rows = []
    for rid in ids:
        v, a, b = bank[rid], records_a[rid], records_b[rid]
        fa, fb = v["form_a"], v["form_b"]
        row = {"id": rid, "group": v["group"], "concept": v.get("concept", "")}
        # headline: same dominant feature for the SAME concept token, lens A vs lens B
        row["cross_lens_pass"] = majority_same(a, b, fa, fa, rule=rule)
        # one feature carries both surface forms, within each lens
        row["cross_form_en"] = majority_same(a, a, fa, fb, rule=rule)
        row[f"cross_form_{spec.tag_b}"] = majority_same(b, b, fa, fb, rule=rule)
        # null floor (a): form_a vs unrelated token, within lens A
        nulls = [majority_same(a, a, fa, nt, rule=rule) for nt in v.get("null_targets", [])]
        nulls = [x for x in nulls if x is not None]
        row["null_hits"] = sum(nulls)
        row["null_total"] = len(nulls)
        # surface label of top-1 under each lens: plurality over the mid-band;
        # a 2-2 tie resolves to the alphabetically first label, so the vote is
        # deterministic (CJK < LATIN < OTHER; DE < EN < OTHER).
        for tag, rec in (("en", a), (spec.tag_b, b)):
            labels = []
            for L in MIDBAND:
                top5 = top5_at(rec, L)
                if top5:
                    labels.append(spec.surface_call(top5[0][0], v))
            row[f"{col}_{tag}"] = max(sorted(set(labels)), key=labels.count) if labels else "NA"
        if spec.divergence:
            row["token_diverges"] = token_diverges(a, b, rule=rule)
        rows.append(row)

    # null floor (b): shuffled pairing, form_a of prompt i under lens A vs form_a of j under lens B
    rng = random.Random(seed)
    cross_ids = [r["id"] for r in rows if r["group"] in spec.cross_groups]
    shuffle_hits = shuffle_total = 0
    for rid in cross_ids:
        others = [x for x in cross_ids if bank[x]["concept"] != bank[rid]["concept"]]
        for oid in rng.sample(others, min(3, len(others))):
            res = majority_same(records_a[rid], records_b[oid], bank[rid]["form_a"], bank[oid]["form_a"], rule=rule)
            if res is not None:
                shuffle_total += 1
                shuffle_hits += int(res)

    def rate(sel):
        vals = [r for r in rows if sel(r) and r["cross_lens_pass"] is not None]
        k = sum(r["cross_lens_pass"] for r in vals)
        return k, len(vals), wilson(k, len(vals))

    groups = sorted({r["group"] for r in rows})
    summary: dict = {"per_group": {}, "rows": rows}
    for grp in groups:
        k, n, (pt, lo, hi) = rate(lambda r, g=grp: r["group"] == g)
        summary["per_group"][grp] = {"pass": k, "n": n, "rate": pt, "ci": [lo, hi]}
    k, n, (pt, lo, hi) = rate(lambda r: r["group"] in spec.cross_groups)
    summary["headline"] = {"pass": k, "n": n, "rate": pt, "ci": [lo, hi]}
    for tag in ("en", spec.tag_b):
        key = f"cross_form_{tag}"
        cf = [r[key] for r in rows if r["group"] in spec.cross_groups and r[key] is not None]
        summary[key] = {"pass": sum(cf), "n": len(cf)}
    null_rows = rows if null_population == "all" else [r for r in rows if r["group"] in spec.cross_groups]
    nk = sum(r["null_hits"] for r in null_rows)
    nn = sum(r["null_total"] for r in null_rows)
    summary["null_within_lens"] = {"pass": nk, "n": nn, "rate": wilson(nk, nn)[0]}
    summary["null_shuffle_cross_lens"] = {
        "pass": shuffle_hits,
        "n": shuffle_total,
        "rate": wilson(shuffle_hits, shuffle_total)[0],
    }
    if spec.divergence:
        div = [r["token_diverges"] for r in rows if r["group"] in spec.cross_groups and r["token_diverges"] is not None]
        summary["divergence_rate"] = {"diverging": sum(div), "n": len(div)}
    la, lb = spec.split_labels

    def flips(sel):
        return sum(1 for r in sel if r[f"{col}_en"] == la and r[f"{col}_{spec.tag_b}"] == lb)

    summary["lens_only_split"] = {}
    for grp in groups:
        sel = [r for r in rows if r["group"] == grp]
        summary["lens_only_split"][grp] = {"pass": flips(sel), "n": len(sel)}
    cross_sel = [r for r in rows if r["group"] in spec.cross_groups]
    summary["lens_only_split"]["all_cross"] = {"pass": flips(cross_sel), "n": len(cross_sel)}
    return summary


def print_summary(summary: dict, spec: PairSpec) -> None:
    """The aggregators' stdout report, from the dict :func:`aggregate_pair` returns."""
    print("\n=== CROSS-LENS DOMINANT-FEATURE AGREEMENT (majority of mid-band) ===")
    for grp, g in summary["per_group"].items():
        print(f"  {grp:14s} {g['pass']:3d}/{g['n']:<3d}  {g['rate']:.2f}  [{g['ci'][0]:.2f}, {g['ci'][1]:.2f}]")
    h = summary["headline"]
    print(f"  {'ALL CROSS':14s} {h['pass']:3d}/{h['n']:<3d}  {h['rate']:.2f}  [{h['ci'][0]:.2f}, {h['ci'][1]:.2f}]")
    cf_en, cf_b = summary["cross_form_en"], summary[f"cross_form_{spec.tag_b}"]
    print("\n=== ONE FEATURE CARRIES BOTH FORMS (within-lens) ===")
    print(f"  EN lens: {cf_en['pass']}/{cf_en['n']}   {spec.tag_b.upper()} lens: {cf_b['pass']}/{cf_b['n']}")
    nw, ns = summary["null_within_lens"], summary["null_shuffle_cross_lens"]
    print("\n=== NULL FLOORS ===")
    print(f"  (a) form vs unrelated token, within-lens: {nw['pass']}/{nw['n']} ({nw['rate']:.2f})")
    print(f"  (b) cross-lens shuffled prompts:          {ns['pass']}/{ns['n']} ({ns['rate']:.2f})")
    if spec.divergence:
        d = summary["divergence_rate"]
        print("\n=== TOKEN DIVERGENCE (descriptive) ===")
        print(f"  top-1 differs between lenses on {d['diverging']}/{d['n']} cross prompts")
    print(f"\n=== LANGUAGE-FOLLOWS-LENS ({spec.surface_caption}, mid-band vote) ===")
    for grp, g in summary["lens_only_split"].items():
        name = "ALL CROSS" if grp == "all_cross" else grp
        print(f"  {name:14s} {spec.split_caption} on {g['pass']}/{g['n']}")


def write_summary(summary: dict, out: str | Path) -> None:
    """The aggregators' summary JSON (``ensure_ascii=False, indent=1``; not ``utils.write_json``, which sorts keys)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


def cli_args(args: Any) -> dict[str, Any]:
    """``vars(args)`` without the subcommand callables (``set_defaults(fn=...)``), ready for ``run_provenance``."""
    items = args.items() if isinstance(args, dict) else vars(args).items()
    return {k: v for k, v in items if not callable(v)}


def write_manifest(anchor: str | Path, args: Any, **fields: Any) -> Path:
    """``<anchor stem>.manifest.json`` beside an output file: ``fields`` plus ``run_provenance(args)``.

    For scripts whose primary outputs are CSVs (or a raw dump whose schema must
    not grow); JSON summaries carry ``provenance`` inline instead.
    """
    path = Path(anchor).with_suffix(".manifest.json")
    write_json({**fields, "provenance": run_provenance(cli_args(args))}, path, atomic=True)
    return path


# --------------------------------------------------------------------------- #
# resume sidecars (lens fitters)
# --------------------------------------------------------------------------- #


def prompt_list_sha1(prompts: Sequence[str]) -> str:
    """Content hash of a prompt list (JSON-encoded, so newlines inside prompts cannot alias)."""
    return hashlib.sha1(json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")).hexdigest()


def check_resume_meta(meta_path: Path, expected: dict[str, Any], artifacts: Sequence[Path]) -> None:
    """Refuse to resume from ``artifacts`` unless ``meta_path`` records ``expected``.

    Nothing present: returns (fresh run). Artifacts present without the sidecar:
    raises (files from before the sidecar existed; delete them to refit).
    Sidecar present with any key of ``expected`` different: raises, naming each
    mismatching key with the on-disk and requested values.
    """
    present = [p for p in artifacts if p.exists()]
    if not present:
        return
    names = ", ".join(str(p) for p in present)
    if not meta_path.exists():
        raise RuntimeError(
            f"{names} exist(s) but the sidecar {meta_path} does not, so the fit they belong to cannot be "
            "verified; delete them (or restore the sidecar) before resuming"
        )
    stored = json.loads(meta_path.read_text())
    diffs = [f"{k}: on disk {stored.get(k)!r}, requested {v!r}" for k, v in expected.items() if stored.get(k) != v]
    if diffs:
        raise RuntimeError(
            f"refusing to resume from {names}: {meta_path} was written for a different fit\n  " + "\n  ".join(diffs)
        )


def write_resume_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    write_json(meta, meta_path, atomic=True)


# --------------------------------------------------------------------------- #
# text folding (bank builder, EN-DE lexical call)
# --------------------------------------------------------------------------- #


def fold_text(s: str) -> str:
    """Strip, casefold, ``ß -> ss``, then drop combining marks after NFKD (``Käse -> kase``)."""
    s = s.strip().casefold().replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))
