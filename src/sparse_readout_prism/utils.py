from __future__ import annotations

import csv
import json
import logging
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(data: dict[str, Any], path: str | Path, *, atomic: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    with target.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")
    if atomic:
        target.replace(path)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSON Lines, atomically (temp file + replace)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text to path atomically (temp file + replace)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str | None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def format_metric_table(metrics: dict[str, Any]) -> str:
    names = [
        "selection_score",
        "row_centered_ev",
        "row_centered_cosine",
        "top1_logit_residual_frac_mean",
        "top1_logit_residual_abs_mean",
        "top8_positive_contrib_coverage_mean",
        "top8_abs_contrib_coverage_mean",
        "val_logit_kl_bits_mean",
        "val_top1_match",
        "val_top5_overlap",
        "dead_feature_rate",
        "feature_usage_entropy",
    ]
    lines = ["| metric | value |", "| --- | ---: |"]
    for name in names:
        if name in metrics and metrics[name] is not None:
            value = metrics[name]
            if isinstance(value, float):
                lines.append(f"| `{name}` | {value:.6g} |")
            else:
                lines.append(f"| `{name}` | {value} |")
    return "\n".join(lines)


def pearson(x: Any, y: Any) -> float:
    """Pearson r over array-likes; nan if <3 points or either input is ~constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Any, y: Any) -> float:
    """Spearman rank correlation (rank-transform then Pearson); nan if <3 points."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    return pearson(rx, ry)


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    mkdir_parents: bool = True,
    fieldnames: list[str] | None = None,
    write_empty: bool = False,
    atomic: bool = False,
) -> None:
    """Write a list of dicts as CSV.

    By default, collects the union of keys across all rows (preserving first-seen
    order), so ragged dicts don't drop columns. Pass an explicit ``fieldnames``
    to fix the column order.

    ``rows`` empty: no-op, unless ``write_empty=True`` (writes a 0-byte file, for
    callers whose downstream readers expect the path to exist). ``atomic=True``
    writes to a sibling ``.tmp`` and replaces it into place in one step.
    """
    path = Path(path)
    if not rows and not write_empty:
        return
    if mkdir_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if atomic:
        target.replace(path)


_LM_HEAD_PATHS = (
    "lm_head",
    "model.lm_head",
    "language_model.lm_head",
    "model.language_model.lm_head",
    "text_model.lm_head",
)


def find_lm_head_with_path(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """Locate the text unembedding (lm_head) and the attribute path it was found at.

    Single source of truth for the lm_head search (plain and multimodal LMs);
    ``find_lm_head`` wraps this when only the module is needed.
    """
    for path in _LM_HEAD_PATHS:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module) and hasattr(obj, "weight"):
            logger.info("lm_head found at %s, weight=%s", path, tuple(obj.weight.shape))
            return obj, path
    raise RuntimeError(f"could not locate lm_head on {type(model).__name__}")


def find_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the text unembedding (lm_head). Works for plain and multimodal LMs."""
    return find_lm_head_with_path(model)[0]


def load_causal_lm(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | str = "auto",
    device_map: str | None = "auto",
    low_cpu_mem_usage: bool = True,
    local_files_only: bool = False,
    prefer_multimodal: bool = False,
) -> tuple[torch.nn.Module, Any]:
    """Load a HuggingFace causal LM via the most-specific auto-class that works.

    The single model loader for the repo. Tries ``AutoModelForCausalLM`` first,
    falls back to ``AutoModelForImageTextToText`` (multimodal models that ship a
    usable ``lm_head``); ``prefer_multimodal=True`` flips that order (Qwen3.5's
    multimodal architecture only loads via the ImageTextToText class).
    ``device_map=None`` loads on CPU so the caller controls placement with
    ``.to(device)``. Returns ``(model.eval(), tokenizer)`` with parameters frozen.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
    auto_classes: list[Any] = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText

        if prefer_multimodal:
            auto_classes.insert(0, AutoModelForImageTextToText)
        else:
            auto_classes.append(AutoModelForImageTextToText)
    except Exception:
        pass

    last_err: Exception | None = None
    for ac in auto_classes:
        try:
            model = ac.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                local_files_only=local_files_only,
            )
            logger.info("loaded %s via %s dtype=%s", model_id, ac.__name__, dtype)
            for param in model.parameters():
                param.requires_grad_(False)
            return model.eval(), tok
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("%s failed for %s: %s", ac.__name__, model_id, e)
    raise RuntimeError(f"could not load {model_id}: {last_err}")


def git_commit() -> str | None:
    """Short git hash of the source checkout, or None (wheel install, no git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return None
    return out.stdout.strip() if out.returncode == 0 else None
