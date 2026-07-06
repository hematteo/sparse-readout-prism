from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sparse_readout_prism.data import load_prism_dataset
from sparse_readout_prism.evaluate import (
    evaluate_model,
    label_features,
)
from sparse_readout_prism.factorizers import build_factorizer
from sparse_readout_prism.train import train_factorizer
from sparse_readout_prism.utils import (
    ensure_dir,
    format_metric_table,
    git_commit,
    read_json,
    resolve_device,
    set_seed,
    write_json,
    write_yaml,
)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run", {})
    seed = int(run_cfg.get("seed", 0))
    # init_seed controls model init RNG; data_seed (in data.data_seed) controls
    # row/hidden sampling. Default both to legacy seed if unspecified.
    init_seed = int(run_cfg.get("init_seed", seed))
    set_seed(init_seed)
    output_dir = ensure_dir(run_cfg.get("output_dir", "results/run"))
    device = resolve_device(run_cfg.get("device", "auto"))

    metrics_path = output_dir / "metrics.json"
    done_marker = output_dir / "DONE"
    if done_marker.exists() and metrics_path.exists():
        cached = read_json(metrics_path)
        if cached.get("finite_metrics"):
            score = cached.get("selection_score")
            score_str = f"{score:.6g}" if isinstance(score, (int, float)) else str(score)
            print(f"[skip] {output_dir} already complete (selection_score={score_str})", flush=True)
            return cached

    dataset = load_prism_dataset(config, seed=seed)
    model = build_factorizer(config, d_model=dataset.d_model).to(device)
    history = train_factorizer(model, dataset, config, device, output_dir=output_dir)
    metrics, usage = evaluate_model(model, dataset, config, device, return_usage=True)
    metrics["train_seconds"] = history["train_seconds"]
    metrics["mean_l0_train"] = history.get("mean_l0_train")
    metrics["l0_controller_enabled"] = history.get("l0_controller_enabled", False)
    metrics["l0_controller_final_coeff"] = history.get("l0_controller_final_coeff")
    metrics["l0_controller_target"] = history.get("l0_controller_target")
    metrics["data_source"] = dataset.source_path
    metrics["used_fallback_data"] = dataset.used_fallback
    metrics["val_overlaps_train"] = dataset.val_overlaps_train
    metrics["device"] = str(device)
    # Read architecture/d_features off the model, not the config: build_factorizer
    # applies defaults, so a config that omits them would be misreported here.
    metrics["architecture"] = model.architecture
    metrics["d_features"] = int(model.d_features)
    metrics["init_seed"] = init_seed
    metrics["data_seed"] = int(config.get("data", {}).get("data_seed", seed))
    metrics["git_commit"] = git_commit()

    steps_run = int(history.get("steps_run", 0))
    bs = int(history.get("batch_size", 0))
    df = int(metrics["d_features"])
    dm = int(dataset.d_model)
    flops_per_step = 6 * bs * dm * df  # fwd+bwd, encoder+decoder ≈ 6 BDF
    try:
        from importlib.metadata import version

        pkg_version = version("sparse-readout-prism")
    except Exception:  # noqa: BLE001
        pkg_version = None
    compute_meta = {
        "architecture": metrics["architecture"],
        "d_features": df,
        "d_model": dm,
        "steps_run": steps_run,
        "batch_size": bs,
        "train_seconds": float(history["train_seconds"]),
        "seconds_per_step": (float(history["train_seconds"] / steps_run) if steps_run > 0 else None),
        "flops_per_step_est": int(flops_per_step),
        "total_train_flops_est": int(flops_per_step) * steps_run,
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "torch_version": torch.__version__,
        "package_version": pkg_version,
        "git_commit": metrics["git_commit"],
    }
    write_json(compute_meta, output_dir / "compute_meta.json", atomic=True)

    write_yaml(config, output_dir / "config.yaml")
    write_json(metrics, output_dir / "metrics.json", atomic=True)
    write_json({"history": history["losses"]}, output_dir / "history.json", atomic=True)

    labels = label_features(model, dataset, usage=usage, top_n=6, max_features=min(256, model.d_features))
    write_json({"features": labels}, output_dir / "feature_labels.json")

    artifacts = config.get("artifacts", {})
    if artifacts.get("save_checkpoint", True):
        ckpt_path = output_dir / "checkpoint.pt"
        ckpt_tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "factorizer": config.get("factorizer", {}),
                "evaluation": config.get("evaluation", {}),
                "row_mean": dataset.row_mean,
                "row_norms": dataset.row_norms,
                "row_token_ids": dataset.row_token_ids,
                "metrics": metrics,
            },
            ckpt_tmp,
        )
        ckpt_tmp.replace(ckpt_path)

    report_path = artifacts.get("report_path")
    if report_path:
        write_basic_report(
            report_path,
            title=f"Sparse Readout Prism: {config.get('run', {}).get('name', 'run')}",
            config=config,
            metrics_table=format_metric_table(metrics),
            notes=failure_notes(metrics),
        )

    train_ckpt = output_dir / "train_ckpt.pt"
    if train_ckpt.exists():
        train_ckpt.unlink()
    done_marker.write_text("done\n", encoding="utf-8")
    return metrics


def failure_notes(metrics: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if metrics.get("dead_feature_rate", 0.0) > 0.8:
        notes.append("Dead feature rate is high; increase steps, lower dictionary size, or improve sampling.")
    if metrics.get("top1_logit_residual_frac_mean", 0.0) > 1.0:
        notes.append("Selected-logit residual is larger than the original logit on average.")
    if metrics.get("val_top1_match", 0.0) < 0.25:
        notes.append("Reconstructed readout has weak top-1 agreement; do not rely on qualitative examples yet.")
    if not notes:
        notes.append("Audit high-usage features for global artifacts before making interpretability claims.")
    return notes


def write_basic_report(
    path: str | Path,
    title: str,
    config: dict[str, Any],
    metrics_table: str,
    notes: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Metrics", "", metrics_table, ""]
    lines.extend(
        [
            "## Config Snapshot",
            "",
            f"- architecture: `{config.get('factorizer', {}).get('architecture')}`",
            f"- d_features: `{config.get('factorizer', {}).get('d_features')}`",
            f"- k: `{config.get('evaluation', {}).get('k', config.get('factorizer', {}).get('k'))}`",
            f"- prism_top_n: `{config.get('training', {}).get('prism_top_n')}`",
            f"- lambda_prism: `{config.get('training', {}).get('lambda_prism')}`",
            f"- row_sampling: `{config.get('training', {}).get('row_sampling')}`",
            "",
            "## Failure-Mode Notes",
            "",
        ]
    )
    for note in notes or ["Review high-usage features and token-frequency artifacts before claiming concepts."]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
