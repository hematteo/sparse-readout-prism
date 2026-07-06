# Sweep configs

Two kinds of file live here:

1. **Run configs** — complete, self-contained run specifications (top-level
   `run` / `data` / `factorizer` / `training` / `evaluation` blocks, optional
   `autoresearch.score_thresholds` read by `evaluate.py`). These are fed
   directly to
   [`scripts/train/train_readout_sae_from_config.py`](../../scripts/train/train_readout_sae_from_config.py)
   via `--config` and run one cell end-to-end.

2. **Grid templates** — sweep *specifications* with top-level `output_root` /
   `fixed` / `grid` blocks (and `k_l0_factor` / `target_l0` knobs). These are
   **not** valid run configs: passing one to `--config` is rejected with a clear
   error (the trainer guards on top-level `grid`/`output_root`/`fixed` and on
   missing `factorizer` / `training` blocks), rather than the old failure mode
   where it silently trained an all-defaults cell on synthetic-fallback data.
   They are meant to be expanded into per-cell run configs by a sweep-expander,
   which is **not shipped in this repo**; reproduce the Appendix-K grids by
   writing one complete run config per cell from the `fixed` + `grid` values.

   **Expansion mapping** (the parts a naive copy misses):
   - TopK-family cells: `factorizer.k = round(k_l0_factor × target_l0)` and
     `evaluation.k` set to the same value.
   - Controller-driven cells (`jumprelu`, `l1_relu`): the grids ship the
     `l0_controller` gains but **not** `enabled` — a per-cell run config must
     set `training.l0_controller.enabled: true` *and* move the cell's
     `target_l0` to `training.l0_controller.target_l0` (`train.py` reads it
     from inside the controller block), or the controller silently stays off
     and the cell trains at an uncontrolled L0.

| Config | Kind | Role |
|---|---|---|
| `paper_phase2_base.yaml` | run config | Complete headline recipe (TopK 32×/k256, Qwen-3.5-2B finalist); override `factorizer.d_features` per model. |
| `paper_phase2_frontier.yaml` | grid template | `k` / width frontier sweep (Appendix K panels). Expand before running. |
| `arch_frontier_160m.yaml` | grid template | Fine-`k` architecture frontier (Pythia-160m; Appendix K). Expand before running. |
| `deepseek_r1_distill_qwen_7b_paper.yaml`, `…_16x_k128.yaml` | run config | R1-Distill-Qwen-7B finalists (32×/k256 and strict-budget 16×/k128). |
| `deepseek_r1_distill_llama_8b_paper.yaml`, `…_16x_k128.yaml` | run config | R1-Distill-Llama-8B finalists. |
| `ministral3_8b_paper.yaml`, `…_16x_k128.yaml` | run config | Ministral-3-8B-Base finalists (32×/k256 and strict-budget 16×/k128). |

The provisional Gemma operating points in the registries were trained with the
same converge recipe as the Qwen models (`paper_phase2_base.yaml` protocol);
no dedicated Gemma run config is shipped — reproduce a Gemma cell by overriding
`factorizer.d_features` (width × d_model from the extraction manifest),
`factorizer.k`, `evaluation.k`, and `data.path` on `paper_phase2_base.yaml`
(see `docs/DATA.md` §4).

The run configs set `data.fallback: error` and a cluster `data.path`
(`${SRP_ARCHIVE_ROOT}/...`); without the dataset synced they exit with a clean
`FileNotFoundError` naming the path. (Grid templates are rejected earlier still,
by the run-config guard, before any data is loaded.)

The cross-model finalist configs (R1-Distill-Qwen-7B, R1-Distill-Llama-8B,
Ministral) are launched with the generic trainer —
`train_readout_sae_from_config.py --config <file>` — and have no dedicated
launch wrapper. Their trained checkpoints are the ones enumerated in
[`../registries/result1_query_fidelity_cluster.yaml`](../registries/result1_query_fidelity_cluster.yaml)
(the Appendix-K finalists for the Qwen / Gemma / Pythia models live in
[`../registries/exp2_selected_sae_checkpoints.yaml`](../registries/exp2_selected_sae_checkpoints.yaml)).
