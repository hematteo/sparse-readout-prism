# Reproducing the paper

This doc maps each paper claim to the script and config that produced it; the
experiment hierarchy and claim boundaries are laid out in the paper itself.

**This repo is metrics-only.** Every script computes and *persists* its metrics
(CSV / JSON / `.pt`), and those committed metric artifacts are the reproducible
deliverables. The repo ships **no** figure-rendering code (zero matplotlib
imports): the scripts under `scripts/figures/` compute and persist the metrics
*behind* each figure, they do not render figures. The paper's figures are
rendered separately in the paper's LaTeX source from these metrics.

## 0. Prerequisites

```bash
uv sync                                  # install Python deps
```

Hardware: the headline results were produced on a single A40 (an HPC
cluster). A single 24 GB GPU is enough for inference and evaluation; SAE
training for the larger models (Qwen-9B) wants ~40 GB.

External data:

- HuggingFace models are downloaded on first use to `$HF_HOME` (default
  `~/.cache/huggingface`). Pin commit hashes via the model configs in
  `configs/models/`.
- The C4 model-native slice used for headline numbers is built by
  `scripts/data/build_query_banks.py --native-source c4` (its provenance /
  revision is pinned in
  `data/query_banks/qwen_gemma_result1_model_native_prompts_c4_manifest.json`).

Pretrained readout-feature dictionaries (the *selected* SAEs cited in the
paper) are published on the Hugging Face Hub at
[`matteohe/sparse-readout-prism`](https://huggingface.co/matteohe/sparse-readout-prism)
(`<model>/<operating_point>/checkpoint.pt`).
`configs/registries/exp2_selected_sae_checkpoints.yaml`
records the canonical manifest (each checkpoint's model, width, `k`, metrics,
and Hub path). Training one from scratch costs ~12 h of GPU time per
model; to exercise the full pipeline now without data or a GPU, run the
synthetic smoke config (see the README quickstart).

### Determinism

Runners seed Python / NumPy / Torch RNG (`sparse_readout_prism.utils.set_seed`),
so a fixed seed reproduces the same sampling and bootstrap draws. This is
seeded-RNG reproducibility, **not** bitwise determinism: on GPU, CUDA reductions
(topk / scatter / matmul) can differ at the last ULPs across hardware and
library versions. We deliberately do not enable
`torch.use_deterministic_algorithms(True)` (it would forbid some kernels the
pipeline relies on); enable it yourself if you need exact bitwise GPU repro.

### Results directory

Every script writes under a top-level `results/` directory. The repo does
not ship a `results/` symlink — point it where you want with a local
symlink if you keep checkpoints on an external drive:

```bash
ln -s /path/to/big-disk/results results
```

Path helpers in `sparse_readout_prism.paths` also honour `$SRP_SSD_ROOT`
for cluster runs (default a `local_snapshots/` dir under the repo).

## 1. End-to-end pipeline (one model, headline numbers)

The pipeline has three distinct evaluation layers. Row reconstruction chooses
whether the dictionary has enough capacity, readout replacement checks whether
the reconstructed LM head behaves like the original on held-out hidden states,
and query fidelity checks the exact scalar queries used by feature displays.
Only the last layer licenses local feature-level interpretation, and only when
the displayed query passes its residual/sign gate.

1. **Extract the readout** — produce `W_U_orig` and final-norm hidden states.
   The script writes a single `.pt` payload `{W_U_orig, h_LN}` to `--out`, plus a
   sibling `<stem>_manifest.json`:
   ```bash
   uv run python scripts/data/extract_model_readout.py \
       --model-id Qwen/Qwen3.5-2B --out results/qwen2b/hLN.pt
   ```
2. **Train the readout-feature SAE** — TopK factorizer on the
   centered+normalized W_U rows. `--data-path` maps to `data.path` and
   `--out-dir` maps to `run.output_dir` in the run config.
   `paper_phase2_base.yaml` is the complete headline recipe (TopK 32×/k256); its
   `factorizer.d_features: 65536` is `32 × d_model` for Qwen-3.5-2B (d_model 2048).
   For a different model, override it with `--set factorizer.d_features=<width×d_model>`.
   ```bash
   uv run python scripts/train/train_readout_sae_from_config.py \
       --config configs/sweeps/paper_phase2_base.yaml \
       --data-path results/qwen2b/hLN.pt --out-dir results/qwen2b/run0
   ```
3. **Task-fidelity evaluation** — paper's headline fidelity gate: margin
   sign agreement + query residuals. `--cell-dir` is the trained run dir (holds
   `checkpoint.pt` + `config.yaml`); `--data-path` is the extract `.pt` from step 1;
   `--bank` is a torch `.pt` margin/query bank loaded via `torch.load` (the
   `{h, margin_items, query_items}` schema documented in the script's module
   docstring). The `.jsonl` files under `data/query_banks/` are case *inputs*,
   not this bank — the `.pt` bank is built separately from those cases plus
   extracted hidden states.
   ```bash
   uv run python scripts/eval/task_fidelity_evaluation.py \
       --cell-dir results/qwen2b/run0 \
       --data-path results/qwen2b/hLN.pt \
       --bank results/qwen2b/task_fidelity_bank.pt
   ```
4. **Compute the metrics behind the paper figures** — see the table below.

## 2. Figure / table → script + config

The `scripts/figures/*.py` entries below compute and persist the metrics behind
each figure (CSV / JSON / `.pt`); they do not render figures. The figures
themselves are assembled in the paper's LaTeX source from these metrics.

When reading this table, keep the claim boundaries in mind: a figure that reports rowEV,
top1, or KL is a replacement/reconstruction diagnostic, while a figure that
interprets feature bars should also report the local query residual and sign
check.

| Paper artifact | Script | Config / inputs |
|---|---|---|
| Method schematics (`fig:srp-schematic` / `readout_query_three_panel`, score-accounting TikZ) | — (paper-side LaTeX/TikZ; no producing script in this repo) | — |
| §4 main fidelity table (Qwen 0.8B/2B/9B) | `scripts/eval/task_fidelity_evaluation.py` | `configs/registries/exp2_selected_sae_checkpoints.yaml` |
| Metrics behind: token decomposition examples | `scripts/figures/compute_sae_paper_examples.py` | hardcoded example sets (`--example-set paper`, the default; `EXAMPLE_SETS` in `research/data/qwen_example_prompts.py`), plus a required `--checkpoint` |
| §5 metrics behind: logit-lens vs Sparse Readout Prism comparison (`fig:srp-lens-prism-comparison`) | `scripts/figures/compute_lens_prism_comparison.py` | defaults reproduce the paper case (`--target-a " verify"`, `--target-b " assume"`, `--k 256`); `--checkpoint` points at the Qwen3.5-2B 32x/k256 selected SAE in `configs/registries/exp2_selected_sae_checkpoints.yaml`. Computes readout states live (reuses `collect_readout_states_batched` from `sparse_readout_prism.research._common.qwen_readout`; no precomputed cache) |
| Table / metrics behind: benchmark-derived task targets | `scripts/run/run_benchmark_derived_query_suite.py`, `scripts/figures/compute_benchmark_task_group_examples.py` | `--target-design task_group_v2 --out-dir results/benchmark_derived_query_suite_task_group_v2_qwen_20260523`; outputs under that dir (the suite does not derive the dir from the target design — pass `--out-dir` explicitly) |
| Metrics behind: family / availability / selection queries | `scripts/figures/compute_general_readout_queries.py` | `configs/registries/result1_query_fidelity_cluster.yaml` |
| Metrics behind: literature-prompt all-layer trace | `scripts/figures/compute_all_layer_literature_prompt.py` | — |
| Baseline comparison metrics (Sparse RP vs nulls) | `scripts/run/run_readout_baseline_comparisons.py` | outputs persisted as CSV/JSON for the paper-side baseline figure |
| Result-1 five-model query-fidelity tables / §4.5 distributional readout metrics | `scripts/run/run_query_fidelity_bank.py` | `configs/registries/result1_query_fidelity_cluster.yaml` + the curated/model-native banks under `data/query_banks/` |
| Appendix J: lexical-edit / readout-side control stress test (`tab:app-lexical-edit-primary-methods`, `tab:app-lexical-edit-cross-model`) | `scripts/run/run_qwen_profanity_suppression_eval.py` | `--checkpoint <qwen2b 32x/k256 checkpoint.pt>`; the primary-methods table reads `baseline_comparison.csv` / `candidate_constrained_summary.csv` (the eval emits these tables only — no paper figure). For the cross-model table, rerun once per model with the checkpoints in `configs/registries/result1_query_fidelity_cluster.yaml` (Qwen3.5-2B fidelity 32x/k256; R1-Distill-Qwen-7B fidelity 32x/k256; Ministral-3-8B-Base strict_budget 16x/k128) |
| Appendix K sweep / selection tables (`tab:app-k-model-finalists`, `tab:app-factorizer-selection-evidence`, `tab:app-model-suite-current`) | — (no figure ships; numbers hand-transcribed from the audited literals in [`data/appendix/appendix_k_sweep_tables.json`](../data/appendix/appendix_k_sweep_tables.json)) | `configs/sweeps/arch_frontier_160m.yaml`, `configs/sweeps/paper_phase2_frontier.yaml` |
| Appendix L: global readout-feature profiles (shell/java, port/cell) | `scripts/analyze/analyze_readout_feature_directions.py` | `--sae-id qwen2b_k256 --context-set interesting_domains --case-ids shell_beach,shell_terminal,java_coffee,java_programming --top-directions 20` (rerun with the `port_harbor,port_network,cell_prison,cell_biology` case-ids for `port_cell`); persists the per-direction metrics behind the global-readout bars |
| Appendix M: metrics behind feature-resolved DLA, vocab-mean verify contrast | `scripts/figures/compute_prism_dla.py` | `--contrast-mode vocab_mean --out-dir results/qwen2b_32x_prism_dla_verify_token_layer_readout` (output basenames are the fixed `qwen2b_32x_prism_dla_*` prefix; the dir distinguishes this vocab-mean run from the default `_verify_assume`) |
| Appendix L: metrics behind additional bark/bass margin case studies | `scripts/figures/compute_sae_paper_examples.py` | `--example-set section43_margin_appendix2 --out-dir results/qwen2b_32x_margin_examples_appendix2 --k 256` (32x/k256 checkpoint; output basenames are the fixed `qwen2b_selected_paper_examples_*`, so a distinct `--out-dir` keeps them from overwriting the default `paper` set) |
| Appendix L: ring fixed-token context delta metrics | `scripts/data/mine_polysemy_features.py` | `--force-pair-ids ring_jewelry_phone`; persists the polysemy delta metrics behind the ring context-delta figure |
| Appendix K: W_U row-norm tail stats (`tab:app-k-row-norm-tail`) | `scripts/analyze/compute_row_norm_tail_stats.py` | `--registry configs/registries/result1_query_fidelity_cluster.yaml --models Qwen3.5-9B,Ministral-3-8B-Base,R1-Distill-Qwen-7B,R1-Distill-Llama-8B` (or `--artifact "label=<extract>.pt"`); centred row norms, no GPU/checkpoint |
| Appendix K: dense / negative-control diagnostic (`omp_diagnostic` in [`data/appendix/appendix_k_sweep_tables.json`](../data/appendix/appendix_k_sweep_tables.json)) | `scripts/eval/dense_control_diagnostic.py` | `--w-u <extract>.pt --checkpoint <ckpt.pt> --setting "Qwen-9B 32x, k=256" --n-rows 20000 --bootstrap 1000` (encoder vs LS/NNLS-on-support vs signed/nonneg OMP vs dense rank-k, scored by rowEV) |
| Appendix L: feature-label audit (`tab:app-main-case-study-feature-audit` data; `tab:app-qualitative-feature-label-audit` counts) | `scripts/analyze/audit_feature_labels.py` | `substrate --feature-ids 36,4095,… --model-id Qwen/Qwen3.5-2B` emits per-feature top rows; `aggregate --annotations data/audit/feature_label_audit_annotations.csv --validate-against data/audit/feature_label_audit.json` tallies the counts (classification is human; see note below) |

### Notes and paper-only artifacts

Like every results table in the paper, the fidelity / selection / baseline /
lexical-edit tables are transcribed by hand from the CSV outputs of the scripts
above. A few artifacts have no producing script and are authored in the paper's
LaTeX source:

- **Schematics** — `fig:srp-schematic` (`readout_query_three_panel.pdf`) and the
  score-accounting TikZ schematic are hand-drawn LaTeX/TikZ; there is no repo
  deliverable.
- **Feature-label audit counts** (`tab:app-qualitative-feature-label-audit`) — the
  per-feature top associated rows are emitted by
  `scripts/analyze/audit_feature_labels.py substrate`, but the
  coherent / ambiguous / token-form classification is a human judgment. The
  count table is reproduced by `… aggregate` from a checked-in annotations CSV
  (`data/audit/feature_label_audit_annotations.csv`, currently covering the main
  `bug` panels); the per-set totals the paper reports are recorded in
  `data/audit/feature_label_audit.json`. Annotating the remaining figure-sets needs
  the displayed feature ids from the figure scripts plus the same human pass.
- **`fig:app-qwen-prism-dla-verify-token`** — `compute_prism_dla.py` persists the
  per-component / per-feature DLA metrics (`qwen2b_32x_prism_dla_*` CSVs plus a
  `manifest.json`) under the `--out-dir` directory; the heatmap and stacked
  component-bar layout shown in the paper are composed in the paper's LaTeX
  source from those metrics.
- The metrics behind the two Appendix-L global-readout figures
  (`shell_java`, `port_cell`) are persisted by
  `scripts/analyze/analyze_readout_feature_directions.py`, not by anything under
  `scripts/figures/` (see the table row above).

## 3. Tests

```bash
uv run pytest -q
```

The test suite is deliberately small: identity-correctness of the
decomposition, schema invariants of the task-fidelity evaluator, and layout
guardrails for the `research` package. CI-safe — no GPU, no model loads.

## 4. Notes on removed exploratory code

The earlier exploratory families have been removed because they were not on any
paper-facing import path: causal / attention-head circuits, CoT-faithfulness and
agentic-injection probes, the 160M transcoder/SAE lineage, the classical / tuned
proto-token-lens renderers, and the per-snapshot proto-token-lens PoC
(`proto_token_lens_poc.py`) together with the readout / logit-trajectory display
renderers (`render_readout.py`, `render_logit_trajectory.py`) and their
readout-token / row-encoder helpers. The pre-formulation feature-mining module
`mine_interesting_features` was likewise removed; its one paper-facing helper,
`collect_readout_states_batched` (batched last-position readout states), now lives
in `sparse_readout_prism.research._common.qwen_readout` next to its single-prompt
twin `collect_readout_state`.
