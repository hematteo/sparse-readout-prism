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
uv sync --extra lens                     # + jlens (Jacobian lens), only for the cross-lens study
```

Hardware: the headline results were produced on a single A40 (an HPC
cluster). A single 24 GB GPU is enough for inference and evaluation; SAE
training for the larger models (Qwen-9B) wants ~40 GB.

External data:

- HuggingFace models are downloaded on first use to `$HF_HOME` (default
  `~/.cache/huggingface`). The paper extractions did **not** pin HF weight
  revisions (they used the latest revision at paper time; the model configs in
  `configs/models/` record `revision: null`). To pin a rerun, pass
  `--revision` to `scripts/data/extract_model_readout.py` — the revision is
  recorded in the extraction manifest.
- The C4 model-native slice used for headline numbers is built by
  `scripts/data/build_query_banks.py --native-source c4` (its provenance /
  revision is pinned in
  `data/query_banks/qwen_gemma_result1_model_native_prompts_c4_manifest.json`).
- The direct-geometry baseline runs (Appendix "Direct-Geometry Alternatives")
  read that same C4 slice as their `model_native` bank; regenerate it with
  `scripts/data/build_query_banks.py --native-source c4 --seed 0 --max-native 10000`
  and pass `--model-native-file qwen_gemma_result1_model_native_prompts_c4.jsonl`.
- The cross-lens study fits its lenses on seeded C4 prompts (`allenai/c4`,
  configs `en` / `zh` / `de`, streaming, `shuffle(seed=0)`) dumped by
  `scripts/run/fit_jlens.py prompts`; the dumps are raw C4 text and are not
  shipped. The lenses need the `lens` extra (`uv sync --extra lens`).
- CoarseWSD-20 (Loureiro, Rezaee, Pilehvar and Camacho-Collados, 2021,
  *Computational Linguistics* 47(2)) for the sense-labelled evaluation: clone
  <https://github.com/danlou/bert-disambiguation> and pass the checkout (or its
  `data/CoarseWSD-20`) as `--data-root` to `scripts/run/run_wsd_feature_alignment.py`.
- The three-seed dictionaries behind the stability appendix are not on the Hub;
  train them from `configs/sweeps/qwen35_2b_seedvar_base.yaml` (see the
  `configs/sweeps/README.md` table for the per-cell overrides).

Pretrained readout-feature dictionaries (the *selected* SAEs cited in the
paper) are published on the Hugging Face Hub at
[`hematteo/sparse-readout-prism`](https://huggingface.co/hematteo/sparse-readout-prism)
(`<model>/<operating_point>/checkpoint.pt`).
`configs/registries/exp2_selected_sae_checkpoints.yaml`
records the canonical manifest (each checkpoint's model, width, `k`, metrics,
and Hub path). Training one from scratch costs ~12 h of GPU time per
model; to exercise the full pipeline now without data or a GPU, run the
synthetic smoke config (see the README quickstart).

### Determinism

The training runner seeds Python / NumPy / Torch RNG
(`sparse_readout_prism.utils.set_seed`); the evaluation and figure scripts use
locally seeded generators (fixed seeds or case-id-derived seeds) rather than
global seeding. Either way a fixed seed reproduces the same sampling and
bootstrap draws. This is
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
| Method schematics (`fig:method-schematic` / `readout_query_one_column`, score-accounting TikZ) | — (paper-side LaTeX/TikZ; no producing script in this repo) | — |
| §4 main fidelity table (Qwen 0.8B/2B/9B) | `scripts/eval/task_fidelity_evaluation.py` | `configs/registries/exp2_selected_sae_checkpoints.yaml` |
| §4 main case-study four-panel figure (`fig:selected-readout-prism-examples`: bug/error, bug/insect, bark/dog, bass/trout) | `scripts/figures/compute_sae_paper_examples.py` | hardcoded example sets (`--example-set section43_polysemy_margin`, the default; `EXAMPLE_SETS` in `research/qwen_example_prompts.py`), plus a required `--checkpoint` |
| §1 intro figure: logit-lens vs Sparse Readout Prism comparison (`fig:main-lens-prism-comparison`) | `scripts/figures/compute_lens_prism_comparison.py` | defaults reproduce the paper case (software-context prompt, `--target-a " bug"`, `--target-b " insect"`, `--k 256`); `--checkpoint` points at the Qwen3.5-2B 32x/k256 selected SAE in `configs/registries/exp2_selected_sae_checkpoints.yaml`. Computes readout states live (reuses `collect_readout_states_batched` from `sparse_readout_prism.research.qwen_readout`; no precomputed cache) |
| Table / metrics behind: benchmark-derived task targets | `scripts/run/run_benchmark_derived_query_suite.py`, `scripts/figures/compute_benchmark_task_group_examples.py` | `--out-dir results/benchmark_derived_query_suite_task_group_v2_qwen_20260523` (the suite uses task-group contrasts — the paper's `task_group_v2` design — and does not derive the dir from it; pass `--out-dir` explicitly) |
| Metrics behind: selected-score / family / selection queries (`fig:qwen-general-readout-scores-basic`, `fig:app-qwen-general-readout-scores-extra`) | `scripts/figures/compute_general_readout_queries.py` | query specs are built in (the paper's jury/guilty panels plus the abstention-family contrast); `--checkpoint` points at the Qwen3.5-2B 32x/k256 selected SAE in `configs/registries/result1_query_fidelity_cluster.yaml` |
| Metrics behind: literature-prompt all-layer trace | `scripts/figures/compute_all_layer_literature_prompt.py` | requires `--checkpoint` (the Qwen3.5-2B 32x/k256 selected SAE); prompt/model defaults reproduce the paper trace |
| `tab:app-direct-geometry-grid`; SRP / best-alternative / lead columns of `tab:readout-score-fidelity-summary` | `scripts/run/run_readout_baseline_comparisons.py` | one run per softcap-free readout, `--model` in Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-9B, Ministral-3-8B-Base, R1-Distill-Qwen-7B, R1-Distill-Llama-8B: `--registry configs/registries/result1_query_fidelity_cluster.yaml --bank-dir data/query_banks --banks curated_ab,case_candidates,model_native --model-native-file qwen_gemma_result1_model_native_prompts_c4.jsonl --operating-point fidelity --methods sparse_rp,nearest_row_ridge_top128,knn_basis_top128,row_cluster_d16384_k256,row_cluster_d65536_k256,row_cluster_hard_d65536,pca_256 --max-native 500 --max-curated 320 --max-cases 40 --max-len 64 --seed 0 --out-root results/direct_geometry_runs/<tag>`; coverage = `accepted_rate` with `accepted_ci_lo/hi` in `baseline_by_method.csv` (unfloored rho_0, 400 base-case resamples). The C4 model-native slice is regenerated with `scripts/data/build_query_banks.py --native-source c4` (see §0) Since 0.2.1 the per-feature coverage/compactness columns for `nearest_row_ridge_top*`, `knn_basis_top*` and `row_cluster_*` are computed on aligned supports (the scalar coverage/sign metrics the tables report are unchanged). |
| Result-1 five-model query-fidelity tables / §4.5 distributional readout metrics; per-model reconstruction with cluster-bootstrap intervals (`tab:app-fidelity-cis`) | `scripts/run/run_query_fidelity_bank.py` | `configs/registries/result1_query_fidelity_cluster.yaml` + the curated/model-native banks under `data/query_banks/`. `tab:app-fidelity-cis` reads, per model, `sign_agreement` (+`_ci_lo/_ci_hi`), `median_rho5` (+CI) and `pass_rho5_lt_0.50` (+CI) from the group metrics: `rho5` is the floored relative error \|m_exact − m_sparse\| / (\|m_exact\| + 0.5) and every interval comes from the same 400 base-case resamples |
| Appendix J lexical-edit / readout-side control stress test (`tab:lexical-control-stress-test`, `tab:lexical-control-primary-methods`, `tab:lexical-control-cross-model-results`) | `scripts/run/run_qwen_profanity_suppression_eval.py` | `--checkpoint <ckpt.pt> --model-id <hf id> --out-dir <out> --device cuda --dtype bfloat16 --batch-size 8 --prompt-limit 16 --pair-limit 9 --max-open-prompts 6 --num-samples-per-prompt 2 --max-new-tokens 24` (the original 16-prompt × 9-pair grid; `baseline_comparison.csv` / `candidate_constrained_summary.csv` at scale 16 give the primary-methods table). Cross-model rows: Qwen3.5-2B and DeepSeek-R1-Distill-Qwen-7B at 32x/k256, Ministral-3-8B-Base at 16x/k128 (`configs/registries/result1_query_fidelity_cluster.yaml`). Note: with the extended scale grid, `choose_operating_point` can select a scale above 16; read the scale-16 rows from `candidate_constrained_summary.csv` |
| Appendix K sweep / selection tables (`tab:app-k-model-finalists`, `tab:app-model-suite-current`, and the other `tab:app-k-*` sweep ledgers) | — (no figure ships; numbers hand-transcribed from the audited literals in [`data/appendix/appendix_k_sweep_tables.json`](../data/appendix/appendix_k_sweep_tables.json)) | `configs/sweeps/arch_frontier_160m.yaml`, `configs/sweeps/paper_phase2_frontier.yaml` |
| Appendix L: global readout-feature profiles (shell/java, port/cell) | `scripts/analyze/analyze_readout_feature_directions.py` | requires `--checkpoint` (the Qwen3.5-2B 32x/k256 selected SAE); `--sae-id qwen2b_k256 --context-set interesting_domains --case-ids shell_beach,shell_terminal,java_coffee,java_programming --top-directions 20 --out-dir results/qwen_pre_token_readout_directions_shell_java` (rerun with the `port_harbor,port_network,cell_prison,cell_biology` case-ids and `--out-dir results/qwen_pre_token_readout_directions_port_cell` — the output basenames are fixed, so the two runs MUST use distinct `--out-dir`s or the second overwrites the first); persists the per-direction metrics behind the global-readout bars |
| Appendix M: metrics behind feature-resolved DLA, vocab-mean verify contrast | `scripts/figures/compute_prism_dla.py` | requires `--checkpoint` (the Qwen3.5-2B 32x/k256 selected SAE); `--contrast-mode vocab_mean --out-dir results/qwen2b_32x_prism_dla_verify_token_layer_readout` (output basenames are the fixed `qwen2b_32x_prism_dla_*` prefix; the dir distinguishes this vocab-mean run from the default `_verify_assume`) |
| Appendix L: metrics behind additional bark/bass margin case studies | `scripts/figures/compute_sae_paper_examples.py` | `--example-set section43_margin_appendix2 --out-dir results/qwen2b_32x_margin_examples_appendix2 --k 256` (32x/k256 checkpoint; output basenames are the fixed `qwen2b_selected_paper_examples_*`, so a distinct `--out-dir` keeps them from overwriting the default four-panel run) |
| §4 / Appendix L: fixed-token context delta metrics (`fig:same-token-polysemy`: bug, ring, bridge; plus the ring appendix panel) | `scripts/data/mine_polysemy_features.py` | `--force-pair-ids bug_insect_software,ring_jewelry_phone,bridge_structure_network` (pairs come from the script's built-in domain candidate bank) |
| Appendix L: technical-domain context deltas (`fig:app-qwen2b-technical-domain-delta`: graph, memory, virus, pipe) | `scripts/data/mine_polysemy_features.py` | `--force-pair-ids graph_chart_datastructure,memory_human_computer,virus_medical_software,pipe_plumbing_unix --out-dir results/qwen2b_32x_domain_polysemy_appendix_focus` (distinct `--out-dir`; output basenames are the fixed `qwen2b_32x_polysemy_*`) |
| Appendix K: W_U row-norm tail stats (`tab:app-k-row-norm-tail`) | `scripts/analyze/compute_row_norm_tail_stats.py` | `--registry configs/registries/result1_query_fidelity_cluster.yaml --models Qwen3.5-9B,Ministral-3-8B-Base,R1-Distill-Qwen-7B,R1-Distill-Llama-8B` (or `--artifact "label=<extract>.pt"`); centred row norms, no GPU/checkpoint |
| Appendix K: dense / negative-control diagnostic (`omp_diagnostic` in [`data/appendix/appendix_k_sweep_tables.json`](../data/appendix/appendix_k_sweep_tables.json)) | `scripts/eval/dense_control_diagnostic.py` | `--w-u <extract>.pt --checkpoint <ckpt.pt> --setting "Qwen-9B 32x, k=256" --n-rows 20000 --bootstrap 1000` (encoder vs LS/NNLS-on-support vs signed/nonneg OMP vs dense rank-k, scored by rowEV) |
| Appendix L: feature-label audit (`tab:app-main-case-study-feature-audit` data; `tab:app-qualitative-feature-label-audit` counts) | `scripts/analyze/audit_feature_labels.py` | `substrate --feature-ids 36,4095,… --model-id Qwen/Qwen3.5-2B` emits per-feature top rows; `aggregate --annotations data/audit/feature_label_audit_annotations.csv --validate-against data/audit/feature_label_audit.json` tallies the counts (classification is human; see note below) |
| Matched-KL frontier metrics (`fig:lexical-matched-kl-frontier`) | `scripts/run/run_qwen_profanity_suppression_eval.py` | once per model (Qwen/Qwen3.5-2B, Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-9B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; 32x/k256 checkpoints): `--checkpoint <ckpt.pt> --model-id <hf id> --out-dir <out> --device cuda --dtype bfloat16 --batch-size 8 --max-open-prompts 0` (full 20-prompt × 17-pair × 14-scale × 8-method grid); the frontier plots the `split == heldout` rows of `candidate_constrained_summary.csv` (`median_kl_bits` vs `mean_bad_prob_reduction` / `candidate_flip_rate` per method and scale) |
| Paired differences at matched KL quoted with `fig:lexical-matched-kl-frontier` (SRP − mean-row / PCA rank-1 / PCA rank-4; term- and prompt-clustered 95% CIs) | `scripts/eval/paired_matched_kl_bootstrap.py` | `--input qwen2b=<run>/candidate_constrained_rows.csv --input qwen0p8b=... --input qwen9b=... --input r1qwen7b=... --out paired_matched_kl_results.json --n-boot 10000 --seed 0` (default `--target-kl 0.02 0.05 0.1 0.2`); CPU, ~17 s for four models. Output is a JSON object whose `comparisons` list holds the 96 paper records (order follows the `--input` order) beside a `provenance` block; reproduces the paper's paired results exactly from the paper CSVs |
| `tab:app-cross-model-nulls` | `scripts/run/run_readout_baseline_comparisons.py` | one run per model Qwen3.5-0.8B/2B/9B, Gemma-4-E2B, Gemma-4-E4B with `--methods sparse_rp,shuffled_row_code,random_support_same_magnitudes` (all other flags as in the row above); coverage = `accepted_rate`, sign = `sign_agreement` in `baseline_by_method.csv` |
| `fig:app-robustness-baseline-comparisons` (Qwen3.5-2B reference panel; the 848-cluster CI paragraph of the fidelity-results subsection) | `scripts/run/run_readout_baseline_comparisons.py` | `--model Qwen3.5-2B --methods sparse_rp,shuffled_row_code,random_support_same_magnitudes,pca_64,pca_256,pca_1024,nearest_row_ridge_top128` (other flags as above); `baseline_by_method.csv`, `baseline_by_method_margin_bin.csv`, `baseline_feature_compactness.csv` |
| `tab:app-error-tails`, `tab:app-error-tail-margins` (and the per-model mean/p95/max ranges quoted in that subsection) | `scripts/eval/analyze_error_tails.py` | `--input-root results/direct_geometry_runs --input-prefix <common dir prefix> --model-tags qwen0p8b,qwen2b,qwen9b,ministral8b,r1qwen7b,r1llama8b --primary-model-tag qwen2b --out-dir results/error_tails --n-boot 10000 --seed 20260711` over the six direct-geometry run dirs; `summary_pooled_by_method.csv` (8,009 contrasts per method), the `sparse_rp` block of `summary_pooled_by_method_margin_bin.csv`, `summary_by_model_method.csv`. Verified to reproduce both tables from the paper's raw run CSVs |
| `tab:app-nearest-rows` | `scripts/analyze/nearest_rows_baseline.py` | `--model-id Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --contrast bug,insect --contrast bug,error --top-n 12 --out-dir results/nearest_rows_qwen2b` (full-vocabulary centering mean, contrast tokens excluded); the `centered_*` columns of `nearest_rows_table.csv` are the table, `nearest_rows.json` holds the raw-cosine ranking too |
| Table 2 predicted-vs-realized readout-side changes, one row per model (`tab:causal-validation-summary`; per-model rows and protocol in `app:causal-validation`) | `scripts/eval/run_causal_contribution_validation.py` | once per model, fidelity operating point (32x, k=256): `--model-id <MODEL_ID> --checkpoint <hf_hub_download("hematteo/sparse-readout-prism", "<model>/k256_32x/checkpoint.pt")> --bank-dir data/query_banks --out-dir results/causal_<TAG> --device cuda --dtype bfloat16 --seed 0`, with defaults `--banks curated_ab,case_candidates,model_native --max-native 300 --top-features 10 --random-per-case 10 --rho-gate 0.5 --n-boot 2000 --max-len 64 --centering live` (`--centering trained` centres on the checkpoint's stored `row_mean`, else the tokenizer's text-token mean; the paper runs used the live full-vocabulary mean, which differs by ~0.04% of a centred row norm on Qwen3.5-2B). Model → Hub dir: `Qwen/Qwen3.5-0.8B`→`qwen3.5-0.8b`, `Qwen/Qwen3.5-2B`→`qwen3.5-2b`, `Qwen/Qwen3.5-9B`→`qwen3.5-9b`, `mistralai/Ministral-3-8B-Base-2512`→`ministral-3-8b`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`→`r1-distill-qwen-7b`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`→`r1-distill-llama-8b`. Table columns come from `summary.json`: r²/CI/slope/n ← `gated_predicted`, Random ← `gated_random_control.r2`; `ungated_predicted` is the all-pairs comparison. The `model_native` bank carries no A/B targets and contributes zero contrasts (a missing file is logged and skipped) |
| Appendix harness self-test line (`app:causal-validation`: r²=1.000, slope 1.04, zero-scoring controls) | `scripts/eval/run_causal_contribution_validation.py` | `--self-test` (CPU, no model/checkpoint; prints `r2=1.000 slope=1.044` and three PASS lines, the third comparing each stored realized change against the dense LM-head margin change) |
| Appendix I: cross-seed stability of contrast explanations (`tab:app-cross-seed-stability`) | `scripts/eval/cross_seed_stability.py` | three same-recipe checkpoints trained from `configs/sweeps/qwen35_2b_seedvar_base.yaml`; `--dictionaries results/qwen35_2b_seedvar/qwen2b_d65536_k256_s0/checkpoint.pt …_s1/checkpoint.pt …_s2/checkpoint.pt --w-u "$SRP_ARCHIVE_ROOT/data/qwen35-2b/qwen35_2b.pt" --bank data/query_banks/qwen_gemma_result1_curated_ab.jsonl --width-tag 32x --out results/seed_stability/cross_seed_stability_32x.json` (defaults `--max-contrasts 150 --seed 0 --top-m 8 --top-r 12 --n-sample 4096 --n-hidden 512`; 16x column: the `qwen2b_d32768_k128_s{0,1,2}` checkpoints, `--width-tag 16x`) `--centering live` is the default (paper); `--centering trained` centres on the dictionaries' stored `row_mean`, else the payload's token-mask mean. Output JSON carries `provenance`. |
| Appendix I: cross-seed feature-group matching incl. the below-null direction check (`tab:app-feature-group-matching`) | `scripts/eval/feature_group_matching.py` | same dictionaries / `--w-u` / `--bank` as above; `--width-tag 32x --direction-check --out results/seed_stability/feature_group_matching_32x.json` (defaults `--n-query 100 --n-null 500 --greedy-pool 200 --greedy-k 3 --min-group-size 4 --direction-null 500 --direction-seed 0`; 16x: the k128 checkpoints, `--width-tag 16x`) `--centering` as above. Since 0.2.1 the frequency-matched null pool is sorted, so null draws no longer depend on `PYTHONHASHSEED`; they are a fresh reproducible sample rather than the paper run's specific draws (the paper's above-null counts were computed before this fix). |
| Appendix I: held-out recovery of the stable core (`tab:app-loo-core-recovery`, plus the 0.444 cross-recipe figure) | `scripts/eval/loo_core_recovery.py` | same dictionaries / `--w-u` / `--bank`; `--width-tag 32x --reference-dict <released Qwen3.5-2B k256_32x checkpoint> --out results/seed_stability/loo_core_recovery_32x.json` (defaults `--side-n 96 --n-clusters 16384 --kmeans-iters 12 --kmeans-seed 0 --n-boot 2000 --bootstrap-seed 1`; 16x: k128 checkpoints, `--width-tag 16x`, no `--reference-dict`; k-means wants a GPU/MPS) `--centering` as above; `--knn-exclude-self` (default, paper) drops the query row from the kNN side set while the SRP and cluster sides keep it — `--no-knn-exclude-self` removes that asymmetry. |
| Appendix K: sparsity budget vs dictionary usage (`tab:app-low-k-dead-features`) | `scripts/train/train_readout_sae_from_config.py` | multi-seed rows: `--config configs/sweeps/qwen35_2b_seedvar_base.yaml` with `--set factorizer.d_features=<D> --set factorizer.k=<k> --set evaluation.k=<k> --set run.seed=<s> --set run.init_seed=<s> --set data.data_seed=<s> --set run.output_dir=results/qwen35_2b_seedvar/qwen2b_d<D>_k<k>_s<s> --set run.name=qwen2b_d<D>_k<k>_s<s>` over cells (32768,256,{0,1,2}) (65536,128,{0,1,2}) (32768,64,{1,2}) (plus the Appendix I family (65536,256,{0,1,2}) (32768,128,{0,1,2})); single-seed rows: `--config configs/sweeps/lowk_qwen35_2b_16x_base.yaml` and `--config configs/sweeps/lowk_qwen35_0p8b_16x_base.yaml` with `--set factorizer.k=<32 or 64> --set evaluation.k=<32 or 64>` and `run.name`/`run.output_dir`; dead / rare / top-1 / KL read from each cell's `metrics.json` |
| Appendix K: Qwen3.5-0.8B 32x/k256 seed window (`tab:app-k-qwen08b-seed-window`) | `scripts/train/train_readout_sae_from_config.py` | `--config configs/sweeps/qwen35_0p8b_paper_topk_32x_k256_s1.yaml` and `--config configs/sweeps/qwen35_0p8b_paper_topk_32x_k256_s2.yaml`, no `--set`; seed 0 row is the archived dictionary registered as `qwen0p8b_k256` in `configs/registries/exp2_selected_sae_checkpoints.yaml` (the same checkpoint is `qwen0p8b_k256_32x` in `result1_query_fidelity_cluster.yaml`) |
| Appendix "Sense-Labelled Evaluation": frozen CoarseWSD-20 readout bundles and coverage statistics (`app:sense-labelled-evaluation` data paragraph: contexts scored, single-token coverage, row-gate coverage, median target rank) | `scripts/run/run_wsd_feature_alignment.py` | `--dataset coarsewsd20 --data-root <bert-disambiguation checkout> --model-id <Qwen/Qwen3.5-2B, Qwen/Qwen3.5-9B, deepseek-ai/DeepSeek-R1-Distill-Llama-8B> --checkpoint <that model's 32x/k256 selected dictionary> --out-dir results/wsd_core/coarsewsd20_<tag> --splits train,test --batch-size 8 (2B) / 4 (9B, R1-Llama) --n-boot 5000 --seed 0`; writes `representations.pt` (input to the two rows below), `audit.json`, `scoring_summary.json`, `metrics.json`; `--analyze-only --out-dir <dir> --n-boot 5000 --seed 0` recomputes `metrics.json` without inference Flags since 0.2.1: `--centering {live,trained}` (default `live`, paper), `--k` (default: the checkpoint's k); `--analyze-only` writes `analysis_config.json` and leaves `run_config.json` untouched; `scoring_summary.json` records `truncation` (cloze prompts are truncated from the left, a 0.2.1 fix); `metrics.json` carries `provenance`. Only CoarseWSD-20 is supported. |
| Sense alignment on CoarseWSD-20 (`tab:app-sense-alignment`; group-size 1/2/4/8 sweep in the appendix prose) | `scripts/analyze/analyze_wsd_sense_groups.py` | `--bundle results/wsd_core/coarsewsd20_<tag>/representations.pt --out results/wsd_sense_groups/coarsewsd20_<tag>__srp.json --group-sizes 1,2,4,8 --selector mean_diff --n-null 200 --n-boot 2000 --seed 0` (defaults: `--basis srp`, train-standardized); the table reads `<stem>_g8.json` fields `word_mean_balanced`, `word_mean_null_balanced`, `word_mean_majority_balanced`, `words_beating_null_p95_balanced`, `words_beating_majority_balanced`, `word_mean_full_account_balanced`, `word_mean_hidden_balanced`; CPU only. Verified against the paper run outputs cell for cell `--revision` (default: the cached `main` snapshot) and, for the `--basis` geometry controls, `--centering` (default `live`) with an optional `--checkpoint` supplying the stored `row_mean`; output carries `provenance` with sorted keys. Verified identical to the paper run outputs after the 0.2.1 refactor. |
| Appendix "Sense-Labelled Evaluation", classifier-framing paragraph (projections vs signed contributions, row-coefficient shuffles, random features) | `scripts/analyze/analyze_wsd_classifier_framing.py` | `--bundle results/wsd_core/coarsewsd20_<tag>/representations.pt --out results/wsd_classifier_framing/coarsewsd20_<tag>.json --ks 5,10,20,50 --primary-k 10 --primary-encoding weighted --n-boot 5000 --n-null-seeds 20 --seed 0` (all defaults); the paragraph quotes `primary.methods.<source>.accuracy`; CPU only Output carries `provenance` with sorted keys; verified identical to the paper run outputs after the 0.2.1 refactor. |
| Fitted lenses for the cross-lens study (inputs to all `fig:cross-lens-*` / `tab:app-cross-lens-*`) | `scripts/run/fit_jlens.py` | `prompts --prompts-json <out>/c4_prompts_en_seed0.json --n-prompts 1000 --seed 0` (default `--min-chars 800 --c4-config en`); `prompts … c4_prompts_zh_seed0.json --n-prompts 1000 --seed 0 --min-chars 300 --c4-config zh`; same with `--c4-config de`; `smoke --model-id Qwen/Qwen3.5-0.8B --prompts-json <en dump> --out <ckpt>/smoke_0p8b.lens.pt --dim-batch 16`; then per corpus `fit --model-id Qwen/Qwen3.5-9B --prompts-json <dump> --n-prompts 100 --shard $i --num-shards 4 --n-layers 12 --dim-batch 8 --ckpt-dir <ckpt> --out-dir <ckpt>` for i in 0..3 and `merge --shard-dir <ckpt> --num-shards 4 --out <out>/qwen35_9b_jlens_{en,zh,de}_seed0_n100.pt`; `--n-prompts 300` for the `*_n300.pt` refits. Needs `uv sync --extra lens` `fit`/`smoke` take `--device` (default `cuda`); `fit` writes `shard{i}.meta.json` beside each shard and refuses to resume a shard whose sidecar differs; `merge` requires the sidecars and writes `<out stem>.meta.json`; `--dim-batch` halves on OOM down to 1. |
| Ridge translators (`tab:app-cross-lens-extension` rows 3–4, `fig:cross-lens-extension`) | `scripts/run/fit_ridge_lens.py` | `--model-id Qwen/Qwen3.5-9B --prompts-json <out>/c4_prompts_en_seed0.json --n-prompts 100 --holdout 10 --layers-from <out>/qwen35_9b_jlens_en_seed0_n100.pt --ckpt-dir <ckpt> --tag ridge_en --out <out>/qwen35_9b_ridgelens_en_seed0_n100.pt`; repeat with `c4_prompts_zh_seed0.json --tag ridge_zh --out …ridgelens_zh_seed0_n100.pt` (layers still from the EN Jacobian lens). Holdout R² and deep top-1 agreement land in `<out>.report.json` `--holdout` must be >= 1; `--device` (default `cuda`); accumulator resume is guarded by `<ckpt>/<tag>.meta.json`; the report carries `provenance`. The lambda grid is scaled per token but applied to the N-token Gram sum, so both paper fits selected the grid's top value at every layer — kept as the paper ran it. |
| Readout dumps behind every cross-lens artifact | `scripts/run/run_cross_lens_readouts.py` | once per (lens, bank): `--model-id Qwen/Qwen3.5-9B --lens <lens.pt> --checkpoint <qwen3.5-9b/k128_8x/checkpoint.pt> --k 128 --prompts data/cross_lens/cross_lens_prompts_en_zh.json --n-positions 1 --decompose-top1 --out <dumps>/en_zh__<lens>.json` for the EN/ZH Jacobian n=100 and n=300 lenses and the EN/ZH ridge translators; `--prompts data/cross_lens/cross_lens_prompts_en_de.json` for the EN/DE Jacobian n=100 and n=300 lenses; three-lens example: `--prompts data/cross_lens/cross_lens_three_lens_prompts.json --n-positions 1 --decompose-top1 --top-feats 10` under the EN, ZH, DE n=100 lenses (optionally `--lens identity --layers-from <EN lens>`) `--k` defaults to the checkpoint's k; `--centering live` (default, paper) or `trained`; `--device` (default `cuda`); `<out>.manifest.json` records k, centering and provenance. |
| `tab:app-cross-lens-families`; `tab:app-cross-lens-extension` rows 1, 3, 4, 5; `fig:cross-lens-extension` (EN–ZH cells) | `scripts/eval/aggregate_cross_lens_en_zh.py` | `--prompts data/cross_lens/cross_lens_prompts_en_zh.json --seed 0 --out <out>/<cell>_summary.json [--rows-csv …]` with `--lens-a/--lens-b` = (jlens_en_n100, jlens_zh_n100) main study; (ridge_en, ridge_zh) ridge row; (jlens_en_n100, ridge_en) Jacobian-vs-ridge row; (jlens_en_n300, jlens_zh_n300) n=300 row. Per-family agreement/CI and the "Lens-only split" column are `per_group` and `lens_only_split`; floors are `null_within_lens` (/159) and `null_shuffle_cross_lens` (/240). Verified to reproduce the paper run's summary key for key `--agreement-rule half` (default, paper: at least half of the mid-band layers; `strict` = more than half, which gives 72/80 on the main cell). Summary carries `provenance`. |
| `tab:app-cross-lens-extension` rows 2, 6; `fig:cross-lens-extension` (EN–DE cells); second-language-pair per-family counts, 73/90 and 76/90 divergence | `scripts/eval/aggregate_cross_lens_en_de.py` | `--lens-a <dumps>/en_de__jlens_en_n100.json --lens-b <dumps>/en_de__jlens_de_n100.json --prompts data/cross_lens/cross_lens_prompts_en_de.json --seed 0 --out <out>/en_de_jlens_n100_summary.json`; same with the `_n300` dumps. Divergence is `divergence_rate`; floors /204 and /270 `--agreement-rule half` (default) and `--null-population all` (default, /204; `cross` drops the 12 controls, /180, the EN–ZH population). Summary carries `provenance`. |
| EN–DE bank + cognate-exclusion report (inputs to the EN–DE cells) | `scripts/data/build_cross_lens_de_bank.py` | `--tokenizer Qwen/Qwen3.5-9B --out data/cross_lens/cross_lens_prompts_en_de.json --report data/cross_lens/cross_lens_prompts_en_de_exclusion_report.json` (shipped outputs are in `data/cross_lens/`) The exclusion report carries `provenance`; the bank is byte-identical to the shipped file. |
| `tab:app-cross-lens-antonym-layers` (and the f112 contributions quoted beside it) | `scripts/analyze/cross_lens_antonym_layers.py` | `run --model-id Qwen/Qwen3.5-9B --lens <out>/qwen35_9b_jlens_en_seed0_n100.pt --checkpoint <k128 checkpoint.pt> --k 128 --out <out>/antonym_layers_en.json` (default `--prompt '"小"的反义词是"'`); same with the ZH lens → `antonym_layers_zh.json`; then `table --dump EN=<out>/antonym_layers_en.json --dump ZH=<out>/antonym_layers_zh.json --layers 24,26,29,final --out-csv <out>/antonym_layers_table.csv --out-features-csv <out>/antonym_layers_features.csv` `run` also accepts `--centering` / `--device` and writes `<out stem>.manifest.json`; `table` writes `<out-csv stem>.manifest.json`. |
| `tab:app-cross-lens-de-antonym-layers` (and the groß/large/big feature ids and 15.16/15.60/22.19 logits in the prose) | `scripts/analyze/cross_lens_three_lens_prompt.py` | `--dump EN=<dumps>/three_lens__jlens_en.json --dump ZH=<dumps>/three_lens__jlens_zh.json --dump DE=<dumps>/three_lens__jlens_de.json --prompt-id antonym_de_01 --layers 24,26,29 --out-csv <out>/three_lens_top1.csv --out-targets-csv <out>/three_lens_targets.csv --out-features-csv <out>/three_lens_features.csv` (table = `top1_token`, `top1_lens_logit`; quoted groß logits = `original_logit` column) Writes `<out-csv stem>.manifest.json`. |
| `fig:cross-lens-butterfly` (metrics only) | `scripts/figures/compute_cross_lens_shared_feature.py` | `--dump EN=<dumps>/en_zh__jlens_en_n100.json --dump ZH=<dumps>/en_zh__jlens_zh_n100.json --prompt-id fac_03 --layer 26 --out-csv <out>/cross_lens_shared_feature.csv --out-json <out>/cross_lens_shared_feature.json` (top-1 logits 38.5 / 39.2, f23180 shares 0.752 / 0.629, largest other +1.69 / +1.55) Writes `<out-csv stem>.manifest.json`, or `provenance` inside `--out-json` when given. |

### Notes and paper-only artifacts

Like every results table in the paper, the fidelity / selection / baseline /
lexical-edit tables are transcribed by hand from the CSV outputs of the scripts
above. A few artifacts have no producing script and are authored in the paper's
LaTeX source:

- **Schematics** — `fig:method-schematic` (`readout_query_one_column.pdf`) and the
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
- **`fig:qwen-prism-dla-verify-token-app`** — `compute_prism_dla.py` persists the
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

The ported experiment code carries its own CPU tests: additive identity of the
weighted row-kNN and k-means centroid baselines, the causal-validation harness
self-test, the paired matched-KL bootstrap on synthetic rows, the cross-lens
aggregators and table builders on stub dumps, sense-group selection and
balanced accuracy on a synthetic bundle, and the seed-stability contrast
pipeline on synthetic dictionaries.

The test suite is fast and CI-safe (no GPU, no model loads): identity
correctness of the decomposition, schema invariants of the task-fidelity
evaluator, checkpoint-registry resolution, the data-loading branches
(token-mask filtering, val-split fallbacks), the trainer's L0 controller and
resume path, the public-API surface, and layout guardrails for the
`research/` package.

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
in `sparse_readout_prism.research.qwen_readout` next to its single-prompt
twin `collect_readout_state`.
