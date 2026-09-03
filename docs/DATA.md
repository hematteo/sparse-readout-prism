# Data: inputs, intermediates, outputs

Where each kind of artifact lives, what its schema is, and how to wire up
your own. For the figure → script + config map, see
[`REPRODUCE.md`](REPRODUCE.md).

## 1. Query banks (paper inputs)

Curated and model-native prompt + target-pair JSONL files live in
[`data/query_banks/`](../data/query_banks/). The full schema and a
file-by-file table are in
[`data/query_banks/README.md`](../data/query_banks/README.md).

A record is one **case** — a prompt, a position to score, two targets
(`A`, `B`), and an expected side. Every script that consumes a bank reads
the same schema; the banks are model-agnostic.

One further paper-input set lives beside them. [`data/cross_lens/`](../data/cross_lens/)
holds the EN–ZH and EN–DE prompt banks of the cross-lens study (one record per
prompt: two surface forms, an unrelated null target, family and control tags)
together with the EN–DE cognate-exclusion report and the one-prompt bank of the
three-lens worked example; schema and provenance are in
[`data/cross_lens/README.md`](../data/cross_lens/README.md).

## 2. Extracted readouts (per-model artifacts)

Run `scripts/data/extract_model_readout.py --model-id <hf-id> --out <path>.pt`
to produce the per-model artifacts. The script writes a **single** `.pt` file at
`--out` holding one payload dict, plus a sibling `<stem>_manifest.json` next to
it (not separate `W_U_orig.pt` / `hLN.pt` files):

| Output | Contents |
|---|---|
| `<out>.pt` payload key `W_U_orig` | `(vocab, d_model)` unembedding matrix, fp32. |
| `<out>.pt` payload key `h_LN` | `(n_prompts, max_len, d_model)` final-norm hidden states (lm_head input) — input to SAE training. |
| `<stem>_manifest.json` | Model id, revision, vocab/d_model, `tie_word_embeddings`, adapter-check results (recon-vs-lm_head identity, post-readout-transform / softcap detection). |

The hidden-state prompt bank is wikitext-2 — specifically what
`scripts/data/extract_model_readout.py` loads (`wikitext-2-raw-v1` / `test`),
with a built-in offline fallback when the dataset can't be fetched; the readout
itself comes straight from the model's `lm_head`.

## 3. SAE checkpoints (pretrained dictionaries)

The selected dictionaries cited in the paper are published on the **Hugging Face
Hub** at [`hematteo/sparse-readout-prism`](https://huggingface.co/hematteo/sparse-readout-prism),
laid out as `<model>/<operating_point>/checkpoint.pt` (download with
`huggingface_hub.hf_hub_download`). The manifest in
[`configs/registries/exp2_selected_sae_checkpoints.yaml`](../configs/registries/exp2_selected_sae_checkpoints.yaml)
records each checkpoint's model / width / `k` / metrics and its Hub path.

A checkpoint is a `.pt` file with:

- `model_state_dict` — the factorizer weights (encoder, decoder, biases).
- `factorizer` — the factorizer config sub-block (`architecture`, `k`,
  `d_features`, ...). `build_factorizer({"factorizer": ckpt["factorizer"]}, d_model)`
  rebuilds the model from this block (or use `load_factorizer(ckpt)`, which
  rebuilds + loads in one call, inferring `d_model` from the checkpoint).
- `row_mean`, `row_norms` — the preprocessing pinned at training time.
  Decomposing against a different preprocessing breaks the identity.

Downloading the released checkpoints skips ~12 h of GPU time per model.

## 4. Training your own

```bash
# 1. extract the readout + hidden states (single .pt payload + manifest json)
uv run python scripts/data/extract_model_readout.py \
    --model-id Qwen/Qwen3.5-2B \
    --out "$SRP_SSD_ROOT/srp/qwen2b/hLN.pt"

# 2. train an SAE on the centered+normalized W_U rows
#    (--data-path -> data.path, --out-dir -> run.output_dir)
uv run python scripts/train/train_readout_sae_from_config.py \
    --config configs/sweeps/paper_phase2_base.yaml \
    --data-path "$SRP_SSD_ROOT/srp/qwen2b/hLN.pt" \
    --out-dir "$SRP_SSD_ROOT/srp/qwen2b/run0"
```

(`paper_phase2_base.yaml` is the complete headline run recipe in
`configs/sweeps/` — TopK 32×/k256; its `factorizer.d_features: 65536` is `32 ×
d_model` for Qwen-3.5-2B, so override it with `--set
factorizer.d_features=<width×d_model>` for another model. See
`docs/REPRODUCE.md` §1.)

To reproduce a **strict-budget (16×/k128)** cell for a model without a
dedicated `*_16x_k128.yaml` config (e.g. the Gemma points), the
full override set is:

```bash
uv run python scripts/train/train_readout_sae_from_config.py \
    --config configs/sweeps/paper_phase2_base.yaml \
    --data-path <extract>.pt --out-dir <run_dir> \
    --set factorizer.d_features=<16 × d_model> \
    --set factorizer.k=128 \
    --set evaluation.k=128
```

(`factorizer.k` and `evaluation.k` both matter — the dedicated
`deepseek_*_16x_k128.yaml` / `ministral3_8b_paper_16x_k128.yaml` configs show
the same three-field change.)

## 5. Adapting to a different model

The **core pipeline** — extraction, SAE training, task-fidelity / query-fidelity
/ baseline evaluation — is **model-agnostic** as long as the model has a `lm_head`
attribute that the helper in `sparse_readout_prism.utils.find_lm_head` can locate
(these paths load via `AutoModelForCausalLM`, falling back to
`AutoModelForImageTextToText`). Currently shipped configs are Pythia-160M and
Qwen-3.5-2B (paper nickname: `qwen2b`).

The per-figure **case-study and mining scripts** are not model-agnostic: the
qualitative producers `scripts/figures/compute_lens_prism_comparison.py`,
`compute_sae_paper_examples.py`, `compute_prism_dla.py`,
`compute_benchmark_task_group_examples.py`,
`compute_general_readout_queries.py`,
`scripts/analyze/analyze_readout_feature_directions.py`,
`scripts/data/mine_polysemy_features.py`, and
`scripts/run/run_benchmark_derived_query_suite.py` default to
`--model-id Qwen/Qwen3.5-2B`, and their prompts and example sets assume that
model. They all share one loader —
`research.qwen_readout.load_qwen_model`, a thin wrapper over
`utils.load_causal_lm` with the multimodal-first class order Qwen3.5 requires
(`AutoModelForImageTextToText`, falling back to `AutoModelForCausalLM`) — so
*loading* another model generally works; the real constraints are that the
dictionary must be trained on **that** model's `W_U`, and that the DLA /
all-layer scripts additionally assume the Qwen3.5 submodule layout
(`model.model.language_model`, `model.model.norm` — they fail with explicit
messages on other architectures).

To add a model:

1. Write `configs/models/<your-model>.yaml` — match the existing files
   ([`pythia-160m.yaml`](../configs/models/pythia-160m.yaml),
   [`qwen35-2b.yaml`](../configs/models/qwen35-2b.yaml)): `model.hf_id` +
   `model.slug`, the `model.arch` shape (`hidden_size`, `vocab_size`), and the
   `extraction` block (`corpus`, `corpus_config`, `corpus_split`, `n_prompts`,
   `max_len`, `dtype`, `apply_token_mask`).
2. Re-run extraction + training. `find_lm_head` walks five canonical
   attribute paths and works for plain causal LMs as well as multimodal
   ones (Qwen-VL, etc.).
3. Models with output softcaps (Gemma family) need the softcap captured
   at extract time. `extract_model_readout.py` runs the detector and
   bails with `exit(2)` if anything looks suspicious; the manifest
   records what it found.

## 6. Where results land

```
results/
├── <model>/                             extract_model_readout.py
│   ├── hLN.pt                           {W_U_orig, h_LN} payload (--out)
│   └── hLN_manifest.json                <stem>_manifest.json
├── <run_dir>/                           train_readout_sae_from_config.py (--out-dir)
│   ├── checkpoint.pt
│   ├── config.yaml
│   ├── metrics.json
│   ├── history.json
│   ├── compute_meta.json
│   ├── feature_labels.json
│   └── DONE
└── <run_dir>/metrics_task_fidelity.json task_fidelity_evaluation.py (in --cell-dir)
```

Repo doesn't ship a `results/` symlink — point it wherever you keep big
artefacts (see [`REPRODUCE.md`](REPRODUCE.md#results-directory)).

## 7. Filename codename glossary

A few internal codenames recur in config and artifact filenames:

- **`result1`** — the Result-1 five-model query-fidelity study (e.g.
  `configs/registries/result1_query_fidelity_cluster.yaml`).
- **`exp2`** — Experiment 2, the selected-SAE fidelity manifest (e.g.
  `configs/registries/exp2_selected_sae_checkpoints.yaml`).
- **`qwen2b`** — historical Qwen nickname used as a results-dir / artifact
  prefix; it maps to the Qwen-3.5-2B model config (see the README, which
  explains `qwen2b`).
- **`proto_token_lens` / `proto_lens`** — the project's historical codename
  (predating the "Sparse Readout Prism" name); it means Sparse Readout Prism and
  survives in some archive/results paths and a few output-dir defaults.
