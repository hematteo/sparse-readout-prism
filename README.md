# Sparse Readout Prism

[![ci](https://img.shields.io/badge/ci-pytest-blue)](.github/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)

Code and configs for the paper
**"Sparse Readout Prism: A Sparse LM-Head Basis for Logit-Lens Readouts"**
(preprint forthcoming).

Sparse Readout Prism factorizes language-model unembedding rows into a
dictionary of reusable **readout features**, then decomposes selected
vocabulary logits and linear logit contrasts into signed feature
contributions plus an explicit residual:

```
   h · W_U[token]              ≈ base + Σ_i z_i (h · d_i) + residual
   h · (W_U[A] − W_U[B])       ≈ Σ_i (z_{A,i} − z_{B,i}) (h · d_i) + residual
```

Logit lens asks which token a hidden state reads out as. Sparse Readout
Prism asks **which features of the unembedding row made that token score
high**, with a per-query fidelity gate that withholds the explanation when
the sparse approximation fails to preserve the held-out logit or margin.

The accompanying paper is distributed separately from this code release; see
the [Citation](#citation) section (and [`CITATION.cff`](CITATION.cff)) for the
reference.

## Selected dictionaries (paper results)

Headline readout-replacement fidelity of the selected dictionaries at their
strongest operating point (32× width, k = 256), on held-out hidden states —
`rowEV` is row-reconstruction explained variance, `top1` is top-token agreement
after replacing `W_U` with its reconstruction, `KL` is the readout KL in bits:

| Model | d_features | rowEV | top1 | KL (bits) |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 32768 | 0.877 | 0.891 | — |
| Qwen3.5-2B | 65536 | 0.847 | 0.887 | — |
| Qwen3.5-9B | 131072 | 0.857 | 0.900 | 0.105 |

Full per-model / per-width / per-`k` numbers (including the fixed-budget k = 128
points and the provisional Gemma results) are in
[`configs/registries/exp2_selected_sae_checkpoints.yaml`](configs/registries/exp2_selected_sae_checkpoints.yaml);
see Appendix K of the paper.

---

## Install

Requires Python 3.11 or 3.12.

```bash
uv sync                 # or: pip install -e .
```

`uv sync` is recommended — it installs from the pinned `uv.lock` and picks
the default PyTorch wheel for your platform (CPU/MPS on macOS, CUDA on Linux).
If your GPU needs a specific CUDA build, pin `torch` to a `download.pytorch.org`
index in `[tool.uv.sources]` locally and re-lock.

## Quickstart (no data, no GPU, ~1 s on CPU)

This whole section is also a runnable notebook,
[`notebooks/demo.ipynb`](notebooks/demo.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hematteo/sparse-readout-prism/blob/main/notebooks/demo.ipynb),
which additionally ranks the contributing features and shows their labels.

No pretrained dictionary download is needed to exercise the method end-to-end.
The training → evaluation → feature-labelling pipeline runs on a self-contained
synthetic config:

```bash
uv run python scripts/train/train_readout_sae_from_config.py \
    --config configs/smoke.yaml --out-dir results/smoke_demo
```

This writes a trained `checkpoint.pt`, `config.yaml`, `metrics.json`,
`history.json`, and `feature_labels.json` under `results/smoke_demo/` (not
checked in; the `results/` tree is gitignored).

Now decompose a hidden state's score for one chosen token against that
dictionary — pure Python, still no model download:

```python
from sparse_readout_prism import decompose_token_logit, load_factorizer, preprocess_rows
from sparse_readout_prism.data import make_synthetic_data

# The smoke run trained on this synthetic readout (seed 7, d_model=128);
# regenerate it so the shapes match the checkpoint.
W_U, hidden = make_synthetic_data(seed=7)                 # W_U: (vocab=1024, d_model=128)
h = hidden.reshape(-1, W_U.shape[1])[0]                   # one final-norm hidden state
row_mean, row_norms, rows_normalized = preprocess_rows(W_U)

# Rebuild the trained factorizer from its checkpoint (build + load + eval, one call).
sae = load_factorizer("results/smoke_demo/checkpoint.pt", freeze=True)

token_id = int((h @ W_U.T).argmax())                     # the token this h reads out as
d = decompose_token_logit(
    h              = h,
    W_row          = W_U[token_id],
    row_mean       = row_mean,
    row_norm       = row_norms[token_id],
    row_normalized = rows_normalized[token_id],
    model          = sae,
    k              = sae.k,
)

# Exact identity: original_logit == base_term + feature_sum + residual_term
print(f"{d.original_logit:.3f} = {d.base_term:.3f} + {d.feature_sum:.3f} + {d.residual_term:.3f}")
print(f"active features: {d.active_feature_indices.tolist()}")
```

This repo ships no figure-rendering code; the paper's figures are rendered
separately in the paper's LaTeX source from the committed metrics.

## Decomposing a real model's logits

The same call works on any HuggingFace causal LM — you just need a dictionary
trained on **that model's** `W_U`. The selected paper dictionaries are published
on the Hugging Face Hub at
[`matteohe/sparse-readout-prism`](https://huggingface.co/matteohe/sparse-readout-prism)
(`<model>/<operating_point>/checkpoint.pt`); download one with
`huggingface_hub.hf_hub_download`, or train your own
(`scripts/data/extract_model_readout.py` + the trainer — Pythia-160M is
CPU-feasible; see [`docs/REPRODUCE.md`](docs/REPRODUCE.md)). The synthetic
smoke checkpoint above is **not** shape-compatible with a real model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_readout_prism import decompose_token_logit, load_factorizer, preprocess_rows

# 1. Get W_U and a final-norm hidden state from the model the dictionary was trained on.
name = "EleutherAI/pythia-160m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, output_hidden_states=True).eval()
W_U = model.get_output_embeddings().weight.detach()       # (vocab, d_model)
inp = tok("The capital of France is", return_tensors="pt")
out = model(**inp)
h = out.hidden_states[-1][0, -1].detach()                 # last-position, post-LN
token_id = int(out.logits[0, -1].argmax())

# 2. Preprocess W_U the same way training did (center + per-row normalize).
row_mean, row_norms, rows_normalized = preprocess_rows(W_U)

# 3. Load the matching trained dictionary and decompose.
#    For a published model, pull the dictionary straight from the Hub, e.g.:
#      from huggingface_hub import hf_hub_download
#      ckpt = hf_hub_download("matteohe/sparse-readout-prism", "qwen3.5-2b/k256_32x/checkpoint.pt")
#    For pythia-160m here, train one locally first (see docs/REPRODUCE.md).
sae = load_factorizer("checkpoint.pt", freeze=True)       # trained on this model's W_U
d = decompose_token_logit(
    h              = h,
    W_row          = W_U[token_id],
    row_mean       = row_mean,
    row_norm       = row_norms[token_id],
    row_normalized = rows_normalized[token_id],
    model          = sae,
    k              = sae.k,
)
print(f"{d.original_logit:.3f} = {d.base_term:.3f} + {d.feature_sum:.3f} + {d.residual_term:.3f}")
```

For paired margins, family contrasts, target-vs-vocabulary-mean, and
winner-vs-competitors decompositions — and for the per-query fidelity gate
used in the paper — see [`evaluate.py`](src/sparse_readout_prism/evaluate.py)
and [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Repository layout

```
src/sparse_readout_prism/         core library
  factorizers.py                  TopK / Matryoshka / JumpReLU / Gated row factorizers
  decompose.py, evaluate.py       linear decomposition + fidelity gates
  data.py, train.py               dataset assembly + training loop
  runner.py                       train -> evaluate -> label-features pipeline
  paths.py                        path helpers
  token_display.py, utils.py      token/text display strings + shared IO helpers
  research/                       flat helpers shared by several scripts/ entry
                                  points (Qwen readout + query-decomposition
                                  toolkits, prompt banks, registry, run IO);
                                  logic used by a single script stays inline in
                                  that script

scripts/                          self-contained CLI entry points
  train/, run/, eval/             SAE training, experiment suites, evaluation
  analyze/, data/                 component inspection, data prep
  figures/                        compute + persist the metrics behind the paper
                                  figures (CSV/JSON example panels, lens grids,
                                  decompositions); no figures are rendered here

configs/
  models/, sweeps/, registries/   model configs, sweep grids, paper-selected SAEs

data/
  query_banks/                    curated prompt banks (paper inputs)
  appendix/                       audited literals behind the Appendix K tables
  audit/                          feature-label audit annotations + counts (Appendix L)

notebooks/demo.ipynb              the Quickstart as a runnable notebook (Colab-ready)

tests/                            pytest suite (identity + schema; no GPU, no model loads)
docs/                             REPRODUCE.md (figure → script + config map),
                                  DATA.md (artifact layout), THIRD_PARTY.md
```

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Figure / table → script + config map. Hardware budget. |
| [`docs/DATA.md`](docs/DATA.md) | Where artefacts live, schema, how to adapt to a different model. |
| [`data/query_banks/README.md`](data/query_banks/README.md) | Paper-input JSONL schema + per-file purpose. |

Pretrained readout-feature dictionaries (the selected SAEs cited in the
paper) are published on the Hugging Face Hub at
[`matteohe/sparse-readout-prism`](https://huggingface.co/matteohe/sparse-readout-prism)
— 17 dictionaries across 8 base models, laid out as
`<model>/<operating_point>/checkpoint.pt`.
[`configs/registries/exp2_selected_sae_checkpoints.yaml`](configs/registries/exp2_selected_sae_checkpoints.yaml)
records the canonical manifest (model, width, `k`, metrics), and the synthetic
smoke pipeline above reproduces the full workflow without any download.

**Model nicknames.** `qwen2b` is a historical shorthand for one of the
selected Qwen checkpoints, not a model identity. It no longer appears in script
or module names; it survives in generated result paths and figure filenames
(`qwen2b_32x_*`) and in registry / CLI operating-point ids (`qwen2b_k256`), kept
so they match the artifact names cited in the paper. The model configs and the registry manifest are the source of truth for
which checkpoint a run used (see [`configs/models/`](configs/models/) and the
registry). Similarly, `proto_token_lens` / `proto_lens` is another historical
project codename — it predates the "Sparse Readout Prism" name and survives only
in some archive paths and output-dir defaults, not in the method itself. The
paper additionally reports further Qwen / Gemma / Ministral
models — those checkpoints are published on the Hub alongside the others; their
training configs and metrics are listed in the registry manifest.

## Tests

```bash
uv run pytest -q
```

The suite is deliberately small and CI-safe: identity correctness of the
decomposition, schema invariants of the task-fidelity evaluator, and layout
guardrails for the `research/` sub-package. No GPU, no model loads, no
figure rendering.

## Related repository

[`learning-to-read-out`](https://github.com/hematteo/learning-to-read-out) is the
companion release: it studies how the `W_U` readout *forms over pretraining*
(parameter-trajectory crosscoders across checkpoints), where this repo
factorizes the *final* readout into a sparse feature basis for logit-lens
analysis.

## Citation

```bibtex
@misc{he2026sparsereadoutprism,
  title  = {Sparse Readout Prism: A Sparse LM-Head Basis for Logit-Lens Readouts},
  author = {He, Matteo and Shen, William F. and Qiu, Xinchi and Lane, Nicholas D.},
  year   = {2026},
  note   = {Preprint forthcoming; see the repository for the up-to-date reference},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
