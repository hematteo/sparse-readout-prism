# Third-Party Code, Data, and Models

This file lists the third-party code, data, and models that the
`sparse-readout-prism` project depends on or redistributes. It is provided as
part of a permanent public release so that downstream users can satisfy the
relevant upstream licenses and citation requirements.

## Code

No third-party source code is vendored in this repository. The Python
dependencies are installed from PyPI, with one exception: the Jacobian-lens
reference implementation (`jlens`, below) is installed from its git repository
through an optional extra. The main runtime dependencies (see `pyproject.toml`)
are:

- `torch`
- `transformers`
- `accelerate`
- `numpy`
- `scipy`
- `datasets`
- `pyyaml`
- `pandas` (BSD-3-Clause; `scripts/eval/analyze_error_tails.py`)
- `scikit-learn` (BSD-3-Clause; `scripts/run/run_wsd_feature_alignment.py`,
  `scripts/analyze/analyze_wsd_classifier_framing.py`)

`torch` installs from the default PyPI index on every platform (CPU/MPS wheels
on macOS, CUDA wheels on Linux). No alternative wheel index is configured; if
your GPU needs a specific CUDA build, add a local `[tool.uv.sources]` pin and
re-lock (see the comment in `pyproject.toml`).

### Jacobian lens (`jlens`)

The cross-lens study (`scripts/run/fit_jlens.py` and the cross-lens scripts
that load its fitted lenses; they import it lazily) uses the reference
Jacobian-lens implementation from <https://github.com/anthropics/jacobian-lens>,
licensed under Apache-2.0. It is not published on PyPI: the optional `lens`
extra (`uv sync --extra lens`) installs it from git, and `uv.lock` pins it to
commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. The rest of the repository
installs and tests without it, and none of its code is vendored here.

## Data

### C4 (`allenai/c4`)

The model-native query bank is sampled from the C4 `en` `validation` split. The
exact dataset revision and the sampling configuration are pinned in the
query-bank manifest (`data/query_banks/*_c4_manifest.json`; the shipped
manifest is `qwen_gemma_result1_model_native_prompts_c4_manifest.json`):

- `source_dataset: allenai/c4`
- `source_config: en`
- `source_split: validation`
- `source_revision: 1588ec454efa1a09f29cd18ddd04fe05fc8653a2`
- `sampling_seed: 0`

License: ODC-BY 1.0, subject to the Common Crawl terms of use.

Citation: Raffel et al. (2020), "Exploring the Limits of Transfer Learning with
a Unified Text-to-Text Transformer" (T5 / C4).

The shipped query bank stores only provenance, not raw text. The 10,000-prompt
C4-sampled model-native slice
(`qwen_gemma_result1_model_native_prompts_c4.jsonl`) is deliberately NOT
committed, because raw C4 web text can contain third-party PII. What ships is
the manifest, which records the build parameters (the pinned C4 revision, the
sampling seed, the text filters, and the exact build command) so the slice is
deterministically reproducible. When the slice is regenerated, each record
carries per-document provenance (document SHA-1 via `source_doc_sha1`, the
pinned `source_revision`, and the `sampling_seed`) rather than relying on a
redistributed copy. The slice is regenerated locally from the pinned revision
with:

```bash
uv run python scripts/data/build_query_banks.py \
    --native-source c4 \
    --c4-revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 \
    --seed 0 --max-native 10000
```

so no raw C4 text is redistributed in this repository.

The cross-lens study fits its lenses on seeded prompt dumps streamed from the
C4 `en`, `zh` and `de` configs (`scripts/run/fit_jlens.py prompts --c4-config
{en,zh,de}`; `train` split, streaming, `shuffle(seed=0)`, 1,000 prompts per
language). The dumps are raw C4 text and are not shipped; they carry the same
license and citation as above (ODC-BY 1.0, subject to the Common Crawl terms of
use).

### WikiText-2 (`wikitext`, config `wikitext-2-raw-v1`)

WikiText-2 is the hidden-state extraction corpus referenced by the model
configs in `configs/models/` (fields `extraction.corpus` = `wikitext`,
`extraction.corpus_config` = `wikitext-2-raw-v1`, and
`extraction.corpus_split` = `test`), matching the shipped extractor
(`scripts/data/extract_model_readout.py`). It is downloaded locally and is not
redistributed here.

License: CC-BY-SA 3.0.

Citation: Merity et al. (2016), "Pointer Sentinel Mixture Models" (WikiText).

### CoarseWSD-20

The sense-labelled evaluation (`scripts/run/run_wsd_feature_alignment.py
--dataset coarsewsd20 --data-root <checkout>`, followed by
`scripts/analyze/analyze_wsd_sense_groups.py` and
`scripts/analyze/analyze_wsd_classifier_framing.py`) reads CoarseWSD-20 from a
local clone of <https://github.com/danlou/bert-disambiguation>
(`data/CoarseWSD-20`). The dataset is not redistributed here; the shipped
outputs are per-model feature bundles and aggregate metrics, not its text.

License: the upstream repository declares no license (no `LICENSE` file, no
license statement in its README, and an empty license field on the GitHub
repository record at the time of this release), so no license can be quoted for
the dataset as given upstream. CoarseWSD-20 is built from English Wikipedia
sentences, whose text is under CC BY-SA. Check the upstream repository before
redistributing the data.

Citation: Loureiro, Rezaee, Pilehvar and Camacho-Collados (2021), "Analysis and
Evaluation of Language Models for Word Sense Disambiguation", *Computational
Linguistics* 47(2), 387-443.

### Benchmark-derived query suites

`scripts/run/run_benchmark_derived_query_suite.py` builds readout-contrast
cases from four public benchmarks, downloaded from the Hugging Face Hub at
runtime. None of their text is redistributed in this repository; the derived
case bank is written under the gitignored `results/` tree.

| Dataset | Hugging Face id | Used for | License |
| --- | --- | --- | --- |
| SQuAD 2.0 | `rajpurkar/squad_v2` (validation) | extractive-QA answer + abstention contrasts | CC-BY-SA 4.0 |
| HotpotQA | `hotpotqa/hotpot_qa` (distractor, validation) | multi-hop QA contrasts | CC-BY-SA 4.0 |
| LegalBench | `nguha/legalbench` (`contract_nli_confidentiality_of_agreement`, test) | yes/no label contrasts | per-task licensing; this task derives from ContractNLI — see the dataset card |
| SecurityEval | `s2e-lab/SecurityEval` (train) | safer-implementation keyword contrasts | see the upstream `s2e-lab/SecurityEval` repository |

Citations: Rajpurkar et al. (2018) for SQuAD 2.0; Yang et al. (2018) for
HotpotQA; Guha et al. (2023) for LegalBench and Koreeda & Manning (2021) for
ContractNLI; Siddiq & Santos (2022) for SecurityEval.

## Models

The project operates on the unembedding (readout) matrices of the base models
below. Only models actually referenced in the configs are listed: the two
standalone model configs in `configs/models/*.yaml` (each with an explicit
`hf_id`) and the additional base models resolved by the five-model fidelity
registry `configs/registries/result1_query_fidelity_cluster.yaml` (each with an
explicit `model_id`). The corresponding training recipes live in
`configs/sweeps/*.yaml`. The Qwen runs standardise on the Qwen3.5 series; earlier
Qwen2.5 pilot configs were removed and are not used.

| Model | Hugging Face id | License |
| --- | --- | --- |
| Pythia 160M | `EleutherAI/pythia-160m` | Apache-2.0 |
| Qwen3.5 0.8B | `Qwen/Qwen3.5-0.8B` | Qwen / Tongyi Qianwen License |
| Qwen3.5 2B | `Qwen/Qwen3.5-2B` | Qwen / Tongyi Qianwen License |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` | Qwen / Tongyi Qianwen License |
| Gemma 4 E2B (instruction-tuned) | `google/gemma-4-E2B-it` | Gemma Terms of Use |
| Gemma 4 E4B (instruction-tuned) | `google/gemma-4-E4B-it` | Gemma Terms of Use |
| Ministral 3 8B Base | `mistralai/Ministral-3-8B-Base-2512` | Mistral License |
| DeepSeek-R1-Distill-Qwen 7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | upstream / base license |
| DeepSeek-R1-Distill-Llama 8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | upstream / base license |

A Llama-3.1-8B base control (the base-vs-distill control for the
DeepSeek-R1-Distill-Llama-8B run) was trained from an off-repo recipe whose
sweep config is not included in this release. It referred to the model only by
slug (`llama31`) without a pinned `hf_id`; the intended base is Meta's Llama 3.1
8B, governed by the Llama 3.1 Community License.

Notes:

- The released readout-feature dictionaries are derivatives of these base
  models' unembedding (output projection) matrices and are therefore subject to
  the corresponding upstream model licenses.
- This applies in particular to the Llama (Llama 3.1, and the Llama-based
  DeepSeek-R1 distill) and Gemma artifacts, whose licenses include specific
  redistribution and naming requirements (for example, the Llama 3.1 Community
  License and the Gemma Terms of Use). Downstream users must comply with those
  upstream terms when redistributing or building on the released dictionaries.
- For Gemma, the prism decomposes the pre-softcap readout; the model's
  post-readout `tanh` logit softcap is recorded in the extraction manifest (see
  `configs/registries/result1_query_fidelity_cluster.yaml`).
- The DeepSeek-R1-Distill models are distilled on top of Qwen and Llama bases.
  Use of the derived dictionaries is governed by the applicable upstream and
  base-model (Qwen, Llama) license terms.
