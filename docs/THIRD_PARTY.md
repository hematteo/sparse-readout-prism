# Third-Party Code, Data, and Models

This file lists the third-party code, data, and models that the
`sparse-readout-prism` project depends on or redistributes. It is provided as
part of a permanent public release so that downstream users can satisfy the
relevant upstream licenses and citation requirements.

## Code

No third-party source code is vendored in this repository. All Python
dependencies are standard packages installed from PyPI. The main runtime
dependencies (see `pyproject.toml`) are:

- `torch`
- `transformers`
- `accelerate`
- `numpy`
- `scipy`
- `datasets`
- `pyyaml`

`torch` installs from the default PyPI index on every platform (CPU/MPS wheels
on macOS, CUDA wheels on Linux). No alternative wheel index is configured; if
your GPU needs a specific CUDA build, add a local `[tool.uv.sources]` pin and
re-lock (see the comment in `pyproject.toml`).

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

### WikiText-2 (`wikitext`, config `wikitext-2-raw-v1`)

WikiText-2 is the hidden-state extraction corpus referenced by the model
configs in `configs/models/` (fields `extraction.corpus` = `wikitext`,
`extraction.corpus_config` = `wikitext-2-raw-v1`, and
`extraction.corpus_split` = `test`), matching the shipped extractor
(`scripts/data/extract_model_readout.py`). It is downloaded locally and is not
redistributed here.

License: CC-BY-SA 3.0.

Citation: Merity et al. (2016), "Pointer Sentinel Mixture Models" (WikiText).

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
