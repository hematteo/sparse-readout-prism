# Query banks

Curated prompt + target-pair JSONL files used as paper inputs. Each line is
one **case**: a prompt, the position to score, and a pair of vocabulary
tokens `(target_a, target_b)` that the readout pipeline decomposes against
each other.

Banks here are model-agnostic (the same JSONL is fed to every model the
paper evaluates). The Result-1 fidelity tables consume the curated banks
unmodified. The large C4-sampled `model_native` slice is not redistributed in
this repo (its raw web text can contain third-party PII); regenerate it
locally from the pinned revision (see below).

## Schema

Every record has these keys:

| Key | Type | Description |
|---|---|---|
| `case_id` | str | Globally unique case id; safe to use as a primary key. |
| `base_case_id` | str | Group id shared by sibling variants (paraphrases, token swaps). |
| `variant_id` | str | Per-variant label within a base case (`default` for the canonical record). |
| `bank` | str | The bank file family — `curated_ab` / `case_candidates` / `model_native`. |
| `family` | str | Semantic family within the bank (e.g. `current_stale`, `c4_model_native`). |
| `prompt` | str | The full prompt fed to the model. |
| `scored_position` | str | Position to read the hidden state from (`last` for end-of-prompt, else an integer). |
| `target_a`, `target_b` | str | Vocabulary tokens to decompose against each other. |
| `target_a_family`, `target_b_family` | str/null | Optional family labels for paired-margin / family-contrast queries. |
| `expected_side` | `"A"` / `"B"` | Which target should win in the correct decomposition. |
| `expected_is_task_label_only` | bool | True if the gold is only the task label, not the model's prediction (excluded from Result-1 scoring). |
| `notes` | str | Free-text annotation (curator note, exclusion reason, ...). |

The C4-sampled `model_native` slice adds C4-provenance fields (`source_dataset`,
`source_config`, `source_split`, `source_revision`, `source_doc_sha1`,
`sampling_seed`) so the slice is reproducibly resolvable; see the matching
`*_c4_manifest.json` for the dataset revision pin and build command.

The C4 slice itself is **not committed** (its raw C4 text can contain
third-party emails and phone numbers). Regenerate it deterministically from
the pinned revision:

```bash
uv run python scripts/data/build_query_banks.py \
    --native-source c4 \
    --c4-revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 \
    --seed 0 --max-native 10000
```

C4 is distributed under ODC-BY 1.0 (subject to Common Crawl terms); see
`THIRD_PARTY.md` for data and model attribution.

## Files

| File | Records | What it is |
|---|---|---|
| `qwen_gemma_result1_curated_ab.jsonl` | 320 | **Canonical**: hand-curated A/B pairs used in the §4 main fidelity table. |
| `qwen_gemma_result1_case_candidates.jsonl` | 40 | Reduced candidate set used as contrast inputs for the null-baseline comparison. |
| `qwen_gemma_result1_model_native_prompts.jsonl` | ~520 | Hand-built "model-native" prompts; run by the fidelity runners alongside the two curated banks. |
| `qwen_gemma_result1_model_native_prompts_c4.jsonl` | 10,000 | C4-sampled "model-native" prompts behind the §4.5 distributional readout metrics. **Not shipped**: regenerated locally from the pinned C4 revision (command above), so no raw C4 text (which can contain third-party PII) is redistributed. Provenance is pinned in `*_c4_manifest.json`. |
| `qwen_gemma_result1_model_native_prompts_c4_manifest.json` | – | Build-time manifest (C4 revision, filters, sampling seed, command) for the C4 slice above. |

## Used by

- **`scripts/data/build_query_banks.py`** — bank builder (writes the three banks from prompt templates; the large C4 slice with `--native-source c4`).
- **`scripts/run/run_query_fidelity_bank.py`** — Result-1 five-model query-fidelity runner; reads these `.jsonl` banks directly (via `--bank-dir`), capturing hidden states with a forward hook.
- **`scripts/eval/task_fidelity_evaluation.py`** — §4 main fidelity gate; consumes a separately-built `.pt` margin/query bank (`{h, margin_items, query_items}`) derived from these cases plus extracted hidden states — **not** the `.jsonl` directly (passing a `.jsonl` to its `--bank` raises `UnpicklingError`; see `docs/reproducing.md` §1).
- **`scripts/run/run_readout_baseline_comparisons.py`** — null-baseline runs against `curated_ab` + `case_candidates`.

See [`docs/reproducing.md`](../../docs/reproducing.md) for the exact
figure-→-metrics-script mapping.

## Adapting the banks

To run the pipeline on your own prompts, write a JSONL with the same schema
above (the optional `*_family` and provenance fields can be omitted) and drop it
in a bank directory that the `.jsonl`-consuming runners read via `--bank-dir`
(`run_query_fidelity_bank.py`, `run_readout_baseline_comparisons.py`). At minimum
each record needs `case_id`, `prompt`, `target_a`, `target_b`, `expected_side`,
`scored_position`. (The `.pt` bank that `task_fidelity_evaluation.py --bank`
consumes is a *different* artifact — built from these cases plus extracted hidden
states, see `docs/reproducing.md` §1 — not the `.jsonl` itself.)
