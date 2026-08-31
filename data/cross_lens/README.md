# Cross-lens prompt banks

Prompt banks for the cross-lens study (paper section "A Fitted Lens Reports the
Language of Its Corpus" and the "Cross-Lens Study Protocol" appendix). Each
record is one **prompt** plus the vocabulary tokens whose transported logits
the readout runner decomposes on the same frozen Qwen3.5-9B hidden states under
two or more fitted lenses. The banks are model-specific in one respect only:
every decomposed surface is a single token under the Qwen3.5 tokenizer.

## Schema

Every record has these keys (`lang_a` / `lang_b` only in the EN-DE bank):

| Key | Type | Description |
|---|---|---|
| `id` | str | Unique prompt id (`ant_01`, `fac_03`, `antonym_de_01`, ...). |
| `group` | str | Prompt family (below). Families starting with `ctrl_` are controls and never pooled into the headline. |
| `concept` | str | The concept the prompt elicits; used to exclude same-concept pairs from the shuffled-pairing null. |
| `prompt` | str | The full prompt fed to the model; the final position is scored. |
| `form_a` | str | The in-context answer form (the language or script the prompt asks for). |
| `form_b` | str | The same concept's canonical form in the other language. |
| `targets` | list[str] | Every surface the runner decomposes: `form_a`, `form_b`, any extra forms, then the nulls. |
| `null_targets` | list[str] | Unrelated tokens (script- or language-matched) for the unrelated-token null floor. |
| `answer` | str | The expected answer (`form_a` without its leading space). |
| `lang_a`, `lang_b` | str | EN-DE bank only: the language of `form_a` / `form_b` (`de` or `en`). |

Each target string is probed by the id of its first token, and the aggregators
compare the dominant readout feature of `form_a`'s decomposition across lenses
on the mid-band layers {21, 24, 26, 29}. Two null comparisons price in loose
matching: the unrelated-token null (`form_a` vs each `null_targets` entry within
one lens) and the shuffled-pairing null (prompt i under lens A vs prompt j of a
different concept under lens B, three seeded draws per prompt).

## Files

| File | Records | What it is |
|---|---|---|
| `cross_lens_prompts_en_zh.json` | 92 | The English-Chinese bank: 80 cross-lens prompts in seven families plus 12 controls. `form_a` is the answer-script form, `form_b` the other-script form; nulls are script-matched. Read by the English- and Chinese-fitted Jacobian lenses (100 and 300 prompts), the two ridge translators, and the Jacobian-vs-ridge comparison. |
| `cross_lens_prompts_en_de.json` | 102 | The English-German bank: 90 cross-lens prompts in seven families plus 12 controls. Both languages are Latin-script, so the surface-language call in the aggregator is lexical and cognates are excluded by construction. Built by `scripts/data/build_cross_lens_de_bank.py`. |
| `cross_lens_prompts_en_de_exclusion_report.json` | - | The builder's report for the EN-DE bank: tokenizer, candidate / kept counts per family, and the 30 of 132 candidates excluded with the reason (`cognate (lev=n)` at normalised edit distance <= 2 after diacritic folding, or `not single-token`). |
| `cross_lens_three_lens_prompts.json` | 2 | One-prompt-per-family bank for the worked example read by three lenses at once: the German antonym prompt `antonym_de_01` (answer `groß`) and its Chinese analogue `ant_01` (answer 大), each with an extended target list covering the English, Chinese and German answer forms plus two nulls. |

### Families

EN-ZH bank (`cross_lens_prompts_en_zh.json`): `antonym_zh` (14), `cloze_en`
(10), `cloze_zh` (18), `exemplar_zh` (8), `fact_zh` (10), `trans_en2zh` (10),
`trans_zh2en` (10); controls `ctrl_digit` (6), `ctrl_propn` (6). The antonym
family includes the prompt used in the Jacobian-lens paper's multilingual
illustration ("the opposite of 小").

EN-DE bank (`cross_lens_prompts_en_de.json`): `antonym_de` (9), `cloze_de`
(14), `cloze_en` (15), `exemplar_de` (10), `fact_de` (8), `trans_de2en` (17),
`trans_en2de` (17); controls `ctrl_digit` (6), `ctrl_propn` (6).

Controls (`ctrl_digit`, `ctrl_propn`) have an expected surface form that does
not depend on language, so no lens-dependent flip is expected; they are
reported separately.

## Provenance

The EN-ZH bank was hand-curated: seven families of short prompts whose answer
is a common concept with a single-token realisation in both scripts, nulls
drawn from unrelated everyday nouns. The EN-DE bank was generated from a
candidate list by `scripts/data/build_cross_lens_de_bank.py` under three rules
fixed before any lens was fitted (single-token surfaces under the Qwen
tokenizer with context-appropriate spacing, cognate exclusion at normalised
Levenshtein <= 2, round-robin language-matched nulls); the exclusion report is
shipped alongside. The three-lens bank reuses `ant_01` and `antonym_de_01`
verbatim with the extended target lists used for the worked example.

The C4 fitting corpora for the lenses (seeded English, Chinese and German
prompt dumps from `scripts/run/fit_jlens.py prompts`) are not committed, since
raw C4 web text can contain third-party PII; regenerate them with the commands
in that script's docstring.

## Used by

- **`scripts/run/run_cross_lens_readouts.py`** reads a bank under one fitted
  lens and writes the per-layer readout dump (top-5 tokens, per-target
  decompositions) that every aggregator consumes.
- **`scripts/eval/aggregate_cross_lens_en_zh.py`** and
  **`scripts/eval/aggregate_cross_lens_en_de.py`** compute agreement rates, null
  floors and the surface-language split from two dumps and the bank.
- **`scripts/analyze/cross_lens_three_lens_prompt.py`** tabulates one prompt
  under several lenses from the three-lens dumps.
- **`scripts/figures/compute_cross_lens_shared_feature.py`** extracts the
  shared-feature numbers behind the two-lens figure for one prompt and layer.

See [`docs/REPRODUCE.md`](../../docs/REPRODUCE.md) for the artifact-to-script
mapping and the exact command lines.

## Adapting the banks

To test a new language pair or lens construction, write a JSON with the same
schema (`{"note": ..., "prompts": [...]}`), check that every entry of `targets`
is a single token under your tokenizer, and keep `concept` filled so the
shuffled-pairing null can exclude same-concept pairs. The EN-ZH aggregator's
surface call is by script (Latin vs CJK) and the EN-DE aggregator's is lexical;
a pair that shares a script needs the lexical variant with its word lists
adapted.
