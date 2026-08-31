# Literature §4 concerns — latent language, multilingual features, lens reliability

Compiled 2026-08-02 by web search. **Verify every entry against the actual paper
page before writing a `\cite{}`.** arXiv IDs are as returned by search; venue/year
for the 2026 preprints is unconfirmed except where noted. Nothing here has been
added to `references.bib`.

Confidence key: **[V]** = fetched/confirmed abstract; **[S]** = search snippet only.

---

## Tier 1 — The direct target of §4: the latent-language line

This is the literature §4's claim ("conclusions drawn from lens token readings about
the language a model computes in are underdetermined") is actually about. None of it
is currently in `references.bib`.

| Work | Claim | Relation to §4 |
|---|---|---|
| **Wendler, Veselovsky, Monea, West — "Do Llamas Work in English? On the Latent Language of Multilingual Transformers", ACL 2024, arXiv 2402.10588** [V] | Logit-lens on Llama-2 with single-token cloze/translation/repetition prompts. Three phases: intermediate embeddings decode a semantically correct next token but give **higher probability to its English version than the input-language version**, then move into an input-language-specific region. Reads as an English-biased concept space. | **The** target. Its measurement *is* a token-identity comparison — English token vs target-language token probability mass — which is exactly the second ambiguity your thesis names. Note it uses the **plain logit lens** (no fitting corpus), so corpus conditionality does not touch it; the *token-unit* critique does, and directly. |
| **Dumas, Wendler, Veselovsky, Monea, West — "Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers", ACL 2025, arXiv 2411.08745** [V] | Activation patching on word translation. Output language is encoded at an **earlier** layer than the concept; language and concept can be changed independently; patching with the mean concept representation across languages *improves* translation. | The strong form of "language-agnostic concept representations," established causally rather than by lens reading. Cite to show you know the strong claims in this area do **not** rest on token rankings — this is the honest scoping h9Tz will demand. |
| **Schut, Gal, Farquhar — "Do Multilingual LLMs Think In English?", arXiv 2502.15603 (Feb 2025)** [S] | Logit lens on French/German/Dutch/Mandarin: models emit representations close to English for semantically-loaded words before translating; steering is more effective with English-computed vectors. | Second direct target, and it *is* lens-based. Your correction applies most cleanly here. |
| **Zhao et al. — "How Do Large Language Models Handle Multilingualism?" arXiv 2402.18815 (verify ID)** [S] | Proposes understand-in-source → think-in-English → generate-in-target workflow. | Widely cited framing your §4 complicates. |
| **"Beyond English-Centric LLMs: What Language Do Multilingual Language Models Think in?", arXiv 2408.10811** [S] | Directly on the pivot-language question. | Background; one-line cite. |

## Tier 2 — Scoop risk: features shared across languages is already published

**Read this section before writing §4's novelty sentence.** Several 2025–26 papers
already report that sparse features are shared across languages and scripts. Your
§4 currently says "the claim this supports is invariance rather than discovery,"
which is the right instinct — but uncited, a reviewer reads it as rediscovery.

| Work | Claim | Threat level |
|---|---|---|
| **Harrasse, Draye, Pandey, Jin, Schölkopf — "Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders", arXiv 2511.10840 (Nov 2025)** [S] | CLTs + attribution graphs: multilingual LLMs use **highly similar features across languages**, with **language-specific decoding emerging in later layers**; a small set of high-frequency final-layer features linearly encode language identity, and intervening on them substitutes one language for another. | **Highest.** This is close to a mechanistic account of the thing §4 observes, published nine months before your resubmission. Must cite and differentiate: they decompose activations across layers with trained transcoders; you decompose the **head's weights**, corpus-free, and use it to audit an *instrument*. |
| **Karne — "One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations", arXiv 2603.08869, UCRL @ ICLR 2026** [S] | Serbian digraphia as a controlled testbed (same language, two scripts, near-perfect character mapping). Identical sentences in different scripts activate **highly overlapping SAE features**, far above random; changing script diverges representations *less* than paraphrasing within a script; invariance strengthens with scale. | **High.** This is your English–German same-script control's stronger cousin — a cleaner design (meaning held *exactly* constant). Cite it, and consider that Serbian digraphia is a better control than cognate-excluded German. Also connects to the digraphic side-bet already in your notes. |
| **Brinkmann et al. — "Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages", arXiv 2501.06346 (NAACL 2025?)** [S] | Morphosyntactic concepts (number, gender, tense) encoded in feature directions shared across many languages. | Medium. Establishes shared-feature-across-languages as known for grammatical concepts. |
| **"Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders", ACL 2025, arXiv 2505.05111** [S] | Monolinguality metric for SAE features; finds strongly language-specific features, mostly mid-to-late FFN layers. | Medium — the complement of your finding (features that *do* track language). |
| **"Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages", arXiv 2507.11230 (Jul 2025)** [S] | As titled. | Medium. |
| **"Causal Language Control in Multilingual Transformers via Sparse Feature Steering", arXiv 2507.13410** [S] | Steers generated language via SAE features, ~90% success with semantic fidelity preserved. | Low–medium; useful as the causal counterpart. |
| **"Language Lives in Sparse Dimensions...", arXiv 2510.07213** [S] | Interpretable/efficient multilingual control via sparse dimensions. | Low. |

**Positioning that survives this:** the novel claims are (a) the basis is fit from
**weights alone**, so it is prior to any corpus — every work above fits on activations
over a corpus, which is precisely the dependence §4 is auditing; and (b) the object
audited is a **fitted instrument**, not the model. "Sparse features are shared across
languages" is *background* for you now, not a finding.

## Tier 3 — Lens reliability: allies for "lens readings underdetermine computation"

| Work | Claim | Use |
|---|---|---|
| **Nadaf — "Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens", arXiv 2604.02608** [S] | 4,032 pairs, 12 tasks, 6 models: function-vector steering succeeds **even when the logit lens cannot decode the answer at any layer**; high-accuracy FVs project to incoherent tokens. One case: 0.880 steering accuracy at L4 vs peak lens readability 0.056. | **Strong ally.** Independent evidence that lens token readings underdetermine computation, from a different direction (steering works where reading fails; yours is reading changes where computation doesn't). Cite in §4's opening or Related Work. |
| **Belrose et al. — tuned lens, arXiv 2303.08112** (already cited) | Per-layer affine translators trained to minimize KL to the final distribution. | You already cite it; add the point that the translator is **trained on a corpus**, which is the dependence you exploit. |
| **"LogitLens4LLMs", arXiv 2503.11667** (already in bib) | Documents logit-lens failure on modern models; top-1 for BLOOM is often the *input* token in early layers; basis drift. | Already cited — but the basis-drift/failure content is directly usable in §4's motivation. |
| **"Indic-TunedLens", arXiv 2602.15038** (already in bib) | Tuned lens for Indian languages. | Already cited; the multilingual-lens practice you are correcting. |
| **Liu & Han — "ICA Lens: Interpreting Language Models Without Training Another Dictionary", arXiv 2606.11722 (Jun 2026)** [V] | ICA recovers interpretable, token-selective directions without training an overcomplete dictionary; argues ICA is "an efficient and complementary first lens," not a weak baseline. | **Baseline risk.** A reviewer may ask why ICA on `W_U` is not in Table 2 alongside PCA-256. Cheap to add; cheaper than being asked. |

## Tier 4 — Script and orthography (relevant to the EN–DE control)

| Work | Claim |
|---|---|
| **"The Latin Substrate: How Language Models Represent and Mediate Script Choice", arXiv 2605.31363 (May 2026)** [S] | Uses the **logit lens** to show consistent latent romanization during transliteration; scripts become increasingly separable across layers; a linear steering direction flips output script preserving semantics; a small set of **late-layer attention heads causally mediate script choice** and transfer across unrelated languages. Directional asymmetry: non-Latin output from a compact gate, Latin output diffuse. |
| **"Multilingual Language Models Encode Script Over Linguistic Structure", arXiv 2604.05090** [S] | As titled — script dominates linguistic structure in encoding. |
| **"Linear Script Representations in Speech Foundation Models Enable Zero-Shot Transliteration", ACL Findings 2026** [S] | Adjacent; linear script directions. |

The Latin Substrate result matters for your framing: if late-layer heads gate script
choice, then a lens reading a *pre-gate* state and reporting a Latin token is
consistent with your account and theirs. Cite it as convergent mechanism rather than
competition.

## Tier 5 — Sparse coding of embeddings/weights (novelty positioning)

Mostly already covered by your Related Work (`murphy2012nnse`, `faruqui2015sparse`,
`subramanian2018spine`, `arora2018polysemy`, `braun2025parameter`,
`gao2025weightsparse`). Gaps found:

- **"Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval", EMNLP 2025** [S] — SAEs over dense embedding spaces.
- **"Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit", arXiv 2512.10092** [S] — SAE decomposition of dense embeddings into described latents.
- **k-sparse autoencoders on static word2vec/GloVe embeddings** [S] — verify the specific citation; it is the nearest classical precedent to sparse-coding a vocabulary matrix and is worth one line in Related Work.

None of these factorize a trained transformer's LM head against a selected-score
reconstruction target, so the core novelty claim holds. But the Related Work sentence
should name the embedding-SAE line explicitly rather than only the 2012–2018 static
work, or it looks like the recent precedent was missed.

---

## What this changes about the plan

1. **Tier 1 is non-negotiable.** Wendler, Dumas, and Schut must appear, and §4 must
   state precisely which of them it touches. Wendler and Schut use lenses; Dumas uses
   patching and is *not* refuted by anything you show.
2. **The strongest version of Track A is now specific:** reproduce Wendler's task
   design (single-token cloze/translation, non-English prompts) and re-run his
   English-token-vs-target-token measurement in the feature basis alongside the token
   measurement. If `large` and 大 load on one readout feature, his phase-2 evidence
   for an English-biased concept space is ambiguous between "English pivot" and
   "shared feature, one surface resolution." That is a headline, it needs no fitted
   lens, and the artifacts are public (epfl-dlab/llm-latent-language).
3. **Tier 2 forces a rewrite of §4's novelty sentence.** "Same feature across
   languages" is established. Your contribution is the weight-only, corpus-free basis
   and the instrument audit. Say that explicitly, with citations, or a reviewer says
   it for you.
4. **Harrasse et al. (2511.10840) is the single most dangerous omission** — closest
   mechanism, published, uncited.
5. **Consider adding ICA to the Table 2 baseline grid.** One more row; pre-empts the
   "why not a cheaper decomposition" question that ICA Lens makes newly available.
