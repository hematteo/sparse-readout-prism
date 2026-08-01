# Resubmission Change Summary — ARR (prev. submission 10103, May 2026 cycle)

Draft for the OpenReview resubmission note. Audience: the previous AC and
reviewers. Appendix letters and section numbers verified against the
compiled PDF on 2026-08-01; re-check both if appendices or sections move.

---

We thank the reviewers and AC for the May-cycle reviews. The metareview asked
for three things: (1) a demonstration of what SRP enables *systematically*
beyond token rankings, (2) incorporation of the discussion-phase experiments
into the manuscript so they can be properly reviewed, and (3) early
definitions of the paper's core objects. This revision restructures the paper
around those requests. Every experiment reported during the discussion phase
now has a named manuscript location, and all quantitative claims have been
re-verified against the underlying result artifacts.

## Response to the metareview

| Metareview request | Revision |
|---|---|
| "What new interpretations can be done in a systematic way?" — e.g. features aligning with different senses | New top-level Section 4, *The Reported Token Can Follow the Lens Corpus*. Its opening paragraph presents the same-token/different-sense analyses (scoped explicitly as worked, audited case analyses); the body of the section is a controlled quantitative study: English- and Chinese-fitted Jacobian lenses report different surface languages on identical hidden states (script divergence on 9/10 factual-recall prompts) while the same SRP feature stays dominant under both readings on 77/80 prompts (95% CI [0.90, 0.99]), with chance floors at 0/159 (within-lens unrelated tokens) and 0/240 (shuffled pairings) and language-neutral controls behaving as expected. Full protocol and per-case results: Appendix K. |
| Discussion-phase experiments "should be included in a new revision" | All are now in the manuscript: simpler-geometry baselines across six models (§3, Table 2; grids in Appendix H); held-out grouping recovery (§3; Appendix I); cross-seed stability (§3; Appendix I); six-model intervention validation (§3, Table 3; Appendix J); blinded label audit (§3; Appendix L); matched-KL editing comparison including all cross-model losses (§3; Appendix N); error-tail and margin-stratified analyses (Appendix H); sparsity/dead-feature checks (Appendix E). |
| Define contrast/readout early | *Readout*, *readout score*, and *contrast* are defined in the first paragraph of the Introduction and used consistently throughout. |

One clarification: the metareview summarized the headline metric as "90% of
the variance of the observations." The paper's headline metrics are sign
agreement (≥0.91 across the suite) and sign-preserved low-error coverage
(0.72–0.81 on the six gated models); we have made both unmissable in the
abstract and Table 1/2 captions.

## Structural changes

- **Thesis and contributions.** The paper is reframed around one claim —
  token identity is too coarse a unit for interpreting vocabulary readouts —
  with three contributions (interface; distinct, reproducible, causally
  predictive structure; cross-lens corpus-conditionality finding), stated on
  page 1. The beyond-token-identity capability is carried by the thesis
  prose rather than claimed as a separate contribution.
- **Evidence as one argument.** Section 3 now runs fidelity → distinctness
  (direct-geometry alternatives) → reproducibility → intervention relevance,
  with the evaluation design (model split, contrast banks) folded into its
  opening.
- **Demotions.** Feature-resolved DLA, the lexical-edit protocol, the score-
  family grid, and additional qualitative displays moved to appendices; the
  main text keeps one-paragraph summaries with scoped claims.
- **Scope statements.** The 8-model reconstruction suite vs 6-model
  interpretation gate is explicit in Table 1; edit claims print the
  cross-model losses; intervention claims are worded as local readout-side
  predictions; the label audit is described as a blinded same-model
  reproducibility audit, not human annotation.

## Point-by-point (reviewer concerns → revision)

| Concern (reviewer) | Revision |
|---|---|
| Novelty vs a standard TopK SAE (j4yY Q3, e1hF W1, pZ3j W1) | Contribution split stated on p.1: TopK is the fitting primitive; the contribution is the factorized readout object and selected-score interface. Distinctness evidence: SRP leads the strongest of six direct-geometry alternatives by 8.9–17.3 coverage points on all six gated models (Table 2); held-out same-recipe dictionaries recover 72–75% of the reproducible grouping core vs 43–49% (row-kNN) and 21–25% (clustering). |
| No downstream/causal validation (j4yY W1, Q6) | §3 intervention study: predicted contributions vs realized readout-side changes at r²=0.83–0.93 across six models (random-direction controls ≤0.06), slopes reported as measured with the small-model overestimation disclosed (Table 3). |
| What analysis do rankings not support? (pZ3j W4, j4yY Q1) | §4 (The Reported Token Can Follow the Lens Corpus), the cross-lens study; the Reading paragraph states the narrow conclusion and its safety relevance. |
| Direct W_U clustering/kNN might match (e1hF W1) | Table 2 + grouping recovery above; capacity-matched k-means (same D, k) included. |
| Seed stability of explanations (e1hF W2) | §3 stability paragraph + Appendix I: 100% of contrasts reproduce above the null's p90 at both widths; 87–91% of used feature groups have counterparts; instability is feature splitting; claims scoped within-recipe with cross-recipe variation disclosed. |
| Edit controls at matched KL (e1hF W3) | §3: SRP beats mean-row and PCA controls at every matched KL on Qwen3.5-2B (12/12 paired, significant); 0.8B/9B losses printed; both candidate explanations reported as tested and rejected (Appendix N). |
| k=128/256 not sparse enough (e1hF) | Appendix E: k=32/64 retraining leaves 64–69% of features dead; k=256 = 0.4% active fraction. |
| Label subjectivity / ambiguous labels (pZ3j W2) | Blinded three-run audit, κ=0.87, 93.4% unanimity; 76/99 coherent, 19 token-form, 4 mixed; 81.8% agreement with the author audit; protocol in Appendix L. |
| Hyperparameter sensitivity (pZ3j W3) | Two widths × three seeds throughout the stability analyses + Appendix E. |
| Median hides tails (j4yY Q7) | Mean/p95/max and margin-stratified errors in Appendix H; near-tie behavior explained structurally in §3. |
| Offset and residual roles (j4yY Q4) | Method §2: offset cancellation for zero-sum contrasts; residual is exact but not orthogonal (overcomplete basis), stated with the reading rule. |
| Contrast selection vs task correctness (j4yY Q5) | §3 (Contrast banks): diagnostic vs task-grounded contrast split; benchmark-derived contrasts summarized in §3. |
| Definitions arrive late (j4yY, AC) | Page-1 definitions; notation split (floored ρ vs unfloored ρ̃) with distinct symbols. |

## New since the discussion phase

- The cross-lens study was completed to its pre-registered scope (80 prompts,
  7 families, controls) and written up in full (§4, Appendix K).
- Compute/hardware and pinned package versions added (Appendix G),
  addressing the two unchecked reproducibility-checklist items.
- Code, trained dictionaries, contrast banks, and the audit protocol
  accompany the submission (links withheld for review).

## Verification note (internal; drop before posting)

All main-text numbers re-verified 2026-07-31 against
`REBUTTAL_RESULTS.md` and raw artifacts (`causal_*/summary.json`,
`exp4v2_*/baseline_by_method.csv`, `cross_lens_v2_summary.{json,txt}`,
stability JSONs). fact-recall 9/10 script divergence and 9/10 feature
agreement are distinct quantities, both correct per the artifact.
