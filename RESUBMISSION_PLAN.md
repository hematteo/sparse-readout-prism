# SRP Resubmission Plan (post ARR 2026-May metareview)

Supersedes the pre-submission revision plan (written against *simulated*
reviews, now at `../rebuttal/plan/REVISION_PLAN.md`; its Edits 1–10 are folded
in below where still live). Companion source-of-truth for numbers:
`REBUTTAL_RESULTS.md`. Real reviews: `../rebuttal/reviews.md`. Metareview:
`../rebuttal/metareview.md`. Rebuttal text as posted: `../rebuttal/posted/`.

**Outcome being answered.** Metareview (AC myvk, 29 Jul 2026): 2.5 Borderline
Findings, venue suggestion BlackboxNLP. Reviewer finals: j4yY 2.5 (conf 2),
e1hF 3.0 (conf 4), pZ3j 3.5 (conf 3). The AC's three operative asks:

1. **Novelty-as-utility:** "what new interpretations can be done in a systematic
   way with this approach?" — with a concrete suggestion: *disambiguation
   datasets, checking discovered features align with different senses*.
2. **Fold the discussion-phase experiments into the paper** ("over 150 new
   experimental runs (!) … most should be included … to ensure a proper
   reviewing process"). The AC pre-committed to treating these as reviewable
   once they are in the manuscript.
3. **Clarity:** define contrast/readout early (j4yY quoted verbatim).

**Strategic read.** The AC never mentioned the causal validation (r²=0.83–0.93)
or the SC2 cross-lens finding (77/80) — the two results that answer ask #1 —
and misstated the fidelity metric as "90% of variance". Conclusion: rebuttal
evidence that lives only in OpenReview comments does not move this AC. The
revision must put the utility results **in the main text, as named
contributions**, and additionally execute the AC's own suggested test (word-sense
disambiguation at dataset scale) so that ask #1 is answered *in the AC's own
terms*. Everything else is incorporation, scoping, and hygiene.

---

## Workstream A — the systematic-utility section (score-moving; new compute)

The one thing that changes the verdict. Two legs, one new main-text section
(becomes the utility centerpiece; suggested title: "What SRP reads that token
rankings cannot").

### A1. Word-sense disambiguation at dataset scale `[exp — NEW]`

The AC's own acceptance test, executed literally. Protocol sketch:

- Dataset: a standard WSD/sense-tagged resource with single-token-friendly
  targets (CoarseWSD-20 is the natural fit: 20 ambiguous nouns, coarse senses,
  ~10k instances; WiC as secondary). Filter to targets tokenizable in the Qwen
  vocab; report the filter rate.
- For each instance, run the model, take the hidden state at the readout
  layer(s), decompose the target-token logit with SRP, and record the top
  positive feature(s).
- **Claim to test (falsifiable, pre-registered thresholds):** instances of the
  same word in different senses select *different* dominant SRP features, and
  the feature→sense mapping is consistent across instances (measure: cluster
  purity / adjusted Rand between feature-assignment and gold senses, with a
  majority-class and an embedding-kNN baseline).
- Controls: logit-lens top-k token overlap (should NOT separate senses — same
  target token either way, which is exactly the point), activation-SAE features
  at the same layer (the "why not an ordinary SAE" comparison, now on utility
  rather than fidelity), row-kNN groupings.
- Scale across ≥2 gated models. This is inference + analysis only — no
  dictionary retraining. Rough budget: hours, not days, on one A40/consumer GPU.

This turns the paper's opening bug/error-vs-bug/insect vignette into a
dataset-scale quantitative result, i.e. exactly "systematic interpretation".

### A2. Corpus-conditionality invariance (the SC2 J-lens finding) `[exp — extend]`

Promote from rebuttal comment to main-text result; this is the chosen utility
showcase (see `../jlens_critique/jlens_critique.tex` §"How to improve" and the
W1 corpus-conditionality section; memory: jlens-srp-exhibit).

- Core result already in hand: English- vs Chinese-fitted J-lenses report
  different tokens on identical frozen states; the same SRP feature dominates
  both readings in 77/80 prompts (96%, CI [0.90, 0.99]); chance floors 0/159
  within-lens, 0/240 shuffled.
- Rebuttal **promised** an extension to "further language pairs and lens
  families" — deliver it: at least one more language pair (e.g. EN–DE or
  EN–ES on the same Qwen3.5-9B pipeline) and one more transport (tuned lens or
  plain logit lens as the second family). Keep the same pre-registered
  agreement/floor metrics.
- Frame around safety-relevant lens use (working-language audits, injection
  detection) exactly as in General Response 2/3 — but keep the
  no-manufactured-surprise rule: show the no-SRP baseline (token rankings
  alone) beside the SRP resolution in the same figure/table.
- Cite Gurnee et al. 2026 (J-lens) as contemporaneous; do not frame as attack.

### A3. Wrong-reason detection (cheap third leg, from existing data) `[rebuttal-able]`

Old plan's Edit 2, still unshipped: a case where the lens confidently prefers
token A but SRP shows the support is a token-form/tokenizer feature, not a
semantic one ("right for the wrong reason"). Pull from the 19/99 token-form
audit rows + prompt-injection Table 20 material. One figure or boxed example in
the new utility section.

---

## Workstream B — incorporate the rebuttal experiments (the AC's ask #2)

Every row of the General-Response summary table must land in the manuscript.
Per item: content → destination, plus framing cautions from `REBUTTAL_RESULTS.md`.

| # | Rebuttal result | Destination | Cautions |
|---|---|---|---|
| B1 | **Causal validation, 6 models** (gated r² 0.83–0.93, slope ≈1 on 7–8B, 1.3–1.7 on small Qwens, random r²≈0) | New main-text RQ (becomes RQ3; see D1) + new `appendices/causal_contribution_validation.tex` + per-model scatter figure | Report slopes as measured with the interference diagnosis *paragraph* (promised to j4yY); scope = readout-side linear ablation, no re-forward |
| B2 | **Simpler-geometry baselines** (36 comparisons; SRP accepted 0.716–0.812 vs best baseline 0.579–0.665; leads 8.9–17.3pp) | Replace/extend main Table 3 + `appendices/robustness_controls.tex` or new baselines appendix | Six-model suite = §5 top-1≥0.75 gate, Gemma excluded — state the inclusion rule in the caption; unfloored ρ here vs floored ρ elsewhere — never mix in one table (see H2) |
| B3 | **Groupings / LOO core recovery** (held-out SRP 0.72–0.75 vs kNN 0.43–0.49 vs cluster 0.21–0.25, non-overlapping CIs) | Same baselines appendix + one main-text sentence | Quote ONLY the LOO-anchored triple; never the dead 0.203-vs-0.214 table; all stability claims scoped *within-recipe* (see H1) |
| B4 | **Cross-seed stability of explanations** (100% above null p90 both widths; aggregate-direction cosine 0.74–0.79; Spearman 0.73–0.77) | New `appendices/basis_stability.tex`; summary sentence in main text; rewrite the limitations "hyperparameter dependence" paragraph to cite it | Report cutoff-free metrics ALONGSIDE Jaccard (no metric-swapping optics); "IDs reshuffle, explanation doesn't" is the frame |
| B5 | **Feature-group matching** (87–91% of used groups reproduce above null p99; ~40% near-exact; residual instability = splitting) | Same stability appendix | Do NOT stack criteria into "~98%"; direction-check numbers only at cos≥0.5; fix thresholds + add the promised threshold-sensitivity curve BEFORE quoting; rubric-score the below-null tail (currently qualitative only); cite Paulo & Belrose 2501.16615 + 2606.12138 |
| B6 | **Matched-KL edit frontier** (Qwen-2B 12/12 wins, CIs exclude 0; 9B/0.8B LOSE to mean-row; R1-Qwen mixed; Ministral/R1-Llama tokenizer-excluded) | Rewrite RQ4/edits subsection with the frontier figure; scoped claim | The claim is display-model-scoped; report both failed mechanistic explanations (fidelity, width); tokenizer exclusions are scope notes, not bugs |
| B7 | **Blinded label audit** (76/99 coherent, Fleiss' κ=0.87, 81.8% agreement with author audit, disagreements skew coherent) | Replace the author-only audit in the feature-audit appendix; κ + rate into main text RQ1/RQ2 | LLM raters (gpt-5.6-sol ×3, frozen prompt) — describe the protocol honestly; the two *human* blind raters remain an open user task and would strengthen this further |
| B8 | **Aggregate non-cherry-picked audit** over the full A/B bank | Add to the audit appendix | Still OPEN (was "remains open" in results doc) — must be run for the revision or the cherry-picking objection survives |
| B9 | **Sparsity/hyperparameter check** (k=32/64: 64–69% dead features; supports k=256 = 0.4% active; 16x/k128 replications) | Extend `appendices/readout_sae_qwen_sweeps.tex`; one sentence in setup | Answers pZ3j W3 + e1hF's k comment; frame k in *fraction-active* terms |
| B10 | **Tail/error distributions** (mean 0.54 / p95 1.78 / max 5.0 vs 1.67/4.19/9.0; margin-stratified; errors concentrate near indifference) | Table 3 companion (mean/p95/max columns) + margin-stratified figure in appendix | Use the corrected tail phrasing ("concentrated, 15% of contrasts, monotone falloff") — the "entirely at |m|<0.5" version was false and already fixed once |
| B11 | **Task-derived contrasts** (Table 19: 300 contrasts, QA/legal/security, sign 0.93–0.97) + prompt-injection (Table 20) | Promote from appendix to a main-text paragraph in the utility section (A3 uses Table 20) | Already in the submission; promotion only |

## Workstream C — writing edits promised in the rebuttal (j4yY + AC ask #3)

All committed in the posted replies; each is small and must actually happen:

- C1. **Define "readout" and "contrast" in the Introduction** (the AC quoted
  this verbatim — treat as a hard requirement). One crisp paragraph, first page.
- C2. **Motivate logit differences up front** (abstract + intro): the model's
  exact signed preference between candidate outputs; the local decision variable
  of logit-lens/DLA/circuit analysis.
- C3. **Sharpen the activation-SAE boundary** in intro + related work: SRP
  factorizes the *weights* (corpus-free, exists before any prompt), an
  activation SAE factorizes corpus activations; SRP's contribution is the
  reusable readout coordinate system, not the sparse optimizer.
- C4. **Offset b̄ justification** in method (TopK capacity argument + centering
  precedent + exact cancellation in zero-sum contrasts).
- C5. **Residual non-orthogonality** paragraph in method (additive-and-exact
  separation; orthogonality impossible for overcomplete dicts; refit gains
  negligible per App. G.1).
- C6. **Contrast selection vs correctness** split (diagnostic vs task-grounded
  contrasts; SRP explains errors as readily as correct decisions).
- C7. **Reading-rule = selective prediction** framing from the intro onward;
  0.91 is a suite summary, not a per-example bar; scope sentence: lens-family
  interpretability method, not a replacement head.
- C8. **Worked numerical example** — `sections/worked_example.tex` exists but is
  not `\input` in main.tex. Surface it after the local-score subsection (old
  Edit 6).
- C9. Abstract/conclusion polish (old Edit 10): one plain-language "what a
  reader gets over the logit lens" sentence; kill vague phrases.

## Workstream D — main-text restructure (space budget)

Adding A1/A2 + B1 + B6 to a long paper requires cuts. Proposed results
architecture:

- D1. **New spine** (agreed 2026-07-31; replaces the earlier RQ-order sketch).
  Current spine = intro → method → setup → calibration (4 fidelity ¶s) →
  "Readout Analyses" (RQ1 competition, RQ2 reweighting, regime ¶, RQ3 DLA,
  RQ4 edits) — i.e. *define → calibrate → demonstrate*; all post-§4 content is
  demonstration, which is the reviewed weakness. New spine = each section makes
  a falsifiable claim:
  1. Introduction — readout/contrast/logit-diff defined ¶2 (C1/C2);
     contributions: basis · fidelity · causality · findings · edits.
  2. Method — + C4 offset ¶, C5 residual ¶, boxed reading rule, worked
     example \input'd (C8; currently orphaned).
  3. Setup — dictionary family + seeds, ONE ρ convention (H2), six-model
     gate stated once, compute budget.
  4. Calibration: faithful & not reducible to simple geometry — fidelity ¶s
     merged; Table 3 = direct-geometry baselines (B2) + tails (B10); one
     stability ¶ (B4/B5 → appendix).
  5. NEW "Interventional validation: the accounts are causal" —
     5.1 predicted-vs-realized ablations (B1, 6-model scatter, slope
     diagnosis); 5.2 edits at matched KL (B6, scoped, losses printed;
     folded here, NOT a standalone closing section — same interventional
     machinery, and the results must not end on the mixed edit outcome).
  6. NEW "Findings: what SRP reads that token rankings cannot" — closes the
     results on the AC-requested material. A2 cross-lens (in hand) + A1 WSD
     + A3 wrong-reason box + B11 pointer + one compressed case study
     (bug/insect, as the qualitative bridge to A1). Internal lead order
     (A1-first vs A2-first) decided AFTER A1 results exist — A2 must be able
     to carry the section if WSD comes back weak.
  7. Related work (+C3) · Conclusion · Limitations (stability + edit scope).
  Blinded-audit κ (B7/B8) lives as the closing ¶ of §4 — the readability
  gate must precede the findings that rely on labels.
  Section identities: §4 faithful · §5 causal · §6 useful — one falsifiable
  claim each. Space: §5 ≈ 1.25 col + 2 figs, §6 ≈ 1.5 col + fig/table, paid
  by DLA demotion (~0.5), case-study compression (~0.75), fidelity merge +
  rule box (~0.5), score-family ¶ → two sentences.
- D2. **Demote the DLA-composition RQ** (near-trivial by linearity — old Edit 8)
  to a paragraph or appendix; keep only if reframed as something DLA alone
  misses.
- D3. Trim the reading-rule prose in method into a boxed four-check summary
  (old Edit 7) to buy room.
- D4. Compress the current qualitative display examples; the utility section
  replaces most of their function.

## Workstream E — artifacts, checklist, reproducibility

- E1. **Release the artifact bundle** (committed in the rebuttal closing): code,
  trained dictionaries, contrast sets, blinded-audit protocol, seeds. This also
  attacks pZ3j's Datasets=1 and j4yY's Reproducibility=3.
- E2. Fix checklist "No" items: **C1 compute budget/hardware** (report GPU-hours
  + hardware; trivial), **C4 package versions** (pin + list).
- E3. Seeds/determinism statement in-text; confirm released dictionaries
  reproduce every table exactly (old Edit 9). NB: decide the dictionary-family
  question (H1) *first*, then verify reproduction against whichever family
  ships.
- E4. Preprint: metadata said "plan to release a non-anonymous preprint" — if
  still intended, time it per ARR anonymity rules for the target cycle.

## H — Landmines and honest-scoping rules (do not lose these)

- H1. **Paper-dict vs seedvar-dict inconsistency.** The paper's released
  dictionary recovers only 0.444 of the seed-stable core — *below kNN* —
  because cross-recipe variation exceeds within-recipe seed variation. Options:
  (a) make a seedvar-family dictionary the paper's operating dictionary and
  regenerate affected tables; (b) keep the paper dict but scope every stability
  claim as within-recipe and disclose the cross-recipe number in the stability
  appendix. Decide early — (a) is cleaner but touches many tables; (b) is
  cheaper but leaves a probe-able seam. Either way, never quote a stability
  number whose dictionary family doesn't match its comparison.
- H2. **Floored vs unfloored ρ.** Paper's 0.69–0.89 pass range = floored ρ over
  8 models; rebuttal harness = stricter unfloored ρ over 6 gated models
  (0.72–0.81). Pick ONE convention for the revision (recommend: unfloored for
  all baseline comparisons, and re-derive the pass-rate table under it), or
  keep both but label every table explicitly.
- H3. **Edit-claim scope.** Qwen-2B wins 12/12; 9B and 0.8B lose to mean-row
  (significant). The paper must print the losses. Frame: dictionary directions
  beat trivial category directions on the display model; advantage is not
  universal; both candidate explanations tested and rejected.
- H4. **Causal slopes >1 on small Qwens** (1.3–1.7): report as measured with the
  interference hypothesis flagged open — this was promised.
- H5. Do-not-quote list: v1 row-cluster baselines (full-rank artifact), the
  0.203-vs-0.214 groupings table, the "8 dictionaries" count (it's 6), any
  "~98% stability" criterion-stacking.
- H6. **Metareview factual error** ("90% of variance"): fix by making the actual
  metric unmissable in abstract + Table 3 caption (sign-preserved low-error
  coverage, sign agreement). Optionally note it neutrally in the resubmission
  changelog; do not litigate.

## Resubmission logistics

- L1. **Target:** next suitable ARR cycle as a **resubmission** (links reviews;
  the AC pre-committed to reviewing the incorporated experiments). Preferred
  venue stays EMNLP-track/ACL unless the user decides otherwise; BlackboxNLP is
  the fallback, not the plan.
- L2. **Changelog document** (ARR resubmission note): a table mapping every
  metareview + reviewer point → paper change with section/table numbers.
  Structure it around the AC's three asks so the AC can verify compliance in
  minutes. Draft alongside the revision, not after.
- L3. Reviewer-reassignment fields: request SAME AC and reviewers (they've seen
  the evidence; e1hF engaged deeply and moved to 3.0; the changelog is written
  for them).
- L4. USER-owned tasks: two human blind raters for the label audit (B7
  strengthener); H1 decision; venue/cycle decision; preprint timing.

## Priority order

1. **A1 + A2** (new compute; long-lead) — start immediately.
2. **H1 + H2 decisions** — they gate which tables get regenerated.
3. **B1, B2, B6** main-text integration + figures (the three big incorporations).
4. **B3–B5, B9, B10** appendices; B8 aggregate audit run.
5. **C1–C9** writing pass; **D1–D4** restructure (after content lands).
6. **E1–E4** artifacts/checklist; **L2** changelog last, against the final diff.

Done-when: every General-Response table row has a manuscript location; the
utility section answers the AC's disambiguation ask with a dataset-scale
number; the changelog maps all three metareview asks to sections.
