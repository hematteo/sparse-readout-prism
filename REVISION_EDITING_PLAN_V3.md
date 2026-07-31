# SRP Conference Revision Plan v3

## Purpose

Rebuild *Sparse Readout Prism* as a focused conference paper, not a catalogue
of factorization diagnostics and applications.

The paper should be remembered for one claim:

> **Token identity is too coarse for interpreting vocabulary readouts.** The
> same token can combine different sense-related readout features, while
> different surface tokens and fitted lenses can express the same underlying
> readout feature. Sparse Readout Prism replaces a token-only account with a
> shared, compositional feature account whose error is explicit.

This thesis unifies the original polysemy results and the rebuttal's cross-lens
finding. It also explains why a sparse factorization of the LM head is useful,
rather than asking reviewers to regard the factorization itself as the entire
novelty.

The revision must still answer the rebuttal's two questions:

1. **SC1:** Is SRP more than a standard TopK SAE applied to the LM head?
2. **SC2:** What analysis does SRP enable that token rankings cannot?

The answer is:

- SRP introduces a new readout-side feature interface for selected vocabulary
  scores;
- it enables feature-level separation and alignment beyond token identity;
- direct-geometry, stability, and intervention tests show that the recovered
  structure is substantive enough to use.

---

## 1. Conference-paper thesis and scope

### Canonical claim

> SRP factorizes the fixed LM head into sparse vocabulary-row coefficients and
> shared readout directions. Any selected token score, logit difference, or
> fixed vocabulary-row combination then separates into query-side feature
> coefficients, context-dependent state projections, and a query-specific
> residual in original logit units. This shared feature space exposes
> distinctions and correspondences that token rankings cannot: different
> contexts can recruit different features for one token, while different
> surface tokens or fitted-lens outputs can share a dominant feature.

Short version:

> **SRP turns token-level readouts into compositional, comparable, and
> residual-checked feature accounts.**

### Scope boundaries

State these positively and consistently:

- TopK is a standard fitting primitive; the contribution is the factorized
  object, selected-score interface, and analyses it enables.
- The SRP basis is learned from weights without a fitting corpus; fitted-lens
  outputs may themselves be corpus-dependent.
- The unit of interpretation is a selected-score explanation or reproducible
  token group, not a uniquely identifiable feature atom.
- Local readout-side ablations test intervention predictions; they do not
  establish a complete circuit-level or generation-level causal mechanism.
- SRP is an interpretability interface, not a reconstructed generation head.
- Compatibility with arbitrary lens transports is claimed only if the exact
  implemented lens form supports the formal derivation in Section 5.3.

---

## 2. Best contributions

Use three contributions. Do not list factorization, projection, and local
decomposition as separate contributions: they are stages of one method.

### Contribution 1 — A compositional readout-feature interface

> **Sparse Readout Prism.** We introduce a weight-only sparse factorization of
> the LM head that turns any selected token logit, logit difference, or fixed
> vocabulary-row combination into signed readout-feature contributions plus an
> explicit residual in original logit units. The decomposition separates what
> the selected score asks about from what the current hidden state supplies,
> without prompt-, task-, or layer-specific training.

This is the central methodological contribution. Its essential identity is

\[
c_i(h,\alpha)
=
\underbrace{\beta_i(\alpha)}_{\text{selected query}}
\underbrace{p_i(h)}_{\text{current state}}.
\]

The contribution is not merely sparse compression of \(W_U\). It is the
readout-side interface induced by that factorization.

### Contribution 2 — Feature-level analysis beyond token identity

> **Separating and aligning token readouts through shared features.** SRP
> supports two complementary operations: it can separate different
> context-dependent feature accounts behind the same surface token, and it can
> align shared feature structure behind different surface tokens or fitted
> vocabulary lenses. This distinguishes a change in token realization from a
> change in the readout structure supporting it.

This is the strongest utility framing. It has two empirical halves:

| Token-level ambiguity | SRP analysis |
|---|---|
| Same token, different contexts or senses | Does its contribution profile separate by sense? |
| Different tokens, same concept | Do they share a dominant readout feature? |
| Different fitted lenses, different reported tokens | Is the disagreement surface-level or feature-level? |

Current evidence fully supports the cross-lens half and qualitatively supports
the same-token half. If the proposed WSD study succeeds, both halves become
systematic main-paper contributions. If it does not, narrow the contribution
to feature-level lens comparison and retain polysemy as a worked example.

### Contribution 3 — Structural characterization and new lens finding

> **Distinct, reproducible, and intervention-relevant readout structure.**
> Across six models, SRP reconstructs selected scores more reliably than
> k-means, hard clustering, kNN, ridge, and PCA; held-out dictionaries recover
> feature groupings that direct row geometry misses; score explanations
> reproduce across independent training runs; and their contributions predict
> realized local readout-side changes. Using this validated feature space, we
> find that English- and Chinese-fitted Jacobian lenses can report different
> surface languages on identical states while the same SRP feature remains
> dominant in 77 of 80 controlled prompts.

In the prose, distinguish the roles:

- geometry, stability, and interventions **characterize the representation**;
- the 77/80 result is the **scientific finding enabled by it**.

### Evidence paragraph

After the contribution list, give one compact numerical preview:

- SRP sign-preserved low-error coverage: 0.72–0.81 on the six gated models,
  leading the strongest alternative by 8.9–17.3 points;
- held-out grouping recovery: SRP 72–75%, kNN 43–49%, clustering 21–25%;
- predicted-versus-realized local effects: \(R^2=0.83\)–\(0.93\), versus at
  most 0.06 for random directions;
- cross-lens dominant-feature agreement: 77/80, with unrelated-token and
  shuffled-prompt controls at 0/159 and 0/240.

---

## 3. Title and one-sentence pitch

Preferred title:

> **From Tokens to Features: Sparse Readout Prism for Language-Model
> Readouts**

More conservative alternative:

> **Sparse Readout Prism: Decomposing and Comparing Language-Model Readouts in
> a Shared Feature Basis**

One-sentence pitch for the abstract and talks:

> Vocabulary lenses report tokens; SRP exposes the shared readout features
> composing those token scores, revealing when one token expresses different
> structures and when different tokens express the same structure.

Do not title or pitch the paper primarily around reconstruction, LM-head
compression, or readout replacement.

---

## 4. Top-level paper structure

Replace the current split between “Calibrating Sparse Readout Accounts” and a
catalogue of “Readout Analyses.” Use claim-based sections rather than RQ labels.

### 1. Introduction — Tokens are an incomplete unit of readout interpretation

The Introduction should establish the complete story on page 1:

1. Define the LM-head readout, selected readout score, and contrast.
2. Explain that a token or token ranking is a surface-level output unit.
3. Introduce the two-sided ambiguity:
   - one token can combine multiple readout structures;
   - different tokens or lenses can express shared readout structure.
4. Show the motivating dual example in Figure 1.
5. Introduce SRP and the query-side/state-side separation.
6. Distinguish SRP from activation SAEs and direct LM-head geometry.
7. Preview the structural evidence and cross-lens finding.
8. Present the three contributions.

Do not frame the paper as a critique of J-lens. Corpus-fitted J-lenses are the
flagship controlled demonstration of a general token-level ambiguity, not the
sole reason SRP exists.

### 2. Sparse Readout Prism

Organize the Method by the new operations it creates:

1. factorizing the output interface;
2. decomposing a selected vocabulary score;
3. comparing feature accounts across contexts or fitted lenses;
4. deciding when an account is safe to interpret.

Move TopK optimization details, hyperparameters, and the full score-family
table to the appendix.

### 3. Evaluation design

Organize metrics by the claim they test:

- **faithfulness:** selected-score residual, sign, replacement behavior;
- **distinctness:** direct-geometry and dense low-rank alternatives;
- **reproducibility:** independent dictionaries and held-out group recovery;
- **intervention relevance:** predicted versus realized local changes;
- **utility:** sense separation and/or cross-lens feature agreement.

State the model split once: eight models enter reconstruction analysis; six
pass the stricter interpretation gate used in the rebuttal comparisons.

### 4. Sparse readout features capture nontrivial structure

Present the evidence as one cumulative argument:

1. selected-score reconstruction is accurate in a declared operating regime;
2. simpler geometry does not recover the same score fidelity or groupings;
3. explanation-level structure reproduces across seeds;
4. feature contributions predict local intervention effects.

The reader should finish this section believing that the representation is
accurate enough, nontrivial enough, and reproducible enough to support the
utility section.

### 5. Beyond token identity

Make this a top-level section and the paper's empirical centerpiece.

#### 5.1 Same token, different feature structure

Preferred conference version: dataset-scale WSD with gold senses.

Fallback version: one compact `bug` example as a worked analysis, explicitly
described as illustrative rather than systematic.

#### 5.2 Different surface tokens, shared feature structure

Present the complete English/Chinese J-lens study:

- identical frozen hidden states;
- 80 controlled prompts;
- 77/80 dominant-feature agreement;
- 0/159 unrelated-token matches;
- 0/240 shuffled-prompt matches;
- no translation pairs, language labels, or lens-fitting corpora used to fit
  the SRP basis.

Use the narrow conclusion:

> In this controlled setting, the reported surface language can follow the
> fitting corpus even when both readings are supported by the same dominant
> LM-head feature. An English surface reading is therefore not, by itself,
> evidence of an English intermediate computation.

Do not claim that models never privilege one language or that the feature is a
language-neutral causal representation.

#### 5.3 Broader example

Use at most one short prompt-injection example to illustrate generality. It
must not compete with the controlled cross-lens result.

### 6. Related work

Position SRP against four adjacent objects:

- vocabulary lenses: token-level outputs without shared feature accounts;
- activation SAEs: corpus/layer activation bases rather than output-row bases;
- output geometry and matrix factorization: structure of \(W_U\) without the
  selected-score feature interface;
- additive attribution: component-level score decomposition rather than
  decomposition of the readout direction itself.

Avoid “prior work cannot” claims broader than the exact comparison.

### 7. Conclusion

Return to the token-versus-feature thesis. State what the method adds, what the
cross-lens study reveals, and where the interpretation boundary lies. Do not
end with reconstruction metrics or a list of speculative applications.

---

## 5. Figure and table plan

Limit the main paper to four high-value visual objects.

### Figure 1 — Beyond token identity

Use a dual-panel motivating/result figure:

- **Panel A: same token, different feature accounts.** `bug` in software and
  insect contexts, or the aggregate WSD result if available.
- **Panel B: different tokens, shared feature.** English- and Chinese-fitted
  J-lens outputs on the same hidden state converging on one SRP feature.

This figure should communicate the entire reason for the paper.

### Figure 2 — Method

One compact schematic showing:

\[
W_U \rightarrow (Z,D,R),
\qquad
(\alpha,h) \rightarrow \beta_i(\alpha)p_i(h)+\epsilon.
\]

Avoid another large qualitative bar plot here.

### Table 1 — Structural alternatives

Compact six-model summary of SRP versus the strongest direct-geometry
alternatives. Report the headline coverage metric and held-out grouping
recovery; move every full method/model grid to the appendix.

### Figure or Table 3 — Reproducibility and interventions

Combine:

- cross-seed explanation or grouping recovery;
- predicted-versus-realized intervention summary;
- random-direction reference.

If the combined visual becomes crowded, retain the intervention scatter in the
main paper and report stability in a compact table sentence.

Do not keep the main DLA figure or fixed-scale lexical-edit table.

---

## 6. Method rewrite

### 6.1 Factorizing the output interface

Start from

\[
W_U=\mathbf 1\mu^\top+ZD_{\mathrm{feat}}+R.
\]

Interpret the factors before describing optimization:

- \(Z\): sparse vocabulary-row coefficients;
- \(D_{\mathrm{feat}}\): shared readout directions;
- \(D_{\mathrm{feat}}h\): state-side readout projections;
- \(R\): raw-space residual used for exact query-specific accounting.

### 6.2 Make query/state separation the central derivation

For vocabulary coefficients \(\alpha\):

\[
\alpha^\top W_U h
=
(\alpha^\top\mathbf1)\mu^\top h
+\sum_i
\underbrace{(\alpha^\top Z_{:i})}_{\beta_i(\alpha)}
\underbrace{(d_i^\top h)}_{p_i(h)}
+\alpha^\top Rh.
\]

Explain:

- score choice changes \(\beta_i(\alpha)\);
- context or layer changes \(p_i(h)\);
- the feature basis is fixed;
- the residual exposes what the fitted basis did not explain for that exact
  query and state.

This equation is the methodological center of the paper.

### 6.3 Formalization gate for fitted-lens comparison

Verify the exact implemented J-lens form before writing a general transport
claim.

- If it is exactly linear/affine into the shared residual space, derive the
  transported decomposition and handle the bias explicitly.
- Otherwise define the feature-level comparison operationally and restrict the
  claim to the tested J-lens construction.

Do not advertise a general “transport calculus” based only on an algebraic
possibility.

### 6.4 Selective interpretation

Compress the reading protocol into four checks:

1. replacement behavior;
2. query-specific residual/error;
3. exact/reconstructed sign agreement;
4. contribution concentration and label audit.

Use one main error definition. If floored and unfloored ratios are both needed,
give them different symbols. State that the shared offset cancels for zero-sum
contrasts and that the residual is exact but not necessarily orthogonal to the
overcomplete dictionary.

---

## 7. Evidence placement

### Main text

Must include:

- direct-geometry comparison across six gated models;
- held-out grouping recovery;
- explanation-level stability summary;
- six-model predicted-versus-realized local ablations;
- full cross-lens study and controls;
- WSD headline result, if completed;
- one explicit statement of the interpretation gate and error tails.

### Appendix

Move or add:

1. full factorization and selected-score identities;
2. model suite, gating, tokenization, and dataset provenance;
3. full direct-geometry method/model tables and CIs;
4. error distributions and exact-margin stratification;
5. seed matching, group recovery, threshold sensitivity, and failure tails;
6. per-model intervention scatters and slope analysis;
7. cross-lens prompts, fitting procedure, controls, and per-case results;
8. WSD protocol and complete results, if run;
9. blinded label-audit inputs, aggregation rule, and outputs;
10. matched-KL lexical-edit frontiers and all negative model results;
11. DLA composition and additional qualitative examples;
12. hyperparameter, sparsity, width, and dead-feature analyses;
13. compute, hardware, package versions, seeds, and reproduction steps.

Every experiment promised in the rebuttal must have a named destination, but
not every experiment deserves main-paper space.

---

## 8. Material to demote

These are supporting analyses, not headline contributions:

- rowEV and reconstructed-head replacement;
- number of evaluated models by itself;
- feature-resolved DLA;
- lexical editing;
- four-panel qualitative polysemy displays;
- exhaustive score-family comparisons;
- hyperparameter sweeps;
- code and dictionary release.

Retain what is needed for credibility and reproducibility, but do not allow
these items to obscure the token-to-feature thesis.

The lexical-edit result should be reported narrowly: Qwen3.5-2B wins the
matched-KL comparisons, but the advantage over mean-row directions is not
universal. DLA should be a short appendix demonstration because its composition
with SRP follows largely from bilinearity.

---

## 9. Highest-value new evidence

### Priority 1 — Dataset-scale sense disambiguation

This directly completes the “same token, different features” half and answers
the AC's suggested test.

Preferred design:

- CoarseWSD-20 or another established sense-tagged dataset;
- targets filtered and tokenization coverage reported;
- contribution profiles computed from held-out contexts;
- cluster purity/ARI and held-out sense prediction;
- majority-class, token-ranking, row-kNN, and activation-SAE controls;
- at least two interpretation-gated models if feasible.

Run it only if the evaluation is controlled and auditable. If it fails or
cannot be completed, narrow the contribution rather than disguising the
qualitative examples as a systematic result.

### Priority 2 — Cross-lens replication

Replicate the corpus-fit result on one additional model or language pair if
feasible. This reduces dependence on one Qwen/language configuration and is
more valuable than adding another qualitative application.

### Priority 3 — Human feature audit

Add two independent blinded human raters if possible. Describe the existing
three language-model runs accurately as a same-model reproducibility audit,
not as independent human annotation.

Do not promise additional languages, models, lenses, or human raters unless
they will appear in the submitted manuscript.

---

## 10. Main-paper budget

| Component | Target |
|---|---:|
| Abstract | 190–220 words |
| Introduction + Figure 1 | 1.25–1.5 pages |
| Method + Figure 2 | 1.5 pages |
| Evaluation design | 0.5–0.75 page |
| Structural evidence | 1.75–2 pages |
| Beyond-token-identity results | 1.25–1.5 pages |
| Related work | 0.5 page |
| Conclusion | 0.25–0.4 page |

If space is tight, cut in this order:

1. extra qualitative examples;
2. lexical editing;
3. DLA;
4. secondary score families;
5. detailed model-specific calibration.

Do not cut the cross-lens study, direct-geometry evidence, local intervention
validation, or query-specific residual boundary.

---

## 11. Consistency decisions before editing

Resolve these before regenerating tables or rewriting claims:

1. **Dictionary family:** do not mix paper-dictionary and seed-variation
   results without explicit cross-recipe disclosure.
2. **Error notation:** select one main \(\rho\) definition or give floored and
   unfloored metrics different symbols.
3. **Lens formalization:** verify the exact J-lens map before claiming general
   transport compatibility.
4. **Model counts:** distinguish eight reconstruction models from six
   interpretation-gated models.
5. **Cross-lens sample:** use 77/80 consistently; remove 14/16 pilot language
   from manuscript-facing material.
6. **Audit description:** distinguish language-model reproducibility runs from
   human evaluation.
7. **Intervention language:** use “predicts local readout-side changes,” not
   unqualified causal language.
8. **Editing scope:** print the cross-model matched-KL losses as well as the
   Qwen3.5-2B wins.

---

## 12. Resubmission change summary

Organize the resubmission note around the AC's requests:

| Concern | Concrete revision |
|---|---|
| What systematic interpretation does SRP enable? | New top-level “Beyond Token Identity” section; controlled cross-lens comparison; WSD if completed |
| Is SRP more than TopK applied to \(W_U\)? | New selected-score interface framing plus direct-geometry, grouping, stability, and intervention evidence |
| New rebuttal experiments were not reviewable | Every rebuttal result placed in a named main or appendix section |
| Contrast/readout definitions arrive late | Defined on page 1 and used consistently |
| Labels are subjective | Blinded audit with protocol and agreement; human audit if completed |
| Stability is unknown | Independent-seed explanation and group recovery analysis |
| Median hides failures | Mean, p95, maximum, and exact-margin-stratified errors |
| Downstream relevance is unclear | Six-model predicted-versus-realized local ablation study |

Write the change summary alongside the paper so section and table references
remain current.

---

## 13. Execution order

### Phase 0 — Freeze the story

- Resolve the eight consistency decisions.
- Decide whether WSD and cross-lens replication will be ready.
- Freeze the title, thesis, three contributions, and four-visual inventory.

### Phase 1 — Rewrite the spine

- Rewrite abstract, introduction, contribution list, and conclusion together.
- Build the dual-panel Figure 1.
- Rewrite the Method around the query/state identity and explicit residual.

### Phase 2 — Build structural evidence

- Consolidate calibration and direct-geometry evidence.
- Add held-out grouping recovery and stability.
- Add the six-model local intervention result.
- Move detailed grids and diagnostics to appendices.

### Phase 3 — Build the utility section

- Add WSD if it passes the quality bar.
- Add the full cross-lens protocol, controls, and result.
- Retain at most one broader example.

### Phase 4 — Appendix and artifacts

- Place every promised rebuttal experiment.
- Add negative results and scope boundaries.
- Verify code, dictionaries, prompts, audits, seeds, compute, and versions.

### Phase 5 — Adversarial editing pass

- Verify every number against `REBUTTAL_RESULTS.md` and final artifacts.
- Search for obsolete 14/16 results, mixed error definitions, and broad causal
  wording.
- Remove repeated motivation and metric exposition.
- Give the abstract and first page to a reader without the rebuttal. They must
  be able to answer:
  1. What is new beyond TopK on the LM head?
  2. Why are token rankings insufficient?
  3. What new finding did SRP enable?

### Phase 6 — Resubmission package

- Complete the change summary with final section/table references.
- Compile and visually inspect the main paper and appendix.
- Confirm that all promised artifacts are accessible and documented.

---

## 14. Done-when criteria

The conference revision is ready only when:

- the title, abstract, Figure 1, Introduction, and Conclusion all express the
  same token-to-feature thesis;
- readout, selected score, and contrast are defined on page 1;
- the contribution list contains one method, one beyond-token-identity
  capability, and one empirical characterization/finding;
- the query-side/state-side decomposition is visibly the methodological core;
- the cross-lens study is a full top-level result rather than a late example;
- same-token sense separation is either evaluated systematically or scoped as
  illustrative;
- direct-geometry, stability, and intervention evidence form a coherent chain
  rather than an experiment catalogue;
- DLA, editing, sweeps, and extra qualitative examples no longer dominate main
  space;
- all negative model results and interpretation limits are disclosed;
- every rebuttal experiment has a manuscript location;
- a new reviewer can summarize the novelty and utility correctly after the
  abstract and first page alone.


---

## Phase 0 decisions — RESOLVED 2026-07-31 (execution session)

1. **Dictionary family:** paper dictionary retained as the operating
   dictionary (H1 option b). All stability/grouping comparisons use the
   seed-variation family end-to-end; the cross-recipe number (paper dict
   recovers 0.444 of LOO cores, below kNN) is disclosed in
   `appendices/basis_stability.tex` §"Cross-Recipe Disclosure".
2. **Error notation:** floored gate keeps \(\rho\); unfloored gets its own
   symbol \(\tilde\rho\), defined in the method reading rule and
   Appendix B. All baseline comparisons quote \(\tilde\rho\); no table
   mixes the two.
3. **Lens formalization:** operational comparison only, restricted to the
   tested J-lens construction (`sections/method.tex`
   §"Comparing and Reading Feature Accounts"). No transport calculus
   claimed.
4. **Model counts:** 8 reconstruction / 6 gated (top-1 ≥ 0.75), stated once
   in Evaluation Design; fidelity table splits gated vs Gemma rows.
5. **Cross-lens sample:** 77/80 [0.90, 0.99] everywhere; verified against
   `readout_prism_workspace/results/jlens/cross_lens_v2_summary.json`.
   No 14/16 numbers anywhere in manuscript.
6. **Audit description:** blinded 3-run LM audit reported as same-model
   reproducibility audit (93.4% unanimity, κ=0.87, 76/4/19 over 99,
   81.8% agreement with author audit); author audit retained in appendix.
7. **Intervention language:** "predicts realized local readout-side
   changes"; slopes reported as measured; scope stated (linear in h, no
   re-forward).
8. **Editing scope:** matched-KL frontier added to
   `appendices/lexical_control_stress_test.tex` §"Matched-KL Frontier";
   0.8B/9B losses and both rejected mechanistic explanations printed.

**WSD (Priority 1): FALLBACK taken.** The 2026-07-31 runs
(`results/wsd_core/coarsewsd20_*`) show SRP profiles ≈ shuffled-SRP
control (0.86 vs 0.84 on 2B; SRP *below* shuffled on 9B and R1-Llama),
far under hidden-state baselines (0.95+). §5.1 is therefore scoped as
illustrative worked analyses; no WSD numbers quoted in the manuscript.
Open user call: whether to disclose the negative WSD result in an
appendix once the runs are final.

**Executed 2026-07-31:** new title; abstract; 3-contribution intro with
page-1 definitions and evidence paragraph; method operations framing +
comparison subsection + \(\tilde\rho\); claim-organized Evaluation
Design; §4 structural-evidence chain (fidelity → six-model
direct-geometry Table 2 → stability → six-model intervention Table 3 →
blinded audit → regime → demoted DLA/edits); new §5 Beyond Token
Identity (5.1 illustrative polysemy, 5.2 cross-lens 77/80 with CJK
worked example); related-work lens paragraph + Gurnee cite; conclusion;
limitations (stability scope + cross-lens scope); new appendices
I (basis_stability), J (intervention_validation), K (cross_lens_study),
direct-geometry grid in robustness_controls, matched-KL subsection in
lexical appendix, blinded-audit paragraph in audit appendix; appendix
roadmap updated; builds clean (51 pp, zero LaTeX warnings).

**Still open:** (a) dual-panel Figure 1 (panel B cross-lens visual) —
current Figure 1 is the bug/insect lens-vs-prism figure; (b) main text
runs ~1.2–1.5 pp over the 8-page budget — cut per §10 order;
(c) `sections/worked_example.tex` remains orphaned (not \input);
(d) verify `gurnee2026workspace` author list and `paulo2025stability`
title against the actual publications before submission;
(e) B8 aggregate non-cherry-picked audit still unrun;
(f) resubmission changelog (§12) not yet drafted.
