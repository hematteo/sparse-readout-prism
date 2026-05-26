# Reviewer-Facing Clarity Pass

Source of truth checked: `CORE_CLAIM.md` and `glossary_and_terminology.md`.

Paper source was not modified. The revised text below is intended as paste-ready replacement text for the indicated local passages.

## Location
`sections/frontmatter.tex`, Abstract.

## Purpose
State the core contribution, the method, the evidence, and the claim boundary in the first reviewer-facing passage.

## Revised text
```tex
\begin{abstract}
Lens-style analyses map language-model hidden states through the final
unembedding into vocabulary logits, but once a logit or contrast is selected, the
readout is still represented by one dense score. That score does not identify the
readout-side directions that support or oppose the target, nor the reconstruction
error of a sparse account. We introduce Sparse Readout Prism (SRP), a sparse
feature decomposition for logit-lens readouts that factorizes the LM head into
sparse token-row coefficients and shared readout feature directions. Given a
hidden state and scalar readout target, SRP decomposes the selected score into
signed feature contributions plus an explicit residual in original logit
coordinates.

We evaluate when this decomposition can be read by measuring LM-head replacement
fidelity, target reconstruction, sign preservation, and null/reference baselines
across multiple model families and readout geometries. In representative
high-fidelity settings, SRP preserves held-out contrast signs at
\(0.949\)--\(0.958\) with median relative reconstruction error below \(0.10\),
and learned sparse structure outperforms null and reference baselines. In
sign-preserved, low-error cases, the same readout-feature basis shows how target
choice, prompt context, residual-stream components, and constrained readout-side
edits change the signed terms.
\end{abstract}
```

## Why this edit helps
- Puts the core contribution in the first half: sparse feature decomposition for logit-lens readouts.
- Replaces the slightly broad "obscuring" with explicit reviewer objects: support, opposition, and reconstruction error.
- Changes "reveals" / "reshape" to more restrained evidence language: "shows" and "change".

## Meaning-preservation checks
- Claims preserved: SRP factorizes the LM head and decomposes selected scalar readout targets into signed terms plus residuals.
- Terms aligned with glossary: "readout", "readout feature directions", "scalar readout target", "signed feature contributions", "residual", "LM-head replacement fidelity".
- Numbers/citations preserved: Numeric results \(0.949\)--\(0.958\), \(0.10\) preserved; no citations in original abstract.
- Possible meaning changes or risks: "when this decomposition can be read" is more interpretive than "test SRP"; acceptable if the paper consistently treats diagnostics as prerequisites for interpretation.

## Flags
- No overclaim found. The abstract stays readout-level and does not imply causal or generation-level evidence.
- "Representative high-fidelity settings" is evidence-sensitive but supported by the reported regime framing.

## Location
`sections/introduction.tex`, first two Introduction paragraphs.

## Purpose
Orient reviewers to the standard logit-lens interface, define the gap, and state the local problem SRP solves.

## Revised text
```tex
Language models map hidden states to vocabulary logits through the final
unembedding matrix, or LM head; we call this interface the \emph{readout}.
Lens-style methods inspect hidden states in vocabulary space, either through the
LM head or through a learned lens
\citep{nostalgebraist2020,belrose2023tuned,pal2023future,ghandeharioun2024patchscopes}.
They can show which tokens or scores a hidden state favors. However, after a
token, contrast, or margin is selected, the inspected quantity remains one dense
scalar dot product with an unembedding row or readout direction. The score alone
does not say which shared readout-side directions support the target, which
oppose it, or how much reconstruction error remains after a sparse account of
that score.

This leaves a local readout-level gap between vocabulary-space inspection and
feature-level accounting. Existing lens and readout methods identify favored
tokens or compare selected scores, but they do not decompose a selected scalar
readout target into sparse, signed, residual-checked readout-side terms in the
original logit units. We address the local problem: for a fixed hidden state and
a fixed linear target, decompose the selected readout score while carrying the
reconstruction residual explicitly. The result is a readout-level decomposition
whose fidelity can be checked for the target at hand.
```

## Why this edit helps
- Makes the gap sentence more direct: reviewers see exactly what the logit lens does not provide.
- Replaces "by itself" with "alone", reducing repetition while preserving scope.
- Keeps the problem local, linear, and readout-level.

## Meaning-preservation checks
- Claims preserved: lens-style methods expose vocabulary scores; SRP addresses local decomposition of a fixed linear target.
- Terms aligned with glossary: "readout", "scalar readout target", "residual-checked", "linear target", "reconstruction residual".
- Numbers/citations preserved: All original citations preserved.
- Possible meaning changes or risks: None substantive.

## Flags
- No terminology drift found in these paragraphs.
- "Feature-level accounting" is readable but should remain tied to readout features, as this passage does.

## Location
`sections/introduction.tex`, Contribution list.

## Purpose
Tell reviewers exactly what the paper contributes and where those contributions stop.

## Revised text
```tex
The paper contributes:
\begin{itemize}
    \item a sparse dictionary factorization of the final unembedding / LM head
    into token-row coefficients and shared readout feature directions;
    \item a scalar readout-target formalism that turns token logits, pairwise
    differences, group contrasts, reference contrasts, and top-competitor margins
    into signed feature contributions plus explicit residuals;
    \item a fidelity evaluation suite covering LM-head replacement, target
    reconstruction, sign preservation, relative error, and sparse-structure
    baselines;
    \item residual-checked readout-level analyses of target choice, prompt
    context, feature-resolved residual-stream attribution, and candidate
    readout-side edit directions.
\end{itemize}
```

## Why this edit helps
- Uses "sparse dictionary factorization", matching the core-claim file.
- Replaces "scalar linear-target decomposition" with the glossary term "scalar readout-target formalism".
- Keeps the last contribution explicitly local and diagnostic-bound.

## Meaning-preservation checks
- Claims preserved: Four original contributions are preserved.
- Terms aligned with glossary: "scalar readout target", "signed feature contributions", "explicit residuals", "LM-head replacement", "target reconstruction", "candidate readout-side edit directions".
- Numbers/citations preserved: No numbers or citations in original list.
- Possible meaning changes or risks: "formalism" slightly broadens "decomposition"; acceptable because the method defines a target family and coefficient identities.

## Flags
- No overclaim found.
## Location
`sections/method.tex`, Method overview paragraph.

## Purpose
Preview the method as a sequence of learned objects, computed quantities, chosen targets, and diagnostics.

## Revised text
```tex
SRP separates four objects that are otherwise collapsed in a lens score. First,
it factorizes LM-head rows into sparse token-row coefficients and shared readout
feature directions. Second, it projects a hidden state onto those directions,
producing readout feature projections before any target is chosen. Third, a
scalar readout target selects a linear combination of unembedding rows and fixes
the sign convention. Finally, target coefficients and hidden-state projections
are multiplied to obtain signed feature terms, with the row-reconstruction
residual carried through to the selected score. This keeps the logit-lens
interface fixed: a hidden state \(h\) is still read out through the LM head
\(\WU\), but selected logits, contrasts, group scores, and margins are
decomposed through the row factorization of \(\WU\).
```

## Why this edit helps
- Makes the method overview answer "what is learned, what is projected, what the target supplies, what is diagnosed".
- Defines the order before the equations, reducing reviewer load.
- Preserves the distinction between projection vector and target-conditioned signed contribution.

## Meaning-preservation checks
- Claims preserved: Same four method steps as original.
- Terms aligned with glossary: "readout feature projections", "scalar readout target", "target coefficients", "signed feature terms", "row-reconstruction residual".
- Numbers/citations preserved: No numbers or citations in original paragraph.
- Possible meaning changes or risks: Removes the final sentence "The subsections below..." as redundant signposting; no technical content lost.

## Flags
- No overclaim found.

## Location
`sections/experimental_setup.tex`, "Claim-to-test map".

## Purpose
Make the experiments section a contract between claims, evidence, controls, and failure modes.

## Revised text
```tex
\paragraph{Claim-to-test map.}
The experiments test the prerequisites for interpreting a local score
decomposition. First, held-out row reconstruction and LM-head replacement on C4
hidden states \citep{raffel2020exploring,dodge2021documenting} ask whether the
fitted readout SAE preserves the readout it replaces. Second, scalar-target
reconstruction of exact score, sign, and relative error asks whether a selected
linear target is reconstructed well enough for its feature bars to be read.
Third, null and reference baselines ask whether learned token-row sparse
structure, rather than dense low-rank structure, lexical neighborhoods, or sparse
capacity alone, accounts for reconstruction gains. We treat sign flips, large
relative error, weak replacement fidelity, or parity with baselines as failure
modes for interpretation. Qualitative displays are therefore validation-gated
demonstrations, not independent evidence of generation-level behavior.
```

## Why this edit helps
- Explicitly maps each experimental block to the claim it licenses.
- Names the controls and what they rule out.
- States failure modes and separates validation from qualitative demonstration.

## Meaning-preservation checks
- Claims preserved: Same evaluation components as original, with failure modes made explicit.
- Terms aligned with glossary: "local score decomposition", "LM-head replacement", "readout SAE", "scalar-target reconstruction", "linear target", "feature bars".
- Numbers/citations preserved: C4 citations preserved; no numbers in original paragraph.
- Possible meaning changes or risks: "weak replacement fidelity" needs operational thresholds elsewhere; if not thresholded, keep as a qualitative failure mode.

## Flags
- This passage adds explicit failure-mode language not present in the original. It does not change results, but it asks the paper to be comfortable treating baseline parity and large residuals as interpretation failures.

## Location
`sections/results_and_analysis.tex`, Results opening.

## Purpose
Frame the Results section as a cumulative argument rather than a list of analyses.

## Revised text
```tex
The results build a cumulative readout-level argument. We first test whether the
factorized readout preserves the original LM-head behavior and selected scalar
targets. We then ask whether learned sparse row-code structure outperforms
controls. Only after those checks do we interpret low-error local decompositions,
compare target and context effects, and show linear extensions to
feature-resolved component attribution and candidate readout-side edits. Unless
stated otherwise, aggregate low-error counts use \(\rho<0.5\).
```

## Why this edit helps
- Matches the requested order: fidelity, target reconstruction/sign preservation, controls, qualitative interpretation, extensions.
- Replaces "faithful enough to read" with concrete diagnostics.
- Prevents readers from treating qualitative examples as the first evidence.

## Meaning-preservation checks
- Claims preserved: Same section flow and \(\rho<0.5\) convention.
- Terms aligned with glossary: "factorized readout", "selected scalar targets", "learned sparse row-code structure", "local decompositions", "candidate readout-side edits".
- Numbers/citations preserved: \(\rho<0.5\) preserved.
- Possible meaning changes or risks: None substantive.

## Flags
- No overclaim found.

## Location
`sections/results_and_analysis.tex`, Subsection "The Sparse Readout Factorization Identifies High-Fidelity Readout Regimes".

## Purpose
Ask when the factorized readout is accurate enough to support local interpretation.

## Revised text
```tex
\paragraph{Question and design.}
When can the factorized readout be used for local score decompositions? We
evaluate held-out row reconstruction, LM-head replacement on C4 hidden states,
and held-out logit-difference reconstruction across the model suite. The target
of this test is preservation of the original readout, measured by top-token
agreement, KL, sign preservation, and relative reconstruction error; it is not a
test of task behavior.

\paragraph{Result, interpretation, and boundary.}
\Cref{tab:readout-query-fidelity-summary} separates row reconstruction,
replacement behavior, and held-out logit-difference reconstruction for the
\(k=256\) settings used below, with
\(\rho=|s_{\mathrm{exact}}-s_{\mathrm{recon}}|/(|s_{\mathrm{exact}}|+0.5)\).
Some settings best preserve top-1/KL; others keep contrast signs despite weaker
full-readout replacement. Gemma keeps signs with larger target errors. Errors
and flips concentrate near small exact margins
(\Cref{fig:readout-query-fidelity-result}), so case studies use signed contrasts
with nontrivial margins and small residuals. RowEV alone is a capacity metric,
not evidence that a local feature-bar display is interpretable.
```

## Why this edit helps
- Starts with the experimental question.
- Names design, metrics, result, interpretation, and boundary.
- Makes RowEV's limited role explicit at the end, where reviewers need the caveat.

## Meaning-preservation checks
- Claims preserved: Replacement, row reconstruction, target fidelity, sign and error diagnostics all preserved.
- Terms aligned with glossary: "local score decompositions", "LM-head replacement", "sign preservation", "relative reconstruction error", "RowEV".
- Numbers/citations preserved: \(k=256\), \(\rho\) definition, figure/table refs preserved.
- Possible meaning changes or risks: Adds "not a test of task behavior"; this is consistent with glossary and limitations.

## Flags
- Existing labels and filenames contain "query" (`readout-query`), but those are LaTeX/file identifiers rather than formal prose. Do not rename unless you also update all refs and files.

## Location
`sections/results_and_analysis.tex`, Subsection "Learned Token-Row Sparse Structure Improves Target Reconstruction".

## Purpose
Test whether reconstruction gains come from learned sparse row structure rather than capacity or simple geometry.

## Revised text
```tex
Are the target-reconstruction gains due to learned sparse row-code structure, or
could generic capacity and output-embedding geometry explain them? We compare SRP
with nearest-row ridge, PCA, shuffled sparse codes, and random sparse patterns on
the full Qwen3.5-2B logit-difference set.

\paragraph{Sparse-structure controls.}
On the full Qwen3.5-2B logit-difference set, SRP has the lowest median error and
largest sign-preserved low-error subset
(\Cref{tab:main-baseline-null-comparison}). Nearest-row ridge
\citep{hoerl1970ridge} and PCA \citep{jolliffe2002principal} retain some sign
structure, while shuffled codes and random sparse patterns approach chance.
These baselines test whether dense low-rank structure, lexical neighborhoods, or
sparse capacity alone are sufficient. The gap supports learned sparse row-code
structure as the useful object for these readout-target reconstructions; it does
not by itself validate every feature label as semantic. The full tail appears in
Appendix~\ref{app:robustness-key-quantitative}.
```

## Why this edit helps
- Converts the subsection opening into a clear control question.
- Names each baseline's role before giving the result.
- Adds a boundary separating reconstruction evidence from label validity.

## Meaning-preservation checks
- Claims preserved: SRP outperforms ridge, PCA, shuffled, and random baselines on the same reported metrics.
- Terms aligned with glossary: "learned sparse row-code structure", "readout-target reconstructions", "feature label".
- Numbers/citations preserved: Table ref and citations preserved; table values remain in table.
- Possible meaning changes or risks: "not by itself validate every feature label as semantic" adds a caveat, not a claim change.

## Flags
- Avoid "semantic" without qualification. The revised caveat uses it only to deny overinterpretation.

## Location
`sections/results_and_analysis.tex`, Subsection "Scalar Readout Targets Extend Beyond Token Pairs".

## Purpose
Show that the scalar readout-target formalism extends to dataset-derived linear contrasts without implying task accuracy.

## Revised text
```tex
Does the same scalar readout-target formalism apply beyond curated token pairs?
We express answer, abstention, label, and safe/action choices as linear
contrasts, then evaluate target reconstruction rather than end-to-end task
performance. We instantiate QA
\citep{rajpurkar2018squad2,yang2018hotpotqa}, ContractNLI labels via LegalBench
\citep{koreeda2021contractnli,guha2023legalbench}, and SecurityEval safe/unsafe
actions \citep{siddiq2022securityeval}; QA contrasts average gold-answer rows
against fixed distractor rows.

\Cref{tab:benchmark-derived-task-targets} shows lower error on answerable-QA,
multi-hop-distractor, and legal-label rows than on SecurityEval, whose
safe/unsafe-action contrast spans many lexical forms. SRP therefore reconstructs
several benchmark-derived linear readout targets beyond curated token pairs.
These rows test readout-level fidelity rather than end-to-end task accuracy;
low-error QA examples appear in
Appendix~\ref{app:benchmark-derived-display-examples}.

A low-error benchmark-derived row means SRP reconstructs the specified answer,
label, abstention, or action contrast at the readout. The weaker SecurityEval row
is consistent with this scope: safe/unsafe actions span more heterogeneous
lexical forms than compact answer or label families, and generation-level
coverage requires separate evaluation.
```

## Why this edit helps
- Puts "not task accuracy" before the dataset names, preventing overreading.
- Uses the glossary phrase "benchmark-derived linear readout targets".
- Keeps SecurityEval interpretation bounded to lexical heterogeneity and readout-level reconstruction.

## Meaning-preservation checks
- Claims preserved: Same datasets, target construction, and qualitative interpretation.
- Terms aligned with glossary: "scalar readout-target formalism", "benchmark-derived linear readout targets", "readout-level fidelity".
- Numbers/citations preserved: All original citations and refs preserved.
- Possible meaning changes or risks: "target reconstruction rather than end-to-end task performance" is a clearer boundary, not a technical change.

## Flags
- The table label `benchmark-derived-task-targets` contains "task"; acceptable as an internal label, but prose should continue to say "benchmark-derived readout target".

## Location
`sections/results_and_analysis.tex`, Subsection "Local Score Decompositions Expose Feature Competition Within Token Rankings".

## Purpose
Interpret qualitative local decompositions only after residual and label caveats are stated.

## Revised text
```tex
Given sign-preserved, low-error cases, what do the signed feature terms show
inside a token ranking? We inspect token-vs-token local score decompositions for
Qwen3.5-2B. Labels summarize high-coefficient rows; the measured quantities are
feature ids, signed contributions, and residuals.

\paragraph{Polysemous-token case studies.}
SRP decomposes low-error Qwen3.5-2B polysemous-token logit differences
(\Cref{fig:selected-readout-prism-examples}). In insect context,
\texttt{bug} draws positive mosquito-, bee-, and beetle-labeled terms against
error-token rows; in software, positives shift toward defect, crash, and debug
rows. The panels have relative errors \(0.185\) and \(0.003\). Additional
\texttt{bark}/\texttt{bass} cases, plotting conventions, and label audits appear
in
\cref{fig:app-selected-readout-prism-extra-examples,app:display-taxonomy,tab:app-main-case-study-feature-audit}.
These sign-preserved, low-\(\rho\) examples show the same surface token drawing
support from different row-labeled feature groups as target and context change.
To calibrate the use of these labels, Appendix~\ref{app:main-case-study-feature-audit}
audits the displayed Qwen feature labels: among 99 labels, 67 are coherent
lexical summaries, 18 are token-form or formatting features, and 14 are
ambiguous. We use only the coherent lexical summaries as label-based
interpretive support below. The next sections separate target and context
effects; aggregate frequencies are in the appendix.
```

## Why this edit helps
- Opens with a question and constrains the answer to sign-preserved, low-error cases.
- Replaces "semantic evidence" with "label-based interpretive support", matching the glossary's feature-label caution.
- Keeps the exact label audit counts while making their function clearer.

## Meaning-preservation checks
- Claims preserved: Same examples, relative errors, refs, and label audit counts.
- Terms aligned with glossary: "local score decompositions", "feature ids", "signed contributions", "residuals", "feature labels".
- Numbers/citations preserved: \(0.185\), \(0.003\), 99/67/18/14 preserved; refs preserved.
- Possible meaning changes or risks: "label-based interpretive support" is weaker than "semantic evidence"; this is intentional claim discipline.

## Flags
- Original phrase "semantic evidence" is claim-sensitive. Prefer the revised wording unless the paper adds an explicit semantic-label validation protocol.

## Location
`sections/results_and_analysis.tex`, Subsection "Target Choice Selects Signed Terms from Shared Readout Feature Projections".

## Purpose
Show that support/opposition is target-conditioned, not a property of the projection vector alone.

## Revised text
```tex
How does target choice change the signed terms when the hidden states are fixed?
We compare scoring questions on the same hidden states, from reference contrasts
to sharper pair and group margins. This isolates the target-coefficient side of
\(c_i(h,q)=\beta_i(q)h^\top d_i\).

\paragraph{Target-family reliability.}
Target coefficients turn projections into signed contributions: a feature may
support a raw logit, drop out after mean subtraction, or become negative evidence
when the comparison token uses it more.

In the five-model Qwen/Gemma subset, 301 top-token-versus-vocabulary-mean cases
give 1,505 model--case rows with median \(\rho=0.021\) and perfect sign
agreement. Pairwise/group contrasts are harder: top-1 versus top-5 margins have
median \(\rho=0.253\), while top-1 versus top-2 margins fall to \(0.824\) sign
agreement. This pattern shows why each display must name its scalar readout
target: reference contrasts are easy to reconstruct, while sharper competitor
margins are the stress test. We interpret each display at its stated score and
report local sign/error; full counts/examples are in
\Cref{tab:app-general-readout-query-families} and
Appendix~\ref{app:readout-target-family-display-example}. These target-family
comparisons show that changing \(q\) changes \(\beta_i(q)\), and therefore which
readout features count as support or opposition. They do not make the projection
vector itself target-specific. The next section holds the token row fixed and
changes the prompt, isolating \(h^\top d_i\).
```

## Why this edit helps
- States the question and the isolated factor before the result.
- Keeps the numeric evidence intact.
- Adds the boundary that projection vectors are not support/opposition until a target supplies coefficients.

## Meaning-preservation checks
- Claims preserved: Same model subset, counts, median \(\rho\), sign agreement, and interpretation.
- Terms aligned with glossary: "scalar readout target", "target coefficients", "readout feature", "support or opposition", "projection vector".
- Numbers/citations preserved: 301, 1,505, \(0.021\), \(0.253\), \(0.824\), refs preserved.
- Possible meaning changes or risks: None substantive.

## Flags
- `tab:app-general-readout-query-families` contains "query" only as a label. Keep prose aligned on "target".

## Location
`sections/results_and_analysis.tex`, Subsection "Context Reweights Fixed Token-Row Coefficients".

## Purpose
Show that context changes hidden-state projections while token-row coefficients stay fixed.

## Revised text
```tex
How does prompt context change the decomposition when the selected token row is
fixed? Here the token-row coefficients stay fixed and only the hidden-state
projections \(h^\top d_i\) change, so bar changes are evidence of context
reweighting in the readout-feature projection coordinates.

\paragraph{Context reweighting.}
In \Cref{fig:same-token-polysemy}, prompt context reweights \texttt{bug},
\texttt{ring}, and \texttt{bridge} toward features whose row labels match the
setting; each centered-token score has \(\rho<0.05\). \texttt{bug} shifts from
mosquito/bee to defect/crash/debug labels, and \texttt{bridge} from
river/crossing to networking labels. These cases support context-dependent
reweighting of readout feature projections for a fixed token row; they do not
show a change in the token-row coefficients or a generation-level intervention.
Additional views are in Appendix~\ref{app:additional-qwen-display-examples}.
```

## Why this edit helps
- Makes the control explicit: same token row, changed prompt.
- Names the measured mechanism: \(h^\top d_i\), not feature activation.
- Adds a clear boundary at the end.

## Meaning-preservation checks
- Claims preserved: Same examples and \(\rho<0.05\).
- Terms aligned with glossary: "token-row coefficients", "hidden-state projections", "readout feature projections".
- Numbers/citations preserved: \(\rho<0.05\), figure and appendix refs preserved.
- Possible meaning changes or risks: "evidence of context reweighting" is appropriate because the design holds token row fixed.

## Flags
- No overclaim found.

## Location
`sections/results_and_analysis.tex`, Subsection "Readout Feature Terms Compose with Direct Logit Attribution".

## Purpose
Position feature-resolved DLA as additive component attribution, not causal circuit discovery.

## Revised text
```tex
Can local SRP terms be split across residual-stream components when an additive
decomposition of the final hidden state is available? Because SRP terms are
linear in the final hidden state, any additive residual-stream decomposition can
pass through the same readout-feature basis. SRP therefore composes with
residual-stream direct logit attribution (DLA)
\citep{elhage2021framework,wang2022ioi,nguyen2024logitprisms}. If
\(h\approx\sum_c h_c\), an SAE feature term for target \(q\) splits into
\[
    M_{c,i}(q)=\beta_i(q)\,h_c^\top d_i .
\]
For \(q=w_A-w_B\), \(\beta_i(q)=z_{A,i}-z_{B,i}\); for
\(q=w_A-\bar{w}_{\mathcal{V}}\),
\(\beta_i(q)=z_{A,i}-\bar{z}_{\mathcal{V},i}\). This is
component-by-feature attribution for a realized forward pass; causal necessity
still requires interventions \citep{geiger2023causalabstraction}.

\Cref{fig:qwen-prism-dla-verify-assume-main} decomposes a Qwen3.5-2B
\texttt{verify}-minus-\texttt{assume} target. The exact margin is \(+4.31\), DLA
sum \(+4.36\), and sparse feature sum \(+4.28\) (\(\rho=0.009\)), separating
verification/checking support from assumption-family opposition. This supports
additive component-by-feature accounting for this realized forward pass, not a
claim that the displayed components are necessary causal mechanisms. A
row-centered \texttt{verify} companion appears in
Appendix~\ref{app:feature-resolved-dla}.
```

## Why this edit helps
- Begins with the experiment question.
- Keeps the equation and citations intact.
- Ends with the causal boundary the glossary asks for.

## Meaning-preservation checks
- Claims preserved: Same DLA composition identity, citations, exact margin, DLA sum, sparse feature sum, and \(\rho\).
- Terms aligned with glossary: "feature-resolved DLA", "component-by-feature attribution", "realized forward pass", "causal necessity".
- Numbers/citations preserved: \(+4.31\), \(+4.36\), \(+4.28\), \(\rho=0.009\), all citations and refs preserved.
- Possible meaning changes or risks: "not necessary causal mechanisms" weakens potential implication; consistent with core claim.

## Flags
- No overclaim found after revision. Current text already contains the needed causal caveat.

## Location
`sections/results_and_analysis.tex`, Subsection "Selected Readout Features Move Held-Out Lexical Scores".

## Purpose
Show the readout edit as a candidate-constrained score-level intervention with explicit limits.

## Revised text
```tex
Do directions selected from local bars transfer to held-out lexical readout
scores? We treat this as a candidate readout-side intervention: the test edits
specified logits along selected readout SAE decoder directions and measures
lexical logit differences, not downstream generation behavior.

\paragraph{Constrained readout edit.}
The test edits logits along ten Qwen3.5-2B decoder directions selected from five
discovery terms, then measures specified lexical logit differences on discovery
and held-out tokens.

\Cref{tab:main-readout-edit} shows that readout SAE directions selected from
discovery tokens move held-out lexical logit differences (flip rate \(0.359\)),
matching the oracle held-out flip rate with larger distribution shift; a
discovery-token-only bias has no held-out effect.
Appendix~\ref{app:lexical-control-stress-test} repeats this on
R1-Distill-Qwen-7B. These results show readout-score movement from discovered
directions in a candidate-constrained test; downstream generation behavior is a
separate evaluation layer.
```

## Why this edit helps
- Makes the intervention scope explicit before the table.
- Uses glossary-preferred "candidate readout-side intervention" and "candidate-constrained test".
- Keeps the score-level boundary at the end.

## Meaning-preservation checks
- Claims preserved: Same edit design, held-out flip rate, oracle comparison, distribution shift caveat, and appendix replication.
- Terms aligned with glossary: "readout-side intervention", "readout SAE decoder directions", "lexical logit differences", "candidate-constrained test".
- Numbers/citations preserved: Ten directions, five discovery terms, flip rate \(0.359\), table and appendix refs preserved.
- Possible meaning changes or risks: "intervention" is acceptable here because the experiment directly edits logits/directions, but it remains readout-level.

## Flags
- Avoid "steering" or "control" in this section unless paired with "constrained", "readout-side", and score-level metrics.

## Location
`sections/related_work.tex`, Related Work.

## Purpose
Position SRP against lenses, output-embedding geometry, sparse dictionaries, attribution, and circuits without overstating novelty or causality.

## Revised text
```tex
\section{Related Work}

\paragraph{Lens-style readouts.}
Lens-style methods expose hidden states through vocabulary distributions,
future-token predictions, continuations, or descriptions
\citep{nostalgebraist2020,belrose2023tuned,pal2023future,ghandeharioun2024patchscopes}.
They are closest to SRP in interface: asking what a hidden state reads out. Lens
variants change the map, layer, or representation; SRP instead keeps the
selected readout score fixed and decomposes it into sparse feature terms plus an
explicit residual.

\paragraph{Output-embedding geometry.}
SRP factorizes the final unembedding, or LM head. Output embeddings have been
studied as learned representations and prediction-shaping geometry
\citep{press2017outputembedding,inan2017tying,yang2018softmaxbottleneck,bostrom2020bpe}.
Token rows, weight tying, softmax rank constraints, and vocabulary segmentation
affect final scores. SRP makes this readout geometry explicit through sparse
token-row coefficients and shared readout directions: target coefficients are
fixed by unembedding-row geometry, while context enters through hidden-state
projections.

\paragraph{Sparse dictionaries.}
Activation-SAE work learns dictionaries over hidden-state distributions and has
established evaluation practice for sparse features
\citep{bricken2023monosemanticity,cunningham2023sparse,gao2024scaling,karvonen2025saebench,makelov2024principled}.
SRP uses the machinery but fits unembedding rows rather than activation vectors:
codes are token-row coefficients, decoder directions are LM-head readout
directions, and \(h^\top d_i\) is a readout feature projection rather than an
activation-SAE code. Local signed terms form only after a scalar readout target
supplies a coefficient.

\paragraph{Logit attribution and circuits.}
Component attribution and logit-prism methods ask where logit evidence is
written in the residual stream
\citep{elhage2021framework,wang2022ioi,nguyen2024logitprisms}. Automated circuit
discovery, causal abstraction, transcoders, and circuit-tracing systems pursue
internal mechanisms and intervention-tested structure
\citep{conmy2023acdc,geiger2023causalabstraction,dunefsky2024transcoders,ameisen2025circuittracing}.
SRP factorizes the rows against which the final hidden state is scored. Because
its terms are linear, signed readout-feature contributions can be split across
residual components in a realized forward pass. This is additive readout-level
accounting; causal interventions still test necessity. Broader positioning
appears in Appendix~\ref{app:extended-related-work}.
```

## Why this edit helps
- Clarifies that SRP is closest to lenses by interface, not by being another learned lens.
- Uses "readout feature projection" instead of "projection" alone in the SAE comparison.
- Tightens the final causal boundary and removes repeated "instead".

## Meaning-preservation checks
- Claims preserved: Same related-work categories and citations.
- Terms aligned with glossary: "lens-style readouts", "readout score", "explicit residual", "token-row coefficients", "readout feature projection", "additive readout-level accounting".
- Numbers/citations preserved: All original citations and appendix ref preserved.
- Possible meaning changes or risks: "SRP instead keeps..." is narrowed to "selected readout score"; this improves claim precision.

## Flags
- No unsupported priority claim found. The section avoids "first", "novel", and "state-of-the-art".

## Location
`sections/conclusion.tex`, Conclusion.

## Purpose
Restate the contribution, evidence standard, and boundaries without expanding the claims.

## Revised text
```tex
\section{Conclusion}

Sparse Readout Prism factorizes the LM head to turn a selected linear readout
score from an opaque dot product into signed readout-feature terms with an
explicit residual in raw logit coordinates. Replacement, target-reconstruction,
sign, and baseline diagnostics identify settings where those terms preserve the
selected readout behavior. In those residual-checked settings, SRP compares
targets and contexts, composes with residual-stream attribution, and supplies
candidate directions for constrained readout-side edits. Causal intervention and
generation-level evaluation remain separate tests; SRP provides a
residual-checked readout-linear basis for choosing where they should focus.

\paragraph{Future work.}
SRP applies to linear readout maps, so a natural next step is to compose it with
tuned lenses, logit-prism variants, and future-token lenses, factorizing the
effective readout used at each layer rather than only the final LM head. This
would allow lens methods to be compared by decoded tokens and by their signed
sparse feature terms and residuals. A complementary direction is a layerwise
residual-stream census: track readout-feature projections before target
selection to study when evidence becomes readout-readable, when it becomes
target-specific, and which residual components supply the corresponding feature
terms.
```

## Why this edit helps
- Adds "linear" to the first sentence, matching the scope.
- Replaces "preserve readout behavior" with "preserve the selected readout behavior".
- Uses "candidate directions" for readout-side edits, keeping intervention claims bounded.

## Meaning-preservation checks
- Claims preserved: Same conclusion and future-work directions.
- Terms aligned with glossary: "selected linear readout score", "signed readout-feature terms", "residual-checked", "candidate directions", "readout-feature projections".
- Numbers/citations preserved: No numbers/citations in original conclusion.
- Possible meaning changes or risks: Slightly narrows "readout behavior" to "selected readout behavior"; this is claim discipline, not technical change.

## Flags
- No overclaim found after adding "linear" and "candidate".

## Location
`sections/limitations.tex`, Limitations.

## Purpose
Make the validity conditions, label caveats, readout-specific caveats, and edit boundary easy for reviewers to find.

## Revised text
```tex
\section*{Limitations}

\paragraph{Scope of score decompositions.}
SRP gives local score decompositions for linear scalar readout targets at the
final LM head. Feature-resolved DLA composes these decompositions with additive
residual-stream attribution, yielding candidate intervention targets rather than
causal circuit claims.

\paragraph{Fidelity and labels.}
Reported diagnostics expose SRP residuals and sign checks in their operating
regimes, but as in other sparse dictionary settings the learned basis can depend
on recipe choices
\citep{elhage2022superposition,karvonen2025saebench,makelov2024principled}. We
therefore report reconstruction error, sign agreement, replacement fidelity, and
stability checks. Feature labels summarize high-scoring unembedding rows; the
measured objects are feature ids, signed terms, and residual errors.

\paragraph{Readout-specific caveats.}
SRP supports token-level, group-contrast, and benchmark-derived readout
decompositions while inheriting vocabulary-geometry caveats: labels and
groupings can reflect tokenizer geometry, token frequency, row norms, or
surface-form artifacts
\citep{press2017outputembedding,inan2017tying,yang2018softmaxbottleneck,bostrom2020bpe}.
Group contrasts reduce single-row dependence; benchmark-derived targets measure
linear readout reconstruction, not task solving; and the readout edit is a
constrained lexical-score test. SRP provides a residual-checked readout-linear
basis for candidate evidence terms, while causal mechanisms and generation-level
effects require targeted interventions.
```

## Why this edit helps
- Keeps limitations focused on the exact ARR reviewer questions: scope, validity, labels, and non-claims.
- Changes "intervention candidates" to "candidate intervention targets" for readability.
- Preserves all citations and caveats.

## Meaning-preservation checks
- Claims preserved: Same limitations and citations.
- Terms aligned with glossary: "linear scalar readout targets", "local score decompositions", "candidate intervention targets", "feature labels", "benchmark-derived targets", "constrained lexical-score test".
- Numbers/citations preserved: All original citations preserved.
- Possible meaning changes or risks: None substantive.

## Flags
- The limitations are already well aligned with the core claim. No unsupported claim found.

## Location
`sections/limitations.tex`, Ethical Considerations.

## Purpose
Keep dual-use and deployment-scope language bounded to readout-level inspection and stress testing.

## Revised text
```tex
\section*{Ethical Considerations}

This work studies pretrained language-model readouts. It collects no new
human-subject data and evaluates no deployed systems. Experiments use public
checkpoints and held-out text-derived hidden states, including C4 continuations;
C4 may contain offensive, private, copyrighted, or sensitive text
\citep{raffel2020exploring,dodge2021documenting}. We use it only to measure
readout reconstruction and make no social-validity claims about individual
tokens, labels, or generated outputs.

SRP makes lexical readout directions and row-level labels easier to inspect and,
in the readout-edit stress test, move. This is dual use: auditing tools could
also inform output manipulation. We frame the edit as a readout-level stress
test, report distribution shift and off-target probes, and reserve
safety-filter, moderation, or deployment-control claims for direct system-level
evaluations.
```

## Why this edit helps
- Current wording is already restrained and aligned.
- No rewrite needed beyond preserving it as the recommended version.

## Meaning-preservation checks
- Claims preserved: Same ethical scope and dual-use boundary.
- Terms aligned with glossary: "readout reconstruction", "row-level labels", "readout-level stress test".
- Numbers/citations preserved: C4 citations preserved.
- Possible meaning changes or risks: None.

## Flags
- No overclaim found.
