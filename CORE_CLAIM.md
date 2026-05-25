# Core Claim and Paper Spine

Purpose: this file is a writing brief for future agents working on the paper.
Use it to keep the abstract, introduction, results framing, limitations, and
conclusion aligned around the same claim.

## TL;DR

Sparse Readout Prism (SRP) is a **sparse feature decomposition for logit lens
readouts**, built from a sparse dictionary factorization of the final
unembedding / LM head. The main contribution is the readout-row basis and the
readout-level analyses it enables. The learned basis can reconstruct or replace
the full LM head, project hidden states into readout-feature coordinates,
support local decompositions of selected readout scores, compose with
residual-stream attribution, and suggest narrow readout-side edit directions.
For local score decompositions, use standard readout objects first: token
logits, logit differences, logit-difference directions, and fixed linear
combinations of LM-head rows. The selected readout direction specifies which
score is decomposed and what sign convention is used, yielding sparse signed
feature terms plus an explicit reconstruction residual for that score.

Use "better logit lens" as the front door, not the ceiling. The motivating
interface is lens style decoding; the technical contribution is a sparse
dictionary factorization of the final unembedding matrix. That factorization
turns the LM head into a reusable readout-feature basis whose uses are checked
against reconstruction and replacement error.

## Literature Grounding

Use standard terms where the literature already has them, and introduce SRP
terms only as paper local specializations.

- **Lens style readout / logit lens:** grounded in work that decodes hidden
  states into vocabulary logits or distributions using the unembedding or a
  learned lens (`nostalgebraist2020`, `belrose2023tuned`, `pal2023future`,
  `ghandeharioun2024patchscopes`). Avoid implying that "sparse feature
  decomposition" is a standard lens term; the standard object is the logit lens,
  and the sparse feature decomposition is SRP's contribution.
- **Readout / unembedding / LM head / output embedding:** grounded in the final
  linear map from hidden states to vocabulary logits and related output embedding
  work (`press2017outputembedding`, `inan2017tying`,
  `yang2018softmaxbottleneck`, `bostrom2020bpe`). In this paper, "readout" is a
  concise alias for the final unembedding / LM head.
- **Sparse dictionary / sparse autoencoder:** grounded in SAE and dictionary
  learning work (`makhzani2013ksparse`, `bricken2023monosemanticity`,
  `cunningham2023sparse`, `gao2024scaling`, `karvonen2025saebench`,
  `makelov2024principled`). Use "sparse code", "decoder direction", and
  "reconstruction error" in their standard SAE senses.
- **Additive logit attribution:** grounded in residual stream decomposition,
  direct logit attribution, and logit prism work (`elhage2021framework`,
  `wang2022ioi`, `nguyen2024logitprisms`). SRP composes with these methods but
  does not by itself establish causal mechanisms.
- **Token logits, logit differences, and readout directions:** token logits,
  logit differences, logit-difference directions, and projections onto
  unembedding rows are standard in logit-lens, DLA, and circuit-analysis work.
  Multi-token label mappings and group-like token sets also have adjacent
  precedents in verbalizer and prompt-classification work. Do not frame
  "scalar readout target" or "linear readout target" as standard terminology or
  as a standalone novelty claim. SRP's role is narrower: it applies the learned
  readout basis to selected token logits, logit differences, and fixed linear
  combinations of LM-head rows.
- **Paper local terms:** "readout feature", "readout feature projection",
  "projection profile", "selected readout direction", "selected readout
  score", and "local score decomposition" are SRP terms or paper-local
  shorthand. Define them on first use and tie each one to the standard object it
  specializes.

## Canonical Claim

SRP learns a sparse basis for the final unembedding / LM head by decomposing
token rows into sparse SAE codes over shared decoder directions. This yields a
reusable readout-side basis for hidden states. Before any token, contrast, or
other readout score is selected, the basis can reconstruct the full logit vector,
replace the LM head, and provide score-independent readout feature projections.
When a selected score fixes a readout direction--for example an unembedding row,
a logit-difference direction, a group or vocabulary-mean contrast, or a fixed
top-competitor margin--the same basis yields feature coefficients for that
direction, signed readout-feature terms, and an explicit reconstruction residual
for that score. The paper's evidence is that this factorized readout preserves the
original LM-head behavior and selected scores in measured regimes,
supporting the corresponding readout-level analyses when diagnostics pass.

## Paper Spine

Lens style methods expose a hidden state through vocabulary logits, token
preferences, or related summaries. This makes them useful inspection tools, but
each chosen logit or contrast is still a single dot product with an unembedding
row or logit-difference direction. SRP keeps the logit lens interface and
factorizes the readout behind it: the unembedding matrix is decomposed into
sparse token row codes and shared SAE decoder directions. Projecting a hidden
state onto those directions gives readout feature projections before a token,
contrast, or margin is selected. The same basis supports full readout
reconstruction, score-independent projection profiles, local decompositions of
selected scores, feature-resolved DLA, and constrained readout-side edits. For a
local score decomposition, the selected readout direction turns projections into
signed feature contributions for a chosen score. The paper shows that the
factorized readout can replace the LM head in useful regimes; that selected
logits, logit differences, and row-combination contrasts can be reconstructed
across models and score families; and that, under these checks, the basis
separates score choice, context reweighting, feature-resolved DLA, and
constrained readout-side edits.

## Narrative Hierarchy

Use this hierarchy when writing high level prose:

1. **Motivating interface:** the logit lens asks what a hidden state reads out.
2. **Gap:** lens scores identify favored tokens or contrasts, but leave each
   selected score as an unembedding row dot product.
3. **Technical move:** SRP factorizes the unembedding matrix into sparse
   token row codes and shared SAE decoder directions.
4. **Basis uses:** the learned basis supports full readout reconstruction,
   score-independent projection profiles, local decompositions of selected
   logits and contrasts, feature-resolved DLA, and readout-side edit directions.
5. **Selected-score use:** a token logit, logit difference, or other fixed
   linear combination of LM-head rows selects weighted sums of readout feature
   projections and defines support/opposition for one score.
6. **Auditability:** full readout uses are checked by replacement and
   reconstruction; local score decompositions report signed feature terms plus
   offset/residual terms.
7. **Evidence:** LM head replacement fidelity, score reconstruction, sign
   agreement, and baselines identify where the factorized readout is
   trustworthy.

This lets the paper say "better logit lens" without sounding like it only
contributes a new visualization of token pair examples, while avoiding the
overclaim that token logits, logit differences, or token/group comparisons are
new by themselves.

## Technical Story

SRP learns a sparse dictionary over unembedding rows:

```tex
W_U \approx \mathbf{1}\mu^\top + ZD_{\mathrm{feat}}
```

where `Z` contains sparse token row codes and `D_{\mathrm{feat}}` contains SAE
decoder directions as rows.

For a hidden state `h`, the factorization gives:

```tex
W_U h \approx \mathbf{1}(\mu^\top h) + Z(D_{\mathrm{feat}}h)
```

This reconstructed logit vector is already a use of SRP: it tests whether the
factorized readout can stand in for the original LM head before any particular
token, contrast, or margin is selected.

The vector `D_{\mathrm{feat}}h` is the hidden state's **readout feature
projection vector** before score selection: it says how strongly the hidden
state projects onto each SAE decoder direction before any token, contrast,
margin, or other readout score has been selected. "Profile" can be used as informal
shorthand for the ranked pattern of these projections, but the technical term
should be "projection vector" or "readout feature projections."
Projection vectors can be compared across prompts, layers, or residual-stream
components without selecting a score. Support and opposition are defined only
after a selected readout direction supplies coefficients.

For a local score decomposition, choose a standard readout object--a token
logit, logit difference, or fixed linear combination of LM-head rows--and write
its readout direction as `q`. This applies the learned readout basis to the
selected score and converts the projection vector into signed feature
contributions at the current hidden state:

```tex
c_i(h, q) = \beta_i(q) \, h^\top d_i
```

The coefficient `\beta_i(q)` says how the selected readout direction uses
feature `i`. The hidden-state projection `h^\top d_i` says how aligned the
current context is with readout feature direction `d_i`. A feature contributes
locally only through the product of both factors.

## Terminology

Use the following terms consistently:

- `d_i`: **readout feature direction** or **SAE decoder direction**.
- `z_{t,i}`: **sparse code entry**, **sparse coefficient**, or **token row
  coefficient**.
- `p_i(h)=h^\top d_i`: **readout feature projection**.
- `p(h)=D_{\mathrm{feat}}h`: **readout feature projection vector** or
  **projection profile**.
- `\beta_i(q)`: **direction coefficient** or **coefficient for the selected
  readout direction**.
- `c_i(h,q)=\beta_i(q)p_i(h)`: **signed feature contribution** or **feature
  term for the selected score**.

Avoid calling `p_i(h)=h^\top d_i` a feature activation in technical prose. In
SAE literature, "feature activation" usually refers to the sparse encoder output
for the object the SAE was trained on. Here the SAE is trained on unembedding
rows, so the sparse activations/codes are the token row sparse codes
`z_{t,i}`. The hidden state enters later through projection onto the learned
SAE decoder directions. If "profile activation" appears in informal prose, make
clear that it means a projection coordinate, not an SAE encoder activation.

## Conceptual Order

Use this order when explaining the paper:

1. Lens style methods reveal what a hidden state reads out, but not what sparse
   readout features support the score.
2. SRP factorizes the unembedding matrix.
3. The factorization can reconstruct or replace the full LM head.
4. The factorization gives reusable readout feature projections,
   `D_{\mathrm{feat}}h`, before any score is selected.
5. Projection profiles, feature-resolved DLA, and readout-side edit directions
   can use the basis without making selected scores the central object.
6. A selected readout direction--usually an unembedding row,
   logit-difference direction, or explicit row-combination contrast--specifies
   the score and turns projection vectors into feature coefficients and signed
   feature contributions.
7. Score decompositions at a hidden state report signed feature terms and
   residual/error terms.
8. Experiments measure when LM head replacement and score reconstruction
   support interpreting those decompositions.

This order matters. Starting with a logit lens example is useful; staying there
too long makes SRP sound narrower than it is. Move quickly from the example to
the readout factorization and sparse feature basis.

## Selected Readout Score Scope

Do not present "scalar readout target" or "linear readout target" as standard
mechanistic-interpretability terminology. Use the standard names whenever
possible: token logits, logit differences, logit-difference directions,
unembedding rows, and projections onto those directions. SRP can also decompose
other explicitly defined scalar linear functions of readout rows:

- raw token logits,
- centered or reference logits,
- pairwise logit differences,
- group contrasts,
- local competitor margins,
- benchmark-derived readout contrasts,
- other explicitly defined linear directions, if their coefficients and
  residual/error terms are reported.

Avoid saying SRP directly explains arbitrary nonlinear quantities. Probabilities,
loss, task accuracy, full generations, and sequence level behavior require a
chosen linear proxy, local linearization, or additional evaluation.

## Validation Framing

Validation is essential, but it should not be the headline claim. The headline
claim is the sparse feature decomposition enabled by the SRP factorization.
Selected readout scores are the local-score interface, not the only way to use
the SRP basis and not a standalone novelty claim. Validation is the standard
that determines when each use of the basis can be interpreted.

Use these diagnostics consistently:

- LM head replacement fidelity,
- score reconstruction error,
- sign agreement for signed targets,
- null and reference baselines,
- local residual/error terms on displayed decompositions.

Preferred stance:

> SRP is a sparse feature decomposition for logit lens readouts, built from a
> sparse factorization of the final unembedding / LM head. The learned basis
> supports full readout reconstruction, score-independent projection profiles,
> and local decompositions of selected readout scores. For the last case, a
> token logit, logit difference, or explicit row-combination contrast fixes a
> readout direction, which turns readout feature projections into signed feature
> terms. The experiments identify regimes where these uses preserve the original
> readout distributions and selected scores.

Avoid making the main spine "SRP explains scores and tells us when the
decomposition is faithful." That wording undersells the readout factorization
and makes validation sound like a novelty rather than a normal requirement.

## Reviewer Facing Emphasis

For ARR, the safest and strongest contributions are:

- **Sparse readout factorization:** SRP factorizes the final unembedding / LM
  head into sparse token-row codes over shared SAE decoder directions, yielding
  a readout-specific sparse feature basis.
- **Score-independent projections:** projecting hidden states onto the learned
  decoder directions gives readout feature projections before any token,
  contrast, or margin is selected.
- **Local score decompositions:** selected token logits, logit differences, and
  fixed LM-head row combinations decompose into signed feature terms plus
  explicit residual/error terms.
- **Compositional readout analyses:** the same basis supports full LM-head
  reconstruction or replacement, feature-resolved DLA, and constrained
  readout-side edits.
- **Empirical demonstrations:** the experiments show that the basis separates
  score choice, context dependence, feature competition, feature-resolved DLA,
  and narrow readout-side lexical score movement in measured settings.

Present validation as the evidence standard for these contributions, not as a
separate contribution. Replacement fidelity, score reconstruction, sign
agreement, null/reference baselines, and local residuals identify when the
readout uses above can be interpreted.

## Language To Prefer

- "sparse feature decomposition"
- "scores decomposed into feature terms"
- "sparse feature basis for the readout"
- "logit lens readout with sparse feature terms"
- "factorized readout"
- "sparse dictionary factorization of the unembedding / LM head"
- "readout feature projection vector"
- "readout feature projections"
- "projection profile before score selection" when a higher level shorthand is
  useful
- "full readout reconstruction"
- "score-independent projection profile"
- "local score decomposition"
- "token logit"
- "logit difference"
- "logit-difference direction"
- "projection onto an unembedding row"
- "fixed linear combination of LM-head rows"
- "selected readout direction `q`"
- "selected readout score `h^\top q`"
- "signed feature contribution for the chosen score"
- "signed feature terms plus residual/error terms"
- "LM head replacement fidelity" and "score reconstruction"

## Language To Avoid

- "SRP explains model behavior" unless tightly qualified.
- "SRP proves causality" or "causal explanation."
- "SRP discovers semantic features" without caveats.
- "Feature labels show..." when the measured object is a feature ID,
  contribution, projection, or residual.
- "Feature activation" for `h^\top d_i`; use "readout feature projection"
  instead.
- "SRP is only a better logit lens" or "SRP is just a logit lens visualization."
- "Targets outside the readout" unless the target is explicitly defined as a
  linear readout direction or proxy compatible with the SRP decomposition.
- "Query" as the formal term; prefer the specific object, such as "token
  logit", "logit difference", "group contrast", or "selected readout score".
- "Scalar readout targets are a standalone contribution" or "SRP formalizes
  token/group comparisons for the first time." Prefer saying that SRP applies a
  sparse readout basis to standard logits, logit differences, and explicit
  row-combination contrasts.
- "Scalar readout targets are the interface for SRP" or other wording implying
  every use of the learned basis requires a selected score.

## Short Slogans

- A logit lens tells you the score; SRP supplies a sparse basis for the readout.
- A sparse feature decomposition for logit lens readouts.
- Factor the unembedding; project hidden states through readout features.
- From vocabulary readouts to reusable readout feature projections.
- From lens outputs to sparse feature decompositions.
