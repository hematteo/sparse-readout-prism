# Core Claim and Paper Spine

Purpose: this file is a writing brief for future agents working on the paper.
Use it to keep the abstract, introduction, results framing, limitations, and
conclusion aligned around the same claim.

## TL;DR

Sparse Readout Prism (SRP) is a **sparse feature decomposition for logit lens
readouts**, built from a sparse dictionary factorization of the final
unembedding / LM head. It starts from the familiar lens question--what
vocabulary evidence does this hidden state expose through the readout?--but
decomposes selected readout scores into sparse feature terms, signed
contributions for the chosen target, and explicit reconstruction residuals.

Use "better logit lens" as the front door, not the ceiling. The motivating
interface is lens style decoding; the technical contribution is a sparse
dictionary factorization of the final unembedding matrix that decomposes
selected LM head scores into feature terms checkable against reconstruction
error.

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
- **Paper local terms:** "readout feature", "readout feature projection",
  "projection profile", "scalar readout target", and "local score
  decomposition" are SRP terms. Define them on first use and tie each one to the
  standard object it specializes.

## Canonical Claim

SRP extends the logit lens view from vocabulary scores to a sparse feature
decomposition of readout scores. It factorizes the final unembedding / LM head
matrix, decomposing token rows into sparse SAE codes over shared decoder
directions. This gives a sparse feature basis for the readout and a reusable
projection basis for final hidden states.
Scalar readout targets--linear functions of unembedding rows, such as logits,
logit differences, and group contrasts--then produce signed feature
contributions for the chosen target, with explicit reconstruction residuals and
fidelity diagnostics.

## Paper Spine

Lens style methods expose a hidden state through vocabulary logits, token
preferences, or related summaries. This makes them useful probes, but each
chosen logit or contrast is still a single dot product with an unembedding row
or readout direction. SRP keeps the logit lens interface and factorizes the
readout behind it: the unembedding matrix is decomposed into sparse token row
codes and shared SAE decoder directions. Projecting a hidden state onto those
directions gives readout feature projections before target selection; choosing
a scalar readout target then turns those projections into signed feature
contributions for the chosen target. The paper shows that this
factorized readout can replace the LM head in useful regimes, reconstruct
selected scalar targets across models and target families, and expose target
choice, context reweighting, DLA by SAE feature, and constrained readout side
edits.

## Narrative Hierarchy

Use this hierarchy when writing high level prose:

1. **Motivating interface:** the logit lens asks what a hidden state reads out.
2. **Gap:** lens scores identify favored tokens or contrasts, but leave each
   selected score as an unembedding row dot product.
3. **Technical move:** SRP factorizes the unembedding matrix into sparse
   token row codes and shared SAE decoder directions.
4. **Improved lens:** hidden states are projected onto readout features, and
   chosen scalar readout targets select weighted sums of those projections.
5. **Auditability:** decompositions report signed feature terms plus
   offset/residual terms, so displayed explanations can be checked against exact
   readout scores.
6. **Evidence:** LM head replacement fidelity, target reconstruction, sign
   agreement, and baselines identify where the factorized readout is
   trustworthy.

This lets the paper say "better logit lens" without sounding like it only
contributes a new visualization of token pair examples.

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

The vector `D_{\mathrm{feat}}h` is the hidden state's **readout feature
projection vector** before target selection: it says how strongly the hidden
state projects onto each SAE decoder direction before any token, contrast, or
scalar readout target has been selected. "Profile" can be used as informal
shorthand for the ranked pattern of these projections, but the technical term
should be "projection vector" or "readout feature projections."

For a chosen scalar readout target `q`, SRP converts the projection vector into
signed feature contributions at the current hidden state:

```tex
c_i(h, q) = \beta_i(q) \, h^\top d_i
```

The target coefficient `\beta_i(q)` says how the target uses feature `i`.
The hidden state projection `h^\top d_i` says how aligned that readout direction
is with the current context. A feature contributes locally only through the
product of both factors.

## Terminology

Use the following terms consistently:

- `d_i`: **readout feature direction** or **SAE decoder direction**.
- `z_{t,i}`: **sparse code entry**, **sparse coefficient**, or **token row
  coefficient**.
- `p_i(h)=h^\top d_i`: **readout feature projection**.
- `p(h)=D_{\mathrm{feat}}h`: **readout feature projection vector** or
  **projection profile**.
- `\beta_i(q)`: **target coefficient**.
- `c_i(h,q)=\beta_i(q)p_i(h)`: **signed feature contribution** or **signed
  readout feature contribution**.

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
3. The factorization gives reusable readout feature projections.
4. Hidden states can be projected onto those features as `D_{\mathrm{feat}}h`.
5. Scalar readout targets turn the projection vector into target coefficients
   and signed feature contributions.
6. Score decompositions at a hidden state report signed feature terms and
   residual/error terms.
7. Experiments measure when LM head replacement and target reconstruction
   support interpreting those decompositions.

This order matters. Starting with a logit lens example is useful; staying there
too long makes SRP sound narrower than it is. Move quickly from the example to
the readout factorization and sparse feature basis.

## Target Scope

SRP directly supports targets that can be represented as scalar linear
functions of readout rows, especially linear combinations of unembedding rows:

- raw token logits,
- centered or reference logits,
- pairwise logit differences,
- group contrasts,
- local competitor margins,
- benchmark readout targets,
- other explicitly defined linear directions, if their coefficients and
  residual/error terms are reported.

Avoid saying SRP directly explains arbitrary nonlinear quantities. Probabilities,
loss, task accuracy, full generations, and sequence level behavior require a
chosen linear proxy, local linearization, or additional evaluation.

## Validation Framing

Validation is essential, but it should not be the headline claim. The headline
claim is the sparse feature decomposition enabled by the SRP factorization.
Scalar readout targets are the interface for decomposing selected scores.
Validation is the standard that determines when the resulting decompositions can
be interpreted.

Use these diagnostics consistently:

- LM head replacement fidelity,
- target reconstruction error,
- sign agreement for signed targets,
- null and reference baselines,
- local residual/error terms on displayed decompositions.

Preferred stance:

> SRP is a sparse feature decomposition for logit lens readouts, built from a
> sparse factorization of the final unembedding / LM head. Scalar readout targets
> turn the resulting readout feature projections into signed feature terms for
> selected scores. The experiments identify regimes where those decompositions
> preserve the original readout behavior and selected target scores.

Avoid making the main spine "SRP explains scores and tells us when the
decomposition is faithful." That wording undersells the readout factorization
and makes validation sound like a novelty rather than a normal requirement.

## Reviewer Facing Emphasis

For ARR, the safest and strongest contribution is:

- a sparse feature decomposition for the logit lens interface;
- a new object of analysis: SAE features of the unembedding/readout matrix;
- a sparse feature basis for hidden states at the final readout;
- a scalar readout target formalism for logits, contrasts, margins, and group
  targets;
- calibrated diagnostics showing where these decompositions can be interpreted;
- demonstrations that the basis reveals target choice, context dependence,
  feature competition, DLA composition, and narrow readout side control.

## Language To Prefer

- "sparse feature decomposition"
- "scores decomposed into feature terms"
- "sparse feature basis for the readout"
- "logit lens readout with sparse feature terms"
- "factorized readout"
- "sparse dictionary factorization of the unembedding / LM head"
- "readout feature projection vector"
- "readout feature projections"
- "projection profile before target selection" when a higher level shorthand is
  useful
- "scalar readout target"
- "linear function of unembedding rows"
- "signed feature contribution for the chosen target"
- "signed feature terms plus residual/error terms"
- "LM head replacement fidelity" and "target reconstruction"

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
- "Query" as the formal term; prefer "scalar readout target" in technical prose
  and use "query" only informally.

## Short Slogans

- A logit lens tells you the score; SRP decomposes it into sparse feature terms.
- A sparse feature decomposition for logit lens readouts.
- Factor the unembedding; project hidden states through readout features.
- From vocabulary readouts to reusable readout feature projections.
- From lens outputs to sparse feature decompositions.
