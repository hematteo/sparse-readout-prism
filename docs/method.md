# Method, in 5 minutes

Sparse Readout Prism factorizes the **unembedding matrix** of a language
model into a dictionary of reusable directions, then decomposes any chosen
logit (or any linear combination of logits) into a sparse sum of
signed feature contributions.

## The setup

The final logit a model assigns to token `t` given a hidden state `h` is
just an inner product:

```
score(h, t) = h · W_U[t]
```

where `W_U ∈ R^{vocab × d_model}` is the unembedding matrix and `W_U[t]`
is its row for token `t`. Logit lens reads `h` against the *whole vocab*
and reports the top tokens; we want to read it against **a chosen
direction** and account for *why* that direction lit up.

## What we train

We train a small dictionary `(D, b)` over the **rows** of `W_U`. After
centering and per-row normalizing,

```
W_U[t] − mean ≈ row_norm[t] · (D · z_t + b)
```

where `z_t` is a sparse code (TopK / Matryoshka / JumpReLU / Gated — paper
compares four) over a feature bank of size `d_features ≫ d_model`. We
recover `W_U[t]` exactly when the residual is zero; the paper's fidelity
gate measures and reports the residual.

Importantly, this is a **sparse-coding problem over the rows of `W_U`** —
no model forward pass, no training data, no labels. Just the matrix.

## The decomposition

Once we have a trained `(D, b, row_mean, row_norm)`, any scored logit
falls out linearly:

```
h · W_U[t] = h · (row_mean + row_norm[t] · (D · z_t + b)) + residual_t
           = base + Σ_i z_{t,i} · (h · D[i]) · row_norm[t] + bias + residual_t
```

The same identity holds for any linear combination of unembedding rows:
margins (`W_U[A] − W_U[B]`), family contrasts (mean over a set), winner
vs. competitors, target vs. vocab mean. All five query kinds share the same
decomposition identity and fidelity gate.

## The fidelity gate

A per-query gate withholds the sparse explanation when the residual is
large relative to the query scale. Without this, decompositions silently
present false explanations on the cases the dictionary fails. The
threshold is cancellation-robust so symmetric contrasts (where `base`
cancels analytically) don't false-trip.

## How to read the experiments

The paper's experiments are a ladder, not one interchangeable score:

1. **Row reconstruction** tests whether the factorizer can reconstruct rows of
   `W_U`. This is a capacity diagnostic, not a feature-display certificate.
2. **Readout replacement** tests whether reconstructed `W_U` preserves logits
   on held-out hidden states (`top1`, KL, top-k overlap).
3. **Query fidelity** tests the exact scalar readout queries later decomposed
   into features (`sign agreement`, relative residual, pass rate).
4. **Feature displays** are interpreted only when their local query passes the
   residual/sign gate.
5. **Baselines and stress tests** use the same query metrics to show what the
   method beats and where its claims stop.

For the full claim-to-experiment map, see
[`experiment_design.md`](experiment_design.md).

## What changes vs. logit lens / SAEs

- **Logit lens**: reads `h` against all vocab, reports top tokens. Doesn't
  say *which features of the row* made that token high-scoring.
- **Activation SAEs**: factorize hidden states `h` into sparse codes.
  Different object, different training, separate dictionary.
- **Sparse Readout Prism**: factorizes the *unembedding rows*. Lets you
  ask "given this hidden state, which features of the readout produced
  this token's score?" — and answer it with an exact additive identity.

## Pointers

- Runtime: see the README quickstart for `from sparse_readout_prism import ...`.
- Full equations + experiments: see the accompanying paper (distributed separately from this code release).
- Experiment-design map: [`experiment_design.md`](experiment_design.md).
- Reproduce: [`reproducing.md`](reproducing.md).
