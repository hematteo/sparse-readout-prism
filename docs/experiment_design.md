# Experiment Design and Claim Boundaries

This note is the paper-ready map from experiments to claims. It is meant to
make the design legible before a reader gets to individual tables, figures, or
scripts.

## One-sentence design

Sparse Readout Prism trains a sparse factorizer on unembedding rows only, then
evaluates whether the reconstructed readout preserves the model-facing logits
and readout queries that later feature displays explain.

## Data roles

| Data / artifact | Used for | Not used for |
|---|---|---|
| `W_U` rows | Training the readout-feature factorizer and computing row reconstruction metrics. | Prompt/task supervision. |
| C4 hidden states | Held-out readout replacement diagnostics: top-token agreement and KL. | Training feature labels or task-specific query selection. |
| Curated A/B query bank | Query-level fidelity: pairwise margins, residuals, sign agreement, and feature-display examples. | Factorizer training. |
| Benchmark-derived task target bank | Task-format query fidelity and display examples built from benchmark answer/distractor or label/action groups. | Factorizer training or benchmark task accuracy. |
| Model-native prompt bank | Distributional readout-query and replacement diagnostics. | Curated case-study selection. |
| Case-candidate bank | Stress examples and contrastive sign-flip diagnostics. | Headline training signal. |
| Baseline/null outputs | Comparison under the same query bank and metrics. | A separate benchmark with different rules. |

The important separation is: the sparse dictionary is learned from `W_U`; all
prompt banks and hidden states are evaluation or display inputs.

## Experiment ladder

| Claim | Experiment | Unit | Metric | Interpretation |
|---|---|---|---|---|
| The factorizer reconstructs readout rows. | Row reconstruction on held-out/readout rows. | Vocabulary row. | rowEV, row cosine, usage diagnostics. | Capacity diagnostic only; high rowEV does not certify readout behavior. |
| Replacing `W_U` with reconstructed `W_U` preserves model-facing readouts. | Hidden-state replacement evaluation. | C4 position / hidden state. | Top-1 agreement, KL bits, top-k overlap. | Tests whether the reconstructed readout behaves like the original readout on natural hidden states. |
| The sparse feature sum preserves the scalar query being interpreted. | Query-fidelity evaluation. | Prompt-query row, clustered by `base_case_id` for uncertainty. | Sign agreement, relative residual `rho`, `rho < threshold`. | Main gate for whether a query-level feature display is interpretable. |
| Learned feature support matters. | Baseline/null comparison. | Same prompt-query rows as Sparse Readout Prism. | Same residual/sign metrics. | Rules out explanations that come merely from sparse displays, dense low-rank reconstruction, or lexical nearest-row structure. |
| Query choice changes reliability and meaning. | Controlled query-family suite. | Query family and prompt-query row. | Residual/sign by family. | Shows selected logits, pairwise margins, family contrasts, and local competitor margins answer different questions. |
| Row reconstruction and readout fidelity can diverge. | Cross-family and post-training stress tests. | Model / operating point. | rowEV versus top1/KL/query residuals. | Defines boundaries; not every high-rowEV factorizer supports feature-level claims. |

## Fidelity gate

Every feature-level display should be read through the local query gate:

```
rho = |s_exact - s_reconstructed| / max(|s_exact|, delta)
```

where `s_exact` is the original readout-query score and `s_reconstructed` is
the score obtained from the sparse reconstruction. The paper uses a denominator
floor for small margins so near-zero ties do not create unstable ratios. For
signed contrasts, the reconstructed sign must also match the exact sign.

A display is interpretable as sparse readout accounting only when the local
query passes the residual/sign checks. If the residual dominates, the sign
flips, or the target tokens fail tokenization controls, the result should be
reported as a failure or withheld from feature-level interpretation.

## Selection versus final evaluation

Operating points are selected by reconstruction, replacement, usage, and
query-fidelity diagnostics. The selected checkpoint is then evaluated on the
frozen query banks with the same residual/sign definitions used by the figures.
When reporting results, keep these roles separate:

- Selection diagnostics choose an operating point.
- Headline query metrics evaluate the selected operating point.
- Feature displays are local examples that must pass their own residual/sign
  gate.
- Stress-test rows show boundary behavior and should not be mixed into the
  primary Qwen feature-display claim unless they pass the same gate.

## Experimental units

The same benchmark can be counted at several granularities. Reports should name
the unit explicitly:

- `case_id`: one concrete prompt and target specification.
- `base_case_id`: a cluster of sibling variants; use this for bootstrap
  uncertainty so paraphrases are not treated as independent evidence.
- prompt-query row: one evaluated scalar query for one prompt/model/checkpoint.
- model-case cell: one model and operating point evaluated on one case.
- query family: selected logit, vs-reference, pairwise, family contrast, or
  local competitor.

## Allowed interpretations

Low-residual Sparse Readout Prism decompositions support claims about how the
final readout scores a chosen hidden state and query in the fitted sparse basis.
They do not, by themselves, identify why the network produced that hidden
state, prove that a feature is a human semantic concept, or establish a unique
causal mechanism. Those claims require separate activation/component
attribution, interventions, and residual accounting.

## The design in one paragraph

The experiments form a ladder from reconstruction to interpretation. First, we
train the sparse factorizer only on unembedding rows and measure row
reconstruction as a capacity diagnostic. Second, we replace the original readout
with the reconstructed readout on held-out hidden states to test top-token and
distributional fidelity. Third, for the readout queries that feature displays
explain, we evaluate exact-versus-reconstructed scalar scores with sign
agreement and relative residual. A feature-level display is interpreted only
when its local query passes this residual/sign gate. Baseline and null methods
use the same query bank and metrics, and cross-family stress tests define the
boundary where row reconstruction no longer implies readout-query fidelity.
