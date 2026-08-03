# SRP Rebuttal — Consolidated Results Report

Compiled 2026-07-10. Companions now at `../rebuttal/plan/{REVISION_PLAN.md,P1_EXPERIMENT_PLAN.md}`;
the active plan is `RESUBMISSION_PLAN.md`.
All experiments complete; this is the source of truth for the rebuttal text and
paper edits. Local artifacts under
`readout_prism_workspace/results/exp3v2_frontier_local/` (abbrev. `LOCAL/`);
cluster artifacts under `/nfs-share/mh2274/readout_prism_workspace/results/`
(abbrev. `CLUSTER/`). Jobs: A 52298, B 52299, C 52303, D 52304, E 52305,
E2 52313, E3 52319, F 52314, H 52318 (all COMPLETED; E2's 0.8B lane retried as E3).

**Models** (all six gated models of the paper; Gemma excluded by the paper's
interpretation gate): Qwen3.5-0.8B / 2B / 9B, Ministral-3-8B-Base,
R1-Distill-Qwen-7B, R1-Distill-Llama-8B. Dictionaries: the paper's 32x/k256
operating points throughout.

---

## 1. Causal validation of feature contributions  ✅ headline result

**Claim tested.** SRP's per-feature contribution c_i = beta_i(q)·(h·d_i) should
predict the realized margin change when h is ablated along the (unit) decoder
direction d_i. Falsifiable: sparse-code misattribution, decoder
non-orthogonality, and the row residual all break the agreement; random
features are the control. ~260 bank contrasts/model, top-10 features + 10
random controls each; gated (paper reading rule) and ungated.

| Model | gated r² [95% CI] | slope | random-control r² | n pairs |
|---|---|---|---|---|
| Ministral-8B | **0.926** [0.913, 0.937] | **0.957** | 0.009 | 2050 |
| R1-Llama-8B | 0.934 [0.920, 0.947] | 1.126 | 0.034 | 1210 |
| R1-Qwen-7B | 0.918 [0.899, 0.933] | 1.082 | 0.000 | 1500 |
| Qwen3.5-2B | 0.834 [0.802, 0.862] | 1.338 | 0.007 | 1820 |
| Qwen3.5-0.8B | 0.886 [0.869, 0.903] | 1.661 | 0.010 | 1610 |
| Qwen3.5-9B | 0.829 [0.813, 0.845] | 1.686 | 0.058 | 1860 |

Ungated ≈ gated everywhere (Δr² < 0.02). **Reading:** the decomposition is
causally predictive on all six models; magnitudes are calibrated (slope≈1) on
the 7–8B models and overstated up to ~1.7x on the small Qwens (sign/rank
preserved) — consistent with more feature interference; report as measured.
Scope: readout-side ablation (linear in h; no re-forward), stated explicitly.

Artifacts: `CLUSTER/causal_<tag>/{summary.json,causal_rows.csv}`;
script `scripts/run/run_causal_contribution_validation.py` (self-test:
r²=1.000/slope 1.044 on residual-free synthetic; controls 0; identity exact).

**Rebuttal use:** the "validated, non-obvious payoff" all three reviewers
wanted; nominally Package D / REVISION_PLAN Edit 1. Directly supports pZ3j
("new analysis that requires the tool") and j4yY (downstream/causal bridge).

---

## 2. Edit frontier at matched KL (e1hF W3)  ✅ executed; honest split

**Protocol.** Paper's RQ4 lexical-suppression task, upgraded per review:
held-out lexicon n 64→240 (12 held-out pairs × 20 prompts), scales extended to
64 (baseline frontiers reach SRP's KL range), controls = mean-row direction,
PCA rank-1, PCA rank-4 (full rank of the 5 discovery rows; capacity-fair),
oracle + discovery-bias + random brackets retained. Frontier = held-out
suppression (Δp_bad primary, flip secondary) vs median next-token KL.
**Paired cluster bootstrap** (same candidates across methods; clustered by
held-out term [12, conservative] and by prompt [20]; 10k resamples).

**Qwen3.5-2B (display): decisive.** SRP > every baseline at every matched KL,
12/12 paired comparisons significant (term-clustered CIs exclude 0):

| matched KL | Δ(SRP − mean-row) [95% CI, term-clustered] |
|---|---|
| 0.02 | +0.032 [+0.014, +0.054] |
| 0.05 | +0.087 [+0.065, +0.115] |
| 0.10 | +0.085 [+0.062, +0.109] |
| 0.20 | +0.074 [+0.048, +0.103] |

(vs PCA rank-4: +0.036 … +0.097, all significant.) SRP reaches the oracle
ceiling at scale 12; trivial directions reach it only at 1.3–5x the KL.

**Cross-model: the advantage is model-specific.**

| Model | SRP vs mean-row at matched KL |
|---|---|
| Qwen3.5-2B | wins, 12/12 significant |
| Qwen3.5-0.8B | loses (Δ −0.030…−0.094, significant) |
| Qwen3.5-9B | loses (Δ −0.086…−0.135, significant) |
| R1-Qwen-7B | mixed (loses to mean-row at 2/4 KLs; beats PCA rank-4) |
| Ministral-8B | edit degenerate: Tekken tokenizer leaves too few single-token profane terms for feature discovery (dictionary itself is fine — see §1) |
| R1-Llama-8B | excluded: 1 tokenizable pair, 0 held-out (rule-based filter) |

Two mechanistic explanations were tested and **failed**: replacement fidelity
(9B is high-fidelity and loses) and dictionary width (registered prediction:
narrow 0.8B should win; it lost). Report both failures. SRP ≥ rank-4 PCA on
nearly all cells; **mean-row is the strongest label-free baseline** and the
final claim is scoped: dictionary-direction edits beat trivial category
directions on the display model (significant, paired); the advantage is not
general across models.

Artifacts: `LOCAL/paired_matched_kl_results.json` (72+ comparisons),
`LOCAL/matched_kl_frontier_two_panel.{png,pdf}` (figure; extend to grid),
`CLUSTER/exp3v2_<tag>/` per model. Note: full-vocab KL (conservative);
oracle uses eval labels (labeled upper bound).

---

## 3. Row-structure baselines: fidelity + groupings (e1hF W1)  ✅ split verdict

**Fidelity half — SRP decisive, vs maximally fair baselines** (Qwen3.5-2B,
1,348 contrasts / 848 base-case clusters; replicated on all six models,
`CLUSTER/exp4v2_<tag>/baseline_by_method.csv`):

| Method | sign | med ρ (unfloored) | accepted (sign & ρ<0.5) |
|---|---|---|---|
| **SRP** | **0.943** | **0.079** | **0.780** |
| nearest-row ridge (top128) | 0.861 | 0.265 | 0.625 |
| k-means dict D=65536, k=256 (sparsity-matched) | 0.835 | 0.384 | 0.596 |
| k-means dict D=16384, k=256 | 0.847 | 0.412 | 0.561 |
| hard cluster assignment (D=65536) | 0.801 | 0.577 | 0.397 |
| weighted zero-fit kNN (top128) | 0.774 | 0.622 | 0.331 |
| PCA-256 | 0.792 | 0.653 | 0.269 |

The D=65536/k=256 centroid dictionary matches SRP's D and k exactly
(near-memorization regime) and still trails by 18 pts accepted / 11 pts sign.
NOTE: v1 of these baselines was invalidated in-house (full-rank projection
artifact; zero-fit over-reconstruction) and redesigned before any results were
used — documented in `readout_baselines.py`.

**Groupings half — CORRECTED 2026-07-12: reviewer's criterion NOT met; this is
a win, not a concession.** The original table anchored J(SRP, kNN)=0.203
(paper dict) against the seed-vs-seed value 0.214 (exp5_seedvar dicts) and
read it as "≈ ceiling". That comparison was invalid twice over: (i) kNN and
cluster sets are deterministic given W_U — their self-agreement is 1.0, so a
seed-vs-seed Jaccard (noise on both sides) is not a ceiling for a
deterministic comparator (noise on one side); (ii) the two rows came from
different dictionary families. Corrected, matched analysis
(`LOCAL/knn_core_recovery_{32x,16x}.json`, seedvar dicts end-to-end,
same pipeline — paper-dict-vs-kNN reproduces 0.2026 in-pipeline):

| Quantity | 32x | 16x |
|---|---|---|
| SRP seed-vs-seed same-side | 0.214 | 0.240 |
| SRP vs kNN (per seed, matched dicts) | 0.129 | 0.125 |
| SRP vs cluster (per seed) | 0.081 | 0.073 |
| kNN recall of seed-stable core (≥2/3 seeds, median 36–44 tok) | 0.42 | 0.36 |
| cluster recall of seed-stable core | 0.20 | 0.15 |
| cross-contrast null | 0.014 | 0.018 |

**Reading:** SRP's groupings overlap row-neighborhood structure well above
null, but kNN recovers only 36–42% and clusters 15–20% of the reproducible
grouping core — the reviewer's "if the groupings look similar" antecedent
fails. **Anchored (LOO) version — what the reply quotes** (core = 2-of-2
non-held-out seeds, all methods on identical cores, 95% contrast-clustered
bootstrap CIs, `LOCAL/loo_core_anchor_{32x,16x}.json`): held-out SRP 0.749
[.73,.77] / 0.720 [.70,.74] vs kNN 0.493 [.44,.54] / 0.432 [.38,.48] vs
cluster 0.249 [.21,.29] / 0.206 [.17,.24]; SRP/kNN ratio 1.52/1.67, CIs
exclude 1 and never overlap. Read recall against the ~0.75 held-out anchor,
not 100%. Caveat for internal use: the paper dict shares only 36% of the
seedvar core and recalls 0.444 on the LOO cores — below kNN (cross-recipe
variation exceeds within-recipe seed variation); the reply therefore uses
the seedvar family end-to-end, consistent with §4, and frames stability as
within-recipe. Superseded artifact: `LOCAL/groupings_overlap_qwen2b.json`
(cross-dictionary anchor; do not quote its 0.203-vs-0.214 comparison).

---

## 4. Cross-seed stability of explanations (e1hF W2)  ✅ clean win

3 seeds × 2 widths (32x/k256, 16x/k128), Qwen3.5-2B, trained from scratch with
independent init/order/data seeds; equal reconstruction verified (val-top1
0.805–0.848; the reviewer's premise controlled). Explanation-level test over
63 curated contrasts (126 side-sets): side token-sets across seed pairs.

| Metric | 32x | 16x |
|---|---|---|
| same-side Jaccard (mean) | 0.214 | 0.240 |
| cross-side leakage | 0.003 | 0.005 |
| cross-contrast null | 0.014 | 0.018 |
| **contrasts reproducing above null p90** | **100%** | **100%** |
| matched-projection corr | 0.53–0.54 | 0.63–0.66 |
| NN decoder cosine (used features) | 0.31–0.34 | 0.52–0.55 |

**Reading:** individual directions do NOT recur across seeds (cos ~0.33 at
32x) — non-uniqueness is real and confirmed — yet the contrast-level semantic
explanation reproduces on every tested contrast, ~15x above null, with no
side leakage. The trustworthy unit of interpretation is the decomposition of a
selected score, not the feature identity (matches the paper's reading-rule
stance). Artifacts: `LOCAL/cross_seed_stability_{32x,16x}.json`;
dictionaries `SSD/results/exp5_seedvar/` (six, with metrics.json).

---

## 4b. Cross-seed FEATURE-GROUP matching (e1hF W2 follow-up)  ✅ favorable

Question: do the feature-level token groups themselves have an equivalent in
another seed's dictionary (structure re-indexed), or are they seed noise?
Protocol fixed before results: 100 features sampled from the bank-used set per
seed pair; group = top-12 centered-row token set; best-of-D Jaccard search
over the full candidate dictionary; null = 500 frequency-matched pseudo-groups
through the IDENTICAL search (prices in best-of-65k inflation); greedy union
of ≤3 candidates (recall@3) for feature splitting. Qwen3.5-2B, 3 seed pairs
× 2 widths.

| Metric (range over 3 pairs) | 32x/k256 | 16x/k128 | null |
|---|---|---|---|
| best-single Jaccard, median | 0.32–0.42 | 0.37–0.43 | 0.059 (p99 ≈ 0.11) |
| groups above null p99 | 89–90% | 87–91% | 1% by constr. |
| strong equivalence (J ≥ 0.5) | 36–41% | 40–47% | ~0 |
| recall@3 (≤3-feature union), median | 0.83–0.90 | 0.86–0.88 | 0.25 |
| recall@3 above null p99 | 91–93% | 92% | — |
| matched decoder cosine, med / p90 | 0.20–0.26 / 0.80–0.89 | 0.40–0.42 / 0.90–0.94 | — |

**Reading:** ~90% of used feature groups have an above-chance counterpart in
an independently trained dictionary; ~40% have a near-exact single match; the
median group is 83–90% covered by a union of ≤3 features — token-group
structure reproduces, indexing does not, and the residual instability is
feature SPLITTING, not noise. Cosine distribution is bimodal (a stable core
recurs almost exactly; the rest recombine); narrower 16x atoms are more stable
(cos med ~0.41 vs ~0.22), consistent with Paulo & Belrose (2025,
arXiv:2501.16615: TopK SAEs most seed-dependent, ~30% shared features at
131k latents) and with "unstable features, reproducible subspaces"
(arXiv:2606.12138). Positioning: atom non-uniqueness is a documented property
of the fitting primitive, not of SRP; SRP's interpretive unit (contrast-level
decomposition, §4) and its group-level structure both reproduce.

**Tail anatomy (below-null groups, direction-level check).** For every
below-null group, max decoder cosine over the full candidate dictionary vs a
random-direction null (p99 ≈ 0.11; generous — decoders are anisotropic, so
only cos ≥ 0.5 is quoted as a counterpart). Below-null groups: 32/300 (32x),
33/300 (16x). Of these, direction-level counterparts at cos ≥ 0.5: 5/32
(32x), 10/33 (16x) — e.g. plural-suffix 0.83, degree-nouns 0.80, Chinese
flowers 0.76, verb-class 0.54. Qualitative read of the remainder (NOT yet
rubric-scored — needs the Table-21 coherent/ambiguous/token-form pass before
being quoted as an audit result): mostly incoherent mixed-script grab-bags of
the kind the label audit rates ambiguous; two features (16877, 30762 at 16x)
fail against both other seeds. Truly seed-specific coherent residue: ~1–2 per
100 queries. Identity-criteria hierarchy for the write-up: atom (rejected,
non-unique) → interpretive feature (token family, splits allowed) → causal
feature (decoder cos ≥ 0.5) → explanation (contrast decomposition; the 100%
result in §4). Cosine/union thresholds are post-hoc for the reply; fix them +
add a threshold-sensitivity curve before the paper appendix.

Artifacts: `LOCAL/feature_group_matching_{32x,16x}.json` (per_query now
includes `max_decoder_cosine` for below-null groups), scripts
`LOCAL/feature_group_matching.py`, `LOCAL/failed_group_direction_check.py`.

---

## 4c. Sparsity/width grid (pZ3j W3, e1hF k comment)  ✅ run 2026-07-12, logged 2026-08-02

Qwen3.5-2B, identical recipe per cell (20k steps, batch 4,096, same data and
schedule; only width, k, and seed differ). Dead = never active on the
evaluation rows; rare = firing rate < 1e-3.

| Setting | seeds | dead | rare | rowEV | top1 | KL |
|---|---|---|---|---|---|---|
| 16x (D=32,768), k=64 | 2 | 0.491–0.512 | 0.632–0.645 | 0.683–0.684 | 0.783–0.816 | 0.367–0.370 |
| 16x (D=32,768), k=256 | 2 | 0.036–0.037 | 0.213–0.218 | 0.801 | 0.807–0.848 | 0.226–0.243 |
| 32x (D=65,536), k=128 | 3 | 0.336–0.352 | 0.628–0.637 | 0.769–0.771 | 0.816–0.826 | 0.241–0.253 |

**Reading:** at matched 16x width, dropping k=256 → k=64 strands half the
dictionary (dead 0.04 → 0.50) and costs fidelity (rowEV 0.80 → 0.68, KL 0.23 →
0.37). Dead rate tracks budget-relative-to-width, not k alone. Seed spread
within a setting is far smaller than between settings.

**Correction to the posted rebuttal.** The e1hF and pZ3j replies state
"k ∈ {32,64} on Qwen3.5-0.8B and Qwen3.5-2B" and "at k=32, 64–69% of features
are never used." Only the k=64 half exists, and only on Qwen3.5-2B: the
`k=64`/2B figure is confirmed (0.491/0.512, i.e. the posted "50.5%"), but
**no k=32 run and no 0.8B low-k run exists on the cluster or locally**
(searched 2026-08-02). Do not quote the k=32 numbers. The manuscript
(App. E.3) reports only the measured cells.

Artifacts: `CLUSTER/pz3j_hparam_grid_20260712/qwen2b_{d32768_k64_s{1,2},
d32768_k256_s{0,2}, d65536_k128_s{0,1,2}}/metrics.json`; metrics + configs
mirrored to `LOCAL/../pz3j_hparam_grid_20260712/` (checkpoints left on the
cluster). `qwen2b_d32768_k256_s1` never finished (no `DONE`).

---

## 5. Scope findings (report as boundaries, not failures)

- **Tokenizer gate on the edit task:** the single-token profanity lexicon does
  not exist in Llama BPE (1 pair) and barely in Tekken (4 pairs) — the edit
  experiment is defined on the 4 Qwen-tokenizer models; stated per-tokenizer.
- **Causal slopes >1 on small Qwens:** magnitude overstatement (1.3–1.7x) with
  preserved sign/rank; candidate explanation = feature interference; left open.
- **9B/0.8B edit losses:** fidelity and width explanations both tested and
  rejected; mean-row is a strong baseline; claim scoped to display model.

## 6. Reviewer mapping

| Reviewer ask | Answer | Section |
|---|---|---|
| e1hF W1 (kNN/cluster basis, groupings) | fidelity win ×6; groupings win (kNN/cluster recover ≤42% of stable core) | §3 |
| e1hF W2 (seed stability of explanations) | 100% reproduce above null, both widths; premise controlled | §4 |
| e1hF W3 (mean-row/PCA controls, matched KL) | executed in full; display-model win (paired, significant); honestly scoped cross-model | §2 |
| e1hF comment (k sparse enough?) | reframe: 256/65,536 = 0.4% active; median 76 features carry 80% of mass; 16x/k128 replications throughout | to write |
| pZ3j (novelty; "new analysis requiring the tool") | causal validation §1; groupings pivot §3 | §1, §3 |
| pZ3j (label audit success rate) | **claim-bearing audit complete:** 3 independently run `gpt-5.6-sol` raters at high reasoning effort in fresh Codex agent contexts; top-20 rows only; 93.4% unanimity, Fleiss' κ=0.87; consensus over 99 display occurrences: 76 coherent / 4 mixed or ambiguous / 19 token-form. Frozen prompt, hashes, JSONL outputs, and majority rule retained. Uncurated contribution-weighted bank audit remains open. | partial |
| pZ3j (hyperparameter sensitivity) | 2 widths × 3 seeds stability (§4) + existing App. K sweeps (point reviewer) | §4 |
| j4yY (downstream/causal motivation) | §1 + framing edits | §1 |
| j4yY (median vs mean, tails) | add mean/p90/p95 columns from existing per-row CSVs | to write |
| j4yY (offset/residual, definitions, abstract) | writing items (REVISION_PLAN Edits 6–10) | to write |

## 7. Remaining work (no compute)

1. Rebuttal text (per-reviewer; concessions explicit; numbers above).
2. Paper edits: new appendices (causal validation; seed stability; groupings +
   redesigned baselines), scoped RQ4 claims, k-reframe, tails columns,
   j4yY clarity edits.
3. Label-audit κ: two blind raters (USER) + aggregate non-cherry-picked rate.
4. Figure assembly: frontier grid (5 panels + fidelity-vs-advantage summary),
   causal scatter (per-model), stability/groupings anchor bar.
