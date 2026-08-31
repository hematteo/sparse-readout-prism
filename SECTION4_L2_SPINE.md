# §4 at L2 — Narrative Spine

Draft skeleton for escalating Section 4 from "we name a phenomenon" (current, L1) to
"a published inference does not follow from its evidence" (L2). Load-bearing sentences
in the order a reader meets them, so the register is visible before any prose is
committed.

**Not** L3: we never claim their results are false, or that Claude's readouts are
corpus artifacts. See §5 (tripwires).

---

## 1. The epistemic contract

L2 lives or dies on assigning the right verb to each claim. Three registers, and they
must not blur:

| Claim | Register | Verb |
|---|---|---|
| The released multilingual eval bank grades non-English prompts on English intermediate tokens | Logical / artifact | **cannot in principle disconfirm** — no hedge |
| "Capacity ≈ 25" counts token vectors, not concepts | Definitional | **counts X, not Y** — no hedge |
| The mid-band English reading reflects the English fitting corpus | Empirical, ours | **underdetermined by** / **does not follow from** |
| The workspace band structure | Supported | **replicates** — we agree |
| Claude specifically routes through English | Out of scope | say nothing |

The first two are facts about public artifacts and can be stated flatly. The third is
ours to defend and gets the careful verb. Blurring them is how L2 slides into L3.

---

## 2. The spine

### §4 title

> **Lens Token Readings Do Not Identify the Language of Intermediate Computation**

(Alt: *What a Fitted Lens Reports Depends on Its Corpus; What It Reads Does Not.*)

Current title — "The Reported Token Can Follow the Lens Corpus" — has a modal verb and
an inanimate subject. Replace it.

### Abstract (replaces the last two sentences)

> Fitted vocabulary lenses are read not only as predictions of what a model will say
> but as reports of what it computes in. We show that second reading is
> underdetermined. English- and Chinese-fitted Jacobian lenses report different surface
> languages on identical hidden states while the same SRP feature stays dominant on
> **77 of 80** prompts; the result holds for a same-script language pair and for a
> second lens construction. The reported surface language follows the lens's fitting
> corpus, so an inference from lens token readings to a model's intermediate
> language — including the claim that multilingual models route through English — does
> not follow from that evidence alone.

### Contribution bullet 3 (replaces "A cross-lens finding: corpus conditionality")

> *A correction to how fitted-lens readings are used.* The surface language a fitted
> lens reports follows its fitting corpus while the dominant readout feature does not,
> across two language pairs and two lens constructions. Conclusions drawn from lens
> token readings about a model's intermediate language are therefore underdetermined,
> and we state what a corpus-invariant reading would require (§4).

### Related Work — "Lens-style readout interfaces"

Add one sentence establishing the debate has more than two participants:

> Whether multilingual models compute in a pivot language has been studied through
> lens readings of intermediate states `\citep{TODO-wendler2024, TODO-dumas}`; §4 shows
> that such readings inherit the lens's fitting corpus.

**Blocked:** neither citation is in `references.bib`. Do not invent keys — add the
entries first, then wire the sentence. This citation is not optional; without it the
section reads as a rebuttal to one unreviewed post rather than a contribution to a
live question.

### §4 ¶1 — Stakes (new; currently absent)

> Fitted lenses transport an intermediate hidden state into the space the LM head reads
> and report the resulting tokens. Increasingly those tokens are read as evidence about
> the state itself: which concept is active, and in which language the model is
> operating. That inference has a dependency the token report does not expose. The
> transport is fitted on a corpus, so the reported surface form is a property of the
> lens as much as of the state, and a token ranking cannot separate the two.

### §4 ¶2 — Replication (new; port from `jlens_critique` §5.1)

> Before qualifying any reading we reproduce the phenomenon the lens was built to
> expose. On Qwen3.5-9B and 50 held-out C4 prompts the three-band depth structure
> reported for a frontier model \citep{gurnee2026workspace} appears cleanly: top-1
> agreement with final logits rises from 0.00 to 0.40 into the final band while KL
> falls from 13 to 2, and mid-band readings hold coherent abstract content across
> roughly 20 layers before the surface token emerges. To our knowledge this is the
> first open-model replication. Our disagreement is with one inference drawn from lens
> readings, not with the workspace structure.

This paragraph is what buys the right to the next one. It is currently nowhere in the
manuscript — "workspace" appears in the compiled PDF only inside a bibliography title.

### §4 ¶3–4 — Design and Result (largely as-is)

Keep. Two required additions:

**(a) The unfitted-lens column.** Add the corpus-free logit lens to the design table.
It is the sharpest cell in the memo's Table 1: the unfitted readout is Chinese at every
layer, and English appears only once the lens is fit on English. Already computed.

**(b) The honest floor.** Insert into Result, before a reviewer derives it:

> Translation-equivalent rows are already aligned in raw unembedding geometry, above a
> random floor, so the factorization does not discover the shared carrier. What it
> supplies is the invariant: a named, counted direction whose dominance is stable while
> the lens's surface report is not. The lens-free agreement rate — the same dominant
> feature for translation-equivalent rows decomposed directly through the LM head — is
> **[N/80]**, and the 77/80 should be read against it.

`[N/80]` does not exist yet and is the highest-priority missing number in the paper.
Frame the claim around invariance, not discovery, and a high floor stops being fatal.

### §4 ¶5 — Correction (replaces the current "Reading" paragraph — **the L2 pivot**)

> Lens token readings have been used to infer the language of a model's intermediate
> computation, most directly in the claim that multilingual prompts are routed through
> English \citep{gurnee2026workspace}. That inference requires the reported surface
> form to be a property of the state rather than of the instrument, and it is not: with
> the state held fixed and only the fitting corpus varied, the reported language flips
> while the dominant readout feature does not. The causal evidence offered in support
> does not separate the two hypotheses either — swapping an English token's lens
> coordinates moves the concept because that token vector loads on the same carrier its
> translation does, so the swap is expected whether or not the intermediate
> representation is English. Two properties of the published evaluation compound this.
> The released multilingual evaluation bank scores non-English prompts by whether
> English intermediate tokens appear; an English-fitted lens graded against
> English-anchored targets cannot disconfirm the English-intermediate reading. And the
> reported workspace capacity counts token vectors rather than concepts, so where
> several surface forms ride one carrier the concept count is smaller than the figure
> suggests. We do not claim these conclusions are false. We claim the evidence offered
> does not distinguish them from the corpus-conditional alternative.

The last two sentences are the L2 boundary. Keep them; they are what makes the
paragraph publishable rather than a fight.

### §4 ¶6 — What this asks of lens practice (new, short)

> Three changes follow, each testable. Report and vary the fitting corpus: a readout
> claim about a model should survive reasonable corpus choices, and the per-corpus
> spread belongs beside the reading. Score concepts rather than spellings: a target
> present under a sibling surface form is scored as a miss by token-anchored metrics,
> and the bias is language- and frequency-dependent. Attach a residual to every
> reading: a decomposition that closes exactly supplies a per-reading reliability
> signal, which a ranking cannot.

### Limitations — replaces the current cross-lens scope paragraph

Keep the existing single-model / single-scale / operational-definition scope. Add:

> We cannot inspect the model the original claims concern, and make no claim about its
> readouts; we show the mechanism is real and available on open models. Our lenses are
> fitted at a smaller budget than the original. The shared carriers are present in raw
> row geometry before any factorization, which is why the claim is about invariance of
> the feature account under corpus change, not about discovery of multilingual
> structure.

---

## 3. Evidence ledger

**A** = in the manuscript · **B** = exists in `jlens_critique`, needs porting ·
**C** = new work.

| Beat | Evidence | Status |
|---|---|---|
| Stakes | prose + latent-language citations | C (citations missing from `.bib`) |
| Replication | band structure, top-1 0.00→0.40, KL 13→2, 50 prompts | **B** |
| Corpus flip, EN–ZH | 77/80 [0.90, 0.99], floors 0/159 · 0/240 | **A** |
| Same-script control | EN–DE 84/90, cognates excluded | **A** |
| Second construction | ridge 67/80; Jacobian vs ridge 76/80 | **A** |
| Unfitted-lens column | logit lens Chinese at every layer | **B** |
| Lens-free floor | `[N/80]` on translation-equivalent rows | **C — top priority** |
| Their causal swap doesn't separate | argument, not experiment — label as such | **B** |
| Eval-bank circularity | public artifact audit | **B**, re-verify (see §4) |
| Capacity in concept units | ~25 token vectors; surface forms per carrier | **B**, needs a measured count if quoted |
| Form instability | 36% workspace band vs 8% final layer | **B**, v1 metric — needs v2 or drop |
| Carrier steering | multi-language flip vs token swap at matched KL | **C** — optional, highest value |

Minimum viable L2: the four **B** items plus the lens-free floor. Everything else is
already written.

---

## 4. Before this goes to print

- **Re-verify the eval-bank claim against the current public artifact**, and quote the
  exact field rather than paraphrasing. It is a factual assertion about someone's
  released file, stated flatly, in a peer-reviewed venue. It must be exactly right and
  it must be re-checked at submission time, not carried over from a July memo.
- **Measure the capacity figure** or cut it. "Four surface forms per concept routinely"
  is an observation, not a count, and the sentence as drafted implies a count.
- **Resolve the form-instability metric** to v2 or leave it out. A provisional metric in
  a correction paragraph is an opening.
- **Add the latent-language citations.** Blocking for the Related Work sentence and for
  the contemporaneity objection.

---

## 5. Tripwires — the L3 boundary

Do not write, in any draft:

- "artifact of" applied to their empirical results without qualification (fine for the
  eval-bank structure, not for their findings)
- any claim about the frontier model's readouts or fitting corpus
- "SRP discovers multilingual structure" — the memo already concedes raw geometry
  separates translation pairs above floor
- "the shared feature is a language-neutral representation" — already correctly
  disclaimed in §4; keep the disclaimer
- "critique of \citep{gurnee2026workspace}" — the target is an inference pattern that
  predates them and generalizes past them, which is what the ridge-translator result
  buys. Frame it that way everywhere.

The ridge translator is the asset that makes L2 safe: it shows corpus conditionality is
not one method's quirk but a property of corpus-fitted lenses generally. Lead with that
whenever the contemporaneity objection comes up.
