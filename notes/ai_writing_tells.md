# AI Writing Tells — Avoidance Checklist for Paper Revision

Purpose: pass-checklist for making SRP prose read as human expert writing at a top-tier venue.
Compiled from Wikipedia's *Signs of AI Writing* / WikiProject AI Cleanup catchphrase catalog,
Kobak et al. 2024 ("Delving into LLM-assisted writing…", Science Advances / arXiv:2406.07016,
excess-vocabulary analysis of 15M PubMed abstracts), the FSU lexical-overrepresentation study
(arXiv:2412.11385), and detector-industry word lists.

**How to use:** none of these is damning alone — reviewers (and detectors) key on *density and
clustering*. Run the grep passes at the bottom, then justify every hit individually. Default
action is delete or replace with something concrete.

---

## 1. Word bank — Tier 1 (high-signal, near-zero legitimate use in our paper)

Empirically the most elevated post-ChatGPT words in scientific abstracts (Kobak et al.; "delve"
alone rose ~654% in biomedical abstracts 2020→2023). Replace on sight:

| Tell | Replace with |
|---|---|
| delve / delves / delving | examine, analyze, test |
| underscore(s) / underscoring | show, indicate — or state the claim directly |
| showcase / showcasing | show, demonstrate |
| pivotal / crucial / vital / paramount | important (once), or quantify why it matters |
| intricate / intricacies | complex — or describe the actual structure |
| meticulous(ly) | (delete; describe the procedure instead) |
| tapestry / realm / landscape / avenue(s) | field, setting, direction — or name it |
| testament (is a testament to) | (delete; state the evidence) |
| harness(ing) / leverage / leveraging | use |
| bolster(ed) / augmenting | support, increase |
| garner(ed) | receive, attract |
| unveil / unveils / unravel(ing) | present, introduce, explain |
| elucidate(s) | explain, clarify |
| burgeoning | growing |
| commendable / remarkable / impressive / exceptional | (delete; let numbers carry it) |
| seamless(ly) | (delete or specify the interface that works) |
| transformative / groundbreaking / revolutionize | (delete; overclaim) |
| multifaceted / holistic / nuanced | (name the facets) |
| interplay / interconnectedness | interaction, relationship |
| foster(ing) / cultivate | enable, encourage, produce |
| myriad / plethora / a wealth of | many, N (count them) |
| noteworthy / notably (as filler) | (delete or make the note) |
| invaluable / valuable insights | (state the insight) |
| embark / endeavor / journey | (delete) |

## 2. Word bank — Tier 2 (fine once, a tell in clusters)

Elevated in LLM text but with legitimate technical uses. Budget: audit any that appear >2–3
times in the paper.

- **Verbs:** highlight(s/ing), emphasize/emphasizing, enhance(s/d)/enhancing, facilitate(s/d),
  encompass(es/ing), encapsulates, exhibit(s/ed), employ(s/ed), utilize(s/d) (→ use),
  streamline(s/d), expedite, illuminate(s), pinpoint(ed), scrutinize(d), surpass(es/ed),
  necessitate(s), align(s) with, resonate with, capitalize on, pave/paving the way,
  shed(ding) light on, hinge(s) on, poised to, grapple/grappling with, revolve(s) around
- **Adjectives:** robust, novel, comprehensive, significant (without a test), substantial,
  compelling, notable, distinctive, foundational, pioneering, formidable, pressing,
  imperative, pronounced, prevalent, enduring, vibrant, rich, profound, diverse array
- **Adverbs/connectives:** Additionally (sentence-initial), Furthermore, Moreover,
  Consequently, Conversely, Subsequently, Ultimately, Thereby, importantly, interestingly,
  effectively, essentially, arguably, particularly, predominantly, potentially (stacked),
  seamlessly, swiftly, strategically
- **Nouns:** advancement(s), capabilities, methodologies (→ methods), exploration, nuances,
  groundwork, challenges (as a section crutch), potential (as vague noun), versatility

## 3. Phrase bank — delete or rewrite on sight

- "It is worth noting that…" / "It is important to note that…" / "It should be emphasized that…"
- "plays a crucial/pivotal/vital/key role in…"
- "in today's fast-paced / ever-evolving world/landscape of…"
- "a growing body of work/literature has…" (→ cite the specific papers and their claims)
- "Recent advances in X have revolutionized/transformed Y…" (as an opener)
- "stands as / serves as / marks / represents a…" (copula avoidance — just say "is")
- "opens exciting avenues for future research" / "paves the way for…"
- "bridge the gap between…" / "unlock the potential of…"
- "In this section, we will explore/delve into…" (empty meta-commentary)
- "In conclusion / Overall / Taken together, these results demonstrate…" (max once, in the
  actual conclusion)
- "Despite these challenges, …" (formula-driven pivot)
- "…, highlighting the importance of X" / "…, underscoring the need for Y" — the trailing
  present-participle impact clause. This is the single most common LLM sentence shape in
  scientific prose: claim + comma + "-ing" phrase asserting vague significance. Cut the
  clause or make it a sentence with content.
- "not only X but also Y" / "It's not just X — it's Y" / "not X, but Y" (negative
  parallelism; Wikipedia flags this as correcting "a misconception nobody actually had")
- "may potentially" / "could possibly suggest" (hedging stacks — hedge once, precisely)
- "various / several / numerous studies" (→ count or cite)
- "Experts argue / Observers have noted / Industry reports suggest" (vague attribution)

## 4. Structural and rhetorical tells

- **Rule of three.** "fast, scalable, and interpretable." One or two triplets per paper is
  human; one per paragraph is a fingerprint. Break the symmetry: two items, or four, or one
  developed properly.
- **Uniform paragraph shape.** Topic sentence → three supports → mini-summary restating the
  topic sentence. Cut the restating closers; end paragraphs on content.
- **Uniform sentence length.** LLM prose sits at a steady 20–25 words/sentence (low
  "burstiness"). Mix short declaratives with long ones. Read sections aloud.
- **Connective-driven transitions.** "Furthermore, … Moreover, … Additionally, …" as
  paragraph openers. Real transitions come from content ("This breaks down when…").
- **Elegant variation.** Cycling synonyms (model/approach/framework/method for the same
  thing) to avoid repetition. In technical writing, repeating the exact term is correct;
  synonym-cycling reads as both AI-ish *and* imprecise.
- **Perfectly symmetric sections.** Every subsection the same length and internal shape.
  Human experts spend three paragraphs on the tricky experiment and one sentence on the
  easy one. Uniform coverage is a tell; asymmetry signals judgment.
- **Bolded-lead-in bullet lists** in running prose (fine in appendix checklists; a tell in
  discussion sections).
- **Summary/recap paragraphs** at the end of every section.

## 5. Tone tells (the ones reviewers punish hardest)

- **Manufactured surprise:** "Surprisingly," "Remarkably," "Strikingly," "Intriguingly"
  before results the setup made predictable. [Standing rule for this paper: none.]
- **Overselling:** "significant" without a statistical test attached; "substantial
  improvements"; superlatives ("unparalleled", "state-of-the-art" unearned).
- **Promotional register:** "boasts", "cutting-edge", "powerful framework". Papers describe;
  ads showcase.
- **Uniformly positive limitations section** ("while our method requires compute, this also
  enables…"). Real limitations sections concede something that actually hurts.
- **False balance / both-sidesing** your own results instead of committing to what the
  evidence supports and hedging only where it's thin — and saying *why* it's thin.

## 6. Punctuation and formatting tells

- **Em-dash overuse** — LLMs reach for them constantly. Audit `---` count in the LaTeX
  source; prefer commas, parentheses, or splitting the sentence. A handful per paper is
  human; several per page is not.
- **Title Case In Every Heading** where the venue style is sentence case (check ACL style).
- **Bold for mechanical emphasis** ("**key takeaway:**").
- **Curly vs straight quotes** inconsistencies (LaTeX mostly handles this; watch pasted text).
- **Colon-titled everything:** "X: A Comprehensive Framework for Y" title shapes, and
  parallel colon-headers throughout.
- **Stacked hyphenated compounds.** LLMs coin compound modifiers freely
  ("sign-preserved low-error subset", "discovery-selected decoder directions"). Keep
  defined technical terms (held-out, per-feature), but unpack ad-hoc coinages —
  especially two or more compounds stacked before one noun — into plain phrases
  ("a larger subset of contrasts preserving sign at low error"). [House rule: use
  hyphenated conjoined words sparingly.]

## 7. Academic-specific integrity checks

- **Hallucinated/mangled citations:** verify every DOI resolves to the cited paper, no
  broken URLs, no invalid arXiv IDs, correct venues/years, no `utm_source=` in URLs.
  (Wikipedia's #1 hard signal; also the fastest way to lose reviewer trust.)
- **Generic related work:** every cited paper should have a specific claim attached, not
  membership in a list. "X et al. showed A; we differ in B" beats "Several works have
  explored sparse readouts [1–7]."
- **Knowledge-cutoff phrasing:** "as of this writing", "recent developments may…" —
  delete.

## 8. What human expert prose does instead

- **Concrete numbers over adjectives:** "recovers 72–75% of the stable core" beats
  "recovers a substantial fraction."
- **Active first person:** "We freeze the encoder" (passive fine where the agent genuinely
  doesn't matter).
- **Committed claims:** state what the evidence supports plainly; hedge only where the
  evidence is thin, with the reason.
- **Baselines beside results,** not buried — reads as confident, not defensive.
- **Plain copulas:** "SRP is a linear readout" beats "SRP serves as a linear readout."
- **Content-bearing transitions** and paragraph endings.

## 9. Grep passes for the LaTeX source

```bash
cd Sparse_Readout_Prism

# Tier 1 vocabulary
grep -rniE 'delve|underscor|showcas|pivotal|crucial|intricate|meticulous|tapestry|realm\b|landscape|testament|harness|leverag|bolster|garner|unveil|elucidat|burgeoning|seamless|transformative|groundbreaking|multifaceted|holistic|interplay|foster|myriad|plethora|invaluable|commendable' \
  --include='*.tex' .

# Phrase bank
grep -rniE 'worth noting|important to note|plays? a (crucial|pivotal|vital|key) role|growing body|paves? the way|bridge the gap|shed(s|ding)? light|not only.*but also|opens? (exciting|new) avenues|in conclusion|taken together|despite these challenges|ever-evolving' \
  --include='*.tex' .

# Manufactured surprise + overselling
grep -rniE 'surprisingly|remarkably|strikingly|intriguingly|novel\b|robust\b|comprehensive\b|significant\b' \
  --include='*.tex' .

# Trailing participle impact clauses (approximate)
grep -rnE ', (highlighting|underscoring|emphasizing|demonstrating|suggesting|indicating|reflecting) ' \
  --include='*.tex' .

# Em dashes and sentence-initial connectives
grep -rn -- '---' --include='*.tex' . | wc -l
grep -rnE '^(Additionally|Furthermore|Moreover|Consequently|Conversely|Ultimately),' \
  --include='*.tex' .
```

Rules of thumb: `novel` ≤1, `robust` ≤2 (fine as a statistics term), `significant` only with
a test, `comprehensive` 0, em dashes ≤ ~5 per paper, sentence-initial
Furthermore/Moreover/Additionally ≤ ~3 total.

---

## Sources

- Wikipedia, *Signs of AI writing* — https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- Wikipedia, *WikiProject AI Cleanup / AI catchphrases* — https://en.wikipedia.org/wiki/Wikipedia:WikiProject_AI_Cleanup/AI_catchphrases
- Kobak et al., "Delving into LLM-assisted writing in biomedical publications through excess
  vocabulary", Science Advances 2025 / arXiv:2406.07016. Full 900-word list (407 style
  words): https://github.com/berenslab/llm-excess-vocab (`results/excess_words.csv`)
- Juzek & Ward, "Why Does ChatGPT 'Delve' So Much?", arXiv:2412.11385 (FSU study, 21 focal
  words incl. delve, resonate, navigate, commendable)
- Reuters Institute, "How AI-generated prose diverges from human writing" —
  https://reutersinstitute.politics.ox.ac.uk/news/how-ai-generated-prose-diverges-human-writing-and-why-it-matters
