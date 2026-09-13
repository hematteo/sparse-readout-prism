# Project: Sparse Readout Prism — paper repo

This repo is a **LaTeX research manuscript** targeted at a top-tier venue. It is *not* a software codebase. Treat `.tex` files as formal scientific prose.

## Where you are

Two different "roots" exist and they are not the same directory:

- **Paper root** — `/Users/m/Desktop/SRP/Sparse_Readout_Prism/`. This file's directory. `main.tex`, `sections/`, `appendices/`, `figures/`, `style/` live here. **Build from here.** Everywhere below, "repo root" means this.
- **Git root / workspace root** — `/Users/m/Desktop/SRP/`, one level up. This is where `git` commands resolve paths from (`git diff -- Sparse_Readout_Prism/sections/…`, not `-- sections/…`) and where sessions usually start. It holds six sibling directories that are **not** the paper; see the workspace `CLAUDE.md` there.

Companion docs (all paths relative to the paper root):
- `REVISION_EDITING_PLAN_V3.md` — **the active plan** for the post-metareview revision (ARR 2026-May → next cycle). Workstreams, cut list, landmines. Start here for any revision task. 804 lines; `Grep` it, don't `Read` it whole.
- `RESUBMISSION_CHANGE_SUMMARY.md` — the OpenReview resubmission note: what changed since submission 10103 and which reviewer point each change answers. Read when you need to know *why* something is the way it is.
- `REBUTTAL_RESULTS.md` — source of truth for every rebuttal-campaign number, including the **do-not-quote list** (§3, §7). Never import a rebuttal number into the paper from anywhere else.
- `SECTION4_L2_SPINE.md` — draft narrative skeleton for escalating the cross-lens section, which is now §5 (it was §4 when the doc was written). Consult only when working on that section's register; it has no bearing on the current §4.
- `glossary_and_terminology.md` — canonical names for methods, models, datasets, metrics. Do not rename.
- `AGENTS.md` — generic prose-style and editing-policy rubric (not a project-specific structural doc). Useful for "how to phrase claims"; not useful as repo onboarding.

Review-cycle record (reviews, metareview, posted responses, old plans): `../rebuttal/` — see its README. The consolidated reviewer objections are in `../rebuttal/reviews.md`; the top two, raised independently by multiple reviewers, are **limited novelty vs. SAEs** and **insufficient evidence of value beyond simpler methods**. Weigh structural and framing decisions against those.

**Stale numbers live in three places outside the paper.** Never source a figure, table or statistic from any of them — go to `REBUTTAL_RESULTS.md`:
- `../rebuttal/drafts/` — superseded rebuttal drafts with invalid numbers
- `../readout_prism_workspace/results/**/report/*.tex` — loose LaTeX table fragments (`tab_groupings.tex`, `tab_j4yy_srp6.tex`, …) from the rebuttal campaign. These are real `.tex` files and a workspace-wide grep for a table or a number will surface them.
- `../_archive/`, `../_history_bundles/` — historical snapshots

# Context discipline

These rules exist to keep the context window clean for prose work. Future agents working in this repo should follow them.

- **Consult companion docs on demand, not preemptively.** `glossary_and_terminology.md` is 76 KB / 463 lines and `REVISION_EDITING_PLAN_V3.md` is 804 lines — neither may be `Read` whole; use `Grep` to look up specific terms. `AGENTS.md` and `RESUBMISSION_CHANGE_SUMMARY.md` are smaller; read them only when the task actually requires them (e.g. claim/scope decisions, structural questions).
- **Do NOT read build artifacts** under any circumstances — they are large and useless for prose work:
  - `main.aux`, `main.bbl`, `main.fls`, `main.fdb_latexmk`, `main.out` — generated, never edit, never read
  - `main.pdf` — open in a viewer, never `Read` (binary)
  - `.DS_Store` — ignore
  - `main.log` — read ONLY when explicitly diagnosing a LaTeX warning, and grep for the warning rather than reading top-to-bottom
  - `main.aux` is the one defensible exception: when checking **pagination or float placement** (which page a section or float landed on), grep it for `\@writefile{toc}` / `\newlabel` rather than recompiling and guessing. Never edit it, never read it whole.
- **One file at a time for polish passes.** Don't `Read` every section file at the start of a session — load only what the current task touches.

# Critical constraints

- **Never edit, invent, reorder, or "fix"** any of:
  - `\cite{}`, `\citet{}`, `\citep{}`, `\ref{}`, `\eqref{}`, `\label{}`, `\autoref{}`
  - Math inside `$...$`, `$$...$$`, `\(...\)`, `\[...\]`, or `equation`/`align`/`gather` environments
  - `references.bib` entries or bibliography keys
  - Method, model, dataset, or metric names defined in `glossary_and_terminology.md`
- **Float placement is tuned; leave it alone.** This repo overrides the global `[H]` rule — `[H]` cannot be applied to `figure*`/`table*` at all, and in ACL two-column it paginates badly. The working convention is `[t]` in the body (7 floats) and `[!tbp]` in the appendices (68 floats), held in place by `style/preamble.tex`: `stfloats`, `placeins`, tightened `\textfloatsep`/`\floatsep`, `\floatpagefraction` at 0.82, and `\AppendixFloatSetup`. Do not "restore" `[H]`, and do not change a float specifier to fix a placement complaint — adjust the preamble lengths or move the float's source position instead. Moving a float's *source position* is allowed, and a float moved between body and appendix should be re-specified to match its destination's convention. **Figure 1 on page 2 is by design, not a placement bug.** In the preprint build no tuning can put it on page 1: the column room below the front panel (~287pt) is smaller than the float (~359pt), and for `figure*` the kernel forbids double floats outright on a page begun with `\twocolumn[...]` (`\@topnewpage` sets `\@dbltopnum` to −1). The only page-1 options — shrinking the figure to ~60% width, or pinning it inside the panel insertion with the caption beside it — were tried and rejected (2026-08-30). Leave it on page 2.
- **No fabricated citations or numbers.** If prose needs a citation that isn't there, flag it in chat. Never paste a plausible-looking `\cite{...}`.

# Editing style

- "Polish" / "tighten" / "fix" = grammar, register, clarity, conciseness. Active voice, concrete subjects, no filler.
- Banned vocabulary: "delve", "leverage", "in the realm of", "it is important to note", "navigate (the/this) landscape", "robustly". Cut hedging ("we believe", "it could be argued").
- Do not rewrite paragraphs wholesale when targeted edits suffice. Prefer the smallest edit that achieves the goal.
- Do not restructure arguments unless explicitly asked. If restructuring is needed, propose the outline first and wait for approval.
- Match the register of the surrounding text — don't import a different paper's voice.
- No end-of-turn summaries. The diff is visible.

# Build

- `latexmk -pdf main.tex` **from the paper root** (this file's directory). PDF lands at `main.pdf`.
- Shell working directory persists between tool calls. If you `cd` elsewhere (e.g. into `../rebuttal/` to read reviews), `cd` back before building — otherwise latexmk reports "Could not find file 'main.tex'", which looks like a broken build and is not.
- Section files live under `sections/`. Appendix files under `appendices/`. Figures under `figures/`. Style files (`acl.sty`, `acl_natbib.bst`) under the paper root and `style/`.

# Submission build mode

`main.tex` line 10 and `style/preamble.tex` carry two switches that **must move together**:

| Target | `main.tex` | `style/preamble.tex` |
|---|---|---|
| ARR review submission | `\usepackage[review]{acl}` | `\preprintmodefalse` |
| arXiv preprint | `\usepackage[preprint]{acl}` | `\preprintmodetrue` |

Current state is the **arXiv preprint build**, so no page limit applies and `main.pdf` is de-anonymized: author block from `style/metadata.tex`, corresponding-author footnote in `main.tex`, and the GitHub/HuggingFace links guarded by `\ifpreprintmode` in `sections/frontmatter.tex`, `sections/limitations.tex` and `appendices/readout_sae_diagnostics_reproducibility.tex`. Build the upload with `./make_arxiv.sh`, which refuses to run unless both switches are set. v1 is public as **arXiv:2609.01936** (<https://arxiv.org/abs/2609.01936>, announced 2026-09-01) from tag `arxiv-v1`; the uploaded tarball, arXiv's render, the official BibTeX and the verification record live in `../arxiv_upload/`. Cite the paper from `../arxiv_upload/arxiv_2609.01936.bib`, and re-paste the abstract into the form at v2 (the working tree's last sentence differs from the record).

If the build is switched back to **ARR review**, the 8-page limit applies again. Limitations, Ethics and References are exempt from that limit; **the Conclusion is not**. Check where §Conclusion starts before declaring a length problem solved — it sits on the page 9/10 boundary in the preprint build (page 10 as of 2026-08-31); grep `main.aux` rather than trusting this note.
