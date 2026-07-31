# Project: Sparse Readout Prism — paper repo

This repo is a **LaTeX research manuscript** targeted at a top-tier venue. It is *not* a software codebase. Treat `.tex` files as formal scientific prose.

Companion docs:
- `RESUBMISSION_PLAN.md` — **the active plan** for the post-metareview revision (ARR 2026-May → next cycle). Workstreams, cut list, landmines. Start here for any revision task.
- `REBUTTAL_RESULTS.md` — source of truth for every rebuttal-campaign number, including the **do-not-quote list** (§3, §7). Never import a rebuttal number into the paper from anywhere else — superseded drafts with invalid numbers exist under `../rebuttal/drafts/`.
- `CORE_CLAIM.md` — what this paper argues (claim core, two-contribution split, target taxonomy, diagnostic suite). Treat as source-of-truth for *scope*, not for headline numbers.
- `glossary_and_terminology.md` — canonical names for methods, models, datasets, metrics. Do not rename.
- `AGENTS.md` — generic prose-style and editing-policy rubric (not a project-specific structural doc). Useful for "how to phrase claims"; not useful as repo onboarding.

Review-cycle record (reviews, metareview, posted responses, old plans): `../rebuttal/` — see its README.

# Context discipline

These rules exist to keep the context window clean for prose work. Future agents working in this repo should follow them.

- **Consult companion docs on demand, not preemptively.** `glossary_and_terminology.md` is 76 KB / 463 lines and must NOT be `Read` whole — use `Grep` to look up specific terms. `CORE_CLAIM.md` and `AGENTS.md` are smaller; read them only when the task actually requires them (e.g. claim/scope decisions, structural questions).
- **Do NOT read build artifacts** under any circumstances — they are large and useless for prose work:
  - `main.aux`, `main.bbl`, `main.fls`, `main.fdb_latexmk`, `main.out` — generated, never edit, never read
  - `main.pdf` — open in a viewer, never `Read` (binary)
  - `.DS_Store` — ignore
  - `main.log` — read ONLY when explicitly diagnosing a LaTeX warning, and grep for the warning rather than reading top-to-bottom
- **One file at a time for polish passes.** Don't `Read` every section file at the start of a session — load only what the current task touches.

# Critical constraints

- **Never edit, invent, reorder, or "fix"** any of:
  - `\cite{}`, `\citet{}`, `\citep{}`, `\ref{}`, `\eqref{}`, `\label{}`, `\autoref{}`
  - Math inside `$...$`, `$$...$$`, `\(...\)`, `\[...\]`, or `equation`/`align`/`gather` environments
  - `references.bib` entries or bibliography keys
  - Method, model, dataset, or metric names defined in `glossary_and_terminology.md`
- **Float pinning is intentional.** Keep `\begin{figure}[H]` / `\begin{table}[H]`. Never relax to `[h]` / `[htbp]` — figures must stay next to the prose that describes them.
- **No fabricated citations or numbers.** If prose needs a citation that isn't there, flag it in chat. Never paste a plausible-looking `\cite{...}`.

# Editing style

- "Polish" / "tighten" / "fix" = grammar, register, clarity, conciseness. Active voice, concrete subjects, no filler.
- Banned vocabulary: "delve", "leverage", "in the realm of", "it is important to note", "navigate (the/this) landscape", "robustly". Cut hedging ("we believe", "it could be argued").
- Do not rewrite paragraphs wholesale when targeted edits suffice. Prefer the smallest edit that achieves the goal.
- Do not restructure arguments unless explicitly asked. If restructuring is needed, propose the outline first and wait for approval.
- Match the register of the surrounding text — don't import a different paper's voice.
- No end-of-turn summaries. The diff is visible.

# Build

- `latexmk -pdf main.tex` from the repo root. PDF lands at `main.pdf`.
- Section files live under `sections/`. Appendix files under `appendices/`. Figures under `figures/`. Style files (`acl.sty`, `acl_natbib.bst`) under repo root and `style/`.

# Recommended setup

The `academic` output style is pinned in `.claude/settings.local.json` and auto-activates for sessions in this repo. It strips Claude Code's default software-engineering system prompt and replaces it with copyeditor instructions tailored to this paper. The style file is at `.claude/output-styles/academic.md`. If a session starts in the wrong style, run `/output-style academic`.
