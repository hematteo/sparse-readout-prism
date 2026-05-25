# Project: Sparse Readout Prism — paper repo

This repo is a **LaTeX research manuscript** targeted at a top-tier venue. It is *not* a software codebase. Treat `.tex` files as formal scientific prose.

Companion context lives in `AGENTS.md` (architecture / claim scaffolding), `CORE_CLAIM.md` (what we are arguing), and `glossary_and_terminology.md` (canonical names — do not rename these). Read those before making non-trivial edits.

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

Activate the `academic` output style for this repo: `/output-style academic`. It strips Claude Code's default software-engineering system prompt and replaces it with copyeditor instructions tailored to this paper. The style file is at `.claude/output-styles/academic.md`.
