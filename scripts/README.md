# scripts/

CLI entrypoints for the Sparse Readout Prism paper. Library code (functions,
classes, decomposition primitives) lives in
[`src/sparse_readout_prism/`](../src/sparse_readout_prism/). The placement rule
is simple: an entrypoint with a unique body holds that body itself
(self-contained — most scripts here); a body shared by several entrypoints
moves into
[`src/sparse_readout_prism/research/`](../src/sparse_readout_prism/research/)
so it imports cleanly without `sys.path` hacks, and the entrypoint becomes a
one-line shim that calls `main()` on it. `research/` is therefore **not** a
mirror of `scripts/` — it contains a module only when that module has more than
one consumer. Its verb sub-packages (`figures/`, `data/`, plus
`_common/`) group those shared bodies by the bucket they back, so a shim and
its body sit at parallel paths.

## Buckets

| Bucket       | Purpose                                                                                                          |
|--------------|------------------------------------------------------------------------------------------------------------------|
| `train/`     | Train readout-row factorizers (TopK, Matryoshka, JumpReLU, ...).                                                 |
| `run/`       | End-to-end experiment suites (query-fidelity bank, baseline/null comparisons, lexical-edit / readout-side control stress test).    |
| `eval/`      | Evaluate trained factorizers (task-fidelity evaluation; dense / negative-control row-reconstruction diagnostic). |
| `analyze/`   | Inspect readout-feature projection directions at the final hidden state; W_U row-norm tail stats; feature-label audit. |
| `figures/`   | Compute and persist the metrics behind the paper figures from `results/` artefacts (metric / eval tables, example panels, lens grids, decompositions). This repo is metrics-only: it persists those metrics as CSV/JSON/.pt; the figures themselves are rendered separately in the paper's LaTeX source from these artefacts. |
| `data/`      | Extract readout + hidden states from a model, build query banks, mine polysemy / example sets.                   |

## Which scripts reproduce the paper?

[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) is the authoritative
figure / table → script + config map. If a script doesn't appear there, it is
**internal research scaffolding**: a probe, baseline, or diagnostic that was
useful while writing the paper but is not part of the headline reproduction
path. These are kept because the canonical scripts still import helpers from
them. They run, but the public surface area is the reproducing-doc subset.

A few modules ship only as importable helper hubs, and live in the
`sparse_readout_prism.research` package rather than `scripts/`: the Qwen
readout toolkit and shared decomposition helpers in `_common/`, the shared
body that computes the metrics behind the SAE paper examples
(`figures/compute_sae_paper_examples.py`), and the feature miner
(`data/mine_polysemy_features.py`). Each is reached through a one-line shim at
the parallel `scripts/` path (the shared `_common/` helpers additionally back
several entrypoints), so it ships even though it is not itself a headline
entrypoint.
The earlier exploratory families (causal / attention-head circuits,
CoT-faithfulness and agentic-injection probes, the 160M transcoder/SAE lineage,
the classical / tuned proto-token-lens renderers, and the per-snapshot
proto-token-lens PoC plus the readout / logit-trajectory display renderers)
have been removed. See `docs/REPRODUCE.md` §4.

For the runtime API of the library helpers (the `decompose_token_logit`
family) see [`../README.md`](../README.md).
