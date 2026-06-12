# scripts/

CLI entrypoints for the Sparse Readout Prism paper. Every script here is
self-contained: it holds its own body and imports only the runtime library
(`sparse_readout_prism`) and the shared paper helpers
([`sparse_readout_prism.research`](../src/sparse_readout_prism/research/) — the
Qwen readout toolkit, query-decomposition toolkit, example-prompt banks, cell
metrics, run IO, and the multi-model run-script registry). The placement rule
is simple: logic used by a single entrypoint stays in that entrypoint; a helper
needed by several entrypoints lives in `research/` so it imports cleanly
without `sys.path` hacks. `tests/test_research_layout.py` enforces both
directions.

[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) is the authoritative
figure / table → script + config map; every script below appears in it.

## Buckets

| Bucket       | Purpose                                                                                                          |
|--------------|------------------------------------------------------------------------------------------------------------------|
| `train/`     | Train readout-row factorizers (TopK, Matryoshka, JumpReLU, ...).                                                 |
| `run/`       | End-to-end experiment suites (query-fidelity bank, baseline/null comparisons, lexical-edit / readout-side control stress test).    |
| `eval/`      | Evaluate trained factorizers (task-fidelity evaluation; dense / negative-control row-reconstruction diagnostic). |
| `analyze/`   | Inspect readout-feature projection directions at the final hidden state; W_U row-norm tail stats; feature-label audit. |
| `figures/`   | Compute and persist the metrics behind the paper figures from `results/` artefacts (metric / eval tables, example panels, lens grids, decompositions). This repo is metrics-only: it persists those metrics as CSV/JSON/.pt; the figures themselves are rendered separately in the paper's LaTeX source from these artefacts. |
| `data/`      | Extract readout + hidden states from a model, build query banks, mine polysemy / example sets.                   |

The earlier exploratory families (causal / attention-head circuits,
CoT-faithfulness and agentic-injection probes, the 160M transcoder/SAE lineage,
the classical / tuned proto-token-lens renderers, and the per-snapshot
proto-token-lens PoC plus the readout / logit-trajectory display renderers)
have been removed. See `docs/REPRODUCE.md` §4.

For the runtime API of the library helpers (the `decompose_token_logit`
family) see [`../README.md`](../README.md).
