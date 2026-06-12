# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026

Initial public research release accompanying the paper.

### Added

- Core `sparse_readout_prism` library: row factorizers (TopK, Matryoshka,
  JumpReLU, Gated), linear logit decomposition, fidelity-gated evaluation, and
  the dataset assembly + training loop with a train -> evaluate -> label-features
  runner pipeline.
- Research scripts and configs (model configs, sweep grids, paper-selected SAE
  registry) plus the curated query-bank assets used as paper inputs.
- CI-safe pytest suite covering decomposition identity correctness, evaluator
  schema invariants, and `research/` layout guardrails (no GPU, no model loads).
- Documentation: method TL;DR, experiment-design map, reproduction map, data
  schema, and query-bank schema.

### Notes

- Pretrained readout-feature dictionaries are published separately on the
  Hugging Face Hub at `matteohe/sparse-readout-prism` (17 dictionaries across 8
  base models), not bundled in this source release.
