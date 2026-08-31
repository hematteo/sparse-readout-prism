"""Shared helpers for the Sparse Readout Prism paper scripts.

This sub-package holds only the code that more than one ``scripts/`` entry
point needs, so it can be imported without ``sys.path`` hacks. The placement
rule is:

  * An entry point keeps its body inline in its ``scripts/`` file
    (self-contained -- this is how every script in ``scripts/`` works).
  * A helper needed by several entry points lives here, as a flat module:
    the Qwen readout toolkit (``qwen_readout``), the query-decomposition
    toolkit (``query_decompose``), the static example-prompt banks
    (``qwen_example_prompts``), cell metrics (``cell_metrics``), run IO
    (``run_io``), the multi-model run-script registry (``registry``), the
    seed-stability contrast pipeline (``seed_stability``), row-geometry
    helpers shared by the baselines and stability scripts (``row_geometry``),
    the cross-lens study toolkit (``cross_lens``), and the CoarseWSD-20
    bundle/statistics helpers (``wsd``).

So ``research/`` is not a mirror of ``scripts/``: every module here backs at
least two consumers (enforced by ``tests/test_research_layout.py``), and no
module here is itself an entry point.

Runtime library helpers used by the documented paper API (data, decompose,
evaluate, factorizers, ...) live one level up in ``sparse_readout_prism``
itself; this sub-package is scoped to the paper-reproduction scripts.
"""
