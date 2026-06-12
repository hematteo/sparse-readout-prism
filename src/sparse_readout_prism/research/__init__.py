"""Shared bodies for the Sparse Readout Prism paper scripts.

This sub-package holds only the code that more than one ``scripts/`` entry
point needs, so it can be imported without ``sys.path`` hacks. The placement
rule is:

  * An entry point with a unique body keeps that body inline in its
    ``scripts/`` file (self-contained -- this is how every ``run/`` script and
    the larger ``figures/`` charts work).
  * A body shared by several consumers lives here, and each entry point that
    needs it is a one-line shim that calls ``main()`` on the module. The shim
    and its body then sit at parallel paths, e.g.::

        scripts/figures/compute_sae_paper_examples.py                     (thin shim)
        sparse_readout_prism.research.figures.compute_sae_paper_examples  (shared body)

So ``research/`` is not a mirror of ``scripts/``: it contains a module only
when that module has more than one consumer (enforced by
``tests/test_research_layout.py``, which allowlists internal helpers factored
out of a single sibling body). Sub-packages ``figures`` / ``data`` group the
shared bodies by the ``scripts/`` bucket they back; ``_common`` holds helpers
with no shim of their own (the Qwen readout toolkit, the selected-score /
query-decomposition helpers, cell metrics, run IO, and the multi-model
run-script registry / single-token helpers in ``_common.registry``).

Runtime library helpers used by the documented paper API (data, decompose,
evaluate, factorizers, ...) live one level up in ``sparse_readout_prism``
itself; this sub-package is scoped to the paper-reproduction scripts.
"""
