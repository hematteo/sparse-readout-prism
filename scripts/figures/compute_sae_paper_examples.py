"""Entrypoint shim — implementation moved to a real package home.

See ``sparse_readout_prism.research.figures.compute_sae_paper_examples``. Kept so sbatch jobs and docs
that invoke ``scripts/compute_sae_paper_examples.py`` by path keep working; importers should use the
package path instead of a flat sibling import.
"""

from sparse_readout_prism.research.figures.compute_sae_paper_examples import main

if __name__ == "__main__":
    main()
