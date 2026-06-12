"""Entrypoint shim — implementation moved to a real package home.

See ``sparse_readout_prism.research.data.mine_polysemy_features``. Kept so sbatch/docs that invoke
``scripts/data/mine_polysemy_features.py`` by path keep working; importers use the package path.
"""

from sparse_readout_prism.research.data.mine_polysemy_features import main

if __name__ == "__main__":
    main()
