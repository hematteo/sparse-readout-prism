"""Sparse Readout Prism: a sparse LM-head basis for logit-lens readouts.

Public API:

    from sparse_readout_prism import (
        preprocess_rows,
        build_factorizer,
        load_factorizer,
        decompose_token_logit,
        Decomposition,
        SAEBase,
        TopKSAE,
    )

See the README for the runtime quickstart, ``docs/REPRODUCE.md`` for the
paper-figure -> script mapping, and ``scripts/`` for CLI entry points.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("sparse-readout-prism")
except PackageNotFoundError:  # editable install before package metadata is built
    __version__ = "0.0.0+unknown"

from sparse_readout_prism.data import (
    PrismDataset,
    load_prism_dataset,
    preprocess_rows,
)
from sparse_readout_prism.decompose import Decomposition, decompose_token_logit
from sparse_readout_prism.factorizers import (
    BatchTopKSAE,
    FactorizerBatch,
    GatedSAE,
    JumpReLUSAE,
    L1ReLUSAE,
    SAEBase,
    TopKSAE,
    build_factorizer,
    load_factorizer,
)

__all__ = [
    "__version__",
    "Decomposition",
    "PrismDataset",
    "decompose_token_logit",
    "load_prism_dataset",
    "preprocess_rows",
    "build_factorizer",
    "load_factorizer",
    "BatchTopKSAE",
    "FactorizerBatch",
    "GatedSAE",
    "JumpReLUSAE",
    "L1ReLUSAE",
    "SAEBase",
    "TopKSAE",
]
