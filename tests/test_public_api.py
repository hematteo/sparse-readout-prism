"""Pin the top-level re-export contract that the README quickstart imports.

Every other test imports from submodules; this one exercises the package facade
(`from sparse_readout_prism import ...`) so a move/rename inside the package that
silently drops a name from `__init__.__all__` is caught here. Import-only: no
model loads, no network."""

from __future__ import annotations

import inspect

import torch

import sparse_readout_prism as srp

# Re-export kinds, grounded in the actual source (factorizers.py / data.py /
# decompose.py), not assumptions:
#   - plain functions
#   - SAE classes: SAEBase + concrete archs, all subclass torch.nn.Module
#   - dataclasses Decomposition / PrismDataset / FactorizerBatch: plain classes,
#     deliberately NOT nn.Module.
_FUNCTIONS = {
    "center_normalize_rows",
    "decompose_token_logit",
    "load_prism_dataset",
    "preprocess_rows",
    "build_factorizer",
    "load_factorizer",
}
_SAE_CLASSES = {
    "SAEBase",
    "BatchTopKSAE",
    "GatedSAE",
    "JumpReLUSAE",
    "L1ReLUSAE",
    "TopKSAE",
}
_PLAIN_CLASSES = {
    "Decomposition",
    "PrismDataset",
    "FactorizerBatch",
}


def test_version_is_nonempty_str() -> None:
    assert isinstance(srp.__version__, str)
    assert srp.__version__  # non-empty


def test_all_names_importable_and_in_dir() -> None:
    assert "__version__" in srp.__all__
    listing = dir(srp)
    for name in srp.__all__:
        assert hasattr(srp, name), f"{name} in __all__ but not importable"
        assert name in listing, f"{name} not in dir(sparse_readout_prism)"


def test_all_list_matches_expected_partition() -> None:
    # Guards against a name being added to __all__ without this test classifying
    # it (or one being dropped). "__version__" is the only non-object entry.
    classified = _FUNCTIONS | _SAE_CLASSES | _PLAIN_CLASSES | {"__version__"}
    assert set(srp.__all__) == classified


def test_functions_are_callable() -> None:
    for name in _FUNCTIONS:
        obj = getattr(srp, name)
        assert inspect.isfunction(obj), f"{name} is not a function"


def test_sae_classes_are_nn_module_subclasses() -> None:
    for name in _SAE_CLASSES:
        obj = getattr(srp, name)
        assert inspect.isclass(obj), f"{name} is not a class"
        assert issubclass(obj, torch.nn.Module), f"{name} is not an nn.Module subclass"


def test_plain_classes_are_classes_not_modules() -> None:
    for name in _PLAIN_CLASSES:
        obj = getattr(srp, name)
        assert inspect.isclass(obj), f"{name} is not a class"
        assert not issubclass(obj, torch.nn.Module), f"{name} should not be an nn.Module"
