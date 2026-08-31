"""Shared test helpers.

``load_script`` imports a ``scripts/`` entry point by path the way
``test_scripts_importable.py`` does: the module is registered in
``sys.modules`` before execution so dataclass decorators can resolve
``cls.__module__``. Test modules use it as ``from conftest import load_script``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(rel_path: str, name: str | None = None) -> ModuleType:
    path = REPO_ROOT / rel_path
    mod_name = name or "_test_script_" + path.stem
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None, rel_path
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
