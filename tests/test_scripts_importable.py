"""Every scripts/ entry point must import cleanly.

Nothing else in the suite imports the scripts (they are exercised via their
CLIs on clusters), so a broken import — a helper renamed in research/ but not
at its call site — is otherwise invisible to CI while leaving the script dead
for a fresh user. Import only; no main() is run, no model/network touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_script_imports(path: Path) -> None:
    name = f"_script_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass decorators resolve cls.__module__ via
    # sys.modules during class creation.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)


def test_scripts_were_collected() -> None:
    # Guard against a silent empty parametrization (e.g. the tree moved).
    assert len(SCRIPTS) >= 10
