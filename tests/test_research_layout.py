"""Anti-regression guards for the research-script sub-package.

Shared research code lives in ``sparse_readout_prism.research``; the
``scripts/`` dir holds self-contained entrypoints that import from the package.
These tests fail if the old flat-package smell creeps back in:

  * a script importing a sibling script by bare module name
    (the ``sys.path.insert(SCRIPT_DIR)`` flat-package pattern), or
  * a package module reintroducing a ``sys.path``/``__file__``-depth hack.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
TESTS = REPO / "tests"
PKG = REPO / "src/sparse_readout_prism/research"

# Every ``research/`` module must back >=2 consumers; single-consumer bodies
# live inline in their ``scripts/`` entry point instead. Add a module here only
# if it is a substantial helper factored out of exactly one sibling for
# readability — currently none.
ALLOWED_SINGLE_CONSUMER: set[str] = set()


def _module_names(directory: Path) -> set[str]:
    return {p.stem for p in directory.rglob("*.py") if p.stem != "__init__"}


def test_no_flat_sibling_imports_in_scripts() -> None:
    sibling_stems = _module_names(SCRIPTS)
    offenders: list[str] = []
    for py in sorted(SCRIPTS.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module
                if mod.split(".")[0] in sibling_stems:
                    offenders.append(f"{py.name}: from {mod} import ... (flat sibling)")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in sibling_stems:
                        offenders.append(f"{py.name}: import {alias.name}")
    assert not offenders, (
        "Sibling-script imports reintroduced — import from "
        "sparse_readout_prism.research instead:\n  " + "\n  ".join(offenders)
    )


def test_research_package_has_no_path_hacks() -> None:
    offenders: list[str] = []
    for py in sorted(PKG.rglob("*.py")):
        src = py.read_text()
        if "sys.path.insert" in src or "sys.path.append" in src:
            offenders.append(f"{py.name}: sys.path mutation")
        if ".resolve().parents[" in src or ".resolve().parent\n" in src:
            offenders.append(f"{py.name}: __file__-depth path logic (use sparse_readout_prism.paths)")
    assert not offenders, "Path hack reintroduced in research package:\n  " + "\n  ".join(offenders)


def _imported_module_paths(py: Path) -> set[str]:
    """Dotted module paths a file imports (both ``from a.b import c`` forms)."""
    out: set[str] = set()
    tree = ast.parse(py.read_text(), filename=str(py))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
            for alias in node.names:
                out.add(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def test_research_modules_have_at_least_two_consumers() -> None:
    """``research/`` holds shared bodies only — a module there must be imported
    by >=2 other files (a single-consumer body belongs inline in its scripts/
    entry point). Internal helpers factored out of one sibling are allowlisted.
    """

    def dotted(p: Path) -> str:
        return ".".join(p.relative_to(SRC).with_suffix("").parts)

    consumer_imports = {py: _imported_module_paths(py) for root in (SCRIPTS, SRC, TESTS) for py in root.rglob("*.py")}

    offenders: list[str] = []
    for mod in sorted(PKG.rglob("*.py")):
        if mod.stem == "__init__":
            continue
        target = dotted(mod)
        n_consumers = sum(1 for py, imps in consumer_imports.items() if py != mod and target in imps)
        if n_consumers < 2 and mod.stem not in ALLOWED_SINGLE_CONSUMER:
            offenders.append(f"{mod.relative_to(REPO)}: {n_consumers} consumer(s)")
    assert not offenders, (
        "research/ modules must back >=2 consumers (or be allowlisted as an "
        "internal helper) — inline single-consumer bodies into their scripts/ "
        "entry point instead:\n  " + "\n  ".join(offenders)
    )
