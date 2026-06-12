"""Tests for the workspace path helpers.

Env-independent: SRP_SSD_ROOT is monkeypatched to a tmp_path so nothing touches
a real SSD."""

from __future__ import annotations

from pathlib import Path

from sparse_readout_prism.paths import repo_root, ssd_root


def test_ssd_root_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SRP_SSD_ROOT", str(tmp_path))
    assert ssd_root() == Path(str(tmp_path))


def test_ssd_root_defaults_under_repo(monkeypatch):
    monkeypatch.delenv("SRP_SSD_ROOT", raising=False)
    assert ssd_root().name == "local_snapshots"


def test_repo_root_contains_pyproject():
    root = repo_root()
    assert (root / "pyproject.toml").exists()
