"""Tests for the shared stats/IO helpers in utils that the run scripts call
(previously each script carried byte-identical private copies)."""

from __future__ import annotations

import json
import math

from sparse_readout_prism.utils import atomic_write_text, pearson, spearman, write_jsonl


def test_pearson_perfect_and_anticorrelated():
    x = [1.0, 2.0, 3.0, 4.0]
    assert abs(pearson(x, [2 * v + 1 for v in x]) - 1.0) < 1e-12
    assert abs(pearson(x, [-v for v in x]) + 1.0) < 1e-12


def test_pearson_nan_guards():
    assert math.isnan(pearson([1.0, 2.0], [1.0, 2.0]))  # <3 points
    assert math.isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # constant input


def test_spearman_monotonic_nonlinear_is_one():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(spearman(x, [v**3 for v in x]) - 1.0) < 1e-12
    assert math.isnan(spearman([1.0, 2.0], [1.0, 2.0]))


def test_write_jsonl_roundtrip_and_temp_cleanup(tmp_path):
    p = tmp_path / "rows.jsonl"
    rows = [{"a": 1}, {"b": "two"}]
    write_jsonl(p, rows)
    assert [json.loads(line) for line in p.read_text().splitlines()] == rows
    assert not (tmp_path / "rows.jsonl.tmp").exists()


def test_atomic_write_text(tmp_path):
    p = tmp_path / "note.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"
    assert not (tmp_path / "note.txt.tmp").exists()
