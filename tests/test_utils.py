"""Tests for the shared stats/IO helpers in utils that the run scripts call
(previously each script carried byte-identical private copies)."""

from __future__ import annotations

import json
import math

from sparse_readout_prism.utils import atomic_write_text, pearson, spearman, write_csv, write_jsonl


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


def test_spearman_ties_use_average_ranks():
    # Tied x -> average ranks [1.5, 1.5, 3]; against y ranks [1, 2, 3] the
    # rank correlation is 1.5 / sqrt(3) ~= 0.8660, NOT 1.0. The old
    # argsort-of-argsort transform assigned the ties arbitrary distinct ranks
    # and reported a perfect 1.0.
    r = spearman([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    assert abs(r - 1.5 / math.sqrt(3.0)) < 1e-9


def test_write_csv_unions_ragged_keys(tmp_path):
    p = tmp_path / "rows.csv"
    write_csv(p, [{"a": 1}, {"a": 2, "b": 3}])
    lines = p.read_text().splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1,"
    assert lines[2] == "2,3"


def test_write_csv_empty_rows_behaviour(tmp_path):
    skipped = tmp_path / "skipped.csv"
    write_csv(skipped, [])
    assert not skipped.exists()  # default: no rows -> no file
    empty = tmp_path / "empty.csv"
    write_csv(empty, [], write_empty=True)
    assert empty.exists() and empty.read_text() == ""


def test_write_csv_atomic_and_explicit_fieldnames(tmp_path):
    p = tmp_path / "ordered.csv"
    write_csv(p, [{"b": 2, "a": 1}], fieldnames=["a", "b"], atomic=True)
    assert p.read_text().splitlines()[0] == "a,b"
    assert not (tmp_path / "ordered.csv.tmp").exists()


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
