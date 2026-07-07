"""Unit tests for research.run_io — the shared Result-1 runner helpers.

All pure dict/tensor logic; the tokenizer-dependent A/B resolver uses the same
stub pattern as tests/test_registry.py. No model loads, no network.
"""

from __future__ import annotations

import json

from sparse_readout_prism.research.run_io import (
    group_rows,
    load_bank,
    margin_row_stats,
    resolve_ab_case,
    run_provenance,
)


class _StubTok:
    """encode() via lookup table; id 0 is the only special token."""

    all_special_ids = [0]
    _TABLE = {
        " cat": [5],
        "cat": [6],
        " bird": [11],
        "bird": [12],
        " dog": [7, 8],
        "dog": [9, 10],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._TABLE.get(text, [1, 2, 3]))


def test_group_rows_by_single_and_tuple_key() -> None:
    rows = [
        {"bank": "a", "method": "x", "v": 1},
        {"bank": "a", "method": "y", "v": 2},
        {"bank": "b", "method": "x", "v": 3},
    ]
    by_bank = group_rows(rows, "bank")
    assert sorted(by_bank) == ["a", "b"]
    assert [r["v"] for r in by_bank["a"]] == [1, 2]
    by_pair = group_rows(rows, ("bank", "method"))
    assert by_pair[("a", "y")] == [rows[1]]


def test_margin_row_stats_bins_and_signs() -> None:
    stats = margin_row_stats({"exact": 1.5, "sparse": 1.2, "resid_term": 0.1})
    assert stats["margin_bin"] == "1-2"
    assert stats["sign_match"] is True
    assert not stats["tiny_margin"]
    assert abs(stats["residual"] - 0.3) < 1e-12
    assert abs(stats["residual_direct"] - 0.3 / 1.5) < 1e-12

    flipped = margin_row_stats({"exact": 0.2, "sparse": -0.1, "resid_term": 0.0})
    assert flipped["margin_bin"] == "<0.5"
    assert flipped["sign_match"] is False
    assert flipped["tiny_margin"]


def test_load_bank_caps_by_base_case_not_rows(tmp_path) -> None:
    rows = [
        {"base_case_id": "b1", "case_id": "c1"},
        {"base_case_id": "b1", "case_id": "c2"},
        {"base_case_id": "b2", "case_id": "c3"},
        {"base_case_id": "b3", "case_id": "c4"},
    ]
    p = tmp_path / "bank.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    kept = load_bank(p, cap=2)
    # cap=2 keeps ALL rows of the first two base cases (3 rows), not 2 raw rows.
    assert [r["case_id"] for r in kept] == ["c1", "c2", "c3"]
    assert len(load_bank(p, cap=None)) == 4


def test_resolve_ab_case_single_token_success_and_audit() -> None:
    audit: list[dict] = []
    skipped: list[dict] = []
    r = {"family": "animals", "case_id": "c1", "target_a": "cat", "target_b": "bird"}
    out = resolve_ab_case(_StubTok(), r, "model_a", audit, skipped)
    assert out == ([5], [11], "animals", "single_token")
    assert [a["side"] for a in audit] == ["A", "B"]
    assert all(a["reason"] == "ok" for a in audit)
    assert skipped == []


def test_resolve_ab_case_collision_and_multi_token_skip() -> None:
    audit: list[dict] = []
    skipped: list[dict] = []
    collision = {"family": "f", "case_id": "c2", "target_a": "cat", "target_b": "cat"}
    assert resolve_ab_case(_StubTok(), collision, "m", audit, skipped) is None
    assert skipped[-1]["reason"] == "ab_collision"

    multi = {"family": "f", "case_id": "c3", "target_a": "cat", "target_b": "dog"}
    assert resolve_ab_case(_StubTok(), multi, "m", audit, skipped) is None
    assert skipped[-1]["case_id"] == "c3"


def test_resolve_ab_case_family_paths() -> None:
    audit: list[dict] = []
    skipped: list[dict] = []
    r = {
        "family": "animals",
        "case_id": "c4",
        "target_a": None,
        "target_a_family": ["cat", "dog"],
        "target_b_family": ["bird"],
    }
    out = resolve_ab_case(_StubTok(), r, "m", audit, skipped)
    # "dog" is multi-token -> dropped from the family, not fatal.
    assert out == ([5], [11], "animals_family", "token_family")

    empty = {
        "family": "animals",
        "case_id": "c5",
        "target_a": None,
        "target_a_family": ["dog"],
        "target_b_family": ["bird"],
    }
    assert resolve_ab_case(_StubTok(), empty, "m", audit, skipped) is None
    assert skipped[-1]["reason"] == "family_empty"


def test_run_provenance_records_args_and_versions() -> None:
    prov = run_provenance({"out_dir": "results/x", "k": 256})
    assert prov["args"] == {"out_dir": "results/x", "k": 256}
    assert prov["package_version"]
    assert prov["torch_version"]
    assert "command" in prov and "timestamp_unix" in prov
