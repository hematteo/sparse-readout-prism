"""Registry resolution: env-var expansion, absolute-entry handling, strict token resolver.

`resolve_registry` is pure (YAML in, dict out) and is the seam where the
shipped Result-1 reproduction path lives — an unexpanded `${SRP_ARCHIVE_ROOT}`
here silently turns every runner cell into a "checkpoint missing" skip, so this
file pins the expansion contract. No model loads, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparse_readout_prism.research.registry import resolve_registry, resolve_single_token_strict

ROOT = Path(__file__).resolve().parents[1]

_REGISTRY_YAML = """
archive_root: ${TEST_SRP_ARCHIVE}
models:
  model_a:
    model_id: org/model-a
    revision: null
    w_u_artifact: data/model_a/model_a.pt
    manifest: data/model_a/manifest.json
    operating_points:
      fidelity:
        id: model_a_k256
        checkpoint: results/model_a/checkpoint.pt
      strict_budget:
        # Embedded-root entry (the historical Ministral/R1-distill style):
        # must resolve to the same place as a root-relative one, not doubled.
        id: model_a_k128
        checkpoint: ${TEST_SRP_ARCHIVE}/results/model_a_16x/checkpoint.pt
"""


def _write_registry(tmp_path: Path) -> Path:
    p = tmp_path / "registry.yaml"
    p.write_text(_REGISTRY_YAML)
    return p


def test_resolve_registry_expands_env_var(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_SRP_ARCHIVE", "/archive/root")
    reg = resolve_registry(_write_registry(tmp_path))
    assert reg["archive_root"] == "/archive/root"
    entry = reg["models"]["model_a"]
    assert entry["w_u_artifact"] == "/archive/root/data/model_a/model_a.pt"
    assert entry["manifest"] == "/archive/root/data/model_a/manifest.json"
    ops = entry["operating_points"]
    assert ops["fidelity"]["checkpoint"] == "/archive/root/results/model_a/checkpoint.pt"
    # Embedded-root entry: absolute after expansion, so NOT joined again.
    assert ops["strict_budget"]["checkpoint"] == "/archive/root/results/model_a_16x/checkpoint.pt"
    # No literal placeholder may survive anywhere.
    assert "${" not in str(reg)


def test_resolve_registry_raises_on_unset_env_var(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TEST_SRP_ARCHIVE", raising=False)
    with pytest.raises(ValueError, match="SRP_ARCHIVE_ROOT|environment variable"):
        resolve_registry(_write_registry(tmp_path))


def test_shipped_result1_registry_resolves(monkeypatch) -> None:
    monkeypatch.setenv("SRP_ARCHIVE_ROOT", "/archive/root")
    reg = resolve_registry(ROOT / "configs/registries/result1_query_fidelity_cluster.yaml")
    assert reg["models"], "shipped registry has no models"
    for name, entry in reg["models"].items():
        assert "${" not in entry["w_u_artifact"], name
        for op_name, op in entry["operating_points"].items():
            ckpt = op["checkpoint"]
            assert "${" not in ckpt, (name, op_name)
            assert ckpt.startswith("/archive/root/"), (name, op_name, ckpt)
            # Doubled-prefix guard for the historical embedded-root entries.
            assert "/archive/root/archive/root" not in ckpt, (name, op_name, ckpt)


class _StubTok:
    """Minimal tokenizer stub: encode() via lookup table, id 0 is special."""

    all_special_ids = [0]
    _TABLE = {
        " cat": [5],
        "cat": [6],
        " dog": [7, 8],
        "dog": [9, 10],
        " eos": [0],
        "eos": [0],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._TABLE.get(text, [1, 2, 3]))


def test_resolve_single_token_strict_prefers_space_variant() -> None:
    tid, variant, reason = resolve_single_token_strict(_StubTok(), "cat")
    assert (tid, variant, reason) == (5, " +s", "")


def test_resolve_single_token_strict_rejects_special_and_multi() -> None:
    tok = _StubTok()
    tid, variant, reason = resolve_single_token_strict(tok, "eos")
    assert tid is None and reason == "special_token"
    tid, _variant, reason = resolve_single_token_strict(tok, "dog")
    assert tid is None and reason.startswith("multi_token(")
    tid, _variant, reason = resolve_single_token_strict(tok, "")
    assert tid is None and reason == "empty"
