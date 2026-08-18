from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "launchers" / "lmcache-layout-fingerprint.py"
)
SPEC = importlib.util.spec_from_file_location("lmcache_layout_fingerprint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_manifest_is_stable_and_content_addressed(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    source = tmp_path / "source.py"
    config.write_text('{"layers": 61}\n', encoding="utf-8")
    source.write_text("LAYOUT_VERSION = 1\n", encoding="utf-8")

    first_manifest, first_digest = MODULE.build_manifest(
        config_paths=[config],
        software_paths=[source],
        values={"chunk_size": "12288", "dcp_size": "8"},
    )
    second_manifest, second_digest = MODULE.build_manifest(
        config_paths=[config],
        software_paths=[source],
        values={"dcp_size": "8", "chunk_size": "12288"},
    )

    assert first_manifest == second_manifest
    assert first_digest == second_digest

    config.write_text('{"layers": 62}\n', encoding="utf-8")
    _, changed_digest = MODULE.build_manifest(
        config_paths=[config],
        software_paths=[source],
        values={"chunk_size": "12288", "dcp_size": "8"},
    )
    assert changed_digest != first_digest


def test_build_manifest_hashes_directory_members(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("A = 1\n", encoding="utf-8")
    _, first = MODULE.build_manifest(
        config_paths=[], software_paths=[source], values={}
    )
    (source / "b.py").write_text("B = 2\n", encoding="utf-8")
    _, second = MODULE.build_manifest(
        config_paths=[], software_paths=[source], values={}
    )
    assert first != second


def test_build_manifest_ignores_generated_python_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("A = 1\n", encoding="utf-8")
    _, before = MODULE.build_manifest(
        config_paths=[], software_paths=[source], values={}
    )

    pycache = source / "__pycache__"
    pycache.mkdir()
    (pycache / "a.cpython-312.pyc").write_bytes(b"runtime bytecode")
    (source / ".ruff_cache").mkdir()
    (source / ".ruff_cache" / "state").write_bytes(b"runtime cache")
    _, after_runtime_files = MODULE.build_manifest(
        config_paths=[], software_paths=[source], values={}
    )

    assert after_runtime_files == before

    (source / "b.py").write_text("B = 2\n", encoding="utf-8")
    _, after_source_change = MODULE.build_manifest(
        config_paths=[], software_paths=[source], values={}
    )
    assert after_source_change != before


def test_build_manifest_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MODULE.build_manifest(
            config_paths=[tmp_path / "missing.json"],
            software_paths=[],
            values={},
        )


def _manifest(digest: str) -> dict[str, object]:
    return {"schema": 1, "digest": digest * 64, "inputs": {"test": digest}}


def test_auto_initializes_empty_l2_directory(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()

    result = MODULE.reconcile(l2, _manifest("a"), policy="auto")

    assert result == "initialized"
    state = json.loads((l2 / MODULE.STATE_FILE).read_text(encoding="utf-8"))
    assert state["digest"] == "a" * 64


def test_auto_removes_stale_state_temporary_files(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    stale = l2 / f"{MODULE.STATE_FILE}.tmp.123"
    stale.write_text("partial", encoding="utf-8")

    result = MODULE.reconcile(l2, _manifest("a"), policy="auto")

    assert result == "initialized"
    assert not stale.exists()


def test_auto_preserves_matching_objects(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    MODULE.reconcile(l2, _manifest("a"), policy="auto")
    cached = l2 / "object-1"
    cached.write_bytes(b"cache")

    result = MODULE.reconcile(l2, _manifest("a"), policy="auto")

    assert result == "compatible"
    assert cached.read_bytes() == b"cache"


def test_auto_clears_mismatched_objects(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    MODULE.reconcile(l2, _manifest("a"), policy="auto")
    (l2 / "object-1").write_bytes(b"cache")
    nested = l2 / "nested"
    nested.mkdir()
    (nested / "object-2").write_bytes(b"cache")

    result = MODULE.reconcile(l2, _manifest("b"), policy="auto")

    assert result == "reset"
    assert not (l2 / "object-1").exists()
    assert not nested.exists()
    state = json.loads((l2 / MODULE.STATE_FILE).read_text(encoding="utf-8"))
    assert state["digest"] == "b" * 64


def test_auto_refuses_unversioned_objects_without_adoption(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    legacy = l2 / "legacy-object"
    legacy.write_bytes(b"cache")

    with pytest.raises(RuntimeError, match="unversioned"):
        MODULE.reconcile(l2, _manifest("a"), policy="auto")

    assert legacy.read_bytes() == b"cache"


def test_auto_adopts_and_clears_unversioned_objects(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    (l2 / "legacy-object").write_bytes(b"cache")

    result = MODULE.reconcile(l2, _manifest("a"), policy="auto", adopt_unmarked=True)

    assert result == "reset"
    assert not (l2 / "legacy-object").exists()


def test_never_preserves_mismatch_and_original_state(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    MODULE.reconcile(l2, _manifest("a"), policy="auto")
    cached = l2 / "object-1"
    cached.write_bytes(b"cache")

    result = MODULE.reconcile(l2, _manifest("b"), policy="never")

    assert result == "mismatch-preserved"
    assert cached.exists()
    state = json.loads((l2 / MODULE.STATE_FILE).read_text(encoding="utf-8"))
    assert state["digest"] == "a" * 64


def test_always_clears_even_when_fingerprint_matches(tmp_path: Path) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    MODULE.reconcile(l2, _manifest("a"), policy="auto")
    cached = l2 / "object-1"
    cached.write_bytes(b"cache")

    result = MODULE.reconcile(l2, _manifest("a"), policy="always")

    assert result == "reset"
    assert not cached.exists()


@pytest.mark.parametrize("policy", ["", "sometimes", "AUTO"])
def test_reconcile_rejects_invalid_policy(tmp_path: Path, policy: str) -> None:
    l2 = tmp_path / "l2"
    l2.mkdir()
    with pytest.raises(ValueError):
        MODULE.reconcile(l2, _manifest("a"), policy=policy)


def test_reconcile_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        MODULE.reconcile(Path("/"), _manifest("a"), policy="auto")
