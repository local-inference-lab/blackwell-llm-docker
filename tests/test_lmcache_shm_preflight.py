from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "launchers" / "lmcache-shm-preflight.py"
)
SPEC = importlib.util.spec_from_file_location("lmcache_shm_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_required_bytes_converts_gib_exactly() -> None:
    assert MODULE.required_bytes("48") == 48 * 1024**3
    assert MODULE.required_bytes("0.5") == 512 * 1024**2


def test_required_bytes_ceil_is_exact_beyond_decimal_context_precision() -> None:
    value = "1.00000000000000000000000000001"
    numerator, denominator = Decimal(value).as_integer_ratio()
    expected = (numerator * 1024**3 + denominator - 1) // denominator

    assert MODULE.required_bytes(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_required_bytes_rejects_invalid_capacity(value: str) -> None:
    with pytest.raises(ValueError):
        MODULE.required_bytes(value)


def test_required_bytes_rejects_unrepresentable_capacity() -> None:
    with pytest.raises(ValueError):
        MODULE.required_bytes("1e1000000")


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", " name"])
def test_shm_path_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        MODULE.shm_path(name, root=tmp_path)


def test_preflight_removes_unheld_stale_file(monkeypatch, tmp_path: Path) -> None:
    stale = tmp_path / "lmcache-l1"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(MODULE, "holders", lambda _path: [])
    monkeypatch.setattr(MODULE, "available_bytes", lambda _path: 4096)

    result = MODULE.preflight("lmcache-l1", expected_bytes=1024, root=tmp_path)

    assert result == stale
    assert not stale.exists()


def test_preflight_rejects_live_holder(monkeypatch, tmp_path: Path) -> None:
    segment = tmp_path / "lmcache-l1"
    segment.write_bytes(b"active")
    monkeypatch.setattr(MODULE, "holders", lambda _path: ["1234"])

    with pytest.raises(RuntimeError, match="active process"):
        MODULE.preflight("lmcache-l1", expected_bytes=1024, root=tmp_path)

    assert segment.exists()


def test_preflight_refuses_path_replaced_during_holder_scan(
    monkeypatch, tmp_path: Path
) -> None:
    segment = tmp_path / "lmcache-l1"
    segment.write_bytes(b"stale")

    def replace_segment(path: Path) -> list[str]:
        path.unlink()
        path.write_bytes(b"new live segment")
        return []

    monkeypatch.setattr(MODULE, "holders", replace_segment)
    monkeypatch.setattr(MODULE, "available_bytes", lambda _path: 4096)

    with pytest.raises(RuntimeError, match="changed during holder inspection"):
        MODULE.preflight("lmcache-l1", expected_bytes=1024, root=tmp_path)

    assert segment.read_bytes() == b"new live segment"


def test_preflight_rejects_non_file_entry(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "lmcache-l1").mkdir()
    monkeypatch.setattr(MODULE, "holders", lambda _path: [])

    with pytest.raises(RuntimeError, match="regular file"):
        MODULE.preflight("lmcache-l1", expected_bytes=1024, root=tmp_path)


def test_preflight_rejects_insufficient_capacity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(MODULE, "available_bytes", lambda _path: 1023)

    with pytest.raises(RuntimeError, match="insufficient capacity"):
        MODULE.preflight("lmcache-l1", expected_bytes=1024, root=tmp_path)


def test_preflight_accepts_exact_capacity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(MODULE, "available_bytes", lambda _path: 1024)

    result = MODULE.preflight("lmcache-l1", expected_bytes=1024, root=tmp_path)

    assert result == tmp_path / "lmcache-l1"


def test_holders_detects_maps_and_file_descriptors(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    shm_root = tmp_path / "shm"
    shm_root.mkdir()
    segment = shm_root / "lmcache-l1"
    segment.write_bytes(b"")

    maps_pid = proc_root / "100"
    maps_pid.mkdir(parents=True)
    (maps_pid / "maps").write_text(f"0-1 rw-s 0 00:00 0 {segment}\n", encoding="utf-8")
    (maps_pid / "fd").mkdir()

    fd_pid = proc_root / "200"
    (fd_pid / "fd").mkdir(parents=True)
    (fd_pid / "maps").write_text("", encoding="utf-8")
    (fd_pid / "fd" / "7").symlink_to(segment)

    assert MODULE.holders(segment, proc_root=proc_root) == ["100", "200"]


def test_holders_fails_closed_when_proc_metadata_is_unreadable(
    monkeypatch, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    process = proc_root / "100"
    process.mkdir(parents=True)
    maps = process / "maps"
    maps.write_text("", encoding="utf-8")
    (process / "fd").mkdir()
    original_read_text = Path.read_text

    def deny_maps(path: Path, *args, **kwargs):
        if path == maps:
            raise PermissionError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_maps)

    with pytest.raises(RuntimeError, match="cannot inspect process 100"):
        MODULE.holders(tmp_path / "lmcache-l1", proc_root=proc_root)
