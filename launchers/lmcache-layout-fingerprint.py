#!/usr/bin/env python3
"""Version a persistent LMCache L2 directory by cache-layout inputs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATE_FILE = ".lmcache-layout-fingerprint.json"
LOCK_FILE = ".lmcache-layout-fingerprint.lock"
STATE_TEMP_PREFIX = f"{STATE_FILE}.tmp."
_IGNORED_SOURCE_DIRS = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
_IGNORED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_symlink():
        return {"path": str(path), "kind": "symlink", "target": os.readlink(path)}
    if path.is_file():
        return {"path": str(path), "kind": "file", "sha256": _sha256_file(path)}
    if not path.is_dir():
        raise ValueError(f"unsupported fingerprint input: {path}")

    members: list[dict[str, str]] = []
    for member in sorted(path.rglob("*"), key=lambda item: str(item.relative_to(path))):
        relative_path = member.relative_to(path)
        if any(part in _IGNORED_SOURCE_DIRS for part in relative_path.parts):
            continue
        if member.suffix in _IGNORED_SOURCE_SUFFIXES:
            continue
        relative = str(relative_path)
        if member.is_symlink():
            members.append(
                {"path": relative, "kind": "symlink", "target": os.readlink(member)}
            )
        elif member.is_file():
            members.append(
                {"path": relative, "kind": "file", "sha256": _sha256_file(member)}
            )
    return {"path": str(path), "kind": "directory", "members": members}


def build_manifest(
    *,
    config_paths: list[Path],
    software_paths: list[Path],
    values: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """Build a canonical manifest and its SHA-256 layout identity."""
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "inputs": {
            "configs": [_path_manifest(path) for path in sorted(config_paths, key=str)],
            "software": [
                _path_manifest(path) for path in sorted(software_paths, key=str)
            ],
            "values": dict(sorted(values.items())),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return {**payload, "digest": digest}, digest


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or state.get("schema") != SCHEMA_VERSION:
        return None
    digest = state.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    return state


def _write_state(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _data_entries(path: Path) -> list[Path]:
    metadata = {STATE_FILE, LOCK_FILE}
    return [entry for entry in path.iterdir() if entry.name not in metadata]


def _remove_stale_state_temps(path: Path) -> None:
    for entry in path.iterdir():
        if entry.name.startswith(STATE_TEMP_PREFIX) and (
            entry.is_file() or entry.is_symlink()
        ):
            entry.unlink()


def _clear_data(path: Path) -> None:
    for entry in _data_entries(path):
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            shutil.rmtree(entry)
    (path / STATE_FILE).unlink(missing_ok=True)


def reconcile(
    l2_path: Path,
    manifest: dict[str, Any],
    *,
    policy: str,
    adopt_unmarked: bool = False,
) -> str:
    """Preserve or reset one dedicated L2 directory under an exclusive lock."""
    if policy not in {"auto", "always", "never"}:
        raise ValueError(f"invalid L2 reset policy: {policy!r}")
    digest = manifest.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("manifest digest must be a SHA-256 hex string")

    l2_path.mkdir(parents=True, exist_ok=True)
    resolved = l2_path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(
            f"refusing to manage filesystem root as LMCache L2: {resolved}"
        )

    lock_path = resolved / LOCK_FILE
    state_path = resolved / STATE_FILE
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _remove_stale_state_temps(resolved)
        previous = _read_state(state_path)
        compatible = previous is not None and previous.get("digest") == digest

        if policy == "never":
            return "compatible" if compatible else "mismatch-preserved"
        if policy == "always":
            _clear_data(resolved)
            _write_state(state_path, manifest)
            return "reset"
        if compatible:
            return "compatible"

        had_data = bool(_data_entries(resolved))
        had_state = state_path.exists()
        if had_data and not had_state and not adopt_unmarked:
            raise RuntimeError(
                "refusing to reset an unversioned non-empty L2 directory; "
                "pass --adopt-unmarked once after confirming that the path is "
                "dedicated to LMCache"
            )
        if had_data or had_state:
            _clear_data(resolved)
            result = "reset"
        else:
            result = "initialized"
        _write_state(state_path, manifest)
        return result


def _parse_values(raw_values: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in raw_values:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"layout value must be key=value: {raw!r}")
        if key in values:
            raise ValueError(f"duplicate layout value: {key}")
        values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-path", type=Path, required=True)
    parser.add_argument("--policy", choices=("auto", "always", "never"), required=True)
    parser.add_argument(
        "--adopt-unmarked",
        action="store_true",
        help="allow auto policy to reset a non-empty directory with no state file",
    )
    parser.add_argument("--config-path", type=Path, action="append", default=[])
    parser.add_argument("--software-path", type=Path, action="append", default=[])
    parser.add_argument("--value", action="append", default=[])
    args = parser.parse_args()

    try:
        values = _parse_values(args.value)
        manifest, digest = build_manifest(
            config_paths=args.config_path,
            software_paths=args.software_path,
            values=values,
        )
        result = reconcile(
            args.l2_path,
            manifest,
            policy=args.policy,
            adopt_unmarked=args.adopt_unmarked,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "digest": digest,
                "l2_path": str(args.l2_path.resolve()),
                "policy": args.policy,
                "result": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
