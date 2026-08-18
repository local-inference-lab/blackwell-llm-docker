#!/usr/bin/env python3
"""Validate a named LMCache shared-memory region before server startup."""

from __future__ import annotations

import argparse
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

_GIB = 1024**3
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")


def required_bytes(size_gb: str) -> int:
    """Convert a positive GiB capacity to bytes without float rounding."""
    if len(size_gb) > 128:
        raise ValueError("shared-memory capacity input is unreasonably long")
    try:
        size = Decimal(size_gb)
    except InvalidOperation as exc:
        raise ValueError(f"invalid shared-memory capacity: {size_gb!r}") from exc
    if not size.is_finite() or size <= 0:
        raise ValueError(f"shared-memory capacity must be positive: {size_gb!r}")
    if size.adjusted() > 9:
        raise ValueError(
            f"shared-memory capacity is too large to represent: {size_gb!r}"
        )
    numerator, denominator = size.as_integer_ratio()
    result = (numerator * _GIB + denominator - 1) // denominator
    if result > sys.maxsize:
        raise ValueError(
            f"shared-memory capacity is too large to represent: {size_gb!r}"
        )
    return result


def shm_path(name: str, *, root: Path = Path("/dev/shm")) -> Path:
    """Return a path below ``root`` for one POSIX-style shared-memory name."""
    if name in {"", ".", ".."} or _SAFE_NAME.fullmatch(name) is None:
        raise ValueError(
            "shared-memory name must start with an alphanumeric character and "
            "contain only alphanumerics, dot, underscore, or hyphen"
        )
    return root / name


def _normalized_proc_target(value: str) -> str:
    deleted_suffix = " (deleted)"
    return value.removesuffix(deleted_suffix)


def holders(path: Path, *, proc_root: Path = Path("/proc")) -> list[str]:
    """Return process IDs that map or hold an FD for ``path``."""
    target = str(path)
    found: set[str] = set()
    try:
        processes = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot inspect process table {proc_root}: {exc}") from exc

    for process in processes:
        if not process.name.isdigit():
            continue
        # ``preflight`` holds the candidate inode open while scanning so a
        # replacement cannot reuse its inode. Ignore that deliberate self-FD.
        if process.name == str(os.getpid()):
            continue
        try:
            for line in (
                (process / "maps")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            ):
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and _normalized_proc_target(fields[5]) == target:
                    found.add(process.name)
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect process {process.name} mappings: {exc}"
            ) from exc

        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    linked = _normalized_proc_target(os.readlink(descriptor))
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"cannot inspect process {process.name} descriptor "
                        f"{descriptor.name}: {exc}"
                    ) from exc
                if linked == target:
                    found.add(process.name)
                    break
        except (FileNotFoundError, ProcessLookupError):
            pass
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect process {process.name} descriptors: {exc}"
            ) from exc

    return sorted(found, key=int)


def available_bytes(path: Path) -> int:
    """Return bytes available to unprivileged allocations in ``path``'s FS."""
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def preflight(
    name: str,
    *,
    expected_bytes: int,
    root: Path = Path("/dev/shm"),
) -> Path:
    """Remove an unheld stale region and verify capacity for a new one."""
    if expected_bytes <= 0:
        raise ValueError("expected_bytes must be positive")
    if not root.is_dir():
        raise RuntimeError(f"shared-memory root is not a directory: {root}")

    path = shm_path(name, root=root)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"shared-memory entry is not a regular file and will not be removed: {path}"
            )
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise RuntimeError(
                f"shared-memory entry changed during holder inspection: {path}"
            ) from exc
        try:
            inspected = os.fstat(descriptor)
            active = holders(path)
            if active:
                raise RuntimeError(
                    f"shared-memory entry {path} is held by an active process: "
                    + ", ".join(active)
                )
            try:
                current = path.stat(follow_symlinks=False)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"shared-memory entry changed during holder inspection: {path}"
                ) from exc
            if (current.st_dev, current.st_ino) != (
                inspected.st_dev,
                inspected.st_ino,
            ):
                raise RuntimeError(
                    f"shared-memory entry changed during holder inspection: {path}"
                )
            path.unlink()
        finally:
            os.close(descriptor)

    available = available_bytes(root)
    if available < expected_bytes:
        raise RuntimeError(
            "insufficient capacity in "
            f"{root}: need {expected_bytes} bytes, available {available} bytes"
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-gb", required=True)
    parser.add_argument("--root", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()

    try:
        expected = required_bytes(args.expected_gb)
        path = preflight(args.name, expected_bytes=expected, root=args.root)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        f"LMCache shared-memory preflight passed: path={path} "
        f"expected_bytes={expected} available_bytes={available_bytes(args.root)}"
    )


if __name__ == "__main__":
    main()
