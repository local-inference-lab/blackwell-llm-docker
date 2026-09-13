from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.jovian_wheel_runtime.verify_release_assets import verify_release


COMMIT = "2" * 40
BETA_TAG = f"jovian-cu133-foundation-beta-{COMMIT}"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_release(directory: Path, promotion: bool = False) -> None:
    wheels = {f"package-{index}.whl": f"wheel {index}".encode() for index in range(6)}
    for name, payload in wheels.items():
        (directory / name).write_bytes(payload)
    manifest = {
        "source": {"publisher": {"commit": COMMIT}},
        "release_tag": BETA_TAG,
        "packages": [
            {"file": name, "sha256": digest(payload)}
            for name, payload in wheels.items()
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    fixed = {
        "extraction.json",
        "foundation.lock",
        "install_foundation.sh",
        "repack-provenance.json",
        "requirements-foundation.txt",
        "requirements-github.txt",
    }
    for name in fixed:
        (directory / name).write_text(name)
    checksummed = list(wheels) + sorted(fixed | {"manifest.json"})
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{digest((directory / name).read_bytes())}  {name}\n"
            for name in checksummed
        )
    )
    archive = "jovian-cu133-foundation.tar.zst"
    (directory / archive).write_bytes(b"archive")
    (directory / f"{archive}.sha256").write_text(f"{digest(b'archive')}  {archive}\n")
    if promotion:
        record = {
            "schema": "local-inference-foundation-promotion/v1",
            "status": "qualified",
            "source_release": BETA_TAG,
            "source_commit": COMMIT,
            "source_manifest_sha256": digest(
                (directory / "manifest.json").read_bytes()
            ),
            "invariant": (
                "Wheel files and wheel SHA-256 digests are unchanged from the "
                "source beta release."
            ),
        }
        (directory / "stable-promotion.json").write_text(json.dumps(record))


def test_accepts_complete_beta_release(tmp_path):
    write_release(tmp_path)

    verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_rejects_changed_wheel(tmp_path):
    write_release(tmp_path)
    (tmp_path / "package-2.whl").write_text("changed")

    with pytest.raises(ValueError, match="package digest mismatch"):
        verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_accepts_complete_stable_promotion(tmp_path):
    write_release(tmp_path, promotion=True)

    verify_release(tmp_path, COMMIT, BETA_TAG, promotion=True)
