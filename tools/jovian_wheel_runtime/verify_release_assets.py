#!/usr/bin/env python3
"""Verify the complete immutable asset contract of a foundation release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIXED_CHECKSUMMED_ASSETS = {
    "extraction.json",
    "foundation.lock",
    "install_foundation.sh",
    "manifest.json",
    "repack-provenance.json",
    "requirements-foundation.txt",
    "requirements-github.txt",
}


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_entries(path: Path) -> dict[str, str]:
    """Read sha256sum output and normalize every entry to its basename."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, declared_path = line.split(maxsplit=1)
        name = Path(declared_path.lstrip("* ")).name
        if name in entries:
            raise ValueError(f"duplicate checksum basename: {name}")
        entries[name] = digest
    return entries


def verify_release(
    directory: Path,
    source_commit: str,
    beta_tag: str,
    promotion: bool,
) -> None:
    """Verify identity, membership, and hashes for one foundation release."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"]["publisher"]["commit"] != source_commit:
        raise ValueError(
            "manifest publisher commit does not match the requested commit"
        )
    if manifest["release_tag"] != beta_tag:
        raise ValueError("manifest release tag does not match the requested beta tag")

    package_files = {package["file"] for package in manifest["packages"]}
    if len(package_files) != 6:
        raise ValueError("foundation manifest must declare exactly six package files")
    archive = "jovian-cu133-foundation.tar.zst"
    expected = (
        FIXED_CHECKSUMMED_ASSETS
        | package_files
        | {"SHA256SUMS", archive, f"{archive}.sha256"}
    )
    if promotion:
        expected.add("stable-promotion.json")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"release asset set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    package_hashes = {
        package["file"]: package["sha256"] for package in manifest["packages"]
    }
    for name, expected_digest in package_hashes.items():
        if sha256(directory / name) != expected_digest:
            raise ValueError(f"package digest mismatch: {name}")

    checksums = checksum_entries(directory / "SHA256SUMS")
    checksummed_assets = package_files | FIXED_CHECKSUMMED_ASSETS
    if checksums.keys() != checksummed_assets:
        raise ValueError("SHA256SUMS membership does not match the foundation contract")
    for name, expected_digest in checksums.items():
        if sha256(directory / name) != expected_digest:
            raise ValueError(f"foundation digest mismatch: {name}")

    archive_checksums = checksum_entries(directory / f"{archive}.sha256")
    if archive_checksums.keys() != {archive}:
        raise ValueError(
            "archive checksum must identify exactly the foundation archive"
        )
    if sha256(directory / archive) != archive_checksums[archive]:
        raise ValueError("foundation archive digest mismatch")

    if promotion:
        promotion_record = json.loads(
            (directory / "stable-promotion.json").read_text(encoding="utf-8")
        )
        expected_promotion = {
            "schema": "local-inference-foundation-promotion/v1",
            "status": "qualified",
            "source_release": beta_tag,
            "source_commit": source_commit,
            "source_manifest_sha256": sha256(manifest_path),
            "invariant": (
                "Wheel files and wheel SHA-256 digests are unchanged from the "
                "source beta release."
            ),
        }
        if promotion_record != expected_promotion:
            raise ValueError("stable promotion record does not match the beta assets")


def main() -> None:
    """Parse arguments and enforce the foundation release contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    args = parser.parse_args()
    verify_release(args.directory, args.source_commit, args.beta_tag, args.promotion)


if __name__ == "__main__":
    main()
