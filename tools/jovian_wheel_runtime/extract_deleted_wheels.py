#!/usr/bin/env python3
"""Recover hash-identified wheel payloads from a Docker image archive.

Docker images may install a wheel in one layer and remove the wheel file in a
later layer. The installed package remains usable, but a normal container
cannot access the original wheel. This tool scans every layer from ``docker
image save`` and emits only payloads whose declared basename and SHA-256
digest match the artifact contract. The declared path remains useful as
provenance, but the basename fallback also handles BuildKit changing a COPY
source directory without changing the immutable wheel payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


@dataclass(frozen=True)
class Artifact:
    path: PurePosixPath
    sha256: str

    @property
    def output_name(self) -> str:
        return self.path.name


def parse_artifact(value: str) -> Artifact:
    try:
        path_text, digest = value.rsplit("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "artifact must have the form IMAGE_PATH=SHA256"
        ) from error
    path = PurePosixPath(path_text.lstrip("./"))
    if not path.name or ".." in path.parts:
        raise argparse.ArgumentTypeError(f"invalid image path: {path_text!r}")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise argparse.ArgumentTypeError(f"invalid SHA-256 digest: {digest!r}")
    return Artifact(path=path, sha256=digest)


def normalized_member_path(name: str) -> PurePosixPath:
    return PurePosixPath(name.lstrip("./"))


def copy_matching_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    artifact: Artifact,
    output_dir: Path,
) -> bool:
    source = archive.extractfile(member)
    if source is None:
        return False
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        digest = hashlib.sha256()
        while chunk := source.read(8 * 1024 * 1024):
            temporary.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        temporary_path.unlink()
        return False
    destination = output_dir / artifact.output_name
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != artifact.sha256:
            raise RuntimeError(f"conflicting output already exists: {destination}")
        temporary_path.unlink()
    else:
        temporary_path.replace(destination)
    return True


def scan_image_archive(
    archive_file: BinaryIO, artifacts: list[Artifact], output_dir: Path
) -> dict[str, str]:
    wanted_by_path = {artifact.path: artifact for artifact in artifacts}
    wanted_by_name = {artifact.output_name: artifact for artifact in artifacts}
    found: dict[str, str] = {}
    with tarfile.open(fileobj=archive_file, mode="r|*") as image_archive:
        for image_member in image_archive:
            if not image_member.isfile():
                continue
            layer_stream = image_archive.extractfile(image_member)
            if layer_stream is None:
                continue
            try:
                with tarfile.open(fileobj=layer_stream, mode="r|*") as layer_archive:
                    for layer_member in layer_archive:
                        if not layer_member.isfile():
                            continue
                        member_path = normalized_member_path(layer_member.name)
                        artifact = wanted_by_path.get(member_path)
                        if artifact is None:
                            artifact = wanted_by_name.get(member_path.name)
                        if artifact is None:
                            continue
                        if copy_matching_member(
                            layer_archive, layer_member, artifact, output_dir
                        ):
                            found[str(artifact.path)] = artifact.sha256
            except tarfile.ReadError:
                # Docker's OCI archive also contains JSON configuration and
                # manifest blobs alongside its tar-formatted filesystem layers.
                continue
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        type=parse_artifact,
        required=True,
        help="image path and required digest as IMAGE_PATH=SHA256",
    )
    args = parser.parse_args()

    outputs = [artifact.output_name for artifact in args.artifact]
    if len(outputs) != len(set(outputs)):
        parser.error("artifact basenames must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    found = scan_image_archive(sys.stdin.buffer, args.artifact, args.output_dir)
    missing = [
        str(artifact.path)
        for artifact in args.artifact
        if str(artifact.path) not in found
    ]
    if missing:
        print(
            "wheel payloads were not found with their required hashes: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    json.dump(
        {"schema": "local-inference-oci-wheel-extraction/v1", "artifacts": found},
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
