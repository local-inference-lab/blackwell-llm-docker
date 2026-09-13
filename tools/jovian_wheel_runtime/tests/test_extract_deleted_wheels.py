from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import PurePosixPath

from tools.jovian_wheel_runtime.extract_deleted_wheels import (
    Artifact,
    scan_image_archive,
)


def tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, payload in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def docker_archive(layers: list[dict[str, bytes]]) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for index, layer in enumerate(layers):
            payload = tar_bytes(layer)
            member = tarfile.TarInfo(f"{index:064x}/layer.tar")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    buffer.seek(0)
    return buffer


def test_recovers_hash_identified_payload_removed_by_later_layer(tmp_path):
    payload = b"source-addressed wheel payload"
    path = "tmp/wheels/runtime.whl"
    artifact = Artifact(
        path=PurePosixPath(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    archive = docker_archive(
        [
            {path: payload},
            {"tmp/wheels/.wh.runtime.whl": b""},
        ]
    )

    found = scan_image_archive(archive, [artifact], tmp_path)

    assert found == {path: artifact.sha256}
    assert (tmp_path / "runtime.whl").read_bytes() == payload


def test_rejects_same_path_with_unexpected_payload(tmp_path):
    path = "tmp/wheels/runtime.whl"
    artifact = Artifact(
        path=PurePosixPath(path),
        sha256=hashlib.sha256(b"expected").hexdigest(),
    )

    found = scan_image_archive(
        docker_archive([{path: b"different"}]), [artifact], tmp_path
    )

    assert found == {}
    assert not (tmp_path / "runtime.whl").exists()


def test_recovers_hash_identified_payload_from_relocated_buildkit_path(tmp_path):
    payload = b"immutable relocated wheel payload"
    declared_path = "tmp/dist/runtime.whl"
    artifact = Artifact(
        path=PurePosixPath(declared_path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    found = scan_image_archive(
        docker_archive([{"build/output/runtime.whl": payload}]),
        [artifact],
        tmp_path,
    )

    assert found == {declared_path: artifact.sha256}
    assert (tmp_path / "runtime.whl").read_bytes() == payload
