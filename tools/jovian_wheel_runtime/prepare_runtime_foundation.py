"""Separate profile-owned ENV from a foundation without changing its layers.

BuildKit exports the pinned foundation into a cached OCI layout. Only the image
configuration and its parent descriptors change. The resulting layout is a named
build context, so docker-container builders need no local registry or daemon
image-store access. The same builder and filesystem cache serve every channel.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"


def read_blob(layout: Path, descriptor: dict) -> dict:
    digest = descriptor["digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("OCI descriptors must use complete SHA256 digests")
    data = (layout / "blobs" / "sha256" / digest[7:]).read_bytes()
    if (
        hashlib.sha256(data).hexdigest() != digest[7:]
        or len(data) != descriptor["size"]
    ):
        raise ValueError("OCI descriptor content does not match its digest/size")
    return json.loads(data)


def write_blob(layout: Path, value: dict, descriptor: dict) -> dict:
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    (layout / "blobs" / "sha256" / digest).write_bytes(data)
    return {**descriptor, "digest": f"sha256:{digest}", "size": len(data)}


def image_manifest(layout: Path) -> tuple[dict, dict]:
    index = json.loads((layout / "index.json").read_text())
    for _ in range(4):
        if len(index.get("manifests", [])) != 1:
            raise ValueError("Foundation layout must contain exactly one image")
        descriptor = index["manifests"][0]
        value = read_blob(layout, descriptor)
        if descriptor["mediaType"] == MANIFEST:
            return descriptor, value
        if descriptor["mediaType"] != INDEX:
            raise ValueError("Unsupported foundation OCI media type")
        index = value
    raise ValueError("Foundation OCI indexes exceed the supported nesting depth")


def neutralize(layout: Path, metadata: dict, defaults: dict[str, str]) -> dict:
    descriptor, manifest = image_manifest(layout)
    config = read_blob(layout, manifest["config"])
    # Docker's classic store identifies images by config digest; its containerd
    # store can report the manifest digest. Both refer to these verified blobs.
    if metadata["Id"] not in {manifest["config"]["digest"], descriptor["digest"]}:
        raise ValueError("OCI foundation is not the inspected source image")
    if config["rootfs"]["diff_ids"] != metadata["RootFS"]["Layers"]:
        raise ValueError("OCI foundation filesystem differs from the source image")
    original_env = config["config"].get("Env", [])
    for name, expected in defaults.items():
        if [item for item in original_env if item.partition("=")[0] == name] != [
            f"{name}={expected}"
        ]:
            raise ValueError(f"Foundation default differs from policy: {name}")
    config["config"]["Env"] = [
        item for item in original_env if item.partition("=")[0] not in defaults
    ]
    manifest["config"] = write_blob(layout, config, manifest["config"])
    descriptor = write_blob(layout, manifest, descriptor)
    (layout / "index.json").write_text(
        json.dumps({"schemaVersion": 2, "manifests": [descriptor]})
    )
    result = copy.deepcopy(metadata)
    result["Id"] = manifest["config"]["digest"]
    result["Config"] = config["config"]
    # A derived config has no original repository tag/digest identity.
    result["RepoTags"], result["RepoDigests"] = [], []
    return {"inspection": result, "manifest_digest": descriptor["digest"]}


def validate_layout(layout: Path, receipt: dict, original: dict) -> None:
    descriptor, manifest = image_manifest(layout)
    config = read_blob(layout, manifest["config"])
    if (
        descriptor["digest"] != receipt["manifest_digest"]
        or manifest["config"]["digest"] != receipt["inspection"]["Id"]
        or config["config"] != receipt["inspection"]["Config"]
        or config["rootfs"]["diff_ids"] != original["RootFS"]["Layers"]
    ):
        raise ValueError("Cached model-neutral foundation identity mismatch")
    for layer in manifest["layers"]:
        digest = layer["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Unsupported foundation layer digest")
        if (layout / "blobs" / "sha256" / digest[7:]).stat().st_size != layer["size"]:
            raise ValueError("Cached foundation layer is incomplete")


def prepare(
    source_image: str,
    metadata: dict,
    policy_path: Path,
    cache: Path,
    builder: str,
    platform: str = "linux/amd64",
) -> tuple[str, dict]:
    policy = json.loads(policy_path.read_text())
    if (
        policy.get("schema_version") != 1
        or policy.get("source_image") != source_image
        or not isinstance(policy.get("environment"), dict)
        or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(value, str)
            or "\x00" in value
            for key, value in policy["environment"].items()
        )
    ):
        raise ValueError("Platform defaults must identify the pinned foundation")
    key = hashlib.sha256(
        json.dumps([metadata["Id"], policy], sort_keys=True).encode()
    ).hexdigest()
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix=f"{key}-", dir=cache) as tmp:
                stage = Path(tmp)
                layout = stage / "layout"
                context = stage / "context"
                context.mkdir()
                subprocess.run(
                    [
                        "docker",
                        "buildx",
                        "build",
                        "--builder",
                        builder,
                        "--platform",
                        platform,
                        "--provenance=false",
                        "--file",
                        "-",
                        "--output",
                        f"type=oci,dest={layout},tar=false",
                        str(context),
                    ],
                    input=f"FROM {source_image}\n".encode(),
                    check=True,
                )
                receipt = neutralize(layout, metadata, policy["environment"])
                receipt["source_image_id"] = metadata["Id"]
                receipt["platform_policy"] = policy
                validate_layout(layout, receipt, metadata)
                (stage / "receipt.json").write_text(json.dumps(receipt, indent=2))
                os.rename(stage, destination)
        receipt = json.loads((destination / "receipt.json").read_text())
        if (
            receipt["source_image_id"] != metadata["Id"]
            or receipt["platform_policy"] != policy
        ):
            raise ValueError("Foundation cache key does not match its receipt")
        layout = destination / "layout"
        validate_layout(layout, receipt, metadata)
        return f"oci-layout://{layout}@{receipt['manifest_digest']}", receipt[
            "inspection"
        ]
