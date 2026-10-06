#!/usr/bin/env python3
"""Build a multi-model image with an inspected foundation and explicit profile identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from prepare_runtime_foundation import prepare


def run(argv: list[str], **kwargs) -> bytes:
    return subprocess.check_output(argv, **kwargs)


def runtime_smoke_command(image: str, gpu: str) -> list[str]:
    """Check native imports and disk-table syscalls with bounded locked memory."""
    return [
        "docker",
        "run",
        "--rm",
        "--device",
        f"nvidia.com/gpu={gpu}",
        "--security-opt",
        "seccomp=unconfined",
        "--ulimit",
        # The probe has two io_uring entries. A finite limit avoids requiring
        # CAP_SYS_RESOURCE merely to raise a rootless daemon's inherited limit.
        "memlock=8388608:8388608",
        image,
        "python",
        "/opt/venv/libexec/verify_qwen38_runtime.py",
        "--foundation",
        "ngc",
        "--require-gpu",
        "--require-io-uring",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("image")
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    root = tools.parents[1]
    bundle = args.bundle.resolve()
    subprocess.run(["sha256sum", "--check", "SHA256SUMS"], cwd=bundle, check=True)
    manifest = bundle / "manifest.json"
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    foundation_lock = dict(
        line.split("=", 1)
        for line in (tools / "foundation.lock").read_text().splitlines()
        if "=" in line
    )
    source_image = foundation_lock["source.image"]
    platform = foundation_lock.get("source.image.platform", "linux/amd64")
    source_commit = run(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    if run(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("Runtime image builds require a clean recipe checkout")
    # Inspect the exact immutable base used by FROM, not a similar local tag.
    present = subprocess.run(
        ["docker", "image", "inspect", source_image], capture_output=True, check=False
    )
    if present.returncode:
        subprocess.run(["docker", "pull", source_image], check=True)
    inspection = json.loads(run(["docker", "image", "inspect", source_image]))
    builder = os.environ.get("BUILDX_BUILDER", foundation_lock.get("buildx.builder", "lil-wheel-cu134-sm120"))
    foundation, neutral_inspection = prepare(
        source_image,
        inspection[0],
        root / "runtime/platform-environment.json",
        Path(
            os.environ.get(
                "LIL_FOUNDATION_CACHE", Path.home() / ".cache/lil-foundation"
            )
        ),
        builder,
        platform,
    )
    with tempfile.TemporaryDirectory(prefix="lil-runtime-build-metadata-") as tmp:
        metadata = Path(tmp)
        (metadata / "foundation.inspect.json").write_text(
            json.dumps([neutral_inspection])
        )
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--builder",
                builder,
                "--platform",
                platform,
                "--file",
                str(tools / "Dockerfile.runtime"),
                "--build-context",
                f"qwen-runtime-bundle={bundle}",
                "--build-context",
                f"runtime-build-metadata={metadata}",
                "--build-context",
                f"model-neutral-foundation={foundation}",
                "--build-arg",
                f"SOURCE_IMAGE={source_image}",
                *(
                    ["--build-arg", f"UV_IMAGE={foundation_lock['uv.container-image']}"]
                    if "uv.container-image" in foundation_lock
                    else []
                ),
                "--build-arg",
                f"RUNTIME_SOURCE_COMMIT={source_commit}",
                "--tag",
                args.image,
                "--load",
                str(root),
            ],
            check=True,
        )
        final = json.loads(run(["docker", "image", "inspect", args.image]))
        (metadata / "final.inspect.json").write_text(json.dumps(final))
        # Re-audit final Config.Env; a child layer can reintroduce baked model policy.
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                "2",
                "--memory",
                "4g",
                "--mount",
                f"type=bind,src={metadata},dst=/build-metadata,readonly",
                args.image,
                "python",
                "-m",
                "runtime.packaging",
                "--image-inspect",
                "/build-metadata/final.inspect.json",
                "--runtime-lock-sha256",
                manifest_digest,
            ],
            check=True,
        )
    print(
        json.dumps(
            {
                "image": final[0]["Id"],
                "layers": len(final[0]["RootFS"]["Layers"]),
                "runtime_manifest_sha256": manifest_digest,
            }
        )
    )
    if os.environ.get("RUNTIME_GPU"):
        subprocess.run(
            runtime_smoke_command(args.image, os.environ["RUNTIME_GPU"]),
            check=True,
        )


if __name__ == "__main__":
    main()
