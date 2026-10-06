#!/usr/bin/env python3
"""Build a runtime image from locally built component bundles, without publishing.

This follows publish_container_channel.py up to the image build: assemble the
verified component bundles over the pinned NGC foundation, prepare the LMCache
cuMem auxiliary source, and build the image. It does not resolve a channel,
download release assets, run GPU qualification, or push anything.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from container_channel import digest
from prepare_runtime_auxiliary import prepare as prepare_auxiliary

ROLES = ("nccl", "flashinfer", "b12x", "vllm", "lmcache", "instanttensor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    for role in ROLES:
        parser.add_argument(f"--{role}-bundle", type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output / "runtime"
    command = [
        "python3",
        str(tools / "assemble_qwen38_runtime_bundle.py"),
        "--output",
        str(runtime),
        "--qwen-lock",
        str(tools / "qwen38-runtime.lock"),
        "--ngc-foundation-lock",
        str(tools / "foundation.lock"),
    ]
    for role in ROLES:
        command += [f"--{role}-bundle", str(getattr(args, f"{role}_bundle").resolve())]
    subprocess.run(command, check=True)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = Path(
        os.environ.get("LIL_COMPONENT_ARCHIVE_CACHE", str(args.output / "archive-cache"))
    )
    manifest["auxiliary"] = {
        "lmcache_cumem": prepare_auxiliary(
            manifest, runtime, cache / "lmcache-cumem-source"
        )
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    checksum = runtime / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(runtime)}\n"
            for path in sorted(runtime.rglob("*"))
            if path.is_file() and path != checksum
        )
    )
    subprocess.run(
        ["bash", str(tools / "build_runtime_image.sh"), str(runtime), args.image],
        check=True,
    )


if __name__ == "__main__":
    main()
