"""Derive serving-image metadata from authenticated build inputs.

Runtime foundation labels may describe a different serving source tree. Clear
those inherited claims while preserving metadata for unchanged dependencies.
The source lock identifies implementation; qualification belongs to the evidence
report for the resulting immutable image, not to a pre-build assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEPENDENCY_PREFIXES = tuple(
    f"local-inference.{name}."
    for name in (
        "cuda",
        "cutlass-dsl",
        "deepgemm",
        "exllamav3",
        "flash-attention",
        "flashinfer",
        "instanttensor",
        "nccl",
        "nccl4py",
        "pytorch",
        "torch",
        "torchvision",
        "rust",
        "transformers",
        "xgrammar",
    )
)


def read_lock(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"Duplicate source-lock key: {key}")
        values[key] = value
    return values


def image_labels(
    lock: dict[str, str], inherited: dict[str, str], source_lock_sha256: str
) -> dict[str, str]:
    """Replace inherited serving claims without changing executable inputs."""
    labels = {
        key: ""
        for key in inherited
        if (
            key.startswith("local-inference.")
            and not key.startswith(DEPENDENCY_PREFIXES)
        )
        or key.startswith("org.opencontainers.image.")
    }
    labels.update(
        {
            "org.opencontainers.image.title": "Jovian Judgement community serving",
            "org.opencontainers.image.description": "GLM-5.3-Flash, Qwen3.8-Flash-Next, DeepSeek-V4-Flash and DeepSeek-V4.1-Flash serving; complete Git sources and model-specific launch profiles",
            "org.opencontainers.image.version": lock["release.version"],
            "org.opencontainers.image.source": "https://github.com/local-inference-lab/blackwell-llm-docker",
            "org.opencontainers.image.revision": lock["vllm.commit"],
            "local-inference.status": "implemented",
            "local-inference.release.name": lock["release.name"],
            "local-inference.release.rootfs-format": "flattened runtime plus source installation",
            "local-inference.release.overlay2-rootfs-layers": lock[
                "runtime.rootfs.layers"
            ],
            "local-inference.runtime.base-image": lock["runtime.base.image"],
            "local-inference.runtime.source-lock.path": "/opt/glm53-flash/source.lock",
            "local-inference.runtime.source-lock.sha256": source_lock_sha256,
            "local-inference.runtime.source-mode": "complete Git bundles",
            "local-inference.runtime.cache-fingerprint": lock[
                "runtime.cache.fingerprint"
            ],
            "local-inference.runtime.cudagraph-mode": lock["runtime.cudagraph.mode"],
            "local-inference.runtime.default.moe-backend": lock.get(
                "runtime.moe-backend.default", "auto"
            ),
            "local-inference.runtime.default.gpu-local.target-page-tokens": "2048",
            "local-inference.runtime.default.recurrent-checkpoint-policy": "auto",
            "local-inference.runtime.lmcache-transfer": lock[
                "runtime.lmcache.transfer"
            ],
            "local-inference.scheduler.max-num-batched-tokens": lock[
                "runtime.scheduler.max-num-batched-tokens"
            ],
            "local-inference.scheduler.prefill-schedule-interval": "1",
            "local-inference.scheduler.prefill-compute-share": lock[
                "runtime.scheduler.prefill-compute-share"
            ],
            "local-inference.model.repository": lock["model.repository"],
            "local-inference.model.update-policy": "resolve requested Hugging Face revision at startup",
            "local-inference.draft.model": lock["draft.repository"],
            "local-inference.draft.quantization": lock["draft.quantization"],
            "local-inference.draft.update-policy": "resolve requested Hugging Face revision at startup",
            "local-inference.glm53.mtp-backends": "attention:B12X,moe:Marlin; private NVFP4 vocabulary head",
            "local-inference.glm53.target-backends": "attention:B12X,moe:B12X,linear:B12X",
            "local-inference.backend.gdn.prefill": "FlashKDA with recurrent checkpoints",
            "local-inference.backend.gdn.decode": "B12X",
            "local-inference.backend.allreduce": "B12X PCIe with NCCL for larger transfers",
            "local-inference.lmcache.version": lock["lmcache.version"],
            "com.nvidia.cuda.version": lock["runtime.cuda.version"],
            "com.nvidia.pytorch.version": lock["runtime.pytorch.version"],
        }
    )
    for name in ("vllm", "b12x", "lmcache"):
        for key, source_key in (
            ("commit", "commit"),
            ("source-revision", "commit"),
            ("tree", "tree"),
            ("package-tree", "package.tree"),
        ):
            labels[f"local-inference.{name}.{key}"] = lock[f"{name}.{source_key}"]
        labels[f"local-inference.{name}.repo"] = lock.get(f"{name}.repository", "")
        labels[f"local-inference.{name}.branch"] = "image-source"
    if "vllm.version" in lock:
        labels["local-inference.vllm.version"] = lock["vllm.version"]
    for key in ("version", "package.version"):
        if f"runtime.liburing.{key}" in lock:
            labels[f"local-inference.liburing.{key}"] = lock[f"runtime.liburing.{key}"]
    for key in ("base.commit", "patch.sha256", "extension.sha256"):
        labels[f"local-inference.flashkda.{key}"] = lock[f"flashkda.{key}"]
    if "flashinfer.commit" in lock:
        labels.update(
            {
                key: ""
                for key in inherited
                if key.startswith("local-inference.flashinfer.")
            }
        )
        for key in (
            "commit",
            "artifact.image",
            "python.wheel.sha256",
            "jit-cache.wheel.sha256",
        ):
            labels[f"local-inference.flashinfer.{key}"] = lock[f"flashinfer.{key}"]
        labels["local-inference.flashinfer.version"] = "0.6.18+cu133"
        labels["local-inference.flashinfer.repo"] = (
            "https://github.com/voipmonitor/flashinfer.git"
        )
    if "vllm.native.extension.sha256" in lock:
        labels["local-inference.vllm.native.extension.sha256"] = lock[
            "vllm.native.extension.sha256"
        ]
        labels["local-inference.ds4.entrypoint"] = "/usr/local/bin/serve-ds4-jovian.sh"
        labels["local-inference.ds41.entrypoint"] = (
            "/usr/local/bin/serve-ds41-jovian.sh"
        )
        labels["local-inference.ds41.engram-table-memory"] = (
            "disk; opt-in mapped pinned RAM"
        )
        labels["local-inference.ds4.target-backends"] = (
            "attention:B12X,moe:B12X-W4A8,linear:DeepGEMM"
        )
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    args = parser.parse_args()
    base = json.load(sys.stdin)[0]
    labels = image_labels(
        read_lock(args.source_lock),
        base["Config"].get("Labels") or {},
        hashlib.sha256(args.source_lock.read_bytes()).hexdigest(),
    )
    arguments = []
    for key, value in sorted(labels.items()):
        if "\0" in key or "\0" in value:
            raise ValueError("Image labels must not contain NUL bytes")
        arguments.append(f"{key}={value}\0".encode())
    sys.stdout.buffer.write(b"".join(arguments))


if __name__ == "__main__":
    main()
