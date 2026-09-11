"""Freeze committed serving sources and build inputs for a flat Docker image."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repository_url(value: str) -> str:
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.path.strip("/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(c.isspace() for c in value)
    ):
        raise ValueError("Source repository must be an HTTPS URL without credentials")
    return value.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm", "b12x", "lmcache", "native-artifact", "uv", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vllm", "b12x", "lmcache"):
        parser.add_argument(
            f"--{name}-repository",
            help="Published source repository; defaults to origin",
        )
    parser.add_argument(
        "--release-name", default="jovian-judgement-community-source-locked"
    )
    parser.add_argument("--release-version", default="source-locked")
    parser.add_argument(
        "--lmcache-native-mode",
        choices=("reuse-all", "cpu-rebuild-cuda-reuse"),
        default="reuse-all",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    roots = {
        name: getattr(args, name).resolve() for name in ("vllm", "b12x", "lmcache")
    }
    for name, root in roots.items():
        if git(root, "status", "--porcelain"):
            raise ValueError(f"{name} must have a clean committed source tree")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = {
        "format": "local-inference-source-lock/v1",
        "implementation.status": "implemented",
        "qualification.contract": "See evidence for the resulting immutable image",
        "release.name": args.release_name,
        "release.version": args.release_version,
        "runtime.base.image": "voipmonitor/vllm@sha256:93ac5228f1cbde2182ca294d8b479259144742af2756a49ff207dd245429bf43",
        "runtime.cuda.version": "13.3",
        "runtime.pytorch.version": "2.13.0",
        "runtime.liburing.version": "2.5",
        "runtime.liburing.package.version": "2.5-1build1",
        "runtime.rootfs.layers": "2",
        "runtime.cudagraph.mode": "FULL_AND_PIECEWISE",
        "runtime.moe-backend.default": "b12x",
        "runtime.scheduler.max-num-batched-tokens": "4096",
        "runtime.scheduler.prefill-compute-share": "0.4",
        "runtime.nccl.channels": "16",
        "runtime.nccl.buffer.bytes": "2097152",
        "runtime.recurrent-checkpoint-policy": "auto",
        "lmcache.native.artifact.image": "localinferencelab/vllm@sha256:3ae04d964f8e7e936ef4b34dd75406b9169fead8062fb8b4b1634f3bbf512827",
        "lmcache.native.mode": args.lmcache_native_mode,
        "runtime.lmcache.transfer": "engine-driven asynchronous shared memory",
        "runtime.lmcache.checkpoints": "atomic target and draft bundles for request_boundaries; independent chunks for explicit aligned",
        "model.repository": "local-inference-lab/GLM-5.3-Flash-NVFP4",
        "draft.repository": "local-inference-lab/GLM-5.3-Flash-DFlash2",
        "draft.quantization": "MXFP8",
    }
    for name, root in roots.items():
        revision = git(root, "rev-parse", "HEAD")
        lock[f"{name}.repository"] = repository_url(
            getattr(args, f"{name}_repository")
            or git(root, "remote", "get-url", "origin")
        )
        lock[f"{name}.commit"] = revision
        lock[f"{name}.tree"] = git(root, "rev-parse", "HEAD^{tree}")
        lock[f"{name}.package.tree"] = git(root, "rev-parse", f"HEAD:{name}")
        bundle = args.output / f"{name}.bundle"
        subprocess.run(
            ["git", "-C", str(root), "bundle", "create", str(bundle), "HEAD"],
            check=True,
        )
        lock[f"{name}.bundle.sha256"] = digest(bundle)
    lock["lmcache.version"] = (
        f"0.5.5.dev0+glm53checkpoints.{lock['lmcache.commit'][:8]}"
    )
    lock["vllm.version"] = (
        f"0.26.1rc0+glm53.{args.release_version}.vllm{lock['vllm.commit'][:8]}"
    )
    patch = "cmake/external_projects/patches/flashkda-packed-checkpoints.patch"
    lock["flashkda.base.commit"] = (
        (args.native_artifact / "flashkda-base.commit").read_text().strip()
    )
    lock["flashkda.patch.sha256"] = digest(roots["vllm"] / patch)
    recorded_patch = (
        (args.native_artifact / "flashkda-patch.sha256").read_text().split()[0]
    )
    if recorded_patch != lock["flashkda.patch.sha256"]:
        raise ValueError(
            "Native artifact patch differs from the committed vLLM dependency"
        )
    lock["flashkda.extension.sha256"] = digest(
        args.native_artifact / "_flashkda_C.abi3.so"
    )
    native_identity = dict(
        line.split("=", 1)
        for line in (args.native_artifact / "native-source.identity")
        .read_text()
        .splitlines()
    )
    for field, path in (
        ("csrc.tree", "csrc"),
        ("cmake.tree", "cmake"),
        ("cmakelists.blob", "CMakeLists.txt"),
    ):
        expected = git(roots["vllm"], "rev-parse", f"HEAD:{path}")
        if native_identity[f"vllm.{field}"] != expected:
            raise ValueError(f"Stable native artifact has mismatched {path} sources")
        lock[f"vllm.native.{field}"] = expected
    lock["vllm.native.extension.sha256"] = digest(
        args.native_artifact / "_C_stable_libtorch.abi3.so"
    )
    lock["vllm.native.identity.sha256"] = digest(
        args.native_artifact / "native-source.identity"
    )
    lock["flashinfer.commit"] = "803c4664f4771ddc418f20a57f752469a237a825"
    lock["flashinfer.artifact.image"] = (
        "voipmonitor/vllm@sha256:79edbc91874d9468e3e6268e1584503e3dec55f2a4d3bdd70d5c43e9b41675c7"
    )
    lock["flashinfer.python.wheel.sha256"] = (
        "3a5b1143d8b933d71881c2d71a2c0677360528344813be1893bafa9f335affea"
    )
    lock["flashinfer.jit-cache.wheel.sha256"] = (
        "9d08679e4ba3cc7c49c5f8035268b7958e9c4d8c7a5e8cd842211da46007809f"
    )
    docker = Path(__file__).resolve().parent
    lmcache_launcher = docker.parents[1] / "launchers/lmcache-mp-wrapper.sh"
    shutil.copy2(lmcache_launcher, args.output / "lmcache-mp-wrapper.sh")
    lock["launcher.lmcache.sha256"] = digest(lmcache_launcher)
    inputs = (
        "Dockerfile.glm53-cache-contracts",
        "build_glm53_cache_contract_image.sh",
        "source_locked_image_labels.py",
        "install_source_bundle.sh",
        "pack_lmcache_native_reuse.py",
        "install_glm53_source_locked.sh",
        "install_vllm_source_version.py",
        "serve-ds4-jovian.sh",
        "Dockerfile.jovian-stable-native",
        "build_jovian_stable_native.py",
        "serve-glm53-flash-nvfp4-dflash2.sh",
        "serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh",
        "serve-glm53-flash-lmcache.sh",
        "serve-glm53-flash-lmcache-cache-complete.sh",
        "serve-glm53-flash-cache-complete.sh",
        "glm53_checkpoint_identity.py",
        "glm53-r18-lmcache-runtime-requirements.txt",
    )
    for filename in inputs:
        lock[f"input.{filename}.sha256"] = digest(docker / filename)
    shutil.copy2(args.uv, args.output / "uv")
    lock["build.uv.sha256"] = digest(args.output / "uv")
    lock["runtime.cache.fingerprint"] = (
        f"cu133-torch213-glm53-vllm{lock['vllm.package.tree'][:8]}"
        f"-b12x{lock['b12x.package.tree'][:8]}-lmcache{lock['lmcache.package.tree'][:8]}"
        f"-flashkda{lock['flashkda.extension.sha256'][:8]}"
        f"-native{lock['vllm.native.extension.sha256'][:8]}"
        f"-fi{lock['flashinfer.commit'][:8]}"
    )
    target = args.output / "source.lock"
    target.write_text("".join(f"{key}={value}\n" for key, value in lock.items()))
    print(f"SOURCE_LOCK_SHA256={digest(target)}")
    print(f"CACHE_FINGERPRINT={lock['runtime.cache.fingerprint']}")


if __name__ == "__main__":
    main()
