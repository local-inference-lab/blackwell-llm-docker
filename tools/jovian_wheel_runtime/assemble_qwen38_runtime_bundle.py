#!/usr/bin/env python3
"""Assemble one verified Qwen3.8 serving runtime from component bundles."""

from __future__ import annotations

import argparse
import configparser
import email.parser
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_SCHEMAS = {
    "foundation": "local-inference-jovian-foundation-bundle/v1",
    "nccl": "local-inference-nccl-cu134-release/v1",
    "flashinfer": "local-inference-flashinfer-wheel-release/v1",
    "b12x": "local-inference-b12x-wheel-release/v1",
    "vllm": "local-inference-vllm-wheel-release/v2",
    "lmcache": "local-inference-lmcache-wheel-release/v1",
    "instanttensor": "local-inference-instanttensor-wheel-release/v1",
}

NGC_FOUNDATION_PACKAGES = {
    "torch": "pytorch.version",
    "torchvision": "torchvision.version",
    "triton": "triton.version",
    "triton-kernels": "triton-kernels.version",
    "flash-attn": "flash-attn.version",
}
FOUNDATION_COMPONENT = "foundation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(bundle: Path) -> None:
    checksum_file = bundle / "SHA256SUMS"
    if not checksum_file.is_file():
        raise ValueError(f"component bundle has no SHA256SUMS: {bundle}")
    declared: set[str] = set()
    for line in checksum_file.read_text().splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator:
            expected, separator, relative = line.partition(" *")
        path = PurePosixPath(relative)
        if (
            not separator
            or len(expected) != 64
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError(f"invalid SHA256SUMS entry in {bundle}: {line!r}")
        candidate = bundle.joinpath(*path.parts)
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"checksum mismatch: {candidate}")
        if relative in declared:
            raise ValueError(f"duplicate checksum entry: {relative}")
        declared.add(relative)
    consumed = {"manifest.json"} | {
        str(path.relative_to(bundle)) for path in (bundle / "wheels").glob("*.whl")
    }
    if not consumed <= declared:
        raise ValueError(f"unchecked component assets: {sorted(consumed - declared)}")


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(matches) != 1:
            raise ValueError(f"wheel must contain one METADATA file: {path}")
        message = email.parser.BytesParser().parsebytes(archive.read(matches[0]))
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError(f"wheel has incomplete package metadata: {path}")
    return name, version


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def require_digest_image(image: str) -> None:
    if re.fullmatch(r"[^\s@]+/[^\s@]+@sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("foundation image must include an immutable SHA-256 digest")


def register_package(name: str, seen: set[str]) -> None:
    key = normalized_name(name)
    if key in seen:
        raise ValueError(f"duplicate runtime package: {name}")
    seen.add(key)


def register_wheel_payload(
    wheel: Path,
    role: str,
    files: dict[str, str],
    entry_points: dict[tuple[str, str], str],
) -> dict[str, str]:
    """Reject competing package owners and retain B12X's exact runtime bytes."""
    b12x_hashes: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            parts = PurePosixPath(name).parts
            if not parts or name.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe wheel member: {wheel.name}:{name}")
            installed = name
            if parts[0].endswith(".data") and len(parts) >= 3:
                scheme, *payload = parts[1:]
                installed = "/".join(payload)
                if scheme not in {"purelib", "platlib"}:
                    installed = f"@{scheme}/{installed}"
            if installed in files:
                raise ValueError(
                    f"wheel file ownership collision: {installed} in "
                    f"{files[installed]} and {wheel.name}"
                )
            files[installed] = wheel.name
            if installed.startswith("flashinfer/b12x/"):
                raise ValueError("runtime requires standalone B12X; FlashInfer embeds B12X")
            if installed.startswith("b12x/"):
                if role != "b12x":
                    raise ValueError(f"{role} wheel claims the B12X import namespace")
                b12x_hashes[installed] = hashlib.sha256(archive.read(name)).hexdigest()
            if name.endswith(".dist-info/entry_points.txt"):
                parser = configparser.ConfigParser(interpolation=None)
                parser.optionxform = str
                parser.read_string(archive.read(name).decode())
                for group in parser.sections():
                    for command, target in parser.items(group):
                        key = (group, command)
                        if key in entry_points:
                            raise ValueError(f"wheel entry point ownership collision: {key}")
                        if target.startswith(("b12x.", "flashinfer.b12x.")) and role != "b12x":
                            raise ValueError(f"{role} wheel claims a B12X entry point: {key}")
                        entry_points[key] = wheel.name
    return b12x_hashes


def runtime_value(manifest: dict[str, object], key: str) -> str | None:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        return None
    value = runtime.get(key)
    return value if isinstance(value, str) else None


def read_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"invalid foundation lock entry: {line!r}")
        if key in values:
            raise ValueError(f"duplicate foundation lock key: {key}")
        values[key] = value
    return values


def ngc_foundation_manifest(path: Path) -> dict[str, object]:
    values = read_lock(path)
    required = {
        "source.image",
        "python.version",
        "cuda.version",
        *NGC_FOUNDATION_PACKAGES.values(),
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"foundation lock is missing keys: {', '.join(missing)}")
    require_digest_image(values["source.image"])
    return {
        "schema": EXPECTED_SCHEMAS["foundation"],
        "status": "implemented",
        "source": {"image": values["source.image"]},
        "runtime": {
            "python": values["python.version"],
            "cuda": values["cuda.version"],
        },
        "packages": [
            {"name": name, "version": values[key]}
            for name, key in NGC_FOUNDATION_PACKAGES.items()
        ],
    }


def application_requirements(packages: list[dict[str, str]]) -> str:
    return "".join(
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n"
        for item in packages
        if item["component"] != FOUNDATION_COMPONENT
    )


def validate_compatibility(manifests: dict[str, dict[str, object]]) -> None:
    foundation = manifests["foundation"]
    foundation_runtime = foundation.get("runtime")
    if not isinstance(foundation_runtime, dict):
        raise ValueError("foundation manifest has no runtime object")
    if foundation_runtime.get("python") != "3.12" or foundation_runtime.get("cuda") != "13.4.1":
        raise ValueError("foundation runtime must use Python 3.12 and CUDA 13.4.1")
    packages = foundation.get("packages")
    if not isinstance(packages, list):
        raise ValueError("foundation manifest has no package list")
    torch_versions = {
        package.get("version")
        for package in packages
        if isinstance(package, dict) and normalized_name(str(package.get("name"))) == "torch"
    }
    if len(torch_versions) != 1:
        raise ValueError("foundation manifest must identify exactly one PyTorch version")
    torch_version = next(iter(torch_versions))
    source = foundation.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("image"), str):
        raise ValueError("foundation manifest must identify its immutable source image")
    builder_image = source["image"]
    require_digest_image(builder_image)

    for role, manifest in manifests.items():
        if role in {"foundation", "nccl"}:
            continue
        python = runtime_value(manifest, "python")
        cuda = runtime_value(manifest, "cuda")
        pytorch = runtime_value(manifest, "pytorch")
        component_builder = runtime_value(manifest, "builder_image")
        if python != "3.12" or not cuda or not cuda.startswith("13.4"):
            raise ValueError(f"{role} does not declare the Python 3.12/CUDA 13.4 ABI")
        if pytorch != torch_version:
            raise ValueError(f"{role} was built against a different PyTorch distribution")
        if component_builder != builder_image:
            raise ValueError(f"{role} was built from a different foundation image")
    nccl_cuda = runtime_value(manifests["nccl"], "cuda")
    if not nccl_cuda or not nccl_cuda.startswith("13.4"):
        raise ValueError("NCCL component does not declare the CUDA 13.4 ABI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qwen-lock", required=True, type=Path)
    foundation = parser.add_mutually_exclusive_group(required=True)
    foundation.add_argument("--foundation-bundle", type=Path)
    foundation.add_argument("--ngc-foundation-lock", type=Path)
    for role in EXPECTED_SCHEMAS:
        if role == "foundation":
            continue
        parser.add_argument(f"--{role}-bundle", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    qwen_lock = args.qwen_lock.resolve()
    if not qwen_lock.is_file():
        raise ValueError(f"Qwen dependency lock does not exist: {qwen_lock}")

    bundles = {
        role: getattr(args, f"{role}_bundle").resolve()
        for role in EXPECTED_SCHEMAS
        if role != "foundation"
    }
    manifests: dict[str, dict[str, object]] = {}
    foundation_lock: Path | None = None
    if args.foundation_bundle is not None:
        foundation_bundle = args.foundation_bundle.resolve()
        verify_checksums(foundation_bundle)
        foundation_manifest = json.loads(
            (foundation_bundle / "manifest.json").read_text()
        )
        bundles["foundation"] = foundation_bundle
    else:
        foundation_lock = args.ngc_foundation_lock.resolve()
        if not foundation_lock.is_file():
            raise ValueError(f"NGC foundation lock does not exist: {foundation_lock}")
        foundation_manifest = ngc_foundation_manifest(foundation_lock)
    if foundation_manifest.get("schema") != EXPECTED_SCHEMAS["foundation"]:
        raise ValueError(
            "unexpected foundation manifest schema: "
            f"{foundation_manifest.get('schema')}"
        )
    manifests["foundation"] = foundation_manifest
    for role, bundle in bundles.items():
        if role == "foundation":
            continue
        verify_checksums(bundle)
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("schema") != EXPECTED_SCHEMAS[role]:
            raise ValueError(f"unexpected {role} manifest schema: {manifest.get('schema')}")
        manifests[role] = manifest
    validate_compatibility(manifests)

    wheel_dir = output / "wheels"
    wheel_dir.mkdir(parents=True)
    packages: list[dict[str, str]] = []
    if foundation_lock is not None:
        foundation_packages = foundation_manifest["packages"]
        assert isinstance(foundation_packages, list)
        packages.extend(
            {
                "component": "foundation",
                "name": str(package["name"]),
                "version": str(package["version"]),
            }
            for package in foundation_packages
            if isinstance(package, dict)
        )
    seen_packages = {normalized_name(package["name"]) for package in packages}
    seen_files: set[str] = set()
    installed_files: dict[str, str] = {}
    entry_point_owners: dict[tuple[str, str], str] = {}
    b12x_files: dict[str, str] = {}
    for role, bundle in bundles.items():
        wheels = sorted((bundle / "wheels").glob("*.whl"))
        if not wheels:
            raise ValueError(f"component bundle contains no wheels: {bundle}")
        for wheel in wheels:
            b12x_files.update(
                register_wheel_payload(wheel, role, installed_files, entry_point_owners)
            )
            name, version = wheel_metadata(wheel)
            register_package(name, seen_packages)
            if wheel.name in seen_files:
                raise ValueError(f"duplicate wheel filename: {wheel}")
            seen_files.add(wheel.name)
            destination = wheel_dir / wheel.name
            shutil.copyfile(wheel, destination)
            packages.append(
                {
                    "component": role,
                    "name": name,
                    "version": version,
                    "file": wheel.name,
                    "sha256": sha256(destination),
                }
            )

    if "b12x/__init__.py" not in b12x_files:
        raise ValueError("B12X component does not provide its import package")
    packages.sort(key=lambda item: normalized_name(item["name"]))
    requirements = output / "requirements-local.txt"
    requirements.write_text(application_requirements(packages))
    foundation_bundle = bundles.get("foundation")
    if foundation_bundle is not None:
        shutil.copyfile(
            foundation_bundle / "foundation-runtime.lock",
            output / "foundation-runtime.lock",
        )
        shutil.copyfile(
            foundation_bundle / "foundation.lock", output / "foundation.lock"
        )
    else:
        assert foundation_lock is not None
        shutil.copyfile(foundation_lock, output / "foundation.lock")
    shutil.copyfile(qwen_lock, output / "qwen38-runtime.lock")
    tool_dir = Path(__file__).resolve().parent
    for name in (
        "install_qwen38_runtime.sh",
        "qwen38_runtime_entrypoint.sh",
        "verify_qwen38_runtime.py",
    ):
        shutil.copyfile(tool_dir / name, output / name)
        (output / name).chmod(0o755)

    component_records: dict[str, object] = {}
    for role, manifest in manifests.items():
        if role == "foundation" and foundation_lock is not None:
            component_records[role] = {
                "contract_sha256": sha256(foundation_lock),
                "source": manifest.get("source"),
            }
        else:
            component_records[role] = {
                "manifest_sha256": sha256(bundles[role] / "manifest.json"),
                "source": manifest.get("source"),
            }
    complete_manifest = {
        "schema": "local-inference-qwen38-cu134-runtime/v1",
        "status": "research-only",
        "purpose": "Source-locked vLLM serving on SM120 GPUs",
        "runtime": {"python": "3.12", "cuda": "13.4.1"},
        "components": component_records,
        "packages": packages,
        "b12x_package": {
            "source": manifests["b12x"]["source"],
            "files": b12x_files,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n"
    )
    checksum_paths = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "SHA256SUMS").open("w") as stream:
        for path in checksum_paths:
            stream.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"runtime bundle assembly failed: {error}", file=sys.stderr)
        sys.exit(1)
