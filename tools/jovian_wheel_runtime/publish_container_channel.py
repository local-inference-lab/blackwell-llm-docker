#!/usr/bin/env python3
"""Build and publish a source-addressed runtime after native GPU checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

from build_runtime_image import runtime_smoke_command
from container_channel import api, digest, download, run
from prepare_runtime_auxiliary import prepare as prepare_auxiliary
from release_changelog import collect_release_changelog, render_release_notes


def execute(args: list[str], **kwargs: object) -> None:
    subprocess.run(args, check=True, **kwargs)


def alias_is_current(assembly: dict, repository: str, ref: str) -> bool:
    """An obsolete recipe or component revision must not replace a channel alias."""
    if ref != "refs/heads/main":
        return False
    recipe = api(f"repos/{repository}/commits/main")
    if recipe["sha"] != assembly["recipe_commit"]:
        return False
    for component in assembly["components"].values():
        head = api(
            f"repos/{component['repository']}/commits/"
            f"{quote(component['branch'], safe='')}"
        )
        if component["observed_branch_commit"] != head["sha"]:
            return False
    return True


def require_idle_gpu(gpu: str) -> None:
    """Reject active or unverifiable devices, not process-free VRAM accounting."""
    if not gpu.startswith("GPU-"):
        raise ValueError("qualification GPU must be an explicit GPU UUID")
    devices = ET.fromstring(
        run(["nvidia-smi", "-i", gpu, "--query", "--xml-format"])
    ).findall("gpu")
    if len(devices) != 1 or devices[0].findtext("uuid") != gpu:
        raise RuntimeError(f"cannot verify qualification GPU identity: {gpu}")
    device = devices[0]
    utilization = device.findtext("utilization/gpu_util", "").split()
    processes = device.find("processes")
    if (
        len(utilization) != 2
        or utilization[1] != "%"
        or not utilization[0].isdigit()
        or not 0 <= int(utilization[0]) <= 100
        or processes is None
        or (processes.text or "").strip() not in ("", "None")
    ):
        raise RuntimeError(f"cannot verify qualification GPU activity: {gpu}")
    # Resident compute or graphics processes can be idle between requests.
    # VRAM usage alone is not an occupancy test: a process-free GPU can use MiB.
    if int(utilization[0]) != 0 or len(processes) != 0:
        raise RuntimeError(f"qualification GPU is busy; no workload was stopped: {gpu}")


def verify_cache_test_report(path: Path) -> dict[str, int]:
    """Require executed coverage for each declared cache contract test module."""
    required = {
        "test_checkpoint_identity",
        "test_checkpoint_index",
        "test_checkpoint_storage",
        "test_fs_native_connector",
        "test_vllm_semantic_checkpoint_transfer",
    }
    cases = list(ET.parse(path).getroot().iter("testcase"))
    executed = set()
    skipped = 0
    for case in cases:
        if case.find("error") is not None or case.find("failure") is not None:
            raise ValueError("cache contract tests contain errors or failures")
        if case.find("skipped") is not None:
            skipped += 1
            continue
        executed.update(
            part for part in case.get("classname", "").split(".") if part in required
        )
    if required - executed:
        raise ValueError(
            f"cache contract modules were not executed: {sorted(required - executed)}"
        )
    return {"passed": len(cases) - skipped, "skipped": skipped}


def stage_assembly_lock(source: Path, directory: Path) -> Path:
    """Give every published assembly the filename used by completion checks."""
    destination = directory / "community-assembly.json"
    destination.write_bytes(source.read_bytes())
    return destination


def publish_release(
    repository: str, assembly: dict, commit: str, notes: Path, assets: list[Path]
) -> None:
    """Repair interrupted uploads before making the release publicly complete."""
    tag = assembly["release_tag"]
    listing = (
        run(
            [
                "gh",
                "api",
                "--paginate",
                f"repos/{repository}/releases?per_page=100",
                "--jq",
                ".[].tag_name",
            ]
        )
        .decode()
        .splitlines()
    )
    if tag not in listing:
        execute(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--target",
                commit,
                "--draft",
                "--prerelease",
                "--title",
                assembly["image"].rsplit(":", 1)[1],
                "--notes-file",
                str(notes),
            ]
        )
    execute(
        [
            "gh",
            "release",
            "upload",
            tag,
            "--repo",
            repository,
            "--clobber",
            *map(str, assets),
        ]
    )
    execute(
        [
            "gh",
            "release",
            "edit",
            tag,
            "--repo",
            repository,
            "--draft=false",
            "--prerelease",
            "--notes-file",
            str(notes),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    root = tools.parents[1]
    assembly = json.loads(args.assembly.read_text())
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    if commit != assembly["recipe_commit"]:
        raise ValueError("checkout is not the recipe selected by the assembly lock")
    if run(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("container build requires a clean recipe checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    publication_repository = os.environ.get(
        "GITHUB_REPOSITORY", "local-inference-lab/blackwell-llm-docker"
    )
    changelog = collect_release_changelog(assembly, publication_repository)
    changelog_path = args.output / "release-changelog.json"
    changelog_path.write_text(json.dumps(changelog, sort_keys=True, indent=2) + "\n")
    bundles = args.output / "components"
    download(assembly, bundles)
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
    for role in assembly["components"]:
        command += [f"--{role}-bundle", str(bundles / role)]
    execute(command)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assembly"] = assembly
    manifest["release_changelog"] = changelog
    auxiliary_cache = (
        Path(
            os.environ.get(
                "LIL_COMPONENT_ARCHIVE_CACHE", str(args.output / "archive-cache")
            )
        )
        / "lmcache-cumem-source"
    )
    manifest["auxiliary"] = {
        "lmcache_cumem": prepare_auxiliary(manifest, runtime, auxiliary_cache)
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
    image = assembly["image"]
    test_results = args.output / "qualification-results"
    test_results.mkdir()
    execute(
        [
            str(tools / "run_serialized_build.sh"),
            "bash",
            str(tools / "build_runtime_image.sh"),
            str(runtime),
            image,
        ]
    )
    base = manifest["components"]["foundation"]["source"]["image"]
    inspection = json.loads(run(["docker", "image", "inspect", base, image]))
    base_layers = inspection[0]["RootFS"]["Layers"]
    image_layers = inspection[1]["RootFS"]["Layers"]
    if (
        len(base_layers) != 67
        or len(image_layers) != 68
        or image_layers[:67] != base_layers
    ):
        raise ValueError(
            "container must retain the 67 foundation layers plus one application layer"
        )
    require_idle_gpu(args.gpu)
    execute(
        [
            "docker",
            "run",
            "--rm",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            image,
            "python",
            "-m",
            "runtime.native_cli_contract",
        ]
    )
    execute(runtime_smoke_command(image, args.gpu))
    execute(
        [
            "docker",
            "run",
            "--rm",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            image,
            "python",
            "-c",
            (
                "from lmcache.lmcache_fs import LMCacheFSClient; "
                "from vllm.v1.core.kv_cache_manager import KVCacheManager; "
                "assert hasattr(KVCacheManager, 'reserve_external_boundary_checkpoint')"
            ),
        ]
    )
    # Cached native wheels must still satisfy the filesystem and checkpoint contracts.
    source = args.output / "lmcache-tests"
    execute(["git", "init", "--quiet", str(source)])
    execute(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--quiet",
            "--depth=1",
            "https://github.com/local-inference-lab/LMCache.git",
            assembly["components"]["lmcache"]["source_commit"],
        ]
    )
    execute(
        [
            "git",
            "-C",
            str(source),
            "checkout",
            "--quiet",
            "--detach",
            assembly["components"]["lmcache"]["source_commit"],
        ]
    )
    execute(
        [
            "docker",
            "run",
            "--rm",
            "--ipc",
            "private",
            "--shm-size",
            "1g",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            "--mount",
            f"type=bind,src={source / 'tests'},dst=/qualification/tests,readonly",
            "--mount",
            f"type=bind,src={test_results},dst=/results",
            "-w",
            "/qualification",
            "-e",
            "LMCACHE_TRACK_USAGE=false",
            image,
            "python",
            "-m",
            "pytest",
            "--noconftest",
            "-q",
            "--junitxml=/results/lmcache-contracts.xml",
            "/qualification/tests/v1/multiprocess/test_checkpoint_identity.py",
            "/qualification/tests/v1/multiprocess/test_checkpoint_index.py",
            "/qualification/tests/v1/multiprocess/test_checkpoint_storage.py",
            "/qualification/tests/v1/storage_backend/test_fs_native_connector.py",
            "/qualification/tests/v1/test_vllm_semantic_checkpoint_transfer.py",
        ],
        timeout=600,
    )
    cache_results = verify_cache_test_report(test_results / "lmcache-contracts.xml")
    receipt = {
        **assembly,
        "status": "qualified",
        "qualification_scope": "Native GPU smoke and LMCache checkpoint/filesystem contract tests; "
        "model-serving performance and full GLM cache E2E remain unqualified.",
        "runtime_manifest_sha256": digest(manifest_path),
        "release_changelog_sha256": digest(changelog_path),
        "layers": 68,
        "cache_contract_tests": cache_results,
    }
    (args.output / "container-release.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    if args.build_only:
        print(
            f"Build and native checks complete: {image}; registry publication was not requested"
        )
        return
    repository = publication_repository
    token = os.environ["GH_TOKEN"]
    actor = os.environ["GITHUB_ACTOR"]
    # Isolate registry credentials without replacing the shared Buildx configuration.
    with tempfile.TemporaryDirectory(prefix="ghcr-publisher-") as registry_config:
        registry_docker = ["docker", "--config", registry_config]
        execute(
            registry_docker
            + ["login", "ghcr.io", "--username", actor, "--password-stdin"],
            input=token.encode(),
            stdout=subprocess.DEVNULL,
        )
        execute(registry_docker + ["push", image])
        inspect = json.loads(run(["docker", "image", "inspect", image]))[0]
        receipt["digest"] = next(
            item
            for item in inspect["RepoDigests"]
            if item.startswith(image.split(":")[0] + "@sha256:")
        )
        receipt_path = args.output / "container-release.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        # Source advances are handled by a subsequent scan; immutable images remain usable.
        promote_alias = alias_is_current(
            assembly, repository, os.environ.get("GITHUB_REF", "")
        )
        if promote_alias:
            execute(["docker", "tag", image, assembly["alias"]])
            execute(registry_docker + ["push", assembly["alias"]])
        receipt["alias_updated"] = promote_alias
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    notes = args.output / "release-notes.md"
    rows = [
        f"# Wheel-built runtime: {assembly['channel']} / {assembly['release_channel']}",
        "",
        f"Image: `{image}`",
        "",
        f"Digest: `{receipt['digest']}`",
        "",
        receipt["qualification_scope"],
        "",
        render_release_notes(changelog).rstrip(),
        "",
        "## Reproducible component inputs",
        "",
        "| Component | Source commit | Wheel release |",
        "|---|---|---|",
    ]
    for role, component in assembly["components"].items():
        rows.append(
            f"| {role} | `{component['source_commit']}` | "
            f"[{component['release']}]({component['release_url']}) |"
        )
    notes.write_text("\n".join(rows) + "\n")
    publish_release(
        repository,
        assembly,
        commit,
        notes,
        [
            receipt_path,
            manifest_path,
            changelog_path,
            stage_assembly_lock(args.assembly, args.output),
        ],
    )


if __name__ == "__main__":
    main()
