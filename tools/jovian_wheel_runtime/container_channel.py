#!/usr/bin/env python3
"""Resolve complete wheel releases into reproducible container build inputs.

Only configured repositories and branches are trusted. Events merely request a
rescan; their payloads never supply executable commands, URLs or source refs.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import os
import posixpath
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from assemble_qwen38_runtime_bundle import (
    EXPECTED_SCHEMAS,
    ngc_foundation_manifest,
    validate_compatibility,
)


class PendingBuild(RuntimeError):
    """A required source revision has no complete published wheel yet."""


def run(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True).stdout


def api(endpoint: str) -> object:
    return json.loads(run(["gh", "api", endpoint]))


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def release_assets(release: dict) -> tuple[dict, dict, dict]:
    """Select exactly one manifest and source archive with its checksum."""
    assets = release["assets"]
    manifests = [a for a in assets if a["name"] == "manifest.json"]
    archives = [a for a in assets if a["name"].endswith(".tar.zst")]
    if len(manifests) != 1 or len(archives) != 1:
        raise PendingBuild("release does not contain one manifest and source archive")
    checksums = [a for a in assets if a["name"] == archives[0]["name"] + ".sha256"]
    if len(checksums) != 1:
        raise PendingBuild("release archive checksum is not published")
    for item in [manifests[0], archives[0], checksums[0]]:
        if item.get("state") != "uploaded" or not item.get("size"):
            raise PendingBuild("release asset upload is incomplete")
        if PurePosixPath(item["name"]).name != item["name"]:
            raise ValueError("release contains an unsafe asset name")
    return manifests[0], archives[0], checksums[0]


def asset_bytes(repository: str, asset: dict) -> bytes:
    return run(
        [
            "gh",
            "api",
            f"repos/{repository}/releases/assets/{int(asset['id'])}",
            "-H",
            "Accept: application/octet-stream",
        ]
    )


def has_source_changes(comparison: dict, paths: list[str]) -> bool:
    # GitHub truncates comparison file lists at 300; incomplete evidence fails closed.
    files = comparison.get("files", [])
    if len(files) >= 300:
        return True
    return any(
        name == prefix or (prefix.endswith("/") and name.startswith(prefix))
        for file in files
        for name in (file["filename"], file.get("previous_filename", ""))
        for prefix in paths
    )


def select_component(role: str, config: dict) -> tuple[dict, dict]:
    repository = config["repository"]
    branch = api(f"repos/{repository}/commits/{quote(config['branch'], safe='')}")
    tip = branch["sha"]
    releases = api(f"repos/{repository}/releases?per_page=100")
    prefix = config["release_prefix"]
    candidates = [
        r
        for r in releases
        if not r["draft"]
        and re.fullmatch(re.escape(prefix) + r"[0-9a-f]{40}", r["tag_name"])
    ]
    # A delayed upload for an older source must never roll the channel backwards.
    candidates.sort(key=lambda r: r["created_at"], reverse=True)
    for release in candidates:
        commit = release["tag_name"][len(prefix) :]
        seed = config.get("bootstrap", {})
        seeded = tip == seed.get("branch_commit") and commit == seed.get(
            "release_commit"
        )
        comparison = {"status": "identical", "files": []}
        if commit != tip and not seeded:
            comparison = api(f"repos/{repository}/compare/{commit}...{tip}")
            if comparison["status"] not in {"ahead", "identical"}:
                continue
            if has_source_changes(comparison, config["source_paths"]):
                continue
        try:
            manifest_asset, archive, checksum = release_assets(release)
        except PendingBuild as error:
            raise PendingBuild(f"{role}: {error}") from error
        manifest_bytes = asset_bytes(repository, manifest_asset)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != EXPECTED_SCHEMAS[role]:
            raise ValueError(f"{role}: unexpected release schema")
        if manifest.get("source", {}).get("commit") != commit:
            raise ValueError(f"{role}: release source identity mismatch")
        if role == "lmcache":
            receipt = manifest.get("community_source", {})
            expected = set(map(str, config["required_reviews"]))
            if (
                not expected <= set(receipt.get("required_reviews", {}))
                or receipt.get("source_commit") != commit
                or not receipt.get("verified_python_modules")
            ):
                raise ValueError(
                    "LMCache release lacks the complete community source proof"
                )
        checksum_data = asset_bytes(repository, checksum).decode().strip().split()
        if (
            len(checksum_data) != 2
            or not re.fullmatch(r"[0-9a-f]{64}", checksum_data[0])
            or checksum_data[1].lstrip("*") != archive["name"]
        ):
            raise ValueError(f"{role}: invalid archive checksum")
        return {
            "repository": repository,
            "branch": config["branch"],
            "release": release["tag_name"],
            "source_commit": commit,
            "observed_branch_commit": tip,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "archive": archive["name"],
            "archive_asset_id": archive["id"],
            "archive_sha256": checksum_data[0],
            "release_url": release["html_url"],
        }, manifest
    raise PendingBuild(
        f"{role}: no complete wheel for branch {config['branch']} at {tip}"
    )


def assembly_identity(channel: dict, components: dict, recipe_commit: str) -> str:
    """Bind the image to all component receipts and the complete build recipe."""
    return hashlib.sha256(
        canonical(
            {
                "channel": channel,
                "components": components,
                "recipe_commit": recipe_commit,
            }
        )
    ).hexdigest()


def completed_publication(repository: str, assembly: dict) -> bool:
    """Do not suppress retries for a release with interrupted asset uploads."""
    tag = assembly["release_tag"]
    page = 1
    release = None
    while True:
        releases = api(f"repos/{repository}/releases?per_page=100&page={page}")
        release = next((item for item in releases if item["tag_name"] == tag), None)
        if release is not None or len(releases) < 100:
            break
        page += 1
    if release is None or release["draft"]:
        return False
    assets = {item["name"]: item for item in release["assets"]}
    lock_name = "community-assembly.json"
    if lock_name not in assets:
        # Releases produced by channel-matrix jobs used a channel-specific
        # basename. Accept only the selected channel, with the same checks.
        lock_name = f"community-assembly-{assembly.get('release_channel', 'main')}.json"
    required = {
        "container-release.json",
        "manifest.json",
        "release-changelog.json",
        lock_name,
    }
    if any(
        name not in assets
        or assets[name].get("state") != "uploaded"
        or not assets[name].get("size")
        for name in required
    ):
        return False
    receipt = json.loads(asset_bytes(repository, assets["container-release.json"]))
    manifest = asset_bytes(repository, assets["manifest.json"])
    changelog = asset_bytes(repository, assets["release-changelog.json"])
    manifest_data = json.loads(manifest)
    changelog_data = json.loads(changelog)
    lock = json.loads(asset_bytes(repository, assets[lock_name]))
    return (
        receipt.get("status") == "qualified"
        and receipt.get("assembly_sha256") == assembly["assembly_sha256"]
        and lock.get("assembly_sha256") == assembly["assembly_sha256"]
        and receipt.get("runtime_manifest_sha256")
        == hashlib.sha256(manifest).hexdigest()
        and receipt.get("release_changelog_sha256")
        == hashlib.sha256(changelog).hexdigest()
        and changelog_data.get("assembly_sha256") == assembly["assembly_sha256"]
        and manifest_data.get("release_changelog") == changelog_data
        and bool(
            re.fullmatch(
                r"ghcr.io/local-inference-lab/vllm@sha256:[0-9a-f]{64}",
                receipt.get("digest", ""),
            )
        )
    )


def channel_config(config: dict, name: str) -> dict:
    """Select branch overrides without duplicating dependencies or build rules."""
    if config.get("schema") != "local-inference-container-channel/v2":
        raise ValueError("unsupported channel schema")
    channels = config["channels"]
    if name not in channels:
        raise ValueError(f"unknown release channel: {name}")
    tags = []
    for key, entry in channels.items():
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key):
            raise ValueError("invalid release channel name")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry["image_tag"]):
            raise ValueError("invalid image tag")
        family = entry.get("channel", config["channel"])
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", family):
            raise ValueError("invalid release family")
        if set(entry["branches"]) - set(config["components"]):
            raise ValueError("branch override names an unknown component")
        changelog = entry.get("changelog", {})
        if set(changelog) - {"required_components"}:
            raise ValueError("release channel contains an unknown changelog policy")
        required = changelog.get("required_components", [])
        if (
            not isinstance(required, list)
            or len(required) != len(set(required))
            or set(required) - set(config["components"])
        ):
            raise ValueError("invalid required changelog component list")
        tags.append(entry["image_tag"])
    if len(tags) != len(set(tags)):
        raise ValueError("release channels must have distinct image tags")
    selected = copy.deepcopy(config)
    entry = selected.pop("channels")[name]
    selected["channel"] = entry.get("channel", selected["channel"])
    selected["release_channel"] = name
    selected["image_tag"] = entry["image_tag"]
    selected["changelog"] = entry.get("changelog", {})
    for role, branch in entry["branches"].items():
        if not isinstance(branch, str) or not branch or branch.startswith("-"):
            raise ValueError("invalid component branch")
        selected["components"][role]["branch"] = branch
    return selected


def resolve(config_path: Path, output: Path, name: str = "main") -> dict:
    config = channel_config(json.loads(config_path.read_text()), name)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", config["channel"]):
        raise ValueError("invalid channel name")
    if not re.fullmatch(
        r"ghcr.io/local-inference-lab/[a-z0-9-]+", config["image_repository"]
    ):
        raise ValueError("image repository is outside the configured organization")
    components, manifests = {}, {}
    for role, component in config["components"].items():
        if not re.fullmatch(
            r"local-inference-lab/[A-Za-z0-9_-]+", component["repository"]
        ):
            raise ValueError(
                "component repository is outside the configured organization"
            )
        components[role], manifests[role] = select_component(role, component)
    manifests["foundation"] = ngc_foundation_manifest(
        config_path.parent / "foundation.lock"
    )
    validate_compatibility(manifests)
    root = config_path.resolve().parents[2]
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    identity = assembly_identity(config, components, commit)
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    result = {
        "schema": "local-inference-container-assembly/v1",
        "channel": config["channel"],
        "release_channel": name,
        "recipe_commit": commit,
        "assembly_sha256": identity,
        "components": components,
        "changelog": config.get("changelog", {}),
        "image": f"{config['image_repository']}:{config['image_tag']}-{date}-{identity[:16]}",
        "alias": f"{config['image_repository']}:{config['image_tag']}",
        "release_tag": f"{config['image_tag']}-{identity}",
        "status": "research-only",
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def resolve_matrix(config_path: Path, directory: Path, repository: str) -> dict:
    """Let a complete channel build while another channel waits for wheels."""
    config = json.loads(config_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    matrix = {"include": []}
    for name in config["channels"]:
        try:
            assembly = resolve(config_path, directory / f"{name}.json", name)
        except PendingBuild as error:
            print(f"{name}: waiting for component build: {error}")
            continue
        if completed_publication(repository, assembly):
            print(f"{name}: assembly already published")
            continue
        matrix["include"].append(
            {
                "channel": name,
                "assembly": base64.b64encode(canonical(assembly)).decode(),
            }
        )
        print(f"{name}: resolved {assembly['image']}")
    return matrix


def safe_extract(tar_path: Path, destination: Path) -> None:
    """Allow internal library links but reject special files and escaping paths."""
    with tarfile.open(tar_path) as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (
                    member.isfile()
                    or member.isdir()
                    or member.issym()
                    or member.islnk()
                )
            ):
                raise ValueError(f"unsafe component archive member: {member.name}")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                parent = str(path.parent) if member.issym() else "."
                resolved = posixpath.normpath(posixpath.join(parent, str(target)))
                if (
                    target.is_absolute()
                    or resolved == ".."
                    or resolved.startswith("../")
                ):
                    raise ValueError(f"unsafe component archive link: {member.name}")
        archive.extractall(destination, members=members, filter="data")


def fetch_archive(component: dict, directory: Path) -> Path:
    """Cache verified release archives by content digest, with atomic publication."""
    identity = component["archive_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("invalid component archive digest")
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / f"{identity}.tar.zst"
    if cached.exists():
        if cached.is_symlink() or digest(cached) != identity:
            raise ValueError(f"cached component archive checksum mismatch: {cached}")
        return cached
    with tempfile.TemporaryDirectory(prefix="upload-", dir=directory) as temporary:
        staged = Path(temporary) / "artifact.tar.zst"
        with staged.open("wb") as stream:
            subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{component['repository']}/releases/assets/{int(component['archive_asset_id'])}",
                    "-H",
                    "Accept: application/octet-stream",
                ],
                stdout=stream,
                check=True,
            )
        if digest(staged) != identity:
            raise ValueError("locked component archive checksum mismatch")
        os.replace(staged, cached)
    return cached


def download(assembly: dict, destination: Path) -> None:
    """Fetch the exact locked asset IDs and verify bytes before extraction."""
    destination.mkdir(parents=True, exist_ok=False)
    for role, component in assembly["components"].items():
        with tempfile.TemporaryDirectory(prefix="component-download-") as temporary:
            compressed = fetch_archive(
                component,
                Path(os.environ.get("LIL_COMPONENT_ARCHIVE_CACHE", temporary)),
            )
            tar_path = Path(temporary) / "component.tar"
            with tar_path.open("wb") as stream:
                subprocess.run(
                    ["zstd", "-dc", str(compressed)], stdout=stream, check=True
                )
            target = destination / role
            target.mkdir()
            safe_extract(tar_path, target)
            if digest(target / "manifest.json") != component["manifest_sha256"]:
                raise ValueError(f"{role}: archive and release manifests disagree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolver = commands.add_parser("resolve")
    resolver.add_argument("--config", type=Path, required=True)
    resolver.add_argument("--output", type=Path, required=True)
    resolver.add_argument("--channel", default="main")
    matrix_parser = commands.add_parser("resolve-matrix")
    matrix_parser.add_argument("--config", type=Path, required=True)
    matrix_parser.add_argument("--output", type=Path, required=True)
    matrix_parser.add_argument("--repository", required=True)
    downloader = commands.add_parser("download")
    downloader.add_argument("--assembly", type=Path, required=True)
    downloader.add_argument("--output", type=Path, required=True)
    completed = commands.add_parser("completed")
    completed.add_argument("--assembly", type=Path, required=True)
    completed.add_argument("--repository", required=True)
    args = parser.parse_args()
    if args.command == "resolve-matrix":
        matrix = resolve_matrix(args.config, args.output, args.repository)
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                stream.write(f"build={str(bool(matrix['include'])).lower()}\n")
                stream.write(f"matrix={canonical(matrix).decode()}\n")
        return
    if args.command == "completed":
        assembly = json.loads(args.assembly.read_text())
        print("true" if completed_publication(args.repository, assembly) else "false")
        return
    if args.command == "download":
        download(json.loads(args.assembly.read_text()), args.output)
        return
    try:
        assembly = resolve(args.config, args.output, args.channel)
    except PendingBuild as error:
        print(f"Waiting for component build: {error}")
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                stream.write("ready=false\n")
        return
    print(f"Resolved {assembly['image']}")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("ready=true\n")
            stream.writelines(
                f"{key}={assembly[key]}\n" for key in ("image", "alias", "release_tag")
            )


if __name__ == "__main__":
    main()
