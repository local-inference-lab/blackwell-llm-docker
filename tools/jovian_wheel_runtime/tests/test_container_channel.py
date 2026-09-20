from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import container_channel as channel


def test_identity_is_order_independent_and_includes_recipe_and_all_components():
    left = channel.assembly_identity(
        {"channel": "jovian"}, {"vllm": "a", "b12x": "b"}, "c"
    )
    assert left == channel.assembly_identity(
        {"channel": "jovian"}, {"b12x": "b", "vllm": "a"}, "c"
    )
    assert left != channel.assembly_identity(
        {"channel": "jovian"}, {"b12x": "z", "vllm": "a"}, "c"
    )
    assert left != channel.assembly_identity(
        {"channel": "jovian"}, {"b12x": "b", "vllm": "a"}, "d"
    )


@pytest.mark.parametrize("name", ["../escape", "/escape"])
def test_archive_rejects_traversal(tmp_path, name):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        channel.safe_extract(archive, tmp_path / "out")


def test_archive_rejects_links(tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp"
        stream.addfile(member)
    with pytest.raises(ValueError, match="unsafe"):
        channel.safe_extract(archive, tmp_path / "out")


def test_archive_accepts_internal_native_library_symlink(tmp_path):
    archive = tmp_path / "native.tar"
    with tarfile.open(archive, "w") as stream:
        library = tarfile.TarInfo("native/lib/libnccl.so.2")
        library.size = 1
        stream.addfile(library, io.BytesIO(b"x"))
        link = tarfile.TarInfo("native/lib/libnccl.so")
        link.type = tarfile.SYMTYPE
        link.linkname = "libnccl.so.2"
        stream.addfile(link)
    channel.safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out/native/lib/libnccl.so").read_bytes() == b"x"


def test_comparison_detects_renamed_sources_and_truncation():
    assert channel.has_source_changes(
        {"files": [{"filename": "docs/x", "previous_filename": "src/x"}]}, ["src/"]
    )
    assert channel.has_source_changes(
        {"files": [{"filename": "docs/x"}] * 300}, ["src/"]
    )
    assert not channel.has_source_changes({"files": [{"filename": "docs/x"}]}, ["src/"])


def make_release(commit, *, uploaded=True):
    return {
        "tag_name": "beta-" + commit,
        "draft": False,
        "created_at": "2026-09-13T00:00:00Z",
        "html_url": "https://github.com/example",
        "assets": [
            {
                "id": index,
                "name": name,
                "size": 1,
                "state": "uploaded" if uploaded else "starter",
            }
            for index, name in enumerate(
                ("manifest.json", "bundle.tar.zst", "bundle.tar.zst.sha256")
            )
        ],
    }


def setup_api(monkeypatch, commit, tip, manifest, *, comparison=None, uploaded=True):
    def fake_api(endpoint):
        if "/commits/" in endpoint:
            return {"sha": tip}
        if "/compare/" in endpoint:
            return comparison
        return [make_release(commit, uploaded=uploaded)]

    monkeypatch.setattr(channel, "api", fake_api)
    monkeypatch.setattr(
        channel,
        "asset_bytes",
        lambda repo, asset: (
            json.dumps(manifest).encode()
            if asset["name"] == "manifest.json"
            else ("a" * 64 + "  bundle.tar.zst\n").encode()
        ),
    )
    return {
        "repository": "local-inference-lab/b12x",
        "branch": "master",
        "release_prefix": "beta-",
        "source_paths": ["b12x/"],
    }


def test_resolve_exact_source(monkeypatch):
    commit = "a" * 40
    manifest = {
        "schema": channel.EXPECTED_SCHEMAS["b12x"],
        "source": {"commit": commit},
    }
    config = setup_api(monkeypatch, commit, commit, manifest)
    selected, actual = channel.select_component("b12x", config)
    assert actual == manifest
    assert selected["observed_branch_commit"] == commit
    assert (
        selected["manifest_sha256"]
        == hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    )


def test_unbuilt_source_change_waits_instead_of_selecting_older_wheel(monkeypatch):
    config = setup_api(
        monkeypatch,
        "a" * 40,
        "b" * 40,
        {},
        comparison={"status": "ahead", "files": [{"filename": "b12x/kernel.py"}]},
    )
    with pytest.raises(channel.PendingBuild):
        channel.select_component("b12x", config)


def test_divergent_release_is_not_accepted(monkeypatch):
    config = setup_api(
        monkeypatch, "a" * 40, "b" * 40, {}, comparison={"status": "diverged"}
    )
    with pytest.raises(channel.PendingBuild):
        channel.select_component("b12x", config)


def test_incomplete_upload_waits(monkeypatch):
    config = setup_api(monkeypatch, "a" * 40, "a" * 40, {}, uploaded=False)
    with pytest.raises(channel.PendingBuild):
        channel.select_component("b12x", config)


def test_lmcache_requires_community_receipt(monkeypatch):
    commit = "a" * 40
    config = setup_api(
        monkeypatch,
        commit,
        commit,
        {"schema": channel.EXPECTED_SCHEMAS["lmcache"], "source": {"commit": commit}},
    )
    config["required_reviews"] = [49, 62]
    with pytest.raises(ValueError, match="complete community"):
        channel.select_component("lmcache", config)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "uploading",
        "draft",
        "checksum",
        "identity",
        "changelog_checksum",
        "changelog_identity",
        "manifest_changelog",
    ],
)
@pytest.mark.parametrize(
    "lock_name",
    [
        "community-assembly.json",
        "community-assembly-beta.json",
        "community-assembly-main.json",
    ],
)
def test_published_assembly_requires_complete_matching_receipt(
    monkeypatch, fault, lock_name
):
    assembly = {
        "release_tag": "beta-example",
        "assembly_sha256": "a" * 64,
        "release_channel": "beta",
    }
    changelog_data = {"assembly_sha256": assembly["assembly_sha256"], "changes": []}
    manifest_data = {"packages": [], "release_changelog": changelog_data}
    manifest = json.dumps(manifest_data).encode()
    changelog = json.dumps(changelog_data).encode()
    receipt = {
        **assembly,
        "status": "qualified",
        "runtime_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "release_changelog_sha256": hashlib.sha256(changelog).hexdigest(),
        "digest": "ghcr.io/local-inference-lab/vllm@sha256:" + "b" * 64,
    }
    if fault == "checksum":
        receipt["runtime_manifest_sha256"] = "c" * 64
    if fault == "changelog_checksum":
        receipt["release_changelog_sha256"] = "c" * 64
    if fault == "identity":
        receipt["assembly_sha256"] = "c" * 64
    if fault == "changelog_identity":
        changelog_data["assembly_sha256"] = "c" * 64
        changelog = json.dumps(changelog_data).encode()
        receipt["release_changelog_sha256"] = hashlib.sha256(changelog).hexdigest()
    if fault == "manifest_changelog":
        manifest_data["release_changelog"] = {**changelog_data, "changes": ["wrong"]}
        manifest = json.dumps(manifest_data).encode()
        receipt["runtime_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    payloads = {
        "container-release.json": json.dumps(receipt).encode(),
        "manifest.json": manifest,
        "release-changelog.json": changelog,
        lock_name: json.dumps(assembly).encode(),
    }
    assets = [
        {"name": name, "state": "uploaded", "size": len(data)}
        for name, data in payloads.items()
    ]
    if fault == "missing":
        assets.pop()
    if fault == "uploading":
        assets[-1]["state"] = "starter"
    monkeypatch.setattr(
        channel,
        "api",
        lambda endpoint: [
            {"tag_name": "beta-example", "draft": fault == "draft", "assets": assets}
        ],
    )
    monkeypatch.setattr(
        channel, "asset_bytes", lambda repo, asset: payloads[asset["name"]]
    )
    assert channel.completed_publication(
        "local-inference-lab/blackwell-llm-docker", assembly
    ) is (fault is None and lock_name != "community-assembly-main.json")


def test_archive_cache_reuses_verified_bytes_without_network(monkeypatch, tmp_path):
    data = b"immutable component archive"
    component = {
        "repository": "local-inference-lab/LMCache",
        "archive_asset_id": 1,
        "archive_sha256": hashlib.sha256(data).hexdigest(),
    }
    calls = []

    def fetch(args, *, stdout, check):
        calls.append(args)
        stdout.write(data)

    monkeypatch.setattr(channel.subprocess, "run", fetch)
    archive = channel.fetch_archive(component, tmp_path)
    assert channel.fetch_archive(component, tmp_path) == archive
    assert len(calls) == 1
    archive.write_bytes(b"damaged cache")
    with pytest.raises(ValueError, match="cached component archive checksum"):
        channel.fetch_archive(component, tmp_path)
    assert len(calls) == 1


def test_failed_archive_verification_does_not_publish_cache_entry(
    monkeypatch, tmp_path
):
    component = {
        "repository": "local-inference-lab/LMCache",
        "archive_asset_id": 1,
        "archive_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        channel.subprocess,
        "run",
        lambda args, *, stdout, check: stdout.write(b"bad archive"),
    )
    with pytest.raises(ValueError, match="locked component archive checksum"):
        channel.fetch_archive(component, tmp_path)
    assert list(tmp_path.iterdir()) == []
