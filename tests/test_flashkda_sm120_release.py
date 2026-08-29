from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "patches/releases/flashkda-sm120-tensormap-r1/flashkda"
LOCK = RELEASE / "integration.lock.json"
PATCH = RELEASE / "integration.patch"
MANIFEST = ROOT / "manifests/flashkda/kimi-k3-sm120-tensormap.json"
DOCKERFILE = ROOT / "Dockerfile.kimi-k3-infernal-invocation-cu133-torch213"
BUILD = ROOT / "build-kimi-k3-infernal-invocation-cu133-torch213.sh"
RECEIPT = ROOT / "validation/kimi-k3-flashkda-sm120-tensormap-20260830.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_flashkda_source_lock_is_self_consistent() -> None:
    lock = json.loads(LOCK.read_text())
    manifest = json.loads(MANIFEST.read_text())
    receipt = json.loads(RECEIPT.read_text())

    assert lock["schema_version"] == 1
    assert lock["base"]["repository"] == manifest["repository"]
    assert lock["base"]["ref"] == manifest["base_ref"]
    assert lock["base"]["commit"] == manifest["base_commit"]
    assert lock["base"]["commit"] == receipt["source"]["commit"]
    assert lock["result"]["tree"] == receipt["source"]["result_tree"]
    assert lock["result"]["patch_sha256"] == sha256(PATCH)
    assert lock["result"]["patch_sha256"] == receipt["source"]["patch_sha256"]
    assert lock["manifest"]["sha256"] == sha256(MANIFEST)
    assert lock["submodules"] == manifest["submodules"]
    assert lock["submodules"]["cutlass"]["commit"] == (
        "5c149f52a436782210263fb2f19b354443a61c6a"
    )
    assert manifest["source_patches"] == [
        {
            "path": "releases/flashkda-sm120-tensormap-r1/flashkda/integration.patch",
            "sha256": sha256(PATCH),
        }
    ]


def test_flashkda_patch_preserves_generic_abi_and_targets_sm120() -> None:
    patch = PATCH.read_text()

    assert "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)" in patch
    assert "__cuda_syscall_cp_async_bulk_unicast" in patch
    assert "WorkspaceLargeSmemLayout" in patch
    assert patch.count("load_workspace_tile(") >= 7
    assert "seq_len=80" in patch
    assert 'assert "__cuda_syscall_cp_async_bulk_unicast" not in symbols' in patch
    assert "-INSTANTIATE_STATE_VARIANTS" not in patch
    assert "-INSTANTIATE_CHECKPOINT_VARIANTS" not in patch


def test_kimi_image_build_composes_and_labels_flashkda() -> None:
    dockerfile = DOCKERFILE.read_text()
    build = BUILD.read_text()

    assert "compose_source flashkda /opt/kimi-k3/flashkda" in dockerfile
    assert "flashkda submodule update --init --recursive" in dockerfile
    assert "-DFETCHCONTENT_SOURCE_DIR_FLASHKDA=/opt/kimi-k3/flashkda" in dockerfile
    assert (
        'local-inference.flashkda.integration.tree="${FLASHKDA_INTEGRATION_TREE}"'
        in dockerfile
    )
    assert 'read_lock flashkda FLASHKDA "${flashkda_composition_root}"' in build
    assert "-flashkda${FLASHKDA_INTEGRATION_TREE:0:10}" in build
    assert "__cuda_syscall_cp_async_bulk_unicast" in build
    assert "assert_label local-inference.flashkda.integration.tree" in build
    assert "assert_label local-inference.flashkda.cutlass.commit" in build

    environment = os.environ.copy()
    environment["PRINT_RELEASE_CONFIG"] = "1"
    output = subprocess.run(
        [str(BUILD)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "flashkda_ref=dev" in output
    assert "flashkda_commit=ee0be888cd0e972f9409bf53756f8c38c6652173" in output
    assert "flashkda_tree=6851531e95decd57c28f166fa6ce8aa381a1aa04" in output
    assert (
        "flashkda_patch_sha256="
        "2119fdd721decd873cdbbbe5263f624e21b8b2e83cede36d9f19e75377895951" in output
    )


def test_receipt_records_the_qualified_memory_envelope() -> None:
    receipt = json.loads(RECEIPT.read_text())

    assert receipt["root_cause"]["checkpoint_abi_caused_delta"] is False
    assert receipt["root_cause"]["vsplit_specific"] is False
    assert receipt["kernel_validation"]["syscall_symbol_present"] is False
    assert receipt["kernel_validation"]["racecheck_errors"] == 0
    assert receipt["live_profile"]["max_model_len"] == 320000
    assert receipt["live_profile"]["kv_cache_bytes_per_rank"] == 1879048192
    assert receipt["live_profile"]["request_errors"] == 0
    assert receipt["rejected_profile"]["first_flashkda_request_completed"] is True
