from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "patches" / "releases" / "kimi-k3-qsrt-tp8-safety"
MANIFEST = ROOT / "manifests" / "b12x" / "kimi-k3-qsrt-tp8-safety.json"
SOURCE_PATCH = (
    ROOT / "manifests" / "b12x" / "patches" / "kimi-k3-qsrt-tp8-qualified-compat.patch"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_b12x_release_locks_qsrt_and_inactive_route_sources() -> None:
    manifest = json.loads(MANIFEST.read_text())
    lock = json.loads((RELEASE / "b12x" / "integration.lock.json").read_text())

    assert manifest["base_ref"] == "refs/heads/agent/pr197-base-glm52-sqg"
    assert manifest["pull_requests"] == [
        {
            "number": 237,
            "head": "7917d618a09cb01cde7df31ac6696dbe86dd72ab",
            "title": "Sanitize inactive QSRT W4A16 routes",
        }
    ]
    assert manifest["source_patches"][0]["sha256"] == _sha256(SOURCE_PATCH)
    assert lock["base"]["commit"] == "6e2c6bfb3991b27509c0e2e70dda5d90bb039691"
    assert lock["pull_requests"][0]["number"] == 237
    assert lock["pull_requests"][0]["head"] == manifest["pull_requests"][0]["head"]
    assert lock["result"]["tree"] == "64dc2cb0c6582f45a3414f5ba40c370aca7600f1"
    assert lock["result"]["patch_sha256"] == _sha256(
        RELEASE / "b12x" / "integration.patch"
    )


def test_vllm_and_lmcache_sources_remain_qualified() -> None:
    vllm = json.loads((RELEASE / "vllm" / "integration.lock.json").read_text())
    lmcache = json.loads((RELEASE / "lmcache" / "integration.lock.json").read_text())

    assert vllm["result"]["tree"] == "12776c0df15ca4087b636c43004b5bc1fde61434"
    assert lmcache["result"]["tree"] == "e045d729bc5c4c63a40e13d032f42923de97812f"


def test_build_script_uses_tp8_safety_release() -> None:
    script = (ROOT / "build-kimi-k3-qsrt-tp16-runtime.sh").read_text()

    assert "release_root=patches/releases/kimi-k3-qsrt-tp8-safety" in script
    assert 'revision="${REVISION:-qsrt-tp8-safety}"' in script
