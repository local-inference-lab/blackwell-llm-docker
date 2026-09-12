"""Authenticate installed source and metadata while preserving native dependencies."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "source_locked_image_labels", RECIPE / "source_locked_image_labels.py"
)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)
native_spec = importlib.util.spec_from_file_location(
    "build_jovian_stable_native", RECIPE / "build_jovian_stable_native.py"
)
native = importlib.util.module_from_spec(native_spec)
native_spec.loader.exec_module(native)
bundle_spec = importlib.util.spec_from_file_location(
    "prepare_glm53_source_bundles", RECIPE / "prepare_glm53_source_bundles.py"
)
bundler = importlib.util.module_from_spec(bundle_spec)
bundle_spec.loader.exec_module(bundler)


@pytest.fixture
def lock():
    values = {
        "release.name": "serving-fixture",
        "release.version": "test",
        "runtime.base.image": "runtime@sha256:" + "a" * 64,
        "runtime.rootfs.layers": "2",
        "runtime.cache.fingerprint": "source-fixture",
        "runtime.cudagraph.mode": "FULL_AND_PIECEWISE",
        "runtime.lmcache.transfer": "engine-driven asynchronous shared memory",
        "runtime.scheduler.max-num-batched-tokens": "4096",
        "runtime.scheduler.prefill-compute-share": "0.4",
        "runtime.cuda.version": "13.3",
        "runtime.pytorch.version": "2.13.0",
        "model.repository": "target/fixture",
        "draft.repository": "draft/fixture",
        "draft.quantization": "MXFP8",
        "lmcache.version": "0.5.5.dev0",
    }
    for name in ("vllm", "b12x", "lmcache"):
        for key in ("commit", "tree", "package.tree"):
            values[f"{name}.{key}"] = f"{name}-{key}"
    for key in ("base.commit", "patch.sha256", "extension.sha256"):
        values[f"flashkda.{key}"] = key
    return values


def test_serving_labels_replace_incompatible_foundation_claims(lock):
    inherited = {
        "local-inference.vllm.composition": "different PR stack",
        "local-inference.cache.target-block-size": "512",
        "local-inference.glm53.mtp-backends": "moe:Humming",
        "local-inference.runtime.source-lock.sha256": "different-lock",
        "local-inference.scheduler.prefill-schedule-interval": "8",
        "org.opencontainers.image.version": "different-image",
    }
    overrides = labels.image_labels(lock, inherited, "b" * 64)
    assert overrides["local-inference.vllm.composition"] == ""
    assert overrides["local-inference.cache.target-block-size"] == ""
    assert "moe:Marlin" in overrides["local-inference.glm53.mtp-backends"]
    assert overrides["local-inference.runtime.source-lock.sha256"] == "b" * 64
    assert overrides["local-inference.scheduler.prefill-schedule-interval"] == "1"
    assert overrides["org.opencontainers.image.version"] == "test"
    assert overrides["local-inference.status"] == "implemented"
    assert all("qualified" not in value for value in overrides.values())


def test_unchanged_dependency_labels_are_preserved_by_inheritance(lock):
    dependency = "local-inference.flashinfer.commit"
    overrides = labels.image_labels(lock, {dependency: "source-ref"}, "b" * 64)
    assert dependency not in overrides
    for name in ("vllm", "b12x", "lmcache"):
        assert overrides[f"local-inference.{name}.commit"] == lock[f"{name}.commit"]


def test_deployment_moe_backend_is_declared_in_image_and_lock(lock):
    lock["runtime.moe-backend.default"] = "b12x"
    overrides = labels.image_labels(lock, {}, "b" * 64)
    assert overrides["local-inference.runtime.default.moe-backend"] == "b12x"
    dockerfile = (RECIPE / "Dockerfile.glm53-cache-contracts").read_text()
    assert "VLLM_DEFAULT_MOE_BACKEND=b12x" in dockerfile


def test_disk_engram_dependency_identity_replaces_inherited_version(lock):
    lock["runtime.liburing.version"] = "2.5"
    lock["runtime.liburing.package.version"] = "2.5-1build1"
    inherited = {"local-inference.liburing.version": "unrelated-version"}
    overrides = labels.image_labels(lock, inherited, "b" * 64)
    assert overrides["local-inference.liburing.version"] == "2.5"
    assert overrides["local-inference.liburing.package.version"] == "2.5-1build1"


def test_source_repository_labels_use_manifest_not_inferred_ownership(lock):
    lock["vllm.repository"] = "https://github.com/voipmonitor/vllm.git"
    overrides = labels.image_labels(lock, {}, "b" * 64)
    assert overrides["local-inference.vllm.repo"] == lock["vllm.repository"]
    assert overrides["local-inference.b12x.repo"] == ""


@pytest.mark.parametrize(
    "url",
    [
        "https://name:secret@example.org/repo.git",
        "/tmp/repo",
        "https://example.org/repo?secret=value",
    ],
)
def test_bundle_metadata_rejects_credentials_and_local_paths(url):
    with pytest.raises(ValueError, match="HTTPS URL without credentials"):
        bundler.repository_url(url)


def test_bundle_metadata_normalizes_github_ssh_url():
    assert (
        bundler.repository_url("git@github.com:voipmonitor/vllm.git")
        == "https://github.com/voipmonitor/vllm.git"
    )


def test_replaced_flashinfer_artifact_does_not_inherit_another_source_ref(lock):
    for key in (
        "commit",
        "artifact.image",
        "python.wheel.sha256",
        "jit-cache.wheel.sha256",
    ):
        lock[f"flashinfer.{key}"] = key
    overrides = labels.image_labels(
        lock, {"local-inference.flashinfer.ref": "unrelated-source"}, "b" * 64
    )
    assert overrides["local-inference.flashinfer.ref"] == ""
    assert overrides["local-inference.flashinfer.commit"] == "commit"
    assert overrides["local-inference.flashinfer.artifact.image"] == "artifact.image"


def test_ambiguous_source_lock_is_rejected(tmp_path):
    source = tmp_path / "source.lock"
    source.write_text("vllm.commit=a\nvllm.commit=b\n")
    with pytest.raises(ValueError, match="Duplicate"):
        labels.read_lock(source)


def git(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


@pytest.fixture
def source_bundle(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "config", "user.name", "Source fixture")
    git(source, "config", "user.email", "source@example.invalid")
    (source / ".gitignore").write_text("*.so\n")
    (source / "module.py").write_text("value = 1\n")
    (source / "obsolete.py").write_text("value = 2\n")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "Source fixture")
    destination = tmp_path / "installed source"
    subprocess.run(
        ["git", "clone", "--quiet", str(source), str(destination)], check=True
    )
    (destination / "extension.so").write_bytes(b"qualified native artifact")
    (source / "obsolete.py").unlink()
    (source / "module.py").write_text("value = 3\n")
    git(source, "add", "-A")
    git(source, "commit", "--quiet", "-m", "Complete source fixture")
    commit = git(source, "rev-parse", "HEAD")
    tree = git(source, "rev-parse", "HEAD^{tree}")
    bundle = tmp_path / "source.bundle"
    git(source, "bundle", "create", str(bundle), "HEAD")
    return destination, bundle, commit, tree


def test_installed_tree_matches_bundle_without_losing_native(source_bundle):
    destination, bundle, commit, tree = source_bundle
    for _ in range(2):
        subprocess.run(
            [
                "bash",
                str(RECIPE / "install_source_bundle.sh"),
                str(bundle),
                commit,
                tree,
                str(destination),
            ],
            check=True,
        )
        assert git(destination, "rev-parse", "HEAD") == commit
        assert git(destination, "status", "--porcelain") == ""
        assert (destination / "module.py").read_text() == "value = 3\n"
        assert not (destination / "obsolete.py").exists()
        assert (
            destination / "extension.so"
        ).read_bytes() == b"qualified native artifact"


def test_cutlass_identity_records_the_actual_committed_tree(source_bundle):
    destination, _, commit, tree = source_bundle
    assert native.cutlass_identity(destination) == {
        "cutlass.commit": git(destination, "rev-parse", "HEAD"),
        "cutlass.tree": git(destination, "rev-parse", "HEAD^{tree}"),
    }
    (destination / "module.py").write_text("uncommitted modification\n")
    with pytest.raises(ValueError, match="committed CUTLASS"):
        native.cutlass_identity(destination)


@pytest.mark.parametrize("mismatch", ["commit", "tree"])
def test_lock_mismatch_does_not_replace_installed_sources(source_bundle, mismatch):
    destination, bundle, commit, tree = source_bundle
    before = git(destination, "rev-parse", "HEAD")
    if mismatch == "commit":
        commit = "0" * 40
    else:
        tree = "0" * 40
    result = subprocess.run(
        [
            "bash",
            str(RECIPE / "install_source_bundle.sh"),
            str(bundle),
            commit,
            tree,
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert git(destination, "rev-parse", "HEAD") == before
    assert (destination / "module.py").read_text() == "value = 1\n"
    assert (destination / "obsolete.py").read_text() == "value = 2\n"
