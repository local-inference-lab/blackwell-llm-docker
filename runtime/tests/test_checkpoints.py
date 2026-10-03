"""Checkpoint variants: FP4-CSF by default, the original checkpoint on request,
and the serving files that let vLLM read an FP4-CSF repository."""

import json
import sys
import types

import pytest

from runtime import launcher
from runtime.launcher import ConfigError, prepare_csf_checkpoint, resolve

CSF = {
    "qwen38-flash-next": (
        "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-CSF",
        "4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e",
        "nvfp4_csf",
        "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
    ),
    "glm53-flash": (
        "local-inference-lab/GLM-5.3-Flash-NVFP4-CSF",
        "20f4777422f833c48b67bb0e554e1bc61dc55ca1",
        "nvfp4_csf",
        "local-inference-lab/GLM-5.3-Flash-NVFP4",
    ),
    "ds41-flash": (
        "local-inference-lab/DeepSeek-V4.1-Flash-MXFP4-CSF",
        "872da235166458bd6ffa9ee3f3c5c4771b63159c",
        "mxfp4_csf",
        "deepseek-ai/DeepSeek-V4.1-Flash",
    ),
}


@pytest.mark.parametrize("profile_id", sorted(CSF))
def test_profiles_serve_their_csf_checkpoint_by_default(profile_id):
    model, revision, method, _ = CSF[profile_id]
    plan = resolve(profile_id, "rtx-pro-6000-pcie", env={})

    assert plan.values["checkpoint"] == "csf"
    assert plan.values["model"] == model
    assert plan.values["revision"] == revision
    assert plan.values["quantization"] == plan.values["load-format"] == method
    assert plan.origins["model"] == "checkpoint:csf"
    assert "--checkpoint" not in plan.argv


@pytest.mark.parametrize("profile_id", sorted(CSF))
def test_original_checkpoint_keeps_the_profile_settings(profile_id):
    _, _, _, original = CSF[profile_id]
    plan = resolve(profile_id, "rtx-pro-6000-pcie", env={"CHECKPOINT": "original"})

    assert plan.values["model"] == original
    assert "revision" not in plan.values
    assert plan.values["load-format"] == "instanttensor"
    assert plan.values.get("quantization") in (None, "modelopt_mixed")


def test_a_model_naming_a_variant_selects_it():
    """Generated Compose files pass MODEL explicitly."""
    csf, revision, _, original = CSF["qwen38-flash-next"]
    by_csf = resolve("qwen38-flash-next", env={"MODEL": csf})
    by_original = resolve("qwen38-flash-next", env={"MODEL": original})

    assert by_csf.values["checkpoint"] == "csf"
    assert by_csf.values["revision"] == revision
    assert by_csf.values["load-format"] == "nvfp4_csf"
    assert by_original.values["checkpoint"] == "original"
    assert by_original.origins["checkpoint"] == "derived:model"
    assert by_original.values["quantization"] == "modelopt_mixed"


def test_another_model_or_revision_keeps_the_profile_settings(tmp_path):
    """A custom checkpoint, or a revision of the original repository such as
    the QAD export, is served with the profile's own settings."""
    custom = resolve("qwen38-flash-next", env={"MODEL": str(tmp_path)})
    qad = resolve("qwen38-flash-next", env={"MODEL_REVISION": "qad-step5500-ple1000"})

    for plan in (custom, qad):
        assert "checkpoint" not in plan.values
        assert plan.values["quantization"] == "modelopt_mixed"
        assert plan.values["load-format"] == "instanttensor"
    assert qad.values["model"] == CSF["qwen38-flash-next"][3]
    assert qad.values["revision"] == "qad-step5500-ple1000"


def test_checkpoint_with_a_local_copy_reads_it_as_csf(tmp_path):
    plan = resolve("glm53-flash", env={"MODEL": str(tmp_path), "CHECKPOINT": "csf"})

    assert plan.values["model"] == str(tmp_path)
    assert plan.values["load-format"] == "nvfp4_csf"
    # The pinned revision belongs to the Hub checkpoint, not to the copy.
    assert "revision" not in plan.values


def test_presets_that_choose_their_checkpoint_keep_it():
    spark = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-spark-tp2")
    tp3 = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp3")
    qwen_tp2 = resolve("qwen38-flash-next", "rtx-pro-6000-pcie", preset="qwen38-tp2")

    assert spark.values["model"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
    assert spark.values["quantization"] == "modelopt_mixed"
    assert "checkpoint" not in spark.values
    assert tp3.values["checkpoint"] == "original"
    assert tp3.values["model"] == CSF["glm53-flash"][3]
    assert qwen_tp2.values["checkpoint"] == "csf"


@pytest.mark.parametrize(
    "env",
    [
        {"CHECKPOINT": "csf", "MOE_BACKEND": "marlin"},
        {"CHECKPOINT": "csf", "ENABLE_EXPERT_PARALLEL": "1"},
        {"CHECKPOINT": "csf", "QUANTIZATION": "modelopt_mixed"},
    ],
)
def test_csf_needs_b12x_experts_without_expert_parallelism(env):
    with pytest.raises(ConfigError, match="FP4-CSF"):
        resolve("glm53-flash", "rtx-pro-6000-pcie", env=env)


def test_profiles_without_variants_reject_a_checkpoint():
    with pytest.raises(ConfigError, match="no csf checkpoint variant"):
        resolve("mimo26-flash", env={"CHECKPOINT": "csf"})
    with pytest.raises(ConfigError, match="checkpoint must be one of"):
        resolve("qwen38-flash-next", env={"CHECKPOINT": "native"})


def _csf_checkpoint(root, schema, quantization_config, nested=False, shard=b"xyz"):
    (root / "metadata").mkdir(parents=True)
    (root / "tensors").mkdir()
    (root / "tensors" / "model-00001.safetensors").write_bytes(shard)
    (root / "build-contract.json").write_text("{}")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "metadata_sha256": {"config.json": "0", "tokenizer.json": "0"},
                "shards": [{"file": "model-00001.safetensors", "target_file_bytes": 3}],
            }
        )
    )
    config = {"architectures": ["X"], "quantization_config": quantization_config}
    if nested:
        config = {"architectures": ["X"], "text_config": config}
    (root / "metadata" / "config.json").write_text(json.dumps(config))
    (root / "metadata" / "tokenizer.json").write_text("{}")
    return root


def _fake_hub(monkeypatch, snapshot_download, hf_hub_download=None):
    """A stand-in huggingface_hub: the runtime CI has none, and the tests
    must not reach the Hub or a real cache."""

    class LocalEntryNotFoundError(Exception):
        pass

    def no_manifest_download(*args, **kwargs):
        raise AssertionError("unexpected manifest download")

    hub = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    hub.errors = errors
    hub.snapshot_download = snapshot_download
    hub.hf_hub_download = hf_hub_download or no_manifest_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return LocalEntryNotFoundError


def _serving_config(plan):
    serving = plan.values["model"]
    assert plan.argv[4] == serving
    return json.loads((launcher.Path(serving) / "config.json").read_text())


def test_nvfp4_csf_serving_files_name_the_reader_and_the_root(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    source = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}
    root = _csf_checkpoint(
        tmp_path / "qwen", "lil-nvfp4-csf-checkpoint/1", source, nested=True
    )
    plan = resolve("qwen38-flash-next", env={"MODEL": str(root), "CHECKPOINT": "csf"})

    prepare_csf_checkpoint(plan)

    config = _serving_config(plan)
    assert config["text_config"]["quantization_config"] == {
        "quant_method": "nvfp4_csf",
        "format_version": 1,
        "checkpoint_root": str(root),
        "source_quantization_config": source,
    }
    assert (launcher.Path(plan.values["model"]) / "tokenizer.json").is_file()
    assert launcher.Path(plan.values["model"]).name.startswith("qwen-")
    assert "revision" not in plan.values
    assert "revision" not in plan.values["speculative-config"]


def test_mxfp4_csf_serving_files_keep_the_source_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    source = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    root = _csf_checkpoint(tmp_path / "ds41", "lil-mxfp4-csf-checkpoint/1", source)
    snapshots = []

    def snapshot_download(repository, revision=None, local_files_only=False):
        snapshots.append((repository, revision, local_files_only))
        return str(root)

    _fake_hub(monkeypatch, snapshot_download)
    plan = resolve("ds41-flash", "rtx-pro-6000-pcie", env={})

    prepare_csf_checkpoint(plan)

    # A complete cached snapshot is used without the network.
    assert snapshots == [(*CSF["ds41-flash"][:2], True)]
    assert _serving_config(plan)["quantization_config"] == {
        "quant_method": "mxfp4_csf",
        "weight_block_size": [128, 128],
        "format_version": 1,
        "checkpoint_root": str(root),
    }
    assert plan.origins["model"].startswith("resolved:FP4-CSF")
    assert launcher.Path(plan.values["model"]).name.startswith(
        "DeepSeek-V4.1-Flash-MXFP4-CSF-"
    )


@pytest.mark.parametrize(
    "schema,shard,match",
    [
        (
            "lil-mxfp4-csf-checkpoint/1",
            b"xyz",
            "is a mxfp4_csf checkpoint, not nvfp4_csf",
        ),
        ("lil-x4t-checkpoint/1", b"xyz", "FP4-CSF schema"),
        ("lil-nvfp4-csf-checkpoint/1", b"x", "incomplete.*tensors/model-00001"),
        (None, b"", "not an FP4-CSF checkpoint"),
    ],
)
def test_csf_serving_rejects_other_checkpoints(
    tmp_path, monkeypatch, schema, shard, match
):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    root = tmp_path / "checkpoint"
    if schema is None:
        root.mkdir()
        (root / "config.json").write_text("{}")
    else:
        _csf_checkpoint(root, schema, {"quant_method": "modelopt"}, shard=shard)
    plan = resolve("glm53-flash", env={"MODEL": str(root), "CHECKPOINT": "csf"})

    with pytest.raises(ConfigError, match=match):
        prepare_csf_checkpoint(plan)


def test_original_checkpoints_are_served_as_they_are(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    plan = resolve("glm53-flash", env={"CHECKPOINT": "original"})
    argv = list(plan.argv)

    prepare_csf_checkpoint(plan)

    assert plan.argv == argv
    assert not (tmp_path / "serving").exists()


def test_hub_checkpoints_download_only_after_the_manifest_matches(
    tmp_path, monkeypatch
):
    """An incomplete cache downloads the rest; a repository that is not FP4-CSF
    is refused before its weights are downloaded."""
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    partial = _csf_checkpoint(
        tmp_path / "partial", "lil-nvfp4-csf-checkpoint/1", {}, shard=b"x"
    )
    complete = _csf_checkpoint(tmp_path / "complete", "lil-nvfp4-csf-checkpoint/1", {})
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "lil-x4t-checkpoint/1"}))
    calls = []

    def snapshot_download(repository, revision=None, local_files_only=False):
        calls.append(("snapshot", local_files_only))
        if repository == "org/other":
            raise not_cached("not cached")
        return str(partial if local_files_only else complete)

    def hf_hub_download(repository, filename, revision=None):
        calls.append(("manifest", filename))
        return str(other if repository == "org/other" else complete / filename)

    not_cached = _fake_hub(monkeypatch, snapshot_download, hf_hub_download)

    plan = resolve("glm53-flash", env={"MODEL": "org/csf", "CHECKPOINT": "csf"})
    prepare_csf_checkpoint(plan)
    assert calls == [
        ("snapshot", True),
        ("manifest", "manifest.json"),
        ("snapshot", False),
    ]
    assert _serving_config(plan)["quantization_config"]["checkpoint_root"] == str(
        complete
    )

    calls.clear()
    plan = resolve("glm53-flash", env={"MODEL": "org/other", "CHECKPOINT": "csf"})
    with pytest.raises(ConfigError, match="FP4-CSF schema"):
        prepare_csf_checkpoint(plan)
    assert calls == [("snapshot", True), ("manifest", "manifest.json")]


def test_images_without_csf_readers_serve_the_original_checkpoint():
    """dk main serves beta and canonical images; an image whose vLLM predates
    the FP4-CSF readers keeps working on the original checkpoint."""
    plan = resolve("glm53-flash", env={}, csf_formats=frozenset())

    assert plan.values["checkpoint"] == "original"
    assert plan.values["model"] == CSF["glm53-flash"][3]
    assert plan.values["quantization"] == "modelopt_mixed"
    assert any("cannot read FP4-CSF" in warning for warning in plan.warnings)
    supported = resolve(
        "glm53-flash", env={}, csf_formats=frozenset(launcher.CSF_FORMATS)
    )
    assert supported.values["checkpoint"] == "csf"


@pytest.mark.parametrize(
    "env", [{"CHECKPOINT": "csf"}, {"MODEL": CSF["qwen38-flash-next"][0]}]
)
def test_an_explicit_csf_choice_needs_the_readers(env):
    with pytest.raises(ConfigError, match="cannot read FP4-CSF"):
        resolve("qwen38-flash-next", env=env, csf_formats=frozenset({"mxfp4_csf"}))


def test_installed_csf_formats_reads_the_loader_registry(tmp_path, monkeypatch):
    package = tmp_path / "vllm" / "model_executor" / "model_loader"
    package.mkdir(parents=True)
    (tmp_path / "vllm" / "__init__.py").write_text("raise RuntimeError('no import')\n")
    (package / "__init__.py").write_text('_LOADERS = {"nvfp4_csf": 1, "auto": 2}\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    assert launcher.installed_csf_formats() == frozenset({"nvfp4_csf"})


def test_csf_identity_follows_the_manifest_not_the_location(tmp_path, monkeypatch):
    """LMCache namespaces and the shared PLE table key on the checkpoint
    content; the serving files are generated, so the manifest names it."""
    from runtime import replicas
    from runtime.cache import resolve_identity

    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    roots = [
        _csf_checkpoint(tmp_path / name, "lil-nvfp4-csf-checkpoint/1", {})
        for name in ("a", "b")
    ]
    plans = []
    for root in roots:
        plan = resolve(
            "glm53-flash",
            env={"MODEL": str(root), "CHECKPOINT": "csf", "CACHE_MODE": "lmcache"},
        )
        prepare_csf_checkpoint(plan)
        plans.append(plan)
    assert plans[0].values["model"] != plans[1].values["model"]
    assert plans[0].target_identity == plans[1].target_identity

    helper = tmp_path / "identity.py"
    helper.write_text(
        "def resolve_checkpoint(model, revision):\n"
        "    raise AssertionError('generated files were hashed')\n"
    )
    resolve_identity(plans[0], {"runtime_lock_sha256": "b" * 64}, helper)
    identity = plans[0].values["kv-transfer-config"]["kv_connector_extra_config"][
        "lmcache.mp.checkpoint_identity"
    ]
    assert identity["target_revision"] == plans[0].target_identity["identity"]
    assert replicas.checkpoint_identity(plans[1], helper) == identity["target_revision"]

    manifest = roots[1] / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"0"', '"1"'))
    changed = resolve("glm53-flash", env={"MODEL": str(roots[1]), "CHECKPOINT": "csf"})
    prepare_csf_checkpoint(changed)
    assert changed.target_identity != plans[0].target_identity
