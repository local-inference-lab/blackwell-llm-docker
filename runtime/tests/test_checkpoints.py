"""Checkpoint variants: FP4-CSF by default, the original checkpoint on request,
and the serving files that let vLLM read an FP4-CSF repository."""

import json
import re
import sys
import types
from pathlib import Path
from typing import NamedTuple

import pytest

from runtime import launcher
from runtime.launcher import ConfigError, prepare_csf_checkpoint, profile, resolve


class Variants(NamedTuple):
    csf: str
    method: str
    original: str
    original_revision: str | None


CSF = {
    "qwen38-flash-next": Variants(
        "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD",
        "nvfp4_csf",
        "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
        None,
    ),
    "glm53-flash": Variants(
        "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
        "nvfp4_csf",
        "local-inference-lab/GLM-5.3-Flash-NVFP4",
        None,
    ),
    "glm53": Variants(
        "local-inference-lab/GLM-5.3-NVFP4-CSF",
        "nvfp4_csf",
        "local-inference-lab/GLM-5.3-NVFP4",
        "1f3bb90cbff7f63cfb5c04c0ab723e6f75b633a8",
    ),
    "ds41-flash": Variants(
        "local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF",
        "mxfp4_csf",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        None,
    ),
    "ds4-flash": Variants(
        "local-inference-lab/DeepSeek-V4-Flash-0731-lossless-CSF",
        "mxfp4_csf",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    ),
    "ds4-vision": Variants(
        "local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF",
        "mxfp4_csf",
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "6821d6ad3681a4b137b066b76094fa82ebd0a380",
    ),
}
DS4 = ("ds4-flash", "ds4-vision")


def csf_revision(profile_id):
    """The FP4-CSF revision the profile pins, or None while it is unpinned."""
    variant = profile("model", profile_id)["checkpoints"]["csf"]
    return variant["options"].get("revision")


@pytest.mark.parametrize("profile_id", sorted(CSF))
def test_profiles_serve_their_csf_checkpoint_by_default(profile_id):
    variants = CSF[profile_id]
    plan = resolve(profile_id, "rtx-pro-6000-pcie", env={})

    assert plan.values["checkpoint"] == "csf"
    assert plan.values["model"] == variants.csf
    assert plan.values.get("revision") == csf_revision(profile_id)
    assert plan.values["quantization"] == plan.values["load-format"] == variants.method
    assert plan.origins["model"] == "checkpoint:csf"
    assert "--checkpoint" not in plan.argv
    # The served API name does not change with the checkpoint.
    defaults = profile("model", profile_id)["defaults"]
    assert plan.values["served-model-name"] == defaults["served-model-name"]


@pytest.mark.parametrize("profile_id", sorted(CSF))
def test_csf_revisions_are_full_commits_once_pinned(profile_id):
    revision = csf_revision(profile_id)
    assert revision is None or re.fullmatch(r"[0-9a-f]{40}", revision)


@pytest.mark.parametrize("profile_id", sorted(CSF))
def test_original_checkpoint_keeps_the_profile_settings(profile_id):
    variants = CSF[profile_id]
    plan = resolve(profile_id, "rtx-pro-6000-pcie", env={"CHECKPOINT": "original"})

    assert plan.values["model"] == variants.original
    assert plan.values.get("revision") == variants.original_revision
    assert plan.values["load-format"] == "instanttensor"
    # GLM-5.3 (744B) is a single-algorithm ModelOpt NVFP4 checkpoint.
    assert plan.values.get("quantization") in (None, "modelopt_mixed", "modelopt_fp4")


def test_a_model_naming_a_variant_selects_it():
    """Generated Compose files pass MODEL explicitly."""
    csf, _, original, _ = CSF["qwen38-flash-next"]
    by_csf = resolve("qwen38-flash-next", env={"MODEL": csf})
    by_original = resolve("qwen38-flash-next", env={"MODEL": original})

    assert by_csf.values["checkpoint"] == "csf"
    assert by_csf.values.get("revision") == csf_revision("qwen38-flash-next")
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
    assert qad.values["model"] == CSF["qwen38-flash-next"].original
    assert qad.values["revision"] == "qad-step5500-ple1000"


def test_checkpoint_with_a_local_copy_reads_it_as_csf(tmp_path):
    plan = resolve("glm53-flash", env={"MODEL": str(tmp_path), "CHECKPOINT": "csf"})

    assert plan.values["model"] == str(tmp_path)
    assert plan.values["load-format"] == "nvfp4_csf"
    # The pinned revision belongs to the Hub checkpoint, not to the copy.
    assert "revision" not in plan.values


def test_presets_that_choose_their_checkpoint_keep_it():
    tp2 = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp2")
    tp3 = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp3")
    qwen_tp2 = resolve("qwen38-flash-next", "rtx-pro-6000-pcie", preset="qwen38-tp2")

    assert tp2.values["model"] == (
        "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"
    )
    assert tp2.values["quantization"] == "nvfp4_csf"
    assert "checkpoint" not in tp2.values
    assert tp3.values["checkpoint"] == "original"
    assert tp3.values["model"] == CSF["glm53-flash"].original
    assert qwen_tp2.values["checkpoint"] == "csf"


def test_the_dflash2_drafter_keeps_the_default_load_format():
    """The DFlash2 drafter is a plain checkpoint; without its own load config
    vLLM would read it with the target's FP4-CSF loader and refuse it."""
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={"SPECULATOR": "dflash2"})
    assert plan.values["load-format"] == "nvfp4_csf"
    spec = json.loads(plan.argv[plan.argv.index("--speculative-config") + 1])
    assert spec["method"] == "dflash"
    assert spec["draft_load_config"] == {"load_format": "auto"}


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

    class OfflineModeIsEnabled(ConnectionError):
        pass

    class EntryNotFoundError(Exception):
        pass

    def no_manifest_download(*args, **kwargs):
        raise AssertionError("unexpected manifest download")

    hub = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    errors.OfflineModeIsEnabled = OfflineModeIsEnabled
    errors.EntryNotFoundError = EntryNotFoundError
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
    assert snapshots == [(CSF["ds41-flash"].csf, csf_revision("ds41-flash"), True)]
    assert _serving_config(plan)["quantization_config"] == {
        "quant_method": "mxfp4_csf",
        "weight_block_size": [128, 128],
        "format_version": 1,
        "checkpoint_root": str(root),
    }
    assert plan.origins["model"].startswith("resolved:FP4-CSF")
    assert launcher.Path(plan.values["model"]).name.startswith(
        "DeepSeek-V4.1-Flash-lossless-CSF-"
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
    assert plan.values["model"] == CSF["glm53-flash"].original
    assert plan.values["quantization"] == "modelopt_mixed"
    assert any("cannot read FP4-CSF" in warning for warning in plan.warnings)
    supported = resolve(
        "glm53-flash", env={}, csf_formats=frozenset(launcher.CSF_FORMATS)
    )
    assert supported.values["checkpoint"] == "csf"


@pytest.mark.parametrize(
    "env", [{"CHECKPOINT": "csf"}, {"MODEL": CSF["qwen38-flash-next"].csf}]
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


def _canonical_vllm(tmp_path, monkeypatch):
    """A vLLM whose ModelOpt recipes declare NVFP4-CSF scales, without the
    dedicated NVFP4-CSF load format."""
    loaders = tmp_path / "vllm" / "model_executor" / "model_loader"
    quantization = tmp_path / "vllm" / "model_executor" / "layers" / "quantization"
    loaders.mkdir(parents=True)
    quantization.mkdir(parents=True)
    (tmp_path / "vllm" / "__init__.py").write_text("raise RuntimeError('no import')\n")
    (loaders / "__init__.py").write_text('_LOADERS = {"mxfp4_csf": 1, "auto": 2}\n')
    (quantization / "modelopt.py").write_text('ENCODING = "weight_scale_encoding"\n')
    monkeypatch.syspath_prepend(str(tmp_path))


def test_recipe_csf_reader_serves_hf_layout_nvfp4_csf(tmp_path, monkeypatch):
    _canonical_vllm(tmp_path / "site", monkeypatch)

    assert launcher.installed_nvfp4_csf_recipes()
    assert launcher.installed_csf_formats() == frozenset({"mxfp4_csf", "nvfp4_csf"})


def test_recipe_csf_reader_loads_hf_layout_through_modelopt(tmp_path, monkeypatch):
    """Without the dedicated load format, the ModelOpt recipes in config.json
    mark the compressed scales and the standard loader reads the snapshot."""
    _canonical_vllm(tmp_path / "site", monkeypatch)
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    root = _hf_layout_checkpoint(tmp_path / "snapshot")
    _fake_hub(monkeypatch, lambda *a, **kw: str(root))
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    prepare_csf_checkpoint(plan)

    assert plan.values["model"] == str(root)
    assert plan.argv[plan.argv.index("--load-format") + 1] == "instanttensor"
    assert plan.argv[plan.argv.index("--quantization") + 1] == "modelopt_mixed"


def _csf_container(root):
    """An NVFP4-CSF container: metadata/, tensors/ and a manifest."""
    (root / "metadata").mkdir(parents=True)
    (root / "tensors").mkdir()
    (root / "build-contract.json").write_text("{}")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lil-nvfp4-csf-checkpoint/1",
                "metadata_sha256": {},
                "shards": [],
            }
        )
    )
    group = {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False}
    (root / "metadata" / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "config_groups": {
                        "group_0": {
                            "weights": group,
                            "input_activations": group,
                            "targets": ["Linear"],
                        }
                    },
                    "ignore": ["lm_head", "model.layers.3.self_attn*"],
                },
            }
        )
    )
    (root / "metadata" / "hf_quant_config.json").write_text("{}")
    (root / "metadata" / "tokenizer.json").write_text("{}")
    expert = "model.layers.3.mlp.experts.0.up_proj"
    header = {
        f"{expert}.weight": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        f"{expert}.weight_scale.nvfp4_csf_fixed": {
            "dtype": "U8",
            "shape": [2],
            "data_offsets": [2, 4],
        },
        f"{expert}.weight_scale.nvfp4_csf_exceptions": {
            "dtype": "U32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.3.self_attn.o_proj.weight": {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [8, 12],
        },
    }
    raw = json.dumps(header).encode()
    (root / "tensors" / "model-00001-of-00001.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + bytes(12)
    )
    return root


def test_recipe_csf_reader_serves_a_container_through_recipes(tmp_path, monkeypatch):
    """An NVFP4-CSF container already stores the compressed scale tensors; the
    launcher adds the recipes and the stored-name index the standard loader
    needs and links the shards."""
    _canonical_vllm(tmp_path / "site", monkeypatch)
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    root = _csf_container(tmp_path / "snapshot")
    _fake_hub(monkeypatch, lambda *a, **kw: str(root))
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    prepare_csf_checkpoint(plan)

    serving = Path(plan.values["model"])
    assert serving.parent == tmp_path / "serving"
    assert plan.argv[plan.argv.index("--quantization") + 1] == "modelopt_mixed"
    assert plan.argv[plan.argv.index("--load-format") + 1] == "instanttensor"
    assert "--revision" not in plan.argv
    recipes = json.loads((serving / "config.json").read_text())["quantization_config"]
    assert recipes["quant_algo"] == "MIXED_PRECISION"
    assert recipes["quantized_layers"] == {
        "model.layers.3.mlp.experts": {
            "group_size": 16,
            "quant_algo": "NVFP4",
            "weight_scale_encoding": "csf",
        }
    }
    shard = "tensors/model-00001-of-00001.safetensors"
    index = json.loads((serving / "model.safetensors.index.json").read_text())
    assert (
        index["weight_map"][
            "model.layers.3.mlp.experts.0.up_proj.weight_scale.nvfp4_csf_fixed"
        ]
        == shard
    )
    assert (serving / shard).is_file()
    assert (serving / "tokenizer.json").is_file()
    assert not (serving / "hf_quant_config.json").exists()


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


def _pin_csf_revisions(monkeypatch, revision="a" * 40):
    """Profiles as they read once the FP4-CSF revisions are pinned."""
    read = launcher.profile

    def pinned(kind, identifier):
        result = read(kind, identifier)
        if kind == "model" and "checkpoints" in result:
            result["checkpoints"]["csf"]["options"].setdefault("revision", revision)
        return result

    monkeypatch.setattr(launcher, "profile", pinned)


@pytest.mark.parametrize("profile_id", DS4)
def test_ds4_csf_needs_the_deepseek_v4_flash_loader_family(profile_id):
    """An image whose MXFP4-CSF loader predates DeepSeek-V4-Flash serves the
    original checkpoint, as an image without FP4-CSF readers does."""
    variants = CSF[profile_id]
    formats = frozenset(launcher.CSF_FORMATS)
    older = resolve(
        profile_id,
        env={},
        csf_formats=formats,
        csf_families=frozenset({"deepseek_v41", "kimi_k3"}),
    )
    assert older.values["checkpoint"] == "original"
    assert older.origins["checkpoint"].startswith("derived:installed vLLM")
    assert older.values["model"] == variants.original
    assert older.values["revision"] == older.values["code-revision"]
    assert older.values["revision"] == variants.original_revision
    assert older.values["load-format"] == "instanttensor"
    assert "quantization" not in older.values
    assert any(
        "cannot read FP4-CSF checkpoints of the deepseek_v4_flash family" in warning
        for warning in older.warnings
    )
    with pytest.raises(ConfigError, match="deepseek_v4_flash family"):
        resolve(
            profile_id,
            env={"CHECKPOINT": "csf"},
            csf_formats=formats,
            csf_families=frozenset({"deepseek_v41"}),
        )
    current = resolve(
        profile_id,
        env={},
        csf_formats=formats,
        csf_families=frozenset({"deepseek_v41", "deepseek_v4_flash"}),
    )
    assert current.values["checkpoint"] == "csf"
    assert current.values["model"] == variants.csf
    without_readers = resolve(profile_id, env={}, csf_formats=frozenset())
    assert without_readers.values["checkpoint"] == "original"


def test_formats_without_a_declared_family_need_only_the_reader():
    plan = resolve(
        "ds41-flash",
        env={},
        csf_formats=frozenset(launcher.CSF_FORMATS),
        csf_families=frozenset(),
    )
    assert plan.values["checkpoint"] == "csf"


def test_installed_csf_families_reads_the_loader_tables(tmp_path, monkeypatch):
    package = tmp_path / "vllm" / "model_executor" / "model_loader"
    package.mkdir(parents=True)
    (tmp_path / "vllm" / "__init__.py").write_text("raise RuntimeError('no import')\n")
    (package / "__init__.py").write_text(
        '_LOADERS = {"mxfp4_csf": 1, "nvfp4_csf": 2}\n'
    )
    (package / "mxfp4_csf_loader.py").write_text(
        "# DeepSeek-V4-Flash and its vision variant\n"
        "FAMILIES = {\n"
        '    "deepseek_v41": (384, 5120, 2304, range(40)),\n'
        '    "deepseek_v4_flash": (256, 4096, 2048, range(43)),\n'
        "}\n"
        'OTHER = {"not_a_family": 1}\n'
    )
    (package / "nvfp4_csf_loader.py").write_text(
        'FAMILIES = {\n    "glm53_nvfp4": (288, 4096, 2048, range(3, 45)),\n}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert launcher.installed_csf_families() == frozenset(
        {"deepseek_v41", "deepseek_v4_flash", "glm53_nvfp4"}
    )
    (package / "mxfp4_csf_loader.py").write_text(
        'FAMILIES = {\n    "deepseek_v41": (384, 5120, 2304),\n}\n'
    )
    assert "deepseek_v4_flash" not in launcher.installed_csf_families()


@pytest.mark.parametrize("profile_id", DS4)
def test_csf_checkpoints_get_no_derived_code_revision(profile_id, monkeypatch):
    """DS4 trusts remote code; its FP4-CSF serving files carry the checkpoint's
    Hugging Face files, so a Hub code revision must not reach vLLM."""
    _pin_csf_revisions(monkeypatch)
    plan = resolve(profile_id, env={})
    assert plan.values["trust-remote-code"] is True
    assert plan.values["revision"] == (csf_revision(profile_id) or "a" * 40)
    assert "code-revision" not in plan.values
    explicit = resolve(profile_id, env={"MODEL_CODE_REVISION": "c" * 40})
    assert explicit.values["code-revision"] == "c" * 40
    original = resolve(profile_id, env={"CHECKPOINT": "original"})
    assert original.values["code-revision"] == CSF[profile_id].original_revision


@pytest.mark.parametrize("profile_id", DS4)
@pytest.mark.parametrize("mode", ["dspark", "mtp"])
def test_ds4_csf_drafters_follow_the_serving_files(
    tmp_path, monkeypatch, profile_id, mode
):
    """The DSpark and MTP drafters live in the target checkpoint; with FP4-CSF
    they are read from the same serving files, without Hub revisions."""
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    _pin_csf_revisions(monkeypatch)
    root = _csf_checkpoint(
        tmp_path / "ds4",
        "lil-mxfp4-csf-checkpoint/1",
        {"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    _fake_hub(monkeypatch, lambda repository, **kwargs: str(root))
    plan = resolve(profile_id, "rtx-pro-6000-pcie", env={"SPECULATOR": mode})
    spec = plan.values["speculative-config"]
    assert spec["method"] == mode and spec["revision"] == plan.values["revision"]
    if mode == "dspark":
        assert spec["model"] == CSF[profile_id].csf

    prepare_csf_checkpoint(plan)

    serving = plan.values["model"]
    assert serving.startswith(str(tmp_path / "serving"))
    assert "revision" not in plan.values and "code-revision" not in plan.values
    assert "revision" not in spec
    assert spec.get("model", serving) == serving
    assert "--revision" not in plan.argv and "--code-revision" not in plan.argv
    assert "--trust-remote-code" in plan.argv
    assert json.loads(plan.argv[plan.argv.index("--speculative-config") + 1]) == spec
    assert _serving_config(plan)["quantization_config"]["quant_method"] == "mxfp4_csf"


def test_a_preset_with_a_checkpoint_of_its_own_refuses_the_other_kind(monkeypatch):
    """CHECKPOINT chooses among the profile's checkpoints; a preset that names
    its own serves it, and a choice of the other kind is an error."""
    real = launcher.deployment_presets()
    own = {
        **real["glm53-tp2"],
        "options": {
            "model": "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark",
            "tensor-parallel-size": 2,
            "load-format": "safetensors",
        },
        "linked_options": {},
    }
    monkeypatch.setattr(
        launcher, "deployment_presets", lambda: {**real, "glm53-own-tp2": own}
    )

    def spark(env):
        return resolve(
            "glm53-flash", "rtx-pro-6000-pcie", preset="glm53-own-tp2", env=env
        )

    with pytest.raises(
        ConfigError,
        match="glm53-own-tp2 serves a non-CSF checkpoint of its own.*CHECKPOINT=csf",
    ):
        spark({"CHECKPOINT": "csf"})
    same_kind = spark({"CHECKPOINT": "original"})
    assert same_kind.values["model"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
    assert same_kind.values["load-format"] == "safetensors"
    assert "checkpoint" not in same_kind.values


def test_an_fp4_csf_checkpoint_outside_the_variants_needs_the_reader(tmp_path):
    env = {
        "MODEL": str(tmp_path),
        "QUANTIZATION": "nvfp4_csf",
        "LOAD_FORMAT": "nvfp4_csf",
    }
    custom = resolve("glm53-flash", env=env)
    assert "checkpoint" not in custom.values
    assert custom.values["load-format"] == "nvfp4_csf"
    with pytest.raises(ConfigError, match="cannot read nvfp4_csf checkpoints$"):
        resolve("glm53-flash", env=env, csf_formats=frozenset({"mxfp4_csf"}))


def test_the_glm_tp2_preset_reads_its_checkpoint_through_serving_files(
    tmp_path, monkeypatch
):
    """glm53-tp2 names a stored FP4-CSF checkpoint of its own; it is
    downloaded at the preset's revision and its MTP drafter follows it."""
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    root = _csf_checkpoint(
        tmp_path / "qad", "lil-nvfp4-csf-checkpoint/1", {"quant_method": "modelopt"}
    )
    downloads = []

    def snapshot_download(repository, revision=None, local_files_only=False):
        downloads.append((repository, revision, local_files_only))
        return str(root)

    _fake_hub(monkeypatch, snapshot_download)
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp2", env={})
    repository, revision = plan.values["model"], plan.values["revision"]

    prepare_csf_checkpoint(plan)

    assert repository == "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"
    assert downloads == [(repository, revision, True)]
    assert _serving_config(plan)["quantization_config"]["quant_method"] == "nvfp4_csf"
    assert "--revision" not in plan.argv and "--code-revision" not in plan.argv
    spec = json.loads(plan.argv[plan.argv.index("--speculative-config") + 1])
    assert "revision" not in spec and "model" not in spec
    assert spec["moe_backend"] == "b12x"


def _pinned_revisions():
    """Every FP4-CSF Hub revision a profile or preset pins."""
    pinned = set()
    for profile_id in CSF:
        options = profile("model", profile_id)["checkpoints"]["csf"]["options"]
        if options.get("revision"):
            pinned.add((options["model"], options["revision"]))
    presets = launcher.read_yaml(launcher.ROOT / "presets.yaml")["presets"]
    for preset in presets.values():
        options = preset.get("options") or {}
        if options.get("load-format") in launcher.CSF_FORMATS and options.get(
            "revision"
        ):
            pinned.add((options["model"], options["revision"]))
    return sorted(pinned)


def test_every_pinned_csf_revision_records_its_manifest():
    records = {
        repository: {
            **launcher.pinned_csf_manifests().get(repository, {}),
            **launcher.pinned_csf_contents().get(repository, {}),
        }
        for repository, _ in _pinned_revisions()
    }
    pinned = _pinned_revisions()

    assert pinned
    for repository, revision in pinned:
        assert re.fullmatch(r"[0-9a-f]{64}", records[repository][revision])


def _offline_hub(monkeypatch, tmp_path, repository, revision, snapshots, error=None):
    """A cache holding other revisions of the repository and no network."""
    cache = tmp_path / "hub"
    folder = cache / f"models--{repository.replace('/', '--')}" / "snapshots"
    for name, root in snapshots.items():
        folder.mkdir(parents=True, exist_ok=True)
        root.rename(folder / name)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    manifest = (folder / "same" / "manifest.json").read_bytes()
    monkeypatch.setattr(
        launcher,
        "pinned_csf_manifests",
        lambda: {repository: {revision: launcher.hashlib.sha256(manifest).hexdigest()}},
    )

    def snapshot_download(repository, revision=None, local_files_only=False):
        raise missing("revision not cached")

    def hf_hub_download(*args, **kwargs):
        raise error or missing("HF_HUB_OFFLINE")

    missing = _fake_hub(monkeypatch, snapshot_download, hf_hub_download)
    return folder


def test_offline_a_cached_revision_with_the_same_manifest_serves_the_pin(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    repository, revision = CSF["glm53-flash"].csf, csf_revision("glm53-flash")
    schema = "lil-nvfp4-csf-checkpoint/1"
    same = _csf_checkpoint(tmp_path / "a", schema, {})
    other = _csf_checkpoint(tmp_path / "b", schema, {}, shard=b"abc")
    manifest = other / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"0"', '"1"'))
    folder = _offline_hub(
        monkeypatch, tmp_path, repository, revision, {"other": other, "same": same}
    )
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    prepare_csf_checkpoint(plan)

    config = _serving_config(plan)
    assert config["quantization_config"]["checkpoint_root"] == str(folder / "same")
    assert "same FP4-CSF manifest" in capsys.readouterr().err


def test_offline_without_a_matching_cached_revision_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    repository, revision = CSF["glm53-flash"].csf, csf_revision("glm53-flash")
    schema = "lil-nvfp4-csf-checkpoint/1"
    same = _csf_checkpoint(tmp_path / "a", schema, {})
    folder = _offline_hub(monkeypatch, tmp_path, repository, revision, {"same": same})
    # An incomplete snapshot with the right manifest is not a substitute.
    (folder / "same" / "tensors" / "model-00001.safetensors").unlink()
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    with pytest.raises(Exception, match="HF_HUB_OFFLINE"):
        prepare_csf_checkpoint(plan)


def test_hub_errors_are_not_masked_by_a_cached_revision(tmp_path, monkeypatch):
    """A missing pinned revision or denied access fails even with a cached twin."""
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    repository, revision = CSF["glm53-flash"].csf, csf_revision("glm53-flash")
    same = _csf_checkpoint(tmp_path / "a", "lil-nvfp4-csf-checkpoint/1", {})
    _offline_hub(
        monkeypatch,
        tmp_path,
        repository,
        revision,
        {"same": same},
        error=OSError("404 Client Error: Revision Not Found"),
    )
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    with pytest.raises(OSError, match="Revision Not Found"):
        prepare_csf_checkpoint(plan)


def _hf_layout_checkpoint(root):
    """A Hugging Face-layout FP4-CSF checkpoint: recipes mark CSF expert scales."""
    (root / "tensors").mkdir(parents=True)
    (root / "tensors" / "model-00001-of-00001.safetensors").write_bytes(b"0")
    expert = "model.language_model.layers.3.mlp.experts"
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        expert: {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                            "weight_scale_encoding": "csf",
                        }
                    },
                }
            }
        )
    )
    shard = "tensors/model-00001-of-00001.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"{expert}.0.up_proj.weight_scale.nvfp4_csf_fixed": shard,
                    f"{expert}.0.up_proj.weight": shard,
                }
            }
        )
    )
    return root


def test_hf_layout_csf_checkpoint_is_served_from_its_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    repository = CSF["glm53-flash"].csf
    root = _hf_layout_checkpoint(tmp_path / "snapshot")
    _fake_hub(monkeypatch, lambda *a, **kw: str(root))
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    prepare_csf_checkpoint(plan)

    assert plan.values["model"] == str(root)
    assert repository in plan.origins["model"]
    assert not (tmp_path / "serving").exists()
    assert plan.argv[plan.argv.index("--load-format") + 1] == "nvfp4_csf"
    assert "--revision" not in plan.argv
    assert len(plan.target_identity["identity"]) == 64


def test_hf_layout_csf_checkpoint_needs_every_indexed_shard(tmp_path, monkeypatch):
    """An incomplete cached snapshot is not served; offline, the launch fails."""
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    root = _hf_layout_checkpoint(tmp_path / "snapshot")
    (root / "tensors" / "model-00001-of-00001.safetensors").unlink()
    hub = {}

    def offline(*args, **kwargs):
        raise hub["missing"]("HF_HUB_OFFLINE")

    hub["missing"] = _fake_hub(monkeypatch, lambda *a, **kw: str(root), offline)
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})

    with pytest.raises(hub["missing"], match="HF_HUB_OFFLINE"):
        prepare_csf_checkpoint(plan)


def test_hf_layout_index_cannot_name_a_shard_outside_the_checkpoint(tmp_path):
    root = _hf_layout_checkpoint(tmp_path / "snapshot")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["escape"] = "../outside.safetensors"
    (tmp_path / "outside.safetensors").write_bytes(b"0")
    (root / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ConfigError, match="outside the checkpoint"):
        launcher.csf_hf_missing_files(root)


def test_hf_layout_identity_covers_the_shard_contents(tmp_path):
    """Two snapshots with one index but different shard blobs differ."""
    digests = set()
    for blob in ("a" * 64, "b" * 64):
        root = _hf_layout_checkpoint(tmp_path / blob[0] / "snapshot")
        shard = root / "tensors" / "model-00001-of-00001.safetensors"
        blobs = tmp_path / blob[0] / "blobs"
        blobs.mkdir()
        (blobs / blob).write_bytes(shard.read_bytes())
        shard.unlink()
        shard.symlink_to(blobs / blob)
        digests.add(launcher.csf_hf_content_digest(root))
    assert len(digests) == 2


@pytest.mark.parametrize("profile_id", ["ds41-flash", *DS4, "glm53-flash"])
def test_a_local_csf_copy_selects_the_variant_that_reads_it(
    tmp_path, monkeypatch, profile_id
):
    """Read with the original settings, a local FP4-CSF copy loads without an
    error and serves wrong output, so its manifest picks the reader."""
    monkeypatch.setattr(launcher, "CSF_SERVING_ROOT", tmp_path / "serving")
    method = CSF[profile_id].method
    source = {"quant_method": "fp8"}
    root = _csf_checkpoint(
        tmp_path / "copy", f"lil-{method.split('_')[0]}-csf-checkpoint/1", source
    )
    plan = resolve(profile_id, "rtx-pro-6000-pcie", env={"MODEL": str(root)})

    assert plan.values["checkpoint"] == "csf"
    assert plan.values["quantization"] == plan.values["load-format"] == method
    assert "revision" not in plan.values
    if method == "mxfp4_csf":
        prepare_csf_checkpoint(plan)
        config = _serving_config(plan)["quantization_config"]
        assert config["quant_method"] == method
        assert config["checkpoint_root"] == str(root)


def test_a_local_hf_layout_csf_copy_selects_the_csf_variant(tmp_path):
    root = _hf_layout_checkpoint(tmp_path / "copy")
    plan = resolve("qwen38-flash-next", env={"MODEL": str(root)})

    assert plan.values["checkpoint"] == "csf"
    assert plan.values["load-format"] == "nvfp4_csf"


def test_the_original_checkpoint_settings_refuse_a_local_csf_copy(tmp_path):
    root = _csf_checkpoint(
        tmp_path / "copy", "lil-mxfp4-csf-checkpoint/1", {"quant_method": "fp8"}
    )
    with pytest.raises(
        ConfigError, match=r"CHECKPOINT=original \(environment:CHECKPOINT\) reads"
    ):
        resolve("ds41-flash", env={"MODEL": str(root), "CHECKPOINT": "original"})


def test_a_local_csf_copy_needs_a_variant_of_its_format(tmp_path):
    root = _csf_checkpoint(
        tmp_path / "copy", "lil-mxfp4-csf-checkpoint/1", {"quant_method": "fp8"}
    )
    with pytest.raises(ConfigError, match="has no checkpoint variant that reads"):
        resolve("glm53-flash", env={"MODEL": str(root)})


def test_a_local_csf_copy_needs_the_reader(tmp_path):
    root = _csf_checkpoint(
        tmp_path / "copy", "lil-mxfp4-csf-checkpoint/1", {"quant_method": "fp8"}
    )
    with pytest.raises(ConfigError, match=f"which MODEL={root} is$"):
        resolve("ds41-flash", env={"MODEL": str(root)}, csf_formats=frozenset())


def test_a_local_copy_of_another_format_keeps_the_profile_settings(tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "lil-x4t-checkpoint/1"}))
    plan = resolve("ds41-flash", env={"MODEL": str(root)})

    assert "checkpoint" not in plan.values
    assert plan.values["load-format"] == "instanttensor"


def test_plain_loader_settings_refuse_a_local_csf_container(tmp_path):
    root = _csf_checkpoint(
        tmp_path / "copy", "lil-mxfp4-csf-checkpoint/1", {"quant_method": "fp8"}
    )
    env = {"MODEL": str(root), "LOAD_FORMAT": "instanttensor"}
    with pytest.raises(ConfigError, match="load-format instanttensor"):
        resolve("ds41-flash", env=env)


def test_a_local_hf_layout_csf_copy_keeps_explicit_recipe_settings(tmp_path):
    """The ModelOpt recipes in its config.json mark the CSF scales, so the
    standard loader reads a Hugging Face-layout copy correctly."""
    root = _hf_layout_checkpoint(tmp_path / "copy")
    env = {
        "MODEL": str(root),
        "LOAD_FORMAT": "instanttensor",
        "QUANTIZATION": "modelopt_mixed",
    }
    plan = resolve("qwen38-flash-next", env=env)

    assert plan.values["load-format"] == "instanttensor"
    assert plan.values["quantization"] == "modelopt_mixed"
