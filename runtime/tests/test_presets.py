"""Deployment overlays must preserve model defaults and explicit override priority."""

import json
import os
import subprocess
import sys

import pytest

from runtime import launcher
from runtime.entrypoint import command
from runtime.launcher import ROOT, ConfigError, deployment_presets, resolve
from runtime.packaging import audit_image_metadata, owned_environment, payload_sources

TP2_KV = 7650410496
QAD_CHECKPOINT = "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"
QAD_REVISION = "dec48abd33efa73c3bb7c95b74eee10cad34f9be"


@pytest.fixture(autouse=True)
def threshold_image(monkeypatch):
    """The glm53-tp2 expectations name the token threshold of images whose vLLM
    still reads it, whatever vLLM the tests run next to."""
    monkeypatch.setattr(launcher, "installed_semantic_a4_prefill", lambda: False)


def tp2(env=None, **kwargs):
    return resolve(
        "glm53-flash",
        "rtx-pro-6000-pcie",
        preset="glm53-tp2",
        env=env or {},
        **kwargs,
    )


def test_tp2_settings_do_not_leak_into_tp4_or_qwen():
    before = resolve("glm53-flash", "rtx-pro-6000-pcie", env={}).public()
    tp2()
    after = resolve("glm53-flash", "rtx-pro-6000-pcie", env={}).public()
    assert after == before
    assert after["settings"]["tensor-parallel-size"]["value"] == 4
    assert after["settings"]["max-num-batched-tokens"]["value"] == 4096
    assert after["environment"]["NCCL_MIN_NCHANNELS"]["value"] == "16"
    assert "kv-cache-memory-bytes" not in after["settings"]
    qwen = resolve(
        "qwen38-flash-next", "rtx-pro-6000-pcie", preset="qwen38-tp2", env={}
    )
    assert qwen.values["tensor-parallel-size"] == 2
    assert qwen.values["max-num-batched-tokens"] == 6019
    assert qwen.environment["VLLM_PLE_CPU_OFFLOAD"] == "1"


@pytest.mark.parametrize("mode,width", [("off", 0), ("mtp", 3), ("dflash2", 7)])
def test_mode_override_rebuilds_the_speculator_instead_of_pinning_mtp(mode, width):
    plan = tp2(argv=["--mode", mode])
    assert plan.values["draft-tokens"] == width
    if width:
        assert plan.values["speculative-config"]["num_speculative_tokens"] == width
    else:
        assert "speculative-config" not in plan.values


def test_explicit_inputs_override_preset_values():
    plan = tp2(
        config={"options": {"max-num-seqs": 8}},
        cli_env={"NCCL_MIN_NCHANNELS": "4"},
        argv=["--max-num-seqs", "16", "--kv-cache-memory-bytes", "3758096384"],
    )
    assert plan.values["max-num-seqs"] == 16
    assert plan.values["kv-cache-memory-bytes"] == 3758096384
    assert plan.environment["NCCL_MIN_NCHANNELS"] == "4"
    assert plan.values["max-cudagraph-capture-size"] == 64
    assert plan.values["cudagraph-capture-sizes"] == [
        1,
        2,
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        52,
        56,
        60,
        64,
    ]


@pytest.mark.parametrize(
    "seqs,sizes",
    [
        (4, [1, 2, 4, 8, 12, 16]),
        (8, [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]),
    ],
)
def test_more_request_slots_keep_a_graph_for_every_verifier_batch(seqs, sizes):
    # MTP3 verifies four rows per request. Raising MAX_NUM_SEQS must not leave
    # 5-7 running requests (20/24/28 rows) without a CUDA graph.
    plan = tp2(cli_env={"MAX_NUM_SEQS": str(seqs)})
    assert plan.values["max-cudagraph-capture-size"] == sizes[-1]
    assert plan.values["cudagraph-capture-sizes"] == sizes


@pytest.mark.parametrize("mode,input_rows", [("mtp", 4096), ("dflash2", 4096 + 8 * 7)])
def test_lmcache_target_budget_follows_the_changed_prefill_budget(mode, input_rows):
    plan = tp2(
        argv=[
            "--mode",
            mode,
            "--cache-mode",
            "lmcache",
            "--max-num-batched-tokens",
            "4096",
        ]
    )
    assert plan.values["cache-object-tokens"] == 4096
    assert plan.values["max-num-scheduled-tokens"] == 4096
    assert plan.values["max-num-batched-tokens"] == input_rows
    assert plan.cache_service


def test_explicit_cache_object_override_is_not_replaced():
    with pytest.raises(ConfigError, match="must match"):
        tp2(argv=["--cache-mode", "lmcache", "--cache-object-tokens", "6144"])


def test_unknown_or_wrong_model_preset_is_rejected():
    with pytest.raises(ConfigError, match="Unknown deployment"):
        resolve("glm53-flash", preset="../../other", env={})
    with pytest.raises(ConfigError, match="different model"):
        resolve("ds4-flash", preset="glm53-tp2", env={})


def test_the_retired_spark_preset_names_its_replacement():
    with pytest.raises(
        ConfigError, match="PRESET=glm53-spark-tp2 was removed; use PRESET=glm53-tp2"
    ):
        resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-spark-tp2", env={})


def test_profile_cli_and_entrypoint_accept_preset_selection():
    assert "--help" not in command([], {"PRESET": "glm53-tp2"}, {})
    raw = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--preset",
            "glm53-tp2",
            "--print-config",
            "--port",
            "5213",
        ],
        cwd=ROOT.parent,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )
    result = json.loads(raw)
    assert result["settings"]["port"]["value"] == 5213
    assert result["settings"]["tensor-parallel-size"]["value"] == 2
    assert result["hardware"] == "rtx-pro-6000-pcie"


def test_presets_are_packaged_and_schema_checked():
    assert "presets.yaml" in payload_sources()
    assert deployment_presets()["glm53-tp2"]["profile"] == "glm53-flash"
    assert "glm53-spark-tp2" not in deployment_presets()


def test_foundation_defaults_are_below_presets_and_user_environment():
    default = resolve("glm53-flash", env={})
    assert default.environment["NCCL_NET_PLUGIN"] == "spcx"
    assert default.environment_origins["NCCL_NET_PLUGIN"] == "platform:foundation"
    assert tp2().environment["NCCL_NET_PLUGIN"] == "none"
    for value in ("spcx", "none", "operator-plugin"):
        plan = resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-tp2",
            env={"NCCL_NET_PLUGIN": value},
        )
        assert plan.environment["NCCL_NET_PLUGIN"] == value
        assert plan.environment_origins["NCCL_NET_PLUGIN"] == "environment"


def test_image_audit_covers_every_preset_environment_name():
    for preset in deployment_presets().values():
        for name in preset["environment"]:
            assert name in owned_environment()
            with pytest.raises(ConfigError, match="profile-owned"):
                audit_image_metadata(
                    {"Id": "sha256:" + "1" * 64, "Config": {"Env": [f"{name}=baked"]}}
                )


@pytest.mark.parametrize("amount", ["0", "-1"])
def test_fixed_kv_allocation_must_be_positive(amount):
    with pytest.raises(ConfigError, match="must be positive"):
        tp2(argv=["--kv-cache-memory-bytes", amount])


def test_glm_tp3_preset_selects_expert_parallel_and_tp3_tuning():
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp3", env={})
    argv = plan.argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "3"
    assert "--enable-expert-parallel" in argv
    assert "--enable-flashinfer-autotune" in argv
    assert argv[argv.index("--moe-backend") + 1] == "flashinfer_cutlass"
    assert argv[argv.index("--max-cudagraph-capture-size") + 1] == "32"
    assert plan.environment["VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE"] == "256KB"
    assert plan.environment["VLLM_GLM53_FP8_DENSE"] == "1"


def test_glm_tp3_preset_rejects_the_external_cache():
    with pytest.raises(ConfigError, match="TP2, TP4 or TP8"):
        resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-tp3",
            env={"CACHE_MODE": "lmcache"},
        )


SAVINGS = frozenset(
    {"VLLM_GLM53_EMBED_HOST", "VLLM_GLM53_VISION_MXFP8", "VLLM_SHARE_PYNCCL_COMMS"}
)


SAVINGS = frozenset(
    {"VLLM_GLM53_EMBED_HOST", "VLLM_GLM53_VISION_MXFP8", "VLLM_SHARE_PYNCCL_COMMS"}
)


def test_a_preset_keeps_its_fallback_recipe_on_an_older_vllm(monkeypatch):
    real = launcher.deployment_presets()
    fallback = {
        **real["glm53-tp2"],
        "vllm_fallback": {
            "requires_environment": sorted(SAVINGS),
            "options": {"max-num-seqs": 4, "kv-cache-memory-bytes": 4190109696},
        },
    }
    monkeypatch.setattr(
        launcher, "deployment_presets", lambda: {**real, "glm53-fallback-tp2": fallback}
    )

    def plan(env=None, **kwargs):
        return resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-fallback-tp2",
            env=env or {},
            **kwargs,
        )

    current = plan(vllm_environment=SAVINGS | {"VLLM_USE_V2_MODEL_RUNNER"})
    assert current.values["max-num-seqs"] == 8
    older = plan(vllm_environment=frozenset({"VLLM_USE_V2_MODEL_RUNNER"}))
    assert older.values["max-num-seqs"] == 4
    assert older.values["cudagraph-capture-sizes"] == [1, 2, 4, 8, 12, 16]
    assert not SAVINGS & set(older.environment)
    assert "fallback" in older.origins["max-num-seqs"]
    # Explicit choices still win over the fallback recipe.
    chosen = plan(
        env={"MAX_NUM_SEQS": "6", "KV_CACHE_MEMORY_BYTES": "3758096384"},
        vllm_environment=frozenset(),
    )
    assert chosen.values["max-num-seqs"] == 6
    assert chosen.values["kv-cache-memory-bytes"] == 3758096384


def test_installed_vllm_environment_reads_envs_without_importing(tmp_path, monkeypatch):
    from runtime.launcher import installed_vllm_environment

    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    (package / "envs.py").write_text(
        'environment_variables = {\n    "VLLM_GLM53_EMBED_HOST": lambda: False,\n}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vllm_environment() == frozenset({"VLLM_GLM53_EMBED_HOST"})


@pytest.mark.parametrize(
    "oracle,formats,expected",
    [
        (
            '    "b12x": Fp8MoeBackend.B12X_MXFP8,\n',
            '    MXFP8 = "mxfp8_e8m0_k32"\n',
            True,
        ),
        (
            '    "marlin": Fp8MoeBackend.MARLIN,\n',
            '    MXFP8 = "mxfp8_e8m0_k32"\n',
            False,
        ),
        (
            '    "b12x": Fp8MoeBackend.B12X_MXFP8,\n',
            '    MXFP4 = "fp4_e8m0_k32"\n',
            False,
        ),
        ('    "b12x": Fp8MoeBackend.B12X_MXFP8,\n', None, False),
    ],
)
def test_installed_b12x_mxfp8_moe_needs_both_packages(
    tmp_path, monkeypatch, oracle, formats, expected
):
    """The MXFP8 drafter stays on b12x only when vLLM maps b12x to its
    B12X_MXFP8 backend and B12X prepares the MXFP8 source format."""
    from runtime.launcher import installed_b12x_mxfp8_moe

    vllm_oracle = tmp_path / "vllm/model_executor/layers/fused_moe/oracle"
    vllm_oracle.mkdir(parents=True)
    (tmp_path / "vllm/__init__.py").write_text(
        "raise RuntimeError('must not import')\n"
    )
    (vllm_oracle / "mxfp8.py").write_text(oracle)
    b12x = tmp_path / "b12x"
    b12x.mkdir()
    (b12x / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    if formats is not None:
        (b12x / "moe/fused_moe").mkdir(parents=True)
        (b12x / "moe/fused_moe/source.py").write_text(formats)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_b12x_mxfp8_moe() is expected


def test_tp2_preset_serves_the_stored_qad_checkpoint():
    """The launch validated on two RTX PRO 6000 Max-Q (8 GiB KV per GPU)."""
    plan = tp2()
    expected = {
        "model": QAD_CHECKPOINT,
        "revision": QAD_REVISION,
        "quantization": "nvfp4_csf",
        "load-format": "nvfp4_csf",
        "served-model-name": "GLM-5.3-Flash",
        "tensor-parallel-size": 2,
        "decode-context-parallel-size": 2,
        "mode": "mtp",
        "max-model-len": -1,
        "max-num-seqs": 8,
        "max-num-batched-tokens": 4096,
        "max-num-scheduled-tokens": 4096,
        "cache-object-tokens": 4096,
        "kv-cache-memory-bytes": TP2_KV,
        "gpu-memory-utilization": 0.985,
        "max-cudagraph-capture-size": 32,
        "cudagraph-capture-sizes": [1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
        "moe-backend": "b12x",
        "cache-mode": "lmcache",
        "cache-l2-enabled": False,
        "expert-activations": "bf16",
        "router-weights": "fp32",
        "prefill-activations": "a4",
    }
    assert {key: plan.values[key] for key in expected} == expected
    # The preset's own checkpoint, not one of the profile's variants.
    assert "checkpoint" not in plan.values
    assert plan.values["speculative-config"] == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
        "moe_backend": "b12x",
        "attention_backend": "B12X",
        "revision": QAD_REVISION,
    }
    assert "--code-revision" not in plan.argv
    expected_env = {
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_MAX_NCHANNELS": "2",
        "NCCL_BUFFSIZE": "1048576",
        "NCCL_NET_PLUGIN": "none",
        "NCCL_TUNER_PLUGIN": "none",
        "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:1",
        "OMP_NUM_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,large_segment_size_mb:12",
        "VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE": "off",
        # Prefill all-reduces use the b12x PCIe DMA path (measured faster at TP2/DCP2).
        "VLLM_PCIE_DMA_MIN_BYTES": "6MB",
        "VLLM_B12X_MLA_CKV_GATHER": "1",
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": "65536",
        "VLLM_GLM53_EMBED_HOST": "1",
        "VLLM_SHARE_PYNCCL_COMMS": "1",
        "VLLM_B12X_MOE_FP4_FORCE_A16": "1",
        "B12X_W4A16_FP32_TOPK_WEIGHTS": "1",
        "B12X_W4A16_A4_PREFILL_MIN_TOKENS": "1536",
        # One target page per DCP rank fills one 4096-token cache object.
        "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE": "auto",
    }
    assert {name: plan.environment.get(name) for name in expected_env} == expected_env
    assert "B12X_W4A16_A4_PREFILL_TERMS" not in plan.environment
    # The RAM tier only: no disk adapter, and the L1 arena may fit the host.
    assert "--l2-adapter" not in plan.cache_service.argv
    assert plan.cache_service.l1_adjustable


def test_tp2_is_described_as_the_default_tp2_recipe():
    presets = deployment_presets()
    assert "default TP2 recipe" in presets["glm53-tp2"]["description"]
    assert "7,296 MiB KV cache per GPU" in presets["glm53-tp2"]["description"]
    assert "LMCache" in presets["glm53-tp2"]["description"]


def test_tp2_serves_only_its_fp4_csf_checkpoint():
    with pytest.raises(
        ConfigError,
        match="PRESET=glm53-tp2 serves an FP4-CSF checkpoint of its own.*"
        "CHECKPOINT=original does not apply",
    ):
        tp2({"CHECKPOINT": "original"})
    same_kind = tp2({"CHECKPOINT": "csf"})
    assert same_kind.values["model"] == QAD_CHECKPOINT
    assert same_kind.values["revision"] == QAD_REVISION


def test_tp2_needs_an_image_that_reads_nvfp4_csf():
    with pytest.raises(
        ConfigError,
        match="cannot read nvfp4_csf checkpoints, which PRESET=glm53-tp2 serves",
    ):
        tp2(csf_formats=frozenset({"mxfp4_csf"}))
    assert tp2(csf_formats=frozenset({"nvfp4_csf"})).values["model"] == (QAD_CHECKPOINT)


def test_tp2_extra_slots_shrink_the_kv_cache_and_both_cache_modes_keep_it():
    sixteen = tp2({"MAX_NUM_SEQS": "16"})
    assert sixteen.values["kv-cache-memory-bytes"] == TP2_KV - 8 * 67108864
    assert sixteen.values["max-cudagraph-capture-size"] == 64
    # The qualified size already leaves room for the LMCache GPU buffers.
    vram = tp2({"CACHE_MODE": "vram"})
    assert vram.values["kv-cache-memory-bytes"] == TP2_KV
    assert vram.cache_service is None
    assert "max-num-scheduled-tokens" not in vram.values
    disk = tp2({"LMCACHE_MODE": "disk"})
    assert disk.values["kv-cache-memory-bytes"] == TP2_KV
    assert "--l2-adapter" in disk.cache_service.argv
    explicit = tp2({"MAX_NUM_SEQS": "16", "KV_CACHE_MEMORY_BYTES": "6442450944"})
    assert explicit.values["kv-cache-memory-bytes"] == 6442450944


def test_tp2_prefill_activation_choices():
    a4 = tp2({"PREFILL_ACTIVATIONS": "a4"})
    assert a4.environment["B12X_W4A16_A4_PREFILL_MIN_TOKENS"] == "1536"
    assert "B12X_W4A16_A4_PREFILL_TERMS" not in a4.environment
    assert a4.environment["VLLM_B12X_MOE_FP4_FORCE_A16"] == "1"
    with pytest.raises(ConfigError, match="prefill-activations must be one of"):
        tp2({"PREFILL_ACTIVATIONS": "a8"})
    with pytest.raises(ConfigError, match="needs expert-activations bf16"):
        tp2({"EXPERT_ACTIVATIONS": "fp4", "PREFILL_ACTIVATIONS": "a4"})


@pytest.mark.parametrize("preset", ["glm53-csf-tp8", "glm53-csf-tp6"])
def test_glm53_presets_share_contended_steps_with_prefill(preset):
    """Like the Flash and DS4.1 profiles, GLM-5.3 keeps decoding while a long
    prompt prefills instead of stalling the running requests."""
    argv = resolve("glm53", "rtx-pro-6000-pcie", preset=preset, env={}).argv
    for flag, value in (
        ("--prefill-compute-share", "0.4"),
        ("--prefill-schedule-interval", "1"),
        ("--max-parallel-prefills", "1"),
    ):
        assert argv[argv.index(flag) + 1] == value, flag
