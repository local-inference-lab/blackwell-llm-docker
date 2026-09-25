"""Deployment overlays must preserve model defaults and explicit override priority."""

import json
import os
import subprocess
import sys

import pytest

from runtime.entrypoint import command
from runtime.launcher import ROOT, ConfigError, deployment_presets, resolve
from runtime.packaging import audit_image_metadata, owned_environment, payload_sources


def spark(env=None, **kwargs):
    return resolve(
        "glm53-flash",
        "rtx-pro-6000-pcie",
        preset="glm53-spark-tp2",
        env=env or {},
        **kwargs,
    )


def test_spark_overlay_preserves_the_bounded_tp2_memory_recipe():
    plan = spark()
    expected = {
        "model": "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark",
        "tensor-parallel-size": 2,
        "decode-context-parallel-size": 2,
        "max-num-seqs": 8,
        "max-num-batched-tokens": 3072,
        "max-model-len": -1,
        "kv-cache-memory-bytes": 4777312256,
        "load-format": "safetensors",
        "max-cudagraph-capture-size": 32,
        "cudagraph-capture-sizes": [1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
    }
    assert {key: plan.values[key] for key in expected} == expected
    assert plan.values["speculative-config"] == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
        "moe_backend": "b12x",
        "attention_backend": "B12X",
    }
    assert plan.environment["NCCL_MIN_NCHANNELS"] == "2"
    assert plan.environment["NCCL_MAX_NCHANNELS"] == "2"
    assert plan.environment["NCCL_BUFFSIZE"] == "1048576"
    assert plan.environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:1"
    assert (
        plan.environment["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True,large_segment_size_mb:12"
    )
    assert "--limit-mm-per-prompt" not in plan.argv
    assert plan.values["additional-config"]["kda_prefill_backend"] == "b12x"
    for name in (
        "VLLM_GLM53_EMBED_HOST",
        "VLLM_GLM53_VISION_MXFP8",
        "VLLM_SHARE_PYNCCL_COMMS",
    ):
        assert plan.environment[name] == "1"


def test_spark_extra_request_slots_shrink_the_preset_kv_allocation():
    sixteen = spark(env={"MAX_NUM_SEQS": "16"})
    assert sixteen.values["max-num-seqs"] == 16
    assert sixteen.values["kv-cache-memory-bytes"] == 4777312256 - 8 * 67108864
    assert sixteen.values["max-cudagraph-capture-size"] == 64
    assert sixteen.origins["kv-cache-memory-bytes"].startswith("derived:")
    fewer = spark(env={"MAX_NUM_SEQS": "4"})
    assert fewer.values["kv-cache-memory-bytes"] == 4777312256
    explicit = spark(env={"MAX_NUM_SEQS": "16", "KV_CACHE_MEMORY_BYTES": "5000000000"})
    assert explicit.values["kv-cache-memory-bytes"] == 5000000000


def test_spark_external_cache_keeps_room_for_its_gpu_buffers():
    lmcache = spark(argv=["--cache-mode", "lmcache"])
    assert lmcache.values["kv-cache-memory-bytes"] == 4777312256 - 201326592
    both = spark(env={"MAX_NUM_SEQS": "16"}, argv=["--cache-mode", "lmcache"])
    assert both.values["kv-cache-memory-bytes"] == 4777312256 - 8 * 67108864 - 201326592


def test_spark_settings_do_not_leak_into_tp4_or_qwen():
    before = resolve("glm53-flash", "rtx-pro-6000-pcie", env={}).public()
    spark()
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
    plan = spark(argv=["--mode", mode])
    assert plan.values["draft-tokens"] == width
    if width:
        assert plan.values["speculative-config"]["num_speculative_tokens"] == width
    else:
        assert "speculative-config" not in plan.values


def test_explicit_inputs_override_preset_values():
    plan = spark(
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
    plan = spark(cli_env={"MAX_NUM_SEQS": str(seqs)})
    assert plan.values["max-cudagraph-capture-size"] == sizes[-1]
    assert plan.values["cudagraph-capture-sizes"] == sizes


@pytest.mark.parametrize("mode,input_rows", [("mtp", 4096), ("dflash2", 4096 + 8 * 7)])
def test_lmcache_target_budget_follows_the_changed_prefill_budget(mode, input_rows):
    plan = spark(
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
        spark(argv=["--cache-mode", "lmcache", "--cache-object-tokens", "4096"])


def test_unknown_or_wrong_model_preset_is_rejected():
    with pytest.raises(ConfigError, match="Unknown deployment"):
        resolve("glm53-flash", preset="../../other", env={})
    with pytest.raises(ConfigError, match="different model"):
        resolve("ds4-flash", preset="glm53-spark-tp2", env={})


def test_profile_cli_and_entrypoint_accept_preset_selection():
    assert "--help" not in command([], {"PRESET": "glm53-spark-tp2"}, {})
    raw = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--preset",
            "glm53-spark-tp2",
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
    assert deployment_presets()["glm53-spark-tp2"]["profile"] == "glm53-flash"


def test_foundation_defaults_are_below_presets_and_user_environment():
    default = resolve("glm53-flash", env={})
    assert default.environment["NCCL_NET_PLUGIN"] == "spcx"
    assert default.environment_origins["NCCL_NET_PLUGIN"] == "platform:foundation"
    assert spark().environment["NCCL_NET_PLUGIN"] == "none"
    for value in ("spcx", "none", "operator-plugin"):
        plan = resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-spark-tp2",
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
        spark(argv=["--kv-cache-memory-bytes", amount])


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
