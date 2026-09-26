"""External-cache launch plans preserve model geometry and transport identity."""

import json

import pytest

from runtime import ConfigError
from runtime.cache import resolve_identity, verify_installed_transfer
from runtime.launcher import make_argv, resolve
from runtime.supervisor import validate_pool


@pytest.mark.parametrize(
    "mode,depth,rows", [("off", 0, 4096), ("mtp", 3, 4096), ("dflash2", 7, 4320)]
)
@pytest.mark.parametrize("dcp", [1, 4])
@pytest.mark.parametrize("dtype", ["fp8", "nvfp4_ds_mla"])
def test_glm_target_budget_is_not_reduced_by_dflash(mode, depth, rows, dcp, dtype):
    plan = resolve(
        "glm53-flash",
        env={
            "CACHE_MODE": "lmcache",
            "SPECULATOR": mode,
            "DCP": str(dcp),
            "KV_CACHE_DTYPE": dtype,
        },
    )
    assert plan.values["draft-tokens"] == depth
    assert plan.values["max-num-batched-tokens"] == rows
    assert plan.values["max-num-scheduled-tokens"] == 4096
    assert plan.values["prefix-cache-retention-interval"] == 0
    assert plan.values["target-page-size"] == "auto"
    assert plan.values["gpu-memory-utilization"] == 0.95
    assert plan.cache_service.environment["CUDA_VISIBLE_DEVICES"] == ""
    assert "UNRESOLVED-CHECKPOINT" in plan.cache_service.namespace
    assert plan.argv.count("--kv-transfer-config") == 1


def test_aligned_direct_transport_retains_interposer_and_checkpoints():
    plan = resolve(
        "glm53-flash",
        env={"CACHE_MODE": "lmcache", "LMCACHE_TRANSFER_MODE": "lmcache_driven"},
    )
    assert plan.values["recurrent-checkpoint-policy"] == "aligned"
    assert plan.values["prefix-cache-retention-interval"] == 4096
    assert plan.values["kv-transfer-config"]["kv_connector"] == "LMCacheMPConnector"
    assert plan.values["enable-cumem-allocator"] is True
    assert "liblmcache_cumem_shareable.so" in plan.environment["LD_PRELOAD"]
    assert not plan.cache_service.environment


@pytest.mark.parametrize("dcp", [1, 2])
def test_glm_tp2_atomic_cache_preserves_3072_token_budget(dcp):
    plan = resolve(
        "glm53-flash",
        env={
            "CACHE_MODE": "lmcache",
            "TP": "2",
            "DCP": str(dcp),
            "SPECULATOR": "mtp",
            "MTP_DEPTH": "3",
            "MAX_NUM_BATCHED_TOKENS": "3072",
            "LMCACHE_CHUNK_SIZE": "3072",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,large_segment_size_mb:12",
        },
    )
    assert plan.values["max-num-batched-tokens"] == 3072
    assert plan.values["max-num-scheduled-tokens"] == 3072
    assert plan.values["recurrent-checkpoint-policy"] == "request_boundaries"
    assert (
        plan.values["kv-transfer-config"]["kv_connector"]
        == "LMCacheRecurrentCheckpointConnector"
    )
    assert (
        plan.environment["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True,large_segment_size_mb:12"
    )
    assert plan.cache_service.environment["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize(
    "extra",
    [
        {"LMCACHE_TRANSFER_MODE": "lmcache_driven"},
        {"RECURRENT_CHECKPOINT_POLICY": "aligned"},
    ],
)
def test_glm_tp2_external_cache_requires_atomic_engine_transport(extra):
    with pytest.raises(ConfigError, match="TP2.*engine-driven.*request-boundary"):
        resolve("glm53-flash", env={"CACHE_MODE": "lmcache", "TP": "2", **extra})


@pytest.mark.parametrize("identifier", ["ds4-flash", "ds4-vision"])
def test_ds4_ram_and_disk_use_one_cpu_only_service(identifier):
    ram = resolve(identifier, env={"LMCACHE_MODE": "ram"})
    disk = resolve(identifier, env={"LMCACHE_MODE": "disk"})
    assert ram.values["cache-l1-gib"] == 24
    assert ram.values["cache-object-tokens"] == 4096
    assert ram.values["gpu-memory-utilization"] == 0.970
    assert "--l2-adapter" not in ram.cache_service.argv
    assert "--l2-adapter" in disk.cache_service.argv
    assert ram.cache_service.environment["CUDA_VISIBLE_DEVICES"] == ""


def test_native_filesystem_cache_exposes_explicit_direct_io_control():
    default = resolve("qwen38-flash-next", env={"LMCACHE_MODE": "disk"})
    direct = resolve(
        "qwen38-flash-next",
        env={"LMCACHE_MODE": "disk", "LMCACHE_L2_ODIRECT": "1"},
    )

    def adapter(plan):
        index = plan.cache_service.argv.index("--l2-adapter")
        return json.loads(plan.cache_service.argv[index + 1])

    assert adapter(default)["use_odirect"] is False
    assert adapter(direct)["use_odirect"] is True
    assert adapter(default)["eviction"] == {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.8,
        "eviction_ratio": 0.2,
    }


def test_qwen_external_cache_preserves_scheduler_and_exact_recurrent_state():
    plan = resolve("qwen38-flash-next", env={"CACHE_MODE": "lmcache"})
    assert plan.values["max-num-batched-tokens"] == 6019
    assert plan.values["max-model-len"] == 262144
    assert plan.values["recurrent-checkpoint-policy"] == "request_boundaries"
    assert plan.values["prefix-cache-retention-interval"] == 0
    assert (
        plan.values["kv-transfer-config"]["kv_connector"]
        == "LMCacheRecurrentCheckpointConnector"
    )
    assert plan.cache_service.identity_required
    assert plan.cache_service.environment["CUDA_VISIBLE_DEVICES"] == ""
    assert plan.values["cache-gpu-workers"] == 1


@pytest.mark.parametrize(("tp", "dcp"), [(2, 1), (2, 2), (4, 1), (4, 2), (4, 4)])
def test_qwen_atomic_cache_supports_tp_dcp_topologies(tp, dcp):
    plan = resolve(
        "qwen38-flash-next",
        env={
            "CACHE_MODE": "lmcache",
            "TP": str(tp),
            "DCP": str(dcp),
        },
    )
    assert plan.values["recurrent-checkpoint-policy"] == "request_boundaries"
    assert (
        plan.values["kv-transfer-config"]["kv_connector"]
        == "LMCacheRecurrentCheckpointConnector"
    )
    assert plan.values["cache-gpu-workers"] == tp


@pytest.mark.parametrize(
    "identifier,dcp,supported,probed",
    [
        ("qwen38-flash-next", 1, False, False),
        ("qwen38-flash-next", 2, True, True),
        ("qwen38-flash-next", 2, False, True),
        ("glm53-flash", 4, False, False),
    ],
)
def test_qwen_dcp_cache_requires_installed_atomic_qsa_transfer(
    identifier, dcp, supported, probed
):
    plan = resolve(
        identifier, env={"CACHE_MODE": "lmcache", "TP": "4", "DCP": str(dcp)}
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return type("Result", (), {"returncode": 0 if supported else 3})()

    if probed and not supported:
        with pytest.raises(ConfigError, match="atomic QSA checkpoint transfer"):
            verify_installed_transfer(plan, run=run)
    else:
        verify_installed_transfer(plan, run=run)
    assert len(calls) == int(probed)
    assert all(devices == "" for _command, devices in calls)


@pytest.mark.parametrize("identifier", ["glm53-flash", "qwen38-flash-next"])
def test_image_serving_exposes_text_only_external_checkpoint_contract(identifier):
    plan = resolve(
        identifier,
        env={"CACHE_MODE": "lmcache"},
        config={"options": {"language-model-only": False}},
    )
    assert any("text requests only" in warning for warning in plan.warnings)
    assert plan.values["language-model-only"] is False
    text = resolve(
        identifier,
        env={"CACHE_MODE": "lmcache"},
        config={"options": {"language-model-only": True}},
    )
    assert not any("text requests only" in warning for warning in text.warnings)


def test_aligned_image_cache_has_no_recurrent_checkpoint_warning():
    plan = resolve("ds4-vision", env={"CACHE_MODE": "lmcache"})
    assert not any("text requests only" in warning for warning in plan.warnings)


def test_ds41_external_cache_keeps_sliding_window_and_engram_independent():
    plan = resolve(
        "ds41-flash", env={"CACHE_MODE": "lmcache", "ENGRAM_TABLE_MEMORY": "ram"}
    )
    assert plan.values["engram-table-memory"] == "ram"
    assert plan.values["block-size"] == 256
    assert plan.values["swa-block-size"] == 128
    assert plan.values["max-num-batched-tokens"] == 4096
    assert plan.values["kv-transfer-config"]["kv_connector"] == "LMCacheMPConnector"
    assert plan.values["gpu-memory-utilization"] == 0.95
    assert plan.cache_service.environment["CUDA_VISIBLE_DEVICES"] == ""
    assert plan.values["cache-gpu-workers"] == 4
    disk = resolve("ds41-flash", env={"LMCACHE_MODE": "disk"})
    assert "--l2-adapter" in disk.cache_service.argv
    assert disk.cache_service.identity_required


@pytest.mark.parametrize("identifier", ["qwen38-flash-next", "ds41-flash"])
def test_cache_transfer_modes_require_the_model_contract(identifier):
    with pytest.raises(ConfigError, match="engine-driven"):
        resolve(
            identifier,
            env={"CACHE_MODE": "lmcache", "LMCACHE_TRANSFER_MODE": "lmcache_driven"},
        )


def test_ds41_native_cpu_cache_remains_unsupported():
    with pytest.raises(ConfigError, match="only.*GLM and Qwen"):
        resolve("ds41-flash", env={"CACHE_MODE": "native"})


def test_qwen_native_cpu_cache_uses_simple_connector_without_lmcache_service():
    plan = resolve(
        "qwen38-flash-next",
        env={
            "CACHE_MODE": "native",
            "NATIVE_KV_OFFLOADING_SIZE_GB": "4",
            "TP": "2",
        },
    )
    assert plan.cache_service is None
    assert plan.environment["VLLM_USE_SIMPLE_KV_OFFLOAD"] == "1"
    assert plan.values["kv-offloading-backend"] == "native"
    assert plan.values["kv-offloading-size"] == 4
    assert plan.values["enable-cumem-allocator"] is True
    assert plan.values["recurrent-checkpoint-policy"] == "aligned"
    assert plan.values["mamba-cache-mode"] == "align"
    assert plan.values["draft-tokens"] == 3
    assert plan.values["max-num-batched-tokens"] == 6019
    assert "--kv-offloading-backend" in plan.argv


@pytest.mark.parametrize(
    "extra,error",
    [
        ({"TP": "2", "DCP": "2"}, "DCP=1"),
        ({"RECURRENT_CHECKPOINT_POLICY": "request_boundaries"}, "aligned"),
        ({"VLLM_USE_SIMPLE_KV_OFFLOAD": "0"}, "Simple|SIMPLE"),
    ],
)
def test_qwen_native_cpu_cache_rejects_incompatible_contract(extra, error):
    with pytest.raises(ConfigError, match=error):
        resolve("qwen38-flash-next", env={"CACHE_MODE": "native", **extra})


@pytest.mark.parametrize(
    "extra",
    [
        {"recurrent-checkpoint-policy": "aligned"},
        {"prefix-cache-retention-interval": 4096},
    ],
)
def test_qwen_rejects_incompatible_checkpoint_contract(extra):
    with pytest.raises(ConfigError, match="Qwen external cache"):
        resolve(
            "qwen38-flash-next",
            env={},
            config={"options": {"cache-mode": "lmcache", **extra}},
        )


@pytest.mark.parametrize(
    "env",
    [
        {"LMCACHE_L1_SIZE_GB": "64", "LMCACHE_L1_GB": "24"},
        {"LMCACHE_CHUNK_SIZE": "8192"},
        {"LMCACHE_SHM_NAME": ""},
        {"LMCACHE_L2_PATH": "/"},
        {"PORT": "60000"},
        {"GLM53_TARGET_BLOCK_SIZE": "2048", "DCP": "4"},
    ],
)
def test_invalid_cache_configuration_fails_before_io(env):
    with pytest.raises(ConfigError):
        resolve("glm53-flash", env={"CACHE_MODE": "lmcache", **env})


def test_pool_must_match_both_identity_and_capacity():
    service = resolve("ds4-flash", env={"CACHE_MODE": "lmcache"}).cache_service
    assert (
        service.shm_name
        == "lmcache_l1_pool_" + service.argv[service.argv.index("--shm-name") + 1]
    )
    pool = {"shm_name": service.shm_name, "pool_size": service.shm_bytes}
    validate_pool({"engine_driven_shm_pool": pool}, service)
    for change in ({"shm_name": "another-service"}, {"pool_size": 1}):
        with pytest.raises(ConfigError, match="no pickle fallback"):
            validate_pool({"engine_driven_shm_pool": {**pool, **change}}, service)


def test_cache_aliases_follow_settings_and_cli_precedence():
    plan = resolve(
        "ds4-flash",
        env={"CACHE_MODE": "vram"},
        config={"environment": {"LMCACHE_MODE": "disk"}},
    )
    assert plan.cache_service is not None
    assert plan.values["cache-l2-enabled"] is True
    plan = resolve(
        "ds4-flash",
        env={"CACHE_MODE": "native", "LMCACHE_MODE": "disk"},
        argv=["--cache-mode", "vram"],
    )
    assert plan.cache_service is None
    with pytest.raises(ConfigError, match="Conflicting environment aliases"):
        resolve("ds4-flash", env={"CACHE_MODE": "vram", "LMCACHE_MODE": "disk"})


def test_explicit_retention_is_not_silently_overridden():
    with pytest.raises(ConfigError, match="retention conflicts"):
        resolve(
            "glm53-flash",
            env={"CACHE_MODE": "lmcache"},
            argv=["--prefix-cache-retention-interval", "512"],
        )


def test_runtime_and_geometry_change_persistent_namespace():
    first = resolve(
        "glm53-flash", env={"CACHE_MODE": "lmcache"}, runtime_identity="a" * 64
    )
    second = resolve(
        "glm53-flash", env={"CACHE_MODE": "lmcache"}, runtime_identity="b" * 64
    )
    dcp = resolve(
        "glm53-flash",
        env={"CACHE_MODE": "lmcache", "DCP": "4"},
        runtime_identity="a" * 64,
    )
    assert len({plan.cache_service.namespace for plan in (first, second, dcp)}) == 3


def test_immutable_identity_pins_both_loaders_before_startup(tmp_path):
    helper = tmp_path / "identity.py"
    helper.write_text(
        'def resolve_checkpoint(model, revision):\n    return {"identity": "a" * 40, "revision": "a" * 40}\n'
    )
    plan = resolve(
        "glm53-flash",
        env={"CACHE_MODE": "lmcache", "SPECULATOR": "dflash2"},
        runtime_identity="b" * 64,
    )
    resolve_identity(plan, {"runtime_lock_sha256": "b" * 64}, helper)
    assert plan.values["revision"] == "a" * 40
    assert plan.values["speculative-config"]["revision"] == "a" * 40
    assert "UNRESOLVED" not in json.dumps(plan.cache_service.argv)
    assert "--revision" in make_argv(plan.values, [])
    identity = plan.values["kv-transfer-config"]["kv_connector_extra_config"][
        "lmcache.mp.checkpoint_identity"
    ]
    assert identity["source_revision"] == "b" * 64


def store_policies(plan) -> list[str]:
    argv = plan.cache_service.argv
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "--l2-store-policy"]


@pytest.mark.parametrize(
    "identifier,env,expected",
    [
        ("qwen38-flash-next", {}, ["checkpoint_on_evict"]),
        (
            "qwen38-flash-next",
            {"LMCACHE_L2_CHECKPOINT_WRITES": "always"},
            ["default"],
        ),
        (
            "qwen38-flash-next",
            {"LMCACHE_L2_CHECKPOINT_WRITES": "on-reuse"},
            ["checkpoint_on_reuse"],
        ),
        ("glm53-flash", {"TP": "2"}, ["checkpoint_on_evict"]),
        ("glm53-flash", {"TP": "2", "LMCACHE_L2_CHECKPOINT_WRITES": "always"}, []),
        (
            "glm53-flash",
            {"TP": "2", "LMCACHE_L2_CHECKPOINT_WRITES": "on-reuse"},
            ["checkpoint_on_reuse"],
        ),
        (
            "qwen38-flash-next",
            {"LMCACHE_L2_CHECKPOINT_WRITES": "on-evict"},
            ["checkpoint_on_evict"],
        ),
        (
            "glm53-flash",
            {"TP": "2", "LMCACHE_L2_CHECKPOINT_WRITES": "on-evict"},
            ["checkpoint_on_evict"],
        ),
    ],
)
def test_l2_checkpoint_writes_select_the_store_policy(identifier, env, expected):
    plan = resolve(
        identifier,
        env={"CACHE_MODE": "lmcache", "LMCACHE_L2_ENABLED": "true", **env},
    )
    assert store_policies(plan) == expected


def test_l2_checkpoint_writes_need_l2_and_request_boundary_checkpoints():
    without_l2 = resolve(
        "qwen38-flash-next",
        env={"CACHE_MODE": "lmcache", "LMCACHE_L2_CHECKPOINT_WRITES": "on-reuse"},
    )
    assert store_policies(without_l2) == ["default"]
    with pytest.raises(ConfigError, match="request-boundary checkpoints"):
        resolve(
            "ds4-flash",
            env={
                "CACHE_MODE": "lmcache",
                "LMCACHE_L2_ENABLED": "true",
                "LMCACHE_L2_CHECKPOINT_WRITES": "on-reuse",
            },
        )


@pytest.mark.parametrize(
    "identifier,env", [("qwen38-flash-next", {}), ("glm53-flash", {"TP": "2"})]
)
def test_on_evict_flushes_at_shutdown_and_extends_the_stop_grace(identifier, env):
    base = {"CACHE_MODE": "lmcache", "LMCACHE_L2_ENABLED": "true", **env}
    always = resolve(
        identifier, env={**base, "LMCACHE_L2_CHECKPOINT_WRITES": "always"}
    ).cache_service
    on_evict = resolve(
        identifier, env={**base, "LMCACHE_L2_CHECKPOINT_WRITES": "on-evict"}
    ).cache_service
    assert "--checkpoint-shutdown-flush-seconds" not in always.argv
    flag = on_evict.argv.index("--checkpoint-shutdown-flush-seconds")
    flush = float(on_evict.argv[flag + 1])
    assert on_evict.stop_grace >= flush + always.stop_grace


def test_on_evict_default_falls_back_without_request_boundary_checkpoints():
    plan = resolve(
        "glm53-flash",
        env={
            "CACHE_MODE": "lmcache",
            "LMCACHE_L2_ENABLED": "true",
            "RECURRENT_CHECKPOINT_POLICY": "aligned",
        },
    )
    assert plan.values["cache-l2-checkpoint-writes"] == "always"
    assert plan.origins["cache-l2-checkpoint-writes"].startswith("derived:")
    with pytest.raises(ConfigError, match="request-boundary"):
        resolve(
            "glm53-flash",
            env={
                "CACHE_MODE": "lmcache",
                "LMCACHE_L2_ENABLED": "true",
                "RECURRENT_CHECKPOINT_POLICY": "aligned",
                "LMCACHE_L2_CHECKPOINT_WRITES": "on-evict",
            },
        )
    ds4 = resolve(
        "ds4-flash", env={"CACHE_MODE": "lmcache", "LMCACHE_L2_ENABLED": "true"}
    )
    assert ds4.values["cache-l2-checkpoint-writes"] == "always"
