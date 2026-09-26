"""Resolve cache geometry and a CPU-service plan without starting processes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from runtime import ConfigError

# LMCACHE_L2_CHECKPOINT_WRITES values that select a checkpoint store policy.
CHECKPOINT_STORE_POLICIES = {
    "on-reuse": "checkpoint_on_reuse",
    "on-evict": "checkpoint_on_evict",
}
# Seconds LMCache may spend writing RAM-only checkpoints to L2 at shutdown
# with on-evict, and the extra time the supervisor waits before SIGKILL.
CHECKPOINT_SHUTDOWN_FLUSH_SECONDS = 30
STOP_GRACE_SECONDS = 10


@dataclass
class CacheService:
    argv: list[str]
    environment: dict[str, str]
    health_url: str
    shm_name: str
    shm_bytes: int
    startup_timeout: float
    directories: list[str] = field(default_factory=list)
    identity_required: bool = False
    namespace: str = ""
    # Seconds between SIGTERM and SIGKILL when the container stops.
    stop_grace: float = STOP_GRACE_SECONDS
    # Delete unused disk-tier namespaces of earlier images or settings.
    prune_stale_tiers: bool = False


def configure(values, origins, environment, env_origins, identifier, runtime_identity):
    """Derive worker/service settings once from the resolved model configuration."""

    def put(key, value):
        values[key], origins[key] = value, "derived:external-cache contract"

    def export(key, value):
        environment[key], env_origins[key] = (
            str(value),
            "derived:external-cache contract",
        )

    cache_mode = values["cache-mode"]
    if cache_mode == "vram":
        return None
    glm = identifier == "glm53-flash"
    qwen = identifier == "qwen38-flash-next"
    ds41 = identifier == "ds41-flash"
    if (
        not glm
        and not qwen
        and not ds41
        and identifier not in {"ds4-flash", "ds4-vision"}
    ):
        raise ConfigError(
            f"{identifier}: external cache is unsupported by this profile; Engram/PLE placement is independent"
        )
    if cache_mode == "native":
        if not (glm or qwen):
            raise ConfigError(
                "Native KV offload is implemented only by the GLM and Qwen profiles"
            )
        if values["cache-native-gib"] <= 0:
            raise ConfigError("cache-native-gib must be positive")
        if qwen:
            if values["decode-context-parallel-size"] != 1:
                raise ConfigError("Qwen native CPU cache requires DCP=1")
            if values.get("recurrent-checkpoint-policy", "auto") not in {
                "auto",
                "aligned",
            }:
                raise ConfigError("Qwen native CPU cache requires aligned checkpoints")
            if environment.get("VLLM_USE_SIMPLE_KV_OFFLOAD", "1") != "1":
                raise ConfigError(
                    "Qwen native CPU cache requires VLLM_USE_SIMPLE_KV_OFFLOAD=1; "
                    "the generic OffloadingConnector is not supported"
                )
            export("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
            put("recurrent-checkpoint-policy", "aligned")
        put("kv-offloading-backend", "native")
        put("kv-offloading-size", values["cache-native-gib"])
        put("enable-cumem-allocator", True)
        return None

    transfer = values["cache-transfer-mode"]
    engine = transfer == "engine_driven"
    if (qwen or ds41) and not engine:
        raise ConfigError(
            "Qwen and DS4.1 external cache require engine-driven transfer"
        )
    chunk = values["cache-object-tokens"]
    for key in (
        "cache-object-tokens",
        "cache-start-timeout",
        "cache-l1-gib",
        "cache-l1-init-gib",
        "cache-l2-gib",
        "cache-cpu-workers",
        "cache-l2-workers",
    ):
        if values[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if values["cache-l1-init-gib"] > values["cache-l1-gib"]:
        raise ConfigError("Initial L1 capacity cannot exceed its configured capacity")
    if "cache-gpu-workers" not in values:
        put("cache-gpu-workers", values["tensor-parallel-size"])
    if values["cache-gpu-workers"] < 1:
        raise ConfigError("cache-gpu-workers must be positive")

    for offset, key in (
        (10000, "cache-port"),
        (10001, "cache-http-port"),
        (10002, "cache-metrics-port"),
    ):
        if key not in values:
            put(key, values["port"] + offset)
    ports = [
        values[key]
        for key in ("port", "cache-port", "cache-http-port", "cache-metrics-port")
    ]
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ConfigError(
            "Model and cache ports must be distinct and in [1, 65535]; set explicit cache ports when automatic offsets overflow"
        )
    for key in ("cache-host", "cache-http-host"):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", values[key]):
            raise ConfigError(f"Invalid {key}")
    if "cache-instance" not in values:
        put("cache-instance", f"{identifier}-{values['port']}")
    if "cache-shm-name" not in values:
        put(
            "cache-shm-name",
            f"lmcache-{values['cache-instance']}-{values['cache-port']}",
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", values["cache-instance"]):
        raise ConfigError(
            "Cache instance must contain only letters, digits, dots, hyphens and underscores"
        )
    shm = values["cache-shm-name"] if engine else ""
    if engine and not re.fullmatch(r"[A-Za-z0-9._-]+", shm):
        raise ConfigError(
            "Engine-driven transfer requires a nonempty valid SHM name; no pickle fallback"
        )

    semantic = False
    if glm:
        tp = values["tensor-parallel-size"]
        if tp not in {2, 4, 8}:
            raise ConfigError("The GLM external-cache profile supports TP2, TP4 or TP8")
        if tp == 2 and not engine:
            raise ConfigError(
                "GLM TP2 requires engine-driven request-boundary cache transfer"
            )
        target_budget = values.get(
            "cache-target-tokens", values["max-num-batched-tokens"]
        )
        if target_budget != chunk:
            raise ConfigError(
                "GLM cache-object-tokens and target scheduler budget must match"
            )
        policy_key = "recurrent-checkpoint-policy"
        if origins[policy_key].startswith("model:") or values[policy_key] == "auto":
            put(policy_key, "request_boundaries" if engine else "aligned")
        semantic = values[policy_key] == "request_boundaries"
        if tp == 2 and not semantic:
            raise ConfigError(
                "GLM TP2 requires engine-driven request-boundary cache transfer"
            )
        if semantic and not engine:
            raise ConfigError(
                "Request-boundary checkpoint bundles require engine-driven transfer"
            )
        if origins["target-page-size"].startswith("model:"):
            put("target-page-size", "auto")
            export("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")
        page = values["target-page-size"]
        if page != "auto" and chunk % (
            int(page) * values["decode-context-parallel-size"]
        ):
            raise ConfigError("Cache objects must contain complete DCP target pages")
        input_rows = target_budget
        if values["mode"] == "dflash2":
            input_rows += values["draft-tokens"] * values["max-num-seqs"]
        put("max-num-batched-tokens", input_rows)
        put("max-num-scheduled-tokens", target_budget)
        retention = 0 if semantic else chunk
        if (
            "prefix-cache-retention-interval" in values
            and not origins["prefix-cache-retention-interval"].startswith(
                ("model:", "common:", "derived:")
            )
            and values["prefix-cache-retention-interval"] not in {"auto", retention}
        ):
            raise ConfigError(
                "Explicit checkpoint retention conflicts with the cache object geometry"
            )
        put("prefix-cache-retention-interval", retention)
        if engine and origins["gpu-memory-utilization"].startswith("model:"):
            put("gpu-memory-utilization", 0.950)
    elif qwen:
        # GDN state must be restored with its exact attention/PLE boundary.
        # Independent aligned chunks are not a substitute for that bundle.
        policy = values.get("recurrent-checkpoint-policy", "auto")
        if policy not in {"auto", "request_boundaries"}:
            raise ConfigError(
                "Qwen external cache requires request-boundary checkpoints"
            )
        if values.get("prefix-cache-retention-interval", 0) not in {0, "auto"}:
            raise ConfigError(
                "Qwen external cache uses exact boundaries, not periodic retention"
            )
        semantic = True
        put("recurrent-checkpoint-policy", "request_boundaries")
        put("prefix-cache-retention-interval", 0)
    else:
        if chunk % (values["block-size"] * values["decode-context-parallel-size"]):
            raise ConfigError("Cache object tokens must align to DS4 DCP cache pages")
        if (
            not ds41
            and engine
            and values["tensor-parallel-size"] == 2
            and (values["max-model-len"] == -1 or values["max-model-len"] >= 1048576)
            and origins["gpu-memory-utilization"].startswith("model:")
        ):
            put("gpu-memory-utilization", 0.970)

    connector = {
        "kv_connector": "LMCacheRecurrentCheckpointConnector"
        if semantic
        else "LMCacheMPConnector",
        "kv_connector_module_path": "lmcache.integration.vllm.recurrent_checkpoint_connector"
        if semantic
        else "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": values["cache-load-failure-policy"],
        "kv_connector_extra_config": {
            "lmcache.mp.host": values["cache-host"],
            "lmcache.mp.port": values["cache-port"],
            "lmcache.mp.mp_transfer_mode": transfer,
        },
    }
    put("kv-transfer-config", connector)
    if not engine:
        if glm:
            put("enable-cumem-allocator", True)
            interposer = "/opt/lmcache/lib/liblmcache_cumem_shareable.so"
            export(
                "LD_PRELOAD",
                interposer
                + (
                    ":" + environment["LD_PRELOAD"]
                    if environment.get("LD_PRELOAD")
                    else ""
                ),
            )
            export("LMCACHE_CUMEM_BROKER_DIR", values["cache-broker-directory"])
        else:
            allocator = environment.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"
            )
            export(
                "PYTORCH_CUDA_ALLOC_CONF",
                allocator.replace(
                    "expandable_segments:True", "expandable_segments:False"
                ),
            )

    argv = [
        "/opt/venv/bin/lmcache",
        "server",
        "--instance-id",
        values["cache-instance"],
        "--host",
        values["cache-host"],
        "--port",
        str(values["cache-port"]),
        "--http-host",
        values["cache-http-host"],
        "--http-port",
        str(values["cache-http-port"]),
        "--prometheus-port",
        str(values["cache-metrics-port"]),
        "--chunk-size",
        str(chunk),
        "--supported-transfer-mode",
        transfer,
        "--separate-object-groups",
        "--l1-size-gb",
        str(values["cache-l1-gib"]),
        "--l1-init-size-gb",
        str(values["cache-l1-init-gib"]),
        "--max-gpu-workers",
        str(values["cache-gpu-workers"]),
        "--max-cpu-workers",
        str(values["cache-cpu-workers"]),
        "--eviction-policy",
        "LRU",
        "--l2-prefetch-policy",
        values["cache-prefetch-policy"],
    ]
    argv += ["--no-l1-use-lazy", "--shm-name", shm] if engine else ["--l1-use-lazy"]
    # A request-boundary checkpoint carries the complete recurrent state and
    # the next turn of the same conversation supersedes it. on-reuse keeps
    # new checkpoints in RAM until a restore proves them useful; on-evict
    # writes the current checkpoint of a conversation when it leaves RAM and
    # at shutdown, and never writes superseded ones.
    store_policy = "default"
    checkpoint_writes = values["cache-l2-checkpoint-writes"]
    if (
        values["cache-l2-enabled"]
        and checkpoint_writes in CHECKPOINT_STORE_POLICIES
        and not semantic
        and origins["cache-l2-checkpoint-writes"].startswith(("model:", "common:"))
    ):
        # The profile default applies to request-boundary checkpoints only;
        # other checkpoint policies keep writing every cache object.
        checkpoint_writes = "always"
        values["cache-l2-checkpoint-writes"] = checkpoint_writes
        origins["cache-l2-checkpoint-writes"] = (
            "derived:no request-boundary checkpoints"
        )
    if values["cache-l2-enabled"] and checkpoint_writes in CHECKPOINT_STORE_POLICIES:
        if not semantic:
            raise ConfigError(
                f"LMCACHE_L2_CHECKPOINT_WRITES={checkpoint_writes} applies only "
                "to request-boundary checkpoints"
            )
        store_policy = CHECKPOINT_STORE_POLICIES[checkpoint_writes]
    stop_grace = STOP_GRACE_SECONDS
    if store_policy == "checkpoint_on_evict":
        argv += [
            "--checkpoint-shutdown-flush-seconds",
            str(CHECKPOINT_SHUTDOWN_FLUSH_SECONDS),
        ]
        stop_grace += CHECKPOINT_SHUTDOWN_FLUSH_SECONDS
    if glm:
        argv += ["--hash-algorithm", "blake3", "--max-workers", "8"]
        if store_policy != "default":
            argv += ["--l2-store-policy", store_policy]
    else:
        argv += [
            "--l1-write-ttl-seconds",
            "600",
            "--l1-read-ttl-seconds",
            "300",
            "--eviction-trigger-watermark",
            "0.90",
            "--eviction-ratio",
            "0.10",
            "--l2-store-policy",
            store_policy,
            "--worker-reap-timeout-seconds",
            "120",
            "--worker-registration-grace-seconds",
            "3600",
        ]
        connector["kv_connector_extra_config"].update(
            {"lmcache.mp.mq_timeout": 60.0, "lmcache.mp.heartbeat_interval": 10.0}
        )
    # Model identity must be resolved before persistent storage is opened. A
    # dry-run uses a visibly unresolved namespace and performs no Hub/disk I/O.
    layout = {
        key: values[key]
        for key in (
            "model",
            "tensor-parallel-size",
            "decode-context-parallel-size",
            "kv-cache-dtype",
            "block-size",
            "mode",
            "draft-tokens",
            "cache-object-tokens",
        )
    }
    layout.update(
        {
            key: values[key]
            for key in (
                "target-page-size",
                "recurrent-page-size",
                "dcp-ckv-gather",
                "cp-kv-cache-interleave-size",
                "recurrent-checkpoint-policy",
                "speculative-config",
            )
            if key in values
        }
    )
    if qwen or ds41:
        layout.update(
            {
                key: values[key]
                for key in (
                    "swa-block-size",
                    "mamba-cache-mode",
                    "mamba-ssm-cache-dtype",
                )
                if key in values
            }
        )
    layout["runtime"] = runtime_identity or "UNBOUND-RUNTIME"
    layout_digest = hashlib.sha256(
        json.dumps(layout, sort_keys=True).encode()
    ).hexdigest()
    path = Path(values["cache-directory"])
    if not path.is_absolute() or path == Path("/"):
        raise ConfigError("Cache directory must be an absolute, dedicated directory")
    namespace = str(path / identifier / layout_digest / "UNRESOLVED-CHECKPOINT")
    directories = []
    if values["cache-l2-enabled"]:
        l2 = {
            "type": "fs_native",
            "base_path": namespace,
            "num_workers": values["cache-l2-workers"],
            "use_odirect": values["cache-l2-odirect"],
            "max_capacity_gb": values["cache-l2-gib"],
        }
        if glm or qwen:
            l2["eviction"] = {
                "eviction_policy": "LRU",
                "trigger_watermark": 0.8,
                "eviction_ratio": 0.2,
            }
        argv += ["--l2-adapter", json.dumps(l2, separators=(",", ":"))]
        if glm and values["cache-prefetch-policy"] == "retain":
            argv += ["--emergency-evict-for-prefetch"]
        directories += [namespace]
        if semantic:
            argv += [
                "--checkpoint-index-path",
                str(Path(namespace) / "semantic-directory.sqlite3"),
            ]
    if glm and not engine:
        directories += [values["cache-broker-directory"]]
    probe = values["cache-http-host"]
    probe = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(probe, probe)
    if ":" in probe:
        probe = f"[{probe}]"
    return CacheService(
        argv,
        {"CUDA_VISIBLE_DEVICES": "", "CUDA_MODULE_LOADING": "LAZY"} if engine else {},
        f"http://{probe}:{values['cache-http-port']}/healthcheck",
        # LMCache's --shm-name is a suffix; the allocator owns this OS name.
        f"lmcache_l1_pool_{shm}" if engine else "",
        int(values["cache-l1-gib"] * 1024**3) if engine else 0,
        values["cache-start-timeout"],
        directories,
        semantic or values["cache-l2-enabled"],
        namespace,
        stop_grace,
        bool(values.get("cache-l2-prune-stale")),
    )


# Asked in a GPU-free child so the supervisor keeps no Torch state. Without
# vLLM #864 the QSA backend refuses KV connectors while nothing enforces it,
# and DCP>1 would silently fall back to KV chunks without recurrent state.
_QSA_ATOMIC_TRANSFER_PROBE = (
    "from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSABackend as B\n"
    "raise SystemExit(0 if B.supports_kv_connector() else 3)"
)


def verify_installed_transfer(plan, run=subprocess.run) -> None:
    """Refuse Qwen DCP>1 external caches unless vLLM keeps QSA checkpoints atomic."""
    service = plan.cache_service
    if (
        not service
        or plan.profile != "qwen38-flash-next"
        or plan.values["decode-context-parallel-size"] == 1
    ):
        return
    result = run(
        [sys.executable, "-c", _QSA_ATOMIC_TRANSFER_PROBE],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ConfigError(
            "Qwen external cache with DCP>1 requires a vLLM build with atomic QSA "
            "checkpoint transfer; use DCP1 or an image that includes it"
        )


def resolve_identity(plan, contract, helper: Path):
    """Pin loaded weights and persistent namespace without re-resolving policy."""
    service = plan.cache_service
    if not service or not service.identity_required:
        return
    spec = importlib.util.spec_from_file_location("lil_checkpoint_identity", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = plan.values
    target = module.resolve_checkpoint(values["model"], values.get("revision"))
    draft = {"identity": "", "revision": ""}
    if values["mode"] == "mtp" or values["mode"] == "dspark":
        draft = target
    elif values["mode"] == "dflash2":
        draft_spec = values["speculative-config"]
        draft = module.resolve_checkpoint(
            draft_spec["model"], draft_spec.get("revision")
        )
    if target["revision"]:
        values["revision"] = target["revision"]
        plan.origins["revision"] = "resolved:checkpoint identity"
    if draft["revision"] and "speculative-config" in values:
        values["speculative-config"]["revision"] = draft["revision"]
    identity = {
        "target_revision": target["identity"],
        "draft_revision": draft["identity"],
        "source_revision": contract["runtime_lock_sha256"],
    }
    config = values["kv-transfer-config"]
    if config["kv_connector"] == "LMCacheRecurrentCheckpointConnector":
        config["kv_connector_extra_config"]["lmcache.mp.checkpoint_identity"] = identity
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    service.argv = [
        arg.replace("UNRESOLVED-CHECKPOINT", suffix) for arg in service.argv
    ]
    service.directories = [
        path.replace("UNRESOLVED-CHECKPOINT", suffix) for path in service.directories
    ]
    service.namespace = service.namespace.replace("UNRESOLVED-CHECKPOINT", suffix)
