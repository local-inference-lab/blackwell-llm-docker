"""Replicas mode: independent TP1 servers behind one conversation-affinity proxy.

``replicas: N`` starts N complete model servers in this container, one per
visible GPU, each on a loopback port, plus ``runtime.replica_proxy`` on the
public host and port. They share one LMCache server when an external cache is
selected, and Qwen3.8-Flash-Next shares one host-RAM copy of its PLE table.
The first process to exit stops the others and sets the container's status.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

from runtime import ConfigError

PLE_SHARED_DIRECTORY = "/dev/shm/lil-ple"


def checkpoint_identity(plan, helper: Path) -> str:
    """Content identity of the target checkpoint; a Hub revision is pinned.

    Every replica must load the same snapshot, and the shared PLE table key
    must change whenever the checkpoint bytes do.
    """
    if not helper.is_file():
        raise ConfigError(
            "The image must package its source-locked checkpoint identity helper"
        )
    spec = importlib.util.spec_from_file_location("lil_checkpoint_identity", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = plan.values
    target = plan.target_identity or module.resolve_checkpoint(
        values["model"], values.get("revision")
    )
    if target["revision"]:
        values["revision"] = target["revision"]
        plan.origins["revision"] = "resolved:checkpoint identity"
        speculative = values.get("speculative-config")
        if speculative and values.get("mode") in ("mtp", "dspark"):
            speculative["revision"] = target["revision"]
    return target["identity"]


def visible_gpus(environment: dict) -> list[str]:
    """GPU identifiers in the order CUDA enumerates them for this container."""
    devices = environment.get("CUDA_VISIBLE_DEVICES")
    if devices:
        return [device.strip() for device in devices.split(",") if device.strip()]
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ConfigError(
            f"Replicas need the list of visible GPUs; nvidia-smi failed: {error}"
        ) from None
    return [line.strip() for line in output.splitlines() if line.strip()]


def loopback_ports(count: int) -> list[int]:
    """Free loopback ports for the replica servers."""
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def replica_servers(
    plan,
    environment: dict,
    bootstrap: list[str],
    *,
    gpus: list[str],
    ports: list[int],
) -> list[tuple[str, list[str], dict]]:
    """Commands of the replica servers followed by the proxy."""
    from runtime.launcher import make_argv

    replicas = plan.values["replicas"]
    if len(gpus) < replicas:
        raise ConfigError(
            f"replicas={replicas} needs {replicas} visible GPUs, the container sees "
            f"{len(gpus)}"
        )
    servers = []
    for index, (gpu, port) in enumerate(zip(gpus, ports)):
        values = {**plan.values, "host": "127.0.0.1", "port": port}
        servers.append(
            (
                f"Replica {index}",
                [*bootstrap, *make_argv(values, plan.passthrough)],
                {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "VLLM_LOGGING_PREFIX": f"[replica {index}] ",
                },
            )
        )
    proxy = [
        sys.executable,
        "-m",
        "runtime.replica_proxy",
        "--host",
        plan.values["host"],
        "--port",
        str(plan.values["port"]),
        *(f"--replica=127.0.0.1:{port}" for port in ports[:replicas]),
    ]
    servers.append(("Replica proxy", proxy, dict(environment)))
    return servers


def run(plan, environment: dict, bootstrap: list[str]) -> int:
    from runtime.supervisor import supervise_replicas

    replicas = plan.values["replicas"]
    gpus = visible_gpus(environment)[:replicas]
    servers = replica_servers(
        plan,
        environment,
        bootstrap,
        gpus=gpus,
        ports=loopback_ports(replicas),
    )
    print(
        f"[lil-serve] Starting {replicas} replicas on GPUs {', '.join(gpus)} behind "
        f"{plan.values['host']}:{plan.values['port']}",
        file=sys.stderr,
        flush=True,
    )
    return supervise_replicas(plan.cache_service, servers, environment, bootstrap)
