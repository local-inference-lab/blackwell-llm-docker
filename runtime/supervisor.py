"""Own cache/server process groups, readiness, SHM checks and bounded shutdown."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from runtime import ConfigError
from runtime.cache import CacheService


def read_json(url: str) -> dict:
    # Internal service readiness must not be redirected through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=1) as response:
        return json.load(response)


def validate_pool(status: dict, service: CacheService) -> None:
    pool = status.get("engine_driven_shm_pool") or {}
    if (
        pool.get("shm_name") != service.shm_name
        or int(pool.get("pool_size") or 0) < service.shm_bytes
    ):
        raise ConfigError(
            "LMCache must advertise the requested SHM arena and capacity; no pickle fallback. "
            f"Expected {service.shm_name!r}/{service.shm_bytes} bytes, "
            f"received {pool.get('shm_name')!r}/{pool.get('pool_size')} bytes"
        )


SHM_ROOT = Path("/dev/shm")
# Share of a filesystem's space (free plus what the tier already holds) that
# a disk tier may claim; the rest stays for the model, logs and other writers.
L2_DISK_SHARE = 0.9
# Arena locks stay open for the supervisor's lifetime. The kernel releases a
# flock when its holder exits, including on SIGKILL or an OOM kill, so a held
# lock proves a running owner and a free one proves the arena is stale.
_ARENA_LOCKS: list[int] = []
# Each running cache holds a shared lock on its disk tier. A tier whose lock
# can be taken exclusively has no running owner that took the lock.
_TIER_LOCKS: list[int] = []
TIER_LOCK = ".lil-tier.lock"
# A tier written this recently may belong to an image that predates the lock.
RECENT_TIER_WRITE_SECONDS = 1800


def _claim_arena(shm: Path) -> None:
    """Lock the arena name for this container or refuse if another owns it."""
    fd = os.open(shm.with_name(shm.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ConfigError(
            f"Refusing to reuse a cache SHM arena owned by another running container: {shm}"
        ) from None
    _ARENA_LOCKS.append(fd)


def _tree_bytes(path: Path) -> int:
    """Bytes allocated on disk by the files below ``path``."""
    total = 0
    stack = [path]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_blocks * 512
                except OSError:
                    continue
    return total


def _tree_newest(path: Path) -> float:
    """Newest modification time of the files below ``path``."""
    newest = 0.0
    stack = [path]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    continue
    return newest


def _disk_tier_paths(service: CacheService) -> list[Path]:
    paths = []
    for index, arg in enumerate(service.argv[:-1]):
        if arg == "--l2-adapter":
            adapter = json.loads(service.argv[index + 1])
            if adapter.get("type") in {"fs", "fs_native"}:
                paths.append(Path(adapter["base_path"]))
    return paths


def report_stale_tiers(service: CacheService) -> None:
    """Name disk-tier namespaces of earlier images and optionally delete them.

    A tier lives in ``<cache>/<model>/<layout>/<checkpoint>``. The layout
    digest covers the runtime, so every image or cache-setting change starts
    a new namespace, and the previous one stays on disk outside the tier cap.
    """
    for base in _disk_tier_paths(service):
        # Only the namespace this launcher derived has the layout above.
        if not service.namespace or base != Path(service.namespace):
            continue
        base.mkdir(parents=True, exist_ok=True)
        fd = os.open(base / TIER_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise ConfigError(
                f"Another container is deleting the LMCache disk tier {base}"
            ) from None
        _TIER_LOCKS.append(fd)
        root = base.parent.parent
        stale = []
        for layout in sorted(root.iterdir()):
            if layout.is_symlink() or not layout.is_dir():
                continue
            for tier in sorted(layout.iterdir()):
                if tier == base or tier.is_symlink() or not tier.is_dir():
                    continue
                stale.append(tier)
        if not stale:
            continue
        now = time.time()
        removed, kept = [], []
        for tier in stale:
            size = _tree_bytes(tier)
            newest = _tree_newest(tier)
            lock = tier / TIER_LOCK
            owner = None
            if lock.exists():
                owner = os.open(lock, os.O_RDWR)
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(owner)
                    kept.append((tier, size, "in use by a running container"))
                    continue
            try:
                if not service.prune_stale_tiers:
                    written = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))
                    kept.append((tier, size, f"last written {written}"))
                elif now - newest < RECENT_TIER_WRITE_SECONDS:
                    kept.append((tier, size, "written in the last 30 minutes"))
                else:
                    shutil.rmtree(tier)
                    removed.append((tier, size, "removed"))
            finally:
                if owner is not None:
                    os.close(owner)
        for label, tiers in (("Removed", removed), ("Kept", kept)):
            if not tiers:
                continue
            total = sum(size for _, size, _ in tiers)
            listing = "; ".join(
                f"{tier} ({size / 1024**3:.0f} GiB, {why})" for tier, size, why in tiers
            )
            announce(
                f"{label} {len(tiers)} disk-tier namespace(s) of earlier images or "
                f"cache settings, {total / 1024**3:.0f} GiB outside LMCACHE_L2_GB: "
                f"{listing}"
            )
        if kept and not service.prune_stale_tiers:
            announce(
                "Set LMCACHE_L2_PRUNE_STALE=1 to delete the ones no running "
                "container uses at startup, or delete them by hand"
            )


def fit_disk_tiers(service: CacheService) -> None:
    """Cap filesystem L2 tiers to what their disk can still hold.

    A tier larger than its disk fills the disk and fails every writer on it,
    including the model server. Existing tier contents count as available,
    since the tier can evict them.
    """
    for index, arg in enumerate(service.argv[:-1]):
        if arg != "--l2-adapter":
            continue
        adapter = json.loads(service.argv[index + 1])
        requested = adapter.get("max_capacity_gb")
        if adapter.get("type") not in {"fs", "fs_native"} or not requested:
            continue
        base = Path(adapter["base_path"])
        probe = base
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        stats = os.statvfs(probe)
        free = stats.f_bavail * stats.f_frsize
        held = _tree_bytes(base) if base.exists() else 0
        usable = math.floor((free + held) * L2_DISK_SHARE / 1024**3)
        if requested <= usable:
            continue
        if usable < 1:
            raise ConfigError(
                f"No space for the LMCache disk tier at {base}: "
                f"{free / 1024**3:.1f} GiB free. Free disk space or set "
                "LMCACHE_L2_ENABLED=0"
            )
        adapter["max_capacity_gb"] = usable
        service.argv[index + 1] = json.dumps(adapter, separators=(",", ":"))
        announce(
            f"LMCache disk tier capped at {usable} GiB instead of {requested:g} GiB: "
            f"{probe} has {free / 1024**3:.0f} GiB free and the tier already holds "
            f"{held / 1024**3:.0f} GiB. Lower LMCACHE_L2_GB or free space to "
            "silence this"
        )


def preflight(service: CacheService, environment: dict) -> None:
    if any("UNRESOLVED-CHECKPOINT" in arg for arg in service.argv):
        raise ConfigError(
            "Persistent cache identity must be resolved before starting processes"
        )
    shm = SHM_ROOT / service.shm_name if service.shm_bytes else None
    if shm is not None:
        _claim_arena(shm)
    interposer = Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so")
    if (
        str(interposer) in environment.get("LD_PRELOAD", "").split(":")
        and not interposer.is_file()
    ):
        raise ConfigError(
            "LMCache-driven GLM transfer requires the packaged CUDA cuMem interposer"
        )
    broker = environment.get("LMCACHE_CUMEM_BROKER_DIR")
    for name in service.directories:
        path = Path(name)
        if path.is_symlink():
            raise ConfigError(f"Refusing a symlinked cache service directory: {path}")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if name == broker and (
            path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077
        ):
            raise ConfigError(
                "The cuMem broker directory must be owned by the serving user with mode 0700"
            )
    # A health response must not accidentally belong to another cache service.
    host = service.argv[service.argv.index("--host") + 1]
    http_host = service.argv[service.argv.index("--http-host") + 1]
    for flag, address in (
        ("--port", host),
        ("--http-port", http_host),
        ("--prometheus-port", http_host),
    ):
        port = int(service.argv[service.argv.index(flag) + 1])
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError as error:
                raise ConfigError(
                    f"Cache service address is unavailable: {address}:{port}"
                ) from error
    if shm is not None:
        # This container holds the arena lock and the cache ports are free, so
        # an existing arena was left by a cache that did not shut down cleanly.
        if shm.exists():
            shm.unlink()
            announce(
                f"Removed the stale cache SHM arena {shm} left by a cache service "
                "that did not shut down cleanly"
            )
        stats = os.statvfs(SHM_ROOT)
        if stats.f_bavail * stats.f_frsize < service.shm_bytes:
            raise ConfigError(
                "Insufficient free /dev/shm for the complete engine-driven L1 arena"
            )
    report_stale_tiers(service)
    fit_disk_tiers(service)


def describe_exit(status: int) -> str:
    if status >= 0:
        return f"status {status}"
    try:
        name = signal.Signals(-status).name
    except ValueError:
        name = f"signal {-status}"
    if status == -signal.SIGKILL:
        # Nothing in this container sends SIGKILL before its own shutdown.
        return f"{name}; the host kernel OOM killer is the usual sender, see dmesg"
    return name


def announce(message: str) -> None:
    # Printed before shutdown starts, so it precedes the model server's own
    # shutdown messages and names the actual cause.
    print(f"[lil-serve] {message}", file=sys.stderr, flush=True)


def stop_groups(children, grace=10):
    """Stop producers before cache, within one shared shutdown deadline.

    Children are appended cache first, model second. Model workers must drain
    their copy leases while the cache RPC service is still available.
    """
    deadline = time.monotonic() + grace
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Reserve the cache flush portion of the shared budget for the cache.
        remaining = max(0.01, deadline - time.monotonic())
        wait = min(10.0, remaining) if child is not children[0] else remaining
        try:
            child.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass
        # Workers may outlive the leader; signal the entire producer group
        # before cache shutdown. Cache ownership still protects undrained copies.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def supervise(
    service: CacheService,
    model_command: list[str],
    environment: dict,
    bootstrap: list[str],
) -> int:
    preflight(service, environment)
    children = []
    requested_signal = 0

    def request_shutdown(signum, _frame):
        nonlocal requested_signal
        requested_signal = signum

    previous = {
        sig: signal.signal(sig, request_shutdown)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        cache = subprocess.Popen(
            [*bootstrap, *service.argv],
            env={**environment, **service.environment},
            start_new_session=True,
        )
        children.append(cache)
        deadline = time.monotonic() + service.startup_timeout
        while not requested_signal:
            status = cache.poll()
            if status is not None:
                raise ConfigError(f"LMCache exited before readiness (status {status})")
            if time.monotonic() >= deadline:
                raise ConfigError("LMCache startup timed out")
            try:
                # /healthcheck can be non-JSON; /status is the transport contract.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(service.health_url, timeout=1):
                    pass
                if service.shm_bytes:
                    validate_pool(
                        read_json(service.health_url.rsplit("/", 1)[0] + "/status"),
                        service,
                    )
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(0.1)
        if requested_signal:
            announce(
                f"Received {signal.Signals(requested_signal).name} before the model "
                "server started; stopping LMCache"
            )
            return 128 + requested_signal
        if cache.poll() is not None:
            raise ConfigError("LMCache exited after its readiness response")
        model = subprocess.Popen(model_command, env=environment, start_new_session=True)
        children.append(model)
        while not requested_signal:
            status = cache.poll()
            if status is not None:
                reason = (
                    f"LMCache exited ({describe_exit(status)}) while the model "
                    "server was running"
                )
                announce(f"{reason}; stopping the model server")
                raise ConfigError(reason)
            status = model.poll()
            if status is not None:
                announce(
                    f"Model server exited ({describe_exit(status)}); stopping LMCache"
                )
                return status if status >= 0 else 128 - status
            time.sleep(0.1)
        announce(
            f"Received {signal.Signals(requested_signal).name} from outside the "
            "container, usually docker stop, a Compose recreation or an image "
            "updater; stopping the model server and LMCache"
        )
        return 128 + requested_signal
    finally:
        stop_groups(children, service.stop_grace)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
