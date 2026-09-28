"""Model/cache process ownership must hold on success and failure."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import pytest

from runtime import ConfigError, supervisor
from runtime.cache import CacheService
from runtime.launcher import resolve


@pytest.mark.parametrize("grace", [20, 50])
def test_stop_drains_model_before_signalling_cache(monkeypatch, grace):
    events = []
    clock = [100.0]

    class Child:
        def __init__(self, pid, delay):
            self.pid, self.delay = pid, delay

        def wait(self, timeout=None):
            events.append(("wait", self.pid, timeout))
            if timeout is not None:
                clock[0] += min(timeout, self.delay)
                if self.delay > timeout:
                    raise subprocess.TimeoutExpired("child", timeout)
            return 0

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        supervisor.os, "killpg", lambda pid, sig: events.append(("signal", pid, sig))
    )
    cache, model = Child(101, 25), Child(102, 100)
    supervisor.stop_groups([cache, model], grace=grace)
    assert events[:4] == [
        ("signal", 102, signal.SIGTERM),
        ("wait", 102, 10),
        ("signal", 102, signal.SIGKILL),
        ("wait", 102, None),
    ]
    assert events[4:6] == [("signal", 101, signal.SIGTERM), ("wait", 101, grace - 10)]
    assert clock[0] == 110 + min(25, grace - 10)


@pytest.mark.parametrize(
    "cache_exit,model_exit,expected",
    [(None, 0, 0), (7, None, "before readiness"), (None, 3, 3)],
)
def test_supervisor_always_stops_its_children(
    monkeypatch, cache_exit, model_exit, expected
):
    service = replace(
        resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service, shm_bytes=0
    )
    children, stopped = [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.environment = command, kwargs["env"]
            self.code = cache_exit if not children else model_exit
            children.append(self)

        def poll(self):
            return self.code

    class HTTP:
        def open(self, *_args, **_kwargs):
            return BytesIO(b"{}")

    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(
        supervisor, "stop_groups", lambda processes, grace: stopped.extend(processes)
    )
    monkeypatch.setattr(supervisor.urllib.request, "build_opener", lambda *_: HTTP())
    if isinstance(expected, str):
        with pytest.raises(ConfigError, match=expected):
            supervisor.supervise(service, ["model"], {"CUDA_VISIBLE_DEVICES": "2"}, [])
    else:
        assert (
            supervisor.supervise(service, ["model"], {"CUDA_VISIBLE_DEVICES": "2"}, [])
            == expected
        )
        assert children[1].environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert children[0].environment["CUDA_VISIBLE_DEVICES"] == ""
    assert stopped == children


def test_wrong_shm_pool_cannot_start_a_model():
    service = resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service
    with pytest.raises(ConfigError, match="no pickle fallback"):
        supervisor.validate_pool(
            {
                "engine_driven_shm_pool": {
                    "shm_name": service.shm_name,
                    "pool_size": service.shm_bytes - 1,
                }
            },
            service,
        )


@pytest.mark.parametrize("valid_pool", [True, False])
def test_non_json_status_retries_but_invalid_pool_is_fatal(monkeypatch, valid_pool):
    service = resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service
    children, stopped, attempts = [], [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.code = None if not children else 0
            children.append(self)

        def poll(self):
            return self.code

    class HTTP:
        def open(self, *_args, **_kwargs):
            return BytesIO(b"{}")

    def status(_url):
        attempts.append(1)
        if len(attempts) == 1:
            raise json.JSONDecodeError("not ready", "", 0)
        return {
            "engine_driven_shm_pool": {
                "shm_name": service.shm_name,
                "pool_size": service.shm_bytes if valid_pool else 0,
            }
        }

    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(
        supervisor, "stop_groups", lambda items, grace: stopped.extend(items)
    )
    monkeypatch.setattr(supervisor.urllib.request, "build_opener", lambda *_: HTTP())
    monkeypatch.setattr(supervisor, "read_json", status)
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    if valid_pool:
        assert supervisor.supervise(service, ["model"], {}, []) == 0
        assert len(children) == 2
    else:
        with pytest.raises(ConfigError, match="no pickle fallback"):
            supervisor.supervise(service, ["model"], {}, [])
        assert len(children) == 1
    assert len(attempts) == 2
    assert stopped == children


@pytest.mark.parametrize(
    "event,expected,announcement",
    [
        (
            "cache-killed",
            "SIGKILL; the host kernel OOM killer",
            "LMCache exited (SIGKILL",
        ),
        ("model-exit", 3, "Model server exited (status 3); stopping LMCache"),
        (
            "sigterm",
            128 + signal.SIGTERM,
            "Received SIGTERM from outside the container",
        ),
    ],
)
def test_supervisor_names_the_stop_reason_before_stopping(
    monkeypatch, capsys, event, expected, announcement
):
    service = replace(
        resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service, shm_bytes=0
    )
    children, stopped = [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.role = "cache" if not children else "model"
            children.append(self)

        def poll(self):
            if len(children) < 2:
                return None
            if event == "cache-killed" and self.role == "cache":
                return -signal.SIGKILL
            if event == "model-exit" and self.role == "model":
                return 3
            if event == "sigterm" and self.role == "model":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return None

    class HTTP:
        def open(self, *_args, **_kwargs):
            return BytesIO(b"{}")

    def stop(processes, grace):
        # The reason must already be visible when shutdown begins.
        assert announcement in capsys.readouterr().err
        stopped.extend(processes)

    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(supervisor, "stop_groups", stop)
    monkeypatch.setattr(supervisor.urllib.request, "build_opener", lambda *_: HTTP())
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    if isinstance(expected, str):
        with pytest.raises(ConfigError, match=expected):
            supervisor.supervise(service, ["model"], {}, [])
    else:
        assert supervisor.supervise(service, ["model"], {}, []) == expected
    assert stopped == children


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _service(shm_bytes=0, l2=None, namespace="", prune=False):
    argv = ["--host", "127.0.0.1", "--http-host", "127.0.0.1"]
    for flag in ("--port", "--http-port", "--prometheus-port"):
        argv += [flag, str(_free_port())]
    if l2 is not None:
        argv += ["--l2-adapter", json.dumps(l2)]
    return CacheService(
        argv,
        {},
        "",
        "lmcache_l1_pool_test",
        shm_bytes,
        1.0,
        [],
        namespace=namespace,
        prune_stale_tiers=prune,
    )


@pytest.fixture
def shm_root(monkeypatch, tmp_path):
    root = tmp_path / "shm"
    root.mkdir()
    locks: list[int] = []
    monkeypatch.setattr(supervisor, "SHM_ROOT", root)
    monkeypatch.setattr(supervisor, "_ARENA_LOCKS", locks)
    yield root
    for fd in locks:
        os.close(fd)


def test_stale_arena_of_a_stopped_cache_is_removed(shm_root, capsys):
    stale = shm_root / "lmcache_l1_pool_test"
    stale.write_bytes(b"left by a killed container")
    supervisor.preflight(_service(shm_bytes=4096), {})
    assert not stale.exists()
    assert "Removed the stale cache SHM arena" in capsys.readouterr().err


def test_arena_of_a_running_cache_is_refused(shm_root):
    arena = shm_root / "lmcache_l1_pool_test"
    arena.write_bytes(b"live")
    owner = os.open(shm_root / "lmcache_l1_pool_test.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ConfigError, match="owned by another running container"):
            supervisor.preflight(_service(shm_bytes=4096), {})
        assert arena.exists()
    finally:
        os.close(owner)


def _statvfs_with_free(monkeypatch, free_bytes):
    real = os.statvfs

    def fake(path):
        stats = real(path)
        return os.statvfs_result(
            (stats.f_bsize, 1, stats.f_blocks, free_bytes, free_bytes)
            + tuple(stats)[5:]
        )

    monkeypatch.setattr(supervisor.os, "statvfs", fake)


def test_disk_tier_larger_than_its_disk_is_capped(monkeypatch, tmp_path, capsys):
    base = tmp_path / "l2"
    base.mkdir()
    _statvfs_with_free(monkeypatch, 100 * 1024**3)
    l2 = {"type": "fs_native", "base_path": str(base), "max_capacity_gb": 512}
    service = _service(l2=l2)
    supervisor.fit_disk_tiers(service)
    adapter = json.loads(service.argv[service.argv.index("--l2-adapter") + 1])
    assert adapter["max_capacity_gb"] == 90
    assert "capped at 90 GiB instead of 512 GiB" in capsys.readouterr().err


def test_disk_tier_that_fits_is_unchanged(monkeypatch, tmp_path, capsys):
    _statvfs_with_free(monkeypatch, 1024 * 1024**3)
    l2 = {
        "type": "fs_native",
        "base_path": str(tmp_path / "l2"),
        "max_capacity_gb": 512,
    }
    service = _service(l2=l2)
    before = list(service.argv)
    supervisor.fit_disk_tiers(service)
    assert service.argv == before
    assert capsys.readouterr().err == ""


def test_disk_tier_without_space_is_refused(monkeypatch, tmp_path):
    _statvfs_with_free(monkeypatch, 100 * 1024**2)
    l2 = {"type": "fs_native", "base_path": str(tmp_path), "max_capacity_gb": 64}
    with pytest.raises(ConfigError, match="No space for the LMCache disk tier"):
        supervisor.fit_disk_tiers(_service(l2=l2))


@pytest.fixture
def tiers(monkeypatch, tmp_path):
    """A model cache root with the current namespace and two older ones."""
    locks: list[int] = []
    monkeypatch.setattr(supervisor, "_TIER_LOCKS", locks)
    root = tmp_path / "lmcache" / "glm53-flash"
    current = root / "layout-new" / "checkpoint"
    older = root / "layout-old" / "checkpoint"
    other_checkpoint = root / "layout-new" / "other-checkpoint"
    for tier, age in ((older, 7200), (other_checkpoint, 7200)):
        tier.mkdir(parents=True)
        page = tier / "page.bin"
        page.write_bytes(b"x" * 8192)
        stamp = time.time() - age
        os.utime(page, (stamp, stamp))
    unrelated = tmp_path / "lmcache" / "qwen38-flash-next" / "layout" / "checkpoint"
    unrelated.mkdir(parents=True)
    l2 = {"type": "fs_native", "base_path": str(current), "max_capacity_gb": 64}
    yield SimpleNamespace(
        current=current,
        older=older,
        other_checkpoint=other_checkpoint,
        unrelated=unrelated,
        l2=l2,
    )
    for fd in locks:
        os.close(fd)


def test_older_disk_tiers_are_named_and_kept_by_default(tiers, capsys):
    supervisor.report_stale_tiers(_service(l2=tiers.l2, namespace=str(tiers.current)))
    err = capsys.readouterr().err
    assert "Kept 2 disk-tier namespace(s)" in err
    assert str(tiers.older) in err and str(tiers.other_checkpoint) in err
    assert "LMCACHE_L2_PRUNE_STALE=1" in err
    for tier in (tiers.older, tiers.other_checkpoint, tiers.unrelated):
        assert tier.exists()
    # The current tier is locked for the container's lifetime.
    probe = os.open(tiers.current / supervisor.TIER_LOCK, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(probe)


def test_prune_removes_only_unused_older_tiers_of_this_model(tiers, capsys):
    # Another running container holds the other checkpoint's tier.
    lock = tiers.other_checkpoint / supervisor.TIER_LOCK
    owner = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(owner, fcntl.LOCK_SH)
    try:
        supervisor.report_stale_tiers(
            _service(l2=tiers.l2, namespace=str(tiers.current), prune=True)
        )
    finally:
        os.close(owner)
    err = capsys.readouterr().err
    assert not tiers.older.exists()
    assert tiers.other_checkpoint.exists()
    assert "in use by a running container" in err
    assert tiers.current.exists() and tiers.unrelated.exists()
    assert "Removed 1 disk-tier namespace(s)" in err


def test_prune_keeps_a_recently_written_tier(tiers, capsys):
    (tiers.older / "page.bin").touch()
    supervisor.report_stale_tiers(
        _service(l2=tiers.l2, namespace=str(tiers.current), prune=True)
    )
    assert tiers.older.exists()
    assert "written in the last 30 minutes" in capsys.readouterr().err


def test_tiers_outside_the_derived_namespace_are_not_managed(tiers, capsys):
    supervisor.report_stale_tiers(_service(l2=tiers.l2, namespace="", prune=True))
    assert tiers.older.exists() and tiers.other_checkpoint.exists()
    assert capsys.readouterr().err == ""
