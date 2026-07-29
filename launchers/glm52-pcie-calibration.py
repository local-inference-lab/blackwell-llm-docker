#!/usr/bin/env python3
"""Cache a lossless SparkInfer PCIe calibration before vLLM starts.

r10 changes (from the venice.kpchosting field failure, 2026-07-29):
- Preflight: every selected GPU must have MIN_FREE_MIB free before the
  probe launches; otherwise exit 3 with one clean status line naming the
  busy GPUs and the processes holding them. No traceback, no torchrun.
- Operational failures never print a Python traceback or an elastic
  error wall. The complete untruncated probe output is written to
  `<cache-dir>/<digest>.failure.log` and a single summary line points
  at it. Exit codes: 0 measured/cache-hit, 2 usage error, 3 skipped by
  preflight, 4 probe failed (details in file).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


SCHEMA_VERSION = 1
MIN_FREE_MIB = 4096
EXIT_SKIPPED_PREFLIGHT = 3
EXIT_PROBE_FAILED = 4
COLLECTIVE_ENV_PREFIXES = (
    "NCCL_",
    "SPARKINFER_PCIE_",
    "B12X_PCIE_",
    "VLLM_PCIE_",
)


class CalibrationSkip(Exception):
    """Calibration cannot run right now; this is a clean, expected skip."""


class CalibrationFailure(Exception):
    """The probe ran and failed; `detail` holds its complete output."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative integers"
        )
    return result


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _file_identity(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError:
        return {"path": str(candidate), "exists": False}
    return {
        "path": str(candidate.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _probe_source_digest() -> str:
    try:
        spec = importlib.util.find_spec("sparkinfer.comm.pcie.overlap_probe")
    except ImportError:
        # find_spec raises (rather than returning None) when the parent
        # package itself is absent; treat both cases as unavailable.
        spec = None
    if spec is None or spec.origin is None:
        return "unavailable"
    try:
        return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _read_optional_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _optional_nvidia_int(value: str) -> int | None:
    normalized = value.strip()
    if normalized.upper() in {"N/A", "[N/A]"}:
        return None
    return int(normalized)


def _runtime_placement() -> dict[str, Any]:
    try:
        cpu_affinity: list[int] | str = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_affinity = "unavailable"
    return {
        "cpu_affinity": cpu_affinity,
        "cpuset_cpus_effective": _read_optional_text(
            "/sys/fs/cgroup/cpuset.cpus.effective"
        ),
        "cpuset_mems_effective": _read_optional_text(
            "/sys/fs/cgroup/cpuset.mems.effective"
        ),
        "kernel": platform.release(),
        "machine": platform.machine(),
    }


def _collective_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(COLLECTIVE_ENV_PREFIXES)
    }


def _run_nvidia_smi(arguments: list[str]) -> str:
    command = ["nvidia-smi", *arguments]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30.0,
        )
    except FileNotFoundError as exc:
        raise CalibrationSkip("nvidia-smi is not available") from exc
    except subprocess.TimeoutExpired as exc:
        raise CalibrationSkip("nvidia-smi timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise CalibrationSkip(f"nvidia-smi failed with exit {exc.returncode}") from exc
    return completed.stdout


def _gpu_inventory(selected: Sequence[int]) -> list[dict[str, Any]]:
    stdout = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,pci.bus_id,pcie.link.gen.current,"
            "pcie.link.width.current,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory: dict[int, dict[str, Any]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index = int(fields[0])
        inventory[index] = {
            "index": index,
            "uuid": fields[1],
            "pci_bus_id": fields[2].lower(),
            "pcie_generation": _optional_nvidia_int(fields[3]),
            "pcie_width": _optional_nvidia_int(fields[4]),
            "driver_version": fields[5],
        }
    missing = [index for index in selected if index not in inventory]
    if missing:
        raise CalibrationSkip(f"selected GPUs missing from nvidia-smi: {missing}")
    return [inventory[index] for index in selected]


def _preflight_free_memory(selected: Sequence[int]) -> None:
    """Refuse to launch the probe when GPUs lack free memory.

    The probe needs roughly 3 GiB per GPU. Launching it against occupied
    GPUs (a still-running previous server instance, another tenant)
    produces an eight-rank CUDA OOM cascade; this check turns that into
    one clean skip line instead."""
    stdout = _run_nvidia_smi(
        [
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    free_by_index: dict[int, int] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            free_by_index[int(fields[0])] = int(fields[1])
        except ValueError:
            continue
    busy = [
        (index, free_by_index[index])
        for index in selected
        if index in free_by_index and free_by_index[index] < MIN_FREE_MIB
    ]
    if not busy:
        return
    busy_text = ", ".join(f"GPU{index}:{free}MiB" for index, free in busy)
    holders = "unavailable"
    try:
        holders_output = _run_nvidia_smi(
            [
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        ).strip()
        if holders_output:
            holders = "; ".join(
                line.strip() for line in holders_output.splitlines()[:4]
            )
    except CalibrationSkip:
        pass
    raise CalibrationSkip(
        f"gpu-memory-busy: {busy_text} free, need {MIN_FREE_MIB}MiB "
        f"(holders: {holders})"
    )


def build_fingerprint_payload(args: argparse.Namespace) -> dict[str, Any]:
    selected = tuple(args.gpus[: args.tp_size])
    if len(selected) != args.tp_size:
        raise ValueError("GPU selection must contain at least tp-size entries")
    return {
        "schema": SCHEMA_VERSION,
        "geometry": {
            "tp_size": args.tp_size,
            "dcp_size": args.dcp_size,
            "indexer_shards": args.indexer_shards,
            "hidden_size": args.hidden_size,
            "tp_rows": args.tp_rows,
            "ckv_record_bytes": args.ckv_record_bytes,
            "context_tokens": args.context_tokens,
            "allreduce_rows": args.allreduce_rows,
        },
        "gpu_order": _gpu_inventory(selected),
        "runtime_placement": _runtime_placement(),
        "software": {
            "cache_fingerprint": os.getenv("LOCAL_INFERENCE_CACHE_FINGERPRINT", ""),
            "python": sys.version,
            "torch": _package_version("torch"),
            "vllm": _package_version("vllm"),
            "sparkinfer": _package_version("sparkinfer"),
            "probe_sha256": _probe_source_digest(),
            "nccl": _file_identity(
                os.getenv("VLLM_NCCL_SO_PATH")
                or os.getenv("NCCL_LOCAL_INFERENCE_PATH")
                or os.getenv("LD_PRELOAD")
            ),
            "collective_environment": _collective_environment(),
        },
    }


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_probe_result(result: dict[str, Any]) -> dict[str, int | str]:
    policy = result.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("probe result has no policy")
    if policy.get("numeric_contract") != "lossless-only":
        raise ValueError("probe result is not lossless-only")
    if policy.get("compressed_dma_requires_explicit_opt_in") is not True:
        raise ValueError("probe result permits automatic compressed DMA")
    tp_policy = policy.get("tp_allreduce")
    if not isinstance(tp_policy, dict) or tp_policy.get("dma_wire_mode") != "bf16":
        raise ValueError("probe result does not use BF16 DMA")
    dma_min_bytes = int(tp_policy["dma_min_bytes"])
    values: dict[str, int | str] = {
        "prefetch_depth": int(policy["dcp_ckv_prefetch_depth"]),
        "query_split": int(policy["dcp_query_split"]),
        "query_split_min_context_tokens": int(
            policy["dcp_query_split_min_context_tokens"]
        ),
        "dma_min_bytes": dma_min_bytes if dma_min_bytes > 0 else "off",
    }
    if values["prefetch_depth"] not in (0, 1):
        raise ValueError("invalid prefetch depth")
    if values["query_split"] not in (0, 1):
        raise ValueError("invalid query-split decision")
    query_min_context = values["query_split_min_context_tokens"]
    if not isinstance(query_min_context, int) or query_min_context < 0:
        raise ValueError("invalid query-split context crossover")
    if bool(values["query_split"]) != bool(query_min_context):
        raise ValueError("inconsistent query-split decision and crossover")
    return values


def _load_record(path: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("schema") != SCHEMA_VERSION
        or record.get("fingerprint") != expected_fingerprint
    ):
        return None
    try:
        validate_probe_result(record["probe"])
    except (KeyError, TypeError, ValueError):
        return None
    return record


def _output_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _terminate_process_group(process: subprocess.Popen[str]) -> str:
    """Stop torchrun and every worker it started, then collect its output."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, _ = process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = process.communicate()
    return _output_text(stdout)


def _run_probe(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.tp_size}",
        "-m",
        "sparkinfer.comm.pcie.overlap_probe",
        "--tp-size",
        str(args.tp_size),
        "--dcp-size",
        str(args.dcp_size),
        "--indexer-shards",
        str(args.indexer_shards),
        "--hidden-size",
        str(args.hidden_size),
        "--tp-rows",
        str(args.tp_rows),
        "--ckv-record-bytes",
        str(args.ckv_record_bytes),
        "--context-tokens",
        ",".join(str(value) for value in args.context_tokens),
        "--allreduce-rows",
        ",".join(str(value) for value in args.allreduce_rows),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(value) for value in args.gpus[: args.tp_size]
    )
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise CalibrationFailure(f"failed to launch torchrun: {exc}", "") from exc
    try:
        stdout, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        captured = _output_text(exc.stdout)
        terminated_output = _terminate_process_group(process)
        if terminated_output:
            captured = terminated_output
        raise CalibrationFailure(
            f"probe timed out after {args.timeout:g}s", captured
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    stdout = _output_text(stdout)
    if process.returncode != 0:
        raise CalibrationFailure(f"probe exited with {process.returncode}", stdout)
    print(stdout, file=sys.stderr, end="")
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationFailure(
            f"probe produced no valid result at {output}: {exc}", stdout
        ) from exc
    try:
        validate_probe_result(result)
    except ValueError as exc:
        raise CalibrationFailure(f"probe result rejected: {exc}", stdout) from exc
    return result


def _write_failure_log(cache_dir: Path, digest: str, reason: str, detail: str) -> Path:
    failure_path = cache_dir / f"{digest}.failure.log"
    try:
        body = (
            f"# PCIe calibration failure\n"
            f"# time: {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n"
            f"# reason: {reason}\n"
            f"# complete probe output follows (nothing truncated)\n\n"
        )
        failure_path.write_text(body + (detail or "(no output)\n"), encoding="utf-8")
    except OSError:
        return Path("(failure log could not be written)")
    return failure_path


def calibrate(args: argparse.Namespace) -> tuple[str, dict[str, Any], Path]:
    payload = build_fingerprint_payload(args)
    digest = fingerprint(payload)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / f"{digest}.json"
    lock_path = args.cache_dir / f"{digest}.lock"

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not args.force:
            cached = _load_record(cache_path, digest)
            if cached is not None:
                return "cache-hit", cached, cache_path

        # Only a real measurement needs free GPUs; the cache hit above
        # is served regardless of current GPU occupancy.
        _preflight_free_memory(args.gpus[: args.tp_size])

        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{digest}.", suffix=".probe.json", dir=args.cache_dir
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            try:
                probe_result = _run_probe(args, temporary)
            except CalibrationFailure as exc:
                failure_path = _write_failure_log(
                    args.cache_dir, digest, exc.reason, exc.detail
                )
                raise CalibrationFailure(
                    f"{exc.reason}; complete output: {failure_path}", ""
                ) from exc
            record = {
                "schema": SCHEMA_VERSION,
                "fingerprint": digest,
                "fingerprint_payload": payload,
                "created_unix": time.time(),
                "probe": probe_result,
            }
            encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return "measured", record, cache_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--dcp-size", type=int, required=True)
    parser.add_argument("--indexer-shards", type=int, required=True)
    parser.add_argument("--gpus", type=_comma_ints, required=True)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--tp-rows", type=int, default=8192)
    parser.add_argument("--ckv-record-bytes", type=int, default=656)
    parser.add_argument(
        "--context-tokens", type=_comma_ints, default=(8192, 65536, 131072)
    )
    parser.add_argument(
        "--allreduce-rows",
        type=_comma_ints,
        default=(1, 8, 32, 128, 512, 2048, 8192),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("XDG_CACHE_HOME", "/cache")) / "pcie-calibration",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.tp_size < 2 or args.dcp_size < 1 or args.tp_size % args.dcp_size:
        print(
            "tp-size must be at least 2 and divisible by positive dcp-size",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if (
        args.indexer_shards < 1
        or args.dcp_size % args.indexer_shards
        or args.tp_size % args.indexer_shards
    ):
        print("indexer-shards must divide TP and DCP", file=sys.stderr)
        raise SystemExit(2)
    if args.timeout <= 0:
        print("timeout must be positive", file=sys.stderr)
        raise SystemExit(2)

    try:
        status, record, cache_path = calibrate(args)
    except CalibrationSkip as exc:
        print(f"PCIe calibration preflight: skipped ({exc})", file=sys.stderr)
        return EXIT_SKIPPED_PREFLIGHT
    except CalibrationFailure as exc:
        print(f"PCIe calibration: {exc.reason}", file=sys.stderr)
        return EXIT_PROBE_FAILED
    values = validate_probe_result(record["probe"])
    print(
        "\t".join(
            (
                status,
                str(values["prefetch_depth"]),
                str(values["query_split"]),
                str(values["query_split_min_context_tokens"]),
                str(values["dma_min_bytes"]),
                str(cache_path),
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
