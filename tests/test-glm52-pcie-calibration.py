from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "launchers" / "glm52-pcie-calibration.py"
SPEC = importlib.util.spec_from_file_location("glm52_pcie_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def _result(*, wire_mode: str = "bf16") -> dict:
    return {
        "policy": {
            "numeric_contract": "lossless-only",
            "compressed_dma_requires_explicit_opt_in": True,
            "dcp_ckv_prefetch_depth": 1,
            "dcp_query_split": 1,
            "dcp_query_split_min_context_tokens": 8192,
            "tp_allreduce": {
                "dma_wire_mode": wire_mode,
                "dma_min_bytes": 24 * 1024 * 1024,
            },
        }
    }


def _result_without_dma_crossover() -> dict:
    result = _result()
    result["policy"]["tp_allreduce"]["dma_min_bytes"] = 0
    return result


def test_validate_probe_result_accepts_only_lossless_policy() -> None:
    assert calibration.validate_probe_result(_result()) == {
        "prefetch_depth": 1,
        "query_split": 1,
        "query_split_min_context_tokens": 8192,
        "dma_min_bytes": 24 * 1024 * 1024,
    }

    with pytest.raises(ValueError, match="BF16 DMA"):
        calibration.validate_probe_result(_result(wire_mode="fp8-ring"))

    assert (
        calibration.validate_probe_result(_result_without_dma_crossover())[
            "dma_min_bytes"
        ]
        == "off"
    )


def test_validate_probe_result_rejects_unsafe_or_inconsistent_policy() -> None:
    result = _result()
    result["policy"]["compressed_dma_requires_explicit_opt_in"] = False
    with pytest.raises(ValueError, match="compressed DMA"):
        calibration.validate_probe_result(result)

    result = _result()
    result["policy"]["dcp_query_split_min_context_tokens"] = 0
    with pytest.raises(ValueError, match="inconsistent query-split"):
        calibration.validate_probe_result(result)

    result = _result()
    result["policy"]["dcp_query_split"] = 0
    with pytest.raises(ValueError, match="inconsistent query-split"):
        calibration.validate_probe_result(result)


@pytest.mark.parametrize(
    ("value", "expected"),
    (("4", 4), ("N/A", None), ("[N/A]", None)),
)
def test_optional_nvidia_int(value: str, expected: int | None) -> None:
    assert calibration._optional_nvidia_int(value) == expected


def _probe_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tp_size=2,
        dcp_size=1,
        indexer_shards=1,
        hidden_size=6144,
        tp_rows=8192,
        ckv_record_bytes=656,
        context_tokens=(8192,),
        allreduce_rows=(1,),
        gpus=(0, 1),
        timeout=5.0,
        cache_dir=tmp_path,
    )


def test_run_probe_reports_timeout_with_captured_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class TimeoutProcess:
        pid = 4242
        returncode = -signal.SIGTERM

        def __init__(self) -> None:
            self.communicate_calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, None]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired("probe", 5.0, output="probe stalled")
            return "probe stalled\nworkers stopped\n", None

    process = TimeoutProcess()
    signals: list[tuple[int, int]] = []
    popen_kwargs: dict[str, object] = {}

    def popen(*args: object, **kwargs: object) -> TimeoutProcess:
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(calibration.subprocess, "Popen", popen)
    monkeypatch.setattr(
        calibration.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )

    with pytest.raises(calibration.CalibrationFailure) as failure:
        calibration._run_probe(_probe_args(tmp_path), tmp_path / "result.json")
    assert failure.value.reason == "probe timed out after 5s"
    assert failure.value.detail == "probe stalled\nworkers stopped\n"
    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.communicate_calls == 2
    assert popen_kwargs["start_new_session"] is True


def test_run_probe_reports_invalid_result_with_probe_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class CompletedProcess:
        pid = 4242
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, None]:
            return "probe complete\n", None

    monkeypatch.setattr(
        calibration.subprocess,
        "Popen",
        lambda *args, **kwargs: CompletedProcess(),
    )
    output = tmp_path / "result.json"
    output.write_text("not JSON", encoding="utf-8")

    with pytest.raises(calibration.CalibrationFailure) as failure:
        calibration._run_probe(_probe_args(tmp_path), output)
    assert failure.value.reason.startswith("probe produced no valid result")
    assert failure.value.detail == "probe complete\n"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--tp-size", "1", "--dcp-size", "1", "--indexer-shards", "1"],
            "tp-size must be at least 2 and divisible by positive dcp-size",
        ),
        (
            ["--tp-size", "8", "--dcp-size", "4", "--indexer-shards", "3"],
            "indexer-shards must divide TP and DCP",
        ),
        (
            [
                "--tp-size",
                "8",
                "--dcp-size",
                "4",
                "--indexer-shards",
                "2",
                "--timeout",
                "0",
            ],
            "timeout must be positive",
        ),
    ],
)
def test_manual_validation_uses_usage_exit_contract(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        calibration.main([*arguments, "--gpus", "0,1,2,3,4,5,6,7"])

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{message}\n"


def test_fingerprint_is_order_sensitive() -> None:
    first = {"gpu_order": [{"uuid": "a"}, {"uuid": "b"}]}
    second = {"gpu_order": [{"uuid": "b"}, {"uuid": "a"}]}

    assert calibration.fingerprint(first) != calibration.fingerprint(second)


def test_cold_probe_timeout_allows_slow_hosts() -> None:
    parser = calibration._build_parser()
    args = parser.parse_args(
        [
            "--tp-size",
            "8",
            "--dcp-size",
            "4",
            "--indexer-shards",
            "2",
            "--gpus",
            "0,2,4,6,1,3,5,7",
        ]
    )

    assert args.timeout == 600.0


def test_collective_environment_tracks_only_relevant_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "8")
    monkeypatch.setenv("SPARKINFER_PCIE_DMA_PIECES", "2")
    monkeypatch.setenv("UNRELATED_VALUE", "ignored")

    environment = calibration._collective_environment()

    assert environment["NCCL_MIN_NCHANNELS"] == "8"
    assert environment["SPARKINFER_PCIE_DMA_PIECES"] == "2"
    assert "UNRELATED_VALUE" not in environment


def test_cache_rejects_wrong_fingerprint_and_numeric_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    record = {
        "schema": calibration.SCHEMA_VERSION,
        "fingerprint": "expected",
        "probe": _result(),
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    assert calibration._load_record(path, "expected") == record
    assert calibration._load_record(path, "different") is None

    record["probe"]["policy"]["numeric_contract"] = "compressed"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert calibration._load_record(path, "expected") is None
