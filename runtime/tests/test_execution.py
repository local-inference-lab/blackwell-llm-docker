"""The resolver must inspect configuration without side effects or secret output."""

import json
import os
import subprocess
import sys

import pytest

from runtime import launcher
from runtime.launcher import ConfigError, ROOT, execute, resolve
from runtime.packaging import make_contract, verify_contract


def test_cli_process_is_cpu_only_and_does_not_execute_unbound_plan():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--profile",
            "ds41-flash",
            "--print-config",
        ],
        cwd=ROOT.parent,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["settings"]["host"]["value"] == "0.0.0.0"
    result = subprocess.run(
        [sys.executable, "-m", "runtime.launcher", "--profile", "ds41-flash"],
        cwd=ROOT.parent,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "build-audited" in result.stderr


def test_model_image_contract_is_checked_and_exec_preserves_private_environment(
    tmp_path, monkeypatch
):
    contract = make_contract(
        {"Id": "sha256:" + "1" * 64, "Config": {"Env": []}}, "2" * 64
    )
    path = tmp_path / "image-contract.json"
    path.write_text(json.dumps(contract))
    verify_contract(path)
    plan = resolve("ds41-flash", env={}, runtime_identity="2" * 64)
    monkeypatch.setenv("HF_TOKEN", "private-value")
    monkeypatch.setenv("TP", "wrong-at-exec")
    monkeypatch.setenv("NCCL_GRAPH_FILE", "")
    called = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: called.append(args))
    execute(plan, path)
    executable, args, env = called[0]
    assert executable == args[0] == "/opt/venv/bin/python"
    assert env["HF_TOKEN"] == "private-value"
    assert "TP" not in env
    assert "NCCL_GRAPH_FILE" not in env
    assert env["OMP_NUM_THREADS"] == "8"
    contract["files"]["profiles/ds41-flash.yaml"] = "0" * 64
    path.write_text(json.dumps(contract))
    with pytest.raises(ConfigError, match="hashes differ"):
        execute(plan, path)


def test_sampling_file_explicitly_replaces_profile_defaults():
    plan = resolve("glm53-flash", env={}, argv=["--generation-config", "auto"])
    assert "override-generation-config" not in plan.values
    combined = resolve(
        "glm53-flash",
        env={},
        argv=["--generation-config", "auto", "--override-generation-config.top_p=1"],
    )
    assert combined.values["override-generation-config"] == {"top_p": 1}
