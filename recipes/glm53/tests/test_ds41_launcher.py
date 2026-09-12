"""Native DS4.1 policy isolation and public Engram storage controls."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "serve-ds41-jovian.sh"


def launch(tmp_path, overrides=None, arguments=()):
    native = tmp_path / "serve-ds41-flash.sh"
    native.write_text(
        '#!/bin/bash\nexec "$PYTHON_BIN" -c '
        "'import json,os,sys; print(json.dumps({"
        '"args":sys.argv[1:],"env":dict(os.environ)}))' + '\' "$@"\n'
    )
    native.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": sys.executable,
        "VLLM_SOURCE_DIR": str(tmp_path),
        **(overrides or {}),
    }
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments], env=env, text=True, capture_output=True
    )


@pytest.mark.parametrize("memory", [None, "ram", "disk"])
def test_engram_storage_is_forwarded_without_general_cpu_offload(tmp_path, memory):
    result = launch(tmp_path, {} if memory is None else {"ENGRAM_TABLE_MEMORY": memory})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["env"]["ENGRAM_TABLE_MEMORY"] == (memory or "disk")
    assert payload["env"]["HOST"] == "0.0.0.0"
    assert payload["env"]["MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert payload["env"]["MODEL_PATH"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert "--cpu-offload-gb" not in payload["args"]


def test_invalid_engram_storage_is_rejected(tmp_path):
    result = launch(tmp_path, {"ENGRAM_TABLE_MEMORY": "ssd"})
    assert result.returncode == 2
    assert "ram or disk" in result.stderr


def test_glm_tuning_is_removed_but_cache_paths_are_preserved(tmp_path):
    result = launch(
        tmp_path,
        {
            "VLLM_GLM53_MTP_DRAFT_HEAD": "nvfp4",
            "B12X_PCIE_ONESHOT_THREADS": "512",
            "B12X_COMPILE_CACHE_DIR": "/cache/b12x",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "NCCL_GRAPH_FILE": "",
        },
    )
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert "VLLM_GLM53_MTP_DRAFT_HEAD" not in env
    assert "B12X_PCIE_ONESHOT_THREADS" not in env
    assert "NCCL_GRAPH_FILE" not in env
    assert env["B12X_COMPILE_CACHE_DIR"] == "/cache/b12x"
    assert env["VLLM_CACHE_ROOT"] == "/cache/vllm"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--override-generation-config", '{"top_p":1}'),
        ('--override-generation-config={"top_p":1}',),
        ("--generation-config", "auto"),
    ],
)
def test_explicit_sampling_configuration_is_not_duplicated(tmp_path, arguments):
    result = launch(tmp_path, arguments=arguments)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["args"]
    assert argv[-len(arguments) :] == list(arguments)
    assert '{"temperature":1.0,"top_p":0.95}' not in argv


def test_sampling_defaults_are_explicit(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["args"]
    assert argv.count("--override-generation-config") == 1
    assert json.loads(argv[argv.index("--override-generation-config") + 1]) == {
        "temperature": 1.0,
        "top_p": 0.95,
    }
