"""Check the portable TP2 profile without weights, GPUs or network access."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    base = tmp_path / "native.sh"
    base.write_text((RECIPE / "serve-glm53-flash-nvfp4-dflash2.sh").read_text())
    base.chmod(0o755)
    scheduler = tmp_path / "scheduler.sh"
    scheduler.write_text(
        (RECIPE / "serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh")
        .read_text()
        .replace(
            "readonly base_launcher=/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh",
            "readonly base_launcher=" + shlex.quote(str(base)),
        )
    )
    scheduler.chmod(0o755)
    wrapper = tmp_path / "tp2.sh"
    wrapper.write_text(
        (RECIPE / "serve-glm-spark-tp2.sh")
        .read_text()
        .replace(
            "/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh",
            shlex.quote(str(scheduler)),
        )
    )

    def run(environment=None, arguments=()):
        return subprocess.run(
            ["bash", str(wrapper), *arguments],
            env={"PATH": os.environ["PATH"], "DRY_RUN": "1", **(environment or {})},
            text=True,
            capture_output=True,
            check=False,
        )

    return run


def rendered(result):
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout)[3:]


def test_capacity_profile_and_b12x_defaults(launch):
    args = rendered(launch())
    values = {
        "--host": "0.0.0.0",
        "--port": "8000",
        "--tensor-parallel-size": "2",
        "--decode-context-parallel-size": "2",
        "--max-model-len": "1048576",
        "--max-num-batched-tokens": "3072",
        "--max-num-seqs": "4",
        "--kv-cache-memory-bytes": "4294967296",
        "--kv-cache-dtype": "fp8_ds_mla",
        "--moe-backend": "b12x",
        "--linear-backend": "b12x",
        "--limit-mm-per-prompt": '{"image":1,"video":0}',
        "--recurrent-checkpoint-policy": "request_boundaries",
    }
    for flag, value in values.items():
        assert args.count(flag) == 1
        assert args[args.index(flag) + 1] == value
    assert "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark" in args
    import json

    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec["method"] == "mtp" and spec["num_speculative_tokens"] == 3
    assert spec["moe_backend"] == "b12x"
    assert spec["draft_sample_method"] == "probabilistic"


@pytest.mark.parametrize("equals", [False, True])
def test_explicit_cli_defaults_are_not_duplicated(launch, equals):
    values = {
        "--kv-cache-memory-bytes": "4000000000",
        "--limit-mm-per-prompt": '{"image":0,"video":0}',
        "--recurrent-checkpoint-policy": "aligned",
    }
    supplied = []
    for key, value in values.items():
        supplied.extend([f"{key}={value}"] if equals else [key, value])
    args = rendered(launch(arguments=supplied))
    for key, value in values.items():
        assert sum(arg == key or arg.startswith(key + "=") for arg in args) == 1
        assert (f"{key}={value}" if equals else value) in args


@pytest.mark.parametrize(
    "environment",
    [{"TP": "4"}, {"DCP": "1"}, {"SPECULATOR": "dflash"}, {"CACHE_MODE": "lmcache"}],
)
def test_unqualified_profiles_fail_with_a_clear_message(launch, environment):
    result = launch(environment)
    assert result.returncode == 2
    assert "profile" in result.stderr or "entrypoint" in result.stderr


def test_help_describes_portable_profile_and_no_clock_mutation(launch):
    result = launch(arguments=["--help"])
    assert result.returncode == 0
    assert "0.0.0.0" in result.stdout
    assert "overclocking is never applied" in result.stdout


def test_specialization_adds_no_filesystem_layer():
    instructions = (RECIPE / "Dockerfile.glm-spark-tp2").read_text().splitlines()
    assert not any(line.startswith(("RUN ", "COPY ", "ADD ")) for line in instructions)
    installer = (RECIPE / "install_glm53_source_locked.sh").read_text()
    assert "install -Dm755 /build-inputs/serve-glm-spark-tp2.sh" in installer
