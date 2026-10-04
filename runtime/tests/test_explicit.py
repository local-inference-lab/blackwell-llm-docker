"""Materialized exports preserve native argv, cache geometry and environment."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from runtime import ConfigError
from runtime.explicit import document, load_plan
from runtime.image_install import install
from runtime.launcher import ROOT, resolve

IDENTITY = "a" * 64
CASES = [
    ("glm53-flash", "off"),
    ("glm53-flash", "mtp"),
    ("glm53-flash", "dflash2"),
    ("ds4-flash", "dspark"),
    ("ds4-vision", "dspark"),
    ("ds41-flash", "dspark"),
    ("qwen38-flash-next", "mtp"),
]


@pytest.mark.parametrize("identifier,mode", CASES)
@pytest.mark.parametrize(
    "cache_mode,l2", [("vram", False), ("lmcache", False), ("lmcache", True)]
)
@pytest.mark.parametrize("hardware", ["native", "rtx-pro-6000-pcie"])
def test_explicit_exports_preserve_resolved_execution(
    identifier, mode, cache_mode, l2, hardware
):
    kwargs = {
        "env": {},
        "config": {
            "options": {"mode": mode, "cache-mode": cache_mode, "cache-l2-enabled": l2}
        },
    }
    unbound = resolve(identifier, hardware, **kwargs)
    bound = resolve(identifier, hardware, runtime_identity=IDENTITY, **kwargs)
    frozen = document(unbound)
    saved = copy.deepcopy(frozen)
    imported = load_plan(frozen, runtime_identity=IDENTITY)
    assert frozen == saved
    assert imported.argv == bound.argv
    assert imported.environment == bound.environment
    assert (asdict(imported.cache_service) if imported.cache_service else None) == (
        asdict(bound.cache_service) if bound.cache_service else None
    )
    assert not any(
        "UNBOUND-RUNTIME" in value for value in imported.environment.values()
    )


@pytest.mark.parametrize("cache", ["native", "lmcache"])
def test_direct_cache_transport_and_native_offload(cache):
    kwargs = {"env": {"CACHE_MODE": cache, "LMCACHE_TRANSFER_MODE": "lmcache_driven"}}
    unbound = resolve("glm53-flash", **kwargs)
    expected = resolve("glm53-flash", runtime_identity=IDENTITY, **kwargs)
    actual = load_plan(document(unbound), runtime_identity=IDENTITY)
    assert actual.argv == expected.argv
    assert actual.environment == expected.environment
    if cache == "lmcache":
        assert asdict(actual.cache_service) == asdict(expected.cache_service)


def test_dflash_slots_are_materialized_once():
    original = resolve(
        "glm53-flash", env={"SPECULATOR": "dflash2", "CACHE_MODE": "lmcache"}
    )
    frozen = document(original)
    assert frozen["options"]["max-num-batched-tokens"] == 4320
    assert frozen["options"]["cache-target-tokens"] == 4096
    loaded = load_plan(frozen, runtime_identity=IDENTITY)
    assert loaded.values["max-num-batched-tokens"] == 4320
    assert loaded.values["max-num-scheduled-tokens"] == 4096


def test_conflicting_cache_port_and_native_connector_is_rejected():
    frozen = document(resolve("glm53-flash", env={"CACHE_MODE": "lmcache"}))
    frozen["options"]["cache-port"] += 10
    with pytest.raises(ConfigError, match="conflict with cache controls"):
        load_plan(frozen, runtime_identity=IDENTITY)


@pytest.mark.parametrize("cache_mode", ["vram", "lmcache", "native"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("kv-transfer-config", {"kv_connector": "UnselectedConnector"}),
        ("kv-offloading-backend", "unselected"),
        ("kv-offloading-size", 1),
        ("enable-cumem-allocator", False),
        ("max-num-scheduled-tokens", 1),
    ],
)
def test_generated_options_cannot_override_or_extend_the_selected_cache_mode(
    cache_mode, key, value
):
    frozen = document(resolve("glm53-flash", env={"CACHE_MODE": cache_mode}))
    frozen["options"][key] = value
    with pytest.raises(ConfigError, match="conflict with cache controls"):
        load_plan(frozen, runtime_identity=IDENTITY)


@pytest.mark.parametrize(
    "argument",
    [
        "--kv-transfer-config.kv_connector=UnselectedConnector",
        "--kv-offloading-backend=native",
        "--kv-offloading-size=1",
        "--enable-cumem-allocator",
        "--max-num-scheduled-tokens=1",
    ],
)
def test_generated_options_cannot_bypass_controls_through_passthrough(argument):
    frozen = document(resolve("glm53-flash", env={"CACHE_MODE": "lmcache"}))
    frozen["passthrough"] = [argument]
    with pytest.raises(ConfigError, match="cannot bypass profile validation"):
        load_plan(frozen, runtime_identity=IDENTITY)


@pytest.mark.parametrize(
    "cache_mode,key",
    [
        ("lmcache", "kv-transfer-config"),
        ("lmcache", "max-num-scheduled-tokens"),
        ("native", "kv-offloading-backend"),
        ("native", "enable-cumem-allocator"),
    ],
)
def test_missing_generated_options_require_regenerating_the_export(cache_mode, key):
    frozen = document(resolve("glm53-flash", env={"CACHE_MODE": cache_mode}))
    del frozen["options"][key]
    with pytest.raises(ConfigError, match="conflict with cache controls"):
        load_plan(frozen, runtime_identity=IDENTITY)


def test_all_image_environment_is_materialized_without_profile_reresolution():
    original = resolve("qwen38-flash-next", env={})
    frozen = document(
        original,
        image_environment={
            "PATH": "/opt/venv/bin:/usr/bin",
            "CUSTOM_IMAGE_DEFAULT": "yes",
        },
    )
    loaded = load_plan(frozen, runtime_identity=IDENTITY)
    assert loaded.environment["CUSTOM_IMAGE_DEFAULT"] == "yes"
    assert loaded.environment["PATH"] == "/opt/venv/bin:/usr/bin"


@pytest.mark.parametrize("location", ["environment", "passthrough", "json"])
def test_credentials_are_not_portable_export_content(location):
    frozen = document(resolve("qwen38-flash-next", env={}))
    if location == "environment":
        frozen["environment"]["HF_TOKEN"] = "private-value"
    elif location == "passthrough":
        frozen["passthrough"] = ["--api-key", "private-value"]
    else:
        frozen["options"].setdefault("additional-config", {})["access_token"] = (
            "private-value"
        )
    with pytest.raises(ConfigError, match="must not contain credentials"):
        load_plan(frozen, runtime_identity=IDENTITY)


def test_missing_required_options_and_empty_nccl_graph_are_rejected():
    frozen = document(resolve("qwen38-flash-next", env={}))
    del frozen["options"]["tensor-parallel-size"]
    with pytest.raises(ConfigError, match="missing required option"):
        load_plan(frozen, runtime_identity=IDENTITY)
    frozen = document(resolve("qwen38-flash-next", env={}))
    frozen["environment"]["NCCL_GRAPH_FILE"] = ""
    with pytest.raises(ConfigError, match="NCCL_GRAPH_FILE must be absent"):
        load_plan(frozen, runtime_identity=IDENTITY)


def test_unset_native_defaults_are_named_without_inventing_values():
    frozen = document(resolve("ds4-flash", env={}))
    assert "linear-backend" in frozen["vllm_defaults"]
    assert "linear-backend" not in frozen["options"]


def test_compose_environment_is_the_single_editable_owner():
    frozen = document(resolve("glm53-flash", "rtx-pro-6000-pcie", env={}))
    environment = frozen.pop("environment")
    frozen["environment_keys"] = sorted(environment)
    environment["NCCL_MIN_NCHANNELS"] = "12"
    loaded = load_plan(
        frozen, runtime_identity=IDENTITY, incoming_environment=environment
    )
    assert loaded.environment["NCCL_MIN_NCHANNELS"] == "12"
    with pytest.raises(ConfigError, match="Missing exported environment"):
        load_plan(frozen, runtime_identity=IDENTITY, incoming_environment={})


def test_native_page_environment_cannot_disagree_with_geometry():
    frozen = document(resolve("glm53-flash", env={}))
    frozen["environment"]["VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE"] = "64"
    with pytest.raises(ConfigError, match="explicit options"):
        load_plan(frozen, runtime_identity=IDENTITY)


@pytest.mark.parametrize("identifier,mode", CASES)
@pytest.mark.parametrize("cache_mode", ["vram", "lmcache"])
def test_installed_explicit_cli_preserves_profile_execution(
    tmp_path, identifier, mode, cache_mode
):
    package = tmp_path / "installed" / "runtime"
    install(
        {"Id": "sha256:" + "1" * 64, "Config": {"Env": ["PATH=/usr/bin"]}},
        IDENTITY,
        None,
        destination=package,
        bin_directory=tmp_path / "bin",
        python_site=tmp_path / "site",
    )
    environment = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"}
    options = ["--mode", mode, "--cache-mode", cache_mode]
    native = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--profile",
            identifier,
            "--hardware",
            "rtx-pro-6000-pcie",
            *options,
            "--print-config",
        ],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    unbound = resolve(identifier, "rtx-pro-6000-pcie", env={}, argv=options)
    exported = document(unbound)
    incoming = exported.pop("environment")
    exported["environment_keys"] = sorted(incoming)
    explicit = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.explicit",
            "--config-json",
            json.dumps(exported),
            "--print-config",
        ],
        cwd=package.parent,
        env={**environment, **incoming},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    expected, actual = json.loads(native.stdout), json.loads(explicit.stdout)
    assert actual["argv"] == expected["argv"]
    assert actual["cache_service"] == expected["cache_service"]
    assert {k: v["value"] for k, v in actual["environment"].items()} == {
        k: v["value"] for k, v in expected["environment"].items()
    }


def test_installed_explicit_execution_invokes_bootstrap_once(tmp_path):
    package = tmp_path / "installed" / "runtime"
    bootstrap = tmp_path / "bootstrap"
    bootstrap.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], "
        "'nccl': os.environ['NCCL_MIN_NCHANNELS']}))\n"
    )
    bootstrap.chmod(0o755)
    install(
        {"Id": "sha256:" + "1" * 64, "Config": {"Env": ["PATH=/usr/bin"]}},
        IDENTITY,
        str(bootstrap),
        destination=package,
        bin_directory=tmp_path / "bin",
        python_site=tmp_path / "site",
    )
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={"CHECKPOINT": "original"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.explicit",
            "--config-json",
            json.dumps(document(plan)),
        ],
        cwd=package.parent,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    # The installed runtime serves the chat template from its own copy.
    templates = str(ROOT / "templates")
    assert json.loads(result.stdout) == {
        "argv": [
            arg.replace(templates, str(package / "templates")) for arg in plan.argv
        ],
        "nccl": plan.environment["NCCL_MIN_NCHANNELS"],
    }
