"""Validate sidecar bind arguments without starting a cache or model server."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "serve-glm53-flash-lmcache.sh",
    "serve-glm53-flash-lmcache-cache-complete.sh",
)


@pytest.mark.parametrize("semantic", [False, True])
def test_semantic_model_budget_is_independent_of_storage_object_size(semantic):
    source = (RECIPE / "serve-glm53-flash-lmcache-cache-complete.sh").read_text()
    source = source.replace("/opt/venv/bin/python", shlex.quote(sys.executable))
    configuration, separator, _ = source.partition('"${lmcache_server_command[@]}" &')
    assert separator
    identity = (
        json.dumps(
            {
                "target_revision": "a" * 40,
                "source_revision": "b" * 40,
                "draft_revision": "a" * 40,
            }
        )
        if semantic
        else ""
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            configuration
            + '\nprintf "%s\\n" "${MAX_NUM_BATCHED_TOKENS}" "${vllm_extra_args[@]}"',
        ],
        env={
            "PATH": os.environ["PATH"],
            "LMCACHE_ENABLED": "1",
            "LMCACHE_L2_ENABLED": "0",
            "LMCACHE_MIN_SHM_GIB": "1",
            "LMCACHE_TRANSFER_MODE": "engine_driven",
            "LMCACHE_CHUNK_SIZE": "4096",
            "LMCACHE_TARGET_TOKEN_BUDGET": "3072",
            "LMCACHE_CHECKPOINT_IDENTITY": identity,
            "SPECULATOR": "mtp",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if semantic:
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "3072",
            "--max-num-scheduled-tokens",
            "3072",
        ]
    else:
        assert result.returncode == 2
        assert "Aligned LMCache requires matching" in result.stderr


def render(wrapper, host=None, dtype=None, settings=None, cwd=None):
    source = (RECIPE / wrapper).read_text()
    # Use the test environment's interpreter for JSON validation; no LMCache
    # or vLLM package is imported while rendering arguments.
    source = source.replace("/opt/venv/bin/python", shlex.quote(sys.executable))
    # The configuration section validates settings and constructs argv. Stop
    # before child-process supervision; no sidecar or vLLM process is launched.
    configuration, separator, _ = source.partition("lmcache_pid=\n")
    assert separator
    if dtype is not None:
        # The standalone wrapper resolves vLLM's dtype after defining its
        # supervisor functions, but before broker or child-process creation.
        configuration, separator, _ = source.partition(
            "if [[ ${transfer_mode} != engine_driven ]]; then"
        )
        assert separator
    environment = {
        "PATH": os.environ["PATH"],
        "LMCACHE_ENABLED": "1",
        "LMCACHE_L2_ENABLED": "0",
        "LMCACHE_MIN_SHM_GIB": "1",
    }
    if host is not None:
        environment["LMCACHE_HTTP_HOST"] = host
    if dtype is not None:
        environment["LMCACHE_KV_CACHE_DTYPE"] = dtype
    environment.update(settings or {})
    return subprocess.run(
        [
            "bash",
            "-c",
            configuration
            + '\nprintf "%s\\0" "${health_url}" "${lmcache_server[@]}" "vllm_dtype=${KV_CACHE_DTYPE:-}"',
        ],
        env=environment,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("transfer", ["engine_driven", "lmcache_driven", "auto"])
def test_engine_transport_has_a_stable_shared_memory_arena(wrapper, transfer):
    result = render(
        wrapper,
        settings={
            "LMCACHE_TRANSFER_MODE": transfer,
            "LMCACHE_INSTANCE_ID": "glm/test",
            "LMCACHE_MP_PORT": "5566",
        },
    )
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    if transfer == "engine_driven":
        assert fields.count("--no-l1-use-lazy") == 1
        assert "--l1-use-lazy" not in fields
        assert fields.count("--shm-name") == 1
        assert fields[fields.index("--shm-name") + 1] == "lmcache-glm_test-5566"
    else:
        assert fields.count("--l1-use-lazy") == 1
        assert "--no-l1-use-lazy" not in fields
        assert "--shm-name" not in fields


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_engine_transport_preserves_explicit_shared_memory_name(wrapper):
    result = render(
        wrapper,
        settings={
            "LMCACHE_TRANSFER_MODE": "engine_driven",
            "LMCACHE_SHM_NAME": "glm-checkpoints.1",
        },
    )
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert fields[fields.index("--shm-name") + 1] == "glm-checkpoints.1"


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("name", ["a/b", "a b", "$(id)"])
def test_invalid_shared_memory_name_fails_before_server_start(wrapper, name):
    result = render(
        wrapper,
        settings={"LMCACHE_TRANSFER_MODE": "engine_driven", "LMCACHE_SHM_NAME": name},
    )
    assert result.returncode == 2
    assert "LMCACHE_SHM_NAME" in result.stderr


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("policy", [None, "default", "retain"])
def test_prefetch_retention_defaults_to_reusable_ram(wrapper, policy):
    settings = {} if policy is None else {"LMCACHE_L2_PREFETCH_POLICY": policy}
    result = render(wrapper, settings=settings)
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert fields.count("--l2-prefetch-policy") == 1
    assert fields[fields.index("--l2-prefetch-policy") + 1] == (policy or "retain")


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("policy", [None, "default", "retain"])
@pytest.mark.parametrize("l2_enabled", ["0", "1"])
def test_retained_filesystem_restore_can_reclaim_unowned_ram(
    wrapper, policy, l2_enabled, tmp_path
):
    settings = {"LMCACHE_L2_ENABLED": l2_enabled, "LMCACHE_L2_PATH": str(tmp_path)}
    if policy is not None:
        settings["LMCACHE_L2_PREFETCH_POLICY"] = policy
    result = render(wrapper, settings=settings)
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    expected = l2_enabled == "1" and policy != "default"
    assert fields.count("--emergency-evict-for-prefetch") == int(expected)
    assert "--write-back-on-evict" not in fields


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("policy", ["all", "Retain", "retain default"])
def test_invalid_retention_policy_fails_before_server_start(wrapper, policy):
    result = render(wrapper, settings={"LMCACHE_L2_PREFETCH_POLICY": policy})
    assert result.returncode == 2
    assert "LMCACHE_L2_PREFETCH_POLICY" in result.stderr


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_server_arguments_preserve_literal_wildcards_without_execution(
    wrapper, tmp_path
):
    (tmp_path / "match.yaml").touch()
    result = render(
        wrapper,
        settings={
            "LMCACHE_SERVER_EXTRA_ARGS": "--max-cpu-workers 4 --pattern *.yaml $(touch${IFS}EXECUTED)"
        },
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert fields[-6:-1] == [
        "--max-cpu-workers",
        "4",
        "--pattern",
        "*.yaml",
        "$(touch${IFS}EXECUTED)",
    ]
    assert not (tmp_path / "EXECUTED").exists()


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "argument",
    [
        "--shm-name other",
        "--chunk-size=512",
        "--no-separate-object-groups",
        "--http-port 9999",
        "--l2-prefetch-policy=default",
        "--no-emergency-evict-for-prefetch",
    ],
)
def test_extra_arguments_cannot_override_transfer_or_readiness_contract(
    wrapper, argument
):
    result = render(wrapper, settings={"LMCACHE_SERVER_EXTRA_ARGS": argument})
    assert result.returncode == 2
    assert "launcher-managed option" in result.stderr


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "arguments", ["--max-cpu-workers 4\n--max-workers 8", "--max-workers 4\r"]
)
def test_extra_arguments_reject_silently_truncated_lines(wrapper, arguments):
    result = render(wrapper, settings={"LMCACHE_SERVER_EXTRA_ARGS": arguments})
    assert result.returncode == 2
    assert "one whitespace-separated line" in result.stderr


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "host,expected,probe,warning",
    [
        (None, "127.0.0.1", "127.0.0.1", False),
        ("0.0.0.0", "0.0.0.0", "127.0.0.1", True),
        ("192.0.2.10", "192.0.2.10", "192.0.2.10", True),
        ("::", "::", "[::1]", True),
        ("::1", "::1", "[::1]", False),
        ("cache.internal", "cache.internal", "cache.internal", True),
        ("127.example.internal", "127.example.internal", "127.example.internal", True),
        ("127.0.0.2", "127.0.0.2", "127.0.0.2", False),
        ("127.999.0.1", "127.999.0.1", "127.999.0.1", True),
    ],
)
def test_http_bind_and_readiness_use_compatible_addresses(
    wrapper, host, expected, probe, warning
):
    result = render(wrapper, host)
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert fields[0] == f"http://{probe}:8085/healthcheck"
    assert fields.count("--http-host") == 1
    assert fields[fields.index("--http-host") + 1] == expected
    assert ("administrative APIs" in result.stderr) == warning


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize(
    "host", ["--host", "localhost;id", "a b", "$(id)", "a/b", "a\nb"]
)
def test_invalid_bind_is_rejected_before_process_start(wrapper, host):
    result = render(wrapper, host)
    assert result.returncode == 2
    assert "LMCACHE_HTTP_HOST" in result.stderr


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("fp8_ds_mla", "fp8"),
        ("fp8", "fp8"),
        ("fp8_e4m3", "fp8_e4m3"),
        ("nvfp4_ds_mla", "nvfp4_ds_mla"),
    ],
)
def test_standalone_cache_storage_dtype_maps_to_vllm(dtype, expected):
    result = render("serve-glm53-flash-lmcache.sh", dtype=dtype)
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\0").split("\0")[-1] == f"vllm_dtype={expected}"


@pytest.mark.parametrize(
    "transfer,resolved",
    [("engine_driven", "request_boundaries"), ("lmcache_driven", "aligned")],
)
@pytest.mark.parametrize(
    "policy_args",
    [
        [],
        ["--recurrent-checkpoint-policy", "auto"],
        ["--recurrent-checkpoint-policy=auto"],
    ],
)
def test_cache_policy_is_resolved_once_before_delegation(
    transfer, resolved, policy_args
):
    unrelated = ["--max-model-len", "65536", "--served-model-name", "model with spaces"]
    result = subprocess.run(
        [
            "bash",
            str(RECIPE / "serve-glm53-flash-cache-complete.sh"),
            *policy_args,
            *unrelated,
        ],
        env={
            "PATH": os.environ["PATH"],
            "CACHE_CONFIG_DRY_RUN": "1",
            "CACHE_MODE": "lmcache",
            "LMCACHE_TRANSFER_MODE": transfer,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("ARGV:"))
    args = shlex.split(line.removeprefix("ARGV:"))
    assert args == unrelated + ["--recurrent-checkpoint-policy", resolved]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--recurrent-checkpoint-policy", "request_boundaries"],
        ["--recurrent-checkpoint-policy=request_boundaries"],
    ],
)
def test_semantic_sidecar_does_not_duplicate_native_checkpoint_policy(args):
    source = (RECIPE / "serve-glm53-flash-lmcache-cache-complete.sh").read_text()
    configuration, separator, _ = source.partition("lmcache_pid=\n")
    assert separator
    _, separator, connector = source.partition("connector_config=$(printf")
    assert separator
    connector, separator, _ = connector.partition('"${base_launcher}" "$@"')
    assert separator
    script = (
        configuration
        + "\nvllm_extra_args=()\nconnector_config=$(printf"
        + connector
        + '\nprintf "%s\\0" "$@" "${vllm_extra_args[@]}"'
    )
    result = subprocess.run(
        ["bash", "-c", script, "policy-test", *args],
        env={
            "PATH": os.environ["PATH"],
            "LMCACHE_ENABLED": "1",
            "LMCACHE_L2_ENABLED": "0",
            "LMCACHE_MIN_SHM_GIB": "1",
            "LMCACHE_TRANSFER_MODE": "engine_driven",
            "LMCACHE_CHECKPOINT_IDENTITY": '{"target_revision":"test"}',
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    fields = result.stdout.rstrip("\0").split("\0")
    assert (
        sum(
            field.split("=", 1)[0] == "--recurrent-checkpoint-policy"
            for field in fields
        )
        == 1
    )


@pytest.mark.parametrize("policy", ["aligned", "request_boundaries"])
@pytest.mark.parametrize("target_identity", ["a" * 64, "c" * 64])
def test_persistent_namespace_uses_effective_model_and_content_identity(
    policy, target_identity
):
    source = (RECIPE / "serve-glm53-flash-cache-complete.sh").read_text()
    resolver = (
        "/opt/venv/bin/python \\\n"
        "          /usr/local/libexec/glm53_checkpoint_identity.py"
    )
    assert source.count(resolver) == 1
    source = source.replace(resolver, "identity_fixture")
    invocation = '\nexec "${cache_launcher}" "$@"'
    assert source.count(invocation) == 1
    source = source.replace(
        invocation,
        'printf "%s\\0" "${LMCACHE_L2_PATH}" '
        '"${LMCACHE_CHECKPOINT_IDENTITY:-}" "${MODEL_REVISION}" "$@"',
    )
    identity = {
        "checkpoint_identity": {
            "target_revision": target_identity,
            "source_revision": "b" * 64,
            "draft_revision": "",
        },
        "model_revision": "",
        "draft_model_revision": "",
    }
    fixture = (
        'identity_fixture() { printf "%s\\0" "$@" >&2; '
        + "printf '%s' "
        + shlex.quote(json.dumps(identity))
        + "; }\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            fixture + source,
            "cache-launcher-test",
            "/models/alternate",
            "--recurrent-checkpoint-policy",
            policy,
        ],
        env={
            "PATH": os.environ["PATH"],
            "CACHE_MODE": "lmcache",
            "LMCACHE_TRANSFER_MODE": "engine_driven",
            "LMCACHE_L2_ENABLED": "1",
            "MODEL": "unused/repository",
            "MTP_DEPTH": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    identity_args = result.stderr.rstrip("\0").split("\0")
    assert identity_args[identity_args.index("--model") + 1] == "/models/alternate"
    fields = result.stdout.rstrip("\0").split("\0")
    assert f"/_models_alternate-{target_identity}/" in fields[0]
    assert "unused" not in fields[0] and "huggingface-main" not in fields[0]
    assert bool(fields[1]) == (policy == "request_boundaries")
    assert fields[2] == ""
    assert fields[3:] == [
        "/models/alternate",
        "--recurrent-checkpoint-policy",
        policy,
    ]
