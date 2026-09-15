"""Resolve model policy without importing vLLM, CUDA, or model checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from runtime import ConfigError

ROOT = Path(__file__).resolve().parent
JIT_PATHS = {
    "XDG_CACHE_HOME": "",
    "VLLM_CACHE_ROOT": "vllm",
    "VLLM_CACHE_DIR": "vllm",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUTE_DSL_CACHE_DIR": "cute-dsl",
    "B12X_CUTE_COMPILE_CACHE_DIR": "b12x/cute",
    "B12X_COMPILE_CACHE_DIR": "b12x/compile",
    "SPARKINFER_COMPILE_CACHE_DIR": "b12x/compile",
    "CUDA_CACHE_PATH": "cuda",
}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET = re.compile(
    r"api[-_]?key|password|secret|authorization|access[-_]?token|hf_token", re.I
)


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently selecting a value."""


# YAML 1.2 booleans: mode names such as "off" are strings, not boolean keys.
UniqueLoader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in rules if tag != "tag:yaml.org,2002:bool"]
    for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.I), list("tTfF")
)


def _mapping(loader: UniqueLoader, node: yaml.MappingNode, deep: bool = False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict:
    try:
        result = yaml.load(path.read_text(), Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Cannot read configuration {path}: {error}") from error
    if not isinstance(result, dict):
        raise ConfigError(f"Expected a mapping in {path}")
    return result


def profile(kind: str, identifier: str) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", identifier):
        raise ConfigError(
            "Profile names must contain only lowercase letters, digits and hyphens"
        )
    path = (
        ROOT / ("hardware" if kind == "hardware" else "profiles") / f"{identifier}.yaml"
    )
    result = read_yaml(path)
    try:
        jsonschema.validate(result, json.loads((ROOT / "schema.json").read_text()))
    except jsonschema.ValidationError as error:
        raise ConfigError(f"Invalid profile {identifier}: {error.message}") from error
    return result


def convert(name: str, value: Any, spec: dict) -> Any:
    kind = spec["type"]
    try:
        if kind == "string":
            value = str(value)
        elif kind == "integer":
            if isinstance(value, bool) or not re.fullmatch(r"[+-]?\d+", str(value)):
                raise ValueError
            value = int(value)
        elif kind == "int-or-auto":
            if str(value) not in {"auto", "None"}:
                value = convert(name, value, {"type": "integer"})
        elif kind == "number":
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
        elif kind == "boolean":
            normalized = str(value).lower()
            if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raise ValueError
            value = normalized in {"true", "1", "yes", "on"}
        elif kind == "object":
            value = json.loads(value) if isinstance(value, str) else value
            if not isinstance(value, dict):
                raise ValueError
        elif kind == "integers":
            if isinstance(value, str):
                value = value.replace(",", " ").split()
            if not isinstance(value, list) or not value:
                raise ValueError
            value = [convert(name, item, {"type": "integer"}) for item in value]
        else:
            raise ConfigError(f"Unknown option type {kind}")
    except (ValueError, TypeError) as error:
        raise ConfigError(f"Invalid value for {name}; expected {kind}") from error
    if "enum" in spec and value not in spec["enum"]:
        raise ConfigError(f"{name} must be one of {', '.join(spec['enum'])}")
    return value


def parse_native(argv: list[str], specs: dict) -> tuple[dict, list[str]]:
    """Consume managed options; retain unknown native options without shell parsing."""
    values: dict[str, Any] = {}
    passthrough: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        index += 1
        if argument == "--":
            continue
        if not argument.startswith("--"):
            raise ConfigError(
                "Use --model for the checkpoint; native options must start with --"
            )
        name, equals, inline = argument[2:].partition("=")
        option, dot, field = name.partition(".")
        name = option.replace("_", "-") + dot + field
        negative = (
            name.startswith("no-")
            and name[3:] in specs
            and specs[name[3:]]["type"] == "boolean"
        )
        if negative:
            name = name[3:]
        root = name.split(".", 1)[0]
        if name in values:
            raise ConfigError(f"Specify --{name} only once")
        if root not in specs:
            if name in {
                "config",
                "kv-transfer-config",
                "kv-offloading-backend",
                "kv-offloading-size",
                "enable-cumem-allocator",
            }:
                raise ConfigError(
                    f"--{name} requires the external-cache/config-file integration; it cannot bypass profile validation"
                )
            passthrough.append(argument)
            while index < len(argv) and not argv[index].startswith("--"):
                passthrough.append(argv[index])
                index += 1
            continue
        if root != name and specs[root]["type"] != "object":
            raise ConfigError(f"--{root} does not accept dotted JSON fields")
        if specs[root]["type"] == "boolean":
            if negative and equals:
                raise ConfigError(f"--no-{name} does not take a value")
            value = not negative if not equals else inline
        elif equals:
            value = inline
        elif specs[root]["type"] == "integers" and root == name:
            value = []
            while index < len(argv) and not argv[index].startswith("--"):
                value.append(argv[index])
                index += 1
        else:
            if index >= len(argv) or argv[index].startswith("--"):
                raise ConfigError(f"--{name} requires a value")
            value = argv[index]
            index += 1
        if root == name:
            values[name] = convert(name, value, specs[name])
        else:
            try:
                values[name] = json.loads(value)
            except (ValueError, TypeError):
                values[name] = value
    return values, passthrough


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if SECRET.search(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1<redacted>@", value)
    return value


@dataclass
class LaunchPlan:
    profile: str
    hardware: str
    values: dict
    environment: dict[str, str]
    origins: dict[str, str]
    environment_origins: dict[str, str]
    argv: list[str]
    passthrough: list[str]
    warnings: list[str]

    def public(self) -> dict:
        # Build public argv from redacted values; never dump the process environment.
        public_argv = list(self.argv)
        redact_next = False
        for index, argument in enumerate(public_argv):
            if argument.startswith("--"):
                redact_next = False
            if redact_next:
                public_argv[index] = "<redacted>"
            elif argument.startswith("--") and SECRET.search(argument.split("=", 1)[0]):
                redact_next = True
                if "=" in argument:
                    public_argv[index] = argument.split("=", 1)[0] + "=<redacted>"
            else:
                try:
                    decoded = json.loads(argument)
                except (ValueError, TypeError):
                    public_argv[index] = _redact(argument)
                else:
                    if isinstance(decoded, (dict, list)):
                        public_argv[index] = json.dumps(
                            _redact(decoded), separators=(",", ":")
                        )
        return {
            "schema_version": 1,
            "status": "implemented",
            "qualification": "CPU configuration tests only; no GPU performance claim",
            "profile": self.profile,
            "hardware": self.hardware,
            "argv": public_argv,
            "settings": {
                key: {"value": _redact(value), "source": self.origins[key]}
                for key, value in self.values.items()
            },
            "environment": {
                key: {
                    "value": "<redacted>" if SECRET.search(key) else _redact(value),
                    "source": self.environment_origins[key],
                }
                for key, value in sorted(self.environment.items())
            },
            "warnings": self.warnings,
        }


def resolve(
    identifier: str,
    hardware: str = "native",
    *,
    env: dict[str, str] | None = None,
    config: dict | None = None,
    argv: list[str] | None = None,
    cli_env: dict[str, str] | None = None,
    runtime_identity: str | None = None,
) -> LaunchPlan:
    incoming = dict(os.environ if env is None else env)
    config = config or {}
    if set(config) - {"options", "environment"}:
        raise ConfigError("Settings file accepts only options and environment mappings")
    for key in ("options", "environment"):
        if not isinstance(config.get(key, {}), dict):
            raise ConfigError(f"Settings {key} must be a mapping")
    for key, value in config.get("environment", {}).items():
        if not NAME.fullmatch(key) or not isinstance(value, str):
            raise ConfigError(
                "Settings environment requires valid names and string values"
            )
    specs = read_yaml(ROOT / "options.yaml")
    common, model, hw = (
        profile("common", "common"),
        profile("model", identifier),
        profile("hardware", hardware),
    )
    cli, passthrough = parse_native(argv or [], specs)
    values, origins = {}, {}
    environment, env_origins = {}, {}

    def set_value(key, value, source):
        if key not in specs:
            raise ConfigError(f"Unknown managed setting: {key}")
        values[key] = convert(key, value, specs[key])
        origins[key] = source

    def set_env(key, value, source):
        if not NAME.fullmatch(key) or not isinstance(value, str) or "\x00" in value:
            raise ConfigError(
                "Environment requires valid names and NUL-free string values"
            )
        environment[key] = value
        env_origins[key] = source

    for layer in (common, model, hw):
        source = f"{layer['kind']}:{layer['id']}"
        for key, value in layer["defaults"].items():
            set_value(key, value, source)
        for key, value in layer["environment"].items():
            set_env(key, value, source)
    for key, value in hw.get("model_environment", {}).get(identifier, {}).items():
        set_env(key, value, f"hardware:{hardware}/{identifier}")

    explicit_env = {**incoming, **config.get("environment", {}), **(cli_env or {})}
    for key, value in config.get("options", {}).items():
        if key not in cli:
            set_value(key, value, "settings:options")
    for key, spec in specs.items():
        if key in cli:
            continue
        candidates = [
            (alias, (cli_env or {})[alias])
            for alias in spec["env"]
            if alias in (cli_env or {})
        ]
        candidate_source = "cli:environment"
        if not candidates:
            if key in config.get("options", {}):
                continue
            candidates = [
                (alias, config.get("environment", {})[alias])
                for alias in spec["env"]
                if alias in config.get("environment", {})
            ]
            candidate_source = "settings:environment"
        if not candidates:
            candidates = [
                (alias, incoming[alias]) for alias in spec["env"] if alias in incoming
            ]
            candidate_source = "environment"
        normalized = []
        for alias, raw in candidates:
            if key == "kv-cache-dtype" and raw == "fp8_ds_mla":
                raw = "fp8"
            if key == "mode" and raw in {"dflash", "none"}:
                raw = {"dflash": "dflash2", "none": "off"}[raw]
            normalized.append((alias, convert(key, raw, spec)))
        if normalized:
            if any(value != normalized[0][1] for _, value in normalized[1:]):
                raise ConfigError(
                    f"Conflicting environment aliases for {key}: {', '.join(alias for alias, _ in normalized)}"
                )
            set_value(
                key,
                normalized[0][1],
                candidate_source + ":" + ",".join(alias for alias, _ in normalized),
            )
    for key, value in cli.items():
        if "." not in key:
            set_value(key, value, "cli")
    if "generation-config" in cli and origins.get(
        "override-generation-config", ""
    ).startswith(("model:", "common:")):
        # Selecting a native generation-config file owns the sampling defaults.
        values.pop("override-generation-config", None)
        origins.pop("override-generation-config", None)

    # Dotted JSON CLI fields refine an explicitly supplied root or its resolved default.
    for key, value in cli.items():
        if "." not in key:
            continue
        root, *parts = key.split(".")
        target = values.setdefault(root, {})
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ConfigError(f"Cannot refine non-object parent in --{key}")
        if not isinstance(target, dict):
            raise ConfigError(f"Cannot refine non-object --{root}")
        target[parts[-1]] = value
        origins[root] = "cli:json-fields"

    # Environment values are explicit only in a model-neutral image. Its build
    # contract is checked before execution; value-equality origin guessing is forbidden.
    consumed = {alias for spec in specs.values() for alias in spec["env"]}
    for key, value in explicit_env.items():
        if key in consumed:
            continue
        if (
            key in environment
            or key.startswith(
                (
                    "VLLM_",
                    "B12X_",
                    "NCCL_",
                    "CUTE_",
                    "TRITON_",
                    "TORCHINDUCTOR_",
                    "CUDA_",
                    "SPARKINFER_",
                    "TILELANG_",
                    "TVM_",
                    "TORCH_EXTENSIONS_",
                    "INSTANTTENSOR_",
                    "SAFETENSORS_",
                )
            )
            or key
            in {
                "OMP_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF",
                "INSTANTTENSOR_BACKEND",
                "XDG_CACHE_HOME",
            }
        ):
            source = (
                "cli:environment"
                if key in (cli_env or {})
                else "settings:environment"
                if key in config.get("environment", {})
                else "environment"
            )
            set_env(key, value, source)
    for key, value in config.get("environment", {}).items():
        if key not in consumed:
            set_env(key, value, "settings:environment")
    for key, value in (cli_env or {}).items():
        if key not in consumed:
            set_env(key, value, "cli:environment")

    def derive(key, value, reason):
        values[key] = value
        origins[key] = f"derived:{reason}"

    if explicit_env.get("EXTRA_VLLM_ARGS"):
        raise ConfigError(
            "EXTRA_VLLM_ARGS is ambiguous shell text; pass native CLI arguments after --"
        )
    for obsolete in (
        "BACKEND",
        "MODE",
        "SPEC_MODE",
        "DS4_OMP_NUM_THREADS",
        "DS4_MAX_CUDAGRAPH_CAPTURE_SIZE",
        "DS4_CUDAGRAPH_CAPTURE_SIZES",
    ):
        if obsolete in explicit_env:
            raise ConfigError(
                f"{obsolete} belongs to a compatibility wrapper; use the documented native option or canonical environment variable"
            )
    fairness = explicit_env.get("FAIRNESS_ENGINE", "compute_share")
    if fairness not in {"none", "compute_share"}:
        raise ConfigError(
            "FAIRNESS_ENGINE must be none or compute_share; micro_slicing is unsupported"
        )
    if fairness == "none" and "prefill-compute-share" not in cli:
        values.pop("prefill-compute-share", None)
        origins.pop("prefill-compute-share", None)

    if (
        "draft-tokens" in values
        and values["draft-tokens"] > 0
        and values["mode"] == "off"
        and origins["mode"].startswith("model:")
    ):
        derive("mode", "mtp", "positive MTP depth")
    mode = values["mode"]
    if mode not in model["modes"]:
        raise ConfigError(f"{identifier} does not define mode {mode}")
    if "speculative-config" not in values:
        tokens = values.get("draft-tokens", model["modes"][mode].get("draft_tokens", 0))
        if tokens < 0:
            raise ConfigError("draft-tokens must be nonnegative")
        if tokens == 0:
            mode = "off"
            derive("mode", "off", "zero draft tokens")
        elif mode == "off":
            raise ConfigError("mode off conflicts with positive draft-tokens")
        if mode != "off":
            spec = copy.deepcopy(model["modes"][mode]["config"])
            spec["num_speculative_tokens"] = tokens
            if identifier == "ds41-flash":
                spec.update(
                    draft_tensor_parallel_size=values["tensor-parallel-size"],
                    enable_adaptive_verification=values["adaptive-verification"],
                    adaptive_verification_cost_scale=values[
                        "adaptive-verification-cost-scale"
                    ],
                )
            elif identifier.startswith("ds4-") and mode == "dspark":
                spec["model"] = values["model"]
            if "draft-model" in values:
                spec["model"] = values["draft-model"]
            if "draft-revision" in values:
                spec["revision"] = values["draft-revision"]
            elif mode == "mtp" and "revision" in values:
                spec["revision"] = values["revision"]
            derive(
                "speculative-config", spec, f"{mode} policy and resolved draft settings"
            )
    else:
        method = values["speculative-config"].get("method")
        mode = "dflash2" if method == "dflash" else method
        if mode not in model["modes"] or mode == "off":
            raise ConfigError(
                "Explicit speculative-config method is not defined by this model profile"
            )
        derive("mode", mode, "explicit speculative-config")
    derive(
        "draft-tokens",
        values.get("speculative-config", {}).get("num_speculative_tokens", 0),
        "effective proposal width",
    )

    if values["cache-mode"] != "vram":
        raise ConfigError(
            f"{identifier}: external cache migration contract '{model['cache']['migration_contract']}' is not executable through this resolver. Use the source-locked cache launcher; no silent GPU-cache fallback is permitted"
        )
    if str(explicit_env.get("LMCACHE_ENABLED", "0")) != "0" or explicit_env.get(
        "LMCACHE_MODE", "off"
    ) not in {"off", "0"}:
        raise ConfigError(
            "External cache requires its lifecycle adapter; LMCACHE settings cannot be ignored"
        )

    if identifier == "glm53-flash":
        for name, key in (
            ("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "target-page-size"),
            ("VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE", "recurrent-page-size"),
        ):
            set_env(name, str(values[key]), origins[key])
        gather = values["dcp-ckv-gather"]
        set_env(
            "VLLM_B12X_MLA_CKV_GATHER",
            str(int(values["decode-context-parallel-size"] > 1))
            if gather == "auto"
            else gather,
            "derived:DCP CKV policy",
        )
        if mode == "off" and not any(
            name in explicit_env
            for name in (
                "VLLM_B12X_DENSE_ACTIVATION_MODE",
                "VLLM_B12X_NVFP4_ACTIVATION_MODE",
            )
        ):
            set_env(
                "VLLM_B12X_NVFP4_ACTIVATION_MODE",
                "quantized",
                "derived:GLM non-speculative activation policy",
            )
        if (
            values["recurrent-checkpoint-policy"] == "aligned"
            and "prefix-cache-retention-interval" not in values
        ):
            derive(
                "prefix-cache-retention-interval", "None", "GPU-local aligned retention"
            )
        if hardware != "native" and "VLLM_PCIE_DMA_MIN_BYTES" not in environment:
            set_env(
                "VLLM_PCIE_DMA_MIN_BYTES",
                "off" if values["decode-context-parallel-size"] > 1 else "6MB",
                "derived:GLM PCIe DCP policy",
            )
    if identifier == "ds41-flash" and "engram-config" not in values:
        derive(
            "engram-config",
            {
                "cpu_offload": False,
                "table_memory": values["engram-table-memory"],
                "disk_resident_scales": values["engram-disk-resident-scales"],
                "projection_tp": values["engram-projection-tp"],
            },
            "Engram table placement, not generic CPU offload",
        )
    if values.get("max-cudagraph-capture-size") == "auto":
        width = (
            values.get("speculative-config", {}).get("num_speculative_tokens", 0) + 1
        )
        if identifier.startswith("ds4-") and mode != "dspark":
            width = 4 if mode == "off" else 8
        derive(
            "max-cudagraph-capture-size",
            max(6, values["max-num-seqs"] * width),
            "bounded verifier rows",
        )
    if "cudagraph-capture-sizes" in values and origins.get(
        "cudagraph-capture-sizes", ""
    ).startswith("model:"):
        cap = values["max-cudagraph-capture-size"]
        if cap != model["defaults"].get("max-cudagraph-capture-size"):
            derive(
                "cudagraph-capture-sizes",
                sorted(
                    {n for n in values["cudagraph-capture-sizes"] if n <= cap} | {cap}
                ),
                "capture-size cap override",
            )

    if values.get("disable-custom-all-reduce"):
        set_env("VLLM_ENABLE_PCIE_ALLREDUCE", "0", "cli:disable-custom-all-reduce")
    if environment.get("NCCL_GRAPH_FILE") == "":
        environment.pop("NCCL_GRAPH_FILE")
        env_origins.pop("NCCL_GRAPH_FILE")
    policy_digest = hashlib.sha256(
        json.dumps([common, model, hw], sort_keys=True).encode()
    ).hexdigest()[:16]
    jit_root = environment.get(
        "XDG_CACHE_HOME",
        f"/cache/jit/{runtime_identity or 'UNBOUND-RUNTIME'}/{identifier}-{policy_digest}",
    )
    cache_paths = {
        name: f"{jit_root}/{suffix}" if suffix else jit_root
        for name, suffix in JIT_PATHS.items()
    }
    for name, path in cache_paths.items():
        if name not in environment:
            set_env(name, path, "derived:runtime-lock and profile identity")
    warnings = [
        "Model parameters preserve recipe intent; changing installed vLLM/B12X requires independent qualification."
    ]
    if identifier.startswith("ds4-") and "linear-backend" not in values:
        warnings.append(
            "DS4 b12x-a8-dglin policy leaves dense selection to native vLLM; confirm DeepGEMM dispatch in serving logs."
        )
    if passthrough:
        warnings.append(
            "Unmanaged native options are forwarded to vLLM; their values are not validated by the profile schema."
        )
    validate(values, environment, identifier)
    command = [
        "/opt/venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        values["model"],
    ]
    for key, value in sorted(values.items()):
        if key == "model" or specs[key].get("control"):
            continue
        if isinstance(value, bool):
            command.append("--" + ("" if value else "no-") + key)
        elif isinstance(value, dict):
            command.extend(
                ["--" + key, json.dumps(value, separators=(",", ":"), allow_nan=False)]
            )
        elif isinstance(value, list):
            command.extend(["--" + key, *map(str, value)])
        else:
            command.extend(["--" + key, str(value)])
    command.extend(passthrough)
    return LaunchPlan(
        identifier,
        hardware,
        values,
        environment,
        origins,
        env_origins,
        command,
        passthrough,
        warnings,
    )


def validate(values: dict, environment: dict, identifier: str) -> None:
    try:
        json.dumps(values, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ConfigError("Configuration must contain finite JSON values") from error
    for key in (
        "tensor-parallel-size",
        "decode-context-parallel-size",
        "pipeline-parallel-size",
        "max-num-seqs",
        "max-num-batched-tokens",
        "block-size",
        "max-cudagraph-capture-size",
    ):
        if not isinstance(values[key], int) or values[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if values["pipeline-parallel-size"] != 1:
        raise ConfigError(
            "These profiles support single-node tensor parallelism, not pipeline parallelism"
        )
    if values["kv-cache-dtype"] not in {"fp8", "fp8_e4m3"}:
        raise ConfigError(
            "The profile migration covers FP8 target KV; other precisions require separate qualification"
        )
    if values["tensor-parallel-size"] % values["decode-context-parallel-size"]:
        raise ConfigError("DCP must divide TP")
    if not 0 < values["gpu-memory-utilization"] <= 1:
        raise ConfigError("gpu-memory-utilization must be in (0, 1]")
    if not 1 <= values["port"] <= 65535:
        raise ConfigError("port must be in [1, 65535]")
    share = values.get("prefill-compute-share")
    if share is not None:
        if share != "auto":
            try:
                numeric_share = float(share)
            except ValueError as error:
                raise ConfigError(
                    "prefill-compute-share must be auto or a value in (0, 1)"
                ) from error
            if not math.isfinite(numeric_share) or not 0 < numeric_share < 1:
                raise ConfigError(
                    "prefill-compute-share must be auto or a value in (0, 1)"
                )
        if values.get("prefill-schedule-interval", 1) != 1:
            raise ConfigError("Compute sharing requires prefill-schedule-interval=1")
    if "prefill-compute-half-life" in values:
        if share != "auto":
            raise ConfigError(
                "prefill-compute-half-life requires automatic compute share"
            )
        half_life = values["prefill-compute-half-life"]
        if half_life not in {"smooth", "responsive"}:
            try:
                seconds = float(half_life)
            except ValueError as error:
                raise ConfigError(
                    "half-life must be smooth, responsive or positive seconds"
                ) from error
            if not math.isfinite(seconds) or seconds <= 0:
                raise ConfigError("half-life must be positive and finite")
    for key in ("max-parallel-prefills", "decode-refill-target"):
        if (
            key in values
            and values[key] != "auto"
            and (not isinstance(values[key], int) or values[key] < 1)
        ):
            raise ConfigError(f"{key} must be auto or a positive integer")
    if (
        values.get("max-parallel-prefills", 1) != 1
        and not values["enable-chunked-prefill"]
    ):
        raise ConfigError("Parallel prefills require chunked prefill")
    captures = values.get("cudagraph-capture-sizes", [])
    if captures and (
        captures != sorted(set(captures))
        or min(captures) < 1
        or max(captures) > values["max-cudagraph-capture-size"]
    ):
        raise ConfigError(
            "Capture sizes must be positive, strictly increasing and within the capture cap"
        )
    if identifier == "glm53-flash":
        for key in ("target-page-size", "recurrent-page-size"):
            page = values[key]
            if page != "auto" and (
                not page.isdigit() or int(page) <= 0 or int(page) % 64
            ):
                raise ConfigError(f"{key} must be auto or a positive multiple of 64")
        target, recurrent = values["target-page-size"], values["recurrent-page-size"]
        if (
            target != "auto"
            and recurrent != "auto"
            and int(target) % int(recurrent)
            and int(recurrent) % int(target)
        ):
            raise ConfigError(
                "Recurrent page spacing must divide, or be a multiple of, the target page"
            )
        if values.get("cp-kv-cache-interleave-size") != values.get(
            "dcp-kv-cache-interleave-size"
        ):
            raise ConfigError(
                "GLM target and draft require matching CP and DCP interleave sizes"
            )
    if "speculative-config" in values:
        depth = values["speculative-config"].get("num_speculative_tokens")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ConfigError("Speculative token count must be positive")
    if identifier == "ds41-flash":
        if "engram-config" in values:
            engram = values["engram-config"]
            if (
                engram.get("table_memory") not in {"ram", "disk"}
                or engram.get("cpu_offload") is not False
            ):
                raise ConfigError(
                    "The DS4.1 profile covers RAM/disk Engram with cpu_offload=false"
                )
        if values["adaptive-verification-cost-scale"] <= 0:
            raise ConfigError("adaptive-verification-cost-scale must be positive")
    if "OMP_NUM_THREADS" in environment:
        omp = environment["OMP_NUM_THREADS"]
        if not omp.isdigit() or int(omp) <= 0:
            raise ConfigError("OMP_NUM_THREADS must be positive")


def execute(plan: LaunchPlan, contract_path: Path) -> None:
    from runtime.packaging import verify_contract

    contract = verify_contract(contract_path)
    if any("UNBOUND-RUNTIME" in value for value in plan.environment.values()):
        raise ConfigError("Execution requires a runtime-bound JIT cache namespace")
    environment = dict(os.environ)
    # Managed aliases must not reach a second resolver. Native runtime variables
    # remain intact, including logging, tracing, credentials and explicit paths.
    for spec in read_yaml(ROOT / "options.yaml").values():
        for name in spec["env"]:
            environment.pop(name, None)
    if environment.get("NCCL_GRAPH_FILE") == "":
        environment.pop("NCCL_GRAPH_FILE")
    environment.update(plan.environment)
    command = [*contract.get("bootstrap", []), *plan.argv]
    os.execvpe(command[0], command, environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--hardware", default="native")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--image-contract", type=Path, default=ROOT / "image-contract.json"
    )
    args, native = parser.parse_known_args()
    try:
        explicit_env = {}
        for item in args.env:
            name, separator, value = item.partition("=")
            if not separator or not NAME.fullmatch(name) or name in explicit_env:
                raise ConfigError("--env requires a unique valid NAME=VALUE")
            explicit_env[name] = value
        runtime_identity = None
        contract = {}
        if not args.print_config or args.image_contract.exists():
            from runtime.packaging import verify_contract

            contract = verify_contract(args.image_contract)
            runtime_identity = contract["runtime_lock_sha256"]
        plan = resolve(
            args.profile,
            args.hardware,
            config=read_yaml(args.settings) if args.settings else None,
            argv=native,
            cli_env=explicit_env,
            runtime_identity=runtime_identity,
        )
        if args.print_config:
            public = plan.public()
            public["bootstrap"] = contract.get("bootstrap", [])
            public["runtime_lock_sha256"] = runtime_identity
            print(json.dumps(public, indent=2, allow_nan=False))
        else:
            execute(plan, args.image_contract)
    except (ConfigError, OSError) as error:
        print(f"Launch configuration error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
