"""Resolve model policy without importing vLLM, CUDA, or model checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform as host_platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
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
    # These default to the home directory, which a recreated container loses.
    "TILELANG_CACHE_DIR": "tilelang",
    "TVM_FFI_CACHE_DIR": "tvm-ffi",
    "TVM_CACHE_DIR": "tvm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "flash-attention-cute-dsl",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "NUMBA_CACHE_DIR": "numba",
    "CUPY_CACHE_DIR": "cupy",
}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Chat-template settings with this prefix name a file in ROOT / "templates".
RUNTIME_TEMPLATE_PREFIX = "runtime:"
SECRET = re.compile(
    r"api[-_]?key|password|secret|authorization|access[-_]?token|hf_token",
    re.IGNORECASE,
)
# Precision of GLM-5.3-Flash's routed FP4 experts on B12X.
GLM_PRECISION_OPTIONS = ("expert-activations", "router-weights", "prefill-activations")
# On images whose vLLM reads B12X_W4A16_A4_PREFILL_MIN_TOKENS, prefill-activations
# a4 sets it to this value. vLLM then runs the prefill rows of every step with
# NVFP4 activations and the decode rows with W4A16, whatever the step size
# (images before vllm #977 used the value as a call-size threshold instead).
# Newer images take B12X_W4A16_A4_PREFILL=1 instead.
GLM_A4_PREFILL_MIN_TOKENS = 1536
# Checkpoints whose routed experts lose accuracy with a4 prefill: the Spark
# checkpoint's pre-QAD experts answered 456 of 500 needle-checksum requests
# with a4 and 488 with a16; the QAD checkpoints answered 477-494 with a4.
GLM_A4_PREFILL_UNQUALIFIED = ("GLM-5.3-Flash-NVFP4-Spark",)


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently selecting a value."""


# YAML 1.2 booleans: mode names such as "off" are strings, not boolean keys.
UniqueLoader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in rules if tag != "tag:yaml.org,2002:bool"]
    for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
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


def platform_environment_path() -> Path:
    """The foundation policy of the machine's image: linux/arm64 images carry
    platform-environment.linux-arm64.json beside the linux/amd64 default."""
    if host_platform.machine() == "aarch64":
        candidate = ROOT / "platform-environment.linux-arm64.json"
        if candidate.is_file():
            return candidate
    return ROOT / "platform-environment.json"


def platform_environment() -> dict[str, str]:
    """Read foundation defaults below model policy and explicit user settings."""
    path = platform_environment_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f"Cannot read {path.name}") from error
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "source_image", "environment"}
        or data["schema_version"] != 1
        or not isinstance(data["source_image"], str)
        or not isinstance(data["environment"], dict)
        or any(
            not NAME.fullmatch(key) or not isinstance(value, str) or "\x00" in value
            for key, value in data["environment"].items()
        )
    ):
        raise ConfigError("Invalid platform environment policy")
    return data["environment"]


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


def deployment_presets() -> dict:
    """Read data-only deployment overlays owned by the container recipe."""
    data = read_yaml(ROOT / "presets.yaml")
    if set(data) != {"schema_version", "presets"} or data["schema_version"] != 1:
        raise ConfigError("Unsupported deployment preset schema")
    presets = data["presets"]
    if not isinstance(presets, dict):
        raise ConfigError("Deployment presets must be a mapping")
    for name, item in presets.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name) or not isinstance(item, dict):
            raise ConfigError("Invalid deployment preset identity")
        required = {
            "profile",
            "hardware",
            "description",
            "options",
            "environment",
            "modes",
            "linked_options",
        }
        kv_fields = {"kv_bytes_per_extra_slot", "kv_bytes_for_external_cache"}
        optional = kv_fields | {"vllm_fallback", "kv_bytes_by_dcp"}
        if not required <= set(item) <= required | optional:
            raise ConfigError(f"Invalid deployment preset fields: {name}")
        fallback = item.get("vllm_fallback")
        if fallback is not None and (
            not isinstance(fallback, dict)
            or set(fallback) != {"requires_environment", "options"}
            or not isinstance(fallback["options"], dict)
            or not isinstance(fallback["requires_environment"], list)
            or not all(
                isinstance(key, str) and key in item["environment"]
                for key in fallback["requires_environment"]
            )
        ):
            raise ConfigError(f"Invalid preset vllm_fallback: {name}")
        for field in kv_fields:
            value = item.get(field, 0)
            if type(value) is not int or value < 0:
                raise ConfigError(f"Invalid preset {field}: {name}")
        by_dcp = item.get("kv_bytes_by_dcp", {})
        if not isinstance(by_dcp, dict) or not all(
            isinstance(size, str)
            and re.fullmatch(r"[1-9][0-9]*", size)
            and int(size) > 1
            and type(amount) is int
            and amount > 0
            for size, amount in by_dcp.items()
        ):
            raise ConfigError(f"Invalid preset kv_bytes_by_dcp: {name}")
        for key in ("profile", "hardware"):
            if not isinstance(item[key], str) or not re.fullmatch(
                r"[a-z][a-z0-9-]*", item[key]
            ):
                raise ConfigError(f"Invalid preset {key}: {name}")
        if not isinstance(item["description"], str) or not all(
            isinstance(item[key], dict)
            for key in ("options", "environment", "modes", "linked_options")
        ):
            raise ConfigError(f"Invalid deployment preset values: {name}")
    return presets


def installed_source(package: str, relative: str) -> str | None:
    """Text of a file in an installed package, or None when it is missing.

    The launcher must not import vLLM, B12X or CUDA, so it reads source text.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        path = Path(location) / relative
        if path.is_file():
            return path.read_text()
    return None


def installed_semantic_a4_prefill() -> bool:
    """Whether the installed vLLM turns hybrid A4 prefill on with
    B12X_W4A16_A4_PREFILL and picks the prefill rows itself. Images before it
    took a token threshold, B12X_W4A16_A4_PREFILL_MIN_TOKENS."""
    source = installed_source("vllm", "utils/b12x.py")
    return source is not None and '"B12X_W4A16_A4_PREFILL"' in source


def installed_vllm_environment() -> frozenset[str] | None:
    """Environment names the installed vLLM declares, or None without vLLM."""
    source = installed_source("vllm", "envs.py")
    if source is None:
        return None
    return frozenset(re.findall(r'^    "(VLLM_[A-Z0-9_]+)"', source, re.MULTILINE))


def installed_csf_formats() -> frozenset[str] | None:
    """FP4-CSF load formats the installed vLLM reads, or None without vLLM.

    A vLLM that reads NVFP4-CSF scales from ModelOpt recipes has no dedicated
    load format but serves Hugging Face-layout NVFP4-CSF checkpoints.
    """
    source = installed_source("vllm", "model_executor/model_loader/__init__.py")
    if source is None:
        return None
    formats = {fmt for fmt in CSF_FORMATS if f'"{fmt}"' in source}
    if installed_nvfp4_csf_recipes():
        formats.add("nvfp4_csf")
    return frozenset(formats)


def installed_nvfp4_csf_recipes() -> bool:
    """Whether the installed vLLM reads NVFP4-CSF expert scales declared by
    ModelOpt recipes (weight_scale_encoding) instead of a dedicated format."""
    loaders = installed_source("vllm", "model_executor/model_loader/__init__.py")
    modelopt = installed_source(
        "vllm", "model_executor/layers/quantization/modelopt.py"
    )
    return (
        loaders is not None
        and '"nvfp4_csf"' not in loaders
        and modelopt is not None
        and "weight_scale_encoding" in modelopt
    )


def installed_csf_families() -> frozenset[str] | None:
    """Checkpoint families the installed vLLM's FP4-CSF loaders accept.

    Each loader lists them in its FAMILIES table, for example deepseek_v41 and
    deepseek_v4_flash in mxfp4_csf_loader.py; a reader of a format can predate
    a family. None without vLLM.
    """
    if installed_source("vllm", "model_executor/model_loader/__init__.py") is None:
        return None
    families: set[str] = set()
    for fmt in CSF_FORMATS:
        source = installed_source(
            "vllm", f"model_executor/model_loader/{fmt}_loader.py"
        )
        table = re.search(
            r"^FAMILIES\b[^=\n]*=\s*\{(.*?)^\}", source or "", re.MULTILINE | re.DOTALL
        )
        if table:
            families.update(
                re.findall(r'^\s+"([a-z][a-z0-9_]*)"\s*:', table.group(1), re.MULTILINE)
            )
    return frozenset(families)


def installed_b12x_mxfp8_moe() -> bool:
    """Whether the installed vLLM and B12X run MXFP8 MoE experts on B12X.

    vLLM maps moe_backend b12x to its B12X_MXFP8 backend (vllm #958) and B12X
    prepares the mxfp8_e8m0_k32 source format (b12x #453).
    """
    oracle = installed_source("vllm", "model_executor/layers/fused_moe/oracle/mxfp8.py")
    formats = installed_source("b12x", "moe/fused_moe/source.py")
    return (
        oracle is not None
        and "Fp8MoeBackend.B12X_MXFP8" in oracle
        and formats is not None
        and '"mxfp8_e8m0_k32"' in formats
    )


# Removed presets name the preset that replaces them, so an old Compose file
# fails with the way forward instead of an unknown name.
RETIRED_PRESETS = {
    "glm53-spark-tp2": (
        "glm53-tp2",
        (
            "it serves the QAD weights of GLM-5.3-Flash from the FP4-CSF checkpoint "
            "GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD (a new download)"
        ),
    ),
}


def deployment_preset(identifier: str) -> dict:
    presets = deployment_presets()
    if identifier not in presets and identifier in RETIRED_PRESETS:
        replacement, reason = RETIRED_PRESETS[identifier]
        raise ConfigError(
            f"PRESET={identifier} was removed; use PRESET={replacement}: {reason}"
        )
    if identifier not in presets:
        raise ConfigError(f"Unknown deployment preset: {identifier}")
    return copy.deepcopy(presets[identifier])


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
            if root in {
                "config",
                "kv-transfer-config",
                "kv-offloading-backend",
                "kv-offloading-size",
                "enable-cumem-allocator",
                "max-num-scheduled-tokens",
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
    cache_service: Any = None
    # Drafter shipped inside the target checkpoint; resolved to a local path
    # at launch, so printing a configuration never downloads anything.
    draft_subfolder: str | None = None
    # Content identity of a target served from generated files (FP4-CSF),
    # which the checkpoint identity helper cannot hash itself.
    target_identity: dict | None = None

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
            "cache_service": _redact(asdict(self.cache_service))
            if self.cache_service
            else None,
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
    preset: str | None = None,
    vllm_environment: frozenset[str] | None = None,
    csf_formats: frozenset[str] | None = None,
    csf_families: frozenset[str] | None = None,
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
    deployment = deployment_preset(preset) if preset else None
    if deployment and deployment["profile"] != identifier:
        raise ConfigError("The deployment preset belongs to a different model profile")
    fallback_reason = None
    if deployment and deployment.get("vllm_fallback") and vllm_environment is not None:
        # A preset that relies on vLLM features newer than the installed vLLM
        # (for example the same recipe in a channel with an older vLLM) keeps
        # its previously qualified values instead.
        fallback = deployment["vllm_fallback"]
        missing = sorted(set(fallback["requires_environment"]) - vllm_environment)
        if missing:
            deployment["options"].update(fallback["options"])
            for name in fallback["requires_environment"]:
                deployment["environment"].pop(name, None)
            fallback_reason = "installed vLLM lacks " + ", ".join(missing)
    if deployment:
        model = copy.deepcopy(model)
        for mode_name, overrides in deployment["modes"].items():
            if (
                mode_name not in model["modes"]
                or not isinstance(overrides, dict)
                or set(overrides) - {"draft_tokens", "config"}
            ):
                raise ConfigError(f"Invalid preset mode: {mode_name}")
            if "draft_tokens" in overrides:
                model["modes"][mode_name]["draft_tokens"] = overrides["draft_tokens"]
            if "config" in overrides:
                if not isinstance(overrides["config"], dict):
                    raise ConfigError("Preset mode config must be a mapping")
                model["modes"][mode_name].setdefault("config", {}).update(
                    overrides["config"]
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

    platform = platform_environment()
    for key, value in platform.items():
        set_env(key, value, "platform:foundation")
    for layer in (common, model, hw):
        source = f"{layer['kind']}:{layer['id']}"
        for key, value in layer["defaults"].items():
            set_value(key, value, source)
        for key, value in layer["environment"].items():
            set_env(key, value, source)
    for key, value in hw.get("model_environment", {}).get(identifier, {}).items():
        set_env(key, value, f"hardware:{hardware}/{identifier}")
    if deployment:
        source = f"preset:{preset}"
        if fallback_reason:
            source += f" (fallback: {fallback_reason})"
        for key, value in deployment["options"].items():
            set_value(key, value, source)
        for key, value in deployment["environment"].items():
            set_env(key, value, f"preset:{preset}")

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
            if alias == "LMCACHE_MODE":
                cache_modes = {
                    "off": ("vram", False),
                    "0": ("vram", False),
                    "ram": ("lmcache", False),
                    "memory": ("lmcache", False),
                    "1": ("lmcache", False),
                    "disk": ("lmcache", True),
                    "ram-disk": ("lmcache", True),
                    "memory-disk": ("lmcache", True),
                }
                if str(raw).lower() not in cache_modes:
                    raise ConfigError("LMCACHE_MODE must select off, ram or disk")
                mode, l2 = cache_modes[str(raw).lower()]
                raw = mode if key == "cache-mode" else l2
            if alias == "LMCACHE_ENABLED":
                if raw not in {"0", "1"}:
                    raise ConfigError("LMCACHE_ENABLED must be 0 or 1")
                raw = "lmcache" if raw == "1" else "vram"
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

    # The checkpoint variant (CHECKPOINT) chooses the checkpoint and the settings
    # that read it, replacing profile defaults only. A model naming a variant's
    # checkpoint selects that variant. Any other model or revision chosen by a
    # preset, settings or the operator keeps the profile settings unless
    # CHECKPOINT is chosen there too; then the variant fills in everything else.
    # A preset that names a checkpoint of its own serves it as configured, also
    # when a variant names the same checkpoint.
    checkpoint_warnings: list[str] = []
    if "checkpoint" in values:
        variants = model.get("checkpoints", {})
        profile_level = ("common:", "model:", "hardware:")

        def chosen(key: str) -> bool:
            return key in values and not origins[key].startswith(profile_level)

        def is_csf(options: dict) -> bool:
            return options.get("load-format") in CSF_FORMATS

        def unreadable(item: dict) -> str | None:
            """The checkpoints the installed vLLM cannot read, if a variant is one."""
            if not is_csf(item["options"]):
                return None
            if (
                csf_formats is not None
                and item["options"]["load-format"] not in csf_formats
            ):
                return "FP4-CSF checkpoints"
            family = item.get("csf_family")
            if family and csf_families is not None and family not in csf_families:
                return f"FP4-CSF checkpoints of the {family} family"
            return None

        name, reason = values["checkpoint"], "derived:model"
        # A local FP4-CSF copy read with a non-CSF variant's settings loads
        # without an error and serves wrong output, so its files choose.
        local_csf = local_csf_format(values["model"]) if chosen("model") else None
        if variants and origins["model"].startswith("preset:"):
            # CHECKPOINT cannot swap a preset's own checkpoint for one of the
            # other kind; the preset's settings read only its own.
            if (
                chosen("checkpoint")
                and name in variants
                and is_csf(variants[name]["options"]) != is_csf(values)
            ):
                kind = "an FP4-CSF" if is_csf(values) else "a non-CSF"
                raise ConfigError(
                    f"PRESET={preset} serves {kind} checkpoint of its own, "
                    f"{values['model']}; CHECKPOINT={name} does not apply to it. "
                    "Choose the checkpoint with the model profile, without this preset"
                )
            name = None
        elif not chosen("checkpoint") and (chosen("model") or chosen("revision")):
            named = [
                variant
                for variant, item in variants.items()
                if item["options"]["model"] == values["model"]
            ]
            if chosen("model") and not named and local_csf:
                named = [
                    variant
                    for variant, item in variants.items()
                    if item["options"].get("load-format") == local_csf
                ]
                if not named:
                    raise ConfigError(
                        f"MODEL={values['model']} is a {local_csf} checkpoint; "
                        f"{identifier} has no checkpoint variant that reads it"
                    )
                reason = f"derived:{local_csf} files at MODEL"
            name = named[0] if chosen("model") and named else None
        if local_csf and name in variants and not is_csf(variants[name]["options"]):
            raise ConfigError(
                f"MODEL={values['model']} is a {local_csf} checkpoint, but "
                f"CHECKPOINT={name} ({origins['checkpoint']}) reads its compressed "
                "scales as plain ones and serves wrong output"
            )
        missing = {
            variant: text
            for variant, item in variants.items()
            if (text := unreadable(item)) is not None
        }
        if name in missing:
            if local_csf:
                raise ConfigError(
                    f"This image's vLLM cannot read {missing[name]}, which "
                    f"MODEL={values['model']} is"
                )
            if chosen("checkpoint") or chosen("model"):
                raise ConfigError(
                    f"This image's vLLM cannot read {missing[name]}; "
                    "use CHECKPOINT=original"
                )
            # An image whose vLLM predates the FP4-CSF reader of this checkpoint
            # serves the checkpoint it can read instead of failing at startup.
            lacking = missing[name]
            name = next(v for v in variants if v not in missing)
            reason = f"derived:installed vLLM cannot read {lacking}"
            checkpoint_warnings.append(
                f"This image's vLLM cannot read {lacking}; serving the {name} checkpoint."
            )
        if name is None:
            values.pop("checkpoint")
            origins.pop("checkpoint")
        elif name not in variants:
            raise ConfigError(f"{identifier} has no {name} checkpoint variant")
        else:
            if name != values["checkpoint"]:
                set_value("checkpoint", name, reason)
            options = variants[name]["options"]
            # A pinned revision belongs to the variant's own checkpoint.
            other = chosen("model") and values["model"] != options["model"]
            for key, value in options.items():
                if not chosen(key) and not (key == "revision" and other):
                    set_value(key, value, f"checkpoint:{name}")
    # An FP4-CSF checkpoint named by a preset or the operator rather than a
    # checkpoint variant has no other checkpoint to fall back to.
    if (
        values.get("load-format") in CSF_FORMATS
        and csf_formats is not None
        and values["load-format"] not in csf_formats
    ):
        raise ConfigError(
            f"This image's vLLM cannot read {values['load-format']} checkpoints"
            + (
                f", which PRESET={preset} serves"
                if origins["load-format"].startswith("preset:")
                else ""
            )
        )

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
                    "FLASHINFER_",
                    "FLASH_ATTENTION_",
                    "NUMBA_",
                    "CUPY_",
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
                "LD_PRELOAD",
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

    configure_expert_precision(
        identifier,
        values,
        origins,
        environment_origins=env_origins,
        explicit_env=explicit_env,
        vllm_environment=vllm_environment,
        set_env=set_env,
        derive=derive,
    )

    if deployment:
        for target, source in deployment["linked_options"].items():
            if target not in specs or source not in values:
                raise ConfigError("Preset option dependency is not declared")
            if origins.get(target, "").startswith(("preset:", "model:", "common:")):
                set_value(target, values[source], f"derived:preset link to {source}")
        # A preset's fixed KV size is qualified at its own request-slot count.
        # Each additional slot needs working memory (CUDA graphs up to the
        # larger verifier-row count, sampler and state buffers), so the KV
        # allocation shrinks unless the operator set it explicitly.
        # The external cache connector keeps its own GPU buffers.
        preset_kv = origins.get("kv-cache-memory-bytes", "").startswith("preset:")
        per_slot = deployment.get("kv_bytes_per_extra_slot", 0)
        base_slots = deployment["options"].get("max-num-seqs")
        reductions = []
        if per_slot and base_slots and values.get("max-num-seqs", 0) > base_slots:
            extra = values["max-num-seqs"] - base_slots
            reductions.append(
                (extra * per_slot, f"{extra} request slots above the preset")
            )
        external = deployment.get("kv_bytes_for_external_cache", 0)
        if external and values.get("cache-mode", "vram") != "vram":
            reductions.append((external, "external cache buffers"))
        if preset_kv and reductions:
            derive(
                "kv-cache-memory-bytes",
                values["kv-cache-memory-bytes"] - sum(size for size, _ in reductions),
                "; ".join(reason for _, reason in reductions),
            )
        # With DCP > 1, context-sized indexer and DCP exchange buffers grow
        # outside vLLM's memory budget, so a preset can carry a KV size per
        # DCP group, qualified at the longest prompt plus a full batch of
        # concurrent prefills. An unlisted group size takes the next larger
        # group's size, which leaves more room.
        by_dcp = {
            int(size): amount
            for size, amount in deployment.get("kv_bytes_by_dcp", {}).items()
        }
        dcp = values.get("decode-context-parallel-size", 1)
        qualified = [size for size in sorted(by_dcp) if size >= dcp]
        if (
            dcp > 1
            and qualified
            and origins.get("kv-cache-memory-bytes", "preset:").startswith(
                ("preset:", "model:", "common:")
            )
        ):
            derive(
                "kv-cache-memory-bytes",
                by_dcp[qualified[0]],
                f"KV size qualified at DCP={qualified[0]}",
            )

    # A repository-specific code revision must not leak to an operator's model.
    if (
        "revision" not in values
        and values["model"] == model["defaults"]["model"]
        and model.get("checkpoint_revision")
    ):
        derive(
            "revision", model["checkpoint_revision"], "profile checkpoint/code revision"
        )
    # vLLM opens an FP4-CSF checkpoint from generated local serving files that
    # carry its Hugging Face files, so no Hub code revision applies to it.
    if (
        values.get("trust-remote-code")
        and "revision" in values
        and "code-revision" not in values
        and values.get("load-format") not in CSF_FORMATS
    ):
        derive(
            "code-revision",
            values["revision"],
            "remote code follows selected checkpoint",
        )

    if identifier == "mimo26-flash" and "max-num-scheduled-tokens" not in values:
        # Qualified MiMo split: half of each step's rows for target tokens,
        # the rest for DFlash verification rows (vllm #881: 4096 / 2048).
        derive(
            "max-num-scheduled-tokens",
            values["max-num-batched-tokens"] // 2,
            "MiMo target share of the step",
        )

    if identifier == "glm53":
        # DCP prefill gathers the full compressed KV cache and the indexer
        # keys on every rank instead of exchanging partial attention: about
        # 1.4x (DCP2) to 2.6x (DCP6) the prefill speed at the same decode.
        # Mixed batches gather for their prefill rows too, so a prompt that
        # arrives while others decode prefills 1.2x (DCP2) to 17x (DCP8)
        # faster.
        gather = values["dcp-ckv-gather"]
        enabled = (
            str(int(values["decode-context-parallel-size"] > 1))
            if gather == "auto"
            else gather
        )
        derived = {
            "VLLM_B12X_MLA_CKV_GATHER": enabled,
            "VLLM_DCP_INDEXER_KEY_GATHER": enabled,
            "VLLM_B12X_MLA_CKV_GATHER_MIXED": enabled,
        }
        if enabled == "1" and values["decode-context-parallel-size"] == 6:
            # DCP6 holds the full 1M context; past the default 524,288-token
            # gather cap its all-gather/reduce-scatter fallback prefills at
            # about 2,000 tokens/s (a 1M prompt: 360 s, 329 s with this cap,
            # for 0.9% of the KV cache). At DCP2/3 the cap gains nothing.
            derived["VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS"] = "1048576"
        for name, value in derived.items():
            if name in explicit_env or (
                vllm_environment is not None and name not in vllm_environment
            ):
                continue
            set_env(name, value, "derived:DCP CKV policy")

    if (
        identifier == "glm53"
        and values["decode-context-parallel-size"] == 6
        and "dcp-comm-backend" not in values
    ):
        # The B12X PCIe DCP all-to-all covers 2, 4 and 8 ranks. At DCP6 the
        # generic all-to-all prefills about seven times slower than
        # all-gather plus reduce-scatter; at DCP3 it still decodes faster.
        derive("dcp-comm-backend", "ag_rs", "GLM DCP6 exchange")

    if identifier == "qwen38-flash-next":
        # B12X QSA shards compressed KV across DCP ranks in groups of four
        # tokens and refuses vLLM's default interleave of one, so every
        # DCP > 1 deployment needs this unless the operator chose a value.
        if values["decode-context-parallel-size"] > 1:
            for key in ("cp-kv-cache-interleave-size", "dcp-kv-cache-interleave-size"):
                if key not in values or origins[key].startswith(
                    ("common:", "model:", "hardware:")
                ):
                    derive(key, 4, "QSA DCP interleave")
        context_length = values["max-model-len"]
        base_context_length = 262144
        if context_length > base_context_length and "hf-overrides" not in values:
            if context_length > 1048576:
                raise ConfigError(
                    "Qwen3.8 Flash Next contexts above 1048576 tokens require "
                    "an explicit HF_OVERRIDES configuration"
                )
            # The checkpoint advertises 262144 positions. Extend its text
            # config before vLLM constructs both target and MTP draft models.
            factor = 2 if context_length <= 524288 else 4
            derive(
                "hf-overrides",
                {
                    "text_config": {
                        "max_position_embeddings": context_length,
                        "rope_parameters": {
                            "mrope_interleaved": True,
                            "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000,
                            "rope_type": "yarn",
                            "factor": factor,
                            "original_max_position_embeddings": base_context_length,
                        },
                    }
                },
                f"Qwen3.8 YaRN for {context_length} tokens",
            )

    replicas = values["replicas"]
    ple_copy_per_replica = False
    if replicas < 1:
        raise ConfigError("replicas must be at least 1")
    configure_nodes(values, replicas, derive)
    if replicas > 1:
        for key in (
            "tensor-parallel-size",
            "pipeline-parallel-size",
            "decode-context-parallel-size",
            "data-parallel-size",
        ):
            if values.get(key, 1) != 1:
                raise ConfigError(
                    f"replicas runs independent single-GPU servers; {key} must be 1, "
                    f"got {values[key]}"
                )
        # One host-RAM PLE table serves every replica instead of one each; a
        # vLLM without shared tables keeps a copy per replica.
        if (
            identifier == "qwen38-flash-next"
            and environment.get("VLLM_PLE_CPU_OFFLOAD") == "1"
            and "VLLM_PLE_TABLE_MEMORY" not in explicit_env
        ):
            ple_copy_per_replica = (
                vllm_environment is not None
                and "VLLM_PLE_SHARED_TABLE_DIR" not in vllm_environment
            )
            if not ple_copy_per_replica:
                from runtime.replicas import PLE_SHARED_DIRECTORY

                set_env("VLLM_PLE_TABLE_MEMORY", "shared", "derived:replicas")
                if "VLLM_PLE_SHARED_TABLE_DIR" not in explicit_env:
                    set_env(
                        "VLLM_PLE_SHARED_TABLE_DIR",
                        PLE_SHARED_DIRECTORY,
                        "derived:replicas",
                    )

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
    draft_subfolder = None
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
            elif identifier == "mimo26-flash":
                spec["draft_tensor_parallel_size"] = values["tensor-parallel-size"]
            elif identifier.startswith("ds4-") and mode == "dspark":
                spec["model"] = values["model"]
            if "draft-model" in values:
                spec["model"] = values["draft-model"]
            elif model["modes"][mode].get("draft_subfolder"):
                draft_subfolder = model["modes"][mode]["draft_subfolder"]
                spec["model"] = f"{values['model']}/{draft_subfolder}"
            if "draft-revision" in values:
                spec["revision"] = values["draft-revision"]
            elif mode in {"mtp", "dspark"} and "revision" in values:
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
    width = values.get("speculative-config", {}).get("num_speculative_tokens", 0) + 1
    if identifier.startswith("ds4-") and mode != "dspark":
        width = 4 if mode == "off" else 8
    if values.get("max-cudagraph-capture-size") == "auto":
        derive(
            "max-cudagraph-capture-size",
            max(6, values["max-num-seqs"] * width),
            "bounded verifier rows",
        )
    if "cudagraph-capture-sizes" in values and origins.get(
        "cudagraph-capture-sizes", ""
    ).startswith(("model:", "preset:")):
        cap = values["max-cudagraph-capture-size"]
        if cap != model["defaults"].get("max-cudagraph-capture-size") or deployment:
            sizes = {n for n in values["cudagraph-capture-sizes"] if n <= cap}
            # A raised cap (for example more request slots) continues the
            # listed sizes in steps of one request's verifier rows, so every
            # running-request count up to the cap still replays a graph.
            step = width if width > 1 else 8
            top = max(sizes, default=0)
            sizes |= {n for n in range(step, cap, step) if n > top}
            derive(
                "cudagraph-capture-sizes",
                sorted(sizes | {cap}),
                "capture-size cap override",
            )

    if values.get("disable-custom-all-reduce"):
        set_env("VLLM_ENABLE_PCIE_ALLREDUCE", "0", "cli:disable-custom-all-reduce")
    if environment.get("NCCL_GRAPH_FILE") == "":
        environment.pop("NCCL_GRAPH_FILE")
        env_origins.pop("NCCL_GRAPH_FILE")
    policy_digest = hashlib.sha256(
        json.dumps(
            [platform, common, model, hw, *([deployment] if deployment else [])],
            sort_keys=True,
        ).encode()
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
        "Model parameters preserve recipe intent; changing installed vLLM/B12X requires independent qualification.",
        *checkpoint_warnings,
    ]
    if identifier.startswith("ds4-") and "linear-backend" not in values:
        warnings.append(
            "DS4 b12x-a8-dglin policy leaves dense selection to native vLLM; confirm DeepGEMM dispatch in serving logs."
        )
    if passthrough:
        warnings.append(
            "Unmanaged native options are forwarded to vLLM; their values are not validated by the profile schema."
        )
    if replicas > 1 and identifier == "qwen38-flash-next" and ple_copy_per_replica:
        warnings.append(
            "The installed vLLM cannot share the PLE table, so every replica keeps "
            "its own host-RAM copy (about 27 GiB each)."
        )
    from runtime.cache import configure as configure_cache

    if (
        values["cache-mode"] != "vram"
        and model.get("cache", {}).get("external") != "implemented"
    ):
        raise ConfigError(
            f"{identifier}: external cache is unsupported by this image's profile"
        )
    cache_service = configure_cache(
        values, origins, environment, env_origins, identifier, runtime_identity
    )
    if values.get("kv-transfer-config", {}).get(
        "kv_connector"
    ) == "LMCacheRecurrentCheckpointConnector" and not values.get(
        "language-model-only", False
    ):
        warnings.append(
            "External recurrent checkpoints support text requests only. "
            "Image/video requests can use native GPU prefix caching but do not "
            "restore their recurrent state from CPU or disk."
        )
    validate(values, environment, identifier)
    command = make_argv(values, passthrough)
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
        cache_service,
        draft_subfolder,
    )


def configure_expert_precision(
    identifier: str,
    values: dict,
    origins: dict,
    *,
    environment_origins: dict,
    explicit_env: dict,
    vllm_environment: frozenset[str] | None,
    set_env,
    derive,
) -> None:
    """Map GLM-5.3-Flash's routed-expert precision options to B12X variables.

    expert-activations bf16 runs the FP4 experts as W4A16 (FP4 weights, BF16
    activations) and fp4 as W4A4. router-weights fp32 combines W4A16 expert
    outputs with FP32 router weights. prefill-activations a4 runs the prefill
    rows of W4A16 calls with NVFP4 activations: B12X_W4A16_A4_PREFILL=1 where
    vLLM picks the rows itself, else B12X_W4A16_A4_PREFILL_MIN_TOKENS (calls of
    at least GLM_A4_PREFILL_MIN_TOKENS tokens). A variable the operator sets
    itself is kept and decides the option it belongs to.
    """
    present = [option for option in GLM_PRECISION_OPTIONS if option in values]
    if not present:
        return
    profile_level = ("common:", "model:", "hardware:")
    if identifier != "glm53-flash":
        raise ConfigError(f"{present[0]} applies to GLM-5.3-Flash only")
    if values.get("moe-backend") != "b12x":
        # Another MoE backend runs the experts in its own precision.
        named = [
            item for item in present if not origins[item].startswith(profile_level)
        ]
        if named:
            raise ConfigError(
                f"{named[0]} applies to B12X routed experts (moe-backend b12x)"
            )
        for option in present:
            values.pop(option)
            origins.pop(option)
        return

    def explicit(option: str) -> bool:
        return origins.get(option, "").startswith(("cli", "settings:", "environment"))

    def effective(option: str, value: str, reason: str) -> None:
        """Make an option describe what runs; refuse an explicit contradiction."""
        if option not in values or values[option] == value:
            return
        if explicit(option):
            raise ConfigError(f"{reason} conflicts with {option} {values[option]}")
        derive(option, value, reason)

    def option_env(option: str, name: str, value: str) -> None:
        """Set an option's variable unless the operator set it, a preset pinned
        it while the option kept its profile default, or the installed vLLM
        does not define it."""
        if option not in values or name in explicit_env:
            return
        if (
            name.startswith("VLLM_")
            and vllm_environment is not None
            and name not in vllm_environment
        ):
            return
        if origins[option].startswith(profile_level) and environment_origins.get(
            name, ""
        ).startswith("preset:"):
            return
        set_env(name, value, origins[option])

    for option, name, decode in (
        (
            "expert-activations",
            "VLLM_B12X_MOE_FP4_FORCE_A16",
            {"1": "bf16", "0": "fp4"},
        ),
        ("router-weights", "B12X_W4A16_FP32_TOPK_WEIGHTS", {"1": "fp32", "0": "bf16"}),
    ):
        if explicit_env.get(name) in decode:
            effective(
                option, decode[explicit_env[name]], f"{name}={explicit_env[name]}"
            )
    threshold_name = "B12X_W4A16_A4_PREFILL_MIN_TOKENS"
    semantic = installed_semantic_a4_prefill()
    if semantic and threshold_name in explicit_env:
        raise ConfigError(
            f"{threshold_name} is no longer read: this image picks the prefill "
            "rows itself. Use prefill-activations (or B12X_W4A16_A4_PREFILL=1)"
        )
    # The variable that turns A4 prefill on in the installed image.
    switch_name = "B12X_W4A16_A4_PREFILL" if semantic else threshold_name
    switch = explicit_env.get(switch_name)
    if switch is not None and "prefill-activations" in values:
        try:
            enabled = int(switch) > 0
        except ValueError:
            enabled = None  # B12X refuses the value at startup
        if enabled is not None:
            effective(
                "prefill-activations",
                "a4" if enabled else "a16",
                f"{switch_name}={switch}",
            )

    bf16 = values.get("expert-activations") == "bf16"
    option_env(
        "expert-activations", "VLLM_B12X_MOE_FP4_FORCE_A16", "1" if bf16 else "0"
    )
    # FP32 router weights belong to the W4A16 combine.
    option_env(
        "router-weights",
        "B12X_W4A16_FP32_TOPK_WEIGHTS",
        "1" if bf16 and values.get("router-weights") == "fp32" else "0",
    )
    prefill = values.get("prefill-activations")
    if prefill not in (None, "a16") and not bf16:
        # FP4 expert activations already quantize prefill and decode alike.
        if explicit("prefill-activations"):
            raise ConfigError(
                f"prefill-activations needs expert-activations bf16; {prefill} "
                "cannot be combined with expert-activations fp4"
            )
        derive("prefill-activations", "a16", "FP4 expert activations")
        prefill = "a16"
    model = Path(str(values.get("model", ""))).name
    if (
        prefill == "a4"
        and model in GLM_A4_PREFILL_UNQUALIFIED
        and switch_name not in explicit_env
    ):
        if explicit("prefill-activations"):
            raise ConfigError(
                f"prefill-activations a4 is not qualified for {values['model']}: "
                "its pre-QAD routed experts lose accuracy with NVFP4 prefill "
                "activations. Use a16, or a QAD checkpoint for a4"
            )
        derive("prefill-activations", "a16", f"{model} (pre-QAD experts)")
        prefill = "a16"
    if prefill is not None:
        on = "1" if semantic else str(GLM_A4_PREFILL_MIN_TOKENS)
        option_env("prefill-activations", switch_name, "0" if prefill == "a16" else on)


CSF_FORMATS = ("nvfp4_csf", "mxfp4_csf")
CSF_SCHEMAS = {
    "lil-nvfp4-csf-checkpoint/1": "nvfp4_csf",
    "lil-mxfp4-csf-checkpoint/1": "mxfp4_csf",
}
CSF_SERVING_ROOT = Path("/tmp/lil-csf")


def csf_serving_config(config: dict, method: str, root: Path) -> dict:
    """config.json that points vLLM's FP4-CSF reader at a checkpoint root."""
    holder = config if "quantization_config" in config else config.get("text_config")
    if not isinstance(holder, dict) or not isinstance(
        holder.get("quantization_config"), dict
    ):
        raise ConfigError("FP4-CSF metadata/config.json has no quantization_config")
    source = holder["quantization_config"]
    location = {"format_version": 1, "checkpoint_root": str(root)}
    if method == "nvfp4_csf":
        holder["quantization_config"] = {
            "quant_method": method,
            **location,
            "source_quantization_config": source,
        }
    else:
        holder["quantization_config"] = {**source, "quant_method": method, **location}
    return config


def csf_format(manifest: Path, source: str, method: str) -> None:
    """Check that a manifest describes an FP4-CSF checkpoint read by method."""
    try:
        schema = json.loads(manifest.read_text()).get("schema")
    except (OSError, ValueError, AttributeError) as error:
        raise ConfigError(
            f"{source} is not an FP4-CSF checkpoint ({error}); "
            "use CHECKPOINT=original for other checkpoints"
        ) from error
    stored = CSF_SCHEMAS.get(schema)
    if stored is None:
        raise ConfigError(
            f"{source} has FP4-CSF schema {schema!r}; this image reads "
            f"{', '.join(sorted(CSF_SCHEMAS))}"
        )
    if stored != method:
        raise ConfigError(f"{source} is a {stored} checkpoint, not {method}")


def local_csf_format(source: str) -> str | None:
    """The FP4-CSF load format a local checkpoint directory needs, if it is one."""
    root = Path(source)
    if not root.is_dir():
        return None
    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            schema = json.loads(manifest.read_text()).get("schema")
        except (OSError, ValueError, AttributeError):
            return None
        return CSF_SCHEMAS.get(schema)
    return "nvfp4_csf" if csf_hf_layout(root) else None


def csf_missing_files(root: Path) -> list[str]:
    """Files the manifest of an FP4-CSF checkpoint lists that are absent or partial."""
    manifest = json.loads((root / "manifest.json").read_text())
    expected: dict[str, int | None] = {"build-contract.json": None}
    for name in manifest.get("metadata_sha256", {}):
        expected[f"metadata/{name}"] = None
    for shard in manifest.get("shards", []):
        expected[f"tensors/{shard['file']}"] = shard.get("target_file_bytes")
    return [
        name
        for name, size in expected.items()
        if not (root / name).is_file()
        or (size is not None and (root / name).stat().st_size != size)
    ]


def csf_hf_layout(root: Path) -> bool:
    """A Hugging Face-layout FP4-CSF checkpoint: config.json and the safetensors
    index at the top, with ModelOpt layer recipes marking CSF expert scales."""
    if (root / "manifest.json").is_file():
        return False
    try:
        config = json.loads((root / "config.json").read_text())
    except (OSError, ValueError):
        return False
    holder = config if "quantization_config" in config else config.get("text_config")
    quant = (holder or {}).get("quantization_config") or {}
    layers = quant.get("quantized_layers") or {}
    return (root / "model.safetensors.index.json").is_file() and any(
        isinstance(recipe, dict) and recipe.get("weight_scale_encoding") == "csf"
        for recipe in layers.values()
    )


def csf_hf_shards(root: Path) -> list[str]:
    """The checkpoint-local shard paths a Hugging Face-layout index names."""
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    for name in shards:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ConfigError(f"{root} indexes a shard outside the checkpoint: {name}")
    return shards


def csf_hf_missing_files(root: Path) -> list[str]:
    """Shards the index of a Hugging Face-layout checkpoint names that are absent."""
    return [name for name in csf_hf_shards(root) if not (root / name).is_file()]


def csf_hf_content_digest(root: Path) -> str:
    """Identity of a Hugging Face-layout checkpoint's content.

    Covers config.json, the index and every indexed shard. A shard in the Hub
    cache is a link to a blob named by its LFS SHA-256; a shard elsewhere is
    identified by its size.
    """
    digest = hashlib.sha256(b"lil-fp4-csf-hf-v1\0")
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update((root / name).read_bytes() + b"\0")
    for name in csf_hf_shards(root):
        target = Path(os.path.realpath(root / name))
        content = (
            target.name
            if target.parent.name == "blobs"
            and re.fullmatch(r"[0-9a-f]{64}", target.name)
            else str(target.stat().st_size)
        )
        digest.update(f"{name}\0{content}\0".encode())
    return digest.hexdigest()


def hub_cache() -> Path:
    """The Hugging Face hub cache directory, located as huggingface_hub does."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(os.environ.get("HF_HOME") or os.path.join(cache, "huggingface")) / "hub"


def pinned_csf_manifests() -> dict[str, dict[str, str]]:
    """Manifest SHA-256 by repository and pinned revision (checkpoints.yaml)."""
    return read_yaml(ROOT / "checkpoints.yaml").get("manifests") or {}


def pinned_csf_contents() -> dict[str, dict[str, str]]:
    """Content digests of pinned Hugging Face-layout FP4-CSF revisions."""
    return read_yaml(ROOT / "checkpoints.yaml").get("contents") or {}


def cached_csf_twin(repository: str, revision: str | None) -> Path | None:
    """A complete cached snapshot with the pinned revision's FP4-CSF manifest.

    checkpoints.yaml records the manifest of every pinned revision. The
    manifest names each shard and metadata file by SHA-256, and revisions that
    change only the model card or license files keep it, so such a snapshot
    holds exactly the pinned checkpoint.
    """
    snapshots = hub_cache() / f"models--{repository.replace('/', '--')}" / "snapshots"
    expected = (pinned_csf_manifests().get(repository) or {}).get(revision)
    if expected is not None and snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir()):
            manifest = snapshot / "manifest.json"
            if (
                manifest.is_file()
                and hashlib.sha256(manifest.read_bytes()).hexdigest() == expected
                and not csf_missing_files(snapshot)
            ):
                return snapshot
    # A Hugging Face-layout revision is identified by its config, index and
    # the LFS SHA-256 of every shard.
    expected = (pinned_csf_contents().get(repository) or {}).get(revision)
    if expected is not None and snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir()):
            if (
                csf_hf_layout(snapshot)
                and not csf_hf_missing_files(snapshot)
                and csf_hf_content_digest(snapshot) == expected
            ):
                return snapshot
    return None


def csf_snapshot(repository: str, revision: str | None, method: str) -> Path:
    """A complete local snapshot of an FP4-CSF Hub repository.

    A complete cached snapshot is used as is, also offline. Otherwise the
    small manifest proves the format before the weights are downloaded. When
    the Hub cannot be reached, a cached revision with the same manifest as
    the pinned one serves it.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    try:
        root = Path(
            snapshot_download(repository, revision=revision, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        root = None
    if root is not None and (root / "manifest.json").is_file():
        csf_format(root / "manifest.json", repository, method)
        if not csf_missing_files(root):
            return root
    if (
        root is not None
        and method == "nvfp4_csf"
        and csf_hf_layout(root)
        and not csf_hf_missing_files(root)
    ):
        return root
    try:
        manifest = hf_hub_download(repository, "manifest.json", revision=revision)
    except EntryNotFoundError:
        # No manifest: a Hugging Face-layout checkpoint, proven by its config
        # and index before the weights are downloaded.
        for name in ("config.json", "model.safetensors.index.json"):
            config = Path(hf_hub_download(repository, name, revision=revision))
        if method != "nvfp4_csf" or not csf_hf_layout(config.parent):
            raise ConfigError(
                f"{repository} is not an FP4-CSF checkpoint; "
                "use CHECKPOINT=original for other checkpoints"
            ) from None
        return Path(snapshot_download(repository, revision=revision))
    except (LocalEntryNotFoundError, OfflineModeIsEnabled):
        # HF_HUB_OFFLINE, or no network. Hub answers such as a missing
        # revision or denied access still fail the launch.
        root = cached_csf_twin(repository, revision)
        if root is None:
            raise
        print(
            f"{repository} revision {revision} is not cached and the Hub cannot "
            f"be reached; serving the cached revision {root.name}, which has "
            "the same FP4-CSF manifest",
            file=sys.stderr,
        )
        return root
    csf_format(Path(manifest), repository, method)
    return Path(snapshot_download(repository, revision=revision))


def prepare_csf_checkpoint(plan: LaunchPlan) -> None:
    """Serve an FP4-CSF checkpoint from a directory vLLM can open.

    An FP4-CSF repository keeps the Hugging Face files under metadata/ and the
    compressed tensors under tensors/. vLLM gets a directory with those files
    and a config.json whose quantization_config names the CSF reader and the
    checkpoint root; the weights stay where they are. vLLM does not download a
    directory, so a Hub checkpoint is downloaded here, at the pinned revision.
    """
    method = plan.values.get("load-format")
    if method not in CSF_FORMATS:
        return
    source = plan.values["model"]
    if Path(source).is_dir():
        root = Path(source).absolute()
    else:
        root = csf_snapshot(source, plan.values.get("revision"), method)
    if method == "nvfp4_csf" and csf_hf_layout(root):
        prepare_hf_layout_csf(plan, source, root)
        return
    csf_format(root / "manifest.json", source, method)
    if method == "nvfp4_csf" and installed_nvfp4_csf_recipes():
        prepare_recipe_csf_container(plan, source, root)
        return
    missing = csf_missing_files(root)
    if missing:
        raise ConfigError(
            f"{source} is incomplete; missing or partial: {', '.join(missing[:5])}"
            + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        )
    try:
        config = json.loads((root / "metadata" / "config.json").read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f"{source} has no readable metadata/config.json") from error
    # The name shows in the vLLM command line, logs and benchmark records.
    name = re.sub(r"[^A-Za-z0-9._-]", "-", Path(source).name) or "checkpoint"
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    serving = CSF_SERVING_ROOT / f"{name}-{digest}"
    if serving.exists():
        shutil.rmtree(serving)
    serving.mkdir(parents=True)
    for item in sorted((root / "metadata").iterdir()):
        if item.is_file() and item.name != "config.json":
            shutil.copyfile(item, serving / item.name)
    (serving / "config.json").write_text(
        json.dumps(csf_serving_config(config, method, root), indent=2) + "\n"
    )
    plan.values["model"] = str(serving)
    plan.origins["model"] = f"resolved:FP4-CSF serving files for {source}"
    # The manifest names every shard and metadata file by SHA-256, so it
    # identifies the checkpoint content wherever the snapshot lives.
    manifest_digest = hashlib.sha256(
        b"lil-fp4-csf-v1\0" + (root / "manifest.json").read_bytes()
    ).hexdigest()
    plan.target_identity = {"identity": manifest_digest, "revision": ""}
    plan.values.pop("revision", None)
    spec = plan.values.get("speculative-config")
    if spec and spec.get("model", source) == source:
        # Drafters read from the target checkpoint follow its serving files.
        spec.pop("revision", None)
        if "model" in spec:
            spec["model"] = str(serving)
    plan.argv = make_argv(plan.values, plan.passthrough)
    print(f"Serving FP4-CSF checkpoint {root} through {serving}", file=sys.stderr)


def safetensors_header(path: Path) -> dict:
    """The JSON header of a safetensors file (tensor names, dtypes, offsets)."""
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size < 1 << 30:
            raise ConfigError(f"{path} is not a safetensors file")
        return json.loads(stream.read(size))


CSF_SCALE_SUFFIX = ".weight_scale.nvfp4_csf_fixed"


def csf_recipe_serving_files(config: dict, root: Path) -> tuple[dict, dict]:
    """config.json and safetensors index that open an NVFP4-CSF container
    through ModelOpt recipes.

    The container's shards already store each compressed expert scale as
    weight_scale.nvfp4_csf_fixed and _exceptions tensors; only the recipes that
    declare them and an index of the stored tensor names are missing.
    """
    weight_map: dict[str, str] = {}
    total = 0
    for shard in sorted((root / "tensors").glob("*.safetensors")):
        for name, info in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            weight_map[name] = f"tensors/{shard.name}"
            total += int(info["data_offsets"][1]) - int(info["data_offsets"][0])
    modules = sorted(
        {
            name[: -len(CSF_SCALE_SUFFIX)].split(".experts.")[0] + ".experts"
            if ".experts." in name
            else name[: -len(CSF_SCALE_SUFFIX)]
            for name in weight_map
            if name.endswith(CSF_SCALE_SUFFIX)
        },
        key=lambda module: [
            int(part) if part.isdigit() else part for part in module.split(".")
        ],
    )
    if not modules:
        raise ConfigError(f"{root} stores no NVFP4-CSF expert scales")
    holder = config if "quantization_config" in config else config.get("text_config")
    if not isinstance(holder, dict) or not isinstance(
        holder.get("quantization_config"), dict
    ):
        raise ConfigError("FP4-CSF metadata/config.json has no quantization_config")
    source = holder["quantization_config"]
    groups = list((source.get("config_groups") or {}).values())
    group = groups[0] if len(groups) == 1 else {}
    weights = group.get("weights") or {
        "dynamic": False,
        "group_size": 16,
        "num_bits": 4,
        "type": "float",
    }
    holder["quantization_config"] = {
        "config_groups": {
            "group_nvfp4_routed_experts": {
                **(
                    {"input_activations": group["input_activations"]}
                    if group.get("input_activations")
                    else {}
                ),
                "weights": weights,
                "targets": modules,
            }
        },
        "ignore": [],
        "producer": source.get("producer"),
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "quantized_layers": {
            module: {
                "group_size": int(weights.get("group_size", 16)),
                "quant_algo": "NVFP4",
                "weight_scale_encoding": "csf",
            }
            for module in modules
        },
    }
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    return config, index


def prepare_recipe_csf_container(plan: LaunchPlan, source: str, root: Path) -> None:
    """Serve an NVFP4-CSF container to a vLLM that reads ModelOpt CSF recipes.

    The serving directory holds the generated config.json and safetensors
    index, the container's other metadata files and a link to its tensors; the
    standard loader then reads the shards where they are.
    """
    missing = csf_missing_files(root)
    if missing:
        raise ConfigError(
            f"{source} is incomplete; missing or partial: {', '.join(missing[:5])}"
            + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        )
    try:
        config = json.loads((root / "metadata" / "config.json").read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f"{source} has no readable metadata/config.json") from error
    config, index = csf_recipe_serving_files(config, root)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", Path(source).name) or "checkpoint"
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    serving = CSF_SERVING_ROOT / f"{name}-{digest}-recipes"
    if serving.exists():
        shutil.rmtree(serving)
    serving.mkdir(parents=True)
    generated = {"config.json", "model.safetensors.index.json", "hf_quant_config.json"}
    for item in sorted((root / "metadata").iterdir()):
        if item.is_file() and item.name not in generated:
            shutil.copyfile(item, serving / item.name)
    (serving / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (serving / "model.safetensors.index.json").write_text(json.dumps(index) + "\n")
    (serving / "tensors").symlink_to(root / "tensors", target_is_directory=True)
    plan.values["model"] = str(serving)
    plan.origins["model"] = f"resolved:NVFP4-CSF recipe serving files for {source}"
    for key, value in (
        ("quantization", "modelopt_mixed"),
        ("load-format", "instanttensor"),
    ):
        plan.values[key] = value
        plan.origins[key] = "resolved:NVFP4-CSF through ModelOpt recipes"
    manifest_digest = hashlib.sha256(
        b"lil-fp4-csf-v1\0" + (root / "manifest.json").read_bytes()
    ).hexdigest()
    plan.target_identity = {"identity": manifest_digest, "revision": ""}
    plan.values.pop("revision", None)
    spec = plan.values.get("speculative-config")
    if spec and spec.get("model", source) == source:
        spec.pop("revision", None)
        if "model" in spec:
            spec["model"] = str(serving)
    plan.argv = make_argv(plan.values, plan.passthrough)
    print(f"Serving NVFP4-CSF checkpoint {root} through {serving}", file=sys.stderr)


def prepare_hf_layout_csf(plan: LaunchPlan, source: str, root: Path) -> None:
    """Serve a Hugging Face-layout FP4-CSF checkpoint from its snapshot.

    Its config.json already names the CSF expert scales, so vLLM reads the
    snapshot directly; no serving directory is needed.
    """
    missing = csf_hf_missing_files(root)
    if missing:
        raise ConfigError(
            f"{source} is incomplete; missing: {', '.join(missing[:5])}"
            + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        )
    plan.values["model"] = str(root)
    plan.origins["model"] = f"resolved:FP4-CSF snapshot of {source}"
    if installed_nvfp4_csf_recipes():
        # The ModelOpt recipes in config.json mark the compressed expert
        # scales; the standard loader reads them as ordinary tensors.
        for key, value in (
            ("quantization", "modelopt_mixed"),
            ("load-format", "instanttensor"),
        ):
            plan.values[key] = value
            plan.origins[key] = "resolved:NVFP4-CSF through ModelOpt recipes"
    plan.target_identity = {
        "identity": csf_hf_content_digest(root),
        "revision": "",
    }
    plan.values.pop("revision", None)
    spec = plan.values.get("speculative-config")
    if spec and spec.get("model", source) == source:
        spec.pop("revision", None)
        if "model" in spec:
            spec["model"] = str(root)
    plan.argv = make_argv(plan.values, plan.passthrough)
    print(f"Serving FP4-CSF checkpoint {root}", file=sys.stderr)


def resolve_draft_subfolder(plan: LaunchPlan) -> None:
    """Point the speculative config at the drafter inside the target checkpoint."""
    if not plan.draft_subfolder or "speculative-config" not in plan.values:
        return
    target = plan.values["model"]
    if Path(target).is_dir():
        root = Path(target)
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        # Fetches only the drafter's files; vLLM downloads the target itself.
        # A cached snapshot is used as is, which also works offline.
        options = {
            "revision": plan.values.get("revision"),
            "allow_patterns": [f"{plan.draft_subfolder}/*"],
        }
        try:
            root = Path(snapshot_download(target, local_files_only=True, **options))
        except LocalEntryNotFoundError:
            root = None
        if root is None or not (root / plan.draft_subfolder / "config.json").is_file():
            root = Path(snapshot_download(target, **options))
    draft = root / plan.draft_subfolder
    if not (draft / "config.json").is_file():
        raise ConfigError(f"Drafter config not found: {draft}/config.json")
    plan.values["speculative-config"]["model"] = str(draft)
    plan.argv = make_argv(plan.values, plan.passthrough)


def mtp_expert_formats(model: str, revision: str | None) -> set[str]:
    """Quantization formats of the MTP experts named by a ModelOpt checkpoint.

    Empty when the checkpoint has no hf_quant_config.json, it cannot be read,
    or it does not list the MTP experts.
    """
    path = Path(model) / "hf_quant_config.json"
    if not Path(model).is_dir():
        from huggingface_hub import hf_hub_download

        # One small file; a cached snapshot is used as is, also offline.
        try:
            path = Path(
                hf_hub_download(
                    model,
                    "hf_quant_config.json",
                    revision=revision,
                    local_files_only=True,
                )
            )
        except OSError:
            try:
                path = Path(
                    hf_hub_download(model, "hf_quant_config.json", revision=revision)
                )
            except OSError:
                return set()
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    # Local Inference Lab exports list quantized_layers at the top level;
    # ModelOpt nests them under "quantization".
    layers = (
        config.get("quantized_layers")
        or (config.get("quantization") or {}).get("quantized_layers")
        or {}
    )
    return {
        str(entry.get("quant_algo"))
        for name, entry in layers.items()
        if name.startswith("mtp.") and name.endswith(".experts")
    }


def resolve_draft_moe_backend(plan: LaunchPlan) -> None:
    """Let vLLM choose the MoE backend of a profile's MTP drafter when b12x cannot run it.

    Profiles run MTP drafter experts on b12x, which takes NVFP4 and MXFP4
    experts, and MXFP8 experts (the Qwen3.8 QAD exports) when the installed
    vLLM and B12X have the MXFP8 path. A checkpoint revision with other MTP
    experts would fail to load. The drafter then gets "auto"; without a
    backend it would inherit the target's --moe-backend b12x. vLLM's choice
    (Marlin for MXFP8) runs them. An explicit speculative-config is kept.
    """
    spec = plan.values.get("speculative-config")
    if (
        not spec
        or spec.get("method") != "mtp"
        or spec.get("moe_backend") != "b12x"
        or not plan.origins["speculative-config"].startswith("derived:")
    ):
        return
    # The drafter's own checkpoint when one is set (--draft-model), else the target.
    drafter = spec.get("model") or plan.values["model"]
    formats = mtp_expert_formats(drafter, spec.get("revision"))
    supported = ("NVFP4", "MXFP4")
    if any("MXFP8" in name for name in formats) and installed_b12x_mxfp8_moe():
        supported += ("MXFP8",)
    unsupported = sorted(
        name for name in formats if not any(kind in name for kind in supported)
    )
    if not unsupported:
        return
    spec["moe_backend"] = "auto"
    plan.argv = make_argv(plan.values, plan.passthrough)
    print(
        f"MTP drafter experts are {', '.join(unsupported)}, which b12x does not "
        "run; the drafter's MoE backend is auto",
        file=sys.stderr,
    )


def chat_template_path(value: str) -> str | None:
    """Resolve a chat-template setting to vLLM's --chat-template value.

    ``checkpoint`` keeps the template shipped with the model (no option), and
    ``runtime:NAME`` names a template installed with these profiles. Any other
    value is a path or template text for vLLM.
    """
    if value == "checkpoint":
        return None
    if not value.startswith(RUNTIME_TEMPLATE_PREFIX):
        return value
    name = value[len(RUNTIME_TEMPLATE_PREFIX) :]
    path = ROOT / name
    if (
        Path(name).is_absolute()
        or path.resolve().parent != (ROOT / "templates").resolve()
        or not path.is_file()
    ):
        raise ConfigError(f"Unknown runtime chat template: {value}")
    return str(path)


def configure_nodes(values: dict, replicas: int, derive) -> None:
    """One tensor-parallel group across several machines (DGX Spark boxes).

    Every node runs this launcher with the same options and its own
    node-rank; rank 0 serves the API and the other ranks run vLLM headless.
    """
    nodes = values.get("nnodes", 1)
    placement = [
        key for key in ("node-rank", "master-addr", "master-port") if key in values
    ]
    if nodes < 1:
        raise ConfigError("nnodes must be at least 1")
    if nodes == 1:
        if placement or values.get("headless"):
            raise ConfigError(f"{', '.join(placement) or 'headless'} needs nnodes > 1")
        return
    if replicas > 1:
        raise ConfigError("replicas and nnodes > 1 cannot be combined")
    if "master-addr" not in values:
        raise ConfigError(
            "nnodes > 1 needs master-addr (MASTER_ADDR), the address of the rank-0 node"
        )
    rank = values.get("node-rank", 0)
    if not 0 <= rank < nodes:
        raise ConfigError(f"node-rank must be between 0 and {nodes - 1}, got {rank}")
    ranks = values.get("tensor-parallel-size", 1) * values.get(
        "pipeline-parallel-size", 1
    )
    if ranks % nodes:
        raise ConfigError(
            f"tensor-parallel-size x pipeline-parallel-size ({ranks}) must split "
            f"evenly across nnodes={nodes}"
        )
    if rank > 0 and not values.get("headless"):
        derive("headless", True, "multi-node worker rank serves no API")


def make_argv(values: dict, passthrough: list[str]) -> list[str]:
    specs = read_yaml(ROOT / "options.yaml")
    command = [
        "/opt/venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        values["model"],
    ]
    for key, value in sorted(values.items()):
        if key == "model" or specs.get(key, {}).get("control"):
            continue
        if key == "chat-template":
            value = chat_template_path(value)
            if value is None:
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
    return command


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
    if {values.get("load-format"), values.get("quantization")} & set(CSF_FORMATS):
        if values.get("load-format") != values.get("quantization"):
            raise ConfigError(
                "FP4-CSF checkpoints need matching --quantization and --load-format "
                "(nvfp4_csf or mxfp4_csf)"
            )
        if values.get("moe-backend") != "b12x" or values.get("enable-expert-parallel"):
            raise ConfigError(
                "FP4-CSF checkpoints decode their experts on B12X with tensor "
                "parallelism only; use CHECKPOINT=original with another MoE "
                "backend or expert parallelism"
            )
    supported_kv = {"fp8", "fp8_e4m3"}
    if identifier == "glm53-flash":
        supported_kv.add("nvfp4_ds_mla")
    if identifier == "mimo26-flash":
        # vllm #882 qualified the exact BF16 cache (half the FP8 capacity).
        supported_kv.add("bfloat16")
    if values["kv-cache-dtype"] not in supported_kv:
        raise ConfigError(
            f"{identifier} supports target KV settings {sorted(supported_kv)}; other precisions require separate qualification"
        )
    if values["tensor-parallel-size"] % values["decode-context-parallel-size"]:
        raise ConfigError("DCP must divide TP")
    if not 0 < values["gpu-memory-utilization"] <= 1:
        raise ConfigError("gpu-memory-utilization must be in (0, 1]")
    if "kv-cache-memory-bytes" in values and values["kv-cache-memory-bytes"] <= 0:
        raise ConfigError("kv-cache-memory-bytes must be positive")
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
    prefill_step_tokens = values.get("max-num-prefill-tokens-per-step")
    if prefill_step_tokens is not None:
        if not isinstance(prefill_step_tokens, int) or prefill_step_tokens < 0:
            raise ConfigError("max-num-prefill-tokens-per-step must be nonnegative")
        if prefill_step_tokens > values["max-num-batched-tokens"]:
            raise ConfigError(
                "max-num-prefill-tokens-per-step cannot exceed max-num-batched-tokens"
            )
        if prefill_step_tokens > 0 and share is None:
            raise ConfigError(
                "max-num-prefill-tokens-per-step requires prefill-compute-share"
            )
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
    if (
        values.get("max-parallel-prefills", 1) == 1
        and values.get("prefill-policy", "round-robin") != "round-robin"
    ):
        raise ConfigError(
            "decode-aware prefill requires max-parallel-prefills > 1 or auto"
        )
    if (
        values.get("prefill-policy", "round-robin") != "decode-aware"
        and values.get("decode-refill-target", "auto") != "auto"
    ):
        raise ConfigError("decode-refill-target requires decode-aware prefill")
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
    prepare_csf_checkpoint(plan)
    resolve_draft_subfolder(plan)
    resolve_draft_moe_backend(plan)
    if plan.values["replicas"] > 1:
        from runtime import replicas

        helper = ROOT / "checkpoint_identity.py"
        if plan.cache_service:
            from runtime.cache import resolve_identity, verify_installed_transfer

            verify_installed_transfer(plan)
            if not helper.is_file():
                raise ConfigError(
                    "The image must package its source-locked checkpoint identity helper"
                )
            resolve_identity(plan, contract, helper)
        if (
            environment.get("VLLM_PLE_TABLE_MEMORY") == "shared"
            and "VLLM_PLE_SHARED_TABLE_IDENTITY" not in environment
        ):
            environment["VLLM_PLE_SHARED_TABLE_IDENTITY"] = (
                replicas.checkpoint_identity(plan, helper)
            )
        raise SystemExit(replicas.run(plan, environment, contract.get("bootstrap", [])))
    if plan.cache_service:
        from runtime.cache import resolve_identity, verify_installed_transfer
        from runtime.supervisor import supervise

        verify_installed_transfer(plan)

        helper = ROOT / "checkpoint_identity.py"
        if not helper.is_file():
            raise ConfigError(
                "The image must package its source-locked checkpoint identity helper"
            )
        resolve_identity(plan, contract, helper)
        command = [
            *contract.get("bootstrap", []),
            *make_argv(plan.values, plan.passthrough),
        ]
        raise SystemExit(
            supervise(
                plan.cache_service, command, environment, contract.get("bootstrap", [])
            )
        )
    command = [*contract.get("bootstrap", []), *plan.argv]
    os.execvpe(command[0], command, environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--profile",
        default=os.environ.get("PROFILE"),
        help="Model profile (or PROFILE): glm53-flash, qwen38-flash-next, ds4-flash, ds4-vision, ds41-flash, mimo26-flash",
    )
    parser.add_argument("--hardware", default=os.environ.get("HARDWARE_PROFILE"))
    parser.add_argument(
        "--preset",
        default=os.environ.get("PRESET"),
        help="Named model/deployment defaults; settings, environment and native arguments can override them",
    )
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--image-contract", type=Path, default=ROOT / "image-contract.json"
    )
    args, native = parser.parse_known_args()
    try:
        deployment = deployment_preset(args.preset) if args.preset else {}
        identifier = args.profile or deployment.get("profile")
        hardware = args.hardware or deployment.get("hardware", "native")
        if not identifier:
            raise ConfigError(
                "Select a model with --profile/PROFILE or --preset/PRESET"
            )
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
            identifier,
            hardware,
            config=read_yaml(args.settings) if args.settings else None,
            argv=native,
            cli_env=explicit_env,
            runtime_identity=runtime_identity,
            preset=args.preset,
            # Inside an image, check the installed vLLM; a CPU-only
            # --print-config outside one resolves the preset as written.
            vllm_environment=installed_vllm_environment() if contract else None,
            csf_formats=installed_csf_formats() if contract else None,
            csf_families=installed_csf_families() if contract else None,
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
