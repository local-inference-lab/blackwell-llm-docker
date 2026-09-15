"""Audit model-neutral image metadata and lock the installed runtime profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from runtime.launcher import ConfigError, JIT_PATHS, ROOT, read_yaml


def owned_environment() -> set[str]:
    """Return configuration names that cannot be baked into a generic image."""
    names = {
        alias
        for item in read_yaml(ROOT / "options.yaml").values()
        for alias in item["env"]
    }
    names.update(JIT_PATHS)
    for directory in (ROOT / "profiles", ROOT / "hardware"):
        for path in directory.glob("*.yaml"):
            data = read_yaml(path)
            names.update(data["environment"])
            for environment in data.get("model_environment", {}).values():
                names.update(environment)
    names.update(
        {
            "FAIRNESS_ENGINE",
            "LMCACHE_MODE",
            "LMCACHE_ENABLED",
            "DS4_OMP_NUM_THREADS",
            "DS4_MAX_CUDAGRAPH_CAPTURE_SIZE",
            "DS4_CUDAGRAPH_CAPTURE_SIZES",
            "MODE",
            "BACKEND",
            "VLLM_DEFAULT_MOE_BACKEND",
        }
    )
    return names


def audit_image_metadata(metadata: dict) -> None:
    config = metadata.get("Config", {})
    environment = config.get("Env") or []
    names = {item.partition("=")[0] for item in environment}
    forbidden = sorted(
        (names & owned_environment())
        | {name for name in names if name.startswith(("VLLM_GLM53_", "VLLM_QWEN3_8_"))}
    )
    if forbidden:
        raise ConfigError(
            "Image Config.Env contains profile-owned values: " + ", ".join(forbidden)
        )
    if not metadata.get("Id", "").startswith("sha256:"):
        raise ConfigError("Image inspection must identify an immutable image ID")


def payload_hashes(root: Path = ROOT) -> dict[str, str]:
    result = {}
    candidates = [
        *root.iterdir(),
        *(root / "profiles").glob("*.yaml"),
        *(root / "hardware").glob("*.yaml"),
    ]
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and (
                path.suffix in {".py", ".json", ".yaml", ".txt", ".sh"}
                or path.name == "lil-serve"
            )
            and relative.parts[0] not in {"tests", "generated", "__pycache__"}
            and path.name != "image-contract.json"
        ):
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def make_contract(
    metadata: dict, runtime_lock_sha256: str, bootstrap: list[str] | None = None
) -> dict:
    audit_image_metadata(metadata)
    if not re.fullmatch(r"[0-9a-f]{64}", runtime_lock_sha256):
        raise ConfigError(
            "runtime-lock-sha256 must identify the source lock or wheel lock"
        )
    return {
        "schema_version": 1,
        "status": "implemented",
        "environment_contract": "model-neutral",
        "foundation_image_id": metadata["Id"],
        "runtime_lock_sha256": runtime_lock_sha256,
        "bootstrap": bootstrap or [],
        "files": payload_hashes(),
        "qualified_serving_receipts": [],
    }


def verify_contract(path: Path) -> dict:
    try:
        contract = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(
            "Execution requires an installed, build-audited image-contract.json; --print-config remains CPU-only"
        ) from error
    if (
        contract.get("schema_version") != 1
        or contract.get("environment_contract") != "model-neutral"
    ):
        raise ConfigError("Unsupported image environment contract")
    if contract.get("files") != payload_hashes():
        raise ConfigError(
            "Installed launcher/profile hashes differ from the image contract"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", contract.get("runtime_lock_sha256", "")):
        raise ConfigError("Image contract has no valid runtime lock identity")
    bootstrap = contract.get("bootstrap", [])
    if (
        not isinstance(bootstrap, list)
        or any(not isinstance(arg, str) for arg in bootstrap)
        or (bootstrap and not Path(bootstrap[0]).is_absolute())
    ):
        raise ConfigError(
            "Image bootstrap must be an absolute executable and an argv list"
        )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-inspect", type=Path, required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--bootstrap", action="append", default=[])
    args = parser.parse_args()
    metadata = json.loads(args.image_inspect.read_text())
    if isinstance(metadata, list):
        if len(metadata) != 1:
            parser.error("Inspect exactly one immutable foundation image")
        metadata = metadata[0]
    try:
        result = make_contract(metadata, args.runtime_lock_sha256, args.bootstrap)
    except ConfigError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
