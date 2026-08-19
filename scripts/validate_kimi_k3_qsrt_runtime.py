#!/usr/bin/env python3
"""Validate the Kimi-K3 QSRT and inactive-route runtime contracts."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


class RuntimeContractError(RuntimeError):
    """The active B12X tree cannot safely serve Kimi-K3 QSRT."""


def _require_parameters(
    function: Callable[..., Any],
    names: tuple[str, ...],
) -> None:
    parameters = inspect.signature(function).parameters
    missing = [name for name in names if name not in parameters]
    if missing:
        raise RuntimeContractError(
            f"{function.__module__}.{function.__name__} is missing {', '.join(missing)}"
        )


def validate_contract(
    plan_weights: Callable[..., Any],
    prepare_weights: Callable[..., Any],
    kernel_type: type,
) -> dict[str, bool]:
    """Validate QSRT atoms-v2 and W4A16 inactive-route behavior.

    Args:
        plan_weights: Active B12X weight-planning function.
        prepare_weights: Active B12X weight-preparation function.
        kernel_type: Active fixed-M W4A16 direct-kernel type.

    Returns:
        A machine-readable contract summary.

    Raises:
        RuntimeContractError: The active package is missing a required contract.
    """
    _require_parameters(plan_weights, ("qsrt_storage_format", "qsrt_profile"))
    _require_parameters(
        prepare_weights,
        ("qsrt_atom_payload", "qsrt_first_atom_slot", "qsrt_layer_index"),
    )

    kernel_kwargs = {
        "activation": "silu",
        "fast_math": False,
        "share_input_across_experts": False,
        "share_expert_scales": True,
        "single_token": False,
        "scale_format": "e8m0_k32",
    }
    fixed_m = kernel_type(compile_time_phase=0, **kernel_kwargs)
    fc2_only = kernel_type(compile_time_phase=2, **kernel_kwargs)
    if not hasattr(fixed_m, "stage_inactive_routes"):
        raise RuntimeContractError(
            "the fixed-M W4A16 kernel is missing stage_inactive_routes"
        )
    if fixed_m.stage_inactive_routes is not True:
        raise RuntimeContractError(
            "the fixed-M W4A16 kernel does not stage inactive routes"
        )
    if fc2_only.stage_inactive_routes is not False:
        raise RuntimeContractError(
            "the FC2-only runtime-M path must validate routes inline"
        )

    return {
        "fixed_m_route_staging": True,
        "fc2_runtime_route_validation": True,
        "qsrt_atoms_v2": True,
    }


def main() -> int:
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.w4a16.kernel import (
        MoEMicroKernelW4A16SmallMDirect,
    )

    result = validate_contract(
        fused_moe.plan_weights,
        fused_moe.prepare_weights,
        MoEMicroKernelW4A16SmallMDirect,
    )
    print(
        "Kimi-K3 QSRT runtime contract: "
        + " ".join(f"{name}={int(value)}" for name, value in sorted(result.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
