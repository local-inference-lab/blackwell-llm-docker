from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_kimi_k3_qsrt_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("validate_kimi_k3_qsrt_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RuntimeContractError = MODULE.RuntimeContractError
validate_contract = MODULE.validate_contract


def _plan_weights(
    *,
    qsrt_storage_format=None,
    qsrt_profile=None,
):
    del qsrt_storage_format, qsrt_profile


def _prepare_weights(
    *,
    qsrt_atom_payload=None,
    qsrt_first_atom_slot=None,
    qsrt_layer_index=None,
):
    del qsrt_atom_payload, qsrt_first_atom_slot, qsrt_layer_index


class _SafeKernel:
    def __init__(self, *, compile_time_phase: int, **_kwargs) -> None:
        self.stage_inactive_routes = compile_time_phase != 2


def test_validator_accepts_qsrt_atoms_v2_and_inactive_route_safety() -> None:
    result = validate_contract(_plan_weights, _prepare_weights, _SafeKernel)

    assert result == {
        "fixed_m_route_staging": True,
        "fc2_runtime_route_validation": True,
        "qsrt_atoms_v2": True,
    }


def test_validator_rejects_binary_package_without_qsrt_kwargs() -> None:
    def old_plan_weights():
        return None

    with pytest.raises(RuntimeContractError, match="qsrt_storage_format"):
        validate_contract(old_plan_weights, _prepare_weights, _SafeKernel)


def test_validator_rejects_compatibility_tree_without_pr227() -> None:
    class UnsafeKernel:
        def __init__(self, **_kwargs) -> None:
            pass

    with pytest.raises(RuntimeContractError, match="stage_inactive_routes"):
        validate_contract(_plan_weights, _prepare_weights, UnsafeKernel)


def test_validator_rejects_fc2_fixed_route_staging() -> None:
    class UnsafeFc2Kernel:
        def __init__(self, **_kwargs) -> None:
            self.stage_inactive_routes = True

    with pytest.raises(RuntimeContractError, match="FC2-only"):
        validate_contract(_plan_weights, _prepare_weights, UnsafeFc2Kernel)
