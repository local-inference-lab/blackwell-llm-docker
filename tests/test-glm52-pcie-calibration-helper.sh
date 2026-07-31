#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
helper="${repo_root}/launchers/serve-glm52-v19.sh"
calibrator="${repo_root}/tests/fake-glm52-pcie-calibrator.sh"

run_helper() {
  env \
    TP=8 \
    DCP=4 \
    GPUS=0,1,2,3,4,5,6,7 \
    PCIE_CALIBRATION_ONLY=1 \
    PCIE_CALIBRATOR="${calibrator}" \
    "$@" \
    "${helper}" 2>&1
}

assert_contains() {
  local output="$1"
  local expected="$2"
  grep -Fqx "${expected}" <<<"${output}" || {
    printf 'missing expected output: %s\n--- output ---\n%s\n' \
      "${expected}" "${output}" >&2
    exit 1
  }
}

measured="$(run_helper \
  FAKE_PCIE_CALIBRATION_BANNER=1 \
  FAKE_EXPECTED_TIMEOUT=600)"
assert_contains "${measured}" "VLLM_DCP_QUERY_SPLIT=1"
assert_contains "${measured}" \
  "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=8192"
assert_contains "${measured}" \
  "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=140000"
assert_contains "${measured}" "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1"
assert_contains "${measured}" "VLLM_PCIE_DMA_MIN_BYTES=25165824"
assert_contains "${measured}" "PCIE_CALIBRATION_STATUS=measured"

# The isolated transport probe does not model the query-split path's E2E
# compute/communication overlap. A negative probe result must not disable the
# release default; operators can still do that explicitly.
split_probe_negative="$(run_helper \
  FAKE_PCIE_QUERY_SPLIT=0 \
  FAKE_PCIE_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0)"
assert_contains "${split_probe_negative}" "VLLM_DCP_QUERY_SPLIT=1"
assert_contains "${split_probe_negative}" \
  "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0"

dcp8_local_fabric="$(run_helper \
  DCP=8 \
  DCP_CKV_PREFETCH_TOPOLOGY=safe \
  PCIE_CALIBRATION=off)"
assert_contains "${dcp8_local_fabric}" "VLLM_DCP_TOPK_OWNER_MERGE=1"

dcp8_cross_numa="$(run_helper \
  DCP=8 \
  DCP_CKV_PREFETCH_TOPOLOGY=unsafe \
  PCIE_CALIBRATION=off)"
assert_contains "${dcp8_cross_numa}" "VLLM_DCP_TOPK_OWNER_MERGE=0"

source_alias="$(run_helper \
  VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=196608)"
assert_contains "${source_alias}" \
  "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=196608"

# An empty Compose GPUS entry must preserve an intentional CUDA device order
# for both calibration and the vLLM launcher that follows it.
ordered_gpus="0,2,4,6,1,3,5,7"
inherited_order="$(run_helper \
  GPUS= \
  CUDA_VISIBLE_DEVICES="${ordered_gpus}" \
  FAKE_EXPECTED_GPUS="${ordered_gpus}")"
assert_contains "${inherited_order}" "PCIE_CALIBRATION_STATUS=measured"

v16_output="$(env \
  GPUS= \
  CUDA_VISIBLE_DEVICES="${ordered_gpus}" \
  DRY_RUN=1 \
  MODEL=/model \
  TP=8 \
  DCP=1 \
  MTP=0 \
  MAX_NUM_SEQS=1 \
  GRAPH=6 \
  "${repo_root}/launchers/serve-glm52-v16.sh")"
assert_contains "${v16_output}" \
  'CUDA_VISIBLE_DEVICES=0\,2\,4\,6\,1\,3\,5\,7'

no_dma="$(run_helper FAKE_PCIE_DMA_MIN_BYTES=off)"
assert_contains "${no_dma}" "VLLM_PCIE_DMA_MIN_BYTES=off"

explicit="$(run_helper \
  DCP_QUERY_SPLIT=0 \
  DCP_CKV_GATHER=1 \
  DCP_CKV_GATHER_MAX_TOKENS=262144 \
  DCP_CKV_PREFETCH_DEPTH=0 \
  PCIE_DMA_MIN_BYTES=12MB)"
assert_contains "${explicit}" "VLLM_DCP_QUERY_SPLIT=0"
assert_contains "${explicit}" \
  "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0"
assert_contains "${explicit}" \
  "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=262144"
assert_contains "${explicit}" "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0"
assert_contains "${explicit}" "VLLM_PCIE_DMA_MIN_BYTES=12MB"
assert_contains "${explicit}" "PCIE_CALIBRATION_STATUS=skipped:all-explicit"

explicit_zero="$(run_helper \
  DCP_QUERY_SPLIT=0 \
  DCP_CKV_GATHER=1 \
  DCP_CKV_PREFETCH_DEPTH=0 \
  PCIE_DMA_MIN_BYTES=0)"
assert_contains "${explicit_zero}" "VLLM_PCIE_DMA_MIN_BYTES=off"
assert_contains "${explicit_zero}" \
  "PCIE_CALIBRATION_STATUS=skipped:all-explicit"

compressed="$(run_helper \
  F8_DMA=ring \
  DCP_CKV_PREFETCH_TOPOLOGY=safe)"
assert_contains "${compressed}" \
  "PCIE_CALIBRATION_STATUS=skipped:explicit-compressed-dma"
assert_contains "${compressed}" "VLLM_PCIE_DMA_MIN_BYTES=6MB"

fallback="$(run_helper \
  FAKE_PCIE_CALIBRATION_FAIL=1 \
  DCP_CKV_PREFETCH_TOPOLOGY=unsafe)"
assert_contains "${fallback}" \
  "PCIE_CALIBRATION_STATUS=failed:fallback-to-topology"
assert_contains "${fallback}" "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0"
assert_contains "${fallback}" "VLLM_PCIE_DMA_MIN_BYTES=6MB"

if run_helper PCIE_CALIBRATION_TIMEOUT=0 >/dev/null; then
  echo "zero calibration timeout was accepted" >&2
  exit 1
fi

if run_helper DCP_CKV_GATHER_MAX_TOKENS=0 >/dev/null; then
  echo "zero CKV gather capacity was accepted" >&2
  exit 1
fi

if run_helper DCP_CKV_GATHER_MAX_TOKENS=not-a-number >/dev/null; then
  echo "non-numeric CKV gather capacity was accepted" >&2
  exit 1
fi

runtime_env_script="${repo_root}/launchers/glm52-pcie-runtime-env.sh"
run_runtime_env_preload() {
  local preload_state="$1"
  local initial_preload="${2:-}"

  RUNTIME_ENV_SCRIPT="${runtime_env_script}" \
    TEST_NCCL_PATH="${calibrator}" \
    bash -c '
    case "$1" in
      unset) unset LD_PRELOAD ;;
      empty) export LD_PRELOAD= ;;
      set) export LD_PRELOAD="$2" ;;
      unexported) LD_PRELOAD="$2" ;;
      *) exit 64 ;;
    esac
    set -u
    source "${RUNTIME_ENV_SCRIPT}"
    export NCCL_LOCAL_INFERENCE_PATH="${TEST_NCCL_PATH}"
    configure_glm52_pcie_runtime_env 1 0
    configure_glm52_pcie_runtime_env 1 0
    [[ "$(declare -p LD_PRELOAD)" == "declare -x "* ]] || {
      printf "LD_PRELOAD was not exported: %s\n" \
        "$(declare -p LD_PRELOAD)" >&2
      exit 65
    }
    printf "%s\n" "${LD_PRELOAD}"
  ' _ "${preload_state}" "${initial_preload}"
}

unset_preload_output="$(run_runtime_env_preload unset)"
assert_contains "${unset_preload_output}" "${calibrator}"

empty_preload_output="$(run_runtime_env_preload empty)"
assert_contains "${empty_preload_output}" "${calibrator}"

preload_output="$(
  run_runtime_env_preload set /tmp/existing-preload.so
)"
assert_contains "${preload_output}" "${calibrator}:/tmp/existing-preload.so"

unexported_preload_output="$(
  run_runtime_env_preload unexported "${calibrator}"
)"
assert_contains "${unexported_preload_output}" "${calibrator}"

unexported_other_output="$(
  run_runtime_env_preload unexported /tmp/unexported-preload.so
)"
assert_contains \
  "${unexported_other_output}" \
  "${calibrator}:/tmp/unexported-preload.so"

space_separated_output="$(
  run_runtime_env_preload set "/tmp/first-preload.so /tmp/second-preload.so"
)"
assert_contains \
  "${space_separated_output}" \
  "${calibrator}:/tmp/first-preload.so /tmp/second-preload.so"

echo "GLM-5.2 PCIe calibration helper: PASS"
