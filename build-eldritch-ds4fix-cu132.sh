#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Eldritch packed-KV build with the DS4 FlashInfer CUTLASS MTP-off
# throughput regression fix.
#
# vLLM:
# - local-inference-lab/vllm codex/eldritch-ds4-regression-fix-20260625
#   @ 3289aec5ee1df516421bffe6d8013980d54daed6
# - based on codex/eldritch-all-experiments-packedkv-20260624 @ b9239499
# - adds:
#   - no-bias FlashInfer CUTLASS MXFP4 MoE preparation without synthetic zero bias
#   - DS4 breakable-cudagraph attention fast path matching v3 graph shape
#   - Lucifer MHC / DeepGEMM o-proj custom-op wrappers needed for clean
#     FLASHINFER_MLA_SPARSE_DSV4 startup without runtime overlays
#
# B12X:
# - voipmonitor/b12x codex/ds4-packedkv-stride-20260624
#   @ f81d8985e2c387d0dec5f9f310a4e7a45be72adc

export IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-ds4fix-v3289aec-b12xf81d898-cu132-20260625}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260608}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:glm-kimi-cu132-build-base-20260608}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"

export MAX_JOBS="${MAX_JOBS:-64}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-main}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-b3baedbbef2686df91b6dc43818ee56fe26ceba2}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-refs/pull/324/head}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-14073b4e1e706506e193231209738c848d092a1f}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/ds4-packedkv-stride-20260624}"
export B12X_COMMIT="${B12X_COMMIT:-f81d8985e2c387d0dec5f9f310a4e7a45be72adc}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/eldritch-ds4-regression-fix-20260625}"
export VLLM_COMMIT="${VLLM_COMMIT:-3289aec5ee1df516421bffe6d8013980d54daed6}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+eldritch.ds4fix.3289aec.b12xf81d898.fib3baedb.cu132.20260625}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.6}"

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-eldritch-ds4fix-20260625"
if [[ "${FORCE_FLASHINFER_SOURCE:-1}" == "1" ]] \
  && compgen -G "${FLASHINFER_WHEEL_DIR}/flashinfer_*.whl" >/dev/null; then
  rm -rf "${FLASHINFER_WHEEL_STASH}"
  mkdir -p "${FLASHINFER_WHEEL_STASH}"
  mv "${FLASHINFER_WHEEL_DIR}"/flashinfer_*.whl "${FLASHINFER_WHEEL_STASH}"/
  restore_flashinfer_wheels() {
    if compgen -G "${FLASHINFER_WHEEL_STASH}/flashinfer_*.whl" >/dev/null; then
      mv "${FLASHINFER_WHEEL_STASH}"/flashinfer_*.whl "${FLASHINFER_WHEEL_DIR}"/
    fi
    rmdir "${FLASHINFER_WHEEL_STASH}" 2>/dev/null || true
  }
  trap restore_flashinfer_wheels EXIT
fi

./build-vllm-b12x-cu132.sh "$@"
