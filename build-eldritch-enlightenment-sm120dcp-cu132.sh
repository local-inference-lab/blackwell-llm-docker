#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Eldritch Enlightenment release + SM120 FlashInfer sparse-MLA DCP support.
#
# vLLM:
# - local-inference-lab/vllm codex/eldritch-sm120-dcp-clean-pr-20260628
#   @ 56fb5d890be75a53aee91446df1fe619e1ed90c1
# - based on codex/eldritch-enlightenment-release-20260627 @ 67e95e7
# - adds PR #60: DCP support for FLASHINFER_MLA_SPARSE_SM120.
#
# B12X:
# - voipmonitor/b12x codex/eldritch-fullstack-20260625
#   @ 284a2eae83754ee1abd31c37b9ca66b68e20b8a8
#
# This is a clean source build. It does not use a runtime overlay and does not
# use VLLM_PATCH_URL.

export IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v56fb5d8-b12x284a2ea-cu132-20260628}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260626}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:glm-kimi-cu132-build-base-20260626}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"

export MAX_JOBS="${MAX_JOBS:-64}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-25dd814e03791e370f96c3148242f0dc8de504ac}"
export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-nv_dev}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-2073ddb2814892014c33ef4cd1c7d4c148baf1fe}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/eldritch-fullstack-20260625}"
export B12X_COMMIT="${B12X_COMMIT:-284a2eae83754ee1abd31c37b9ca66b68e20b8a8}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/eldritch-sm120-dcp-clean-pr-20260628}"
export VLLM_COMMIT="${VLLM_COMMIT:-56fb5d890be75a53aee91446df1fe619e1ed90c1}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+eldritch.enlightenment.56fb5d8.b12x284a2ea.fi25dd814.cu132.20260628}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.6}"

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-eldritch-enlightenment-sm120dcp-20260628"
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
