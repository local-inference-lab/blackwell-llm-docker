#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Reproducible source build for GLM-5.2 Dark Devotion with PR31 vLLM and
# B12X PR15 packed W4A16 ModelOpt scale reuse + GLM k_segments==12 speed retile.
#
# This is a clean source build. It does not use a runtime overlay and does not
# use VLLM_PATCH_URL.

export IMAGE="${IMAGE:-voipmonitor/vllm:glm52-dark-devotion-pr31-pr15-w4a16scale-vllm79f154c-b12xaecc88f-cu132-20260622}"
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
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-9c5ed7c194e7412780862491742fc655daaad6ac}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-refs/pull/324/head}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-9ca30487a6d1a484757f2d87f532c5f6707b9f25}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/w4a16-modelopt-packed-micro-scales-20260622}"
export B12X_COMMIT="${B12X_COMMIT:-aecc88fa7eee17e3d2f1ac9591b2285ad54fc849}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/dark-devotion-dcp4-mtp3-globaltopk-fix-20260621}"
export VLLM_COMMIT="${VLLM_COMMIT:-79f154c998acd315bd999c8909cfc24085c23f85}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+dark.devotion.pr31.pr15.w4a16scale.79f154c.b12xaecc88f.fi9c5ed7c.cu132.20260622}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-dark-devotion-pr31-pr15"
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
