#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Reproducible source build for the GLM-5.2 Dark Devotion release stack.
#
# vLLM branch:
# - dev/dark-devotion @ 1bcacdeb
# - PR33 MXFP8 sparse-indexer WK loader
# - PR35 serialized MXFP8 MTP module loader
# - PR36 DCP draft KV sharding default
# - PR34 parser fix (local port of upstream vLLM PR #46149)
#
# B12X branch:
# - master @ 5af873a
# - PR15 W4A16 packed-scale memory/speed fix
# - PR15 follow-up: keep legacy E4M3 scale decode outside W4A16
# - PR14 GLM TP6 odd-head prefill split
#
# This is a clean source build. It does not use a runtime overlay and does not
# use VLLM_PATCH_URL.

export IMAGE="${IMAGE:-voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622}"
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
export B12X_REF="${B12X_REF:-codex/dark-devotion-pr14-pr15-20260622}"
export B12X_COMMIT="${B12X_COMMIT:-aaf1891861ab86e78561326f13156d69a51a3ed8}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/dark-devotion-release-20260622}"
export VLLM_COMMIT="${VLLM_COMMIT:-ec656676100a756912d6966c4232ea436c55d792}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+dark.devotion.release.ec65667.b12xaaf1891.fi9c5ed7c.cu132.20260622}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-dark-devotion-release-20260622"
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
