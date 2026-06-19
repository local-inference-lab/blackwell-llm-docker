#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Reproducible source build for local-inference-lab/vllm dev/dark-devotion
# plus the current GLM/Kimi/DFlash/DeepSeek tool-call fix stack:
#
#   base: local-inference-lab/vllm dev/dark-devotion @ 774f42d
#   added:
#     - PR26 GLM 5.2 DCP MTP metadata and graph-capture state fixes
#     - PR27 stale spec proposer metadata cleanup
#     - PR24 DFlash sliding-window draft KV under DCP
#     - PR24 follow-up: keep DCP-replicated SlidingWindowSpec groups separate
#     - PR22 MiMo/Qwen3 speculative streaming reasoning/tool-call fixes
#     - DeepSeek V4 dangling-reasoning tool-call recovery
#
# This builds the fixed vLLM branch directly. It does not use a runtime overlay
# and does not use VLLM_PATCH_URL.

export IMAGE="${IMAGE:-voipmonitor/vllm:dark-devotion-772aac6-b12x5b2e018-fi9c5ed7c-pr26-pr27-pr24-pr22-ds4tool-cu132-20260619}"
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
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-33a715e3d9634b64a351855c74ad64e2d9359c7e}"

export B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
export B12X_REF="${B12X_REF:-master}"
export B12X_COMMIT="${B12X_COMMIT:-5b2e018d1c5228436a3ca23f67b17dab55c9cf65}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/dark-devotion-dcp-dflash-tools-20260619}"
export VLLM_COMMIT="${VLLM_COMMIT:-772aac698351de5373bdbaafc0d6e0f95f11a46b}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+dark.devotion.772aac6.b12x5b2e018.fi9c5ed7c.pr26.pr27.pr24.pr22.ds4tool.cu132.20260619}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"

exec ./build-vllm-b12x-cu132.sh "$@"
