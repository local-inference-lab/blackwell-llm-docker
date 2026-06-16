#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Reproducible source build for chthonic-consecration plus the DS4/GLM/Kimi
# runtime fix stack:
#
#   base: local-inference-lab/vllm dev/chthonic-consecration @ 225f431
#   added:
#     - Recover DeepSeek V4 tool calls trapped in reasoning
#     - PR44297 structured-output/spec-decode </think> FSM fixes
#     - PR20 B12X NVFP4 forced-W4A16 prepare fix
#     - Kimi EAGLE3 draft config + DCP draft MLA seq-lens fix
#     - PR22 MiMo-V2 speculative streaming reasoning/tool-call fixes
#
# This intentionally builds the fixed vLLM branch directly. It is not a Python
# file overlay and does not use VLLM_PATCH_URL.

export IMAGE="${IMAGE:-voipmonitor/vllm:chthonic-consecration-92962d2-b12x0ff2847-pr20-kimi-eagle3-pr22-cu132-20260616}"
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
export FLASHINFER_REF="${FLASHINFER_REF:-refs/pull/3395/head}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-b41aa8dd2fb93c49b1c6134bd1953040f8089d51}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-refs/pull/324/head}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-33a715e3d9634b64a351855c74ad64e2d9359c7e}"

export B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
export B12X_REF="${B12X_REF:-master}"
export B12X_COMMIT="${B12X_COMMIT:-0ff2847b0c55c599c8acabb32e694ce07faa1247}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/chthonic-pr20-kimi-eagle3-pr22-20260616}"
export VLLM_COMMIT="${VLLM_COMMIT:-92962d26689cc6953e3a9f6f55e92ebc16086ec3}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+chthonic.consecration.92962d2.b12x0ff2847.pr20.kimi.eagle3.pr22.cu132.20260616}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-main}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"

exec ./build-vllm-b12x-cu132.sh "$@"
