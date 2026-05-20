#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:glm-kimi-cu132-20260518}"
MAX_JOBS="${MAX_JOBS:-128}"
VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-128}"
NVCC_THREADS="${NVCC_THREADS:-1}"
VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_REF="${FLASHINFER_REF:-main}"
B12X_REPO="${B12X_REPO:-https://github.com/local-inference-lab/b12x.git}"
B12X_REF="${B12X_REF:-codex/glm51-kimi-b12x-a16-cpuhangfix-cutedsl45-20260512}"
VLLM_REPO="${VLLM_REPO:-https://github.com/voipmonitor/vllm.git}"
VLLM_REF="${VLLM_REF:-codex/glm51-kimi-canonical-rebase-test-20260514}"
LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
CUTLASS_REF="${CUTLASS_REF:-main}"
VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.20.2+glm51kimi.cu132.20260518}"

resolve_ref() {
  local repo="$1"
  local ref="$2"
  local sha=""

  if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
    printf '%s\n' "${ref}"
    return
  fi

  sha="$(git ls-remote "${repo}" "refs/heads/${ref}" | awk 'NR == 1 {print $1}')"
  if [[ -z "${sha}" ]]; then
    sha="$(git ls-remote "${repo}" "refs/tags/${ref}^{}" | awk 'NR == 1 {print $1}')"
  fi
  if [[ -z "${sha}" ]]; then
    sha="$(git ls-remote "${repo}" "${ref}" | awk 'NR == 1 {print $1}')"
  fi
  if [[ -z "${sha}" ]]; then
    echo "Unable to resolve ${repo} ${ref}" >&2
    exit 1
  fi
  printf '%s\n' "${sha}"
}

if [[ "${PIN_SOURCE_COMMITS}" == "1" ]]; then
  NCCL_COMMIT="${NCCL_COMMIT:-$(resolve_ref "${NCCL_REPO}" "${NCCL_REF}")}"
  FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-$(resolve_ref "${FLASHINFER_REPO}" "${FLASHINFER_REF}")}"
  B12X_COMMIT="${B12X_COMMIT:-$(resolve_ref "${B12X_REPO}" "${B12X_REF}")}"
  VLLM_COMMIT="${VLLM_COMMIT:-$(resolve_ref "${VLLM_REPO}" "${VLLM_REF}")}"
  LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-$(resolve_ref "${LAUNCHER_REPO}" "${LAUNCHER_REF}")}"
  CUTLASS_COMMIT="${CUTLASS_COMMIT:-$(resolve_ref "${CUTLASS_REPO}" "${CUTLASS_REF}")}"
else
  NCCL_COMMIT="${NCCL_COMMIT:-}"
  FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-}"
  B12X_COMMIT="${B12X_COMMIT:-}"
  VLLM_COMMIT="${VLLM_COMMIT:-}"
  LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-}"
  CUTLASS_COMMIT="${CUTLASS_COMMIT:-}"
fi

echo "Building ${IMAGE}"
echo "  MAX_JOBS=${MAX_JOBS}"
echo "  VLLM_MAX_JOBS=${VLLM_MAX_JOBS}"
echo "  NVCC_THREADS=${NVCC_THREADS}"
echo "  VLLM_NVCC_THREADS=${VLLM_NVCC_THREADS}"
echo "  FLASHINFER_REF=${FLASHINFER_REF} ${FLASHINFER_COMMIT}"
echo "  B12X_REF=${B12X_REF} ${B12X_COMMIT}"
echo "  VLLM_REF=${VLLM_REF} ${VLLM_COMMIT}"
echo "  LAUNCHER_REF=${LAUNCHER_REF} ${LAUNCHER_COMMIT}"
echo "  CUTLASS_REF=${CUTLASS_REF} ${CUTLASS_COMMIT}"
echo "  NCCL_REF=${NCCL_REF} ${NCCL_COMMIT}"

DOCKER_BUILDKIT=1 docker build \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg VLLM_MAX_JOBS="${VLLM_MAX_JOBS}" \
  --build-arg NVCC_THREADS="${NVCC_THREADS}" \
  --build-arg VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS}" \
  --build-arg NCCL_REPO="${NCCL_REPO}" \
  --build-arg NCCL_REF="${NCCL_REF}" \
  --build-arg NCCL_COMMIT="${NCCL_COMMIT}" \
  --build-arg FLASHINFER_REPO="${FLASHINFER_REPO}" \
  --build-arg FLASHINFER_REF="${FLASHINFER_REF}" \
  --build-arg FLASHINFER_COMMIT="${FLASHINFER_COMMIT}" \
  --build-arg B12X_REPO="${B12X_REPO}" \
  --build-arg B12X_REF="${B12X_REF}" \
  --build-arg B12X_COMMIT="${B12X_COMMIT}" \
  --build-arg VLLM_REPO="${VLLM_REPO}" \
  --build-arg VLLM_REF="${VLLM_REF}" \
  --build-arg VLLM_COMMIT="${VLLM_COMMIT}" \
  --build-arg VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION}" \
  --build-arg LAUNCHER_REPO="${LAUNCHER_REPO}" \
  --build-arg LAUNCHER_REF="${LAUNCHER_REF}" \
  --build-arg LAUNCHER_COMMIT="${LAUNCHER_COMMIT}" \
  --build-arg CUTLASS_REPO="${CUTLASS_REPO}" \
  --build-arg CUTLASS_REF="${CUTLASS_REF}" \
  --build-arg CUTLASS_COMMIT="${CUTLASS_COMMIT}" \
  --progress=plain \
  -f Dockerfile.glm-kimi-cu132 \
  -t "${IMAGE}" \
  "$@" \
  .
