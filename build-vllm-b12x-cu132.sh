#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:vllm-b12x-cu132}"
SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:vllm-b12x-cu132-system-base}"
BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:vllm-b12x-cu132-build-base}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-1}"
PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"
MAX_JOBS="${MAX_JOBS:-64}"
VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
NVCC_THREADS="${NVCC_THREADS:-1}"
VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_REF="${FLASHINFER_REF:-refs/pull/3395/head}"
FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-1}"
DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
DEEPGEMM_REF="${DEEPGEMM_REF:-refs/pull/324/head}"
B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
B12X_REF="${B12X_REF:-refs/pull/11/head}"
VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
VLLM_REF="${VLLM_REF:-dev/black-benediction}"
VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-}"
LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-}"
CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
CUTLASS_REF="${CUTLASS_REF:-main}"
VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+black.benediction.b12x.cu132}"
TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO:-https://github.com/triton-lang/triton.git}"
TRITON_KERNELS_REF="${TRITON_KERNELS_REF:-}"
INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-main}"
HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.4}"
VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-}"

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
  DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-$(resolve_ref "${DEEPGEMM_REPO}" "${DEEPGEMM_REF}")}"
  B12X_COMMIT="${B12X_COMMIT:-$(resolve_ref "${B12X_REPO}" "${B12X_REF}")}"
  VLLM_COMMIT="${VLLM_COMMIT:-$(resolve_ref "${VLLM_REPO}" "${VLLM_REF}")}"
  LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-$(resolve_ref "${LAUNCHER_REPO}" "${LAUNCHER_REF}")}"
  CUTLASS_COMMIT="${CUTLASS_COMMIT:-$(resolve_ref "${CUTLASS_REPO}" "${CUTLASS_REF}")}"
  if [[ -n "${TRITON_KERNELS_REF}" ]]; then
    TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-$(resolve_ref "${TRITON_KERNELS_REPO}" "${TRITON_KERNELS_REF}")}"
  else
    TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-}"
  fi
  if [[ -n "${INSTANTTENSOR_REF}" ]]; then
    INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-$(resolve_ref "${INSTANTTENSOR_REPO}" "${INSTANTTENSOR_REF}")}"
  else
    INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-}"
  fi
else
  NCCL_COMMIT="${NCCL_COMMIT:-}"
  FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-}"
  DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-}"
  B12X_COMMIT="${B12X_COMMIT:-}"
  VLLM_COMMIT="${VLLM_COMMIT:-}"
  LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-}"
  CUTLASS_COMMIT="${CUTLASS_COMMIT:-}"
  TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-}"
  INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-}"
fi

echo "Building ${IMAGE}"
echo "  SYSTEM_BASE_IMAGE=${SYSTEM_BASE_IMAGE}"
echo "  BUILD_BASE_IMAGE_TAG=${BUILD_BASE_IMAGE_TAG}"
echo "  BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}"
echo "  PUSH_BASE_IMAGE=${PUSH_BASE_IMAGE}"
echo "  MAX_JOBS=${MAX_JOBS}"
echo "  VLLM_MAX_JOBS=${VLLM_MAX_JOBS}"
echo "  NVCC_THREADS=${NVCC_THREADS}"
echo "  VLLM_NVCC_THREADS=${VLLM_NVCC_THREADS}"
echo "  FLASHINFER_REF=${FLASHINFER_REF} ${FLASHINFER_COMMIT}"
echo "  FLASHINFER_BUILD_CUBIN=${FLASHINFER_BUILD_CUBIN}"
echo "  DEEPGEMM_REF=${DEEPGEMM_REF} ${DEEPGEMM_COMMIT}"
echo "  B12X_REF=${B12X_REF} ${B12X_COMMIT}"
echo "  VLLM_REF=${VLLM_REF} ${VLLM_COMMIT}"
echo "  VLLM_PATCH_URL=${VLLM_PATCH_URL}"
echo "  VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}"
echo "  VLLM_PATCH_FILE=${VLLM_PATCH_FILE}"
echo "  LAUNCHER_REF=${LAUNCHER_REF} ${LAUNCHER_COMMIT}"
echo "  VLLM_REQUIRED_LAUNCHERS=${VLLM_REQUIRED_LAUNCHERS}"
echo "  CUTLASS_REF=${CUTLASS_REF} ${CUTLASS_COMMIT}"
echo "  TRITON_KERNELS_REF=${TRITON_KERNELS_REF} ${TRITON_KERNELS_COMMIT}"
echo "  INSTANTTENSOR_REF=${INSTANTTENSOR_REF} ${INSTANTTENSOR_COMMIT}"
echo "  NCCL_REF=${NCCL_REF} ${NCCL_COMMIT}"
echo "  HUMMING_KERNELS_SPEC=${HUMMING_KERNELS_SPEC}"
echo "  VLLM_RUNTIME_EXTRA_PACKAGES=${VLLM_RUNTIME_EXTRA_PACKAGES}"

if [[ "${BUILD_BASE_IMAGE}" == "1" ]]; then
  DOCKER_BUILDKIT=1 docker build \
    --target vllm-b12x-cu132-system-base-build \
    --build-arg NCCL_REPO="${NCCL_REPO}" \
    --build-arg NCCL_REF="${NCCL_REF}" \
    --build-arg NCCL_COMMIT="${NCCL_COMMIT}" \
    --progress=plain \
    -f Dockerfile.vllm-b12x-cu132 \
    -t "${SYSTEM_BASE_IMAGE}" \
    "$@" \
    .

  DOCKER_BUILDKIT=1 docker build \
    --target vllm-b12x-cu132-build-base-build \
    --build-arg VLLM_B12X_CU132_SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE}" \
    --progress=plain \
    -f Dockerfile.vllm-b12x-cu132 \
    -t "${BUILD_BASE_IMAGE_TAG}" \
    "$@" \
    .

  if [[ "${PUSH_BASE_IMAGE}" == "1" ]]; then
    docker push "${SYSTEM_BASE_IMAGE}"
    docker push "${BUILD_BASE_IMAGE_TAG}"
  fi
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg VLLM_B12X_CU132_SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE}" \
  --build-arg VLLM_B12X_CU132_BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE_TAG}" \
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
  --build-arg FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN}" \
  --build-arg DEEPGEMM_REPO="${DEEPGEMM_REPO}" \
  --build-arg DEEPGEMM_REF="${DEEPGEMM_REF}" \
  --build-arg DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT}" \
  --build-arg B12X_REPO="${B12X_REPO}" \
  --build-arg B12X_REF="${B12X_REF}" \
  --build-arg B12X_COMMIT="${B12X_COMMIT}" \
  --build-arg VLLM_REPO="${VLLM_REPO}" \
  --build-arg VLLM_REF="${VLLM_REF}" \
  --build-arg VLLM_COMMIT="${VLLM_COMMIT}" \
  --build-arg VLLM_PATCH_URL="${VLLM_PATCH_URL}" \
  --build-arg VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256}" \
  --build-arg VLLM_PATCH_FILE="${VLLM_PATCH_FILE}" \
  --build-arg VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION}" \
  --build-arg LAUNCHER_REPO="${LAUNCHER_REPO}" \
  --build-arg LAUNCHER_REF="${LAUNCHER_REF}" \
  --build-arg LAUNCHER_COMMIT="${LAUNCHER_COMMIT}" \
  --build-arg VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS}" \
  --build-arg CUTLASS_REPO="${CUTLASS_REPO}" \
  --build-arg CUTLASS_REF="${CUTLASS_REF}" \
  --build-arg CUTLASS_COMMIT="${CUTLASS_COMMIT}" \
  --build-arg TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO}" \
  --build-arg TRITON_KERNELS_REF="${TRITON_KERNELS_REF}" \
  --build-arg TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT}" \
  --build-arg INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO}" \
  --build-arg INSTANTTENSOR_REF="${INSTANTTENSOR_REF}" \
  --build-arg INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT}" \
  --build-arg HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC}" \
  --build-arg VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES}" \
  --progress=plain \
  -f Dockerfile.vllm-b12x-cu132 \
  -t "${IMAGE}" \
  "$@" \
  .
