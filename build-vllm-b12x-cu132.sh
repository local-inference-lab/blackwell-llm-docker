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
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.5.2}"
TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.2}"
TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.9}"
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

local_patch_sha=""
if [[ -n "${VLLM_PATCH_FILE}" ]]; then
  [[ -f "${VLLM_PATCH_FILE}" ]] || {
    echo "VLLM_PATCH_FILE does not exist: ${VLLM_PATCH_FILE}" >&2
    exit 1
  }
  local_patch_sha="$(sha256sum "${VLLM_PATCH_FILE}" | awk '{print $1}')"
  if [[ -n "${VLLM_PATCH_SHA256}" && "${local_patch_sha}" != "${VLLM_PATCH_SHA256}" ]]; then
    echo "VLLM_PATCH_FILE SHA256 mismatch: got ${local_patch_sha}, expected ${VLLM_PATCH_SHA256}" >&2
    exit 1
  fi
fi

runtime_files_sha="$({
  sha256sum Dockerfile.vllm-b12x-cu132
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | awk '{print $1}')"

cache_hash="$(printf '%s\n' \
  "SYSTEM_BASE_IMAGE=${SYSTEM_BASE_IMAGE}" \
  "BUILD_BASE_IMAGE_TAG=${BUILD_BASE_IMAGE_TAG}" \
  "NCCL_REPO=${NCCL_REPO}" \
  "NCCL_REF=${NCCL_REF}" \
  "NCCL_COMMIT=${NCCL_COMMIT}" \
  "FLASHINFER_REPO=${FLASHINFER_REPO}" \
  "FLASHINFER_REF=${FLASHINFER_REF}" \
  "FLASHINFER_COMMIT=${FLASHINFER_COMMIT}" \
  "FLASHINFER_BUILD_CUBIN=${FLASHINFER_BUILD_CUBIN}" \
  "DEEPGEMM_REPO=${DEEPGEMM_REPO}" \
  "DEEPGEMM_REF=${DEEPGEMM_REF}" \
  "DEEPGEMM_COMMIT=${DEEPGEMM_COMMIT}" \
  "B12X_REPO=${B12X_REPO}" \
  "B12X_REF=${B12X_REF}" \
  "B12X_COMMIT=${B12X_COMMIT}" \
  "VLLM_REPO=${VLLM_REPO}" \
  "VLLM_REF=${VLLM_REF}" \
  "VLLM_COMMIT=${VLLM_COMMIT}" \
  "VLLM_PATCH_URL=${VLLM_PATCH_URL}" \
  "VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}" \
  "VLLM_PATCH_FILE_SHA256=${local_patch_sha}" \
  "VLLM_BUILD_VERSION=${VLLM_BUILD_VERSION}" \
  "LAUNCHER_REPO=${LAUNCHER_REPO}" \
  "LAUNCHER_REF=${LAUNCHER_REF}" \
  "LAUNCHER_COMMIT=${LAUNCHER_COMMIT}" \
  "CUTLASS_REPO=${CUTLASS_REPO}" \
  "CUTLASS_REF=${CUTLASS_REF}" \
  "CUTLASS_COMMIT=${CUTLASS_COMMIT}" \
  "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" \
  "TOKENSPEED_MLA_VERSION=${TOKENSPEED_MLA_VERSION}" \
  "TVM_FFI_VERSION=${TVM_FFI_VERSION}" \
  "TRITON_KERNELS_REPO=${TRITON_KERNELS_REPO}" \
  "TRITON_KERNELS_REF=${TRITON_KERNELS_REF}" \
  "TRITON_KERNELS_COMMIT=${TRITON_KERNELS_COMMIT}" \
  "INSTANTTENSOR_REPO=${INSTANTTENSOR_REPO}" \
  "INSTANTTENSOR_REF=${INSTANTTENSOR_REF}" \
  "INSTANTTENSOR_COMMIT=${INSTANTTENSOR_COMMIT}" \
  "HUMMING_KERNELS_SPEC=${HUMMING_KERNELS_SPEC}" \
  "VLLM_RUNTIME_EXTRA_PACKAGES=${VLLM_RUNTIME_EXTRA_PACKAGES}" \
  "RUNTIME_FILES_SHA256=${runtime_files_sha}" \
  | sha256sum | awk '{print substr($1, 1, 16)}')"
vllm_cache_id="${VLLM_COMMIT:0:10}"
b12x_cache_id="${B12X_COMMIT:0:10}"
vllm_cache_id="${vllm_cache_id:-unpinned}"
b12x_cache_id="${b12x_cache_id:-unpinned}"
CACHE_FINGERPRINT="${CACHE_FINGERPRINT:-vllm${vllm_cache_id}-b12x${b12x_cache_id}-${cache_hash}}"
if [[ ! "${CACHE_FINGERPRINT}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ || "${CACHE_FINGERPRINT}" == *..* ]]; then
  echo "Invalid CACHE_FINGERPRINT: ${CACHE_FINGERPRINT}" >&2
  exit 1
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
echo "  CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}"
echo "  TOKENSPEED_MLA_VERSION=${TOKENSPEED_MLA_VERSION}"
echo "  TVM_FFI_VERSION=${TVM_FFI_VERSION}"
echo "  TRITON_KERNELS_REF=${TRITON_KERNELS_REF} ${TRITON_KERNELS_COMMIT}"
echo "  INSTANTTENSOR_REF=${INSTANTTENSOR_REF} ${INSTANTTENSOR_COMMIT}"
echo "  NCCL_REF=${NCCL_REF} ${NCCL_COMMIT}"
echo "  HUMMING_KERNELS_SPEC=${HUMMING_KERNELS_SPEC}"
echo "  VLLM_RUNTIME_EXTRA_PACKAGES=${VLLM_RUNTIME_EXTRA_PACKAGES}"
echo "  CACHE_FINGERPRINT=${CACHE_FINGERPRINT}"

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
    --build-arg CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION}" \
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
  --build-arg CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION}" \
  --build-arg TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION}" \
  --build-arg TVM_FFI_VERSION="${TVM_FFI_VERSION}" \
  --build-arg TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO}" \
  --build-arg TRITON_KERNELS_REF="${TRITON_KERNELS_REF}" \
  --build-arg TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT}" \
  --build-arg INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO}" \
  --build-arg INSTANTTENSOR_REF="${INSTANTTENSOR_REF}" \
  --build-arg INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT}" \
  --build-arg HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC}" \
  --build-arg VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES}" \
  --build-arg CACHE_FINGERPRINT="${CACHE_FINGERPRINT}" \
  --progress=plain \
  -f Dockerfile.vllm-b12x-cu132 \
  -t "${IMAGE}" \
  "$@" \
  .

image_cache_fingerprint="$(docker image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.cache.fingerprint"}}')"
[[ "${image_cache_fingerprint}" == "${CACHE_FINGERPRINT}" ]] || {
  echo "Image cache fingerprint mismatch: got ${image_cache_fingerprint}, expected ${CACHE_FINGERPRINT}" >&2
  exit 1
}

image_env="$(docker image inspect "${IMAGE}" --format '{{range .Config.Env}}{{println .}}{{end}}')"
cache_root="/cache/jit/${CACHE_FINGERPRINT}"
for expected in \
  "LOCAL_INFERENCE_CACHE_FINGERPRINT=${CACHE_FINGERPRINT}" \
  "XDG_CACHE_HOME=${cache_root}" \
  "VLLM_CACHE_ROOT=${cache_root}/vllm" \
  "TRITON_CACHE_DIR=${cache_root}/triton" \
  "TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor" \
  "B12X_CUTE_COMPILE_CACHE_DIR=${cache_root}/b12x-cute" \
  "MM_SPARSE_ATTN_AOT_CACHE=${cache_root}/minfer/mm_sparse_attn"; do
  grep -Fxq "${expected}" <<<"${image_env}" || {
    echo "Image is missing cache environment: ${expected}" >&2
    exit 1
  }
done
