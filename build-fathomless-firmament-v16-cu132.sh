#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Unified GLM 5.2 and DS4/DSpark v16 build. Every source is pinned by commit;
# model-specific behavior lives in two serve helpers selected by one dispatcher.
export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v16-vllm3104056-b12x90172a5-fi801d57a-cu132-20260713}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260626}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:glm-kimi-cu132-build-base-20260626}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"
export PUSH_IMAGE="${PUSH_IMAGE:-0}"

export MAX_JOBS="${MAX_JOBS:-64}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS=1

export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
export NCCL_COMMIT="${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-codex/sm120-dspark-stack-20260711}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-801d57a08958c13d375ddbb6be3be4808f48a708}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"

# B12X master plus the CuTe compile fallback in PR #28.
export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/ff-v15-cute-compile-fallback-20260709}"
export B12X_COMMIT="${B12X_COMMIT:-90172a504e96d246e07cb1ebad3b291532445560}"

# Current FF plus DS4 PR #88, GLM PRs #90/#91, and stream-lifetime PR #93.
export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v16-unified-20260712}"
export VLLM_COMMIT="${VLLM_COMMIT:-3104056237fffb5cf925909036cf244c4426ff7d}"
export VLLM_PATCH_URL=
export VLLM_PATCH_SHA256=
export VLLM_PATCH_FILE=
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v16.vllm3104056.b12x90172a5.fi801d57a.cu132.20260713}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
export VLLM_REQUIRED_LAUNCHERS="serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
export TRITON_KERNELS_REF=
export TRITON_KERNELS_COMMIT=

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

./build-vllm-b12x-cu132.sh "$@"

docker run --rm --entrypoint /usr/local/bin/serve-glm52-v16.sh \
  -e DRY_RUN=1 \
  -e MODEL=lukealonso/GLM-5.2-NVFP4 \
  -e MTP=0 \
  -e DCP=2 \
  -e MOE_MODE=a16 \
  -e ONLINE_QUANT=mxfp8 \
  "${IMAGE}"

docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=2 \
  "${IMAGE}"

docker run --rm --entrypoint /usr/local/bin/serve-fathomless-firmament.sh \
  -e MODEL_FAMILY=glm52 \
  -e DRY_RUN=1 \
  -e MODEL=lukealonso/GLM-5.2-NVFP4 \
  "${IMAGE}"

if [[ "${PUSH_IMAGE}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
