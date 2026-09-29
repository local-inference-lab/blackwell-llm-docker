#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

release_name="${RELEASE_NAME:-kimi-k3-infernal-invocation-cu133-torch213}"
release_date="${RELEASE_DATE:-$(date -u +%Y%m%d)}"
revision="${REVISION:-r1}"
vllm_composition_root="patches/releases/kimi-k3-infernal-invocation-runtime-r1"
b12x_composition_root="patches/releases/kimi-k3-hh-runtime-r1"
flashkda_composition_root="patches/releases/flashkda-sm120-tensormap-r1"
instanttensor_repo="${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}"
instanttensor_commit="${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"
instanttensor_libaio_repo="${INSTANTTENSOR_LIBAIO_REPO:-https://github.com/1g4-mirror/libaio.git}"
instanttensor_libaio_commit="${INSTANTTENSOR_LIBAIO_COMMIT:-1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448}"
instanttensor_libaio_tree="${INSTANTTENSOR_LIBAIO_TREE:-c9442e111b747e9329ea782c6edb9d13a827cc08}"
cutlass_dsl_version="${CUTLASS_DSL_VERSION:-4.6.0}"
torchvision_version="${TORCHVISION_VERSION:-0.28.0}"
flashinfer_version="${FLASHINFER_VERSION:-0.6.15.post1}"
triton_kernels_repo="${TRITON_KERNELS_REPO:-https://github.com/triton-lang/triton.git}"
triton_kernels_ref="${TRITON_KERNELS_REF:-v3.5.1}"
triton_kernels_commit="${TRITON_KERNELS_COMMIT:-0add68262ab0a2e33b84524346cb27cbb2787356}"

read_lock() {
  local component="$1"
  local prefix="$2"
  local composition_root="$3"
  local lock="${composition_root}/${component}/integration.lock.json"
  local patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}" || { printf 'Missing composition lock: %s\n' "${lock}" >&2; exit 1; }
  test -f "${patch}" || { printf 'Missing integration patch: %s\n' "${patch}" >&2; exit 1; }
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" | sha256sum -c - >/dev/null

  export "${prefix}_REPO=$(jq -er '.base.repository' "${lock}")"
  export "${prefix}_REF=$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${lock}")"
  export "${prefix}_COMMIT=$(jq -er '.base.commit' "${lock}")"
  export "${prefix}_PATCH_FILE=${composition_root#patches/}/${component}/integration.patch"
  export "${prefix}_PATCH_SHA256=$(jq -er '.result.patch_sha256' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.result.tree' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
  export "${prefix}_PRS=$(jq -er '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
}

read_lock vllm VLLM "${vllm_composition_root}"
read_lock b12x B12X "${b12x_composition_root}"
read_lock flashkda FLASHKDA "${flashkda_composition_root}"
FLASHKDA_CUTLASS_COMMIT="$(jq -er '.submodules.cutlass.commit' \
  "${flashkda_composition_root}/flashkda/integration.lock.json")"
export FLASHKDA_CUTLASS_COMMIT
flashkda_manifest="manifests/flashkda/kimi-k3-sm120-tensormap.json"
echo "$(jq -er '.manifest.sha256' \
  "${flashkda_composition_root}/flashkda/integration.lock.json")  ${flashkda_manifest}" \
  | sha256sum -c - >/dev/null
vllm_package_version="${VLLM_PACKAGE_VERSION:-0.11.2.dev280+infernal.${VLLM_INTEGRATION_TREE:0:7}.cu133.torch213}"

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'flashkda_ref=%s\n' "${FLASHKDA_REF}"
  printf 'flashkda_commit=%s\n' "${FLASHKDA_COMMIT}"
  printf 'flashkda_tree=%s\n' "${FLASHKDA_INTEGRATION_TREE}"
  printf 'flashkda_patch=%s\n' "${FLASHKDA_PATCH_FILE}"
  printf 'flashkda_patch_sha256=%s\n' "${FLASHKDA_PATCH_SHA256}"
  printf 'flashkda_cutlass_commit=%s\n' "${FLASHKDA_CUTLASS_COMMIT}"
  exit 0
fi

base_image="${BASE_IMAGE:-voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2}"
if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi
base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'Set ALLOW_DIRTY_BUILD=1 for an image whose recipe is not committed.\n' >&2
  git status --short >&2
  exit 1
fi

cache_fingerprint="cu133-torch213-vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}-flashkda${FLASHKDA_INTEGRATION_TREE:0:10}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-infernal-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-fkda${FLASHKDA_INTEGRATION_TREE:0:7}-cu133-torch213-${release_date}-${revision}}"

printf 'release=%s\n' "${release_name}"
printf 'base=%s (%s)\n' "${base_image}" "${base_image_id}"
printf 'vllm=%s + %s -> %s\n' "${VLLM_COMMIT}" "${VLLM_PRS}" "${VLLM_INTEGRATION_TREE}"
printf 'vllm-package=%s\n' "${vllm_package_version}"
printf 'b12x=%s + %s -> %s\n' "${B12X_COMMIT}" "${B12X_PRS}" "${B12X_INTEGRATION_TREE}"
printf 'flashkda=%s -> %s\n' "${FLASHKDA_COMMIT}" "${FLASHKDA_INTEGRATION_TREE}"
printf 'instanttensor=%s\n' "${instanttensor_commit}"
printf 'flashinfer=%s\n' "${flashinfer_version}"
printf 'triton-kernels=%s\n' "${triton_kernels_commit}"
printf 'image=%s\n' "${image}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "VLLM_PATCH_FILE=${VLLM_PATCH_FILE}" \
  --build-arg "VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}" \
  --build-arg "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg "VLLM_PRS=${VLLM_PRS}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_PATCH_FILE=${B12X_PATCH_FILE}" \
  --build-arg "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "FLASHKDA_REPO=${FLASHKDA_REPO}" \
  --build-arg "FLASHKDA_REF=${FLASHKDA_REF}" \
  --build-arg "FLASHKDA_COMMIT=${FLASHKDA_COMMIT}" \
  --build-arg "FLASHKDA_PATCH_FILE=${FLASHKDA_PATCH_FILE}" \
  --build-arg "FLASHKDA_PATCH_SHA256=${FLASHKDA_PATCH_SHA256}" \
  --build-arg "FLASHKDA_INTEGRATION_TREE=${FLASHKDA_INTEGRATION_TREE}" \
  --build-arg "FLASHKDA_INTEGRATION_LOCK_SHA256=${FLASHKDA_INTEGRATION_LOCK_SHA256}" \
  --build-arg "FLASHKDA_CUTLASS_COMMIT=${FLASHKDA_CUTLASS_COMMIT}" \
  --build-arg "INSTANTTENSOR_REPO=${instanttensor_repo}" \
  --build-arg "INSTANTTENSOR_REF=${instanttensor_commit}" \
  --build-arg "INSTANTTENSOR_COMMIT=${instanttensor_commit}" \
  --build-arg "INSTANTTENSOR_LIBAIO_REPO=${instanttensor_libaio_repo}" \
  --build-arg "INSTANTTENSOR_LIBAIO_COMMIT=${instanttensor_libaio_commit}" \
  --build-arg "INSTANTTENSOR_LIBAIO_TREE=${instanttensor_libaio_tree}" \
  --build-arg "CUTLASS_DSL_VERSION=${cutlass_dsl_version}" \
  --build-arg "FLASHINFER_VERSION=${flashinfer_version}" \
  --build-arg "TORCHVISION_VERSION=${torchvision_version}" \
  --build-arg "TRITON_KERNELS_REPO=${triton_kernels_repo}" \
  --build-arg "TRITON_KERNELS_REF=${triton_kernels_ref}" \
  --build-arg "TRITON_KERNELS_COMMIT=${triton_kernels_commit}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --file Dockerfile.kimi-k3-infernal-invocation-cu133-torch213 \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
assert_label() {
  local key="$1"
  local expected="$2"
  jq -e --arg key "${key}" --arg expected "${expected}" \
    '.[$key] == $expected' <<<"${labels}" >/dev/null || {
      printf 'Image label %s does not match %s\n' "${key}" "${expected}" >&2
      exit 1
    }
}

assert_label local-inference.docker.commit "${docker_commit}"
assert_label local-inference.runtime.base-id "${base_image_id}"
assert_label local-inference.cache.fingerprint "${cache_fingerprint}"
assert_label local-inference.vllm.integration.tree "${VLLM_INTEGRATION_TREE}"
assert_label local-inference.vllm.package-version "${vllm_package_version}"
assert_label local-inference.b12x.integration.tree "${B12X_INTEGRATION_TREE}"
assert_label local-inference.flashkda.integration.tree "${FLASHKDA_INTEGRATION_TREE}"
assert_label local-inference.flashkda.patch_sha256 "${FLASHKDA_PATCH_SHA256}"
assert_label local-inference.flashkda.cutlass.commit "${FLASHKDA_CUTLASS_COMMIT}"
assert_label local-inference.instanttensor.commit "${instanttensor_commit}"
assert_label local-inference.instanttensor.libaio.repo "${instanttensor_libaio_repo}"
assert_label local-inference.instanttensor.libaio.commit "${instanttensor_libaio_commit}"
assert_label local-inference.instanttensor.libaio.tree "${instanttensor_libaio_tree}"
assert_label local-inference.cutlass-dsl.version "${cutlass_dsl_version}"
assert_label local-inference.flashinfer.version "${flashinfer_version}"
assert_label local-inference.torchvision.version "${torchvision_version}"
assert_label local-inference.triton-kernels.commit "${triton_kernels_commit}"

docker run --rm --entrypoint /bin/bash "${image}" -lc \
  '! /usr/local/cuda/bin/cuobjdump --dump-elf-symbols \
      /opt/kimi-k3/vllm/vllm/_flashkda_C.abi3.so \
      | grep -Fq __cuda_syscall_cp_async_bulk_unicast'

docker run --rm --entrypoint /opt/venv/bin/python "${image}" \
  /opt/local-inference/verify_kimi_k3_cu133_runtime.py \
  --instanttensor-version 0.1.9 \
  --cutlass-dsl-version "${cutlass_dsl_version}" \
  --flashinfer-version "${flashinfer_version}" \
  --torchvision-version "${torchvision_version}" \
  --vllm-version "${vllm_package_version}" \
  --runtime-label "infernal-invocation" \
  --vllm-source-root /opt/kimi-k3/vllm \
  --b12x-source-root /opt/kimi-k3/b12x

if [[ "${RUN_NCCL_SMOKE:-1}" == 1 ]]; then
  docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint torchrun "${image}" \
    --standalone --nproc-per-node=16 \
    /opt/local-inference/torch_nccl_smoke.py
fi

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
