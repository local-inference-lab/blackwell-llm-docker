#!/usr/bin/env bash
# Compile Kimi-K3 from verified Infernal Invocation and B12X merge stacks.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

source_root=patches/releases/kimi-k3-upstream-aligned-20260918
release_name="${RELEASE_NAME:-kimi-k3-upstream-aligned}"
release_date="${RELEASE_DATE:-20260918}"
revision="${REVISION:-r38}"
runtime_foundation_image="${RUNTIME_FOUNDATION_IMAGE:-voipmonitor/vllm@sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2}"
base_image="${BASE_IMAGE:-${runtime_foundation_image}}"
runtime_foundation="${RUNTIME_FOUNDATION:-1}"
flashinfer_wheel_image="${FLASHINFER_WHEEL_IMAGE:-voipmonitor/vllm:flashinfer-wheels-fi1ac6942-cu133-torch213-20260820-r1@sha256:477a3b55b973df48b08a6dfae4a2a1e64c975a990dda22f65e31acd5217b86bb}"
flashinfer_commit=1ac6942776b383c6b03c7a5805a22e72a3e3349f
flashinfer_version=0.6.18+cu133
cutlass_dsl_version=4.6.2

read_source_lock() {
  local component=$1 prefix=$2
  local lock="${source_root}/${component}/source.lock.json"
  test -f "${lock}" || {
    printf 'Missing source lock: %s\n' "${lock}" >&2
    exit 1
  }
  jq -e \
    '.schema_version == 1 and .component == $component and
     (.source_patches | length) == 0' \
    --arg component "${component}" "${lock}" >/dev/null

  export "${prefix}_REPO=$(jq -er '.source.repository' "${lock}")"
  export "${prefix}_REF=$(jq -er '.source.ref | sub("^refs/heads/"; "")' "${lock}")"
  export "${prefix}_COMMIT=$(jq -er '.source.commit' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.source.tree' "${lock}")"
  export "${prefix}_UPSTREAM_BASE=$(jq -er '.upstream_base.commit' "${lock}")"
  export "${prefix}_MERGE_HEADS=$(jq -r \
    '[.pull_requests[].head, (.qualification_candidate_commits // [])[].head] | join(",")' \
    "${lock}")"
  export "${prefix}_PRS=$(jq -r '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
}

read_source_lock vllm VLLM
read_source_lock b12x B12X
read_source_lock lmcache LMCACHE

test "${VLLM_UPSTREAM_BASE}" = b5f995e73e6b7fe27c9927477e277a151ebcc9e9
test "${B12X_UPSTREAM_BASE}" = 36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8
test "${LMCACHE_UPSTREAM_BASE}" = "${LMCACHE_COMMIT}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'The clean runtime recipe must be committed before build.\n' >&2
  git status --short >&2
  exit 1
fi

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi

base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
flashinfer_base_id="${base_image_id}"

if [[ "${runtime_foundation}" = 1 ]]; then
  foundation_labels="$(docker image inspect "${base_image}" \
    --format '{{json .Config.Labels}}')"
  jq -e \
    '."local-inference.runtime.foundation.source-packages" == "absent" and
     ."local-inference.cuda.version" == "13.3" and
     ."local-inference.torch.version" == "2.13.0" and
     ."local-inference.flashinfer.version" == "0.6.18+cu133" and
     ."local-inference.rust.toolchain" == "1.95"' \
    <<<"${foundation_labels}" >/dev/null
  flashinfer_base_id="$(jq -er '."local-inference.runtime.base-id"' \
    <<<"${foundation_labels}")"
fi

if ! docker image inspect "${flashinfer_wheel_image}" >/dev/null 2>&1; then
  docker pull "${flashinfer_wheel_image}" || \
    BASE_IMAGE="${base_image}" IMAGE="${flashinfer_wheel_image}" \
      FLASHINFER_COMMIT="${flashinfer_commit}" \
      FLASHINFER_VERSION="${flashinfer_version}" \
      CUTLASS_DSL_VERSION="${cutlass_dsl_version}" \
      ./build-flashinfer-cu133-torch213-wheels.sh
fi

flashinfer_labels="$(docker image inspect "${flashinfer_wheel_image}" \
  --format '{{json .Config.Labels}}')"
jq -e \
  --arg base_id "${flashinfer_base_id}" \
  --arg commit "${flashinfer_commit}" \
  --arg version "${flashinfer_version}" \
  --arg cutlass "${cutlass_dsl_version}" \
  '."local-inference.runtime.base-id" == $base_id and
   ."local-inference.flashinfer.commit" == $commit and
   ."local-inference.flashinfer.version" == $version and
   ."local-inference.cutlass-dsl.version" == $cutlass' \
  <<<"${flashinfer_labels}" >/dev/null
flashinfer_wheel_image_id="$(docker image inspect "${flashinfer_wheel_image}" \
  --format '{{.Id}}')"

docker_commit="$(git rev-parse HEAD)"
source_lock_sha256="$({
  sha256sum \
    "${source_root}/vllm/source.lock.json" \
    "${source_root}/b12x/source.lock.json" \
    "${source_root}/lmcache/source.lock.json"
} | sha256sum | cut -d' ' -f1)"
cache_fingerprint="cu133-torch213-kimi-k3-clean-vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}-lmcache${LMCACHE_INTEGRATION_TREE:0:10}"
vllm_package_version="${VLLM_PACKAGE_VERSION:-0.26.1rc0+kimi.k3.aligned.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}}"
core_image="${CORE_IMAGE:-voipmonitor/vllm:kimi-k3-upstream-aligned-core-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-cu133-torch213-${release_date}-${revision}}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-nativekv-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-cu133-torch213-${release_date}-${revision}}"

printf 'base=%s id=%s\n' "${base_image}" "${base_image_id}"
printf 'runtime_foundation=%s image=%s\n' \
  "${runtime_foundation}" "${runtime_foundation_image}"
printf 'vllm=%s tree=%s base=%s prs=%s\n' \
  "${VLLM_COMMIT}" "${VLLM_INTEGRATION_TREE}" "${VLLM_UPSTREAM_BASE}" "${VLLM_PRS}"
printf 'b12x=%s tree=%s base=%s prs=%s\n' \
  "${B12X_COMMIT}" "${B12X_INTEGRATION_TREE}" "${B12X_UPSTREAM_BASE}" "${B12X_PRS}"
printf 'lmcache=%s tree=%s\n' "${LMCACHE_COMMIT}" "${LMCACHE_INTEGRATION_TREE}"
printf 'core_image=%s\nimage=%s\n' "${core_image}" "${image}"
printf 'flashinfer_wheels=%s\n' "${flashinfer_wheel_image}"
printf 'flashinfer_wheels_id=%s\n' "${flashinfer_wheel_image_id}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "RUNTIME_FOUNDATION=${runtime_foundation}" \
  --build-arg "RUNTIME_FOUNDATION_IMAGE=${runtime_foundation_image}" \
  --build-arg "FLASHINFER_WHEEL_IMAGE=${flashinfer_wheel_image}" \
  --build-arg "FLASHINFER_WHEEL_IMAGE_ID=${flashinfer_wheel_image_id}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg 'VLLM_PATCH_FILE=' \
  --build-arg 'VLLM_PATCH_SHA256=' \
  --build-arg "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg "VLLM_PRS=${VLLM_PRS}" \
  --build-arg "VLLM_UPSTREAM_BASE=${VLLM_UPSTREAM_BASE}" \
  --build-arg "VLLM_MERGE_HEADS=${VLLM_MERGE_HEADS}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg 'B12X_PATCH_FILE=' \
  --build-arg 'B12X_PATCH_SHA256=' \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "B12X_UPSTREAM_BASE=${B12X_UPSTREAM_BASE}" \
  --build-arg "B12X_MERGE_HEADS=${B12X_MERGE_HEADS}" \
  --build-arg "LMCACHE_REPO=${LMCACHE_REPO}" \
  --build-arg "LMCACHE_REF=${LMCACHE_REF}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg 'LMCACHE_PATCH_FILE=' \
  --build-arg 'LMCACHE_PATCH_SHA256=' \
  --build-arg "LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_INTEGRATION_LOCK_SHA256=${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  --build-arg "LMCACHE_PRS=${LMCACHE_PRS}" \
  --build-arg "LMCACHE_UPSTREAM_BASE=${LMCACHE_UPSTREAM_BASE}" \
  --build-arg "LMCACHE_MERGE_HEADS=${LMCACHE_MERGE_HEADS}" \
  --build-arg 'LMCACHE_BUILD_VERSION=0.5.2+glm52dcp.5' \
  --build-arg 'INSTANTTENSOR_REPO=https://github.com/voipmonitor/InstantTensor.git' \
  --build-arg 'INSTANTTENSOR_COMMIT=49b4010afc1cae0441e71fe0b0bffc24fa05e932' \
  --build-arg 'INSTANTTENSOR_LIBAIO_REPO=https://github.com/sailfishos-mirror/libaio.git' \
  --build-arg 'INSTANTTENSOR_LIBAIO_COMMIT=1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448' \
  --build-arg 'INSTANTTENSOR_LIBAIO_TREE=c9442e111b747e9329ea782c6edb9d13a827cc08' \
  --build-arg "CUTLASS_DSL_VERSION=${cutlass_dsl_version}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg "FLASHINFER_VERSION=${flashinfer_version}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --build-arg "RELEASE_NAME=${release_name}-core" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.deepseek-infernal-invocation-cu133-torch213 \
  --tag "${core_image}" \
  .

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${core_image}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "SOURCE_LOCK_SHA256=${source_lock_sha256}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "VLLM_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg "LMCACHE_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --file Dockerfile.kimi-k3-production-clean-runtime \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
jq -e --arg expected "${source_lock_sha256}" \
  '."local-inference.runtime.source-lock.sha256" == $expected and
   ."local-inference.runtime.host-kv-default" == "native" and
   ."local-inference.runtime.source-mode" == "compiled-installed-package" and
   ."local-inference.runtime.source-overlay" == "absent"' \
  <<<"${labels}" >/dev/null

docker run --rm --entrypoint /opt/venv/bin/python "${image}" -c \
  'import pathlib, vllm, b12x, lmcache; root=pathlib.Path("/opt/venv/lib/python3.12/site-packages"); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); assert pathlib.Path(b12x.__file__).resolve().is_relative_to(root); assert pathlib.Path(lmcache.__file__).resolve().is_relative_to(root)'

docker run --rm \
  --env "EXPECTED_VLLM_COMMIT=${VLLM_COMMIT}" \
  --env "EXPECTED_VLLM_TREE=${VLLM_INTEGRATION_TREE}" \
  --env "EXPECTED_B12X_COMMIT=${B12X_COMMIT}" \
  --env "EXPECTED_B12X_TREE=${B12X_INTEGRATION_TREE}" \
  --env "EXPECTED_LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --env "EXPECTED_LMCACHE_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --entrypoint /bin/bash "${image}" -lc '
    set -euo pipefail
    test ! -e /opt/kimi-k3-qsrt
    test "$(git -C /opt/infernal-invocation/vllm rev-parse HEAD)" = "${EXPECTED_VLLM_COMMIT}"
    test "$(git -C /opt/infernal-invocation/vllm rev-parse "HEAD^{tree}")" = "${EXPECTED_VLLM_TREE}"
    test "$(git -C /opt/infernal-invocation/b12x rev-parse HEAD)" = "${EXPECTED_B12X_COMMIT}"
    test "$(git -C /opt/infernal-invocation/b12x rev-parse "HEAD^{tree}")" = "${EXPECTED_B12X_TREE}"
    test "$(git -C /opt/infernal-invocation/lmcache rev-parse HEAD)" = "${EXPECTED_LMCACHE_COMMIT}"
    test "$(git -C /opt/infernal-invocation/lmcache rev-parse "HEAD^{tree}")" = "${EXPECTED_LMCACHE_TREE}"
    test -f /opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so
    test -f /opt/venv/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
    for launcher in \
      /usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii \
      /usr/local/bin/serve-kimi-k3-full-mxfp4-dspark-ii \
      /usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii \
      /usr/local/bin/lmcache-mp-wrapper.sh \
      /usr/local/bin/serve-kimi-k3-production-dspark-ii; do
      bash -n "${launcher}"
    done
    ! grep -Fq "export MAX_IMAGES_PER_PROMPT=" \
      /usr/local/bin/serve-kimi-k3-production-dspark-ii
    grep -Fq "if [[ -n \"\${MAX_IMAGES_PER_PROMPT:-}\" ]]" \
      /usr/local/bin/serve-kimi-k3-full-mxfp4-dspark-ii
  '

if [[ "${RUN_NCCL_SMOKE:-0}" == 1 ]]; then
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
