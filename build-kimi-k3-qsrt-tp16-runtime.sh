#!/usr/bin/env bash
# Build the source-locked Kimi-K3 official MXFP4 and QSRT-K2 runtime image.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

release_root=patches/releases/kimi-k3-qsrt-tp8-safety
vllm_lock="${release_root}/vllm/integration.lock.json"
b12x_lock="${release_root}/b12x/integration.lock.json"
lmcache_lock="${release_root}/lmcache/integration.lock.json"
base_image="${BASE_IMAGE:-voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971}"
release_name="${RELEASE_NAME:-kimi-k3-production-dspark-lmcache}"
release_date="${RELEASE_DATE:-20260819}"
revision="${REVISION:-qsrt-tp8-safety}"

for path in \
  "${vllm_lock}" \
  "${release_root}/vllm/integration.patch" \
  "${b12x_lock}" \
  "${release_root}/b12x/integration.patch" \
  "${lmcache_lock}" \
  "${release_root}/lmcache/integration.patch"; do
  [[ -f "${path}" ]] || {
    printf 'Release composition artifact is missing: %s\n' "${path}" >&2
    exit 1
  }
done

read_lock() {
  local lock=$1 prefix=$2 component=$3
  local patch="${release_root}/${component}/integration.patch"
  local expected_patch_sha repo ref commit integration_tree lock_sha prs
  expected_patch_sha="$(jq -er '.result.patch_sha256' "${lock}")"
  echo "${expected_patch_sha}  ${patch}" | sha256sum -c - >/dev/null
  repo="$(jq -er '.base.repository' "${lock}")"
  ref="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${lock}")"
  commit="$(jq -er '.base.commit' "${lock}")"
  integration_tree="$(jq -er '.result.tree' "${lock}")"
  lock_sha="$(sha256sum "${lock}" | cut -d' ' -f1)"
  prs="$(jq -er '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
  export "${prefix}_REPO=${repo}"
  export "${prefix}_REF=${ref}"
  export "${prefix}_COMMIT=${commit}"
  export "${prefix}_PATCH_FILE=${patch}"
  export "${prefix}_PATCH_SHA256=${expected_patch_sha}"
  export "${prefix}_INTEGRATION_TREE=${integration_tree}"
  export "${prefix}_INTEGRATION_LOCK_SHA256=${lock_sha}"
  export "${prefix}_PRS=${prs}"
}

read_lock "${vllm_lock}" VLLM vllm
read_lock "${b12x_lock}" B12X b12x
read_lock "${lmcache_lock}" LMCACHE lmcache

source_overlay_sha256="$(sha256sum runtime/kimi-k3-qsrt/source-overlay/sitecustomize.py | cut -d' ' -f1)"
cache_fingerprint="cu133-torch213-kimi-k3-vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-production-dspark-lmcache-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-cu133-torch213-${release_date}-${revision}}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'The image recipe must be committed before build.\n' >&2
  git status --short >&2
  exit 1
fi

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi
base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"

printf 'image=%s\n' "${image}"
printf 'base=%s id=%s\n' "${base_image}" "${base_image_id}"
printf 'vllm_base=%s tree=%s prs=%s\n' \
  "${VLLM_COMMIT}" "${VLLM_INTEGRATION_TREE}" "${VLLM_PRS}"
printf 'b12x_base=%s tree=%s prs=%s\n' \
  "${B12X_COMMIT}" "${B12X_INTEGRATION_TREE}" "${B12X_PRS}"
printf 'lmcache_base=%s tree=%s prs=%s\n' \
  "${LMCACHE_COMMIT}" "${LMCACHE_INTEGRATION_TREE}" "${LMCACHE_PRS}"

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
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_PATCH_FILE=${B12X_PATCH_FILE}" \
  --build-arg "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "LMCACHE_REPO=${LMCACHE_REPO}" \
  --build-arg "LMCACHE_REF=${LMCACHE_REF}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg "LMCACHE_PATCH_FILE=${LMCACHE_PATCH_FILE}" \
  --build-arg "LMCACHE_PATCH_SHA256=${LMCACHE_PATCH_SHA256}" \
  --build-arg "LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_INTEGRATION_LOCK_SHA256=${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  --build-arg "LMCACHE_PRS=${LMCACHE_PRS}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --build-arg "SOURCE_OVERLAY_SHA256=${source_overlay_sha256}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "RELEASE_ARTIFACT_DIR=${release_root}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.kimi-k3-qsrt-tp16-runtime \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
assert_label() {
  local key=$1 expected=$2
  jq -e --arg key "${key}" --arg expected "${expected}" \
    '.[$key] == $expected' <<<"${labels}" >/dev/null || {
    printf 'Image label %s does not match %s\n' "${key}" "${expected}" >&2
    exit 1
  }
}

assert_label local-inference.docker.commit "${docker_commit}"
assert_label local-inference.runtime.base-id "${base_image_id}"
assert_label local-inference.vllm.integration.tree "${VLLM_INTEGRATION_TREE}"
assert_label local-inference.b12x.integration.tree "${B12X_INTEGRATION_TREE}"
assert_label local-inference.lmcache.integration.tree "${LMCACHE_INTEGRATION_TREE}"
assert_label local-inference.flash-attention.forward.sha256 \
  f8dfc8321baef79d8ad4ce5f8e18365e215f567da631638498b26330a5aca449

docker run --rm \
  --entrypoint /opt/venv/bin/python "${image}" -c \
  'import pathlib, sys; assert "vllm" not in sys.modules; import vllm; expected=pathlib.Path("/opt/kimi-k3-qsrt/vllm"); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(expected)'
docker run --rm \
  --entrypoint /opt/venv/bin/python "${image}" -c \
  'import lmcache, pathlib; expected=pathlib.Path("/opt/kimi-k3-qsrt/lmcache"); assert pathlib.Path(lmcache.__file__).resolve().is_relative_to(expected)'
docker run --rm --entrypoint /bin/bash "${image}" -lc \
  'sha256sum /opt/infernal-invocation/vllm/vllm/vllm_flash_attn/cute/flash_fwd.py'

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
