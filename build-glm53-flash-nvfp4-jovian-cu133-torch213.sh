#!/usr/bin/env bash
set -euo pipefail

# Build a reproducible GLM-5.3-Flash NVFP4 runtime with ModelOpt MXFP8
# DFlash2 support from pinned branch bases and pull-request heads.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

base_image_overridden=0
vllm_package_version_overridden=0
image_overridden=0
[[ -v BASE_IMAGE ]] && base_image_overridden=1
[[ -v VLLM_PACKAGE_VERSION ]] && vllm_package_version_overridden=1
[[ -v IMAGE ]] && image_overridden=1

release_date=${RELEASE_DATE:-20260828}
revision=${REVISION:-r1}
composition_root=patches/releases/glm53-flash-nvfp4-jovian
base_image=${BASE_IMAGE:-voipmonitor/vllm@sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2}
allow_dirty_build=${ALLOW_DIRTY_BUILD:-0}
push_image=${PUSH_IMAGE:-0}

case "${revision}" in
  r[1-9]|r[1-9][0-9]*) ;;
  *) printf 'REVISION must use the rN form; got %s\n' "${revision}" >&2; exit 2 ;;
esac
[[ "${release_date}" =~ ^[0-9]{8}$ ]] || {
  printf 'RELEASE_DATE must use YYYYMMDD; got %s\n' "${release_date}" >&2
  exit 2
}
case "${allow_dirty_build}" in
  0 | 1) ;;
  *) printf 'ALLOW_DIRTY_BUILD must be 0 or 1; got %s\n' "${allow_dirty_build}" >&2; exit 2 ;;
esac
case "${push_image}" in
  0 | 1) ;;
  *) printf 'PUSH_IMAGE must be 0 or 1; got %s\n' "${push_image}" >&2; exit 2 ;;
esac
if ((allow_dirty_build == 1 && push_image == 1)); then
  printf 'PUSH_IMAGE=1 is forbidden when ALLOW_DIRTY_BUILD=1.\n' >&2
  exit 2
fi

read_lock() {
  local component=$1 prefix=$2
  local lock="${composition_root}/${component}/integration.lock.json"
  local patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}" || { printf 'Missing composition lock: %s\n' "${lock}" >&2; exit 1; }
  test -f "${patch}" || { printf 'Missing integration patch: %s\n' "${patch}" >&2; exit 1; }
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" | sha256sum -c - >/dev/null

  export "${prefix}_REPO=$(jq -er '.base.repository' "${lock}")"
  export "${prefix}_REF=$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${lock}")"
  export "${prefix}_COMMIT=$(jq -er '.base.commit' "${lock}")"
  export "${prefix}_PATCH_FILE=${composition_root#patches/releases/}/${component}/integration.patch"
  export "${prefix}_PATCH_SHA256=$(jq -er '.result.patch_sha256' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.result.tree' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
  export "${prefix}_PRS=$(jq -er '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
  export "${prefix}_MERGE_HEADS=$(jq -er '[.pull_requests[].head] | join(",")' "${lock}")"
}

read_lock vllm VLLM
read_lock b12x B12X
source_lock_sha256="$({
  printf '%s\n' "${VLLM_INTEGRATION_LOCK_SHA256}"
  printf '%s\n' "${B12X_INTEGRATION_LOCK_SHA256}"
} | sha256sum | cut -d' ' -f1)"

test "${VLLM_REF}" = dev/jovian-judgement
test "${B12X_REF}" = master
test "${VLLM_PRS}" = "491@b77333ca8824897ff6ddf96a62208ea406c555a9"
test "${B12X_PRS}" = "250@dd8cf60505e0363ab9d6ef6b2116c3a37216a2f1"
test "$(jq -er '.source_patches | length' "${composition_root}/vllm/integration.lock.json")" = 0
test "$(jq -er '.source_patches | length' "${composition_root}/b12x/integration.lock.json")" = 0

docker_commit="$(git rev-parse HEAD)"
worktree_dirty=0
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  worktree_dirty=1
fi
if ((worktree_dirty == 1 && allow_dirty_build == 0)); then
  printf 'Commit the runtime recipe or set ALLOW_DIRTY_BUILD=1.\n' >&2
  git status --short >&2
  exit 1
fi

vllm_package_version=${VLLM_PACKAGE_VERSION:-0.26.1rc0+glm53.flash.dflash2.mxfp8.${revision}.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}}
cache_fingerprint="cu133-torch213-glm53-dflash2-mxfp8-vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}"
qualified_image="voipmonitor/vllm:glm53-flash-dflash2-mxfp8-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-fi1ac6942-cu133-torch213-${release_date}-${revision}-git${docker_commit:0:12}"
image=${IMAGE:-${qualified_image}}
release_status=qualified

override_reasons=()
((base_image_overridden == 1)) && override_reasons+=(BASE_IMAGE)
((vllm_package_version_overridden == 1)) && override_reasons+=(VLLM_PACKAGE_VERSION)
((allow_dirty_build == 1)) && override_reasons+=(ALLOW_DIRTY_BUILD)
if ((${#override_reasons[@]} > 0)); then
  if ((image_overridden == 0)); then
    printf 'Build overrides (%s) require an explicit non-release IMAGE tag.\n' \
      "$(IFS=,; printf '%s' "${override_reasons[*]}")" >&2
    exit 2
  fi
  image_tag=${image##*:}
  case "${image_tag}" in
    dev-* | test-* | scratch-* | *-dev-* | *-test-* | *-scratch-*) ;;
    *)
      printf 'Override IMAGE tags must contain a dev, test, or scratch marker; got %s.\n' \
        "${image}" >&2
      exit 2
      ;;
  esac
  release_status=research-only
fi

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'base=%s\nimage=%s\nstatus=%s\ndocker_commit=%s\n' \
    "${base_image}" "${image}" "${release_status}" "${docker_commit}"
  printf 'vllm_ref=%s\nvllm_commit=%s\nvllm_prs=%s\nvllm_tree=%s\n' \
    "${VLLM_REF}" "${VLLM_COMMIT}" "${VLLM_PRS}" "${VLLM_INTEGRATION_TREE}"
  printf 'b12x_ref=%s\nb12x_commit=%s\nb12x_prs=%s\nb12x_tree=%s\n' \
    "${B12X_REF}" "${B12X_COMMIT}" "${B12X_PRS}" "${B12X_INTEGRATION_TREE}"
  printf 'vllm_package_version=%s\n' "${vllm_package_version}"
  printf 'torch=2.13.0\ncuda=13.3\nnccl=2.31.2\nflashinfer=0.6.18+cu133\n'
  exit 0
fi

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi
base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
base_labels="$(docker image inspect "${base_image}" --format '{{json .Config.Labels}}')"
jq -e '
  ."local-inference.runtime.foundation.source-packages" == "absent" and
  ."local-inference.cuda.version" == "13.3" and
  ."local-inference.torch.version" == "2.13.0" and
  ."local-inference.flashinfer.version" == "0.6.18+cu133" and
  ."local-inference.cutlass-dsl.version" == "4.6.2" and
  ."local-inference.instanttensor.version" == "0.1.9"
' <<<"${base_labels}" >/dev/null

printf 'base=%s (%s)\nimage=%s\n' "${base_image}" "${base_image_id}" "${image}"
printf 'vllm=%s + %s -> %s\nb12x=%s + %s -> %s\n' \
  "${VLLM_COMMIT}" "${VLLM_PRS}" "${VLLM_INTEGRATION_TREE}" \
  "${B12X_COMMIT}" "${B12X_PRS}" "${B12X_INTEGRATION_TREE}"

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
  --build-arg "VLLM_MERGE_HEADS=${VLLM_MERGE_HEADS}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_PATCH_FILE=${B12X_PATCH_FILE}" \
  --build-arg "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "B12X_MERGE_HEADS=${B12X_MERGE_HEADS}" \
  --build-arg "SOURCE_LOCK_SHA256=${source_lock_sha256}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "RELEASE_STATUS=${release_status}" \
  --file Dockerfile.glm53-flash-nvfp4-jovian-cu133-torch213 \
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

assert_label local-inference.runtime.base-id "${base_image_id}"
assert_label local-inference.vllm.commit "${VLLM_COMMIT}"
assert_label local-inference.vllm.integration.tree "${VLLM_INTEGRATION_TREE}"
assert_label local-inference.b12x.commit "${B12X_COMMIT}"
assert_label local-inference.b12x.integration.tree "${B12X_INTEGRATION_TREE}"
assert_label local-inference.runtime.source-lock.sha256 "${source_lock_sha256}"
assert_label local-inference.vllm.integration.prs "${VLLM_PRS}"
assert_label local-inference.vllm.merge-heads "${VLLM_MERGE_HEADS}"
assert_label local-inference.b12x.integration.prs "${B12X_PRS}"
assert_label local-inference.b12x.merge-heads "${B12X_MERGE_HEADS}"
assert_label local-inference.status "${release_status}"
assert_label local-inference.backend.indexer.decode "B12X_C4"
assert_label local-inference.backend.indexer.prefill "DeepGEMM_C4"

docker run --rm --entrypoint /opt/venv/bin/python "${image}" \
  /opt/local-inference/verify_glm53_flash_nvfp4_runtime.py \
  --vllm-version "${vllm_package_version}" \
  --vllm-tree "${VLLM_INTEGRATION_TREE}" \
  --b12x-tree "${B12X_INTEGRATION_TREE}"

launcher_output="$(docker run --rm -e DRY_RUN=1 "${image}")"
grep -Fq -- '--revision 520de24eabf507659eaef7c70f14fd584527facc' <<<"${launcher_output}"
grep -Fq -- '--tensor-parallel-size 4' <<<"${launcher_output}"
grep -Fq -- '--load-format instanttensor' <<<"${launcher_output}"
grep -Fq -- '--max-num-seqs 16' <<<"${launcher_output}"
grep -Fq -- '--max-model-len 262144' <<<"${launcher_output}"
grep -Fq -- '--max-num-batched-tokens 4096' <<<"${launcher_output}"
grep -Fq -- '--attention-backend B12X' <<<"${launcher_output}"
grep -Fq -- '--moe-backend b12x' <<<"${launcher_output}"
grep -Fq -- '--linear-backend b12x' <<<"${launcher_output}"
grep -Fq -- '--mamba-cache-mode align' <<<"${launcher_output}"
grep -Fq -- '--quantization modelopt_mixed' <<<"${launcher_output}"
grep -Fq -- '--block-size 256' <<<"${launcher_output}"
grep -Fq -- '--enable-prefix-caching' <<<"${launcher_output}"
grep -Fq -- '--compilation-config' <<<"${launcher_output}"
grep -Fq -- 'cudagraph_mode\":\"FULL' <<<"${launcher_output}"
if grep -Fq -- '--disable-custom-all-reduce' <<<"${launcher_output}"; then
  printf 'The default launcher must retain B12X PCIe all-reduce.\n' >&2
  exit 1
fi

mtp_launcher_output="$(docker run --rm -e DRY_RUN=1 -e SPECULATOR=mtp -e MTP=3 "${image}")"
grep -Fq -- 'num_speculative_tokens\":3' <<<"${mtp_launcher_output}"
grep -Fq -- 'moe_backend\":\"humming' <<<"${mtp_launcher_output}"
grep -Fq -- 'attention_backend\":\"B12X' <<<"${mtp_launcher_output}"

dflash_launcher_output="$(docker run --rm -e DRY_RUN=1 -e SPECULATOR=dflash "${image}")"
grep -Fq -- 'method\":\"dflash' <<<"${dflash_launcher_output}"
grep -Fq -- 'model\":\"local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8' <<<"${dflash_launcher_output}"
grep -Fq -- 'revision\":\"b6d33aa93fc1ac5b23a88251a1c0ce0bfe2ad17c' <<<"${dflash_launcher_output}"
grep -Fq -- 'num_speculative_tokens\":7' <<<"${dflash_launcher_output}"
grep -Fq -- 'kv_cache_dtype\":\"auto' <<<"${dflash_launcher_output}"
printf '%s\n' "${launcher_output}"

if ((push_image == 1)); then docker push "${image}"; fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
