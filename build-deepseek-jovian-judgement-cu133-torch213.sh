#!/usr/bin/env bash
set -euo pipefail

# Build the CUDA 13.3 DeepSeek-V4-Flash text and vision runtime from immutable
# Jovian Judgement, B12X, and LMCache source-composition locks.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

release_name=${RELEASE_NAME:-jovian-judgement-deepseek-v4-flash-cu133-torch213}
release_date=${RELEASE_DATE:-20260904}
revision=${REVISION:-r4}
composition_root=${COMPOSITION_ROOT:-patches/releases/jovian-judgement-ds4-r4}
base_image=${BASE_IMAGE:-voipmonitor/vllm@sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2}
runtime_foundation=${RUNTIME_FOUNDATION:-1}
runtime_foundation_image=${RUNTIME_FOUNDATION_IMAGE:-${base_image}}
flashinfer_wheel_image=${FLASHINFER_WHEEL_IMAGE:-voipmonitor/vllm:flashinfer-wheels-fi803c466-cu133-torch213-20260904-r1@sha256:79edbc91874d9468e3e6268e1584503e3dec55f2a4d3bdd70d5c43e9b41675c7}
flashinfer_repo=${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}
flashinfer_ref=${FLASHINFER_REF:-port/jj-ds4-vision-20260903}
flashinfer_commit=${FLASHINFER_COMMIT:-803c4664f4771ddc418f20a57f752469a237a825}
instanttensor_repo=${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}
instanttensor_commit=${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}
instanttensor_libaio_repo=${INSTANTTENSOR_LIBAIO_REPO:-https://github.com/sailfishos-mirror/libaio.git}
instanttensor_libaio_commit=${INSTANTTENSOR_LIBAIO_COMMIT:-1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448}
instanttensor_libaio_tree=${INSTANTTENSOR_LIBAIO_TREE:-c9442e111b747e9329ea782c6edb9d13a827cc08}
exllamav3_repo=${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}
exllamav3_commit=${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}

case "${revision}" in
  r[1-9]|r[1-9][0-9]*) ;;
  *) printf 'REVISION must use the rN form; got %s\n' "${revision}" >&2; exit 2 ;;
esac
[[ "${release_date}" =~ ^[0-9]{8}$ ]] || {
  printf 'RELEASE_DATE must use YYYYMMDD; got %s\n' "${release_date}" >&2
  exit 2
}

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
  export "${prefix}_PATCH_FILE=${composition_root#patches/}/${component}/integration.patch"
  export "${prefix}_PATCH_SHA256=$(jq -er '.result.patch_sha256' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.result.tree' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
  export "${prefix}_PRS=$(jq -er '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
  export "${prefix}_UPSTREAM_BASE=$(jq -r '.upstream_base.commit // .composition.upstream_base // ""' "${lock}")"
  export "${prefix}_MERGE_HEADS=$(jq -r '(.merge_heads // .composition.merge_heads // []) | join(",")' "${lock}")"
}

read_lock vllm VLLM
read_lock b12x B12X
read_lock lmcache LMCACHE

test "${VLLM_REF}" = dev/jovian-judgement
test "${B12X_REF}" = master

vllm_package_version=${VLLM_PACKAGE_VERSION:-0.26.1rc0+jovian.judgement.cu133.${revision}.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}}
flashinfer_version=${FLASHINFER_VERSION:-0.6.18+cu133}
lmcache_build_version=${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.5}
cache_fingerprint="cu133-torch213-vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}-lmcache${LMCACHE_INTEGRATION_TREE:0:10}"
image=${IMAGE:-voipmonitor/vllm:jovian-judgement-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-fi${flashinfer_commit:0:7}-cu133-torch213-${release_date}-${revision}}

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'release=%s\nrevision=%s\nbase=%s\nimage=%s\n' \
    "${release_name}" "${revision}" "${base_image}" "${image}"
  printf 'runtime_foundation=%s\nruntime_foundation_image=%s\n' \
    "${runtime_foundation}" "${runtime_foundation_image}"
  printf 'vllm_ref=%s\nvllm_commit=%s\nvllm_tree=%s\nvllm_patch=%s\n' \
    "${VLLM_REF}" "${VLLM_COMMIT}" "${VLLM_INTEGRATION_TREE}" "${VLLM_PATCH_FILE}"
  printf 'b12x_ref=%s\nb12x_commit=%s\nb12x_tree=%s\nb12x_patch=%s\n' \
    "${B12X_REF}" "${B12X_COMMIT}" "${B12X_INTEGRATION_TREE}" "${B12X_PATCH_FILE}"
  printf 'lmcache_ref=%s\nlmcache_commit=%s\nlmcache_tree=%s\nlmcache_patch=%s\n' \
    "${LMCACHE_REF}" "${LMCACHE_COMMIT}" "${LMCACHE_INTEGRATION_TREE}" "${LMCACHE_PATCH_FILE}"
  printf 'torch=2.13.0\ncuda=13.3\nnccl=2.31.2\nflashinfer=%s\n' "${flashinfer_version}"
  exit 0
fi

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}" || {
    printf 'The pinned CUDA 13.3 runtime foundation is unavailable. Build it with ./build-kimi-k3-runtime-foundation.sh.\n' >&2
    exit 1
  }
fi
base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"

case "${runtime_foundation}" in
  0|1) ;;
  *) printf 'RUNTIME_FOUNDATION must be 0 or 1; got %s\n' "${runtime_foundation}" >&2; exit 2 ;;
esac

if [[ "${runtime_foundation}" == 1 ]]; then
  base_labels="$(docker image inspect "${base_image}" --format '{{json .Config.Labels}}')"
  jq -e '
    ."local-inference.runtime.foundation.source-packages" == "absent" and
    ."local-inference.cuda.version" == "13.3" and
    ."local-inference.torch.version" == "2.13.0" and
    ."local-inference.flashinfer.version" == "0.6.18+cu133" and
    ."local-inference.cutlass-dsl.version" == "4.6.2" and
    ."local-inference.instanttensor.version" == "0.1.9"
  ' <<<"${base_labels}" >/dev/null
fi

if ! docker image inspect "${flashinfer_wheel_image}" >/dev/null 2>&1; then
  docker pull "${flashinfer_wheel_image}" || \
    BASE_IMAGE="${base_image}" IMAGE="${flashinfer_wheel_image}" \
      FLASHINFER_REPO="${flashinfer_repo}" \
      FLASHINFER_REF="${flashinfer_ref}" \
      FLASHINFER_COMMIT="${flashinfer_commit}" \
      FLASHINFER_VERSION="${flashinfer_version}" \
      ./build-flashinfer-cu133-torch213-wheels.sh
fi

flashinfer_labels="$(docker image inspect "${flashinfer_wheel_image}" \
  --format '{{json .Config.Labels}}')"
jq -e \
  --arg repo "${flashinfer_repo}" \
  --arg ref "${flashinfer_ref}" \
  --arg commit "${flashinfer_commit}" \
  --arg version "${flashinfer_version}" \
  '."local-inference.flashinfer.repo" == $repo and
   ."local-inference.flashinfer.ref" == $ref and
   ."local-inference.flashinfer.commit" == $commit and
   ."local-inference.flashinfer.version" == $version' \
  <<<"${flashinfer_labels}" >/dev/null
if [[ "${runtime_foundation}" == 0 ]]; then
  jq -e --arg base_id "${base_image_id}" \
    '."local-inference.runtime.base-id" == $base_id' \
    <<<"${flashinfer_labels}" >/dev/null
fi
flashinfer_wheel_image_id="$(docker image inspect "${flashinfer_wheel_image}" \
  --format '{{.Id}}')"

docker_commit="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'Set ALLOW_DIRTY_BUILD=1 for an image whose recipe is not committed.\n' >&2
  git status --short >&2
  exit 1
fi

printf 'release=%s\nbase=%s (%s)\nimage=%s\n' \
  "${release_name}" "${base_image}" "${base_image_id}" "${image}"
printf 'vllm=%s + %s -> %s\n' "${VLLM_COMMIT}" "${VLLM_PRS}" "${VLLM_INTEGRATION_TREE}"
printf 'b12x=%s + %s -> %s\n' "${B12X_COMMIT}" "${B12X_PRS}" "${B12X_INTEGRATION_TREE}"
printf 'lmcache=%s + %s -> %s\n' "${LMCACHE_COMMIT}" "${LMCACHE_PRS}" "${LMCACHE_INTEGRATION_TREE}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "FLASHINFER_WHEEL_IMAGE=${flashinfer_wheel_image}" \
  --build-arg "FLASHINFER_WHEEL_IMAGE_ID=${flashinfer_wheel_image_id}" \
  --build-arg "FLASHINFER_REPO=${flashinfer_repo}" \
  --build-arg "FLASHINFER_REF=${flashinfer_ref}" \
  --build-arg "FLASHINFER_COMMIT=${flashinfer_commit}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "VLLM_PATCH_FILE=${VLLM_PATCH_FILE}" \
  --build-arg "VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}" \
  --build-arg "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg "VLLM_PRS=${VLLM_PRS}" \
  --build-arg "VLLM_UPSTREAM_BASE=${VLLM_UPSTREAM_BASE}" \
  --build-arg "VLLM_MERGE_HEADS=${VLLM_MERGE_HEADS}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_PATCH_FILE=${B12X_PATCH_FILE}" \
  --build-arg "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "B12X_UPSTREAM_BASE=${B12X_UPSTREAM_BASE}" \
  --build-arg "B12X_MERGE_HEADS=${B12X_MERGE_HEADS}" \
  --build-arg "LMCACHE_REPO=${LMCACHE_REPO}" \
  --build-arg "LMCACHE_REF=${LMCACHE_REF}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg "LMCACHE_PATCH_FILE=${LMCACHE_PATCH_FILE}" \
  --build-arg "LMCACHE_PATCH_SHA256=${LMCACHE_PATCH_SHA256}" \
  --build-arg "LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_INTEGRATION_LOCK_SHA256=${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  --build-arg "LMCACHE_PRS=${LMCACHE_PRS}" \
  --build-arg "LMCACHE_UPSTREAM_BASE=${LMCACHE_UPSTREAM_BASE}" \
  --build-arg "LMCACHE_MERGE_HEADS=${LMCACHE_MERGE_HEADS}" \
  --build-arg "LMCACHE_BUILD_VERSION=${lmcache_build_version}" \
  --build-arg "INSTANTTENSOR_REPO=${instanttensor_repo}" \
  --build-arg "INSTANTTENSOR_COMMIT=${instanttensor_commit}" \
  --build-arg "INSTANTTENSOR_LIBAIO_REPO=${instanttensor_libaio_repo}" \
  --build-arg "INSTANTTENSOR_LIBAIO_COMMIT=${instanttensor_libaio_commit}" \
  --build-arg "INSTANTTENSOR_LIBAIO_TREE=${instanttensor_libaio_tree}" \
  --build-arg "EXLLAMAV3_REPO=${exllamav3_repo}" \
  --build-arg "EXLLAMAV3_COMMIT=${exllamav3_commit}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg "FLASHINFER_VERSION=${flashinfer_version}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --build-arg "RUNTIME_FOUNDATION=${runtime_foundation}" \
  --build-arg "RUNTIME_FOUNDATION_IMAGE=${runtime_foundation_image}" \
  --file Dockerfile.deepseek-infernal-invocation-cu133-torch213 \
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
assert_label local-inference.runtime.foundation.enabled "${runtime_foundation}"
assert_label local-inference.runtime.foundation.image "${runtime_foundation_image}"
assert_label local-inference.runtime.host-kv-default off
assert_label local-inference.flashinfer.repo "${flashinfer_repo}"
assert_label local-inference.flashinfer.commit "${flashinfer_commit}"
assert_label local-inference.vllm.integration.tree "${VLLM_INTEGRATION_TREE}"
assert_label local-inference.b12x.integration.tree "${B12X_INTEGRATION_TREE}"
assert_label local-inference.lmcache.integration.tree "${LMCACHE_INTEGRATION_TREE}"
assert_label local-inference.instanttensor.commit "${instanttensor_commit}"
assert_label local-inference.instanttensor.libaio.repo "${instanttensor_libaio_repo}"
assert_label local-inference.instanttensor.libaio.commit "${instanttensor_libaio_commit}"
assert_label local-inference.instanttensor.libaio.tree "${instanttensor_libaio_tree}"
assert_label local-inference.nccl.version 2.31.2
assert_label local-inference.exllamav3.repo "${exllamav3_repo}"
assert_label local-inference.exllamav3.commit "${exllamav3_commit}"

docker run --rm --entrypoint /opt/venv/bin/python "${image}" \
  /opt/local-inference/verify_deepseek_infernal_cu133_runtime.py \
  --vllm-version "${vllm_package_version}" \
  --flashinfer-version "${flashinfer_version}" \
  --lmcache-version "${lmcache_build_version}" \
  --instanttensor-version 0.1.9

launcher_output="$(
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODE=dspark -e DSPARK_TOKENS=5 -e MAX_NUM_SEQS=16 \
    -e TP_SIZE=2 -e GRAPH=auto -e LOAD_FORMAT=instanttensor "${image}" 2>&1
)"
grep -Fq 'DS4 launch: variant=text mode=dspark depth=fixed' <<<"${launcher_output}"
grep -Fq 'backend=b12x-a8' <<<"${launcher_output}"
grep -Fq 'tp=2 dcp=1 max_seqs=16 graph=96' <<<"${launcher_output}"
grep -Fq -- '--attention-backend B12X' <<<"${launcher_output}"
printf '%s\n' "${launcher_output}"

vision_launcher_output="$(
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODE=dspark -e DS4_MODEL_VARIANT=vision \
    -e MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
    -e MAX_NUM_SEQS=4 -e MAX_NUM_BATCHED_TOKENS=4096 \
    -e TP_SIZE=2 -e GRAPH=auto -e LOAD_FORMAT=instanttensor "${image}" 2>&1
)"
grep -Fq 'DS4 launch: variant=vision mode=dspark depth=fixed' \
  <<<"${vision_launcher_output}"
grep -Fq 'tp=2 dcp=1 max_seqs=4 graph=16' <<<"${vision_launcher_output}"
grep -Fq 'num_speculative_tokens\":3' <<<"${vision_launcher_output}"
grep -Fq -- '--revision 6821d6ad3681a4b137b066b76094fa82ebd0a380' \
  <<<"${vision_launcher_output}"
grep -Fq -- '--max-model-len 1048576' <<<"${vision_launcher_output}"
grep -Fq -- '--gpu-memory-utilization 0.975' <<<"${vision_launcher_output}"

vision_lmcache_output="$(
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODE=dspark -e DS4_MODEL_VARIANT=vision \
    -e MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
    -e LMCACHE_MODE=ram -e LMCACHE_TRANSFER_MODE=auto \
    -e MAX_NUM_SEQS=4 -e TP_SIZE=2 -e GRAPH=auto "${image}" 2>&1
)"
grep -Fq -- '--max-model-len 900000' <<<"${vision_lmcache_output}"
grep -Fq -- '--gpu-memory-utilization 0.951' <<<"${vision_lmcache_output}"

text_lmcache_output="$(
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODE=dspark -e DSPARK_TOKENS=5 \
    -e LMCACHE_MODE=disk -e MAX_MODEL_LEN=1048576 \
    -e MAX_NUM_SEQS=8 -e TP_SIZE=2 -e GRAPH=auto "${image}" 2>&1
)"
grep -Fq -- '--gpu-memory-utilization 0.965' <<<"${text_lmcache_output}"
grep -Fq 'direct_lmcache=1 lmcache_memory_profile=qualified' \
  <<<"${text_lmcache_output}"

if text_unsafe_output="$(
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODE=dspark -e DSPARK_TOKENS=5 \
    -e LMCACHE_MODE=disk -e MAX_MODEL_LEN=1048576 \
    -e GPU_MEMORY_UTILIZATION=0.975 \
    -e MAX_NUM_SEQS=8 -e TP_SIZE=2 -e GRAPH=auto "${image}" 2>&1
)"; then
  printf 'Text direct LMCache accepted an unqualified memory profile:\n%s\n' \
    "${text_unsafe_output}" >&2
  exit 1
fi
grep -Fq 'requires GPU_MEMORY_UTILIZATION at or below 0.965' \
  <<<"${text_unsafe_output}"

docker run --rm --entrypoint /opt/venv/bin/python "${image}" -c \
  'import importlib, os, pathlib, torch; ext = importlib.import_module("exllamav3_ext"); assert hasattr(ext, "exl3_gemm"); assert pathlib.Path(os.environ["VLLM_EXL3_ENCODER_SOURCE"], "modules/quant/exl3_lib/quantize.py").is_file()'

if [[ "${RUN_NCCL_SMOKE:-0}" == 1 ]]; then
  smoke_gpus=${NCCL_SMOKE_GPUS:-0,1,2,3}
  smoke_ranks=${NCCL_SMOKE_RANKS:-4}
  docker run --rm --gpus "\"device=${smoke_gpus}\"" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint torchrun "${image}" \
    --standalone --nproc-per-node="${smoke_ranks}" \
    /opt/local-inference/torch_nccl_smoke.py
fi

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then docker push "${image}"; fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
