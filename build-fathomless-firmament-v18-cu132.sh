#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v18-vllma4486e3-b12xaaf9b2c-fi801d57a-cu132-20260717}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/fathomless-firmament-v18-final-20260717}"
export VLLM_COMMIT="${VLLM_COMMIT:-a4486e37f71baeb2f8389a73739fcf23bd0d6d14}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v18.vllma4486e3.b12xaaf9b2c.fi801d57a.cu132.20260717}"

export B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
export B12X_REF="${B12X_REF:-master}"
export B12X_COMMIT="${B12X_COMMIT:-aaf9b2c2cae9ec44fa6031988da184e5d379c89c}"

export VLLM_REQUIRED_LAUNCHERS="serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-hybrid-v17.sh"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

./build-fathomless-firmament-v16-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_COMMIT}" '."local-inference.b12x.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "801d57a08958c13d375ddbb6be3be4808f48a708" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null

docker run --rm -i --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
from vllm import envs
from vllm.v1.attention.backends.mla import b12x_mla_sparse

assert hasattr(envs, "VLLM_DCP_QUERY_SPLIT")
assert hasattr(envs, "VLLM_B12X_MLA_CKV_GATHER")
assert hasattr(b12x_mla_sparse, "_global_causal_lens_for_ckv_gather")
PY

dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
    -e DRY_RUN=1 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" | tee "/tmp/fathomless-firmament-v18-${name}.txt"
}

dry_run tp8-dcp4 -e TP=8 -e DCP=4
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/fathomless-firmament-v18-tp8-dcp4.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/fathomless-firmament-v18-tp8-dcp4.txt
grep -q -- '--load-format instanttensor' /tmp/fathomless-firmament-v18-tp8-dcp4.txt

dry_run tp8-dcp8 -e TP=8 -e DCP=8
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/fathomless-firmament-v18-tp8-dcp8.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/fathomless-firmament-v18-tp8-dcp8.txt

dry_run tp8-dcp2 -e TP=8 -e DCP=2
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' /tmp/fathomless-firmament-v18-tp8-dcp2.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' /tmp/fathomless-firmament-v18-tp8-dcp2.txt

dry_run forced-off \
  -e TP=8 \
  -e DCP=4 \
  -e DCP_QUERY_SPLIT=0 \
  -e DCP_CKV_GATHER=0
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' /tmp/fathomless-firmament-v18-forced-off.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' /tmp/fathomless-firmament-v18-forced-off.txt

docker run --rm --entrypoint /usr/local/bin/serve-fathomless-firmament.sh \
  -e MODEL_FAMILY=glm52 \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=4 \
  "${IMAGE}" | grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$'

docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 \
  -e MODE=mtp \
  -e BACKEND=b12x \
  -e TP_SIZE=2 \
  "${IMAGE}" >/tmp/fathomless-firmament-v18-ds4-dry-run.txt

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
