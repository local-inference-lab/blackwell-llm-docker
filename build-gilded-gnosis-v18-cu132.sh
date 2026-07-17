#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Unified GLM 5.2 and DS4/DSpark image built from dev/gilded-gnosis plus the
# explicitly pinned DSpark, SM120 PCIe, DCP-prefill, and NF3 decode stacks.
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v18-vllmce5dee9-b12x66dff47-fi801d57a-cu132-20260717}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v18-final-20260717}"
export VLLM_COMMIT="${VLLM_COMMIT:-ce5dee91dc90076bf653759ffa54777971a6caab}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v18.vllmce5dee9.b12x66dff47.fi801d57a.cu132.20260717}"

# Temporary source pin for lukealonso/b12x#36. Replace the repository/ref with
# upstream master after the PR is merged; the immutable commit remains audited.
export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/nf3-grid188-decode-20260717}"
export B12X_COMMIT="${B12X_COMMIT:-66dff47de0ffb16934f812951a9fb5e8b3769536}"

export VLLM_REQUIRED_LAUNCHERS="serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

./build-fathomless-firmament-v16-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_COMMIT}" '."local-inference.b12x.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "801d57a08958c13d375ddbb6be3be4808f48a708" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null

docker run --rm -i --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
from vllm import envs
from vllm.config.quantization import resolve_quantization_config
from vllm.v1.attention.backends.mla import b12x_mla_sparse
from b12x.moe.fused.w4a16.kernel import (
    w4a16_hybrid_mapped_grid188_mapping_proof,
)

assert hasattr(envs, "VLLM_DCP_QUERY_SPLIT")
assert hasattr(envs, "VLLM_B12X_MLA_CKV_GATHER")
assert hasattr(envs, "VLLM_NF3_GRID188_DECODE")
assert hasattr(b12x_mla_sparse, "_global_causal_lens_for_ckv_gather")
assert resolve_quantization_config("nvfp4_nf3_hybrid", {"linear": {"weight": "mxfp8"}})
proof = w4a16_hybrid_mapped_grid188_mapping_proof()
assert proof["grid_x"] == 188
assert len(proof["fc1_tasks"]) == 128
assert len(proof["fc2_tasks"]) == 768
PY

dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
    -e DRY_RUN=1 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" | tee "/tmp/gilded-gnosis-v18-${name}.txt"
}

dry_run tp8-dcp4 -e TP=8 -e DCP=4
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/gilded-gnosis-v18-tp8-dcp4.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/gilded-gnosis-v18-tp8-dcp4.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v18-tp8-dcp4.txt

dry_run tp8-dcp8 -e TP=8 -e DCP=8
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/gilded-gnosis-v18-tp8-dcp8.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/gilded-gnosis-v18-tp8-dcp8.txt

dry_run forced-off \
  -e TP=8 \
  -e DCP=4 \
  -e DCP_QUERY_SPLIT=0 \
  -e DCP_CKV_GATHER=0
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' /tmp/gilded-gnosis-v18-forced-off.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' /tmp/gilded-gnosis-v18-forced-off.txt

docker run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v18.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  "${IMAGE}" | tee /tmp/gilded-gnosis-v18-nf3.txt
grep -q '^ONLINE_QUANT=nf3-mxfp8$' /tmp/gilded-gnosis-v18-nf3.txt
grep -q '^VLLM_NF3_GRID188_DECODE=1$' /tmp/gilded-gnosis-v18-nf3.txt
grep -q 'shared_experts' /tmp/gilded-gnosis-v18-nf3.txt
grep -q -- '--quantization nvfp4_nf3_hybrid' /tmp/gilded-gnosis-v18-nf3.txt
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' /tmp/gilded-gnosis-v18-nf3.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v18-nf3.txt

docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52-hybrid \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  "${IMAGE}" | grep -q '^ONLINE_QUANT=nf3-mxfp8$'

ds4_dry_run() {
  local mode="$1"
  local backend="$2"
  local tp="$3"
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 \
    -e MODE="${mode}" \
    -e BACKEND="${backend}" \
    -e TP_SIZE="${tp}" \
    -e MODEL=/model \
    "${IMAGE}" 2>&1 | tee "/tmp/gilded-gnosis-v18-ds4-${mode}-${backend}.txt"
}

ds4_dry_run mtp0 b12x-a16 2
ds4_dry_run mtp2 b12x-a8 2
ds4_dry_run mtp3 b12x-a8-dglin 2
ds4_dry_run dspark lucifer-cutlass 4
grep -q 'graph=512 load_format=instanttensor instanttensor_backend=BUFFERED' \
  /tmp/gilded-gnosis-v18-ds4-mtp2-b12x-a8.txt
grep -q 'graph=384 load_format=instanttensor instanttensor_backend=BUFFERED' \
  /tmp/gilded-gnosis-v18-ds4-dspark-lucifer-cutlass.txt
if grep -q -- '--revision' /tmp/gilded-gnosis-v18-ds4-mtp0-b12x-a16.txt; then
  echo 'ERROR: DS4 helper injected an HF revision for a local model path' >&2
  exit 1
fi

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
