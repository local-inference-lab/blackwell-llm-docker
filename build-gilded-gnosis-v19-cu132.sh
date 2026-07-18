#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Unified GLM 5.2 and DS4/DSpark image built from the rebased Gilded Gnosis
# branch plus the independently reviewable gg-rebased PR stack.
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v19-vllm2152d08-b12xbc85ef3-fi801d57a-cu132-20260718}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/voipmonitor/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v19-20260718}"
export VLLM_COMMIT="${VLLM_COMMIT:-2152d08149d097c077d59fb3c9fde2ad7af525fb}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v19.vllm2152d08.b12xbc85ef3.fi801d57a.cu132.20260718}"

# Current lukealonso/b12x master plus ready-for-review Grid188 PR #36.
export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/nf3-grid188-decode-20260717}"
export B12X_COMMIT="${B12X_COMMIT:-bc85ef36192cb6e444d42ba7be86e1e125cca98a}"

export VLLM_REQUIRED_LAUNCHERS="serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-v19.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh serve-glm52-hybrid-v19.sh"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

# Reuse the audited CUDA 13.2/NCCL/InstantTensor/FlashInfer build substrate and
# its v18 compatibility assertions while overriding all mutable source pins.
./build-gilded-gnosis-v18-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_COMMIT}" '."local-inference.b12x.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "801d57a08958c13d375ddbb6be3be4808f48a708" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null

docker run --rm --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
from vllm import envs
from vllm.distributed.device_communicators import symm_mem_pcie_barrier
from vllm.model_executor.layers import fp8_draft_head
from vllm.v1.worker.gpu.spec_decode import capacity
from vllm.v1.worker.gpu.spec_decode.dspark import online_sts

assert hasattr(envs, "VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE")
assert hasattr(envs, "VLLM_SYMM_MEM_PCIE_SAFE_BARRIER")
assert hasattr(envs, "VLLM_DSPARK_FP8_DRAFT_HEAD")
assert hasattr(envs, "VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH")
assert hasattr(envs, "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE")
assert hasattr(symm_mem_pcie_barrier, "install_pcie_safe_barrier")
assert hasattr(fp8_draft_head, "Fp8DraftHead")
assert hasattr(capacity, "DSparkDynamicDraftDepthController")
assert hasattr(capacity, "CapacityBasedVerificationManager")
assert hasattr(online_sts, "DSparkOnlineSTS")
print("v19 rebased runtime symbols: PASS")
PY

dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e DRY_RUN=1 \
    -e MODEL_FAMILY=glm52 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" | tee "/tmp/gilded-gnosis-v19-${name}.txt"
}

dry_run tp8-dcp4 -e TP=8 -e DCP=4
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/gilded-gnosis-v19-tp8-dcp4.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/gilded-gnosis-v19-tp8-dcp4.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v19-tp8-dcp4.txt

dry_run tp6-dcp6-mtp3 -e TP=6 -e DCP=6 -e MTP=3
grep -q -- '--tensor-parallel-size 6' /tmp/gilded-gnosis-v19-tp6-dcp6-mtp3.txt
grep -q -- '--decode-context-parallel-size 6' /tmp/gilded-gnosis-v19-tp6-dcp6-mtp3.txt
grep -q 'num_speculative_tokens.*3' /tmp/gilded-gnosis-v19-tp6-dcp6-mtp3.txt

docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52-hybrid \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  "${IMAGE}" | tee /tmp/gilded-gnosis-v19-nf3.txt
grep -q '^ONLINE_QUANT=nf3-mxfp8$' /tmp/gilded-gnosis-v19-nf3.txt
grep -q '^VLLM_NF3_GRID188_DECODE=1$' /tmp/gilded-gnosis-v19-nf3.txt
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' /tmp/gilded-gnosis-v19-nf3.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v19-nf3.txt

ds4_dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e MODEL_FAMILY=ds4 \
    -e DRY_RUN=1 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" 2>&1 | tee "/tmp/gilded-gnosis-v19-ds4-${name}.txt"
}

ds4_dry_run default \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=4
grep -q 'load_format=instanttensor instanttensor_backend=BUFFERED' \
  /tmp/gilded-gnosis-v19-ds4-default.txt

ds4_dry_run adaptive \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=4 \
  -e DSPARK_CAPACITY=1 \
  -e DSPARK_DYNAMIC_DRAFT_DEPTH=1 \
  -e DSPARK_FP8_DRAFT_HEAD=1
grep -q 'dspark_capacity_verification_mode' \
  /tmp/gilded-gnosis-v19-ds4-adaptive.txt

docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc \
  'grep -q "export VLLM_DSPARK_FP8_DRAFT_HEAD" /usr/local/bin/serve-ds4-flash.sh && grep -q "export VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH" /usr/local/bin/serve-ds4-flash.sh'

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
